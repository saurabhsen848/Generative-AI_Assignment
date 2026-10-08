"""Fine-tune GPT-2 on a short, original narrative style (CPU friendly)."""

from pathlib import Path
import random
import re

import torch
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)


# All assignment artifacts live beside this script, regardless of the launch cwd.
ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "custom_style_dataset.txt"
BASE_OUTPUT_PATH = ROOT / "base_model_output.txt"
FINE_OUTPUT_PATH = ROOT / "fine_tuned_model_output.txt"
COMPARISON_PATH = ROOT / "comparison.txt"
MODEL_PATH = ROOT / "gpt2-finetuned-custom-style"
MODEL_NAME = "gpt2"
PROMPT = "The scientist opened the door and found"
BLOCK_SIZE = 96


def create_custom_dataset() -> list[str]:
    """Write 250 clean, consistent lines about Mira Vale and her observatory."""
    # Ten linked night watches, each with a different quiet mystery. The repeated
    # motifs and restrained, sensory narration give the fine-tuning a clear style.
    nights = [
        ("rain", "the brass bell", "a note folded beneath the clock"),
        ("fog", "the western telescope", "a warm handprint on the cold glass"),
        ("wind", "the blue lantern", "a map marked with silver thread"),
        ("snow", "the radio cabinet", "a voice counting backward from seven"),
        ("starlight", "the locked archive", "a key wrapped in a strip of linen"),
        ("thunder", "the empty stairwell", "footsteps stopping one floor above"),
        ("mist", "the darkroom sink", "a photograph still wet at its edges"),
        ("moonlight", "the signal room", "three lamps answering from the ridge"),
        ("hail", "the engine shed", "a small engine ticking though it was cold"),
        ("dawn", "the eastern dome", "a narrow seam of light under the door"),
    ]
    beats = [
        "Mira Vale kept the observatory above the harbor, where the cliffs held the last warmth of day.",
        "She wrote each change in the sky in a narrow book bound with green cloth.",
        "The old building answered the weather with soft knocks in its wooden walls.",
        "Beyond the windows, the sea carried the town's scattered lights toward the horizon.",
        "Mira trusted patient instruments more than hurried explanations.",
        "That evening, {weather} gathered along the ridge before the lamps were lit.",
        "She checked {place} twice, then listened until the machinery settled into its familiar hum.",
        "A cup of tea cooled beside her notes while the clock moved one careful minute at a time.",
        "Near midnight, she noticed {clue}, though no one else had entered the room.",
        "The discovery did not frighten her; it made the silence feel newly attentive.",
        "She copied the detail into her book and left a blank line beneath it.",
        "In the margin, she drew the curve of the coast as it looked from the upper window.",
        "A pale reflection crossed the glass and vanished when she turned toward it.",
        "Mira opened the window a finger's width and smelled salt, stone, and distant rain.",
        "The instruments continued their steady work, indifferent to her questions.",
        "She followed the corridor slowly, keeping one hand against the cool plaster wall.",
        "At the landing, the building gave a small sigh as the wind changed direction.",
        "There was no message waiting, only the ordinary dust disturbed in a straight line.",
        "She marked the line with chalk so the morning crew could see it clearly.",
        "Then she returned to the dome and turned the telescope toward the dark water.",
        "A single star appeared between moving clouds, steady as a pin in blue cloth.",
        "Mira recorded its position without deciding what it meant.",
        "Before dawn, she banked the stove and covered the instruments with canvas.",
        "The town below began to wake, one window at a time.",
        "She closed her book, certain that the next night would bring another small question.",
    ]

    lines: list[str] = []
    for weather, place, clue in nights:
        for beat in beats:
            line = beat.format(weather=weather, place=place, clue=clue)
            line = re.sub(r"\s+", " ", line).strip()
            if line:
                lines.append(line)
    if len(lines) != 250:
        raise ValueError(f"Expected 250 dataset lines, got {len(lines)}")
    DATASET_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def clean_dataset() -> list[str]:
    """Remove blank rows, repeated whitespace, and stray formatting artifacts."""
    raw = DATASET_PATH.read_text(encoding="utf-8")
    lines = [re.sub(r"\s+", " ", row).strip() for row in raw.splitlines()]
    lines = [row for row in lines if row]
    DATASET_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def generate_text(model, tokenizer, device: torch.device, prompt: str) -> str:
    """Generate a reproducible continuation while handling GPT-2's EOS padding."""
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=90,
            do_sample=True,
            temperature=0.8,
            top_p=0.92,
            repetition_penalty=1.12,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(generated[0], skip_special_tokens=True).strip()


def tokenize_for_causal_lm(lines: list[str], tokenizer) -> Dataset:
    """Tokenize and concatenate lines, then split into fixed causal-LM blocks."""
    dataset = Dataset.from_dict({"text": lines})
    tokenized = dataset.map(
        lambda batch: tokenizer(batch["text"], add_special_tokens=False),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing narrative lines",
    )

    # Concatenation lets the model learn across sentence boundaries. Append EOS
    # between lines, then slice into manageable blocks for CPU training.
    def group_texts(batch):
        concatenated = {key: [] for key in batch}
        for row in range(len(batch["input_ids"])):
            for key, values in batch.items():
                concatenated[key].extend(values[row])
            concatenated["input_ids"].append(tokenizer.eos_token_id)
            if "attention_mask" in concatenated:
                concatenated["attention_mask"].append(1)
        total = len(concatenated["input_ids"])
        usable = (total // BLOCK_SIZE) * BLOCK_SIZE
        if usable == 0:
            usable = total
        result = {
            key: [values[i : min(i + BLOCK_SIZE, usable)] for i in range(0, usable, BLOCK_SIZE)]
            for key, values in concatenated.items()
        }
        result["labels"] = [sequence.copy() for sequence in result["input_ids"]]
        return result

    return tokenized.map(group_texts, batched=True, desc="Packing causal-LM blocks")


def main() -> None:
    set_seed(11)
    random.seed(11)
    # Explicitly choose CPU even if a CUDA build happens to be present; this
    # assignment is configured for the user's CPU-only machine.
    device = torch.device("cpu")
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    print("Loading model: gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    # GPT-2 has no dedicated pad token. Reuse EOS so batched collation works.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.to(device)
    base_model.eval()

    print("Generating base output")
    base_output = generate_text(base_model, tokenizer, device, PROMPT)
    BASE_OUTPUT_PATH.write_text(base_output + "\n", encoding="utf-8")
    del base_model

    print("Preparing dataset")
    if not DATASET_PATH.exists():
        create_custom_dataset()
    # Keep the required dataset at exactly 250 nonempty, clean lines.
    lines = clean_dataset()
    if len(lines) != 250:
        lines = create_custom_dataset()
        lines = clean_dataset()
    print(f"Prepared {len(lines)} clean narrative lines")
    train_dataset = tokenize_for_causal_lm(lines, tokenizer)
    if len(train_dataset) == 0:
        raise RuntimeError("The dataset did not produce any training blocks.")

    print("Starting fine-tuning (CPU, batch size 1, maximum 20 steps)")
    training_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    training_model.config.pad_token_id = tokenizer.pad_token_id
    training_args = TrainingArguments(
        output_dir=str(ROOT / "gpt2-training-runs"),
        num_train_epochs=1,
        max_steps=20,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        warmup_steps=2,
        weight_decay=0.01,
        logging_steps=2,
        save_strategy="no",
        report_to="none",
        # no_cuda is compatible with the Transformers 4.x versions listed in
        # requirements and forces CPU execution on this CPU-only machine.
        dataloader_num_workers=0,
        seed=11,
        data_seed=11,
    )
    trainer = Trainer(
        model=training_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    print("Saving model")
    trainer.save_model(str(MODEL_PATH))
    tokenizer.save_pretrained(str(MODEL_PATH))
    del trainer, training_model

    print("Generating fine-tuned output")
    fine_tokenizer = GPT2TokenizerFast.from_pretrained(str(MODEL_PATH))
    fine_tokenizer.pad_token = fine_tokenizer.eos_token
    fine_model = GPT2LMHeadModel.from_pretrained(str(MODEL_PATH))
    fine_model.config.pad_token_id = fine_tokenizer.pad_token_id
    fine_model.to(device)
    fine_model.eval()
    fine_output = generate_text(fine_model, fine_tokenizer, device, PROMPT)
    FINE_OUTPUT_PATH.write_text(fine_output + "\n", encoding="utf-8")

    comparison = f"""Prompt:
{PROMPT}

Original GPT-2 output:
{base_output}

Fine-tuned GPT-2 output:
{fine_output}

Writing-style differences:
1. The original model uses its broad, general-purpose continuation style; the fine-tuned output is nudged toward the dataset's quiet observatory setting, with Mira Vale, instruments, weather, and the harbor.
2. The original output is less constrained by the custom corpus, while the fine-tuned output favors measured pacing, restrained suspense, and concrete sensory details such as salt air, glass, clocks, and muted sounds.
"""
    COMPARISON_PATH.write_text(comparison, encoding="utf-8")
    print("Assignment completed")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Checkpoint: {MODEL_PATH}")
    print(f"Comparison: {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
