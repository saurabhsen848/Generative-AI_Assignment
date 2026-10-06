"""Assignment 7: Creating and Fixing Mode Collapse in a GAN.

Run from the VS Code terminal with: python assignment7_mode_collapse.py
Fashion-MNIST is downloaded automatically to ./data when needed.
"""

from pathlib import Path
import random
import socket
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Assignment settings
NGF_BROKEN = 16
NDF_BROKEN = 32  # Smaller discriminator keeps CPU training lighter; G remains NGF=16.
LR_BROKEN = 0.01
BETA1 = 0.5
NUM_EPOCHS = 18
BATCH_SIZE = 128  # Fewer batches keeps full 18-epoch CPU runs practical.
LATENT_SIZE = 100
IMAGE_SIZE = 28
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_a7"
DATA_DIR = Path(__file__).resolve().parent / "data"
FASHION_MNIST_MIRRORS = [
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
]


class Generator(nn.Module):
    """Small DCGAN generator; deliberately has no BatchNorm2d layers."""

    def __init__(self, ngf: int = NGF_BROKEN):
        super().__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(LATENT_SIZE, ngf * 4, 7, 1, 0, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(ngf, 1, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.network(noise)


class Discriminator(nn.Module):
    def __init__(self, ndf: int = NDF_BROKEN):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image).view(-1)


def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def train_gan(loader: DataLoader, device: torch.device, label_smoothing: bool):
    generator = Generator(NGF_BROKEN).to(device)
    discriminator = Discriminator(NDF_BROKEN).to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)

    criterion = nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=LR_BROKEN, betas=(BETA1, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=LR_BROKEN, betas=(BETA1, 0.999))
    history = {"generator": [], "discriminator": []}

    for epoch in range(NUM_EPOCHS):
        total_g = 0.0
        total_d = 0.0
        for real_images, _ in loader:
            real_images = real_images.to(device)
            batch = real_images.size(0)
            real_value = 0.9 if label_smoothing else 1.0
            real_targets = torch.full((batch,), real_value, device=device)
            fake_targets = torch.zeros(batch, device=device)

            # Train discriminator on real and detached generated images.
            discriminator.zero_grad(set_to_none=True)
            real_loss = criterion(discriminator(real_images), real_targets)
            noise = torch.randn(batch, LATENT_SIZE, 1, 1, device=device)
            fake_images = generator(noise)
            fake_loss = criterion(discriminator(fake_images.detach()), fake_targets)
            loss_d = real_loss + fake_loss
            loss_d.backward()
            optimizer_d.step()

            # Train generator to make discriminator classify fakes as real.
            generator.zero_grad(set_to_none=True)
            generated_targets = torch.full((batch,), 1.0, device=device)
            loss_g = criterion(discriminator(fake_images), generated_targets)
            loss_g.backward()
            optimizer_g.step()
            total_d += loss_d.item()
            total_g += loss_g.item()

        mean_d = total_d / len(loader)
        mean_g = total_g / len(loader)
        history["discriminator"].append(mean_d)
        history["generator"].append(mean_g)
        model_name = "Fixed" if label_smoothing else "Broken"
        print(
            f"{model_name} Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"D loss: {mean_d:.4f} | G loss: {mean_g:.4f}",
            flush=True,
        )
    return generator, history


@torch.no_grad()
def save_grid(generator: Generator, device: torch.device, path: Path, seed: int) -> torch.Tensor:
    generator.eval()
    # Reuse latent vectors across runs for a fair visual comparison.
    gen = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(64, LATENT_SIZE, 1, 1, generator=gen, device=device)
    images = generator(noise).cpu().clamp(-1, 1)
    fig, axes = plt.subplots(8, 8, figsize=(9, 9))
    fig.suptitle(path.stem.replace("_", " ").title(), fontsize=16)
    for axis, image in zip(axes.flat, images):
        axis.imshow(image.squeeze(0), cmap="gray", vmin=-1, vmax=1)
        axis.axis("off")
    fig.subplots_adjust(top=0.92, wspace=0.04, hspace=0.04)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return images


def save_loss_plot(history: dict, path: Path, title: str) -> None:
    epochs = range(1, len(history["generator"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history["generator"], marker="o", label="Generator loss")
    ax.plot(epochs, history["discriminator"], marker="o", label="Discriminator loss")
    ax.set(title=title, xlabel="Epoch", ylabel="Binary cross-entropy loss")
    ax.set_xticks(list(epochs))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_comparison(broken: torch.Tensor, fixed: torch.Tensor, path: Path) -> None:
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.8))
    fig.suptitle("Broken GAN vs. GAN with real-label smoothing", fontsize=16)
    for row, (images, heading) in enumerate(((broken, "Broken model"), (fixed, "Label smoothing"))):
        axes[row, 0].set_ylabel(heading, fontsize=12)
        for col in range(8):
            axes[row, col].imshow(images[col].squeeze(0), cmap="gray", vmin=-1, vmax=1)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def sample_diversity(images: torch.Tensor) -> tuple[float, int]:
    """Return mean pixel distance and coarse distinct count for the saved grid."""
    array = images.numpy().reshape(len(images), -1)
    pairwise = np.abs(array[:, None, :] - array[None, :, :]).mean(axis=2)
    upper = pairwise[np.triu_indices(len(array), k=1)]
    coarse = np.round(array, 1)
    distinct = len(np.unique(coarse, axis=0))
    return float(upper.mean()), distinct


def load_fashion_mnist(transform):
    """Load a valid local torchvision cache, or download with timeout and retries."""
    print("Checking Fashion-MNIST...", flush=True)
    try:
        local_dataset = datasets.FashionMNIST(
            root=str(DATA_DIR), train=True, download=False, transform=transform
        )
        print("Dataset ready.", flush=True)
        return local_dataset
    except RuntimeError:
        # No complete torchvision-processed cache; download() will reuse any
        # valid raw files already present and fetch only missing/corrupt files.
        pass

    print("Downloading Fashion-MNIST...", flush=True)
    # GitHub's HTTPS copy is the primary mirror. Keep torchvision's upstream
    # Fashion-MNIST host as a fallback.
    datasets.FashionMNIST.mirrors = FASHION_MNIST_MIRRORS
    previous_timeout = socket.getdefaulttimeout()
    last_error = None
    try:
        # torchvision's download_url uses urllib without an explicit timeout.
        # A socket default prevents a stalled connection from hanging forever.
        socket.setdefaulttimeout(30)
        for attempt in range(1, 4):
            try:
                dataset = datasets.FashionMNIST(
                    root=str(DATA_DIR), train=True, download=True, transform=transform
                )
                print("Dataset ready.", flush=True)
                return dataset
            except Exception as error:
                last_error = error
                if attempt == 3:
                    break
                wait_seconds = 2 ** attempt
                print(
                    f"Download attempt {attempt}/3 failed ({error}). "
                    f"Retrying in {wait_seconds} seconds...",
                    flush=True,
                )
                time.sleep(wait_seconds)
    finally:
        socket.setdefaulttimeout(previous_timeout)

    raise RuntimeError(
        "Fashion-MNIST could not be downloaded after 3 attempts. "
        "Check the network connection and try again."
    ) from last_error


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(7)
    torch.manual_seed(7)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        # Cap intra-op workers to keep CPU use predictable on a laptop.
        torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f"Using device: {device}", flush=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = load_fashion_mnist(transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Reliable default for Windows; avoids worker startup overhead.
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    print("Starting broken GAN...", flush=True)
    broken_generator, broken_history = train_gan(loader, device, label_smoothing=False)
    broken_images = save_grid(
        broken_generator, device, OUTPUT_DIR / "broken_output_screenshot.png", seed=2026
    )
    save_loss_plot(broken_history, OUTPUT_DIR / "loss_broken.png", "Broken GAN training losses")
    print("Saved broken output.", flush=True)

    # Keep only the small sample grid and loss values before training the second model.
    del broken_generator
    print("Starting fixed GAN...", flush=True)
    fixed_generator, fixed_history = train_gan(loader, device, label_smoothing=True)
    fixed_images = save_grid(
        fixed_generator, device, OUTPUT_DIR / "improved_output_screenshot.png", seed=2026
    )
    save_loss_plot(fixed_history, OUTPUT_DIR / "loss_fixed.png", "GAN with label smoothing losses")
    print("Saved improved output.", flush=True)
    save_comparison(broken_images, fixed_images, OUTPUT_DIR / "comparison_broken_vs_fixed.png")
    print("Saved comparison.", flush=True)
    print("Saved loss graphs.", flush=True)

    # Describe the actual sampled grids cautiously: this is a visual check, not a
    # formal measurement of mode coverage or image quality.
    broken_distance, broken_unique = sample_diversity(broken_images)
    fixed_distance, fixed_unique = sample_diversity(fixed_images)
    if fixed_distance > broken_distance * 1.10:
        observation = (
            "The broken grid had lower average pixel differences between generated samples, while the "
            "label-smoothed grid had greater sample variation. Review the saved comparison to judge "
            "whether this variation forms recognizable Fashion-MNIST items."
        )
        assessment = (
            "The smoothed samples had higher pixel variation in this run, which is a limited sign of "
            "improvement; pixel variation alone does not establish better image quality. "
        )
    elif fixed_distance < broken_distance * 0.90:
        observation = (
            "The label-smoothed grid had lower average pixel differences between generated samples than "
            "the broken grid. Review the comparison figure to see whether the images repeat or differ "
            "in recognizable ways."
        )
        assessment = (
            "The pixel-variation check does not indicate an improvement in sample diversity in this run. "
            "The figure should be judged visually as well. "
        )
    else:
        observation = (
            "The grids had similar average pixel variation; inspect comparison_broken_vs_fixed.png to "
            "judge the appearance and variety of the generated items."
        )
        assessment = "This run did not show a clear change in sample variation by this simple check. "

    explanation = (
        "Assignment 7: Creating and Fixing Mode Collapse in a GAN\n\n"
        "1. What modification caused the model to perform poorly?\n"
        "The generator was deliberately reduced to NGF=16, BatchNorm2d was removed from the generator, "
        "and both networks used a very high learning rate of 0.01. These changes made adversarial training "
        "less stable.\n\n"
        "2. What did the generated images look like?\n"
        f"{observation} The broken output is saved as broken_output_screenshot.png and both runs are "
        "shown in comparison_broken_vs_fixed.png. Mode collapse is not claimed from loss values alone.\n\n"
        "3. Which technique was used to improve the model?\n"
        "One-sided label smoothing was applied to real images: the discriminator target for a real image "
        "was set to 0.9 instead of 1.0. The architecture and learning rate remained the same, and the "
        "fixed run started from fresh weights.\n\n"
        "4. Did the solution improve the output? Why?\n"
        f"{assessment} Label smoothing can prevent the discriminator from becoming too confident, which can "
        "provide more useful gradients to the generator. For context, the saved 64-image grids had mean "
        f"pairwise pixel distances of {broken_distance:.3f} (broken) and {fixed_distance:.3f} (smoothed); "
        f"coarsely distinct samples: {broken_unique}/64 and {fixed_unique}/64. These simple statistics do "
        "not replace visual inspection and the conclusion is limited to this run.\n"
    )
    (OUTPUT_DIR / "assignment7_explanation.txt").write_text(explanation, encoding="utf-8")
    print(f"All requested images and the explanation were saved in: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
