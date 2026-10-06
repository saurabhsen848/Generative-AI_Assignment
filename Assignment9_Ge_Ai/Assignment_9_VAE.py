"""Using a Variational Autoencoder for Compression and Denoising.

Run with: python Assignment_9_VAE.py
The script downloads MNIST through tensorflow.keras.datasets when needed.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers


# Fixed seeds make initialization, image selection, and added noise repeatable.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

LATENT_DIMS = [2, 16, 32]
EPOCHS = 5  # Increase for better results; 5 keeps the assignment practical.
BATCH_SIZE = 128
NOISE_STDDEV = 0.35
RESULTS_DIR = "results"


class VAE(tf.keras.Model):
    """A compact dense VAE with a custom training step and KL loss."""

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_hidden = layers.Dense(256, activation="relu")
        self.z_mean_layer = layers.Dense(latent_dim)
        self.z_log_var_layer = layers.Dense(latent_dim)
        self.decoder_hidden = layers.Dense(256, activation="relu")
        self.decoder_output = layers.Dense(784, activation="sigmoid")
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.loss_tracker, self.reconstruction_tracker, self.kl_tracker]

    def encode(self, inputs):
        hidden = self.encoder_hidden(inputs)
        return self.z_mean_layer(hidden), self.z_log_var_layer(hidden)

    def reparameterize(self, mean, log_var):
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon

    def decode(self, latent):
        return self.decoder_output(self.decoder_hidden(latent))

    def call(self, inputs, training=False):
        mean, log_var = self.encode(inputs)
        latent = self.reparameterize(mean, log_var) if training else mean
        return self.decode(latent)

    def train_step(self, data):
        images = data[0] if isinstance(data, (tuple, list)) else data
        with tf.GradientTape() as tape:
            mean, log_var = self.encode(images)
            reconstruction = self.decode(self.reparameterize(mean, log_var))
            # Sum pixel-wise binary cross entropy, then average over the batch.
            bce = tf.keras.backend.binary_crossentropy(images, reconstruction)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(bce, axis=1))
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1)
            )
            loss = reconstruction_loss + kl_loss
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.reconstruction_tracker.update_state(reconstruction_loss)
        self.kl_tracker.update_state(kl_loss)
        return {metric.name: metric.result() for metric in self.metrics}


def save_reconstruction_grid(originals, reconstructed, filename, title):
    """Save a two-row original/reconstruction comparison."""
    fig, axes = plt.subplots(2, len(originals), figsize=(12, 3.2))
    for i in range(len(originals)):
        axes[0, i].imshow(originals[i].reshape(28, 28), cmap="gray")
        axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Loading MNIST...")
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32").reshape(-1, 784) / 255.0
    x_test = x_test.astype("float32").reshape(-1, 784) / 255.0
    selected = x_test[:10]
    rng = np.random.default_rng(SEED)
    noise = rng.normal(0.0, NOISE_STDDEV, selected.shape).astype("float32")
    noisy = np.clip(selected + noise, 0.0, 1.0)

    models = {}
    rows = []
    for latent_dim in LATENT_DIMS:
        print(f"\nTraining VAE with latent dimension {latent_dim} for {EPOCHS} epochs...")
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED + latent_dim)
        model = VAE(latent_dim)
        model.compile(optimizer=tf.keras.optimizers.Adam())
        model.fit(x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=2)
        models[latent_dim] = model

        # Encode to the latent mean, then decode that representation.
        latent_mean, _ = model.encode(selected)
        reconstructed = model.decode(latent_mean).numpy()
        per_image_mse = np.mean(np.square(selected - reconstructed), axis=1)
        average_mse = float(np.mean(per_image_mse))
        compression_ratio = 784 / latent_dim
        save_reconstruction_grid(
            selected, reconstructed,
            os.path.join(RESULTS_DIR, f"latent_{latent_dim}_reconstruction.png"),
            f"Latent dimension {latent_dim} | Test MSE {average_mse:.6f}",
        )
        rows.append({
            "Latent Dimension": latent_dim,
            "Compression Ratio": f"{compression_ratio:g}:1",
            "Reconstruction Error": average_mse,
            "Reconstruction Quality": "Pending comparison",
            "Denoising Observation": "Pending comparison",
        })
        print("Per-image reconstruction MSE:", ", ".join(f"{value:.6f}" for value in per_image_mse))

    # Compare denoising using each trained model on the identical noisy images.
    denoised_by_dim = {}
    for dim in LATENT_DIMS:
        noisy_mean, _ = models[dim].encode(noisy)
        denoised_by_dim[dim] = models[dim].decode(noisy_mean).numpy()
    denoising_mse = {
        dim: float(np.mean(np.square(selected - output)))
        for dim, output in denoised_by_dim.items()
    }
    best_dim = min(rows, key=lambda row: row["Reconstruction Error"])["Latent Dimension"]
    ordered_errors = [row["Reconstruction Error"] for row in rows]
    for index, row in enumerate(rows):
        dim = row["Latent Dimension"]
        quality_rank = sorted(ordered_errors).index(row["Reconstruction Error"]) + 1
        if index == 0:
            trend = "Baseline latent dimension"
        elif ordered_errors[index] < ordered_errors[index - 1]:
            trend = "Improved over previous dimension"
        elif ordered_errors[index] > ordered_errors[index - 1]:
            trend = "Lower quality than previous dimension"
        else:
            trend = "Same MSE as previous dimension"
        row["Reconstruction Quality"] = f"{trend}; rank {quality_rank}/3 by MSE"
        noisy_baseline = float(np.mean(np.square(selected - noisy)))
        change = "reduced" if denoising_mse[dim] < noisy_baseline else "not reduced"
        row["Denoising Observation"] = (
            f"Denoised MSE {denoising_mse[dim]:.6f}; {change} vs noisy MSE {noisy_baseline:.6f}"
        )

    # Main reconstruction figure for the smallest latent space.
    latent_mean_2, _ = models[2].encode(selected)
    recon2 = models[2].decode(latent_mean_2).numpy()
    save_reconstruction_grid(selected, recon2, os.path.join(RESULTS_DIR, "reconstruction_latent_2.png"), "Original and reconstructed MNIST images (latent dimension 2)")

    # Three rows show the same original, noisy, and denoised examples.
    denoised = denoised_by_dim[32]
    fig, axes = plt.subplots(3, 10, figsize=(12, 4.8))
    for i in range(10):
        for row_index, images in enumerate((selected, noisy, denoised)):
            axes[row_index, i].imshow(images[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            axes[row_index, i].axis("off")
    for idx, label in enumerate(("Original", "Noisy", "Denoised (latent 32)")):
        axes[idx, 0].set_ylabel(label, fontsize=9)
    fig.suptitle(f"Gaussian noise (standard deviation {NOISE_STDDEV}) and VAE denoising")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "denoising_results.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Write results and plot measured reconstruction error against latent size.
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "results_table.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["Latent Dimension"], df["Reconstruction Error"], marker="o")
    ax.set(title="Reconstruction MSE by latent dimension", xlabel="Latent dimension", ylabel="Average test-image MSE")
    ax.set_xticks(LATENT_DIMS)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "latent_dimension_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nLatent Dimension | Compression Ratio | Reconstruction Error (MSE) | Reconstruction Quality")
    for row in rows:
        print(f"{row['Latent Dimension']:16} | {row['Compression Ratio']:17} | {row['Reconstruction Error']:.6f}                  | {row['Reconstruction Quality']}")
    if ordered_errors[-1] < ordered_errors[0]:
        increase_result = "In this run, reconstruction MSE improved from latent dimension 2 to 32."
    elif ordered_errors[-1] > ordered_errors[0]:
        increase_result = "In this run, reconstruction MSE did not improve from latent dimension 2 to 32."
    else:
        increase_result = "In this run, dimensions 2 and 32 had the same reconstruction MSE."
    print("\nConclusion:")
    print("- Latent dimension 2 gives the highest compression (392:1).")
    print("- Larger latent dimensions generally preserve more information, though the measured result depends on training.")
    print(f"- {increase_result}")
    print(f"- Reconstruction quality changes with latent dimension; best measured MSE here: dimension {best_dim}.")
    print("- The VAE can be used for image denoising by reconstructing noisy inputs.")
    print(f"\nAll images and the results table were saved in: {os.path.abspath(RESULTS_DIR)}")


if __name__ == "__main__":
    main()
