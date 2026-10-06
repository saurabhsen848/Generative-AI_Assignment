# Assignment 9: Using a Variational Autoencoder for Compression and Denoising

## Objective

Use a Variational Autoencoder (VAE) on MNIST to study image compression, reconstruction, denoising, and the effect of changing the latent dimension. The program uses TensorFlow/Keras, NumPy, Matplotlib, and Pandas.

**Assignment 8 code note:** No Assignment 8 files were present in this project folder, and the parent user directory is not accessible in this environment. The Assignment 9 program therefore uses a small dense VAE with separate encoder and decoder methods. If you add your Assignment 8 implementation to this folder, its model can be substituted in those methods.

## Part A: Compression

Each MNIST image has 28 × 28 = 784 pixel values. The program trains models using latent dimensions 2, 16, and 32. The latent representation size is the selected latent dimension, and compression ratio is calculated as `784 / latent_dimension` (392:1, 49:1, and 24.5:1).

## Part B: Reconstruction

The same first 10 MNIST test images are explicitly encoded to their latent means and decoded. The program compares each reconstruction with its original, calculates MSE for each image and the average MSE, and saves one original/reconstruction comparison image per latent dimension.

## Part C: Denoising

Reproducible Gaussian noise is added to the same 10 test images. Values are clipped to [0, 1]. Each trained VAE reconstructs the noisy images, and the denoising figure shows three rows: Original, Noisy, and Denoised (using latent dimension 32). The CSV records the denoised MSE against the clean originals and compares it to the noisy-input MSE.

## Part D: Changing Latent Dimension

The three independently trained models are compared by compression ratio and average reconstruction MSE. Reconstruction quality is ranked by actual MSE, and the table states whether quality improved or declined versus the previous dimension. The terminal conclusion reports whether dimension 32 improved MSE relative to dimension 2 for this run.

## Results

On completion, the program creates `results/` with:

- `reconstruction_latent_2.png`
- `reconstruction_latent_16.png`
- `reconstruction_latent_32.png`
- `denoising_results.png`
- `latent_dimension_comparison.png`
- `results_table.csv` (columns: Latent Dimension, Compression Ratio, Reconstruction Error, Reconstruction Quality, Denoising Observation)

All error values and quality descriptions are calculated from the run. No MSE values are pre-filled.

## Conclusion

Latent dimension 2 provides the greatest compression. Larger latent dimensions can preserve more image detail, while measured reconstruction quality depends on the trained models; the program checks this using actual MSE. The VAE also attempts to denoise by reconstructing noisy inputs.

## How to run the program

Install packages in the VS Code terminal:

```bash
python -m pip install tensorflow numpy matplotlib pandas scikit-learn
```

Run from this project folder:

```bash
python Assignment_9_VAE.py
```

MNIST downloads automatically through `tensorflow.keras.datasets` when first run, so initial execution requires network access. The random seed is fixed. The script trains each model for 5 epochs; change `EPOCHS` near the top of the Python file to train longer.
