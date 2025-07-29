# Compare predictions without ground truths
In 2-Photon microscopy, we don't have a labelled dataset, so how can we assert which solution delivers better predictions?

## 1. Proxy Ground Truth via Slow Scans
Acquire a small region of interest at **very low frame‐rate** or **elevated excitation power**, and average ≥ N (e.g. 20–50) raw frames. See [2.1-Means_Medians.ipynb](2.1-Means_Medians.ipynb).

> May introduce motion blur or photodamage.
> Hinders fast dynamics.

- PSNR and SSIM computed against the averaged image.
- RMSE of pixel intensities.

## Synthetic Noise Injection on High‑SNR Patches
Basically the idea is to estimate the ground truth only for those pixels having high photon count.

1. Select patches from your data that visually appear high‑SNR (e.g. somata regions).
2. Fit a Poisson–Gaussian noise model (estimate photon‑count λ and detector variance σ²).
3. Re‑simulate noisy versions of the patch by drawing
4. Run each denoiser on these synthetic noisy patches

- PSNR, SSIM, RMSE relative to the original high‑SNR patch.
- Edge‐preservation index (e.g. Pratt’s figure of merit).

## No‑Reference Evaluation Metrics
BRISQUE, NIQE, PIQE rank denoisers by perceptual quality.

## Check that the residual is uncorrelated
Compute the residual image for each denoiser. Check that the residual:
    - Has zero mean,
    - Is spatially uncorrelated (flat autocorrelation),
    - Has the same global noise statistics as your raw data (e.g. same histogram of intensities).

If one algorithm leaves structured features in the residual, it’s over‑smoothing; if it leaves heavy tails, it’s under‑denoising.

## Evaluate Downstream Task Performance
Segmentation