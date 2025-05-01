# Neural Texture Compression Results

This directory will store the results of training and evaluation runs.

## Directory Structure

When you run the training script, it will create a timestamped subdirectory for each run:

```
results/
├── 20250429-190000/  # Example timestamped directory
│   ├── config.yaml   # Copy of the configuration used
│   ├── logs/         # TensorBoard logs
│   ├── checkpoint_500.pth  # Model checkpoint at step 500
│   ├── checkpoint_1000.pth # Model checkpoint at step 1000
│   └── model_final.pth     # Final model checkpoint
├── quick_test/       # Results from quick test run
│   └── ...
├── quick_evaluation/ # Evaluation results from quick test
│   ├── texture_set_1/
│   │   ├── compression_mip0.png  # Visualization of compression at mip level 0
│   │   ├── features.png          # Visualization of grid features
│   │   ├── fourier.png           # Visualization of Fourier transforms
│   │   ├── rate_distortion_psnr.png  # Rate-distortion curve (PSNR)
│   │   ├── rate_distortion_ssim.png  # Rate-distortion curve (SSIM)
│   │   └── metrics.txt           # Metrics summary
│   └── ...
└── ...
```

## Viewing Results

### TensorBoard

You can view the training progress using TensorBoard:

```bash
pip install tensorboard
tensorboard --logdir=results
```

Then open http://localhost:6006 in your browser.

### Visualizations

The evaluation script generates various visualizations:

1. **Compression Visualizations**: Original vs. reconstructed textures at different mip levels
2. **Feature Visualizations**: Visualization of the grid features (G0 and G1)
3. **Fourier Transforms**: Visualization of the Fourier transforms of the grid features
4. **Rate-Distortion Curves**: Plots of rate vs. distortion metrics (PSNR, SSIM)

### Metrics

The evaluation script also generates a metrics.txt file with a summary of the metrics:

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity) - if enabled
- BD-rate (Bjøntegaard-Delta rate) - compared to ASTC

## Using Results in the Frontend

After running the evaluation, you can copy the results to the frontend:

```bash
mkdir -p vision-ops/public/compression-results
cp -r results/quick_evaluation/* vision-ops/public/compression-results/
```

Then update the neural-video-compression page to use the actual results instead of placeholder images.
