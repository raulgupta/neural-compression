# 💻 Neural Graphics Texture Compression

A PyTorch implementation of the asymmetric autoencoder framework for neural texture compression as described in the paper ["Neural Graphics Texture Compression Supporting Random Access"](https://arxiv.org/abs/2407.00021) by Farhadzadeh et al.

## 🎛 Features

- **Asymmetric Autoencoder Framework**: Efficient neural compression architecture
- **Multi-Channel Support**: Compress texture sets with any number of channels
- **Multi-Resolution Reconstruction**: Support for mip-levels essential for texture filtering
- **Random Access Capability**: Enables efficient texel sampling during rendering
- **High Compression Ratio**: Achieves 240:1 compression with high quality reconstruction
- **GPU Acceleration**: Optimized for GPU training and inference

## 🏗️ Architecture

The implementation consists of four main components:

1. **Global Transformer**: A convolutional encoder that captures detailed information in a bottleneck-latent space.
2. **Grid Constructor**: Maps the latent representation to a pair of grid features (dual-bank features).
3. **Grid Sampler**: Samples the grids based on texture coordinates and mip level.
4. **Texture Synthesizer**: A fully connected network that reconstructs texels at specific positions and mip levels.

## 📊 Performance

Our implementation achieves impressive compression results:

- **Compression Ratio**: 240:1
- **Average PSNR**: 27.33 dB
- **Average SSIM**: 0.7581
- **BD-Rate Savings**: -88.67% compared to ASTC

## 🖼️ Texture Sets

This implementation uses high-quality PBR texture sets from [AmbientCG](https://ambientcg.com/), a free and CC0 licensed resource for 3D artists and developers. The texture sets include:

- **Diffuse/Albedo Maps**: Base color information
- **Normal Maps**: Surface detail for realistic lighting
- **Displacement Maps**: Height information for surface detail
- **Roughness Maps**: Surface smoothness/roughness properties

These multi-channel texture sets provide an excellent test case for our compression algorithm, demonstrating its ability to handle diverse texture types while maintaining visual fidelity across all channels.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/raulgupta/neural-compression.git
cd neural-compression

# Make the setup script executable
chmod +x setup_and_run.sh

# Run quick test (1-3 hours on a decent GPU)
./setup_and_run.sh quick

# Run default training (18-25 hours on a high-end GPU)
./setup_and_run.sh default

# Run high quality training (34-48 hours on a high-end GPU)
./setup_and_run.sh high_quality
```

## 📁 Project Structure

```
neural-compression/
├── experiments/            # Training and evaluation scripts
│   ├── configs/            # Configuration files
│   ├── train.py            # Training script
│   └── evaluate.py         # Evaluation script
├── models/                 # Model architecture
│   ├── compression_model.py    # Main model
│   ├── global_transformer.py   # Encoder
│   ├── grid_constructor.py     # Grid feature constructor
│   ├── grid_sampler.py         # Grid sampler
│   └── texture_synthesizer.py  # Decoder
├── utils/                  # Utility functions
│   ├── data_loader.py          # Data loading utilities
│   ├── metrics.py              # Evaluation metrics
│   ├── visualization.py        # Visualization utilities
│   └── multi_texture_evaluate.py # Multi-texture evaluation
├── setup_and_run.sh        # Setup and training script
└── requirements.txt        # Python dependencies
```

## 🔧 Configuration Options

We provide three configuration files:

1. **quick_test.yaml**: For quick testing with reduced steps and model complexity (1-3 hours)
2. **default.yaml**: Default configuration with balanced quality and training time (18-25 hours)
3. **high_quality.yaml**: High quality configuration with more steps and larger model (34-48 hours)

## 📝 Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{farhadzadeh2024neural,
  title={Neural Graphics Texture Compression Supporting Random Access},
  author={Farhadzadeh, Farzad and Hou, Qiqi and Le, Hoang and Said, Amir and Rauwendaal, Randall and Bourd, Alex and Porikli, Fatih},
  journal={arXiv preprint arXiv:2407.00021},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
