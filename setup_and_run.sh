#!/bin/bash
# Setup and run script for neural texture compression

# Exit on error
set -e

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r neural-compression/requirements.txt

# Check for CUDA
if python -c "import torch; print(torch.cuda.is_available())"; then
    echo "CUDA is available, installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "CUDA is not available, installing PyTorch CPU version..."
    pip install torch torchvision
fi

# Create data directory
echo "Creating data directory..."
mkdir -p data/textures

# Check if texture data exists
if [ -z "$(ls -A data/textures)" ]; then
    echo "No texture data found. Please add texture sets to data/textures directory."
    echo "Each texture set should be in its own subdirectory."
    echo "Example structure:"
    echo "  data/textures/texture_set_1/"
    echo "    - diffuse.png"
    echo "    - normal.png"
    echo "    - displacement.png"
    echo "  data/textures/texture_set_2/"
    echo "    - ..."
fi

# Create results directory
echo "Creating results directory..."
mkdir -p results

# Run quick test if requested
if [ "$1" == "quick" ]; then
    echo "Running quick test..."
    python neural-compression/experiments/train.py \
        --config neural-compression/experiments/configs/quick_test.yaml \
        --output_dir results/quick_test \
        --gpu 0
    
    # Run evaluation
    echo "Running evaluation..."
    python neural-compression/experiments/evaluate.py \
        --model_path results/quick_test/model_final.pth \
        --texture_dir data/textures \
        --output_dir results/quick_evaluation \
        --gpu 0
elif [ "$1" == "default" ]; then
    echo "Running default training..."
    python neural-compression/experiments/train.py \
        --config neural-compression/experiments/configs/default.yaml \
        --output_dir results/default \
        --gpu 0
    
    # Run evaluation
    echo "Running evaluation..."
    python neural-compression/experiments/evaluate.py \
        --model_path results/default/model_final.pth \
        --texture_dir data/textures \
        --output_dir results/default_evaluation \
        --gpu 0
elif [ "$1" == "high_quality" ]; then
    echo "Running high quality training..."
    python neural-compression/experiments/train.py \
        --config neural-compression/experiments/configs/high_quality.yaml \
        --output_dir results/high_quality \
        --gpu 0
    
    # Run evaluation
    echo "Running evaluation..."
    python neural-compression/experiments/evaluate.py \
        --model_path results/high_quality/model_final.pth \
        --texture_dir data/textures \
        --output_dir results/high_quality_evaluation \
        --gpu 0
else
    echo "Usage: $0 [quick|default|high_quality]"
    echo "  quick: Run quick test with reduced steps and model complexity"
    echo "  default: Run default training"
    echo "  high_quality: Run high quality training"
fi

echo "Done!"
