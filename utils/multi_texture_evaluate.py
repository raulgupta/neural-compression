import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import sys

# Add the correct path to pytorch-scripts
sys.path.append(os.path.expanduser("~/pytorch-scripts"))

# Now import from the correct modules
from models import CompressionModel
from utils.visualization import visualize_compression, visualize_features
from utils.metrics import calculate_psnr, calculate_ssim
from utils.data_loader import TextureDataset

# Find the latest checkpoint
latest_checkpoint = sys.argv[1] if len(sys.argv) > 1 else None
if not latest_checkpoint:
    results_dir = os.path.expanduser("~/results/quick_test")
    latest_dir = sorted(glob(f"{results_dir}/*/"), key=os.path.getmtime)[-1]
    checkpoints = glob(f"{latest_dir}/checkpoint_*.pth")
    if not checkpoints:
        print(f"No checkpoints found in {latest_dir}")
        sys.exit(1)
    latest_checkpoint = max(checkpoints, key=os.path.getctime)

print(f"Loading checkpoint: {latest_checkpoint}")

# Create output directory
output_dir = os.path.expanduser("~/results/multi_texture_evaluation")
os.makedirs(output_dir, exist_ok=True)

# Load the checkpoint
checkpoint = torch.load(latest_checkpoint)
config = checkpoint["config"]

# Get the number of channels from the checkpoint
in_channels = checkpoint["model_state_dict"]["global_transformer.initial_conv.weight"].shape[1]
print(f"Model was trained with {in_channels} channels")

# Create model with the correct number of channels
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CompressionModel(
    in_channels=in_channels,
    encoder_channels=config["model"]["global_transformer"]["channels"],
    grid_channels=config["model"]["grid_constructor"]["grid_channels"],
    quantization_bits=config["model"]["grid_constructor"]["quantization_bits"],
    hidden_dim=config["model"]["texture_synthesizer"]["hidden_dim"],
    num_residual_blocks=config["model"]["texture_synthesizer"]["num_residual_blocks"],
    positional_encoding_levels=config["model"]["texture_synthesizer"]["positional_encoding_levels"],
    use_attention=config["model"]["global_transformer"].get("use_attention", False)
)

# Load model weights
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Load the texture dataset
texture_dir = os.path.expanduser("~/data/textures")
dataset = TextureDataset(root_dir=texture_dir, crop_size=256)

if len(dataset) == 0:
    print(f"No textures found in {texture_dir}. Using random textures instead.")
    # Create random textures for testing
    num_textures = 5
    textures = [torch.rand(1, in_channels, 256, 256, device=device) for _ in range(num_textures)]
    texture_names = [f"random_texture_{i}" for i in range(num_textures)]
else:
    print(f"Found {len(dataset)} textures in {texture_dir}")
    textures = [dataset[i]["texture_set"].unsqueeze(0).to(device) for i in range(len(dataset))]
    texture_names = [f"texture_{i}" for i in range(len(dataset))]

# Calculate compression ratio for a 256x256 texture
h, w = 256, 256
original_bits = h * w * in_channels * 32  # 32 bits per float
compressed_bits = sum(c * (h // 8) * (w // 8) * 4 for c in config["model"]["grid_constructor"]["grid_channels"])
compression_ratio = original_bits / compressed_bits
print(f"Compression ratio: {compression_ratio:.1f}:1")
print(f"Bits per pixel per channel: {compressed_bits / (h * w * in_channels):.4f}")

# Create a summary file
with open(os.path.join(output_dir, "summary.txt"), "w") as f:
    f.write(f"Model checkpoint: {latest_checkpoint}\n")
    f.write(f"Number of input channels: {in_channels}\n")
    f.write(f"Compression ratio: {compression_ratio:.1f}:1\n")
    f.write(f"Bits per pixel per channel: {compressed_bits / (h * w * in_channels):.4f}\n\n")
    f.write(f"Grid channels: G0={config['model']['grid_constructor']['grid_channels'][0]}, G1={config['model']['grid_constructor']['grid_channels'][1]}\n")
    f.write(f"Quantization bits: {config['model']['grid_constructor']['quantization_bits']}\n\n")
    f.write("Texture Results:\n")

# Process each texture
all_psnrs = []
all_ssims = []

for texture_idx, (texture, name) in enumerate(zip(textures, texture_names)):
    print(f"\nProcessing texture {texture_idx+1}/{len(textures)}: {name}")
    
    # Create directory for this texture
    texture_dir = os.path.join(output_dir, name)
    os.makedirs(texture_dir, exist_ok=True)
    
    # Generate grid features
    with torch.no_grad():
        g0, g1 = model.encode(texture, quantization_mode="hard")
        
        # Save grid features visualization for this texture
        fig = visualize_features(g0.cpu(), g1.cpu())
        fig.savefig(os.path.join(texture_dir, "grid_features.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        # Process each mip level
        texture_psnrs = []
        texture_ssims = []
        
        for mip_level in range(5):  # 0 to 4
            # Skip higher mip levels if texture is too small
            h, w = texture.shape[2], texture.shape[3]
            scale_factor = 2 ** mip_level
            h_m, w_m = h // scale_factor, w // scale_factor
            
            if h_m < 8 or w_m < 8:
                print(f"  Skipping mip level {mip_level} (too small: {h_m}x{w_m})")
                continue
                
            print(f"  Processing mip level {mip_level} ({h_m}x{w_m})")
            
            # Create directory for this mip level
            mip_dir = os.path.join(texture_dir, f"mip{mip_level}")
            os.makedirs(mip_dir, exist_ok=True)
            
            # Reconstruct texture
            reconstructed = model.reconstruct_texture(g0, g1, mip_level=mip_level)
            
            # Downsample original for comparison
            original_m = torch.nn.functional.interpolate(
                texture, size=(h_m, w_m), mode="bilinear", align_corners=False
            )
            
            # Ensure reconstructed has the same size
            if reconstructed.shape[2:] != original_m.shape[2:]:
                reconstructed = torch.nn.functional.interpolate(
                    reconstructed, size=original_m.shape[2:], mode="bilinear", align_corners=False
                )
            
            # Calculate metrics
            psnr = calculate_psnr(original_m.cpu(), reconstructed.cpu())
            texture_psnrs.append(psnr)
            
            # Calculate SSIM with appropriate window size
            try:
                ssim = calculate_ssim(original_m.cpu(), reconstructed.cpu())
                texture_ssims.append(ssim)
            except Exception as e:
                print(f"  SSIM calculation failed: {e}")
                texture_ssims.append(0.0)
            
            print(f"  Mip level {mip_level} - PSNR: {psnr:.2f} dB, SSIM: {texture_ssims[-1]:.4f}")
            
            # Visualize compression
            fig = visualize_compression(
                original_m[0].cpu(),
                reconstructed[0].cpu(),
                channel_names=[f"Channel {i}" for i in range(in_channels)]
            )
            fig.savefig(os.path.join(mip_dir, "compression.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
            
            # Save individual channels
            for i in range(min(3, in_channels)):  # Save first 3 channels for visualization
                plt.figure(figsize=(12, 6))
                
                # Original
                plt.subplot(1, 2, 1)
                plt.imshow(original_m[0, i].cpu().numpy(), cmap='viridis')
                plt.title(f"Original - Channel {i}")
                plt.colorbar()
                
                # Reconstructed
                plt.subplot(1, 2, 2)
                plt.imshow(reconstructed[0, i].cpu().numpy(), cmap='viridis')
                plt.title(f"Reconstructed - Channel {i}")
                plt.colorbar()
                
                plt.tight_layout()
                plt.savefig(os.path.join(mip_dir, f"channel_{i}.png"), dpi=300, bbox_inches="tight")
                plt.close()
        
        # Calculate average metrics for this texture
        avg_psnr = sum(texture_psnrs) / len(texture_psnrs) if texture_psnrs else 0
        avg_ssim = sum(texture_ssims) / len(texture_ssims) if texture_ssims else 0
        
        all_psnrs.append(avg_psnr)
        all_ssims.append(avg_ssim)
        
        # Write metrics to summary file
        with open(os.path.join(output_dir, "summary.txt"), "a") as f:
            f.write(f"\n{name}:\n")
            f.write(f"  Average PSNR: {avg_psnr:.2f} dB\n")
            f.write(f"  Average SSIM: {avg_ssim:.4f}\n")
            f.write("  Per-mip metrics:\n")
            for m, (psnr, ssim) in enumerate(zip(texture_psnrs, texture_ssims)):
                f.write(f"    Mip {m}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}\n")

# Calculate overall average metrics
overall_psnr = sum(all_psnrs) / len(all_psnrs) if all_psnrs else 0
overall_ssim = sum(all_ssims) / len(all_ssims) if all_ssims else 0

# Write overall metrics to summary file
with open(os.path.join(output_dir, "summary.txt"), "a") as f:
    f.write("\nOverall Metrics:\n")
    f.write(f"  Average PSNR across all textures: {overall_psnr:.2f} dB\n")
    f.write(f"  Average SSIM across all textures: {overall_ssim:.4f}\n")

# Create a bar chart of PSNR values
plt.figure(figsize=(12, 6))
plt.bar(texture_names, all_psnrs)
plt.axhline(y=overall_psnr, color='r', linestyle='-', label=f'Average: {overall_psnr:.2f} dB')
plt.xlabel('Texture')
plt.ylabel('PSNR (dB)')
plt.title('PSNR Across Different Textures')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "psnr_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()

# Create a bar chart of SSIM values
plt.figure(figsize=(12, 6))
plt.bar(texture_names, all_ssims)
plt.axhline(y=overall_ssim, color='r', linestyle='-', label=f'Average: {overall_ssim:.4f}')
plt.xlabel('Texture')
plt.ylabel('SSIM')
plt.title('SSIM Across Different Textures')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ssim_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"\nEvaluation complete! Results saved to {output_dir}/")
print(f"Overall PSNR: {overall_psnr:.2f} dB")
print(f"Overall SSIM: {overall_ssim:.4f}")
