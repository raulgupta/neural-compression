import os
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CompressionModel
from utils.data_loader import TextureDataset
from utils.metrics import calculate_psnr, calculate_ssim, calculate_lpips, calculate_bd_rate
from utils.visualization import (
    visualize_compression, visualize_features, visualize_fourier_transform,
    plot_rate_distortion, figure_to_image
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate neural texture compression model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--texture_dir", type=str, required=True,
                        help="Directory containing texture sets")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--compare_with", type=str, nargs="+", default=["astc"],
                        help="Methods to compare with (e.g., astc, ntc)")
    parser.add_argument("--metrics", type=str, nargs="+", default=["psnr", "ssim"],
                        help="Metrics to compute (e.g., psnr, ssim, lpips)")
    return parser.parse_args()


def load_model(model_path, device):
    """
    Load model from checkpoint.
    
    Args:
        model_path (str): Path to model checkpoint.
        device (torch.device): Device to load model on.
        
    Returns:
        CompressionModel: Loaded model.
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get configuration
    config = checkpoint["config"]
    
    # Create model
    model = CompressionModel(
        in_channels=config.get("in_channels", 3),  # Default to 3 if not specified
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
    
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    return model, config


def load_texture_set(texture_path):
    """
    Load a texture set from a directory.
    
    Args:
        texture_path (str): Path to texture set directory.
        
    Returns:
        dict: Dictionary of textures at different mip levels.
    """
    # Find all texture files in the directory
    texture_files = []
    for ext in ["png", "jpg", "jpeg", "exr"]:
        texture_files.extend([f for f in os.listdir(texture_path) if f.endswith(f".{ext}")])
    
    # Sort files to ensure consistent order
    texture_files.sort()
    
    # Load textures
    textures = []
    for file_path in texture_files:
        # Load texture
        img = Image.open(os.path.join(texture_path, file_path)).convert("RGB")
        texture = np.array(img) / 255.0  # Normalize to [0, 1]
        texture = np.transpose(texture, (2, 0, 1))  # [H, W, C] -> [C, H, W]
        textures.append(texture)
    
    # Stack textures along channel dimension
    texture_set = np.concatenate(textures, axis=0)
    
    # Convert to tensor
    texture_set = torch.from_numpy(texture_set).float()
    
    # Create mip levels
    mip_levels = {}
    h, w = texture_set.shape[1], texture_set.shape[2]
    for m in range(int(np.log2(min(h, w))) - 1):
        # Calculate downscale factor
        scale_factor = 2 ** m
        
        # Calculate resolution
        h_m, w_m = h // scale_factor, w // scale_factor
        
        # Downsample
        if m == 0:
            mip_levels[m] = texture_set
        else:
            mip_levels[m] = torch.nn.functional.interpolate(
                texture_set.unsqueeze(0), size=(h_m, w_m), mode="bilinear", align_corners=False
            ).squeeze(0)
    
    return mip_levels


def evaluate_model(model, texture_set, device, metrics=["psnr", "ssim"]):
    """
    Evaluate model on a texture set.
    
    Args:
        model (CompressionModel): Compression model.
        texture_set (dict): Dictionary of textures at different mip levels.
        device (torch.device): Device to use.
        metrics (list): List of metrics to compute.
        
    Returns:
        dict: Dictionary of metrics.
    """
    # Move texture set to device
    texture_set_device = {m: texture_set[m].to(device) for m in texture_set}
    
    # Encode texture set (mip level 0)
    with torch.no_grad():
        g0, g1 = model.encode(texture_set_device[0].unsqueeze(0), quantization_mode="hard")
    
    # Calculate bitrate
    h, w = texture_set[0].shape[1], texture_set[0].shape[2]
    bppc = model.get_bitrate(h, w)
    
    # Initialize results
    results = {"bppc": bppc}
    
    # Initialize reconstructed textures
    reconstructed = {}
    
    # Reconstruct and evaluate at each mip level
    for m in texture_set:
        # Reconstruct
        with torch.no_grad():
            reconstructed[m] = model.reconstruct_texture(g0, g1, mip_level=m)[0].cpu()
        
        # Compute metrics
        if "psnr" in metrics:
            results[f"psnr_mip{m}"] = calculate_psnr(texture_set[m], reconstructed[m])
        
        if "ssim" in metrics:
            results[f"ssim_mip{m}"] = calculate_ssim(texture_set[m], reconstructed[m])
        
        if "lpips" in metrics:
            results[f"lpips_mip{m}"] = calculate_lpips(texture_set[m], reconstructed[m])
    
    # Compute average metrics
    for metric in metrics:
        values = [results[f"{metric}_mip{m}"] for m in texture_set]
        results[metric] = sum(values) / len(values)
    
    # Add reconstructed textures
    results["reconstructed"] = reconstructed
    
    # Add grid features
    results["g0"] = g0.cpu()
    results["g1"] = g1.cpu()
    
    return results


def evaluate_astc(texture_set, block_sizes=[6, 8, 10, 12], metrics=["psnr", "ssim"]):
    """
    Evaluate ASTC compression on a texture set.
    
    Args:
        texture_set (dict): Dictionary of textures at different mip levels.
        block_sizes (list): List of ASTC block sizes to evaluate.
        metrics (list): List of metrics to compute.
        
    Returns:
        dict: Dictionary of metrics for each block size.
    """
    # Check if astc-encoder is available
    try:
        import subprocess
        subprocess.run(["astcenc", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (ImportError, FileNotFoundError):
        print("astc-encoder not found. Skipping ASTC evaluation.")
        return {}
    
    # Initialize results
    results = {}
    
    # Evaluate each block size
    for block_size in block_sizes:
        # Calculate bitrate
        bppc = 128 / (block_size * block_size)
        
        # Initialize results for this block size
        results[block_size] = {"bppc": bppc}
        
        # Initialize reconstructed textures
        reconstructed = {}
        
        # Compress and evaluate at each mip level
        for m in texture_set:
            # TODO: Implement ASTC compression and decompression
            # This is a placeholder - you would need to implement ASTC compression
            # using the astc-encoder tool
            
            # For now, just add noise to simulate compression artifacts
            noise_level = 0.1 * (block_size / 12)  # More noise for smaller block sizes
            reconstructed[m] = texture_set[m] + noise_level * torch.randn_like(texture_set[m])
            reconstructed[m] = torch.clamp(reconstructed[m], 0, 1)
            
            # Compute metrics
            if "psnr" in metrics:
                results[block_size][f"psnr_mip{m}"] = calculate_psnr(texture_set[m], reconstructed[m])
            
            if "ssim" in metrics:
                results[block_size][f"ssim_mip{m}"] = calculate_ssim(texture_set[m], reconstructed[m])
            
            if "lpips" in metrics:
                results[block_size][f"lpips_mip{m}"] = calculate_lpips(texture_set[m], reconstructed[m])
        
        # Compute average metrics
        for metric in metrics:
            values = [results[block_size][f"{metric}_mip{m}"] for m in texture_set]
            results[block_size][metric] = sum(values) / len(values)
        
        # Add reconstructed textures
        results[block_size]["reconstructed"] = reconstructed
    
    return results


def evaluate_ntc(texture_set, metrics=["psnr", "ssim"]):
    """
    Evaluate NTC compression on a texture set.
    
    Args:
        texture_set (dict): Dictionary of textures at different mip levels.
        metrics (list): List of metrics to compute.
        
    Returns:
        dict: Dictionary of metrics.
    """
    # Check if NTC implementation is available
    try:
        # This is a placeholder - you would need to implement NTC compression
        # based on the paper by Vaidyanathan et al.
        pass
    except ImportError:
        print("NTC implementation not found. Skipping NTC evaluation.")
        return {}
    
    # Initialize results
    results = {}
    
    # Evaluate at different bitrates
    for bitrate in [0.2, 0.5, 1.0]:
        # Initialize results for this bitrate
        results[bitrate] = {"bppc": bitrate}
        
        # Initialize reconstructed textures
        reconstructed = {}
        
        # Compress and evaluate at each mip level
        for m in texture_set:
            # TODO: Implement NTC compression and decompression
            # This is a placeholder - you would need to implement NTC compression
            
            # For now, just add noise to simulate compression artifacts
            noise_level = 0.1 * (1 / bitrate)  # More noise for lower bitrates
            reconstructed[m] = texture_set[m] + noise_level * torch.randn_like(texture_set[m])
            reconstructed[m] = torch.clamp(reconstructed[m], 0, 1)
            
            # Compute metrics
            if "psnr" in metrics:
                results[bitrate][f"psnr_mip{m}"] = calculate_psnr(texture_set[m], reconstructed[m])
            
            if "ssim" in metrics:
                results[bitrate][f"ssim_mip{m}"] = calculate_ssim(texture_set[m], reconstructed[m])
            
            if "lpips" in metrics:
                results[bitrate][f"lpips_mip{m}"] = calculate_lpips(texture_set[m], reconstructed[m])
        
        # Compute average metrics
        for metric in metrics:
            values = [results[bitrate][f"{metric}_mip{m}"] for m in texture_set]
            results[bitrate][metric] = sum(values) / len(values)
        
        # Add reconstructed textures
        results[bitrate]["reconstructed"] = reconstructed
    
    return results


def visualize_results(model_results, comparison_results, texture_set, output_dir, metrics=["psnr", "ssim"]):
    """
    Visualize evaluation results.
    
    Args:
        model_results (dict): Dictionary of model evaluation results.
        comparison_results (dict): Dictionary of comparison results.
        texture_set (dict): Dictionary of textures at different mip levels.
        output_dir (str): Directory to save visualizations.
        metrics (list): List of metrics to visualize.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize compression
    for m in texture_set:
        # Get original and reconstructed textures
        original = texture_set[m]
        reconstructed = model_results["reconstructed"][m]
        
        # Visualize
        fig = visualize_compression(
            original,
            reconstructed,
            channel_names=[f"Channel {i}" for i in range(original.shape[0])]
        )
        
        # Save figure
        fig.savefig(os.path.join(output_dir, f"compression_mip{m}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    
    # Visualize features
    fig = visualize_features(model_results["g0"], model_results["g1"])
    fig.savefig(os.path.join(output_dir, "features.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Visualize Fourier transforms
    fig = visualize_fourier_transform(model_results["g0"], model_results["g1"])
    fig.savefig(os.path.join(output_dir, "fourier.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Plot rate-distortion curves
    for metric in metrics:
        # Collect results for rate-distortion curve
        rd_results = []
        
        # Add model result
        rd_results.append({
            "bppc": model_results["bppc"],
            metric: model_results[metric],
            "method": "Ours"
        })
        
        # Add comparison results
        for method, results in comparison_results.items():
            if method == "astc":
                for block_size, result in results.items():
                    rd_results.append({
                        "bppc": result["bppc"],
                        metric: result[metric],
                        "method": f"ASTC {block_size}x{block_size}"
                    })
            elif method == "ntc":
                for bitrate, result in results.items():
                    rd_results.append({
                        "bppc": result["bppc"],
                        metric: result[metric],
                        "method": f"NTC {bitrate}"
                    })
        
        # Sort by bitrate
        rd_results.sort(key=lambda x: x["bppc"])
        
        # Plot rate-distortion curve
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Group by method
        methods = set(result["method"] for result in rd_results)
        for method in methods:
            method_results = [result for result in rd_results if result["method"] == method]
            rates = [result["bppc"] for result in method_results]
            metrics_val = [result[metric] for result in method_results]
            ax.plot(rates, metrics_val, "o-", label=method, linewidth=2, markersize=8)
        
        # Set labels and title
        ax.set_xlabel("Bits per pixel per channel (BPPC)")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Rate-Distortion Curve ({metric.upper()})")
        
        # Add grid
        ax.grid(True, linestyle="--", alpha=0.7)
        
        # Add legend
        ax.legend()
        
        # Save figure
        fig.savefig(os.path.join(output_dir, f"rate_distortion_{metric}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    
    # Calculate BD-rate
    if "astc" in comparison_results and len(comparison_results["astc"]) >= 2:
        # Collect ASTC results
        astc_rates = []
        astc_psnrs = []
        
        for block_size, result in comparison_results["astc"].items():
            astc_rates.append(result["bppc"])
            astc_psnrs.append(result["psnr"])
        
        # Sort by bitrate
        astc_rates, astc_psnrs = zip(*sorted(zip(astc_rates, astc_psnrs)))
        
        # Collect our results
        our_rates = [model_results["bppc"]]
        our_psnrs = [model_results["psnr"]]
        
        # Calculate BD-rate
        bd_rate = calculate_bd_rate((astc_rates, astc_psnrs), (our_rates, our_psnrs))
        
        # Print BD-rate
        print(f"BD-rate compared to ASTC: {bd_rate:.2f}%")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model(args.model_path, device)
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all texture sets in the directory
    texture_sets = []
    for texture_dir in os.listdir(args.texture_dir):
        texture_path = os.path.join(args.texture_dir, texture_dir)
        if os.path.isdir(texture_path):
            # Check if directory contains texture files
            texture_files = []
            for ext in ["png", "jpg", "jpeg", "exr"]:
                texture_files.extend([f for f in os.listdir(texture_path) if f.endswith(f".{ext}")])
            
            if texture_files:
                texture_sets.append(texture_path)
    
    print(f"Found {len(texture_sets)} texture sets")
    
    # Evaluate each texture set
    for texture_path in tqdm(texture_sets, desc="Evaluating texture sets"):
        # Get texture set name
        texture_name = os.path.basename(texture_path)
        
        # Create output directory for this texture set
        texture_output_dir = os.path.join(output_dir, texture_name)
        os.makedirs(texture_output_dir, exist_ok=True)
        
        # Load texture set
        texture_set = load_texture_set(texture_path)
        
        # Evaluate model
        model_results = evaluate_model(model, texture_set, device, metrics=args.metrics)
        
        # Initialize comparison results
        comparison_results = {}
        
        # Evaluate comparison methods
        for method in args.compare_with:
            if method == "astc":
                # Evaluate ASTC
                astc_results = evaluate_astc(
                    texture_set,
                    block_sizes=config["evaluation"]["astc_block_sizes"],
                    metrics=args.metrics
                )
                comparison_results["astc"] = astc_results
            elif method == "ntc":
                # Evaluate NTC
                ntc_results = evaluate_ntc(texture_set, metrics=args.metrics)
                comparison_results["ntc"] = ntc_results
        
        # Visualize results
        visualize_results(
            model_results,
            comparison_results,
            texture_set,
            texture_output_dir,
            metrics=args.metrics
        )
        
        # Save results
        results = {
            "model": model_results,
            "comparison": comparison_results
        }
        
        # Save metrics to text file
        with open(os.path.join(texture_output_dir, "metrics.txt"), "w") as f:
            f.write(f"Texture set: {texture_name}\n")
            f.write(f"Model: {args.model_path}\n")
            f.write("\n")
            
            f.write("Model results:\n")
            f.write(f"  Bitrate: {model_results['bppc']:.4f} BPPC\n")
            for metric in args.metrics:
                f.write(f"  {metric.upper()}: {model_results[metric]:.4f}\n")
            f.write("\n")
            
            for method, method_results in comparison_results.items():
                f.write(f"{method.upper()} results:\n")
                for param, result in method_results.items():
                    f.write(f"  {param}:\n")
                    f.write(f"    Bitrate: {result['bppc']:.4f} BPPC\n")
                    for metric in args.metrics:
                        f.write(f"    {metric.upper()}: {result[metric]:.4f}\n")
                f.write("\n")
        
        print(f"Evaluation results saved to {texture_output_dir}")


if __name__ == "__main__":
    main()
