import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
import lpips


def calculate_psnr(original, reconstructed, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between original and reconstructed images.
    
    Args:
        original (torch.Tensor or np.ndarray): Original image.
        reconstructed (torch.Tensor or np.ndarray): Reconstructed image.
        max_val (float): Maximum value of the images.
        
    Returns:
        float: PSNR value in dB.
    """
    # Convert to numpy if tensors
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()
    
    # Ensure shapes match
    assert original.shape == reconstructed.shape, "Shapes must match"
    
    # Calculate MSE
    mse = np.mean((original - reconstructed) ** 2)
    
    # Avoid division by zero
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    
    return psnr

def calculate_ssim(original, reconstructed, multichannel=True):
    """
    Calculate Structural Similarity Index (SSIM) between original and reconstructed images.
    """
    print(f"SSIM - Input shapes: original: {original.shape}, reconstructed: {reconstructed.shape}")
    
    # Convert to numpy if tensors
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
        print(f"SSIM - Converted original to numpy: {original.shape}")
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()
        print(f"SSIM - Converted reconstructed to numpy: {reconstructed.shape}")
    
    # Ensure shapes match
    assert original.shape == reconstructed.shape, "Shapes must match"
    
    # For multi-channel images with batch dimension, we need to handle each image separately
    if original.ndim == 4:  # [B, C, H, W]
        # Process first image in batch for simplicity
        original = original[0]  # Now [C, H, W]
        reconstructed = reconstructed[0]  # Now [C, H, W]
    
    # Transpose if needed (SSIM expects [H, W, C])
    if original.ndim == 3 and original.shape[0] <= 15:  # Assuming C <= 15
        original = np.transpose(original, (1, 2, 0))
        reconstructed = np.transpose(reconstructed, (1, 2, 0))
        print(f"SSIM - After transpose: original: {original.shape}, reconstructed: {reconstructed.shape}")
    
    # Determine appropriate window size based on image dimensions
    min_dim = min(original.shape[0], original.shape[1])  # Now correctly using H, W dimensions
    win_size = min(7, min_dim - (min_dim % 2) + 1)  # Ensure it's odd and <= min_dim
    
    # Ensure win_size is at least 3 (minimum for SSIM)
    win_size = max(3, win_size)
    print(f"SSIM - Using window size: {win_size} for image with min dimension: {min_dim}")
    
    # Calculate SSIM
    try:
        print(f"SSIM - Calculating with multichannel={multichannel}, win_size={win_size}")
        ssim_value = ssim(
            original, 
            reconstructed, 
            data_range=1.0,
            multichannel=multichannel,
            win_size=win_size,
            channel_axis=-1 if original.ndim == 3 else None  # Explicitly specify channel axis
        )
        print(f"SSIM - Calculation successful: {ssim_value}")
    except Exception as e:
        print(f"SSIM - Calculation failed with error: {e}")
        print(f"SSIM - Image shape: {original.shape}, win_size: {win_size}")
        # Return a default value if SSIM calculation fails
        ssim_value = 0.0
    
    return ssim_value

def calculate_lpips(original, reconstructed, net_type='alex'):
    """
    Calculate Learned Perceptual Image Patch Similarity (LPIPS) between original and reconstructed images.
    
    Args:
        original (torch.Tensor): Original image in range [0, 1].
        reconstructed (torch.Tensor): Reconstructed image in range [0, 1].
        net_type (str): Network type for LPIPS ('alex', 'vgg', or 'squeeze').
        
    Returns:
        float: LPIPS value (lower is better).
    """
    # Ensure inputs are tensors
    if not isinstance(original, torch.Tensor):
        original = torch.from_numpy(original)
    if not isinstance(reconstructed, torch.Tensor):
        reconstructed = torch.from_numpy(reconstructed)
    
    # Ensure shapes match
    assert original.shape == reconstructed.shape, "Shapes must match"
    
    # Initialize LPIPS model
    loss_fn = lpips.LPIPS(net=net_type)
    
    # Move to same device as inputs
    device = original.device
    loss_fn = loss_fn.to(device)
    
    # Normalize to [-1, 1] if in [0, 1]
    if original.min() >= 0 and original.max() <= 1:
        original = 2 * original - 1
        reconstructed = 2 * reconstructed - 1
    
    # Ensure batch dimension
    if original.dim() == 3:
        original = original.unsqueeze(0)
        reconstructed = reconstructed.unsqueeze(0)
    
    # Calculate LPIPS
    with torch.no_grad():
        lpips_value = loss_fn(original, reconstructed)
    
    return lpips_value.item()


def calculate_bd_rate(rd_curve1, rd_curve2):
    """
    Calculate Bjøntegaard-Delta rate (BD-rate) between two rate-distortion curves.
    
    Args:
        rd_curve1 (tuple): First rate-distortion curve as (rates, psnrs).
        rd_curve2 (tuple): Second rate-distortion curve as (rates, psnrs).
        
    Returns:
        float: BD-rate savings in percentage.
    """
    try:
        import bjontegaard as bd
        
        # Extract rates and PSNRs
        rates1, psnrs1 = rd_curve1
        rates2, psnrs2 = rd_curve2
        
        # Calculate BD-rate using Akima interpolation (more accurate)
        bd_rate = bd.bd_rate(rates1, psnrs1, rates2, psnrs2, method='akima')
        
        return bd_rate
    except ImportError:
        print("bjontegaard package not installed. Please install it with: pip install bjontegaard")
        return None


def evaluate_texture_compression(original_textures, reconstructed_textures, metrics=["psnr", "ssim"]):
    """
    Evaluate texture compression using multiple metrics.
    
    Args:
        original_textures (dict): Dictionary of original textures at different mip levels.
        reconstructed_textures (dict): Dictionary of reconstructed textures at different mip levels.
        metrics (list): List of metrics to compute.
        
    Returns:
        dict: Dictionary of metrics.
    """
    results = {}
    
    # Ensure same mip levels
    mip_levels = set(original_textures.keys()).intersection(set(reconstructed_textures.keys()))
    
    # Compute metrics for each mip level
    for mip_level in mip_levels:
        original = original_textures[mip_level]
        reconstructed = reconstructed_textures[mip_level]
        
        # Initialize results for this mip level
        results[mip_level] = {}
        
        # Compute metrics
        if "psnr" in metrics:
            results[mip_level]["psnr"] = calculate_psnr(original, reconstructed)
        
        if "ssim" in metrics:
            results[mip_level]["ssim"] = calculate_ssim(original, reconstructed)
        
        if "lpips" in metrics:
            results[mip_level]["lpips"] = calculate_lpips(original, reconstructed)
    
    # Compute average metrics across all mip levels
    results["average"] = {}
    for metric in metrics:
        if metric in ["psnr", "ssim", "lpips"]:
            values = [results[mip_level][metric] for mip_level in mip_levels]
            results["average"][metric] = sum(values) / len(values)
    
    return results


# Example usage
if __name__ == "__main__":
    # Create sample textures
    original = torch.rand(3, 256, 256)
    reconstructed = original + 0.1 * torch.randn(3, 256, 256)
    reconstructed = torch.clamp(reconstructed, 0, 1)
    
    # Calculate metrics
    psnr = calculate_psnr(original, reconstructed)
    ssim_value = calculate_ssim(original, reconstructed)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    
    # Calculate LPIPS if available
    try:
        lpips_value = calculate_lpips(original, reconstructed)
        print(f"LPIPS: {lpips_value:.4f}")
    except:
        print("LPIPS calculation failed (lpips package may not be installed)")
    
    # Calculate BD-rate if available
    try:
        rates1 = [0.1, 0.2, 0.3, 0.4]
        psnrs1 = [30, 33, 35, 37]
        rates2 = [0.08, 0.16, 0.24, 0.32]
        psnrs2 = [30, 33, 35, 37]
        
        bd_rate = calculate_bd_rate((rates1, psnrs1), (rates2, psnrs2))
        print(f"BD-rate: {bd_rate:.2f}%")
    except:
        print("BD-rate calculation failed (bd_rate_calculator may not be installed)")
