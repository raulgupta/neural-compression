import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap
import io
from PIL import Image


def tensor_to_numpy(tensor):
    """
    Convert a PyTorch tensor to a numpy array.
    
    Args:
        tensor (torch.Tensor): Input tensor.
        
    Returns:
        np.ndarray: Numpy array.
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    if tensor.requires_grad:
        tensor = tensor.detach()
    
    return tensor.numpy()


def visualize_texture_set(texture_set, channel_names=None, figsize=(15, 10)):
    """
    Visualize a texture set with multiple channels.
    
    Args:
        texture_set (torch.Tensor or np.ndarray): Texture set of shape [C, H, W].
        channel_names (list, optional): Names of the channels.
        figsize (tuple, optional): Figure size.
        
    Returns:
        matplotlib.figure.Figure: Figure object.
    """
    # Convert to numpy if tensor
    if isinstance(texture_set, torch.Tensor):
        texture_set = tensor_to_numpy(texture_set)
    
    # Get number of channels
    num_channels = texture_set.shape[0]
    
    # Create default channel names if not provided
    if channel_names is None:
        channel_names = [f"Channel {i}" for i in range(num_channels)]
    
    # Create figure
    fig, axes = plt.subplots(1, num_channels, figsize=figsize)
    if num_channels == 1:
        axes = [axes]
    
    # Plot each channel
    for i, ax in enumerate(axes):
        if i < num_channels:
            # Get channel
            channel = texture_set[i]
            
            # Normalize if needed
            if channel.min() < 0 or channel.max() > 1:
                channel = (channel - channel.min()) / (channel.max() - channel.min())
            
            # Plot
            im = ax.imshow(channel, cmap='viridis')
            ax.set_title(channel_names[i])
            ax.axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig


def visualize_compression(original, reconstructed, channel_names=None, figsize=(15, 15)):
    """
    Visualize original and reconstructed textures side by side.
    
    Args:
        original (torch.Tensor or np.ndarray): Original texture set of shape [C, H, W].
        reconstructed (torch.Tensor or np.ndarray): Reconstructed texture set of shape [C, H, W].
        channel_names (list, optional): Names of the channels.
        figsize (tuple, optional): Figure size.
        
    Returns:
        matplotlib.figure.Figure: Figure object.
    """
    # Convert to numpy if tensor
    if isinstance(original, torch.Tensor):
        original = tensor_to_numpy(original)
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = tensor_to_numpy(reconstructed)
    
    # Get number of channels
    num_channels = original.shape[0]
    
    # Create default channel names if not provided
    if channel_names is None:
        channel_names = [f"Channel {i}" for i in range(num_channels)]
    
    # Create figure
    fig, axes = plt.subplots(num_channels, 3, figsize=figsize)
    if num_channels == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each channel
    for i in range(num_channels):
        # Get channel
        orig_channel = original[i]
        recon_channel = reconstructed[i]
        diff_channel = np.abs(orig_channel - recon_channel)
        
        # Normalize if needed
        if orig_channel.min() < 0 or orig_channel.max() > 1:
            orig_channel = (orig_channel - orig_channel.min()) / (orig_channel.max() - orig_channel.min())
        if recon_channel.min() < 0 or recon_channel.max() > 1:
            recon_channel = (recon_channel - recon_channel.min()) / (recon_channel.max() - recon_channel.min())
        
        # Plot original
        axes[i, 0].imshow(orig_channel, cmap='viridis')
        axes[i, 0].set_title(f"Original: {channel_names[i]}")
        axes[i, 0].axis('off')
        
        # Plot reconstructed
        axes[i, 1].imshow(recon_channel, cmap='viridis')
        axes[i, 1].set_title(f"Reconstructed: {channel_names[i]}")
        axes[i, 1].axis('off')
        
        # Plot difference
        im = axes[i, 2].imshow(diff_channel, cmap='hot')
        axes[i, 2].set_title(f"Difference: {channel_names[i]}")
        axes[i, 2].axis('off')
        
        # Add colorbar for difference
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig


def visualize_features(g0, g1, figsize=(15, 10)):
    """
    Visualize grid features G0 and G1.
    
    Args:
        g0 (torch.Tensor or np.ndarray): Grid features G0 of shape [B, C, H, W].
        g1 (torch.Tensor or np.ndarray): Grid features G1 of shape [B, C, H, W].
        figsize (tuple, optional): Figure size.
        
    Returns:
        matplotlib.figure.Figure: Figure object.
    """
    # Convert to numpy if tensor
    if isinstance(g0, torch.Tensor):
        g0 = tensor_to_numpy(g0)
    if isinstance(g1, torch.Tensor):
        g1 = tensor_to_numpy(g1)
    
    # Get batch and channel dimensions
    batch_size, g0_channels, height, width = g0.shape
    _, g1_channels, _, _ = g1.shape
    
    # Create figure
    fig, axes = plt.subplots(2, max(g0_channels, g1_channels), figsize=figsize)
    
    # Plot G0 features
    for i in range(g0_channels):
        # Get feature map (first batch)
        feature = g0[0, i]
        
        # Normalize
        feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)
        
        # Plot
        axes[0, i].imshow(feature, cmap='viridis')
        axes[0, i].set_title(f"G0 Channel {i}")
        axes[0, i].axis('off')
    
    # Plot G1 features
    for i in range(g1_channels):
        # Get feature map (first batch)
        feature = g1[0, i]
        
        # Normalize
        feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)
        
        # Plot
        axes[1, i].imshow(feature, cmap='viridis')
        axes[1, i].set_title(f"G1 Channel {i}")
        axes[1, i].axis('off')
    
    # Hide empty subplots
    for i in range(max(g0_channels, g1_channels)):
        if i >= g0_channels:
            axes[0, i].axis('off')
        if i >= g1_channels:
            axes[1, i].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_fourier_transform(g0, g1, figsize=(15, 10)):
    """
    Visualize Fourier transforms of grid features G0 and G1.
    
    Args:
        g0 (torch.Tensor or np.ndarray): Grid features G0 of shape [B, C, H, W].
        g1 (torch.Tensor or np.ndarray): Grid features G1 of shape [B, C, H, W].
        figsize (tuple, optional): Figure size.
        
    Returns:
        matplotlib.figure.Figure: Figure object.
    """
    # Convert to numpy if tensor
    if isinstance(g0, torch.Tensor):
        g0 = tensor_to_numpy(g0)
    if isinstance(g1, torch.Tensor):
        g1 = tensor_to_numpy(g1)
    
    # Get batch and channel dimensions
    batch_size, g0_channels, height, width = g0.shape
    _, g1_channels, _, _ = g1.shape
    
    # Create figure
    fig, axes = plt.subplots(2, max(g0_channels, g1_channels), figsize=figsize)
    
    # Plot G0 Fourier transforms
    for i in range(g0_channels):
        # Get feature map (first batch)
        feature = g0[0, i]
        
        # Compute Fourier transform
        fft = np.fft.fftshift(np.fft.fft2(feature))
        magnitude = np.abs(fft)
        
        # Log scale for better visualization
        magnitude = np.log1p(magnitude)
        
        # Normalize
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        
        # Plot
        axes[0, i].imshow(magnitude, cmap='viridis')
        axes[0, i].set_title(f"G0 FFT Channel {i}")
        axes[0, i].axis('off')
    
    # Plot G1 Fourier transforms
    for i in range(g1_channels):
        # Get feature map (first batch)
        feature = g1[0, i]
        
        # Compute Fourier transform
        fft = np.fft.fftshift(np.fft.fft2(feature))
        magnitude = np.abs(fft)
        
        # Log scale for better visualization
        magnitude = np.log1p(magnitude)
        
        # Normalize
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        
        # Plot
        axes[1, i].imshow(magnitude, cmap='viridis')
        axes[1, i].set_title(f"G1 FFT Channel {i}")
        axes[1, i].axis('off')
    
    # Hide empty subplots
    for i in range(max(g0_channels, g1_channels)):
        if i >= g0_channels:
            axes[0, i].axis('off')
        if i >= g1_channels:
            axes[1, i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_rate_distortion(results, metric="psnr", figsize=(10, 6)):
    """
    Plot rate-distortion curve.
    
    Args:
        results (list): List of dictionaries with 'bppc' and metric keys.
        metric (str): Metric to plot (e.g., 'psnr', 'ssim').
        figsize (tuple, optional): Figure size.
        
    Returns:
        matplotlib.figure.Figure: Figure object.
    """
    # Extract rates and metrics
    rates = [result["bppc"] for result in results]
    metrics = [result[metric] for result in results]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot rate-distortion curve
    ax.plot(rates, metrics, 'o-', linewidth=2, markersize=8)
    
    # Set labels and title
    ax.set_xlabel("Bits per pixel per channel (BPPC)")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Rate-Distortion Curve ({metric.upper()})")
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Annotate points
    for i, (rate, metric_val) in enumerate(zip(rates, metrics)):
        ax.annotate(
            f"{metric_val:.2f}",
            (rate, metric_val),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )
    
    plt.tight_layout()
    return fig


def figure_to_image(fig):
    """
    Convert a matplotlib figure to a PIL Image.
    
    Args:
        fig (matplotlib.figure.Figure): Figure to convert.
        
    Returns:
        PIL.Image: PIL Image.
    """
    # Save figure to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    
    # Create PIL Image
    img = Image.open(buf)
    
    return img


# Example usage
if __name__ == "__main__":
    # Create sample textures
    original = np.random.rand(4, 256, 256)
    reconstructed = original + 0.1 * np.random.randn(4, 256, 256)
    reconstructed = np.clip(reconstructed, 0, 1)
    
    # Create sample grid features
    g0 = np.random.rand(1, 8, 32, 32)
    g1 = np.random.rand(1, 8, 32, 32)
    
    # Visualize textures
    fig1 = visualize_texture_set(original, channel_names=["Diffuse R", "Diffuse G", "Diffuse B", "Normal"])
    plt.savefig("texture_set.png")
    
    # Visualize compression
    fig2 = visualize_compression(original, reconstructed, channel_names=["Diffuse R", "Diffuse G", "Diffuse B", "Normal"])
    plt.savefig("compression.png")
    
    # Visualize features
    fig3 = visualize_features(g0, g1)
    plt.savefig("features.png")
    
    # Visualize Fourier transforms
    fig4 = visualize_fourier_transform(g0, g1)
    plt.savefig("fourier.png")
    
    # Plot rate-distortion curve
    results = [
        {"bppc": 0.1, "psnr": 30, "ssim": 0.9},
        {"bppc": 0.2, "psnr": 33, "ssim": 0.93},
        {"bppc": 0.3, "psnr": 35, "ssim": 0.95},
        {"bppc": 0.4, "psnr": 37, "ssim": 0.97}
    ]
    fig5 = plot_rate_distortion(results)
    plt.savefig("rate_distortion.png")
    
    print("Visualization examples saved to disk")
