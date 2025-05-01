import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time
import random
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CompressionModel
from utils.data_loader import TextureDataset, TextureDataLoader
from utils.metrics import calculate_psnr, calculate_ssim
from utils.visualization import visualize_compression, visualize_features, figure_to_image


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train neural texture compression model")
    parser.add_argument("--config", type=str, default="experiments/configs/default.yaml",
                        help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use")
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_model(config, in_channels):
    """Create compression model from configuration."""
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
    return model


def create_optimizer(config, model):
    """Create optimizer from configuration."""
    optimizer_config = config["training"]["optimizer"]
    
    if optimizer_config["type"].lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["training"]["stages"][0]["learning_rate"],
            betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
            weight_decay=optimizer_config.get("weight_decay", 0.0)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_config['type']}")
    
    return optimizer


def create_loss_function(config):
    """Create loss function from configuration."""
    loss_config = config["training"]["loss"]
    
    if loss_config["type"].lower() == "mse":
        loss_fn = nn.MSELoss()
    elif loss_config["type"].lower() == "mse+lpips":
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="alex")
            
            def loss_fn(pred, target):
                mse_loss = nn.MSELoss()(pred, target)
                lpips_loss = lpips_fn(pred, target).mean()
                return mse_loss + loss_config["lambda_lpips"] * lpips_loss
        except ImportError:
            print("LPIPS not installed. Falling back to MSE loss.")
            loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_config['type']}")
    
    return loss_fn


def sample_coordinates(batch_size, num_points, device):
    """Sample random coordinates in range [-1, 1]."""
    x = torch.rand(batch_size, num_points, device=device) * 2 - 1
    y = torch.rand(batch_size, num_points, device=device) * 2 - 1
    return x, y


def sample_mip_level(max_mip_level, batch_size, device, uniform_prob=0.1):
    """
    Sample mip level with exponential distribution.
    
    Args:
        max_mip_level (int): Maximum mip level.
        batch_size (int): Batch size.
        device (torch.device): Device to create tensor on.
        uniform_prob (float): Probability of sampling from uniform distribution.
        
    Returns:
        torch.Tensor: Sampled mip levels.
    """
    if random.random() < uniform_prob:
        # Sample from uniform distribution
        mip_levels = torch.randint(0, max_mip_level + 1, (batch_size,), device=device)
    else:
        # Sample from exponential distribution
        # Use rate parameter log(4) as in the paper
        rate = np.log(4)
        mip_levels = torch.tensor(
            np.random.exponential(1 / rate, batch_size).astype(int),
            device=device
        )
        # Clamp to valid range
        mip_levels = torch.clamp(mip_levels, 0, max_mip_level)
    
    return mip_levels


def train_step(model, texture_set, optimizer, loss_fn, device, mip_levels, num_points=1024):
    """
    Perform a single training step.
    
    Args:
        model (CompressionModel): Compression model.
        texture_set (torch.Tensor): Texture set of shape [B, C, H, W].
        optimizer (torch.optim.Optimizer): Optimizer.
        loss_fn (callable): Loss function.
        device (torch.device): Device to use.
        mip_levels (list): List of mip levels to train on.
        num_points (int): Number of points to sample per batch.
        
    Returns:
        float: Loss value.
    """
    # Move texture set to device
    texture_set = texture_set.to(device)
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Encode texture set
    g0, g1 = model.encode(texture_set)
    
    # Sample random coordinates
    batch_size = texture_set.shape[0]
    x, y = sample_coordinates(batch_size, num_points, device)
    
    # Sample random mip level for each batch
    mip_level = sample_mip_level(max(mip_levels), batch_size, device)
    
    # Initialize loss
    total_loss = 0.0
    
    # Process each batch separately
    for i in range(batch_size):
        # Get mip level for this batch
        m = mip_level[i].item()
        
        # Skip if mip level is not in the list
        if m not in mip_levels:
            continue
        
        # Calculate downscale factor
        scale_factor = 2 ** m
        
        # Downsample texture set to get ground truth
        h, w = texture_set.shape[2], texture_set.shape[3]
        h_m, w_m = h // scale_factor, w // scale_factor
        
        # Use interpolation to downsample
        texture_m = torch.nn.functional.interpolate(
            texture_set[i:i+1], size=(h_m, w_m), mode="bilinear", align_corners=False
        )
        
        # Decode at sampled coordinates
        texels = model.decode(
            g0[i:i+1], g1[i:i+1], x[i:i+1], y[i:i+1], m
        )
        
        # Sample ground truth at the same coordinates
        # Convert normalized coordinates [-1, 1] to pixel coordinates [0, h_m-1] and [0, w_m-1]
        x_pixel = ((x[i] + 1) * (w_m - 1) / 2).long().clamp(0, w_m - 1)
        y_pixel = ((y[i] + 1) * (h_m - 1) / 2).long().clamp(0, h_m - 1)
        
        # Gather ground truth texels
        gt_texels = []
        for j in range(texture_m.shape[1]):
            gt_texels.append(texture_m[0, j, y_pixel, x_pixel])
        gt_texels = torch.stack(gt_texels, dim=1)
        
        # Compute loss
        loss = loss_fn(texels, gt_texels)
        total_loss += loss
    
    # Normalize loss by batch size
    total_loss /= batch_size
    
    # Backward pass
    total_loss.backward()
    
    # Update weights
    optimizer.step()
    
    return total_loss.item()


def validate(model, texture_set, loss_fn, device, mip_levels, num_points=1024):
    """
    Validate the model on a texture set.
    
    Args:
        model (CompressionModel): Compression model.
        texture_set (torch.Tensor): Texture set of shape [B, C, H, W].
        loss_fn (callable): Loss function.
        device (torch.device): Device to use.
        mip_levels (list): List of mip levels to validate on.
        num_points (int): Number of points to sample per batch.
        
    Returns:
        dict: Dictionary of metrics.
    """
    # Move texture set to device
    texture_set = texture_set.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Encode texture set
    with torch.no_grad():
        g0, g1 = model.encode(texture_set, quantization_mode="hard")
    
    # Initialize metrics
    metrics = {m: {"loss": 0.0, "psnr": 0.0, "ssim": 0.0} for m in mip_levels}
    
    # Process each mip level
    for m in mip_levels:
        # Calculate downscale factor
        scale_factor = 2 ** m
        
        # Downsample texture set to get ground truth
        h, w = texture_set.shape[2], texture_set.shape[3]
        h_m, w_m = h // scale_factor, w // scale_factor
        
        # Use interpolation to downsample
        texture_m = torch.nn.functional.interpolate(
            texture_set, size=(h_m, w_m), mode="bilinear", align_corners=False
        )
        
        # Reconstruct full texture at this mip level
        with torch.no_grad():
            reconstructed = model.reconstruct_texture(g0, g1, mip_level=m)
            
            # Ensure reconstructed texture has the same size as the ground truth
            if reconstructed.shape[2:] != texture_m.shape[2:]:
                print(f"Resizing reconstructed texture from {reconstructed.shape[2:]} to {texture_m.shape[2:]}")
                reconstructed = torch.nn.functional.interpolate(
                    reconstructed, size=texture_m.shape[2:], mode="bilinear", align_corners=False
                )
        
        # Compute metrics
        metrics[m]["loss"] = loss_fn(reconstructed, texture_m).item()
        metrics[m]["psnr"] = calculate_psnr(reconstructed.cpu(), texture_m.cpu())
        metrics[m]["ssim"] = calculate_ssim(reconstructed.cpu(), texture_m.cpu())
    
    # Compute average metrics
    avg_metrics = {
        "loss": sum(metrics[m]["loss"] for m in mip_levels) / len(mip_levels),
        "psnr": sum(metrics[m]["psnr"] for m in mip_levels) / len(mip_levels),
        "ssim": sum(metrics[m]["ssim"] for m in mip_levels) / len(mip_levels)
    }
    
    # Set model back to training mode
    model.train()
    
    return {"mip_levels": metrics, "average": avg_metrics}


def train(config, output_dir, device):
    """
    Train the neural texture compression model.
    
    Args:
        config (dict): Configuration dictionary.
        output_dir (str): Directory to save results.
        device (torch.device): Device to use.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create tensorboard writer
    writer = SummaryWriter(os.path.join(output_dir, "logs"))
    
    # Create dataset
    dataset = TextureDataset(
        root_dir=config["dataset"]["texture_dir"],
        crop_size=config["dataset"]["crop_size"]
    )
    
    # Create data loader
    dataloader = TextureDataLoader(
        dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"]
    )
    
    # Get a sample batch to determine input channels
    sample_batch = next(iter(dataloader))
    in_channels = sample_batch["texture_set"].shape[1]
    
    # Create model
    model = create_model(config, in_channels)
    model = model.to(device)
    
    # Create optimizer
    optimizer = create_optimizer(config, model)
    
    # Create loss function
    loss_fn = create_loss_function(config)
    if isinstance(loss_fn, nn.Module):
        loss_fn = loss_fn.to(device)
    
    # Training loop
    global_step = 0
    
    # Iterate through training stages
    for stage_idx, stage_config in enumerate(config["training"]["stages"]):
        print(f"Starting training stage {stage_idx + 1}/{len(config['training']['stages'])}")
        
        # Update crop size
        dataset.crop_size = stage_config["crop_size"]
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = stage_config["learning_rate"]
        
        # Get mip levels for this stage
        mip_levels = stage_config["mip_levels"]
        
        # Get quantization mode for this stage
        quantization_mode = stage_config["quantization"]
        
        # Training steps for this stage
        steps = stage_config["steps"]
        
        # Progress bar
        pbar = tqdm(range(steps), desc=f"Stage {stage_idx + 1}")
        
        # Iterate through steps
        for step in pbar:
            # Get batch
            batch = next(iter(dataloader))
            texture_set = batch["texture_set"]
            
            # Train step
            loss = train_step(
                model, texture_set, optimizer, loss_fn, device, mip_levels
            )
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss:.4f}"})
            
            # Log to tensorboard
            writer.add_scalar("train/loss", loss, global_step)
            
            # Validate more frequently for quick tests
            validation_frequency = 500  # Reduced from 1000 for quicker feedback
            if global_step % validation_frequency == 0:
                # Validate
                metrics = validate(model, texture_set, loss_fn, device, mip_levels)
                
                # Log to tensorboard
                writer.add_scalar("val/loss", metrics["average"]["loss"], global_step)
                writer.add_scalar("val/psnr", metrics["average"]["psnr"], global_step)
                writer.add_scalar("val/ssim", metrics["average"]["ssim"], global_step)
                
                # Log mip level metrics
                for m in mip_levels:
                    writer.add_scalar(f"val/loss_mip{m}", metrics["mip_levels"][m]["loss"], global_step)
                    writer.add_scalar(f"val/psnr_mip{m}", metrics["mip_levels"][m]["psnr"], global_step)
                    writer.add_scalar(f"val/ssim_mip{m}", metrics["mip_levels"][m]["ssim"], global_step)
                
                # Visualize results
                with torch.no_grad():
                    # Encode
                    g0, g1 = model.encode(texture_set.to(device), quantization_mode="hard")
                    
                    # Reconstruct at mip level 0
                    reconstructed = model.reconstruct_texture(g0, g1, mip_level=0)
                    
                    # Ensure reconstructed texture has the same size as the input for visualization
                    if reconstructed.shape[2:] != texture_set.shape[2:]:
                        print(f"Resizing reconstructed texture for visualization from {reconstructed.shape[2:]} to {texture_set.shape[2:]}")
                        reconstructed = torch.nn.functional.interpolate(
                            reconstructed, size=texture_set.shape[2:], mode="bilinear", align_corners=False
                        )
                    
                    # Visualize compression
                    fig = visualize_compression(
                        texture_set[0].cpu(),
                        reconstructed[0].cpu(),
                        channel_names=[f"Channel {i}" for i in range(texture_set.shape[1])]
                    )
                    writer.add_figure("compression", fig, global_step)
                    
                    # Visualize features
                    fig = visualize_features(g0.cpu(), g1.cpu())
                    writer.add_figure("features", fig, global_step)
                
                # Save model checkpoint
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "stage": stage_idx,
                    "config": config
                }, os.path.join(output_dir, f"checkpoint_{global_step}.pth"))
            
            # Increment global step
            global_step += 1
    
    # Final validation
    metrics = validate(model, texture_set, loss_fn, device, mip_levels)
    
    # Log final metrics
    print("Final metrics:")
    print(f"Average PSNR: {metrics['average']['psnr']:.2f} dB")
    print(f"Average SSIM: {metrics['average']['ssim']:.4f}")
    
    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "stage": len(config["training"]["stages"]) - 1,
        "config": config
    }, os.path.join(output_dir, "model_final.pth"))
    
    # Close tensorboard writer
    writer.close()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    # Train model
    train(config, output_dir, device)


if __name__ == "__main__":
    main()
