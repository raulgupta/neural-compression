import torch
import torch.nn as nn
import torch.nn.functional as F


class GridSampler(nn.Module):
    """
    Grid Sampler for neural texture compression.
    
    As described in the paper, this samples the grid features (G0 and G1)
    based on texture coordinates and mip level.
    """
    
    def __init__(self):
        """Initialize the Grid Sampler."""
        super().__init__()
    
    def sample_grid(self, grid, x, y, mip_level, mode="nearest"):
        """
        Sample a grid at the given coordinates and mip level.
        
        Args:
            grid (torch.Tensor): Grid features of shape [B, C, H, W].
            x (torch.Tensor): X coordinates in range [-1, 1].
            y (torch.Tensor): Y coordinates in range [-1, 1].
            mip_level (int): Mip level to sample from.
            mode (str): Sampling mode, either "nearest" or "bilinear".
            
        Returns:
            torch.Tensor: Sampled features.
        """
        batch_size, channels, height, width = grid.shape
        print(f"GRID SAMPLER INPUT - grid: {grid.shape}, x: {x.shape}, y: {y.shape}, mip_level: {mip_level}")
        
        # Calculate stride for the given mip level
        # For mip levels <= 3, stride is 1
        # For mip levels > 3, stride is 2^(mip_level-3)
        stride = 2 ** max(0, mip_level - 3)
        
        # Convert normalized coordinates [-1, 1] to grid coordinates [0, H/W]
        x = (x + 1) * (width - 1) / 2
        y = (y + 1) * (height - 1) / 2
        
        if mode == "nearest":
            # Round to nearest integer
            x = torch.round(x).long()
            y = torch.round(y).long()
            
            # Apply stride
            x_tl = torch.clamp(x, 0, width - 1)
            y_tl = torch.clamp(y, 0, height - 1)
            
            # Sample the four corners with stride
            x_tr = torch.clamp(x_tl + stride, 0, width - 1)
            y_tr = y_tl
            x_bl = x_tl
            y_bl = torch.clamp(y_tl + stride, 0, height - 1)
            x_br = x_tr
            y_br = y_bl
            
            # Get batch indices
            batch_idx = torch.arange(batch_size, device=grid.device)[:, None]
            batch_idx = batch_idx.expand(-1, x.shape[1])
            
            # Sample the grid at the four corners
            tl = grid[batch_idx, :, y_tl, x_tl].transpose(1, 2)  # [B, C, N]
            tr = grid[batch_idx, :, y_tr, x_tr].transpose(1, 2)  # [B, C, N]
            bl = grid[batch_idx, :, y_bl, x_bl].transpose(1, 2)  # [B, C, N]
            br = grid[batch_idx, :, y_br, x_br].transpose(1, 2)  # [B, C, N]
            
            # Permute to match the expected dimensions [B, N, C]
            tl = tl.permute(0, 2, 1)  # [B, N, C]
            tr = tr.permute(0, 2, 1)  # [B, N, C]
            bl = bl.permute(0, 2, 1)  # [B, N, C]
            br = br.permute(0, 2, 1)  # [B, N, C]
            
            print(f"NEAREST MODE - tl: {tl.shape}, tr: {tr.shape}, bl: {bl.shape}, br: {br.shape}")
            return tl, tr, bl, br
            
        elif mode == "bilinear":
            # Calculate integer coordinates and fractional parts
            x0 = torch.floor(x).long()
            y0 = torch.floor(y).long()
            
            # Apply stride
            x0 = torch.clamp(x0, 0, width - 1 - stride)
            y0 = torch.clamp(y0, 0, height - 1 - stride)
            
            # Calculate the four corners with stride
            x1 = x0 + stride
            y1 = y0 + stride
            
            # Calculate weights for bilinear interpolation
            wx = (x - x0.float()) / stride
            wy = (y - y0.float()) / stride
            
            # Ensure weights are in [0, 1]
            wx = torch.clamp(wx, 0, 1)
            wy = torch.clamp(wy, 0, 1)
            
            # Get batch indices
            batch_idx = torch.arange(batch_size, device=grid.device)[:, None]
            batch_idx = batch_idx.expand(-1, x.shape[1])
            
            # Sample the grid at the four corners
            tl = grid[batch_idx, :, y0, x0].transpose(1, 2)  # [B, N, C]
            tr = grid[batch_idx, :, y0, x1].transpose(1, 2)  # [B, N, C]
            bl = grid[batch_idx, :, y1, x0].transpose(1, 2)  # [B, N, C]
            br = grid[batch_idx, :, y1, x1].transpose(1, 2)  # [B, N, C]
            
            print(f"BILINEAR MODE - Coordinates: x0: {x0.shape}, y0: {y0.shape}, x1: {x1.shape}, y1: {y1.shape}")
            print(f"BILINEAR MODE - Weights: wx: {wx.shape}, wy: {wy.shape}")
            print(f"BILINEAR MODE - Corners: tl: {tl.shape}, tr: {tr.shape}, bl: {bl.shape}, br: {br.shape}")
            
            # Bilinear interpolation
            print(f"BILINEAR MODE - tl shape before transpose: {tl.shape}")
            
            # The issue is that tl, tr, bl, br have shape [B, C, N] but we need [B, N, C]
            # Let's transpose them to match the expected dimensions
            tl = tl.permute(0, 2, 1)  # [B, N, C]
            tr = tr.permute(0, 2, 1)  # [B, N, C]
            bl = bl.permute(0, 2, 1)  # [B, N, C]
            br = br.permute(0, 2, 1)  # [B, N, C]
            print(f"BILINEAR MODE - After permute: tl: {tl.shape}, tr: {tr.shape}, bl: {bl.shape}, br: {br.shape}")
            
            # Now use simple broadcasting with unsqueezed weights
            wx_unsqueezed = wx.unsqueeze(-1)  # [B, N, 1]
            wy_unsqueezed = wy.unsqueeze(-1)  # [B, N, 1]
            print(f"BILINEAR MODE - Unsqueezed weights: wx_unsqueezed: {wx_unsqueezed.shape}, wy_unsqueezed: {wy_unsqueezed.shape}")
            
            # This should work now with proper broadcasting
            top = tl * (1 - wx_unsqueezed) + tr * wx_unsqueezed  # [B, N, C]
            bottom = bl * (1 - wx_unsqueezed) + br * wx_unsqueezed  # [B, N, C]
            result = top * (1 - wy_unsqueezed) + bottom * wy_unsqueezed  # [B, N, C]
            
            print(f"BILINEAR MODE - Success! top: {top.shape}, bottom: {bottom.shape}, result: {result.shape}")
            return result
            
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")
    
    def forward(self, g0, g1, x, y, mip_level):
        """
        Forward pass of the Grid Sampler.
        
        Args:
            g0 (torch.Tensor): First grid features from the Grid Constructor.
            g1 (torch.Tensor): Second grid features from the Grid Constructor.
            x (torch.Tensor): X coordinates in range [-1, 1].
            y (torch.Tensor): Y coordinates in range [-1, 1].
            mip_level (int or torch.Tensor): Mip level to sample from.
            
        Returns:
            tuple: Sampled features (Y0, Y1).
        """
        print(f"GRID SAMPLER FORWARD - g0: {g0.shape}, g1: {g1.shape}, x: {x.shape}, y: {y.shape}, mip_level: {mip_level}")
        
        # Ensure mip_level is a tensor
        if isinstance(mip_level, int):
            mip_level = torch.tensor(mip_level, device=g0.device)
        
        # Sample G0 using nearest mode (concatenate the four corners)
        tl0, tr0, bl0, br0 = self.sample_grid(g0, x, y, mip_level, mode="nearest")
        y0 = torch.cat([tl0, tr0, bl0, br0], dim=-1)  # [B, N, 4*C]
        
        # Sample G1 using bilinear mode
        y1 = self.sample_grid(g1, x, y, mip_level, mode="bilinear")  # [B, N, C]
        
        print(f"GRID SAMPLER FORWARD - Output shapes: y0: {y0.shape}, y1: {y1.shape}")
        return y0, y1


# Example usage
if __name__ == "__main__":
    # Create sample grid features
    batch_size = 4
    g0_channels = 16
    g1_channels = 16
    height, width = 32, 32  # After downscaling by 8
    
    g0 = torch.randn(batch_size, g0_channels, height, width)
    g1 = torch.randn(batch_size, g1_channels, height, width)
    
    # Create sample coordinates
    num_points = 1000
    x = torch.rand(batch_size, num_points) * 2 - 1  # Range [-1, 1]
    y = torch.rand(batch_size, num_points) * 2 - 1  # Range [-1, 1]
    
    # Create the Grid Sampler
    sampler = GridSampler()
    
    # Sample at different mip levels
    for mip_level in range(0, 10):
        y0, y1 = sampler(g0, g1, x, y, mip_level)
        
        # Print shapes
        print(f"Mip level: {mip_level}")
        print(f"Y0 shape: {y0.shape}")
        print(f"Y1 shape: {y1.shape}")
