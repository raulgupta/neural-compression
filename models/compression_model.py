import torch
import torch.nn as nn
import torch.nn.functional as F

from .global_transformer import GlobalTransformer
from .grid_constructor import GridConstructor
from .grid_sampler import GridSampler
from .texture_synthesizer import TextureSynthesizer


class CompressionModel(nn.Module):
    """
    Complete neural texture compression model.
    
    This integrates all components of the pipeline:
    1. Global Transformer (Encoder)
    2. Grid Constructor
    3. Grid Sampler
    4. Texture Synthesizer (Decoder)
    """
    
    def __init__(self, in_channels, encoder_channels=[64, 128, 256], 
                 grid_channels=[16, 16], quantization_bits=4, hidden_dim=32,
                 num_residual_blocks=4, positional_encoding_levels=10,
                 use_attention=False):
        """
        Initialize the compression model.
        
        Args:
            in_channels (int): Number of input channels in the texture set.
            encoder_channels (list): List of channel dimensions for the encoder.
            grid_channels (list): List of channel dimensions for G0 and G1.
            quantization_bits (int): Number of bits for quantization.
            hidden_dim (int): Dimension of the hidden layers in the decoder.
            num_residual_blocks (int): Number of residual blocks in the decoder.
            positional_encoding_levels (int): Number of levels for positional encoding.
            use_attention (bool): Whether to use attention blocks in the encoder.
        """
        super().__init__()
        
        # Global Transformer (Encoder)
        self.global_transformer = GlobalTransformer(
            in_channels=in_channels,
            channels=encoder_channels,
            use_attention=use_attention
        )
        
        # Grid Constructor
        self.grid_constructor = GridConstructor(
            in_channels=encoder_channels[-1],
            grid_channels=grid_channels,
            quantization_bits=quantization_bits
        )
        
        # Grid Sampler
        self.grid_sampler = GridSampler()
        
        # Texture Synthesizer (Decoder)
        self.texture_synthesizer = TextureSynthesizer(
            g0_channels=grid_channels[0],
            g1_channels=grid_channels[1],
            out_channels=in_channels,
            hidden_dim=hidden_dim,
            num_residual_blocks=num_residual_blocks,
            positional_encoding_levels=positional_encoding_levels
        )
        
    def encode(self, x, quantization_mode="uniform_noise"):
        """
        Encode a texture set.
        
        Args:
            x (torch.Tensor): Input texture set of shape [B, C, H, W].
            quantization_mode (str): Quantization mode to use.
            
        Returns:
            tuple: Pair of grid features (G0, G1).
        """
        # Global Transformer
        z = self.global_transformer(x)
        
        # Grid Constructor
        g0, g1 = self.grid_constructor(z, quantization_mode=quantization_mode)
        
        return g0, g1
    
    def decode(self, g0, g1, x, y, mip_level):
        """
        Decode texels at specific coordinates and mip level.
        
        Args:
            g0 (torch.Tensor): First grid features from the Grid Constructor.
            g1 (torch.Tensor): Second grid features from the Grid Constructor.
            x (torch.Tensor): X coordinates in range [-1, 1].
            y (torch.Tensor): Y coordinates in range [-1, 1].
            mip_level (int or torch.Tensor): Mip level to sample from.
            
        Returns:
            torch.Tensor: Reconstructed texels.
        """
        # Grid Sampler
        y0, y1 = self.grid_sampler(g0, g1, x, y, mip_level)
        
        # Texture Synthesizer
        texels = self.texture_synthesizer(y0, y1, mip_level, x, y)
        
        return texels
    
    def forward(self, x, coords=None, mip_level=0, quantization_mode="uniform_noise"):
        """
        Forward pass of the compression model.
        
        Args:
            x (torch.Tensor): Input texture set of shape [B, C, H, W].
            coords (tuple, optional): Tuple of (x, y) coordinates to decode.
                If None, random coordinates will be sampled.
            mip_level (int, optional): Mip level to sample from.
            quantization_mode (str, optional): Quantization mode to use.
            
        Returns:
            torch.Tensor: Reconstructed texels.
        """
        # Encode
        g0, g1 = self.encode(x, quantization_mode=quantization_mode)
        
        # Generate random coordinates if not provided
        if coords is None:
            batch_size = x.shape[0]
            num_points = 1024
            x_coords = torch.rand(batch_size, num_points, device=x.device) * 2 - 1
            y_coords = torch.rand(batch_size, num_points, device=x.device) * 2 - 1
        else:
            x_coords, y_coords = coords
        
        # Decode
        texels = self.decode(g0, g1, x_coords, y_coords, mip_level)
        
        return texels
    
    def get_bitrate(self, h, w):
        """
        Calculate the bitrate in bits-per-pixel-per-channel (BPPC).
        
        Args:
            h (int): Height of the original texture.
            w (int): Width of the original texture.
            
        Returns:
            float: Bitrate in BPPC.
        """
        return self.grid_constructor.get_bitrate(h, w)
    
    def reconstruct_texture(self, g0, g1, mip_level=0, resolution=None):
        """
        Reconstruct a complete texture at a specific mip level.
        
        Args:
            g0 (torch.Tensor): First grid features from the Grid Constructor.
            g1 (torch.Tensor): Second grid features from the Grid Constructor.
            mip_level (int, optional): Mip level to reconstruct.
            resolution (tuple, optional): Resolution of the output texture.
                If None, it will be inferred from the grid size and mip level.
            
        Returns:
            torch.Tensor: Reconstructed texture.
        """
        batch_size = g0.shape[0]
        
        # Infer resolution if not provided
        if resolution is None:
            grid_h, grid_w = g0.shape[2], g0.shape[3]
            h = grid_h * 8 // (2 ** mip_level)
            w = grid_w * 8 // (2 ** mip_level)
        else:
            h, w = resolution
        
        # Generate grid of coordinates
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, h, device=g0.device),
            torch.linspace(-1, 1, w, device=g0.device),
            indexing='ij'
        )
        
        # Reshape coordinates
        x_coords = x_coords.reshape(1, -1).expand(batch_size, -1)
        y_coords = y_coords.reshape(1, -1).expand(batch_size, -1)
        
        # Decode in batches to avoid OOM
        max_points = 65536  # Adjust based on available memory
        num_points = h * w
        num_batches = (num_points + max_points - 1) // max_points
        
        texels_list = []
        for i in range(num_batches):
            start_idx = i * max_points
            end_idx = min((i + 1) * max_points, num_points)
            
            x_batch = x_coords[:, start_idx:end_idx]
            y_batch = y_coords[:, start_idx:end_idx]
            
            texels_batch = self.decode(g0, g1, x_batch, y_batch, mip_level)
            texels_list.append(texels_batch)
        
        # Concatenate batches
        texels = torch.cat(texels_list, dim=1)
        
        # Reshape to texture
        texels = texels.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2)
        
        return texels


# Example usage
if __name__ == "__main__":
    # Create a sample texture set with 5 channels
    batch_size = 1
    in_channels = 5
    height, width = 256, 256
    
    x = torch.rand(batch_size, in_channels, height, width)
    
    # Create the compression model
    model = CompressionModel(in_channels)
    
    # Encode
    g0, g1 = model.encode(x)
    
    # Calculate bitrate
    bppc = model.get_bitrate(height, width)
    print(f"Bitrate: {bppc:.4f} BPPC")
    
    # Reconstruct at different mip levels
    for mip_level in range(0, 5):
        texels = model.reconstruct_texture(g0, g1, mip_level=mip_level)
        print(f"Mip level {mip_level}: {texels.shape}")
