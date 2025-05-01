import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding as described in the paper.
    
    This is based on the encoding used in NeRF and similar methods,
    which maps coordinates to a higher-dimensional space using
    sinusoidal functions at different frequencies.
    """
    
    def __init__(self, num_levels=10, include_identity=True):
        """
        Initialize the positional encoding.
        
        Args:
            num_levels (int): Number of frequency levels to use.
            include_identity (bool): Whether to include the original coordinates.
        """
        super().__init__()
        self.num_levels = num_levels
        self.include_identity = include_identity
        
        # Frequency multipliers: 2^0, 2^1, 2^2, ...
        self.freq_bands = 2 ** torch.arange(num_levels).float()
        
    def forward(self, x, y):
        """
        Forward pass of the positional encoding.
        
        Args:
            x (torch.Tensor): X coordinates in range [-1, 1].
            y (torch.Tensor): Y coordinates in range [-1, 1].
            
        Returns:
            torch.Tensor: Encoded coordinates.
        """
        # Ensure inputs are float tensors
        x = x.float()
        y = y.float()
        
        # Reshape to [batch_size, num_points, 1]
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
        
        # Apply frequency bands
        x_enc = x * self.freq_bands.to(x.device)
        y_enc = y * self.freq_bands.to(y.device)
        
        # Apply sin and cos to each frequency
        x_sin = torch.sin(x_enc)
        x_cos = torch.cos(x_enc)
        y_sin = torch.sin(y_enc)
        y_cos = torch.cos(y_enc)
        
        # Concatenate all encodings
        out = torch.cat([x_sin, x_cos, y_sin, y_cos], dim=-1)
        
        # Optionally include the original coordinates
        if self.include_identity:
            out = torch.cat([x, y, out], dim=-1)
        
        return out


class LinearResidualBlock(nn.Module):
    """
    Linear residual block for the texture synthesizer.
    
    This is a fully connected version of the residual block,
    used in the decoder part of the pipeline.
    """
    
    def __init__(self, dim):
        """
        Initialize the linear residual block.
        
        Args:
            dim (int): Dimension of the input and output.
        """
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.linear3 = nn.Linear(dim, dim)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        """
        Forward pass of the linear residual block.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        identity = x
        
        out = self.linear1(x)
        out = self.leaky_relu(out)
        out = self.linear2(out)
        out = self.leaky_relu(out)
        out = self.linear3(out)
        
        out += identity
        return out


class TextureSynthesizer(nn.Module):
    """
    Texture Synthesizer (Decoder) for neural texture compression.
    
    As described in the paper, this is a fully connected network that
    reconstructs texels at specific positions and mip levels.
    """
    
    def __init__(self, g0_channels, g1_channels, out_channels, hidden_dim=32, 
                 num_residual_blocks=4, positional_encoding_levels=10):
        """
        Initialize the Texture Synthesizer.
        
        Args:
            g0_channels (int): Number of channels in G0 (multiplied by 4 due to concatenation).
            g1_channels (int): Number of channels in G1.
            out_channels (int): Number of output channels (texture channels).
            hidden_dim (int): Dimension of the hidden layers.
            num_residual_blocks (int): Number of residual blocks to use.
            positional_encoding_levels (int): Number of levels for positional encoding.
        """
        super().__init__()
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            num_levels=positional_encoding_levels,
            include_identity=True
        )
        
        # Calculate input dimension
        # 4*g0_channels: concatenated features from G0 (4 corners)
        # g1_channels: features from G1
        # 1: normalized mip level
        # 4*positional_encoding_levels + 2: positional encoding (if include_identity=True)
        pos_enc_dim = 4 * positional_encoding_levels + 2
        input_dim = 4 * g0_channels + g1_channels + 1 + pos_enc_dim
        
        # Initial linear layer
        self.initial_linear = nn.Linear(input_dim, hidden_dim)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            LinearResidualBlock(hidden_dim) for _ in range(num_residual_blocks)
        ])
        
        # Final linear layer
        self.final_linear = nn.Linear(hidden_dim, out_channels)
        
    def forward(self, y0, y1, mip_level, x, y):
        """
        Forward pass of the Texture Synthesizer.
        
        Args:
            y0 (torch.Tensor): Sampled features from G0 of shape [B, N, 4*C].
            y1 (torch.Tensor): Sampled features from G1 of shape [B, N, C].
            mip_level (torch.Tensor): Normalized mip level in range [0, 1].
            x (torch.Tensor): X coordinates in range [-1, 1].
            y (torch.Tensor): Y coordinates in range [-1, 1].
            
        Returns:
            torch.Tensor: Reconstructed texels of shape [B, N, out_channels].
        """
        # Ensure mip_level is a tensor with the right shape
        if isinstance(mip_level, (int, float)):
            mip_level = torch.tensor(mip_level, device=y0.device)
        
        # Normalize mip level to [0, 1]
        if mip_level.dim() == 0:
            # Scalar mip level
            mip_level = mip_level.view(1, 1).expand(y0.shape[0], y0.shape[1])
        
        # Apply positional encoding
        pos_enc = self.positional_encoding(x, y)
        
        # Concatenate all inputs
        mip_level = mip_level.unsqueeze(-1)  # [B, N, 1]
        inputs = torch.cat([y0, y1, mip_level, pos_enc], dim=-1)
        
        # Initial linear layer
        x = self.initial_linear(inputs)
        x = self.leaky_relu(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Final linear layer
        x = self.final_linear(x)
        
        # Sigmoid activation to ensure output is in [0, 1]
        x = torch.sigmoid(x)
        
        return x


# Example usage
if __name__ == "__main__":
    # Create sample inputs
    batch_size = 4
    num_points = 1000
    g0_channels = 16
    g1_channels = 16
    out_channels = 5  # e.g., RGB + normal + displacement
    
    y0 = torch.randn(batch_size, num_points, 4 * g0_channels)
    y1 = torch.randn(batch_size, num_points, g1_channels)
    mip_level = 3
    x = torch.rand(batch_size, num_points) * 2 - 1  # Range [-1, 1]
    y = torch.rand(batch_size, num_points) * 2 - 1  # Range [-1, 1]
    
    # Create the Texture Synthesizer
    synthesizer = TextureSynthesizer(g0_channels, g1_channels, out_channels)
    
    # Forward pass
    texels = synthesizer(y0, y1, mip_level, x, y)
    
    # Print shapes
    print(f"Y0 shape: {y0.shape}")
    print(f"Y1 shape: {y1.shape}")
    print(f"Output shape: {texels.shape}")
