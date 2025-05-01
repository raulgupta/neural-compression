import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block as described in the paper."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        
        out += identity
        return out


class AttentionBlock(nn.Module):
    """Self-attention block for the global transformer."""
    
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Reshape for attention computation
        query = self.query(x).view(batch_size, C, -1)
        key = self.key(x).view(batch_size, C, -1).permute(0, 2, 1)
        value = self.value(x).view(batch_size, C, -1)
        
        # Compute attention map
        attention = F.softmax(torch.bmm(query, key), dim=2)
        
        # Apply attention to value
        out = torch.bmm(attention, value)
        out = out.view(batch_size, C, H, W)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        return out


class GlobalTransformer(nn.Module):
    """
    Global Transformer (Encoder) for neural texture compression.
    
    As described in the paper, this is a convolutional encoder that maps
    a texture set to a bottleneck latent representation.
    """
    
    def __init__(self, in_channels, channels=[64, 128, 256], use_attention=False):
        """
        Initialize the Global Transformer.
        
        Args:
            in_channels (int): Number of input channels in the texture set.
            channels (list): List of channel dimensions for each layer.
            use_attention (bool): Whether to use attention blocks.
        """
        super().__init__()
        
        self.use_attention = use_attention
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            # Downsampling convolution (stride 2)
            self.down_blocks.append(
                nn.Conv2d(channels[i], channels[i+1], kernel_size=5, stride=2, padding=2)
            )
            
            # Residual blocks
            self.down_blocks.append(ResidualBlock(channels[i+1]))
            
            # Attention block (optional)
            if use_attention:
                self.down_blocks.append(AttentionBlock(channels[i+1]))
        
        # Final convolution to produce latent representation
        self.final_conv = nn.Conv2d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        """
        Forward pass of the Global Transformer.
        
        Args:
            x (torch.Tensor): Input texture set of shape [B, C, H, W].
            
        Returns:
            torch.Tensor: Bottleneck latent representation of shape [B, C', H/8, W/8].
        """
        # Initial convolution
        x = self.initial_conv(x)
        
        # Downsampling blocks
        for block in self.down_blocks:
            x = block(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Apply 0.5 * tanh to constrain output to [-0.5, 0.5] as mentioned in the paper
        x = 0.5 * torch.tanh(x)
        
        return x


# Example usage
if __name__ == "__main__":
    # Create a sample texture set with 5 channels
    batch_size = 4
    in_channels = 5
    height, width = 256, 256
    
    x = torch.randn(batch_size, in_channels, height, width)
    
    # Create the Global Transformer
    transformer = GlobalTransformer(in_channels)
    
    # Forward pass
    z = transformer(x)
    
    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {z.shape}")
    print(f"Downscale factor: {height / z.shape[2]}")
