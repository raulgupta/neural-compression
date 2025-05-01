import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricQuantizer(nn.Module):
    """
    Asymmetric scalar quantizer as described in the paper.
    
    This quantizer maps values to a discrete set of values within the range
    [-2^(B-1)/(2^B+1), 1/2], where B is the number of bits.
    """
    
    def __init__(self, bits=4):
        """
        Initialize the asymmetric quantizer.
        
        Args:
            bits (int): Number of bits for quantization.
        """
        super().__init__()
        self.bits = bits
        self.n_levels = 2 ** bits
        
        # Compute quantization range as described in the paper
        self.min_val = -2 ** (bits - 1) / (2 ** bits + 1)
        self.max_val = 0.5
        self.scale = (self.max_val - self.min_val) / (self.n_levels - 1)
        
    def forward(self, x, mode="uniform_noise"):
        """
        Forward pass of the asymmetric quantizer.
        
        Args:
            x (torch.Tensor): Input tensor to quantize.
            mode (str): Quantization mode:
                - "uniform_noise": Add uniform noise during training (Ballé et al. 2017)
                - "ste": Use straight-through estimator for quantization
                - "hard": Hard quantization (for inference)
                
        Returns:
            torch.Tensor: Quantized tensor.
        """
        # Clamp values to the quantization range
        x = torch.clamp(x, self.min_val, self.max_val)
        
        if mode == "uniform_noise":
            # During training, add uniform noise as a differentiable approximation
            # This is the approach from Ballé et al. 2017
            noise_scale = self.scale / 2
            noise = torch.zeros_like(x).uniform_(-noise_scale, noise_scale)
            return x + noise
            
        elif mode == "ste":
            # Straight-through estimator
            # Forward: quantize
            # Backward: pass gradients through unchanged
            x_q = self.quantize(x)
            return x + (x_q - x).detach()
            
        elif mode == "hard":
            # Hard quantization (for inference)
            return self.quantize(x)
            
        else:
            raise ValueError(f"Unknown quantization mode: {mode}")
    
    def quantize(self, x):
        """
        Perform hard quantization.
        
        Args:
            x (torch.Tensor): Input tensor to quantize.
            
        Returns:
            torch.Tensor: Quantized tensor.
        """
        # Scale to [0, n_levels-1]
        x_scaled = (x - self.min_val) / self.scale
        
        # Quantize
        x_quantized = torch.round(x_scaled)
        
        # Scale back to original range
        return x_quantized * self.scale + self.min_val


class GridConstructor(nn.Module):
    """
    Grid Constructor for neural texture compression.
    
    As described in the paper, this maps the bottleneck latent representation
    to a pair of grid features (G0 and G1).
    """
    
    def __init__(self, in_channels, grid_channels=[16, 16], quantization_bits=4):
        """
        Initialize the Grid Constructor.
        
        Args:
            in_channels (int): Number of input channels in the latent representation.
            grid_channels (list): List of channel dimensions for G0 and G1.
            quantization_bits (int): Number of bits for quantization.
        """
        super().__init__()
        
        self.grid_channels = grid_channels
        
        # Linear projections for G0 and G1
        self.projection_g0 = nn.Conv2d(in_channels, grid_channels[0], kernel_size=1)
        self.projection_g1 = nn.Conv2d(in_channels, grid_channels[1], kernel_size=1)
        
        # Quantizers for G0 and G1
        self.quantizer_g0 = AsymmetricQuantizer(bits=quantization_bits)
        self.quantizer_g1 = AsymmetricQuantizer(bits=quantization_bits)
        
    def forward(self, z, quantization_mode="uniform_noise"):
        """
        Forward pass of the Grid Constructor.
        
        Args:
            z (torch.Tensor): Bottleneck latent representation from the Global Transformer.
            quantization_mode (str): Quantization mode to use.
            
        Returns:
            tuple: Pair of grid features (G0, G1).
        """
        # Linear projections
        g0 = self.projection_g0(z)
        g1 = self.projection_g1(z)
        
        # Quantization
        g0 = self.quantizer_g0(g0, mode=quantization_mode)
        g1 = self.quantizer_g1(g1, mode=quantization_mode)
        
        return g0, g1
    
    def get_bitrate(self, h, w):
        """
        Calculate the bitrate in bits-per-pixel-per-channel (BPPC).
        
        Args:
            h (int): Height of the original texture.
            w (int): Width of the original texture.
            
        Returns:
            float: Bitrate in BPPC.
        """
        # Calculate total number of bits
        total_bits = sum(c * (h // 8) * (w // 8) * self.quantizer_g0.bits for c in self.grid_channels)
        
        # Calculate BPPC
        bppc = total_bits / (h * w)
        
        return bppc


# Example usage
if __name__ == "__main__":
    # Create a sample latent representation
    batch_size = 4
    in_channels = 256
    height, width = 32, 32  # After downscaling by 8
    
    z = torch.randn(batch_size, in_channels, height, width) * 0.1
    
    # Create the Grid Constructor
    constructor = GridConstructor(in_channels)
    
    # Forward pass
    g0, g1 = constructor(z)
    
    # Print shapes
    print(f"Input shape: {z.shape}")
    print(f"G0 shape: {g0.shape}")
    print(f"G1 shape: {g1.shape}")
    
    # Calculate bitrate
    original_height, original_width = height * 8, width * 8
    bppc = constructor.get_bitrate(original_height, original_width)
    print(f"Bitrate: {bppc:.4f} BPPC")
