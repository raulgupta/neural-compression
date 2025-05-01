import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class TextureDataset(Dataset):
    """
    Dataset for loading texture sets.
    
    A texture set consists of multiple textures (e.g., diffuse, normal, displacement)
    that are stored as separate image files in a directory.
    """
    
    def __init__(self, root_dir, crop_size=256, transform=None, extensions=["png", "jpg", "jpeg", "exr"]):
        """
        Initialize the texture dataset.
        
        Args:
            root_dir (str): Directory containing texture sets.
            crop_size (int): Size of random crops during training.
            transform (callable, optional): Optional transform to be applied on a sample.
            extensions (list): List of file extensions to consider.
        """
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.transform = transform
        self.extensions = extensions
        
        # Find all texture set directories
        self.texture_sets = []
        for texture_dir in os.listdir(root_dir):
            texture_path = os.path.join(root_dir, texture_dir)
            if os.path.isdir(texture_path):
                # Check if directory contains texture files
                texture_files = []
                for ext in extensions:
                    texture_files.extend(glob.glob(os.path.join(texture_path, f"*.{ext}")))
                
                if texture_files:
                    self.texture_sets.append(texture_path)
        
        print(f"Found {len(self.texture_sets)} texture sets")
    
    def __len__(self):
        return len(self.texture_sets)
    
    def __getitem__(self, idx):
        texture_path = self.texture_sets[idx]
        
        # Load all texture files in the directory
        texture_files = []
        for ext in self.extensions:
            texture_files.extend(glob.glob(os.path.join(texture_path, f"*.{ext}")))
        
        # Sort files to ensure consistent order
        texture_files.sort()
        
        # Load textures
        textures = []
        for file_path in texture_files:
            texture = self._load_texture(file_path)
            textures.append(texture)
        
        # Stack textures along channel dimension
        texture_set = torch.cat(textures, dim=0)
        
        # Apply random crop
        if self.crop_size > 0:
            texture_set = self._random_crop(texture_set, self.crop_size)
        
        # Apply additional transforms
        if self.transform:
            texture_set = self.transform(texture_set)
        
        return {
            "texture_set": texture_set,
            "path": texture_path,
            "num_channels": texture_set.shape[0]
        }
    
    def _load_texture(self, file_path):
        """
        Load a texture file.
        
        Args:
            file_path (str): Path to the texture file.
            
        Returns:
            torch.Tensor: Loaded texture.
        """
        # Handle different file formats
        if file_path.endswith(".exr"):
            # For EXR files, use OpenEXR or similar library
            # This is a placeholder - you would need to implement EXR loading
            raise NotImplementedError("EXR loading not implemented")
        else:
            # For standard image formats
            img = Image.open(file_path).convert("RGB")
            texture = TF.to_tensor(img)  # Scales to [0, 1]
        
        return texture
    
    def _random_crop(self, texture, crop_size):
        """
        Apply random crop to a texture.
        
        Args:
            texture (torch.Tensor): Texture to crop.
            crop_size (int): Size of the crop.
            
        Returns:
            torch.Tensor: Cropped texture.
        """
        _, h, w = texture.shape
        
        # Ensure crop size is not larger than texture
        crop_size = min(crop_size, h, w)
        
        # Random crop
        top = torch.randint(0, h - crop_size + 1, (1,)).item()
        left = torch.randint(0, w - crop_size + 1, (1,)).item()
        
        return texture[:, top:top+crop_size, left:left+crop_size]


class TextureDataLoader:
    """
    Data loader for texture sets.
    
    This is a wrapper around torch.utils.data.DataLoader that provides
    additional functionality for texture sets.
    """
    
    def __init__(self, dataset, batch_size=4, shuffle=True, num_workers=4):
        """
        Initialize the texture data loader.
        
        Args:
            dataset (TextureDataset): Dataset to load from.
            batch_size (int): Batch size.
            shuffle (bool): Whether to shuffle the data.
            num_workers (int): Number of worker threads.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # Create PyTorch DataLoader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


# Example usage
if __name__ == "__main__":
    # Create dataset
    dataset = TextureDataset(
        root_dir="data/textures",
        crop_size=256
    )
    
    # Create data loader
    dataloader = TextureDataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )
    
    # Iterate over batches
    for batch in dataloader:
        texture_set = batch["texture_set"]
        path = batch["path"]
        num_channels = batch["num_channels"]
        
        print(f"Texture set shape: {texture_set.shape}")
        print(f"Path: {path}")
        print(f"Number of channels: {num_channels}")
        break
