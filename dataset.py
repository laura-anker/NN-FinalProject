import torch 
import numpy as np 
from torch.utils.data import Dataset

# Dataset class used for loading L/ab channel data for colorization
class ColorizationDataset(Dataset):
    # Create the dataset with the L/ab data and augmentation flag
    def __init__(self, l_data, ab_data, augment=False):
        self.l_data = l_data
        self.ab_data = ab_data
        self.augment = augment

        assert self.l_data.shape[0] == self.ab_data.shape[0], "Mismatch in L and ab channel counts"

    # Return the size of the dataset
    def __len__(self):
        return self.l_data.shape[0]
    
    # Load, optionally augment, normalize, and return L/ab tensors for a given index
    def __getitem__(self, idx):
        # Load the L/ab channels for the given index
        L = self.l_data[idx]        # Shape: (H, W)
        ab = self.ab_data[idx]      # Shape: (H, W, 2)

        # Optionally apply a horizontal flip augmentation
        if self.augment:
            if np.random.rand() < 0.5:
                L = np.flip(L, axis=1).copy()
                ab = np.flip(ab, axis=1).copy()

        # Normalize each of the L/ab channels 
        L = (L.astype("float32") / 255.0)                   # 0 to 255 → 0 to 1
        a = (ab[...,0].astype("float32") - 128) / 128.0     # 0 to 255 → -1 to +1
        b = (ab[...,1].astype("float32") - 128) / 128.0     # 0 to 255 → -1 to +1

        # Restack the normalized ab channels: (H, W, 2) 
        ab_norm = np.stack([a, b], axis=-1)

        # Convert to a channel first format
        L = L[np.newaxis, :, :]                 # Shape: (1, H, W)
        ab_norm = ab_norm.transpose(2, 0, 1)    # Shape: (2, H, W)

        # Return tensors for the L and ab channels
        return {
            "L": torch.from_numpy(L),
            "ab": torch.from_numpy(ab_norm),
        }