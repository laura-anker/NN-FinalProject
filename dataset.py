import torch 
import numpy as np 
from torch.utils.data import Dataset, DataLoader

# TODO add comment about class 
class ColorizationDataset(Dataset):
    # Create the dataset with L/ab data and the optional transform
    def __init__(self, l_data, ab_data, augment=False):
        self.l_data = l_data
        self.ab_data = ab_data
        self.augment = augment

        assert self.l_data.shape[0] == self.ab_data.shape[0], "Mismatch in L and ab channel counts"

    # Return the size of the dataset
    def __len__(self):
        return self.l_data.shape[0]
    
    # Load, optionally augment, normalize, and return L/ab tensors for index idx
    def __getitem__(self, idx):
        # Load the L/ab channels for the given index
        L = self.l_data[idx]        # Shape: (H, W) = (224, 224)
        ab = self.ab_data[idx]      # Shape: (H, W, 2) = (224, 224, 2)

        # TODO Apply the optional transform on the raw data
        if self.augment:
            L = L
            ab = ab

        # Normalize each of the L/ab channels 
        L = (L.astype("float32") / 255.0)                   # 0 to 255 → 0 to 1
        a = (ab[...,0].astype("float32") - 128) / 128.0     # 0 to 255 → -1 to +1
        b = (ab[...,1].astype("float32") - 128) / 128.0     # 0 to 255 → -1 to +1

        # Restack the normalized ab channels: (H, W, 2) 
        ab_norm = np.stack([a, b], axis=-1)

        # Convert to a channel first format: (C, H, W)
        L = L[np.newaxis, :, :]                 # Shape: (1, 224, 224)
        ab_norm = ab_norm.transpose(2, 0, 1)    # Shape: (2, 224, 224)

        # Return tensors for the L and ab channels
        return {
            'L': torch.tensor(L, dtype=torch.float32),
            'ab': torch.tensor(ab_norm, dtype=torch.float32)
        }