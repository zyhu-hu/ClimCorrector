from torch.utils.data import Dataset
import numpy as np
import torch
import glob
import h5py

class climsim_dataset_processed_h5_preload(Dataset):
    def __init__(self, 
                 parent_path, 
                 ):
        """
        This "preload" version of the climsim_dataset_h5 class loads the entire dataset into memory. 
        This code is intended to the validation dataset which only has one input/target file.
        Args:
            parent_path (str): Path to the parent directory containing the .h5 files.
        """
        self.parent_path = parent_path
        self.input_path = f'{parent_path}/val_input.h5'
        self.target_path = f'{parent_path}/val_target.h5'
        print('input paths:', self.input_path)
        print('target paths:', self.target_path)

    
        self.input_file = h5py.File(self.input_path, 'r')
        self.target_file = h5py.File(self.target_path, 'r')
        self.total_samples = self.input_file['data'].shape[0]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError("Index out of bounds")

        x = self.input_file['data'][idx]
        y = self.target_file['data'][idx]
        
        lat, tod,toy = x[-3:]
        x = (x - self.input_mean) / self.input_std
        y = (y - self.target_mean) / self.target_std
        
        # Convert numpy arrays to torch tensors with float32 dtype
        x = torch.from_numpy(x).to(torch.float32)
        y = torch.from_numpy(y).to(torch.float32)

        return x, y