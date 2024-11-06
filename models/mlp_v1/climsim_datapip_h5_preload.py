from torch.utils.data import Dataset
import numpy as np
import torch
import glob
import h5py
import nvtx

class climsim_dataset_h5_preload(Dataset):
    def __init__(self, 
                 parent_path, 
                 input_mean,
                 input_std,
                 target_mean,
                 target_std,         
                 input_clip=False,
                 target_clip=True,
                 target_clip_value=100):
        """
        This "preload" version of the climsim_dataset_h5 class loads the entire dataset into memory. 
        This code is intended to the validation dataset which only has one input/target file.
        Args:
            parent_path (str): Path to the parent directory containing the .h5 files.
            input_mean (float): Mean of the input data.
            input_std (float): Standard deviation of the input data.
            target_mean (float): Mean of the target data.
            target_std (float): Standard deviation of the target data.
            input_clip (bool): Whether to clip the input data. not implemented yet
            target_clip (bool): Whether to clip the target data.
            target_clip_value (float): Value to clip the target data to. the QDIFF contains huge negative extremes and may worth clipping.
        """
        self.parent_path = parent_path
        self.input_path = f'{parent_path}/val_input.h5'
        self.target_path = f'{parent_path}/val_target.h5'
        print('input paths:', self.input_path)
        print('target paths:', self.target_path)

    
        self.input_file = h5py.File(self.input_path, 'r')
        self.target_file = h5py.File(self.target_path, 'r')
        self.total_samples = self.input_file['data'].shape[0]

        self.input_mean = input_mean
        self.input_std = input_std
        self.target_mean = target_mean
        self.target_std = target_std
        self.input_clip = input_clip
        self.target_clip = target_clip
        self.target_clip_value = target_clip_value
    
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
        lat_norm = lat/90.
        tod_cos = np.cos(tod/24.*2*np.pi)
        tod_sin = np.sin(tod/24.*2*np.pi)
        toy_cos = np.cos(toy/365.*2*np.pi)
        toy_sin = np.sin(toy/365.*2*np.pi)
        x = np.concatenate((x[:-3], [lat_norm, tod_cos, tod_sin, toy_cos, toy_sin]))
        if self.target_clip:
            y = np.clip(y, -self.target_clip_value, self.target_clip_value)
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)