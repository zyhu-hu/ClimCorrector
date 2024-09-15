from torch.utils.data import Dataset
import numpy as np
import torch
import glob
import h5py

class climsim_dataset_h5(Dataset):
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
        self.input_paths = glob.glob(f'{parent_path}/**/train_input.h5', recursive=True)
        print('input paths:', self.input_paths)
        if not self.input_paths:
            raise FileNotFoundError("No 'train_input.h5' files found under the specified parent path.")
        self.target_paths = [path.replace('train_input.h5', 'train_target.h5') for path in self.input_paths]

        # Initialize lists to hold the samples count per file
        self.samples_per_file = []
        for input_path in self.input_paths:
            with h5py.File(input_path, 'r') as file:  # Open the file to read the number of samples
                #dataset is named 'data' in the h5 file
                self.samples_per_file.append(file['data'].shape[0])
                
        self.cumulative_samples = np.cumsum([0] + self.samples_per_file)
        self.total_samples = self.cumulative_samples[-1]

        self.input_files = {}
        self.target_files = {}
        for input_path, target_path in zip(self.input_paths, self.target_paths):
            self.input_files[input_path] = h5py.File(input_path, 'r')
            self.target_files[target_path] = h5py.File(target_path, 'r')
        
        self.input_mean = input_mean
        self.input_std = input_std
        self.target_mean = target_mean
        self.target_std = target_std
        self.input_clip = input_clip
        self.target_clip = target_clip
        self.target_clip_value = target_clip_value
    
    def __len__(self):
        return self.total_samples
    
    def _find_file_and_index(self, idx):
        file_idx = np.searchsorted(self.cumulative_samples, idx+1) - 1
        local_idx = idx - self.cumulative_samples[file_idx]
        return file_idx, local_idx
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError("Index out of bounds")
        
        file_idx, local_idx = self._find_file_and_index(idx)
        # Open the HDF5 files and read the data for the given index
        input_file = self.input_files[self.input_paths[file_idx]]
        target_file = self.target_files[self.target_paths[file_idx]]
        x = input_file['data'][local_idx]
        y = target_file['data'][local_idx]

        lat, tod,toy = x[-3:]
        x = (x - self.input_mean) / self.input_std
        y = (y - self.target_mean) / self.target_std
        lat_norm = lat/90.
        tod_cos = np.cos(tod/24.*2*np.pi)
        tod_sin = np.sin(tod/24.*2*np.pi)
        toy_cos = np.cos(toy/365.*2*np.pi)
        x = np.concatenate((x[:-3], [lat_norm, tod_cos, tod_sin, toy_cos]))
        if self.target_clip:
            y = np.clip(y, -self.target_clip_value, self.target_clip_value)
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)