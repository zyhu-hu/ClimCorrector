import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os
import re
import netCDF4
import copy
import string
import h5py
from tqdm import tqdm
from typing import Literal
# import torch

class data_utils:
    def __init__(self,
                 input_mean = None,
                 input_std = None,
                 output_mean = None,
                 output_std = None,
                 normalize = False,
                 save_h5=True,
                 save_npy=False):
       
        self.save_h5 = save_h5
        self.save_npy = save_npy
        self.data_path = None
        self.input_vars = []
        self.input_vars_standard = [] # those having time, lat and lon dimensions
        self.input_vars_attribute = [] # those that do not have all of time, lat and lon dimensions
        self.target_vars = []
        self.input_feature_len = None
        self.target_feature_len = None
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std
        self.normalize = normalize
        
        self.p0 = 1e5 # code assumes this will always be a scalar
        self.ps_index = None

        self.num_levels = 26

        self.train_regexps = None
        self.train_stride_sample = None
        self.train_filelist = None
        self.val_regexps = None
        self.val_stride_sample = None
        self.val_filelist = None
        self.test_regexps = None
        self.test_stride_sample = None
        self.test_filelist = None

        # physical constants from E3SM_ROOT/share/util/shr_const_mod.F90
        self.grav    = 9.80616    # acceleration of gravity ~ m/s^2
        self.cp      = 1.00464e3  # specific heat of dry air   ~ J/kg/K
        self.lv      = 2.501e6    # latent heat of evaporation ~ J/kg
        self.lf      = 3.337e5    # latent heat of fusion      ~ J/kg
        self.lsub    = self.lv + self.lf    # latent heat of sublimation ~ J/kg
        self.rho_air = 101325/(6.02214e26*1.38065e-23/28.966)/273.15 # density of dry air at STP  ~ kg/m^3
                                                                    # ~ 1.2923182846924677
                                                                    # SHR_CONST_PSTD/(SHR_CONST_RDAIR*SHR_CONST_TKFRZ)
                                                                    # SHR_CONST_RDAIR   = SHR_CONST_RGAS/SHR_CONST_MWDAIR
                                                                    # SHR_CONST_RGAS    = SHR_CONST_AVOGAD*SHR_CONST_BOLTZ
        self.rho_h20 = 1.e3       # density of fresh water     ~ kg/m^ 3
        
        self.v1_inputs_standard = ['T',
                          'Q',
                          'U',
                          'V',
                          'PS',
                          'SOLIN',
                          'LHFLX',
                          'SHFLX',
                          'SNOWHLND']
        self.v1_inputs_attribute = ['attri_lat',
                                     'TOD',
                                     'TOY']        
        self.v1_outputs = ['SDIFF',
                           'QDIFF',
                           'UDIFF',
                           'VDIFF']

        self.var_lens = {#inputs
                        'T':self.num_levels,
                        'Q':self.num_levels,
                        'U':self.num_levels,
                        'V':self.num_levels,
                        'PS':1,
                        'SOLIN':1,
                        'LHFLX':1,
                        'SHFLX':1,
                        'SNOWHLND':1,
                        #outputs
                        'SdIFF':self.num_levels,
                        'QdIFF':self.num_levels,
                        'UdIFF':self.num_levels,
                        'VdIFF':self.num_levels,
                        }
                        
    def set_to_v1_vars(self):
        '''
        This function sets the inputs and outputs to the V1 subset.
        It also indicates the index of the surface pressure variable.
        '''
        self.input_vars_standard = self.v1_inputs_standard
        self.input_vars_attribute = self.v1_inputs_attribute
        self.input_vars = self.v1_inputs_standard + self.v1_inputs_attribute
        self.target_vars = self.v1_outputs
        self.ps_index = 104
        self.input_feature_len = 112
        self.target_feature_len = 104

    def get_xrdata(self, file, file_vars_standard = None, file_vars_attribute = None):
        '''
        This function reads in a file and returns an xarray dataset with the variables specified.
        file_vars must be a list of strings.
        '''
        ds = xr.open_dataset(file, engine = 'netcdf4')

        ds_final = xr.Dataset()

        # Handle file_vars_standard
        if file_vars_standard is not None:
            ds_standard = ds[file_vars_standard].isel(time=[0, 3])
            ds_final = xr.merge([ds_final, ds_standard])

        # Handle file_vars_attribute
        if file_vars_attribute is not None:
            for var in file_vars_attribute:
                if var == 'attri_lat':
                    # Broadcast lat to have (time, lat, lon) dimensions
                    lat_data = ds['lat']
                    lat_broadcasted = np.tile(lat_data, (ds_final.sizes['time'], ds_final.sizes['lon'], 1)).transpose(0, 2, 1)
                    ds_final['attri_lat'] = (('time', 'lat', 'lon'), lat_broadcasted.astype(np.float32))
                elif var == 'TOD':
                    # Compute TOD (time of day) from ds.time
                    tod_data = (ds_final.time.dt.hour + ds_final.time.dt.minute / 60.0).values
                    tod_data = np.tile(tod_data[:, np.newaxis, np.newaxis], (1, ds_final.sizes['lat'], ds_final.sizes['lon']))
                    ds_final['TOD'] = (('time', 'lat', 'lon'), tod_data.astype(np.float32))
                elif var == 'TOY':
                    # Compute TOY (time of year) from ds.time
                    doy = ds_final.time.dt.dayofyear.values
                    toy_data = np.tile(doy[:, np.newaxis, np.newaxis], (1, ds_final.sizes['lat'], ds_final.sizes['lon']))
                    ds_final['TOY'] = (('time', 'lat', 'lon'), toy_data.astype(np.float32))

        return ds_final
    
    def get_input(self, input_file):
        '''
        This function reads in a file and returns an xarray dataset with the input variables for the emulator.
        '''
        # read inputs
        return self.get_xrdata(input_file, self.input_vars_standard, self.input_vars_attribute)


    def get_target(self, input_file):
        '''
        This function reads in a file and returns an xarray dataset with the target variables for the emulator.
        '''
        
        ds_target = self.get_xrdata(input_file, self.target_vars)
        return ds_target
    
    def set_regexps(self, data_split, regexps):
        '''
        This function sets the regular expressions used for getting the filelist for train, val, scoring, and test.
        '''
        assert data_split in ['train', 'val', 'test'], 'Provided data_split is not valid. Available options are train, val, and test.'
        if data_split == 'train':
            self.train_regexps = regexps
        elif data_split == 'val':
            self.val_regexps = regexps
        elif data_split == 'test':
            self.test_regexps = regexps

    def set_stride_sample(self, data_split, stride_sample):
        '''
        This function sets the stride_sample for train, val, scoring, and test.
        '''
        assert data_split in ['train', 'val', 'test'], 'Provided data_split is not valid. Available options are train, val, and test.'
        if data_split == 'train':
            self.train_stride_sample = stride_sample
        elif data_split == 'val':
            self.val_stride_sample = stride_sample
        elif data_split == 'test':
            self.test_stride_sample = stride_sample

    def set_filelist(self, data_split, start_idx = 0, end_idx = -1):
        '''
        This function sets the filelists corresponding to data splits for train, val, and test.
        '''
        filelist = []
        assert data_split in ['train', 'val', 'test'], 'Provided data_split is not valid. Available options are train, val, and test.'
        if data_split == 'train':
            assert self.train_regexps is not None, 'regexps for train is not set.'
            assert self.train_stride_sample is not None, 'stride_sample for train is not set.'
            for regexp in self.train_regexps:
                filelist = filelist + glob.glob(self.data_path + regexp)
            self.train_filelist = sorted(filelist)[start_idx:end_idx:self.train_stride_sample]
        elif data_split == 'val':
            assert self.val_regexps is not None, 'regexps for val is not set.'
            assert self.val_stride_sample is not None, 'stride_sample for val is not set.'
            for regexp in self.val_regexps:
                filelist = filelist + glob.glob(self.data_path + regexp)
            self.val_filelist = sorted(filelist)[start_idx:end_idx:self.val_stride_sample]
        elif data_split == 'test':
            assert self.test_regexps is not None, 'regexps for test is not set.'
            assert self.test_stride_sample is not None, 'stride_sample for test is not set.'
            for regexp in self.test_regexps:
                filelist = filelist + glob.glob(self.data_path + regexp)
            self.test_filelist = sorted(filelist)[start_idx:end_idx:self.test_stride_sample]

    def get_filelist(self, data_split):
        '''
        This function returns the filelist corresponding to data splits for train, val, and test.
        '''
        assert data_split in ['train', 'val', 'test'], 'Provided data_split is not valid. Available options are train, val, and test.'
        if data_split == 'train':
            assert self.train_filelist is not None, 'filelist for train is not set.'
            return self.train_filelist
        elif data_split == 'val':
            assert self.val_filelist is not None, 'filelist for val is not set.'
            return self.val_filelist
        elif data_split == 'test':
            assert self.test_filelist is not None, 'filelist for test is not set.'
            return self.test_filelist
        
    def load_ncdata_with_generator(self, data_split):
        '''
        This function works as a dataloader when training the emulator with raw netCDF files.
        This can be used as a dataloader during training or it can be used to create entire datasets.
        When used as a dataloader for training, I/O can slow down training considerably.
        This function also normalizes the data.
        '''
        filelist = self.get_filelist(data_split)
        def gen():
            for file in filelist:
                # read inputs
                ds_input = self.get_input(file)
                # read targets
                ds_target = self.get_target(file)
                
                # normalization, scaling
                if self.normalize:
                    ds_input = (ds_input - self.input_mean)/self.input_std
                    ds_target = (ds_target - self.output_mean)/self.output_std

                # stack
                # ds = ds.stack({'batch':{'sample','ncol'}})
                ds_input = ds_input.stack({'batch':['time', 'lat', 'lon']})
                ds_input = ds_input.to_stacked_array(new_dim='mlvar', sample_dims=['batch'])
                # dso = dso.stack({'batch':{'sample','ncol'}})
                ds_target = ds_target.stack({'batch':['time', 'lat', 'lon']})
                ds_target = ds_target.to_stacked_array(new_dim='mlvar', sample_dims=['batch'])
                yield (ds_input.values, ds_target.values)

        class IterableNumpyDataset:
            def __init__(this_self, data_generator, output_shapes):
                this_self.data_generator = data_generator
                this_self.output_shapes = output_shapes

            def __iter__(this_self):
                for item in this_self.data_generator:
                    input_array = np.array(item[0], dtype=np.float32)
                    target_array = np.array(item[1], dtype=np.float32)

                    # Assert final dimensions are correct.
                    assert input_array.shape[-1] == this_self.output_shapes[0][-1]
                    assert target_array.shape[-1] == this_self.output_shapes[1][-1]

                    yield (input_array, target_array)

            def as_numpy_iterator(this_self):
                for item in this_self.data_generator:
                    input_array = np.array(item[0], dtype=np.float32)
                    target_array = np.array(item[1], dtype=np.float32)

                    # Assert final dimensions are correct.
                    assert input_array.shape[-1] == this_self.output_shapes[0][-1]
                    assert target_array.shape[-1] == this_self.output_shapes[1][-1]

                    yield (input_array, target_array)

        dataset = IterableNumpyDataset(
            gen(),
            ((None, self.input_feature_len), (None, self.target_feature_len)),
        )

        return dataset
    
    def save_data(self,
                 data_split, 
                 save_path = ''):
        '''
        This function saves the training data as a .npy file (also with option to save .h5).
        '''
        data_loader = self.load_ncdata_with_generator(data_split)
        npy_iterator = list(data_loader.as_numpy_iterator())
        npy_input = np.concatenate([npy_iterator[x][0] for x in range(len(npy_iterator))])
        if self.normalize:
            # replace inf and nan with 0
            npy_input[np.isinf(npy_input)] = 0 
            npy_input[np.isnan(npy_input)] = 0

        # if save_path not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # add "/" to the end of save_path if it does not exist
        if save_path[-1] != '/':
            save_path = save_path + '/'

        npy_input = np.float32(npy_input)
        if self.save_npy:
            with open(save_path + data_split + '_input.npy', 'wb') as f:
                np.save(f, npy_input)
        if self.save_h5:
            h5_path = save_path + data_split + '_input.h5'
            with h5py.File(h5_path, 'w') as hdf:
                hdf.create_dataset('data', data=npy_input, dtype=npy_input.dtype)
        del npy_input
        
        npy_target = np.concatenate([npy_iterator[x][1] for x in range(len(npy_iterator))])
        npy_target = np.float32(npy_target)

        if self.save_npy:
            with open(save_path + data_split + '_target.npy', 'wb') as f:
                np.save(f, npy_target)
        if self.save_h5:
            h5_path = save_path + data_split + '_target.h5'
            with h5py.File(h5_path, 'w') as hdf:
                hdf.create_dataset('data', data=npy_target, dtype=npy_target.dtype)

    
        