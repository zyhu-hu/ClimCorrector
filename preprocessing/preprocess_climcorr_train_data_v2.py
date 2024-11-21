# preprocess_climcorr_train_data.py
import numpy as np
import h5py
import gc

def preprocess_climcorr_train_data(year):
    """
    Preprocesses ClimCorr data for a given year.
    :param year: The year as a string (e.g., '1980').
    """
    parent_path = f'/n/home00/zeyuanhu/scratch/climcorr_preprocessing/v2_iter2/{year}/'
    input_mean = np.load('/n/home00/zeyuanhu/ClimCorrector/preprocessing/normalization/inputs/input_mean_v2_iter2_40year_sub23.npy')
    input_std = np.load('/n/home00/zeyuanhu/ClimCorrector/preprocessing/normalization/inputs/input_std_v2_iter2_40year_sub23.npy')
    target_mean_dc = np.load('/n/home00/zeyuanhu/ClimCorrector/preprocessing/normalization/outputs/target_dc_mean_v2_iter2_40year_sub23.npy')
    target_std_dc = np.load('/n/home00/zeyuanhu/ClimCorrector/preprocessing/normalization/outputs/target_dc_std_v2_iter2_40year_sub23.npy')
    target_mean_sum = np.load('/n/home00/zeyuanhu/ClimCorrector/preprocessing/normalization/outputs/target_sum_mean_v2_iter2_40year_sub23.npy')
    target_std_sum = np.load('/n/home00/zeyuanhu/ClimCorrector/preprocessing/normalization/outputs/target_sum_std_v2_iter2_40year_sub23.npy')

    # make std for cloud variables to be max(std, 1e-5)
    input_std[26*4:26*6] = np.maximum(input_std[26*4:26*6], 1e-5)

    input_path = f'{parent_path}/train_input.h5'
    target_path_dc = f'{parent_path}/train_target_dc.h5'
    target_path_sum = f'{parent_path}/train_target_sum.h5'
    input_file = h5py.File(input_path, 'r')
    target_file_dc = h5py.File(target_path_dc, 'r')
    target_file_sum = h5py.File(target_path_sum, 'r')

    x = input_file['data']
    y_dc = target_file_dc['data']
    y_sum = target_file_sum['data']
    
    lat= x[:,-4]
    lon = x[:,-3]
    tod = x[:,-2]
    toy = x[:,-1]
    x = (x - input_mean) / input_std
    y_dc = (y_dc - target_mean_dc) / target_std_dc
    y_sum = (y_sum - target_mean_sum) / target_std_sum
    lat_norm = lat/90.
    lon_cos = np.cos(lon/360.*2*np.pi)
    lon_sin = np.sin(lon/360.*2*np.pi)
    tod_cos = np.cos(tod/24.*2*np.pi)
    tod_sin = np.sin(tod/24.*2*np.pi)
    toy_cos = np.cos(toy/365.*2*np.pi)
    toy_sin = np.sin(toy/365.*2*np.pi)

    x = np.concatenate((x[:,:-4], np.array([lat_norm, lon_cos, lon_sin, tod_cos, tod_sin, toy_cos, toy_sin]).T), axis=1)
    y_dc = np.clip(y_dc, -100, 100)
    y_sum = np.clip(y_sum, -100, 100)

    xshape = x.shape
    x = x.reshape(xshape[0]//(96*144),96,144,xshape[1])
    yshape = y_dc.shape
    y_dc = y_dc.reshape(yshape[0]//(96*144),96,144,yshape[1])
    y_sum = y_sum.reshape(yshape[0]//(96*144),96,144,yshape[1])
    

    assert xshape[0] == yshape[0]

    target_path = f'/n/home00/zeyuanhu/scratch/climcorr_preprocessing/v2_iter2_processed/{year}/'
    h5_path = target_path + 'train_input.h5'
    with h5py.File(h5_path, 'w') as hdf:
        hdf.create_dataset('data', data=x, dtype=np.float32)

    h5_path = target_path + 'train_target_dc.h5'
    with h5py.File(h5_path, 'w') as hdf:
        hdf.create_dataset('data', data=y_dc, dtype=np.float32)

    h5_path = target_path + 'train_target_sum.h5'
    with h5py.File(h5_path, 'w') as hdf:
        hdf.create_dataset('data', data=y_sum, dtype=np.float32)
        
    # Clear variables
    del x, y_dc, y_sum, input_mean, input_std, target_mean_dc, target_std_dc, target_mean_sum, target_std_sum
    gc.collect()  # Force garbage collection to free memory
    
    return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise ValueError("Please provide the year as an argument.")
    year = sys.argv[1]
    preprocess_climcorr_train_data(year)