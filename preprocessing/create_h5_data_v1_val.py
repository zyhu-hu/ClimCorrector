from utils.data_utils import *
import argparse

def main(regexps, data_path, save_path, start_idx, stride_sample):
    data = data_utils(normalize=False,
                  save_h5=True,
                  save_npy=False
                  )
    # set the path to model outputs
    data.data_path = data_path
    # set the input and target features
    data.set_to_v1_vars()
    print(regexps)
    # data.set_regexps(data_split='train', regexps=regexps)
    # data.set_stride_sample(data_split='train', stride_sample=2)
    # data.set_filelist(data_split='train', start_idx=start_idx)

    # set regular expressions for selecting training data
    data.set_regexps(data_split = 'val', 
                    regexps = regexps)
    # set temporal subsampling
    data.set_stride_sample(data_split = 'val', stride_sample = stride_sample)
    # create list of files to extract data from
    data.set_filelist(data_split = 'val', start_idx=start_idx)
    print('files to be processed (top 10 and last 10 are printed out below):')
    print(data.get_filelist('val')[:10])
    print(data.get_filelist('val')[-10:])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data.save_data(data_split='val', save_path=save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process E3SM-MMF data.')
    parser.add_argument('regexps', type=str, nargs='+', help='Regular expressions for selecting data files.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data files.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save processed data.')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index of the data file to be processed.')
    parser.add_argument('--stride_sample', type=int, default=1, help='Temporal subsampling rate.')
    args = parser.parse_args()

    main(args.regexps, args.data_path, args.save_path, args.start_idx, args.stride_sample)