#!/bin/bash
#python make_hdf5.py --dataset I128 --batch_size 256 --data_root data
#python calculate_inception_moments.py --dataset I128_hdf5 --data_root data


#python make_hdf5.py --resolution 64 --split 'train' --data_root 'data/Imagenet_all/' --out_path 'mock_data'

#python make_hdf5.py --resolution 64 --split 'train' --data_root '../../anyshot_longtail/data/Imagenet_all/' --out_path 'mock_data' --longtail
#python make_hdf5.py --resolution 64 --split 'train' --data_root '../../anyshot_longtail/data/Imagenet_all/' --out_path 'mock_data'

python calculate_inception_moments.py --resolution 64 --split 'train' --data_root 'mock_data' --longtail --load_in_mem
python calculate_inception_moments.py --resolution 64 --split 'train' --data_root 'mock_data' --load_in_mem --out_path 'mock_data'
