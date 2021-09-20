#!/bin/bash
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# All contributions by Andy Brock:
# Copyright (c) 2019 Andy Brock
#
# MIT License
#
# ImageNet
python make_hdf5.py --resolution 64 --split 'train' --data_root '../../anyshot_longtail/data/Imagenet_all/' --out_path 'mock_data'
python calculate_inception_moments.py --resolution 64 --split 'train' --data_root 'mock_data' --load_in_mem --out_path 'mock_data'
python make_hdf5.py --resolution 64 --split 'val' --data_root '../../anyshot_longtail/data/Imagenet_all/' --out_path 'mock_data'
python calculate_inception_moments.py --resolution 64 --split 'val' --data_root 'mock_data' --load_in_mem --out_path 'mock_data'

# ImageNet-LT
python make_hdf5.py --resolution 64 --split 'train' --data_root '../../anyshot_longtail/data/Imagenet_all/' --out_path 'mock_data' --longtail
python calculate_inception_moments.py --resolution 64 --split 'train' --data_root 'mock_data' --longtail --load_in_mem
python make_hdf5.py --resolution 64 --split 'val' --data_root '../../anyshot_longtail/data/Imagenet_all/' --out_path 'mock_data'
python calculate_inception_moments.py --resolution 64 --split 'val' --data_root 'mock_data' --load_in_mem --out_path 'mock_data' --stratified_moments
