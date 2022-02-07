#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

resolution=$2 # 64,128,256
dataset=$1 #'imagenet', 'imagenet_lt',  'coco', [a transfer dataset, such as 'cityscapes']
out_path=''
path_imnet=''
path_swav='swav_800ep_pretrain.pth.tar'
path_classifier_lt='resnet50_uniform_e90.pth'


##################
#### ImageNet ####
##################
if [ $dataset = 'imagenet' ]; then
  python data_utils/make_hdf5.py --resolution $resolution --split 'train' --data_root $path_imnet --out_path $out_path --feature_extractor 'classification' --feature_augmentation
  python data_utils/make_hdf5.py --resolution $resolution --split 'train' --data_root $path_imnet --out_path $out_path --save_features_only --feature_extractor 'selfsupervised' --feature_augmentation --pretrained_model_path $path_swav
  python data_utils/make_hdf5.py --resolution $resolution --split 'val' --data_root $path_imnet --out_path $out_path --save_images_only
  ## Calculate inception moments
  for split in 'train' 'val'; do
    python data_utils/calculate_inception_moments.py --resolution $resolution --split $split --data_root $out_path --load_in_mem --out_path $out_path
  done
  # Compute NNs
  python data_utils/make_hdf5_nns.py --resolution $resolution --split 'train' --feature_extractor 'classification' --data_root $out_path --out_path $out_path --k_nn 50
  python data_utils/make_hdf5_nns.py --resolution $resolution --split 'train' --feature_extractor 'selfsupervised' --data_root $out_path --out_path $out_path --k_nn 50

elif [ $dataset = 'imagenet_lt' ]; then
  python data_utils/make_hdf5.py --resolution $resolution --which_dataset 'imagenet_lt' --split 'train' --data_root $path_imnet --out_path $out_path --feature_extractor 'classification' --feature_augmentation --pretrained_model_path $path_classifier_lt
  python data_utils/make_hdf5.py --resolution $resolution --which_dataset 'imagenet_lt' --split 'val' --data_root $path_imnet --out_path $out_path --save_images_only
  # Calculate inception moments
  python data_utils/calculate_inception_moments.py --resolution $resolution --which_dataset 'imagenet_lt' --split 'train' --data_root $out_path --out_path $out_path
  python data_utils/calculate_inception_moments.py --resolution $resolution --split 'val' --data_root $out_path --out_path $out_path --stratified_moments
  # Compute NNs
  python data_utils/make_hdf5_nns.py --resolution $resolution --which_dataset 'imagenet_lt' --split 'train' --feature_extractor 'classification' --data_root $out_path --out_path $out_path --k_nn 5

elif [ $dataset = 'coco' ]; then
  path_split=("train" "val")
  split=("train" "test")
  for i in "${!path_split[@]}"; do
    coco_data_path='COCO/022719/'${path_split[i]}'2017'
    coco_instances_path='datasets/coco/annotations/instances_'${path_split[i]}'2017.json'
    coco_stuff_path='datasets/coco/annotations/stuff_'${path_split[i]}'2017.json'
    python data_utils/make_hdf5.py --resolution $resolution --which_dataset 'coco' --split ${split[i]} --data_root $coco_data_path --instance_json $coco_instances_path --stuff_json $coco_stuff_path --out_path $out_path --feature_extractor 'selfsupervised' --feature_augmentation --pretrained_model_path $path_swav
    python data_utils/make_hdf5.py --resolution $resolution --which_dataset 'coco' --split ${split[i]} --data_root $coco_data_path --instance_json $coco_instances_path --stuff_json $coco_stuff_path --out_path $out_path --feature_extractor 'classification' --feature_augmentation

    # Calculate inception moments
    python data_utils/calculate_inception_moments.py --resolution $resolution --which_dataset 'coco' --split ${split[i]} --data_root $out_path --load_in_mem --out_path $out_path
    # Compute NNs
    python data_utils/make_hdf5_nns.py --resolution $resolution --which_dataset 'coco' --split ${split[i]} --feature_extractor 'selfsupervised' --data_root $out_path --out_path $out_path --k_nn 5
    python data_utils/make_hdf5_nns.py --resolution $resolution --which_dataset 'coco' --split ${split[i]} --feature_extractor 'classification' --data_root $out_path --out_path $out_path --k_nn 5

  done
# Transfer datasets
else
  python data_utils/make_hdf5.py --resolution $resolution --which_dataset $dataset --split 'train' --data_root $3 --feature_extractor 'classification' --out_path $out_path
    # Compute NNs
  python data_utils/make_hdf5.py --resolution $resolution --which_dataset $dataset --split 'train' --data_root $3 --feature_extractor 'selfsupervised' --pretrained_model_path $path_swav --save_features_only --out_path $out_path
    # Compute NNs
  # Compute NNs
  python data_utils/make_hdf5_nns.py --resolution $resolution --which_dataset $dataset --split 'train' --feature_extractor 'classification' --data_root $out_path --out_path $out_path --k_nn 5
  python data_utils/make_hdf5_nns.py --resolution $resolution --which_dataset $dataset --split 'train' --feature_extractor 'selfsupervised' --data_root $out_path --out_path $out_path --k_nn 5

fi
