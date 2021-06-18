#!/bin/bash

resolution=$1 # 64,128,256
dataset='imagenet'
path_imnet='../anyshot_longtail/data/Imagenet_all/'
path_swav='/private/home/acasanova/anyshot_longtail/swav_pretrained/swav_800ep_pretrain.pth.tar'
path_classifier_lt='/private/home/acasanova/classifier-balancing/resnet50_joint/resnet50_uniform_e90.pth'
out_path='mock_data'
split='train'


##################
#### ImageNet ####
##################
#python data_utils/make_hdf5.py --resolution $resolution --split 'train' --data_root '../../anyshot_longtail/data/Imagenet_all/' --out_path $out_path --feature_extractor 'classification' --feature_augmentation
#python data_utils/make_hdf5.py --resolution $resolution --split 'train' --data_root '../../anyshot_longtail/data/Imagenet_all/' --out_path $out_path --save_features_only --feature_extractor 'selfsupervised' --feature_augmentation --pretrained_model_path $path_swav
#python data_utils/make_hdf5.py --resolution $resolution --split 'val' --data_root '../../anyshot_longtail/data/Imagenet_all/' --out_path 'mock_data'

# Calculate inception moments
#python data_utils/calculate_inception_moments.py --resolution $resolution --split 'train' --data_root $out_path --load_in_mem --out_path $out_path
#python data_utils/calculate_inception_moments.py --resolution $resolution --split 'val' --data_root $out_path --load_in_mem --out_path $out_path

# Compute NNs
#python data_utils/make_hdf5_nns.py --resolution $resolution --split 'train' --feature_extractor 'classification' --data_root $out_path --out_path $out_path --k_nn 500
#python data_utils/make_hdf5_nns.py --resolution $resolution --split 'train' --feature_extractor 'selfsupervised' --data_root $out_path --out_path $out_path --k_nn 500

#####################
#### ImageNet-LT ####
#####################
#python data_utils/make_hdf5.py --resolution $resolution --which_dataset 'imagenet_lt' --split 'train' --data_root $path_imnet --out_path $out_path --feature_extractor 'classification' --feature_augmentation --pretrained_model_path $path_classifier_lt
#python data_utils/make_hdf5.py --resolution $resolution --which_dataset 'imagenet_lt' --split 'val' --data_root '../../anyshot_longtail/data/Imagenet_all/' --out_path $out_path --save_images_only

# Compute NNs
#python data_utils/make_hdf5_nns.py --resolution $resolution --which_dataset 'imagenet_lt' --split 'train' --feature_extractor 'classification' --data_root $out_path --out_path $out_path --k_nn 500

# Calculate inception moments
#python data_utils/calculate_inception_moments.py --resolution $resolution --which_dataset 'imagenet_lt' --split 'train' --data_root 'mock_data' --load_in_mem
#python data_utils/calculate_inception_moments.py --resolution $resolution --split 'val' --data_root 'mock_data' --load_in_mem --out_path 'mock_data' --stratified_moments


####################
#### COCO-Stuff ####
####################
resolution=128

#coco_data_path='/datasets01/COCO/022719/train2017'
#coco_instances_path='/checkpoint/acasanova/LostGANs/datasets/coco/annotations/instances_train2017.json'
#coco_stuff_path='/checkpoint/acasanova/LostGANs/datasets/coco/annotations/stuff_train2017.json'
#python data_utils/make_hdf5.py --resolution $resolution --which_dataset 'coco' --split 'train' --data_root $coco_data_path --instance_json $coco_instances_path --stuff_json $coco_stuff_path --out_path $out_path --feature_extractor 'selfsupervised' --feature_augmentation --pretrained_model_path $path_swav
#
#coco_data_path='/datasets01/COCO/022719/val2017'
#coco_instances_path='/checkpoint/acasanova/LostGANs/datasets/coco/annotations/instances_val2017.json'
#coco_stuff_path='/checkpoint/acasanova/LostGANs/datasets/coco/annotations/stuff_val2017.json'
#python data_utils/make_hdf5.py --resolution $resolution --which_dataset 'coco' --split 'test' --data_root $coco_data_path --instance_json $coco_instances_path --stuff_json $coco_stuff_path --out_path $out_path --feature_extractor 'selfsupervised' --feature_augmentation --pretrained_model_path $path_swav


# Calculate inception moments
python data_utils/calculate_inception_moments.py --resolution $resolution --which_dataset 'coco' --split 'train' --data_root $out_path --load_in_mem --out_path $out_path

# Compute NNs
#python data_utils/make_hdf5_nns.py --resolution $resolution --which_dataset 'coco' --split 'train' --feature_extractor 'selfsupervised' --data_root $out_path --out_path $out_path --k_nn 500


#TODO: transfer