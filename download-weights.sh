#!/bin/sh

wget https://dl.fbaipublicfiles.com/ic_gan/cc_icgan_biggan_imagenet_res256.tar.gz
tar -xvf cc_icgan_biggan_imagenet_res256.tar.gz
wget https://dl.fbaipublicfiles.com/ic_gan/icgan_biggan_imagenet_res256.tar.gz
tar -xvf icgan_biggan_imagenet_res256.tar.gz
wget https://dl.fbaipublicfiles.com/ic_gan/stored_instances.tar.gz
tar -xvf stored_instances.tar.gz
curl -L -o swav_pretrained.pth.tar -C - 'https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar'
