# <p align="center"> IC-GAN: Instance-Conditioned GAN </p>
Official Pytorch code of [Instance-Conditioned GAN](https://arxiv.org/abs/2109.05070) by Arantxa Casanova, Marlène Careil, Jakob Verbeek, Michał Drożdżal, Adriana Romero-Soriano. 
![IC-GAN results](./figures/github_image.png?raw=true)

## Generate images with IC-GAN in a Colab Notebook
We provide a [Google Colab notebook](https://colab.research.google.com/github/facebookresearch/ic_gan/blob/main/inference/icgan_colab.ipynb) to generate images with IC-GAN and its class-conditional counter part. We also invite users to check out the [demo on Replicate](https://replicate.ai/arantxacasanova/ic_gan), courtesy of [Replicate](https://replicate.ai/home).

The figure below depicts two instances, unseen during training and downloaded from [Creative Commons search](https://search.creativecommons.org), and the generated images with IC-GAN and class-conditional IC-GAN when conditioning on the class "castle":
<p align="center">
  <img src="./figures/icgan_transfer_all_github.png?raw=true">
</p>

Additionally, and inspired by [this Colab](https://colab.research.google.com/github/eyaler/clip_biggan/blob/main/ClipBigGAN.ipynb), we provide the funcionality in the same Colab notebook to guide generations with text captions, using the [CLIP model](https://github.com/openai/CLIP). 
As an example, the following Figure shows three instance conditionings and a text caption (top), followed by the resulting generated images with IC-GAN (bottom), when optimizing the noise vector following CLIP's gradient for 100 iterations. 
<p align="center">
  <img src="./figures/icgan_clip.png?raw=true">
</p>


*Credit for the three instance conditionings, from left to right, that were modified with a resize and central crop:* [1: "Landscape in Bavaria" by shining.darkness, licensed under CC BY 2.0](https://search.creativecommons.org/photos/92ef279c-4469-49a5-aa4b-48ad746f2dc4), [2: "Fantasy Landscape - slolsss" by Douglas Tofoli is marked with CC PDM 1.0](https://search.creativecommons.org/photos/13646adc-f1df-437a-a0dd-8223452ee46c), [3: "How to Draw Landscapes Simply" by Kuwagata Keisai is marked with CC0 1.0](https://search.creativecommons.org/photos/2ab9c3b7-de99-4536-81ed-604ee988bd5f)


## Requirements
* Python 3.8 
* Cuda v10.2 / Cudnn v7.6.5
* gcc v7.3.0
* Pytorch 1.8.0
* A conda environment can be created from `environment.yaml` by entering the command: `conda env create -f environment.yml`, that contains the aforemention version of Pytorch and other required packages. 
* Faiss: follow the instructions in the [original repository](https://github.com/facebookresearch/faiss).


## Overview 

This repository consists of four main folders:
* `data_utils`: A common folder to obtain and format the data needed to train and test IC-GAN, agnostic of the specific backbone. 
* `inference`: Scripts to test the models both qualitatively and quantitatively.
* `BigGAN_PyTorch`: It provides the training, evaluation and sampling scripts for IC-GAN with a BigGAN backbone. The code base comes from [Pytorch BigGAN repository](https://github.com/ajbrock/BigGAN-PyTorch), made available under the MIT License. It has been modified to [add additional utilities](#biggan-changelog) and it enables IC-GAN training on top of it.
* `stylegan2_ada_pytorch`: It provides the training, evaluation and sampling scripts for IC-GAN with a StyleGAN2 backbone. The code base comes from [StyleGAN2 Pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch), made available under the [Nvidia Source Code License](https://nvlabs.github.io/stylegan2-ada-pytorch/license.html). It has been modified to [add additional utilities](#stylegan-changelog) and it enables IC-GAN training on top of it.


## (Python script) Generate images with IC-GAN
Alternatively, we can <b> generate images with IC-GAN models </b> directly from a python script, by following the next steps:
1) Download the desired pretrained models (links below) and the [pre-computed 1000 instance features from ImageNet](https://dl.fbaipublicfiles.com/ic_gan/stored_instances.tar.gz) and extract them into a folder `pretrained_models_path`. 

| model | backbone | class-conditional? | training dataset | resolution | url |
|-------------------|-------------------|-------------------|---------------------|--------------------|--------------------|
| IC-GAN | BigGAN | No | ImageNet | 256x256 | [model](https://dl.fbaipublicfiles.com/ic_gan/icgan_biggan_imagenet_res256.tar.gz) | 
| IC-GAN (half capacity) | BigGAN | No | ImageNet | 256x256 | [model](https://dl.fbaipublicfiles.com/ic_gan/icgan_biggan_imagenet_res256_halfcap.tar.gz) | 
| IC-GAN | BigGAN | No | ImageNet | 128x128 | [model](https://dl.fbaipublicfiles.com/ic_gan/icgan_biggan_imagenet_res128.tar.gz) | 
| IC-GAN | BigGAN | No | ImageNet | 64x64 | [model](https://dl.fbaipublicfiles.com/ic_gan/icgan_biggan_imagenet_res64.tar.gz) | 
| IC-GAN | BigGAN | Yes | ImageNet | 256x256 | [model](https://dl.fbaipublicfiles.com/ic_gan/cc_icgan_biggan_imagenet_res256.tar.gz) | 
| IC-GAN (half capacity) | BigGAN | Yes | ImageNet | 256x256 | [model](https://dl.fbaipublicfiles.com/ic_gan/cc_icgan_biggan_imagenet_res256_halfcap.tar.gz) | 
| IC-GAN | BigGAN | Yes | ImageNet | 128x128 | [model](https://dl.fbaipublicfiles.com/ic_gan/cc_icgan_biggan_imagenet_res128.tar.gz) | 
| IC-GAN | BigGAN | Yes | ImageNet | 64x64 | [model](https://dl.fbaipublicfiles.com/ic_gan/cc_icgan_biggan_imagenet_res64.tar.gz) | 
| IC-GAN | BigGAN | Yes | ImageNet-LT | 256x256 | [model](https://dl.fbaipublicfiles.com/ic_gan/cc_icgan_biggan_imagenetlt_res256.tar.gz) | 
| IC-GAN | BigGAN | Yes | ImageNet-LT | 128x128 | [model](https://dl.fbaipublicfiles.com/ic_gan/cc_icgan_biggan_imagenetlt_res128.tar.gz) | 
| IC-GAN | BigGAN | Yes | ImageNet-LT | 64x64 | [model](https://dl.fbaipublicfiles.com/ic_gan/cc_icgan_biggan_imagenetlt_res64.tar.gz) | 
| IC-GAN | BigGAN | No | COCO-Stuff | 256x256 | [model](https://dl.fbaipublicfiles.com/ic_gan/icgan_biggan_coco_res256.tar.gz) | 
| IC-GAN | BigGAN | No | COCO-Stuff | 128x128 | [model](https://dl.fbaipublicfiles.com/ic_gan/icgan_biggan_coco_res128.tar.gz) | 
| IC-GAN | StyleGAN2 | No | COCO-Stuff | 256x256 | [model](https://dl.fbaipublicfiles.com/ic_gan/icgan_stylegan2_coco_res256.tar.gz) | 
| IC-GAN | StyleGAN2 | No | COCO-Stuff | 128x128 | [model](https://dl.fbaipublicfiles.com/ic_gan/icgan_stylegan2_coco_res128.tar.gz) | 

2) Execute: 
```
python inference/generate_images.py --root_path [pretrained_models_path] --model [model] --model_backbone [backbone] --resolution [res]
```
* `model` can be chosen from `["icgan", "cc_icgan"]` to use the IC-GAN or the class-conditional IC-GAN model respectively.
* `backbone` can be chosen from `["biggan", "stylegan2"]`.
* `res` indicates the resolution at which the model has been trained. For ImageNet, choose one in `[64, 128, 256]`, and for COCO-Stuff, one in `[128, 256]`.

This script results in a .PNG file where several generated images are shown, given an instance feature (each row), and a sampled noise vector (each grid position).
   
<b>Additional and optional parameters</b>:
* `index`: (None by default), is an integer from 0 to 999 that choses a specific instance feature vector out of the 1000 instances that have been selected with k-means on the ImageNet dataset and stored in `pretrained_models_path/stored_instances`.
* `swap_target`: (None by default) is an integer from 0 to 999 indicating an ImageNet class label. This label will be used to condition the class-conditional IC-GAN, regardless of which instance features are being used.
* `which_dataset`: (ImageNet by default) can be chosen from `["imagenet", "coco"]` to indicate which dataset (training split) to sample the instances from. 
* `trained_dataset`: (ImageNet by default) can be chosen from `["imagenet", "coco"]` to indicate the dataset in which the IC-GAN model has been trained on. 
* `num_imgs_gen`: (5 by default), it changes the number of noise vectors to sample per conditioning. Increasing this number results in a bigger .PNG file to save and load.
* `num_conditionings_gen`: (5 by default), it changes the number of conditionings to sample. Increasing this number results in a bigger .PNG file to save and load.
* `z_var`: (1.0 by default) controls the truncation factor for the generation. 
* Optionally, the script can be run with the following additional options `--visualize_instance_images --dataset_path [dataset_path]` to visualize the ground-truth images corresponding to the conditioning instance features, given a path to the dataset's ground-truth images `dataset_path`. Ground-truth instances will be plotted as the leftmost image for each row.

## Data preparation 
<div id="data-preparation">
<details>
<summary>ImageNet</summary>
<br>
   <ol>
      <li>Download dataset from <a href="https://image-net.org/download.php"> here </a>.
      </li>
      <li>Download <a href="https://github.com/facebookresearch/swav"> SwAV </a> feature extractor weights from <a href="https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar"> here </a>. </li>
      <li> Replace the paths in data_utils/prepare_data.sh: <code>out_path</code> by the path where hdf5 files will be stored, <code>path_imnet</code> by the path where ImageNet dataset is downloaded, and <code>path_swav</code> by the path where SwAV weights are stored. </li>
      <li> Execute <code>./data_utils/prepare_data.sh imagenet [resolution]</code>, where <code>[resolution]</code> can be an integer in {64,128,256}. This script will create several hdf5 files:
         <ul> <li> <code>ILSVRC[resolution]_xy.hdf5</code> and <code>ILSVRC[resolution]_val_xy.hdf5</code>, where images and labels are stored for the training and validation set          respectively. </li>
           <li> <code>ILSVRC[resolution]_feats_[feature_extractor]_resnet50.hdf5</code> that contains the instance features for each image. </li>
            <li> <code>ILSVRC[resolution]_feats_[feature_extractor]_resnet50_nn_k[k_nn].hdf5</code> that contains the list of [k_nn] neighbors for each of the instance features. </li> </ul> </li>
   </ol> </br>
</details>

<details>
<summary>ImageNet-LT</summary>
<br>
   <ol>
      <li>Download ImageNet dataset from <a href="https://image-net.org/download.php"> here </a>. Following <a href="https://github.com/zhmiao/OpenLongTailRecognition-OLTR"> ImageNet-LT </a>, the file <code>ImageNet_LT_train.txt</code> can be downloaded from <a href="https://drive.google.com/drive/u/1/folders/1j7Nkfe6ZhzKFXePHdsseeeGI877Xu1yf" > this link </a> and later stored in the folder <code>./BigGAN_PyTorch/imagenet_lt</code>.
      </li>
      <li>Download the pre-trained weights of the ResNet on ImageNet-LT from <a href="https://dl.fbaipublicfiles.com/classifier-balancing/ImageNet_LT/models/resnet50_uniform_e90.pth"> this link</a>, provided by the <a href="https://github.com/facebookresearch/classifier-balancing"> classifier-balancing repository </a>. </li>
      <li> Replace the paths in data_utils/prepare_data.sh: <code>out_path</code> by the path where hdf5 files will be stored, <code>path_imnet</code> by the path where ImageNet dataset is downloaded, and <code>path_classifier_lt</code> by the path where the pre-trained ResNet50 weights are stored. </li>
      <li> Execute <code>./data_utils/prepare_data.sh imagenet_lt [resolution]</code>, where <code>[resolution]</code> can be an integer in {64,128,256}. This script will create several hdf5 files:
         <ul> <li> <code>ILSVRC[resolution]longtail_xy.hdf5</code>, where images and labels are stored for the training and validation set          respectively. </li>
           <li> <code>ILSVRC[resolution]longtail_feats_[feature_extractor]_resnet50.hdf5</code> that contains the instance features for each image. </li>
            <li> <code>ILSVRC[resolution]longtail_feats_[feature_extractor]_resnet50_nn_k[k_nn].hdf5</code> that contains the list of [k_nn] neighbors for each of the instance features. </li> </ul> </li>
   </ol> </br>
</details>

<details>
<summary>COCO-Stuff</summary>
<br>
   <ol>
      <li>Download the dataset following the <a href="https://github.com/WillSuen/LostGANs/blob/master/INSTALL.md"> LostGANs' repository instructions </a>.
      </li>
      <li>Download <a href="https://github.com/facebookresearch/swav"> SwAV </a> feature extractor weights from <a href="https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar"> here </a>. </li>
      <li> Replace the paths in data_utils/prepare_data.sh: <code>out_path</code> by the path where hdf5 files will be stored, <code>path_imnet</code> by the path where ImageNet dataset is downloaded, and <code>path_swav</code> by the path where SwAV weights are stored. </li>
      <li> Execute <code>./data_utils/prepare_data.sh coco [resolution]</code>, where <code>[resolution]</code> can be an integer in {128,256}. This script will create several hdf5 files:
         <ul> <li> <code>COCO[resolution]_xy.hdf5</code> and <code>COCO[resolution]_val_test_xy.hdf5</code>, where images and labels are stored for the training and evaluation set respectively. </li>
           <li> <code>COCO[resolution]_feats_[feature_extractor]_resnet50.hdf5</code> that contains the instance features for each image. </li>
            <li> <code>COCO[resolution]_feats_[feature_extractor]_resnet50_nn_k[k_nn].hdf5</code> that contains the list of [k_nn] neighbors for each of the instance features. </li> </ul> </li>
   </ol> </br>
</details>

<details>
<summary>Other datasets</summary>
<br>
   <ol>
      <li>Download the corresponding dataset and store in a folder <code>dataset_path</code>.
      </li>
      <li>Download <a href="https://github.com/facebookresearch/swav"> SwAV </a> feature extractor weights from <a href="https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar"> here </a>. </li>
      <li> Replace the paths in data_utils/prepare_data.sh: <code>out_path</code> by the path where hdf5 files will be stored and <code>path_swav</code> by the path where SwAV weights are stored. </li>
      <li> Execute <code>./data_utils/prepare_data.sh [dataset_name] [resolution] [dataset_path]</code>, where <code>[dataset_name]</code> will be the dataset name, <code>[resolution]</code> can be an integer, for example 128 or 256, and <code>dataset_path</code> contains the dataset images. This script will create several hdf5 files:
         <ul> <li> <code>[dataset_name][resolution]_xy.hdf5</code>, where images and labels are stored for the training set. </li>
           <li> <code>[dataset_name][resolution]_feats_[feature_extractor]_resnet50.hdf5</code> that contains the instance features for each image. </li>
            <li> <code>[dataset_name][resolution]_feats_[feature_extractor]_resnet50_nn_k[k_nn].hdf5</code> that contains the list of <code>k_nn</code> neighbors for each of the instance features. </li> </ul> </li>
   </ol> </br>
</details>


<details>
<summary>How to subsample an instance feature dataset with k-means</summary>
<br>
To downsample the instance feature vector dataset, after we have prepared the data, we can use the k-means algorithm:
<code>
python data_utils/store_kmeans_indexes.py --resolution [resolution] --which_dataset [dataset_name] --data_root [data_path]
   </code>
   <ul> <li> Adding <code>--gpu</code> allows the faiss library to compute k-means leveraging GPUs, resulting in faster execution. </li>
      <li> Adding the parameter <code>--feature_extractor [feature_extractor]</code> chooses which feature extractor to use, with <code>feature_extractor</code> in <code>['selfsupervised', 'classification'] </code>, if we are using swAV as feature extactor or the ResNet pretrained on the classification task on ImageNet, respectively. </li>
      <li> The number of k-means clusters can be set with <code>--kmeans_subsampled [centers]</code>, where <code>centers</code> is an integer. </li> </ul>
</br>
</details>
</div>

## How to train the models

#### BigGAN or StyleGAN2 backbone
Training parameters are stored in JSON files in `[backbone_folder]/config_files/[dataset]/*.json`, where `[backbone_folder]` is either BigGAN_Pytorch or stylegan2_ada_pytorch and `[dataset]` can either be ImageNet, ImageNet-LT or COCO_Stuff.
```
cd BigGAN_PyTorch
python run.py --json_config config_files/<dataset>/<selected_config>.json --data_root [data_root] --base_root [base_root]
```
or 
```
cd stylegan_ada_pytorch
python run.py --json_config config_files/<dataset>/<selected_config>.json --data_root [data_root] --base_root [base_root]
```
where:
* `data_root` path where the data has been prepared and stored, following the previous section (<a href="./README.md#data-preparation">Data preparation</a>). 
* `base_root` path where to store the model weights and logs.


Note that one can create other JSON files to modify the training parameters.

#### Other backbones
To be able to run IC-GAN with other backbones, we provide some orientative steps:
* Place the new backbone code in a new folder under `ic_gan`  (`ic_gan/new_backbone`).
* Modify the relevant piece of code in the GAN architecture to allow instance features as conditionings (for both generator and discriminator). 
* Create a `trainer.py` file with the training loop to train an IC-GAN with the new backbone. The `data_utils` folder provides the tools to prepare the dataset, load the data and conditioning sampling to train an IC-GAN. The IC-GAN with BigGAN backbone [`trainer.py`](BigGAN_PyTorch/trainer.py) file can be used as an inspiration.


   
## How to test the models
<b>To obtain the FID and IS metrics on ImageNet and ImageNet-LT</b>: 
1) Execute:
``` 
python inference/test.py --json_config [BigGAN-PyTorch or stylegan-ada-pytorch]/config_files/<dataset>/<selected_config>.json --num_inception_images [num_imgs] --sample_num_npz [num_imgs] --eval_reference_set [ref_set] --sample_npz --base_root [base_root] --data_root [data_root] --kmeans_subsampled [kmeans_centers] --model_backbone [backbone]
```
To obtain the tensorflow IS and FID metrics, use an environment with the Python <3.7 and Tensorflow 1.15. Then:

2) Obtain Inception Scores and pre-computed FID moments:
 ``` 
 python ../data_utils/inception_tf13.py --experiment_name [exp_name] --experiment_root [base_root] --kmeans_subsampled [kmeans_centers] 
 ```

For stratified FIDs in the ImageNet-LT dataset, the following parameters can be added `--which_dataset 'imagenet_lt' --split 'val' --strat_name [stratified_split]`, where `stratified_split` can be in `[few,low, many]`.
    
3) (Only needed once) Pre-compute reference moments with tensorflow code:
 ```
 python ../data_utils/inception_tf13.py --use_ground_truth_data --data_root [data_root] --split [ref_set] --resolution [res] --which_dataset [dataset]
 ```

4) (Using this [repository](https://github.com/bioinf-jku/TTUR)) FID can be computed using the pre-computed statistics obtained in 2) and the pre-computed ground-truth statistics obtain in 3). For example, to compute the FID with reference ImageNet validation set: 
```python TTUR/fid.py [base_root]/[exp_name]/TF_pool_.npz [data_root]/imagenet_val_res[res]_tf_inception_moments_ground_truth.npz ``` 

<b>To obtain the FID metric on COCO-Stuff</b>:
1) Obtain ground-truth jpeg images:  ```python data_utils/store_coco_jpeg_images.py --resolution [res] --split [ref_set] --data_root [data_root] --out_path [gt_coco_images] --filter_hd [filter_hd] ```
2) Store generated images as jpeg images: ```python sample.py --json_config ../[BigGAN-PyTorch or stylegan-ada-pytorch]/config_files/<dataset>/<selected_config>.json --data_root [data_root] --base_root [base_root] --sample_num_npz [num_imgs] --which_dataset 'coco' --eval_instance_set [ref_set] --eval_reference_set [ref_set] --filter_hd [filter_hd] --model_backbone [backbone] ```
3) Using this [repository](https://github.com/bioinf-jku/TTUR), compute FID on the two folders of ground-truth and generated images.

where:
* `dataset`: option to select the dataset in `['imagenet', 'imagenet_lt', 'coco']
* `exp_name`: name of the experiment folder.
* `data_root`: path where the data has been prepared and stored, following the previous section ["Data preparation"](#data-preparation). 
* `base_root`: path where to find the model (for example, where the pretrained models have been downloaded). 
* `num_imgs`: needs to be set to 50000 for ImageNet and ImageNet-LT (with validation set as reference) and set to 11500 for ImageNet-LT (with training set as reference). For COCO-Stuff, set to 75777, 2050, 675, 1375 if using the training, evaluation, evaluation seen or evaluation unseen set as reference.
* `ref_set`: set to `'val'` for ImageNet, ImageNet-LT (and COCO) to obtain metrics with the validation (evaluation) set as reference, or set to `'train'` for ImageNet-LT or COCO to obtain metrics with the training set as reference.
* `kmeans_centers`: set to 1000 for ImageNet and to -1 for ImageNet-LT. 
* `backbone`: model backbone architecture in `['biggan','stylegan2']`.
* `res`: integer indicating the resolution of the images (64,128,256).
* `gt_coco_images`: folder to store the ground-truth JPEG images of that specific split.
* `filter_hd`: only valid for `ref_set=val`. If -1, use the entire evaluation set; if 0, use only conditionings and their ground-truth images with seen class combinations during training (eval seen); if 1, use only conditionings and their ground-truth images with unseen class combinations during training (eval unseen). 


## Utilities for GAN backbones
We change and provide extra utilities to facilitate the training, for both BigGAN and StyleGAN2 base repositories.

### BigGAN change log
The following changes were made:

* BigGAN architecture:
    * In `train_fns.py`: option to either have the optimizers inside the generator and discriminator class, or directly in the `G_D` wrapper module. Additionally, added an option to augment both generated and real images with augmentations from [DiffAugment](https://github.com/mit-han-lab/data-efficient-gans).
    * In `BigGAN.py`: added a function `get_condition_embeddings` to handle the conditioning separately.
    * Small modifications to `layers.py` to adapt the batchnorm function calls to the pytorch 1.8 version. 
    
* Training utilities: 
    * Added `trainer.py` file (replacing train.py):
        * Training now allows the usage of DDP for faster single-node and multi-node training.
        * Training is performed by epochs instead of by iterations.
        * Option to stop the training by using early stopping or when experiments diverge. 
    * In `utils.py`:
        * Replaced `MultiEpochSampler` for `CheckpointedSampler` to allow experiments to be resumable when using epochs and fixing a bug where `MultiEpochSampler` would require a long time to fetch data permutations when the number of epochs increased.
        * ImageNet-LT: Added option to use different class distributions when sampling a class label for the generator.
        * ImageNet-LT: Added class balancing (uniform and temperature annealed).
        * Added data augmentations from [DiffAugment](https://github.com/mit-han-lab/data-efficient-gans).

* Testing utilities:
    * In `calculate_inception_moments.py`: added option to obtain moments for ImageNet-LT dataset, as well as stratified moments for many, medium and few-shot classes (stratified FID computation).
    * In `inception_utils.py`: added option to compute [Precision, Recall, Density, Coverage](https://github.com/clovaai/generative-evaluation-prdc) and stratified FID.
    
* Data utilities:
    * In `datasets.py`, added option to load ImageNet-LT dataset.
    * Added ImageNet-LT.txt files with image indexes for training and validation split. 
    * In `utils.py`: 
        * Separate functions to obtain the data from hdf5 files (`get_dataset_hdf5`) or from directory (`get_dataset_images`), as well as a function to obtain only the data loader (`get_dataloader`). 
        * Added the function `sample_conditionings` to handle possible different conditionings to train G with.
        
* Experiment utilities:
    * Added JSON files to launch experiments with the proposed hyper-parameter configuration.
    * Script to launch experiments with either the [submitit tool](https://github.com/facebookincubator/submitit) or locally in the same machine (run.py). 

### StyleGAN2 change log 
<div id="stylegan-changelog">
<ul>
   <li> Multi-node DistributedDataParallel training.  </li>
   <li> Added early stopping based on the training FID metric.  </li>
   <li> Automatic checkpointing when jobs are automatically rescheduled on a cluster.  </li>
   <li> Option to load dataset from hdf5 file.  </li>
   <li> Replaced the usage of Click python package by an `ArgumentParser`. </li>
   <li> Only saving best and last model weights. </li>
   </ul>
</div>

## Acknowledgements
We would like to thanks the authors of the [Pytorch BigGAN repository](https://github.com/ajbrock/BigGAN-PyTorch) and [StyleGAN2 Pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch), as our model requires their repositories to train IC-GAN with BigGAN or StyleGAN2 bakcbone respectively. 
Moreover, we would like to further thank the authors of [generative-evaluation-prdc](https://github.com/clovaai/generative-evaluation-prdc), [data-efficient-gans](https://github.com/mit-han-lab/data-efficient-gans), [faiss](https://github.com/facebookresearch/faiss) and [sg2im](https://github.com/google/sg2im) as some components were borrowed and modified from their code bases. Finally, we thank the author of [WanderCLIP](https://colab.research.google.com/github/eyaler/clip_biggan/blob/main/WanderCLIP.ipynb) as well as the following repositories, that we use in our Colab notebook: [pytorch-pretrained-BigGAN](https://github.com/huggingface/pytorch-pretrained-BigGAN) and [CLIP](https://github.com/openai/CLIP).

## License
The majority of IC-GAN is licensed under CC-BY-NC, however portions of the project are available under separate license terms: BigGAN and [PRDC](https://github.com/facebookresearch/ic_gan/blob/main/data_utils/compute_pdrc.py) are licensed under the MIT license; [COCO-Stuff loader](https://github.com/facebookresearch/ic_gan/blob/main/data_utils/cocostuff_dataset.py) is licensed under Apache License 2.0; [DiffAugment](https://github.com/facebookresearch/ic_gan/blob/main/BigGAN_PyTorch/diffaugment_utils.py) is licensed under BSD 2-Clause Simplified license; StyleGAN2 is licensed under a NVIDIA license, available here: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/LICENSE.txt. In the Colab notebook, [CLIP](https://github.com/openai/CLIP) and [pytorch-pretrained-BigGAN](https://github.com/huggingface/pytorch-pretrained-BigGAN) code is used, both licensed under the MIT license.

## Disclaimers
THE DIFFAUGMENT SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE CLIP SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

THE PYTORCH-PRETRAINED-BIGGAN SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Cite the paper
If this repository, the paper or any of its content is useful for your research, please cite:
```
@inproceedings{casanova2021instanceconditioned,
      title={Instance-Conditioned GAN}, 
      author={Arantxa Casanova and Marlène Careil and Jakob Verbeek and Michal Drozdzal and Adriana Romero-Soriano},
      booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
      year={2021}
}
```
