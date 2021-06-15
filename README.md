# IC-GAN

BigGAN
StyleGAN
ComputePRDC
DIffAugment



###BigGAN: utils modifications (link to PR)
* Experiment utilities:
    * Added JSON files to launch experiments with the proposed hyper-parameter configuration.
    * Script to launch experiments with either submitit tool or locally in the same machine (run.py). 


* Training utilities: 
    * Added trainer.py file (replacing train.py):
    * Training now allows the usage of DDP for faster single-node and multi-node training.
    * Training is performed by epochs instead of by iterations.
    * Option to stop the training by using early stopping or when experiments diverge. 
    * Replaced MultiEpochSampler  for CheckpointedSampler to allow experiments to be resumable when using epochs and fixing a bug where MultiEpochSampler would require a long time to fetch data permutations when the number of epochs increased.
    * ImageNet-LT: Added option to use different class distributions when sampling a class label for the generator.
    * ImageNet-LT: Added class balancing (uniform and temperature annealed).
    * DiffAugment set of augmentations.

* Testing utilities:
    * In calculate_inception_moments: added option to obtain moments for ImageNet-LT dataset, as well as stratified moments for many, medium and few-shot classes (stratified FID computation).
    * Option to compute PRDC (cite) and stratified FID in inception_utils.
    
* Data utilities:
    * In datasets.py, added option to load ImageNet-LT dataset.
    * Added ImageNet-LT .txt files with image indexes for training and validation split. 
    * Separate function to obtain the data from hdf5 files (get_dataset_hdf5) or from directory (get_dataset_images), as well as a function to obtain only the data loader (get_dataloader). 
    * Added the function sample_conditionings() to handle possible different conditionings to train G with.

* BigGAN architecture:
    * Added option to augment both generated and real images with augmentations from (cite DiffAugment).
    * Option to either have the optimizers inside the generator and discriminator class, or directly in the G_D wrapper module.
    * Added a function get_condition_embeddings to handle the conditioning separately.
    * Small modifications to layers.py to adapt the batchnorm function calls to the pytorch 1.8 version. 
    
    
## Data preparation 
./scripts/utils/prepare_data.sh: change data path to dataset and the output path.
## How to train the models
(Change output path and data path accordingly)
python run.py --json_config scripts/config_files/<dataset>/<selected_config>.json
## How to obtain metrics
* python test.py --json_config scripts/config_files/ImageNet/biggan_res64_ddp.json --batch_size 64 --num_inception_images 50000 --seed 0 --sample_num_npz 50000 --eval_reference_set 'val' --eval_prdc --sample_npz
* python test.py --json_config scripts/config_files/ImageNet-LT/biggan_res64.json --batch_size 64 --num_inception_images 115000 --seed 0 --sample_num_npz 115000 --eval_reference_set 'train' --sample_npz
* ###TODO: obtain TF1.3 in gan_lt_pyt1.8_ddp
* python inception_tf13.py --json_config scripts/config_files/ImageNet/biggan_res64_ddp.json




# StyleGAN2

### Train unconditional
python run.py --outdir=/checkpoint/acasanova/stylegan_training-runs --slurm_logdir /checkpoint/acasanova/submitit_logs_anyshot/ --data=/private/home/acasanova/anyshot_longtail/data/COCO128_xy.hdf5 --gpus=2 --exp_name 'stylegan2_coco128_unconditional' --aug=noaug --lrate 0.0025 --slurm=1
python run.py --outdir=/checkpoint/acasanova/stylegan_training-runs --slurm_logdir /checkpoint/acasanova/submitit_logs_anyshot/ --data=/private/home/acasanova/anyshot_longtail/data/ILSVRC128_xy.hdf5 --gpus=8 --nodes=2 --batch 1024 --exp_name 'stylegan2_imagenet128_unconditional' --aug=noaug --lrate 0.0025 --slurm=1