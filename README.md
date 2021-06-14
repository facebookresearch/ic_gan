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
