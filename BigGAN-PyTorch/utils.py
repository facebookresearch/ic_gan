#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Utilities file
This file contains utility functions for bookkeeping, logging, and data loading.
Methods which directly affect training should either go in layers, the model,
or train_fns.py.
'''

from __future__ import print_function
import sys
import os
import numpy as np
import time
import datetime
import json
import pickle
from argparse import ArgumentParser
import animal_hash
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import datasets as dset
import shutil
import torch.distributed as dist

import h5py as h5

def prepare_parser():
    usage = 'Parser for all scripts.'
    parser = ArgumentParser(description=usage)

    parser.add_argument("--json_config", type=str, default='',
                        help="Json config from where to load the configuration parameters.")

    ### Dataset/Dataloader stuff ###
    parser.add_argument(
        '--resolution', type=int, default=64,
        help='Resolution to train with '
             '(default: %(default)s)')
    parser.add_argument(
        '--augment', action='store_true', default=False,
        help='Augment with random crops and flips (default: %(default)s)')
    parser.add_argument(
        '--num_workers', type=int, default=8,
        help='Number of dataloader workers; consider using less for HDF5 '
             '(default: %(default)s)')
    parser.add_argument(
        '--no_pin_memory', action='store_false', dest='pin_memory', default=True,
        help='Pin data into memory through dataloader? (default: %(default)s)')
    parser.add_argument(
        '--shuffle', action='store_true', default=False,
        help='Shuffle the data (strongly recommended)? (default: %(default)s)')
    parser.add_argument(
        '--load_in_mem', action='store_true', default=False,
        help='Load all data into memory? (default: %(default)s)')
    parser.add_argument(
        '--use_multiepoch_sampler', action='store_true', default=False,
        help='Use the multi-epoch sampler for dataloader? (default: %(default)s)')
    parser.add_argument(
        '--use_checkpointable_sampler', action='store_true', default=False,
        help='Use the checkpointable sampler for dataloader? (default: %(default)s)')
    parser.add_argument(
        '--use_balanced_sampler', action='store_true', default=False,
        help='Use the class balanced sampler for dataloader? (default: %(default)s)')
    parser.add_argument(
        '--longtail_temperature', type=int, default=1,
        help='Temperature to relax longtail_distribution')

    parser.add_argument(
        '--longtail', action='store_true', default=False,
        help='Use long-tail version of the dataset')
    parser.add_argument(
        '--longtail_gen', action='store_true', default=False,
        help='Use long-tail version of class conditioning sampling for generator.')
    parser.add_argument(
        '--custom_distrib_gen', action='store_true', default=False,
        help='Use custom distribution for sampling class conditionings in generator.')

    ### Data augmentation ###
    parser.add_argument(
        '--DiffAugment', type=str, default='',
        help='DiffAugment policy')
    parser.add_argument(
        '--DA', action='store_true', default=False,
        help='Diff Augment for GANs (default: %(default)s)')
    parser.add_argument(
        '--hflips', action='store_true', default=False,
        help='Use horizontal flips in data augmentation.'
             '(default: %(default)s)')

    ### Model stuff ###
    parser.add_argument(
        '--model', type=str, default='BigGAN',
        help='Name of the model module (default: %(default)s)')
    parser.add_argument(
        '--G_param', type=str, default='SN',
        help='Parameterization style to use for G, spectral norm (SN) or SVD (SVD)'
             ' or None (default: %(default)s)')
    parser.add_argument(
        '--D_param', type=str, default='SN',
        help='Parameterization style to use for D, spectral norm (SN) or SVD (SVD)'
             ' or None (default: %(default)s)')
    parser.add_argument(
        '--G_ch', type=int, default=64,
        help='Channel multiplier for G (default: %(default)s)')
    parser.add_argument(
        '--D_ch', type=int, default=64,
        help='Channel multiplier for D (default: %(default)s)')
    parser.add_argument(
        '--G_depth', type=int, default=1,
        help='Number of resblocks per stage in G? (default: %(default)s)')
    parser.add_argument(
        '--D_depth', type=int, default=1,
        help='Number of resblocks per stage in D? (default: %(default)s)')
    parser.add_argument(
        '--D_thin', action='store_false', dest='D_wide', default=True,
        help='Use the SN-GAN channel pattern for D? (default: %(default)s)')
    parser.add_argument(
        '--G_shared', action='store_true', default=True,
        help='Use shared embeddings in G? (default: %(default)s)')
    parser.add_argument(
        '--shared_dim', type=int, default=0,
        help='G''s shared embedding dimensionality; if 0, will be equal to dim_z. '
             '(default: %(default)s)')
    parser.add_argument(
        '--dim_z', type=int, default=120,
        help='Noise dimensionality: %(default)s)')
    parser.add_argument(
        '--z_var', type=float, default=1.0,
        help='Noise variance: %(default)s)')
    parser.add_argument(
        '--hier', action='store_true', default=False,
        help='Use hierarchical z in G? (default: %(default)s)')
    parser.add_argument(
        '--syncbn', action='store_true', default=False,
        help='Sync batch norm? (default: %(default)s)')
    parser.add_argument(
        '--cross_replica', action='store_true', default=False,
        help='Cross_replica batchnorm in G?(default: %(default)s)')
    parser.add_argument(
        '--mybn', action='store_true', default=False,
        help='Use my batchnorm (which supports standing stats?) %(default)s)')
    parser.add_argument(
        '--G_nl', type=str, default='relu',
        help='Activation function for G (default: %(default)s)')
    parser.add_argument(
        '--D_nl', type=str, default='relu',
        help='Activation function for D (default: %(default)s)')
    parser.add_argument(
        '--G_attn', type=str, default='64',
        help='What resolutions to use attention on for G (underscore separated) '
             '(default: %(default)s)')
    parser.add_argument(
        '--D_attn', type=str, default='64',
        help='What resolutions to use attention on for D (underscore separated) '
             '(default: %(default)s)')
    parser.add_argument(
        '--norm_style', type=str, default='bn',
        help='Normalizer style for G, one of bn [batchnorm], in [instancenorm], '
             'ln [layernorm], gn [groupnorm] (default: %(default)s)')

    ### Model init stuff ###
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed to use; affects both initialization and '
             ' dataloading. (default: %(default)s)')
    parser.add_argument(
        '--G_init', type=str, default='ortho',
        help='Init style to use for G (default: %(default)s)')
    parser.add_argument(
        '--D_init', type=str, default='ortho',
        help='Init style to use for D(default: %(default)s)')
    parser.add_argument(
        '--skip_init', action='store_true', default=False,
        help='Skip initialization, ideal for testing when ortho init was used '
             '(default: %(default)s)')

    ### Optimizer stuff ###
    parser.add_argument(
        '--G_lr', type=float, default=5e-5,
        help='Learning rate to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_lr', type=float, default=2e-4,
        help='Learning rate to use for Discriminator (default: %(default)s)')
    parser.add_argument(
        '--G_B1', type=float, default=0.0,
        help='Beta1 to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_B1', type=float, default=0.0,
        help='Beta1 to use for Discriminator (default: %(default)s)')
    parser.add_argument(
        '--G_B2', type=float, default=0.999,
        help='Beta2 to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_B2', type=float, default=0.999,
        help='Beta2 to use for Discriminator (default: %(default)s)')

    ### Batch size, parallel, and precision stuff ###
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
        '--G_batch_size', type=int, default=0,
        help='Batch size to use for G; if 0, same as D (default: %(default)s)')
    parser.add_argument(
        '--num_G_accumulations', type=int, default=1,
        help='Number of passes to accumulate G''s gradients over '
             '(default: %(default)s)')
    parser.add_argument(
        '--num_D_steps', type=int, default=2,
        help='Number of D steps per G step (default: %(default)s)')
    parser.add_argument(
        '--num_D_accumulations', type=int, default=1,
        help='Number of passes to accumulate D''s gradients over '
             '(default: %(default)s)')
    parser.add_argument(
        '--split_D', action='store_true', default=False,
        help='Run D twice rather than concatenating inputs? (default: %(default)s)')
    parser.add_argument(
        '--num_epochs', type=int, default=100,
        help='Number of epochs to train for (default: %(default)s)')
    parser.add_argument(
        '--parallel', action='store_true', default=False,
        help='Train with multiple GPUs (default: %(default)s)')
    parser.add_argument(
        '--G_fp16', action='store_true', default=False,
        help='Train with half-precision in G? (default: %(default)s)')
    parser.add_argument(
        '--D_fp16', action='store_true', default=False,
        help='Train with half-precision in D? (default: %(default)s)')
    parser.add_argument(
        '--D_mixed_precision', action='store_true', default=False,
        help='Train with half-precision activations but fp32 params in D? '
             '(default: %(default)s)')
    parser.add_argument(
        '--G_mixed_precision', action='store_true', default=False,
        help='Train with half-precision activations but fp32 params in G? '
             '(default: %(default)s)')
    parser.add_argument(
        '--accumulate_stats', action='store_true', default=False,
        help='Accumulate "standing" batchnorm stats? (default: %(default)s)')
    parser.add_argument(
        '--num_standing_accumulations', type=int, default=16,
        help='Number of forward passes to use in accumulating standing stats? '
             '(default: %(default)s)')

    ### Bookkeping stuff ###
    parser.add_argument(
        '--G_eval_mode', action='store_true', default=False,
        help='Run G in eval mode (running/standing stats?) at sample/test time? '
             '(default: %(default)s)')
    parser.add_argument(
        '--save_every', type=int, default=2000,
        help='Save every X iterations (default: %(default)s)')
    parser.add_argument(
        '--num_save_copies', type=int, default=2,
        help='How many copies to save (default: %(default)s)')
    parser.add_argument(
        '--num_best_copies', type=int, default=2,
        help='How many previous best checkpoints to save (default: %(default)s)')
    parser.add_argument(
        '--which_best', type=str, default='IS',
        help='Which metric to use to determine when to save new "best"'
             'checkpoints, one of IS or FID (default: %(default)s)')
    parser.add_argument(
        '--no_fid', action='store_true', default=False,
        help='Calculate IS only, not FID? (default: %(default)s)')
    parser.add_argument(
        '--test_every', type=int, default=5000,
        help='Test every X iterations (default: %(default)s)')
    parser.add_argument(
        '--num_inception_images', type=int, default=50000,
        help='Number of samples to compute inception metrics with '
             '(default: %(default)s)')
    parser.add_argument(
        '--hashname', action='store_true', default=False,
        help='Use a hash of the experiment name instead of the full config '
             '(default: %(default)s)')
    parser.add_argument(
        '--base_root', type=str, default='',
        help='Default location to store all weights, samples, data, and logs '
             ' (default: %(default)s)')
    parser.add_argument(
        '--data_root', type=str, default='data',
        help='Default location where data is stored (default: %(default)s)')
    parser.add_argument(
        '--weights_root', type=str, default='weights',
        help='Default location to store weights (default: %(default)s)')
    parser.add_argument(
        '--logs_root', type=str, default='logs',
        help='Default location to store logs (default: %(default)s)')
    parser.add_argument(
        '--samples_root', type=str, default='samples',
        help='Default location to store samples (default: %(default)s)')
    parser.add_argument(
        '--pbar', type=str, default='mine',
        help='Type of progressbar to use; one of "mine" or "tqdm" '
             '(default: %(default)s)')
    parser.add_argument(
        '--name_suffix', type=str, default='',
        help='Suffix for experiment name for loading weights for sampling '
             '(consider "best0") (default: %(default)s)')
    parser.add_argument(
        '--experiment_name', type=str, default='',
        help='Optionally override the automatic experiment naming with this arg. '
             '(default: %(default)s)')
    parser.add_argument(
        '--config_from_name', action='store_true', default=False,
        help='Use a hash of the experiment name instead of the full config '
             '(default: %(default)s)')

    ### EMA Stuff ###
    parser.add_argument(
        '--ema', action='store_true', default=False,
        help='Keep an ema of G''s weights? (default: %(default)s)')
    parser.add_argument(
        '--ema_decay', type=float, default=0.9999,
        help='EMA decay rate (default: %(default)s)')
    parser.add_argument(
        '--use_ema', action='store_true', default=False,
        help='Use the EMA parameters of G for evaluation? (default: %(default)s)')
    parser.add_argument(
        '--ema_start', type=int, default=20000,
        help='When to start updating the EMA weights (default: %(default)s)')

    ### Numerical precision and SV stuff ###
    parser.add_argument(
        '--adam_eps', type=float, default=1e-6,
        help='epsilon value to use for Adam (default: %(default)s)')
    parser.add_argument(
        '--BN_eps', type=float, default=1e-5,
        help='epsilon value to use for BatchNorm (default: %(default)s)')
    parser.add_argument(
        '--SN_eps', type=float, default=1e-6,
        help='epsilon value to use for Spectral Norm(default: %(default)s)')
    parser.add_argument(
        '--num_G_SVs', type=int, default=1,
        help='Number of SVs to track in G (default: %(default)s)')
    parser.add_argument(
        '--num_D_SVs', type=int, default=1,
        help='Number of SVs to track in D (default: %(default)s)')
    parser.add_argument(
        '--num_G_SV_itrs', type=int, default=1,
        help='Number of SV itrs in G (default: %(default)s)')
    parser.add_argument(
        '--num_D_SV_itrs', type=int, default=1,
        help='Number of SV itrs in D (default: %(default)s)')

    parser.add_argument(
        '--class_cond', action='store_true', default=False,
        help='Use classes as conditioning')
    parser.add_argument(
        '--constant_conditioning', action='store_true', default=False,
        help='Use a a class-conditioning vector where the input label is always 0?  (default: %(default)s)')


    parser.add_argument(
        '--which_dataset', type=str, default='imagenet',
        choices=['imagenet','coco'],
        help='Dataset choice.')

    ### Ortho reg stuff ###
    parser.add_argument(
        '--G_ortho', type=float, default=0.0,  # 1e-4 is default for BigGAN
        help='Modified ortho reg coefficient in G(default: %(default)s)')
    parser.add_argument(
        '--D_ortho', type=float, default=0.0,
        help='Modified ortho reg coefficient in D (default: %(default)s)')
    parser.add_argument(
        '--toggle_grads', action='store_true', default=True,
        help='Toggle D and G''s "requires_grad" settings when not training them? '
             ' (default: %(default)s)')

    ### Which train functions/setup ###
    parser.add_argument(
        '--which_train_fn', type=str, default='GAN',
        help='How2trainyourbois (default: %(default)s)')
    parser.add_argument(
        '--run_setup', type=str, default='slurm',
        help='If local_debug or slurm (default: %(default)s)')
    parser.add_argument(
        '--ddp_train', action='store_true', default=False,
        help='If use DDP for training')
    parser.add_argument(
        '--n_nodes', type=int, default=1,
        help='Number of nodes for ddp (default: %(default)s)')
    parser.add_argument(
        '--n_gpus_per_node', type=int, default=1,
        help='Number of gpus per node for ddp (default: %(default)s)')
    parser.add_argument(
        '--stop_when_diverge', action='store_true', default=False,
        help='Stop the experiment if there is signs of divergence. '
             '(default: %(default)s)')
    parser.add_argument(
        '--es_patience', type=int, default=50,
        help='Epochs for early stopping patience')
    parser.add_argument(
        '--deterministic_run', action='store_true', default=False,
        help='Set deterministic cudnn and set the seed at each epoch'
             '(default: %(default)s)')


    ### Testing parameters ###
    parser.add_argument(
        '--eval_prdc', action='store_true', default=False,
        help='(Eval) Evaluate prdc '
             ' (default: %(default)s)')

    ### Resume training stuff
    parser.add_argument(
        '--load_weights', type=str, default='',
        help='Suffix for which weights to load (e.g. best0, copy0) '
             '(default: %(default)s)')
    parser.add_argument(
        '--resume', action='store_true', default=False,
        help='Resume training? (default: %(default)s)')

    ### Log stuff ###
    parser.add_argument(
        '--logstyle', type=str, default='%3.3e',
        help='What style to use when logging training metrics?'
             'One of: %#.#f/ %#.#e (float/exp, text),'
             'pickle (python pickle),'
             'npz (numpy zip),'
             'mat (MATLAB .mat file) (default: %(default)s)')
    parser.add_argument(
        '--log_G_spectra', action='store_true', default=False,
        help='Log the top 3 singular values in each SN layer in G? '
             '(default: %(default)s)')
    parser.add_argument(
        '--log_D_spectra', action='store_true', default=False,
        help='Log the top 3 singular values in each SN layer in D? '
             '(default: %(default)s)')
    parser.add_argument(
        '--sv_log_interval', type=int, default=10,
        help='Iteration interval for logging singular values '
             ' (default: %(default)s)')

    return parser


# Arguments for sample.py; not presently used in train.py
def add_sample_parser(parser):
    parser.add_argument(
        '--sample_npz', action='store_true', default=False,
        help='Sample "sample_num_npz" images and save to npz? '
             '(default: %(default)s)')
    parser.add_argument(
        '--sample_num_npz', type=int, default=50000,
        help='Number of images to sample when sampling NPZs '
             '(default: %(default)s)')
    parser.add_argument(
        '--sample_sheets', action='store_true', default=False,
        help='Produce class-conditional sample sheets and stick them in '
             'the samples root? (default: %(default)s)')
    parser.add_argument(
        '--sample_interps', action='store_true', default=False,
        help='Produce interpolation sheets and stick them in '
             'the samples root? (default: %(default)s)')
    parser.add_argument(
        '--sample_sheet_folder_num', type=int, default=-1,
        help='Number to use for the folder for these sample sheets '
             '(default: %(default)s)')
    parser.add_argument(
        '--sample_random', action='store_true', default=False,
        help='Produce a single random sheet? (default: %(default)s)')
    parser.add_argument(
        '--sample_trunc_curves', type=str, default='',
        help='Get inception metrics with a range of variances?'
             'To use this, specify a startpoint, step, and endpoint, e.g. '
             '--sample_trunc_curves 0.2_0.1_1.0 for a startpoint of 0.2, '
             'endpoint of 1.0, and stepsize of 1.0.  Note that this is '
             'not exactly identical to using tf.truncated_normal, but should '
             'have approximately the same effect. (default: %(default)s)')
    parser.add_argument(
        '--sample_inception_metrics', action='store_true', default=False,
        help='Calculate Inception metrics with sample.py? (default: %(default)s)')
    return parser

activation_dict = {'inplace_relu': nn.ReLU(inplace=True),
                   'relu': nn.ReLU(inplace=False),
                   'ir': nn.ReLU(inplace=True), }


class CenterCropLongEdge(object):
    """Crops the given PIL Image on the long edge.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


class RandomCropLongEdge(object):
    """Crops the given PIL Image on the long edge with a random start point.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        size = (min(img.size), min(img.size))
        # Only step forward along this edge if it's the long edge
        i = (0 if size[0] == img.size[0]
             else np.random.randint(low=0, high=img.size[0] - size[0]))
        j = (0 if size[1] == img.size[1]
             else np.random.randint(low=0, high=img.size[1] - size[1]))
        return transforms.functional.crop(img, i, j, size[0], size[1])

    def __repr__(self):
        return self.__class__.__name__

class CheckpointedSampler(torch.utils.data.Sampler):
    r"""Resumable sample with a random generated initialized with a given seed.

    Arguments:
        data_source (Dataset): dataset to sample from
        num_epochs (int) : Number of times to loop over the dataset
        start_itr (int) : which iteration to begin from
    """

    def __init__(self, data_source, start_itr=0, start_epoch=0, batch_size=128,
                 class_balanced=False, custom_distrib_gen=False,
                 samples_per_class=None, class_probabilities=None,
                 longtail_temperature=1, seed=0):
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.start_itr = start_itr % (len(self.data_source)//batch_size)
        self.start_epoch = start_epoch
        self.batch_size = batch_size
        self.class_balanced = class_balanced
        self.custom_distrib_gen = custom_distrib_gen
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        if self.class_balanced:
            print('Class balanced sampling.')
            self.weights = make_weights_for_balanced_classes(self.data_source.labels,
                                                        1000,
                                                        self.custom_distrib_gen,
                                                        longtail_temperature,
                                                        samples_per_class=samples_per_class,
                                                        class_probabilities=class_probabilities
                                                             )
            self.weights = torch.DoubleTensor(self.weights)

        # Resumable data loader
        print('Using the generator ', self.start_epoch,
              ' times to resume where we left off.')
       # print('Later, we will resume at iteration ', self.start_itr)
        for epoch in range(self.start_epoch):
            self.sample_epoch_perm()

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(
                self.num_samples))

    def sample_epoch_perm(self):
        if self.class_balanced:
            out = [torch.multinomial(self.weights, len(self.data_source),
                                     replacement=True, generator=self.generator)]
        else:
            out = [torch.randperm(len(self.data_source), generator=self.generator)]
        return out

    def __iter__(self):
        out = self.sample_epoch_perm()
        output = torch.cat(out).tolist()
        return iter(output)

    def __len__(self):
        return len(self.data_source)

def make_weights_for_balanced_classes(labels, nclasses,
                                      custom_distrib_gen=False,
                                      longtail_temperature=1, samples_per_class=None,
                                      class_probabilities=None
                                      ):
    if custom_distrib_gen:
        # temperature controlled distribution
        print('Temperature controlled distribution for balanced classes! '
              'Temperature:', longtail_temperature)
        class_prob = torch.log(torch.DoubleTensor(class_probabilities))
        weight_per_class = torch.exp(class_prob / longtail_temperature) / \
                          torch.sum(torch.exp(
                              class_prob / longtail_temperature))
    else:
        count = [0] * nclasses
        for item in labels:
            count[item] += 1
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            #Standard class balancing
            weight_per_class[i] = N / float(count[i])
    # Convert weighting per class to weighting per example
    weight = [0] * len(labels)
    for idx, val in enumerate(labels):
        # Uniform probability of selecting a sample, given a class
        # p(x|y)p(y)
        weight[idx] = (1/samples_per_class[val])*weight_per_class[val]
    print('Sum weight', sum(weight))
    return weight

def get_dataset_hdf5(resolution, data_path, load_in_mem=False, augment=False,longtail=False,
                local_rank=0, copy_locally=False, ddp=True, tmp_dir='', **kwargs):

    dataset_name_prefix = 'ILSVRC' # if which_dataset == 'imagenet' else 'COCO'
    # HDF5 file name
    hdf5_filename = '%s%i%s' % (
        dataset_name_prefix, resolution,
        '' if not longtail else 'longtail')

    # Data paths
    data_path_xy = os.path.join(data_path, hdf5_filename + '_xy.hdf5')
    #'/scratch/slurm_tmpdir/'+str(os.environ.get("SLURM_JOB_ID"))

    # Optionally copy the data locally in the cluster.
    if copy_locally:
        tmp_file = os.path.join(tmp_dir, hdf5_filename + '_xy.hdf5')
        print(tmp_file)
        # Only copy locally for the first device in each machine
        if local_rank == 0: #device == 'cuda:0':
            shutil.copy2(data_path_xy, tmp_file)
        data_path_xy = tmp_file

        # Wait for the main process to copy the data locally
        if ddp:
            dist.barrier()

    # Data transforms
    if augment:
        print('>>>>>>Augmenting images from hdf5 file with flips!')
        transform_list = transforms.RandomHorizontalFlip()
    else:
        transform_list = None
    dataset = dset.ILSVRC_HDF5(root=data_path_xy, transform=transform_list,
                            load_in_mem=load_in_mem)
    return dataset

def get_dataset_images(resolution, data_path, load_in_mem=False, augment=False,longtail=False, **kwargs):
    # Data transforms
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    transform_list = [CenterCropLongEdge(), transforms.Resize(resolution)]
    transform_list = transforms.Compose(transform_list + [
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)])
    if augment:
        print('>>>>>>Augmenting images from directory with flips!')
        transform_list = transforms.Compose(transform_list + [
            transforms.RandomHorizontalFlip()])

    dataset = dset.ImageFolder(root=data_path, transform=transform_list,
                            load_in_mem=load_in_mem, longtail=longtail)
    return dataset

#Modified to be able to do class-balancing
class DistributedSampler(torch.utils.data.sampler.Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True,
                 weights=None):
        if num_replicas is None:
            if not torch.dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.dist.get_world_size()
        if rank is None:
            if not torch.dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.weights = weights

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            if self.weights is not None:
                print('using class balanced!')
                indices = torch.multinomial(self.weights, len(self.dataset),
                                  replacement=True, generator=g).tolist()
            else:
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def get_dataloader(dataset, batch_size=64,
                   num_workers=8, shuffle=True,
                   pin_memory=True, drop_last=True, start_itr=0,
                   start_epoch=0, use_checkpointable_sampler=False,
                   use_balanced_sampler=False, custom_distrib_gen=False,
                   samples_per_class=None, class_probabilities=None,
                   seed=0, longtail_temperature=1, rank=0, world_size=-1,
                   **kwargs):

    # Prepare loader; the loaders list is for forward compatibility with
    # using validation / test splits.
   # if use_multiepoch_sampler:
    loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory,
                     'drop_last':drop_last}
    print('Dropping last batch? ', drop_last)
    # Otherwise, it has issues dividing the batch for accumulations
    # if longtail:
    #   loader_kwargs.update({'drop_last': drop_last})
    if use_checkpointable_sampler:
        print('Using checkpointable sampler from start_itr %d..., using seed %d' %
              (start_itr,seed))

        sampler = CheckpointedSampler(dataset, start_itr, start_epoch,
                                    batch_size,
                                    class_balanced=use_balanced_sampler,
                                    custom_distrib_gen=custom_distrib_gen,
                                    longtail_temperature=longtail_temperature,
                                    samples_per_class=samples_per_class,
                                    class_probabilities=class_probabilities,
                                    seed=seed)
        loader = DataLoader(dataset, batch_size=batch_size,
                                  sampler=sampler, shuffle=False,
                            worker_init_fn=seed_worker,
                            **loader_kwargs)
    else:
        if use_balanced_sampler:
            print('Balancing real data! Custom? ', custom_distrib_gen)
            weights = make_weights_for_balanced_classes(dataset.labels,
                                                        1000,
                                                        custom_distrib_gen,
                                                        longtail_temperature,
                                                        samples_per_class=samples_per_class,
                                                        class_probabilities=class_probabilities)
            weights = torch.DoubleTensor(weights)
        else:
            weights = None
        if world_size == -1:
            if use_balanced_sampler:
                sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,
                                                                         len(
                                                                             weights))
                shuffle = False
            else:
                sampler = None
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank, weights=weights)
            shuffle=False
        print('Loader workers?', loader_kwargs, ' with shuffle?', shuffle)
        loader = DataLoader(dataset, batch_size=batch_size,
                                      shuffle=shuffle, sampler=sampler,
                            worker_init_fn=seed_worker if use_checkpointable_sampler else None,
                                      **loader_kwargs)

    return loader

# Utility file to seed rngs
def seed_rng(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() + worker_id

# Utility to peg all roots to a base root
# If a base root folder is provided, peg all other root folders to it.
def update_config_roots(config):
    if config['base_root']:
        print('Pegging all root folders to base root %s' % config['base_root'])
        for key in ['weights', 'logs', 'samples']:
            config['%s_root' % key] = '%s/%s' % (config['base_root'], key)
    return config


# Utility to prepare root folders if they don't exist; parent folder must exist
def prepare_root(config):
    for key in ['weights_root', 'logs_root', 'samples_root']:
        if not os.path.exists(config[key]):
            print('Making directory %s for %s...' % (config[key], key))
            os.mkdir(config[key])


# Simple wrapper that applies EMA to a model. COuld be better done in 1.0 using
# the parameters() and buffers() module functions, but for now this works
# with state_dicts using .copy_
class ema(object):
    def __init__(self, source, target, decay=0.9999, start_itr=0):
        self.source = source
        self.target = target
        self.decay = decay
        # Optional parameter indicating what iteration to start the decay at
        self.start_itr = start_itr
        # Initialize target's params to be source's
        self.source_dict = self.source.state_dict()
        self.target_dict = self.target.state_dict()
        print('Initializing EMA parameters to be source parameters...')
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.source_dict[key].data)
                # target_dict[key].data = source_dict[key].data # Doesn't work!

    def update(self, itr=None):
        # If an iteration counter is provided and itr is less than the start itr,
        # peg the ema weights to the underlying weights.
        if itr and itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.target_dict[key].data * decay
                                                 + self.source_dict[key].data * (1 - decay))


# Apply modified ortho reg to a model
# This function is an optimized version that directly computes the gradient,
# instead of computing and then differentiating the loss.
def ortho(model, strength=1e-4, blacklist=[]):
    with torch.no_grad():
        for param in model.parameters():
            # Only apply this to parameters with at least 2 axes, and not in the blacklist
            if len(param.shape) < 2 or any([param is item for item in blacklist]):
                continue
            w = param.view(param.shape[0], -1)
            grad = (2 * torch.mm(torch.mm(w, w.t())
                                 * (1. - torch.eye(w.shape[0], device=w.device)), w))
            param.grad.data += strength * grad.view(param.shape)


# Default ortho reg
# This function is an optimized version that directly computes the gradient,
# instead of computing and then differentiating the loss.
def default_ortho(model, strength=1e-4, blacklist=[]):
    with torch.no_grad():
        for param in model.parameters():
            # Only apply this to parameters with at least 2 axes & not in blacklist
            if len(param.shape) < 2 or param in blacklist:
                continue
            w = param.view(param.shape[0], -1)
            grad = (2 * torch.mm(torch.mm(w, w.t())
                                 - torch.eye(w.shape[0], device=w.device), w))
            param.grad.data += strength * grad.view(param.shape)


# Convenience utility to switch off requires_grad
def toggle_grad(model, on_or_off):
    for param in model.parameters():
        param.requires_grad = on_or_off


# Function to join strings or ignore them
# Base string is the string to link "strings," while strings
# is a list of strings or Nones.
def join_strings(base_string, strings):
    return base_string.join([item for item in strings if item])


# Save a model's weights, optimizer, and the state_dict
def save_weights(G, D, state_dict, weights_root, experiment_name,
                 name_suffix=None, G_ema=None, embedded_optimizers=True,
                 G_optim=None, D_optim=None):
    root = '/'.join([weights_root, experiment_name])
    if not os.path.exists(root):
        os.mkdir(root)
    if name_suffix:
        print('Saving weights to %s/%s...' % (root, name_suffix))
    else:
        print('Saving weights to %s...' % root)
    torch.save(G.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['G', name_suffix])))
    torch.save(D.state_dict(),
               '%s/%s.pth' % (root, join_strings('_', ['D', name_suffix])))
    torch.save(state_dict,
               '%s/%s.pth' % (root, join_strings('_', ['state_dict', name_suffix])))

    if embedded_optimizers:
      torch.save(G.optim.state_dict(),
                  '%s/%s.pth' % (
                  root, join_strings('_', ['G_optim', name_suffix])))
      torch.save(D.optim.state_dict(),
                  '%s/%s.pth' % (
                  root, join_strings('_', ['D_optim', name_suffix])))
    else:
      torch.save(G_optim.state_dict(),
                  '%s/%s.pth' % (
                  root, join_strings('_', ['G_optim', name_suffix])))
      torch.save(D_optim.state_dict(),
                  '%s/%s.pth' % (
                  root, join_strings('_', ['D_optim', name_suffix])))
    if G_ema is not None:
        torch.save(G_ema.state_dict(),
                   '%s/%s.pth' % (root, join_strings('_', ['G_ema', name_suffix])))


# Load a model's weights, optimizer, and the state_dict
def load_weights(G, D, state_dict, weights_root, experiment_name,
                 name_suffix=None, G_ema=None, strict=True, load_optim=True,
                 eval=False, map_location=None, embedded_optimizers=True,
                 G_optim=None, D_optim=None):
    root = '/'.join([weights_root, experiment_name])
    if not os.path.exists(root):
      print('Not loading data, experiment folder does not exist yet!')
      print(root)
      if eval:
        raise ValueError('Make sure foder exists')
      return

    if name_suffix:
        print('Loading %s weights from %s...' % (name_suffix, root))
    else:
        print('Loading weights from %s...' % root)
    if G is not None:
        G.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['G', name_suffix])),
                       map_location=map_location),
            strict=strict)
        if load_optim:
          if embedded_optimizers:
            G.optim.load_state_dict(torch.load('%s/%s.pth' % (
                    root, join_strings('_', ['G_optim', name_suffix])), map_location=map_location))
          else:
            G_optim.load_state_dict(torch.load('%s/%s.pth' % (
                    root, join_strings('_', ['G_optim', name_suffix])), map_location=map_location))
    if D is not None:
        D.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['D', name_suffix])),
                       map_location=map_location), strict=strict)
        if load_optim:
          if embedded_optimizers:
            D.optim.load_state_dict(
                torch.load('%s/%s.pth' % (
                    root, join_strings('_', ['D_optim', name_suffix])),
                             map_location=map_location))
          else:
            D_optim.load_state_dict(
                torch.load('%s/%s.pth' % (
                    root, join_strings('_', ['D_optim', name_suffix])),
                             map_location=map_location))
    # Load state dict
    for item in state_dict:
      try:
        state_dict[item] = torch.load('%s/%s.pth' % (
            root, join_strings('_', ['state_dict', name_suffix])),
                                          map_location=map_location)[item]
      except:
        print('No values to load')
    if G_ema is not None:
        G_ema.load_state_dict(
            torch.load('%s/%s.pth' % (root, join_strings('_', ['G_ema', name_suffix])),
                       map_location=map_location), strict=strict)


''' MetricsLogger originally stolen from VoxNet source code.
    Used for logging inception metrics'''


class MetricsLogger(object):
    def __init__(self, fname, reinitialize=False):
        self.fname = fname
        self.reinitialize = reinitialize
        if os.path.exists(self.fname):
            if self.reinitialize:
                print('{} exists, deleting...'.format(self.fname))
                os.remove(self.fname)

    def log(self, record=None, **kwargs):
        """
        Assumption: no newlines in the input.
        """
        if record is None:
            record = {}
        record.update(kwargs)
        record['_stamp'] = time.time()
        with open(self.fname, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=True) + '\n')


# Logstyle is either:
# '%#.#f' for floating point representation in text
# '%#.#e' for exponent representation in text
# 'npz' for output to npz # NOT YET SUPPORTED
# 'pickle' for output to a python pickle # NOT YET SUPPORTED
# 'mat' for output to a MATLAB .mat file # NOT YET SUPPORTED
class MyLogger(object):
    def __init__(self, fname, reinitialize=False, logstyle='%3.3f'):
        self.root = fname
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        self.reinitialize = reinitialize
        self.metrics = []
        self.logstyle = logstyle  # One of '%3.3f' or like '%3.3e'

    # Delete log if re-starting and log already exists
    def reinit(self, item):
        if os.path.exists('%s/%s.log' % (self.root, item)):
            if self.reinitialize:
                # Only print the removal mess
                if 'sv' in item:
                    if not any('sv' in item for item in self.metrics):
                        print('Deleting singular value logs...')
                else:
                    print('{} exists, deleting...'.format('%s_%s.log' % (self.root, item)))
                os.remove('%s/%s.log' % (self.root, item))

    # Log in plaintext; this is designed for being read in MATLAB(sorry not sorry)
    def log(self, itr, **kwargs):
        for arg in kwargs:
            if arg not in self.metrics:
                if self.reinitialize:
                    self.reinit(arg)
                self.metrics += [arg]
            if self.logstyle == 'pickle':
                print('Pickle not currently supported...')
                # with open('%s/%s.log' % (self.root, arg), 'a') as f:
                # pickle.dump(kwargs[arg], f)
            elif self.logstyle == 'mat':
                print('.mat logstyle not currently supported...')
            else:
                with open('%s/%s.log' % (self.root, arg), 'a') as f:
                    f.write('%d: %s\n' % (itr, self.logstyle % kwargs[arg]))


# Write some metadata to the logs directory
def write_metadata(logs_root, experiment_name, config, state_dict):
    with open(('%s/%s/metalog.txt' %
               (logs_root, experiment_name)), 'w') as writefile:
        writefile.write('datetime: %s\n' % str(datetime.datetime.now()))
        writefile.write('config: %s\n' % str(config))
        writefile.write('state: %s\n' % str(state_dict))


"""
Very basic progress indicator to wrap an iterable in.

Author: Jan Schlüter
Andy's adds: time elapsed in addition to ETA, makes it possible to add
estimated time to 1k iters instead of estimated time to completion.
"""


def progress(items, desc='', total=None, min_delay=0.1, displaytype='s1k'):
    """
    Returns a generator over `items`, printing the number and percentage of
    items processed and the estimated remaining processing time before yielding
    the next item. `total` gives the total number of items (required if `items`
    has no length), and `min_delay` gives the minimum time in seconds between
    subsequent prints. `desc` gives an optional prefix text (end with a space).
    """
    total = total or len(items)
    t_start = time.time()
    t_last = 0
    for n, item in enumerate(items):
        t_now = time.time()
        if t_now - t_last > min_delay:
            print("\r%s%d/%d (%6.2f%%)" % (
                desc, n + 1, total, n / float(total) * 100), end=" ")
            if n > 0:

                if displaytype == 's1k':  # minutes/seconds for 1000 iters
                    next_1000 = n + (1000 - n % 1000)
                    t_done = t_now - t_start
                    t_1k = t_done / n * next_1000
                    outlist = list(divmod(t_done, 60)) + list(divmod(t_1k - t_done, 60))
                    print("(TE/ET1k: %d:%02d / %d:%02d)" % tuple(outlist), end=" ")
                else:  # displaytype == 'eta':
                    t_done = t_now - t_start
                    t_total = t_done / n * total
                    outlist = list(divmod(t_done, 60)) + list(divmod(t_total - t_done, 60))
                    print("(TE/ETA: %d:%02d / %d:%02d)" % tuple(outlist), end=" ")

            sys.stdout.flush()
            t_last = t_now
        yield item
    t_total = time.time() - t_start
    print("\r%s%d/%d (100.00%%) (took %d:%02d)" % ((desc, total, total) +
                                                   divmod(t_total, 60)))

def sample_conditioning_values(z_, y_, model_type='BigGAN',
                               ddp=False, constant_conditioning=False):
    with torch.no_grad():
      z_.sample_()
      if model_type == 'BigGAN':
        y_.sample_()
        if constant_conditioning:
          return z_, torch.zeros_like(y_)
        else:
          if ddp:
            return z_, y_
          else:
            return z_, y_.data.clone()

# Sample function for use with inception metrics
def sample(G, sample_conditioning_func, config, model_type='BigGAN', device='cuda'):
    conditioning = sample_conditioning_func()
    with torch.no_grad():
      if model_type == 'BigGAN':
        z_, y_ = conditioning
        y_ = y_.long()
        z_ = z_.to(device, non_blocking=True)
        y_ = y_.to(device,non_blocking=True)

        if config['parallel']:
            G_z = nn.parallel.data_parallel(G,(z_,y_))
        else:
            G_z = G(z_, y_)
        return G_z, y_


# Sample function for sample sheets
def sample_sheet(G, classes_per_sheet, num_classes, samples_per_class, parallel,
                 samples_root, experiment_name, folder_number, z_=None):
    # Prepare sample directory
    if not os.path.isdir('%s/%s' % (samples_root, experiment_name)):
        os.mkdir('%s/%s' % (samples_root, experiment_name))
    if not os.path.isdir('%s/%s/%d' % (samples_root, experiment_name, folder_number)):
        os.mkdir('%s/%s/%d' % (samples_root, experiment_name, folder_number))
    # loop over total number of sheets
    for i in range(num_classes // classes_per_sheet):
        ims = []
        y = torch.arange(i * classes_per_sheet, (i + 1) * classes_per_sheet, device='cuda')
        for j in range(samples_per_class):
            if (z_ is not None) and hasattr(z_, 'sample_') and classes_per_sheet <= z_.size(0):
                z_.sample_()
            else:
                z_ = torch.randn(classes_per_sheet, G.dim_z, device='cuda')
            with torch.no_grad():
                if parallel:
                    o = nn.parallel.data_parallel(G, (z_[:classes_per_sheet], G.shared(y)))
                else:
                    o = G(z_[:classes_per_sheet], G.shared(y))

            ims += [o.data.cpu()]
        # This line should properly unroll the images
        out_ims = torch.stack(ims, 1).view(-1, ims[0].shape[1], ims[0].shape[2],
                                           ims[0].shape[3]).data.float().cpu()
        # The path for the samples
        image_filename = '%s/%s/%d/samples%d.jpg' % (samples_root, experiment_name,
                                                     folder_number, i)
        torchvision.utils.save_image(out_ims, image_filename,
                                     nrow=samples_per_class, normalize=True)


# Interp function; expects x0 and x1 to be of shape (shape0, 1, rest_of_shape..)
def interp(x0, x1, num_midpoints):
    lerp = torch.linspace(0, 1.0, num_midpoints + 2, device='cuda').to(x0.dtype)
    return ((x0 * (1 - lerp.view(1, -1, 1))) + (x1 * lerp.view(1, -1, 1)))


# interp sheet function
# Supports full, class-wise and intra-class interpolation
def interp_sheet(G, num_per_sheet, num_midpoints, num_classes, parallel,
                 samples_root, experiment_name, folder_number, sheet_number=0,
                 fix_z=False, fix_y=False, device='cuda'):
    # Prepare zs and ys
    if fix_z:  # If fix Z, only sample 1 z per row
        zs = torch.randn(num_per_sheet, 1, G.dim_z, device=device)
        zs = zs.repeat(1, num_midpoints + 2, 1).view(-1, G.dim_z)
    else:
        zs = interp(torch.randn(num_per_sheet, 1, G.dim_z, device=device),
                    torch.randn(num_per_sheet, 1, G.dim_z, device=device),
                    num_midpoints).view(-1, G.dim_z)
    if fix_y:  # If fix y, only sample 1 z per row
        ys = sample_1hot(num_per_sheet, num_classes)
        ys = G.shared(ys).view(num_per_sheet, 1, -1)
        ys = ys.repeat(1, num_midpoints + 2, 1).view(num_per_sheet * (num_midpoints + 2), -1)
    else:
        ys = interp(G.shared(sample_1hot(num_per_sheet, num_classes)).view(num_per_sheet, 1, -1),
                    G.shared(sample_1hot(num_per_sheet, num_classes)).view(num_per_sheet, 1, -1),
                    num_midpoints).view(num_per_sheet * (num_midpoints + 2), -1)
    # Run the net--note that we've already passed y through G.shared.
    if G.fp16:
        zs = zs.half()
    with torch.no_grad():
        if parallel:
            out_ims = nn.parallel.data_parallel(G, (zs, ys)).data.cpu()
        else:
            out_ims = G(zs, ys).data.cpu()
    interp_style = '' + ('Z' if not fix_z else '') + ('Y' if not fix_y else '')
    image_filename = '%s/%s/%d/interp%s%d.jpg' % (samples_root, experiment_name,
                                                  folder_number, interp_style,
                                                  sheet_number)
    torchvision.utils.save_image(out_ims, image_filename,
                                 nrow=num_midpoints + 2, normalize=True)


# Convenience debugging function to print out gradnorms and shape from each layer
# May need to rewrite this so we can actually see which parameter is which
def print_grad_norms(net):
    gradsums = [[float(torch.norm(param.grad).item()),
                 float(torch.norm(param).item()), param.shape]
                for param in net.parameters()]
    order = np.argsort([item[0] for item in gradsums])
    print(['%3.3e,%3.3e, %s' % (gradsums[item_index][0],
                                gradsums[item_index][1],
                                str(gradsums[item_index][2]))
           for item_index in order])


# Get singular values to log. This will use the state dict to find them
# and substitute underscores for dots.
def get_SVs(net, prefix):
    d = net.state_dict()
    return {('%s_%s' % (prefix, key)).replace('.', '_'):
                float(d[key].item())
            for key in d if 'sv' in key}


# Name an experiment based on its config
def name_from_config(config):
    name = '_'.join([
        item for item in [
            'Big%s' % config['which_train_fn'],
            config['dataset'],
            config['model'] if config['model'] != 'BigGAN' else None,
            'seed%d' % config['seed'],
            'Gch%d' % config['G_ch'],
            'Dch%d' % config['D_ch'],
            'Gd%d' % config['G_depth'] if config['G_depth'] > 1 else None,
            'Dd%d' % config['D_depth'] if config['D_depth'] > 1 else None,
            'bs%d' % config['batch_size'],
            'Gfp16' if config['G_fp16'] else None,
            'Dfp16' if config['D_fp16'] else None,
            'nDs%d' % config['num_D_steps'] if config['num_D_steps'] > 1 else None,
            'nDa%d' % config['num_D_accumulations'] if config['num_D_accumulations'] > 1 else None,
            'nGa%d' % config['num_G_accumulations'] if config['num_G_accumulations'] > 1 else None,
            'Glr%2.1e' % config['G_lr'],
            'Dlr%2.1e' % config['D_lr'],
            'GB%3.3f' % config['G_B1'] if config['G_B1'] != 0.0 else None,
            'GBB%3.3f' % config['G_B2'] if config['G_B2'] != 0.999 else None,
            'DB%3.3f' % config['D_B1'] if config['D_B1'] != 0.0 else None,
            'DBB%3.3f' % config['D_B2'] if config['D_B2'] != 0.999 else None,
            'Gnl%s' % config['G_nl'],
            'Dnl%s' % config['D_nl'],
            'Ginit%s' % config['G_init'],
            'Dinit%s' % config['D_init'],
            'G%s' % config['G_param'] if config['G_param'] != 'SN' else None,
            'D%s' % config['D_param'] if config['D_param'] != 'SN' else None,
            'Gattn%s' % config['G_attn'] if config['G_attn'] != '0' else None,
            'Dattn%s' % config['D_attn'] if config['D_attn'] != '0' else None,
            'Gortho%2.1e' % config['G_ortho'] if config['G_ortho'] > 0.0 else None,
            'Dortho%2.1e' % config['D_ortho'] if config['D_ortho'] > 0.0 else None,
            config['norm_style'] if config['norm_style'] != 'bn' else None,
            'cr' if config['cross_replica'] else None,
            'Gshared' if config['G_shared'] else None,
            'hier' if config['hier'] else None,
            'ema' if config['ema'] else None,
            config['name_suffix'] if config['name_suffix'] else None,
        ]
        if item is not None])
    # dogball
    if config['hashname']:
        return hashname(name)
    else:
        return name


# A simple function to produce a unique experiment name from the animal hashes.
def hashname(name):
    h = hash(name)
    a = h % len(animal_hash.a)
    h = h // len(animal_hash.a)
    b = h % len(animal_hash.b)
    h = h // len(animal_hash.c)
    c = h % len(animal_hash.c)
    return animal_hash.a[a] + animal_hash.b[b] + animal_hash.c[c]


# Get GPU memory, -i is the index
def query_gpu(indices):
    os.system('nvidia-smi -i 0 --query-gpu=memory.free --format=csv')


# Convenience function to count the number of parameters in a module
def count_parameters(module):
    print('Number of parameters: {}'.format(
        sum([p.data.nelement() for p in module.parameters()])))


# Convenience function to sample an index, not actually a 1-hot
def sample_1hot(batch_size, num_classes, device='cuda'):
    return torch.randint(low=0, high=num_classes, size=(batch_size,),
                         device=device, dtype=torch.int64, requires_grad=False)


# A highly simplified convenience class for sampling from distributions
# One could also use PyTorch's inbuilt distributions package.
# Note that this class requires initialization to proceed as
# x = Distribution(torch.randn(size))
# x.init_distribution(dist_type, **dist_kwargs)
# x = x.to(device,dtype)
# This is partially based on https://discuss.pytorch.org/t/subclassing-torch-tensor/23754/2
class Distribution(torch.Tensor):
    # Init the params of the distribution
    def init_distribution(self, dist_type, class_prob=None, **kwargs):
        self.dist_type = dist_type
        self.dist_kwargs = kwargs
        if self.dist_type == 'normal':
            self.mean, self.var = kwargs['mean'], kwargs['var']
        elif self.dist_type == 'categorical':
            self.num_categories = kwargs['num_categories']
        elif self.dist_type == 'categorical_longtail':
            print('(class conditioning sampler) using longtail distribution')
            self.num_categories = kwargs['num_categories']
            self.class_prob = torch.DoubleTensor(class_prob)
        elif self.dist_type == 'categorical_longtail_temperature':
            print('(class conditioning sampler) Softening the long-tail distribution with temperature ',
                  kwargs['temperature'])
            self.num_categories = kwargs['num_categories']
            self.class_prob = torch.log(torch.DoubleTensor(class_prob))
            self.class_prob = torch.exp(self.class_prob/kwargs['temperature']) / \
                              torch.sum(torch.exp(self.class_prob/kwargs['temperature']))
    def seed_generator(self, seed):
        self.generator.manual_seed(seed)
    def sample_(self):
        if self.dist_type == 'normal':
            self.normal_(self.mean, self.var)
        elif self.dist_type == 'categorical':
            self.random_(0, self.num_categories)
        elif 'categorical_longtail' in self.dist_type or \
                 'categorical_longtail_temperature' in self.dist_type:
            self.data = torch.multinomial(self.class_prob, len(self),
                                          replacement=True).to(self.device)
            # return self.variable

    # Silly hack: overwrite the to() method to wrap the new object
    # in a distribution as well
    # def to(self, *args, **kwargs):
    #     new_obj = Distribution(self)
    #     new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
    #     new_obj.data = super().to(*args, **kwargs)
    #     return new_obj


# Convenience function to prepare a z and y vector
def prepare_z_y(G_batch_size, dim_z, nclasses, device='cuda',
                fp16=False, z_var=1.0, longtail_gen=False, custom_distrib=False,
                longtail_temperature=1, class_probabilities=None):
    z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
    z_.init_distribution('normal', mean=0, var=z_var)
  #  z_ = z_.to(device, torch.float16 if fp16 else torch.float32)

    if fp16:
        z_ = z_.half()

    y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
    if longtail_gen:
        y_.init_distribution('categorical_longtail', num_categories=nclasses, class_prob=class_probabilities)
    elif custom_distrib:
        y_.init_distribution('categorical_longtail_temperature',
                             num_categories=nclasses,
                             temperature=longtail_temperature, class_prob=class_probabilities)
    else:
        y_.init_distribution('categorical', num_categories=nclasses)
   # y_ = y_.to(device, torch.int64)
    return z_, y_


def initiate_standing_stats(net):
    for module in net.modules():
        if hasattr(module, 'accumulate_standing'):
            module.reset_stats()
            module.accumulate_standing = True


def accumulate_standing_stats(net, z, y, nclasses, num_accumulations=16):
    initiate_standing_stats(net)
    net.train()
    for i in range(num_accumulations):
        with torch.no_grad():
            z.normal_()
            y.random_(0, nclasses)
            x = net(z, net.shared(y))  # No need to parallelize here unless using syncbn
    # Set to eval mode
    net.eval()


# This version of Adam keeps an fp32 copy of the parameters and
# does all of the parameter updates in fp32, while still doing the
# forwards and backwards passes using fp16 (i.e. fp16 copies of the
# parameters and fp16 activations).
#
# Note that this calls .float().cuda() on the params.
import math
from torch.optim.optimizer import Optimizer


class Adam16(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        params = list(params)
        super(Adam16, self).__init__(params, defaults)

    # Safety modification to make sure we floatify our state
    def load_state_dict(self, state_dict):
        super(Adam16, self).load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['exp_avg'] = self.state[p]['exp_avg'].float()
                self.state[p]['exp_avg_sq'] = self.state[p]['exp_avg_sq'].float()
                self.state[p]['fp32_p'] = self.state[p]['fp32_p'].float()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
          closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data.float()
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()
                    # Fp32 copy of the weights
                    state['fp32_p'] = p.data.float()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], state['fp32_p'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                state['fp32_p'].addcdiv_(-step_size, exp_avg, denom)
                p.data = state['fp32_p'].half()

        return loss