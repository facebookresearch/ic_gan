""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

import os
import functools
import math
import numpy as np
from tqdm import tqdm, trange
import argparse
import time
import pickle
import json

from imageio import imwrite as imsave

import utils
# Import my stuff
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data_utils.inception_utils as inception_utils
import data_utils.utils as data_utils

class Tester:
    def __init__(self, config):
        self.config = vars(config) if not isinstance(config, dict) else config
        #   self.config['experiment_name'] = self.config['exp_name'] if 'exp_name' in self.config else self.config['experiment_name']
        if 'checkpoints_dir' in self.config:
            self.config['base_root'] = self.config['checkpoints_dir']

    def get_sampling_funct(self, instance_set='train', reference_set='train',
    which_dataset='imagenet'):
        # Class labels will follow either a long-tail
        # distribution(if reference==train) or a uniform distribution
        # otherwise).

        if self.config['longtail']:
            class_probabilities = np.load('imagenet_lt/imagenet_lt_class_prob.npy',
                                          allow_pickle=True)
            samples_per_class = np.load('imagenet_lt/imagenet_lt_samples_per_class.npy',
                                        allow_pickle=True)
        else:
            class_probabilities, samples_per_class = None, None

        if (reference_set=='val' and instance_set=='val') and config['which_dataset'] == 'coco':
            # using evaluation set
            test_part = True
        else:
            test_part = False

        print('Truncation value for z? ', self.config['z_var'])
        z_, y_ = utils.prepare_z_y(self.config['batch_size'], self.G.dim_z,
                                   self.config['n_classes'],
                                   device='cuda', fp16=self.config['G_fp16'],
                                   longtail_gen=self.config['longtail'] if
                                   reference_set=='train' else False,z_var=self.config['z_var'],
                                   class_probabilities=class_probabilities
                                   )

        if self.config['instance_cond']:
            dataset = data_utils.get_dataset_hdf5(
                **{**self.config, 'data_path': self.config['data_root'],
                   'batch_size': self.config['batch_size'],
                   'load_in_mem_feats': self.config['load_in_mem'],
                   'split': instance_set,
                   'test_part': test_part,
                   'augment': False,
                   'ddp':False})

        else:
            dataset = None

        weights_sampling=None
        nn_sampling_strategy = 'instance_balance'
        if self.config['instance_cond'] and self.config['class_cond'] and self.config['longtail']:
            nn_sampling_strategy = 'nnclass_balance'
            if reference_set == 'val':
                print('Sampling classes uniformly for generator.')
                # Sampling classes uniformly
                weights_sampling = None
            else:
                print('Balancing with weights=samples_per_class (long-tailed).')
                weights_sampling = samples_per_class

        sample_conditioning = functools.partial(
            utils.sample_conditioning_values,
            z_=z_, y_=y_, constant_conditioning = self.config['constant_conditioning'],
            batch_size=self.config['batch_size'], weights_sampling=weights_sampling,
            dataset=dataset, class_cond=self.config['class_cond'],
            instance_cond=self.config['instance_cond'], nn_sampling_strategy=nn_sampling_strategy)

        # Prepare Sample function for use with inception metrics
        sample = functools.partial(utils.sample,G=self.G,
                                   sample_conditioning_func=sample_conditioning,
                                   config=self.config, class_cond=self.config['class_cond'],
                                   instance_cond=self.config['instance_cond'])


        # Get reference statistics to compute FID
        im_prefix = 'I' if which_dataset == 'imagenet' else 'COCO'
        if reference_set == 'train':
            im_filename = '%s%i_%s' % (im_prefix,
                self.config['resolution'], '' if not self.config['longtail'] else 'longtail')
        else:
            im_filename = '%s%i_%s%s' % (im_prefix, self.config['resolution'], '_val',
                '_test' if test_part else '')
        print('Using ', im_filename, 'for Inception metrics.')
        return sample, im_filename, dataset

    def __call__(self) -> float:

        self.config = utils.update_config_roots(self.config, change_weight_folder=False)
        # Prepare state dict, which holds things like epoch # and itr #
        self.state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0,
                           'save_best_num': 0,
                           'best_IS': 0, 'best_FID': 999999,
                           'config': self.config}
        # Get override some parameters from trained model in experiment config
        utils.load_weights(None, None, self.state_dict, self.config['weights_root'],
                           self.config['experiment_name'], self.config['load_weights'],
                           None,
                           strict=False, load_optim=False, eval=True)

        # Ignore items which we might want to overwrite from the command line
        print('Before loading config ', self.config['sample_num_npz'])
        for item in self.state_dict['config']:
            if item not in ['base_root', 'data_root',
                            'batch_size','num_workers', 'weights_root',
                            'logs_root', 'samples_root', 'eval_reference_set',
                            'eval_instance_set', 'which_dataset',
                            'seed', 'eval_prdc',
                            'use_balanced_sampler', 'custom_distrib',
                            'longtail_temperature', 'longtail_gen',
                            'num_inception_images', 'sample_num_npz',
                            'load_in_mem', 'split', 'z_var', 'kmeans_subsampled',
                            'filter_hd', 'n_subsampled_data', 'feature_augmentation']:
                self.config[item] = self.state_dict['config'][item]

        device = 'cuda'

        # Seed RNG
        utils.seed_rng(self.config['seed'])

        # Prepare root folders if necessary
        utils.prepare_root(self.config)

        import torch
        # Setup cudnn.benchmark for free speed
        torch.backends.cudnn.benchmark = True

        # Import the model--this line allows us to dynamically select different files.
        model = __import__(self.config['model'])
        experiment_name = (
            self.config['experiment_name'] if self.config['experiment_name']
            else utils.name_from_config(self.config))
        print('Experiment name is %s' % experiment_name)

        # Next, build the model
        self.G = model.Generator(**self.config).to(device)
        utils.count_parameters(self.G)

        # Load weights
        print('Loading weights...')
        # Select best according to best checkpoint
        best_fid = 1e5
        best_name_final = ''
        for name_best in ['best0', 'best1']:
            try:
                root = '/'.join(
                    [self.config['weights_root'], self.config['experiment_name']])
                state_dict_loaded = torch.load('%s/%s.pth' % (
                    root, utils.join_strings('_', ['state_dict', name_best])))
                print('For name best ', name_best, ' we have an FID: ',
                      state_dict_loaded['best_FID'])
                if state_dict_loaded['best_FID'] < best_fid:
                    best_fid = state_dict_loaded['best_FID']
                    best_name_final = name_best
            except:
                print('Checkpoint with name ', name_best, ' not in folder.')
        print('Final name selected is ', best_name_final)
        self.config['load_weights'] = best_name_final

        print('Loading weights...')
        # Here is where we deal with the ema--load ema weights or load normal weights
        utils.load_weights(self.G if not (self.config['use_ema']) else None, None,
                           self.state_dict,
                           self.config['weights_root'], experiment_name,
                           self.config['load_weights'],
                           self.G if self.config['ema'] and self.config['use_ema'] else None,
                           strict=False, load_optim=False)

        if config['G_eval_mode']:
            print('Putting G in eval mode..')
            self.G.eval()
        else:
            print('G is in %s mode...' % ('training' if self.G.training else 'eval'))

        # Get sampling function and reference statistics for FID
        print('Eval reference set is ', self.config['eval_reference_set'])
        sample, im_reference_filename, dataset = \
            self.get_sampling_funct(instance_set=self.config['eval_instance_set'],
                                    reference_set=self.config['eval_reference_set'],
                                    which_dataset=self.config['which_dataset'])

        if config['which_dataset'] == 'coco':
            image_format = 'jpg'
        else:
            image_format = 'png'
        if self.config['eval_instance_set'] == 'test' and config['which_dataset'] == 'coco':
            # using evaluation set
            test_part = True
        else:
            test_part = False
        path_samples = os.path.join(self.config['samples_root'], self.config['experiment_name'],
                                    '%s_images_seed%i%s%s%s' % (config['which_dataset'],
                                    config['seed'], '_test' if test_part else '',
                                    '_hd' + str(self.config['filter_hd']) if self.config[
                                                                                 'filter_hd'] > -1 else '',
                                        '' if self.config['kmeans_subsampled']==-1 else
                                            '_'+str(self.config['kmeans_subsampled'])+'centers'))

        print('Path samples will be ', path_samples)
        if not os.path.exists(path_samples):
            os.makedirs(path_samples)

        if not os.path.exists(os.path.join(self.config['samples_root'],
                                           self.config['experiment_name'])):
            os.mkdir(os.path.join(self.config['samples_root'],
                                  self.config['experiment_name']))
        print('Sampling %d images and saving them with %s format...' % (self.config[
            'sample_num_npz'], image_format))
        counter_i = 0
        for i in trange(
                int(np.ceil(
                    self.config['sample_num_npz'] / float(
                        self.config['batch_size'])))):
            with torch.no_grad():
                images, labels, _ = sample()

                fake_imgs = images.cpu().detach().numpy().transpose(0, 2, 3, 1)
                fake_imgs = fake_imgs*0.5 + 0.5
                for fake_img in fake_imgs:
                    imsave("%s/%06d.%s" % (
                    path_samples, counter_i, image_format), fake_img)
                    counter_i+=1
                    if counter_i >= self.config['sample_num_npz']:
                        break


if __name__ == "__main__":
    parser = utils.prepare_parser()
    parser = utils.add_sample_parser(parser)
    config = vars(parser.parse_args())
    if config['json_config'] != "":
        data = json.load(open(config['json_config']))
        for key in data.keys():
            config[key] = data[key]
    else:
        print('No json file to load configuration from')


    tester = Tester(config)

    tester()

    # executor = submitit.SlurmExecutor(
    #     folder='/checkpoint/acasanova/submitit_logs_biggan',
    #     max_num_timeout=10)
    # executor.update_parameters(
    #     gpus_per_node=1, partition='prioritylab',
    #     comment='Neurips deadline',
    #     cpus_per_task=8, mem=128000,
    #     time=100, job_name='testing_'+config['experiment_name'])
    # executor.submit(tester)
    # import time
    #
    # time.sleep(1)
