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
                            'filter_hd', 'n_subsampled_data']:
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


        if self.config['eval_reference_set'] == 'val' and self.config['longtail']:
            stratified_m = True
        else:
            stratified_m = False
        if self.config['longtail']:
            samples_per_class = np.load('imagenet_lt/imagenet_lt_samples_per_class.npy',
                                    allow_pickle=True)
        else:
            samples_per_class = None

        get_inception_metrics = inception_utils.prepare_inception_metrics(
            im_reference_filename, samples_per_class, self.config['parallel'], self.config['no_fid'],
            self.config['data_root'], prdc=self.config['eval_prdc'], stratified_fid=stratified_m)

        # If computing PRDC, we need a loader to obtain reference Inception features
        if self.config['eval_prdc']:
            prdc_ref_set = data_utils.get_dataset_hdf5(
                    **{**self.config, 'data_path': self.config['data_root'],
                       'load_in_mem_feats': self.config['load_in_mem'],
                       'kmeans_subsampled':False,
                       'test_part': True if self.config['which_dataset']=='coco' and
                                            self.config['eval_reference_set']=='val' else False,
                       'split': self.config['eval_reference_set'], 'ddp': False})
            prdc_loader = data_utils.get_dataloader(
                **{**self.config, 'dataset': prdc_ref_set,
                   'batch_size': self.config['batch_size'],
                   'use_checkpointable_sampler':False, 'shuffle':True,
                   'drop_last':False})
        else:
            prdc_loader=None


        # Get metrics
        print('Number of inception images is ', self.config['num_inception_images'])
        eval_metrics = get_inception_metrics(sample,
                                 num_inception_images=self.config['num_inception_images'],
                                 num_splits=10, prints=False,
                                 loader_ref=prdc_loader,
                                num_pr_images=self.config['num_inception_images']
                                if (self.config['longtail']
                                    and self.config['eval_reference_set']=='train') else 10000)
        eval_metrics_dict = dict()
        if self.config['eval_prdc']:
            IS_mean, IS_std, FID, stratified_FID, prdc_metrics = eval_metrics
        else:
            IS_mean, IS_std, FID, stratified_FID = eval_metrics
        if stratified_m:
            eval_metrics_dict['stratified_FID'] = stratified_FID

        eval_metrics_dict['IS_mean'] = IS_mean
        eval_metrics_dict['IS_std'] = IS_std
        eval_metrics_dict['FID'] = FID
        eval_metrics_dict['tested_itr'] = self.state_dict['itr']
        print(eval_metrics_dict)
        if self.config['eval_prdc']:
            eval_metrics_dict = {**prdc_metrics, **eval_metrics_dict}

        add_suffix = ''
        if self.config['z_var']!=1.0:
            add_suffix = '_z_var'+str(self.config['z_var'])
        if not os.path.exists(
                os.path.join(self.config['samples_root'], experiment_name)):
            os.mkdir(os.path.join(self.config['samples_root'], experiment_name))
        np.save(os.path.join(self.config['samples_root'], experiment_name,
                             'eval_metrics_reference_' +
                             self.config['eval_reference_set'] +
                             '_instances_'+ self.config['eval_instance_set'] +
                             '_kmeans' + str(self.config['kmeans_subsampled'])+
                             '_seed'+str(self.config['seed'])+add_suffix+'.npy'),
                eval_metrics_dict)
        print('Computed metrics:')
        for key, value in eval_metrics_dict.items():
            print(key, ': ', value)

        if self.config['sample_npz']:
        #TODO: save images for COCO
            # Sample a number of images and save them to an NPZ, for use with TF-Inception
            # Lists to hold images and labels for images
            if not os.path.exists(os.path.join(self.config['samples_root'],
                                               self.config['experiment_name'])):
                os.mkdir(os.path.join(self.config['samples_root'],
                                      self.config['experiment_name']))
            x, y = [], []
            print('Sampling %d images and saving them to npz...' % self.config[
                'sample_num_npz'])
            dict_tosave = {}
            counter_i = 0
            for i in trange(
                    int(np.ceil(
                        self.config['sample_num_npz'] / float(
                            self.config['batch_size'])))):
                with torch.no_grad():
                    images, labels, _ = sample()
                x += [images.cpu().numpy()]
                if self.config['class_cond']:
                    y += [labels.cpu().numpy()]
            if self.config['which_dataset'] == 'imagenet':
                x = np.concatenate(x, 0)[:self.config['sample_num_npz']]
                if self.config['class_cond']:
                    y = np.concatenate(y, 0)[:self.config['sample_num_npz']]

                np_filename = '%s/%s/samples%s_seed%i.pickle' % (
                    self.config['samples_root'], self.config['experiment_name'],
                    '_kmeans' + str(self.config['kmeans_subsampled']) if
                    self.config['kmeans_subsampled'] >-1 else '', self.config['seed'])
                print('Saving npy to %s...' % np_filename)
                dict_tosave['x'] = x
                dict_tosave['y'] = y
                file_to_store = open(np_filename, "wb")
                pickle.dump(dict_tosave, file_to_store,protocol=4)
                file_to_store.close()

                if self.config['longtail'] and self.config['eval_reference_set'] == 'val':
                    print('Also storing stratified samples')

                    for strat_name in ['_many', '_low', '_few']:
                        np_filename = '%s/%s/samples%s_seed%i_strat%s.pickle' % (
                            self.config['samples_root'],
                            self.config['experiment_name'],
                            '_kmeans' + str(self.config['kmeans_subsampled']) if
                            self.config['kmeans_subsampled'] > -1 else '',
                            self.config['seed'], strat_name)
                        print(np_filename)
                        if strat_name == '_many':
                            x_ = x[samples_per_class[y] >= 100]
                            y_ = y[samples_per_class[y] >= 100]
                        elif strat_name == '_low':
                            x_ = x[samples_per_class[y] < 100]
                            y_ = y[samples_per_class[y] < 100]
                            x_ = x_[samples_per_class[y_] > 20]
                            y_ = y_[samples_per_class[y_] > 20]
                        elif strat_name == '_few':
                            x_ = x[samples_per_class[y] <= 20]
                            y_ = y[samples_per_class[y] <= 20]
                        dict_tosave = {}
                        dict_tosave['x'] = x_
                        dict_tosave['y'] = y_
                        file_to_store = open(np_filename, "wb")
                        pickle.dump(dict_tosave, file_to_store, protocol=4)
                        file_to_store.close()

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
