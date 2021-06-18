""" Convert dataset to HDF5
    This script preprocesses a dataset and saves it (images and labels) to 
    an HDF5 file for improved I/O. """
import os
import sys
from argparse import ArgumentParser
from tqdm import tqdm, trange
import h5py as h5

import numpy as np
import torch
import utils


def prepare_parser():
    usage = 'Parser for ImageNet HDF5 scripts.'
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '--resolution', type=int, default=128,
        help='Which Dataset resolution to train on, out of 64, 128, 256, 512 (default: %(default)s)')
    parser.add_argument(
        '--split', type=str, default='train',
        help='Which Dataset to convert: train, val (default: %(default)s)')
    parser.add_argument(
        '--data_root', type=str, default='data',
        help='Default location where data is stored (default: %(default)s)')
    parser.add_argument(
      '--out_path', type=str, default='data',
      help='Default location where data in hdf5 format will be stored (default: %(default)s)')
    parser.add_argument(
        '--num_workers', type=int, default=16,
        help='Number of dataloader workers (default: %(default)s)')
    parser.add_argument(
        '--chunk_size', type=int, default=500,
        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
        '--compression', action='store_true', default=False,
        help='Use LZF compression? (default: %(default)s)')

    parser.add_argument(
        '--feature_extractor', type=str, default='classification',
        choices=['classification','selfsupervised'],
        help='Choice of feature extractor')
    parser.add_argument(
        '--backbone_feature_extractor', type=str, default='resnet50',
        choices=['resnet50'],
        help='Choice of feature extractor backbone')
    parser.add_argument(
        '--k_nn', type=int, default=100,
        help='Number of nearest neighbors (default: %(default)s)')


    parser.add_argument(
        '--which_dataset', type=str, default='imagenet',
        choices=['imagenet', 'imagenet_lt', 'coco', 'cityscapes', 'cub', 'metfaces','sketches','cartoon'],
        help='Dataset choice.')

    return parser


def run(config):
    # Update compression entry
    config['compression'] = 'lzf' if config[
        'compression'] else None  # No compression; can also use 'lzf'

    test_part=False
    if config['split'] == 'test':
        config['split'] = 'val'
        test_part= True
    if config['which_dataset'] in ['imagenet','imagenet_lt']:
        dataset_name_prefix = 'ILSVRC'
    elif config['which_dataset'] == 'coco':
        dataset_name_prefix = 'COCO'
    elif config['which_dataset'] == 'cityscapes':
        dataset_name_prefix = 'CS'
    elif config['which_dataset'] == 'metfaces':
        dataset_name_prefix = 'METF'
    elif config['which_dataset'] == 'sketches':
        dataset_name_prefix = 'SKETCH'
    elif config['which_dataset'] == 'cartoon':
        dataset_name_prefix = 'CARTOON'
    elif config['which_dataset'] == 'cub':
        dataset_name_prefix = 'CUB'
    else:
        raise ValueError('Dataset not prepared.')

    train_dataset = utils.get_dataset_hdf5(
        **{'resolution':config['resolution'], 'data_path': config['data_root'],
           'load_in_mem_feats': True, 'compute_nns': True,
           'longtail':config['which_dataset']=='imagenet_lt',
           'split': config['split'], 'instance_cond': True,
        'feature_extractor': config['feature_extractor'],
        'backbone_feature_extractor':config['backbone_feature_extractor'],
           'k_nn': config['k_nn'], 'ddp': False,
           'which_dataset': config['which_dataset'],
           'test_part': test_part
           })

    for k_nn in [500, 100, 50, 20, 5]: #[1200]: #
        all_nns = np.array(train_dataset.sample_nns)[:, :k_nn]
        all_nns_radii = train_dataset.kth_values[:,k_nn]
        print('NNs shape ', all_nns.shape, all_nns_radii.shape)
        labels_ = torch.Tensor(train_dataset.labels)
        acc = np.array([(labels_[all_nns[:,i_nn]] ==
              labels_).sum() for i_nn in range(k_nn)]).sum()/\
        (len(labels_)*k_nn)
        print('For k ', k_nn, ' accuracy:', acc)

        h5file_name_nns = config[
                              'out_path'] + '/%s%i%s%s%s_feats_%s_%s_nn_k%i.hdf5' % (
            dataset_name_prefix,
                              config['resolution'],
                              '' if config['which_dataset']!='imagenet_lt' else 'longtail',
                              '_val' if config['split'] == 'val' else '',
                                '_test' if test_part else '',
            config['feature_extractor'], config['backbone_feature_extractor'],k_nn)
        print('Filename is ', h5file_name_nns)

        with h5.File(h5file_name_nns, 'w') as f:
            nns_dset = f.create_dataset('sample_nns', all_nns.shape, dtype='int64',
                                                 maxshape=all_nns.shape,
                                                 chunks=(config['chunk_size'], all_nns.shape[1]),
                                                 compression=config['compression'])
            nns_dset[...] = all_nns

            nns_radii_dset = f.create_dataset('sample_nns_radius', all_nns_radii.shape,
                                        dtype='float',
                                        maxshape=all_nns_radii.shape,
                                        chunks=(
                                        config['chunk_size'],),
                                        compression=config['compression'])
            nns_radii_dset[...] = all_nns_radii



def main():
    # parse command line and run
    parser = prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()
