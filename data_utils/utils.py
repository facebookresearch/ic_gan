import sys
import os
import numpy as np
import time
import datetime
import json
import math
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import shutil
import torch.distributed as dist

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from data_utils.resnet import resnet50
import data_utils.datasets_common as dset
from data_utils.cocostuff_dataset import CocoStuff


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


def load_pretrained_feature_extractor(pretrained_path='',
                                      feature_extractor='classification',
                                      backbone_feature_extractor='resnet50'):
    if backbone_feature_extractor == 'resnet50':
        print('using resnet50 to extract features')
        net = resnet50(pretrained=False if pretrained_path != '' else True,
                       classifier_run=False).cuda()
    else:
        raise ValueError('Not implemented for other backbones other than ResNet50.')
    if pretrained_path != '':
        print("Loading pretrained weights from: ", pretrained_path)

        # original saved file with DataParallel
        state_dict = torch.load(pretrained_path)
        if not feature_extractor == 'selfsupervised':
            state_dict = state_dict['state_dict_best']['feat_model']

        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module.' in k:
                name = k[7:]  # remove `module.`
            elif '_feature_blocks.' in k:
                name = k.replace('_feature_blocks.', '')
            else:
                name = k
            if name in net.state_dict().keys():
                new_state_dict[name] = v
            else:
                print('key ', name, ' not in dict')

        for key in net.state_dict().keys():
            if key not in new_state_dict.keys():
                print('Network key ', key, ' not in dict to load')
        if not feature_extractor == 'selfsupervised':
            state_dict = torch.load(pretrained_path)['state_dict_best']['classifier']
            # create new OrderedDict that does not contain `module.`
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
        # load params
        net.load_state_dict(new_state_dict, strict=False if
        feature_extractor == 'selfsupervised' else True)
    else:
        print('Using pretrained weights on full ImageNet.')
    return net

def get_dataset_images(resolution, data_path, load_in_mem=False, augment=False,longtail=False,
                       split='train', test_part=False, which_dataset='imagenet', instance_json='',
                       stuff_json='', **kwargs):
    # Data transforms
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    if which_dataset not in ['coco']:
        transform_list = [CenterCropLongEdge(), transforms.Resize(resolution)]
    else:
        transform_list = [transforms.Resize(resolution)]
    transform_list = transforms.Compose(transform_list + [
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)])
    if augment:
        print('>>>>>>Augmenting images from directory with flips!')
        transform_list = transforms.Compose(transform_list + [
            transforms.RandomHorizontalFlip()])

    if which_dataset not in ['coco']:
        which_dataset_file = dset.ImageFolder
        dataset_kwargs = {}
    else:
        print('Using coco-stuff dataset class')
        which_dataset_file = CocoStuff
        dataset_kwargs = {'image_dir': data_path,
                          'instances_json': instance_json,
                          'stuff_json': stuff_json,
                          'image_size': resolution,
                          'iscrowd':True if split=='train' else False,
                          'test_part': test_part}
    dataset = which_dataset_file(root=data_path, transform=transform_list,
                            load_in_mem=load_in_mem, split=split, longtail=longtail, **dataset_kwargs)
    return dataset


def get_dataset_hdf5(resolution, data_path, augment=False,longtail=False,
                local_rank=0, copy_locally=False, ddp=True, tmp_dir='',
                class_cond=True, instance_cond=False, feature_extractor='classification',
                backbone_feature_extractor='resnext50',
                which_nn_balance='instance_balance', which_dataset='imagenet', split='train',
                test_part=False, kmeans_subsampled=-1, n_subsampled_data=-1,
                feature_augmentation=False, filter_hd=-1, k_nn=50,
                load_in_mem_feats=False, compute_nns=False, **kwargs):

    if which_dataset in ['imagenet', 'imagenet_lt']:
        dataset_name_prefix = 'ILSVRC'
    elif which_dataset == 'coco':
        dataset_name_prefix = 'COCO'
    elif which_dataset == 'cityscapes':
        dataset_name_prefix = 'CS'
    elif which_dataset == 'metfaces':
        dataset_name_prefix = 'METF'
    elif which_dataset == 'sketches':
        dataset_name_prefix = 'SKETCH'
    elif which_dataset == 'cartoon':
        dataset_name_prefix = 'CARTOON'
    elif which_dataset == 'cub':
        dataset_name_prefix = 'CUB'
    else:
        raise ValueError('Dataset not prepared.')
    # HDF5 file name
    hdf5_filename = '%s%i%s%s%s' % (
        dataset_name_prefix,
        resolution,
        '' if not longtail else 'longtail',
        '_val' if split == 'val' else '',
        '_test' if test_part else ''
    )

    # Data paths
    data_path_xy = os.path.join(data_path, hdf5_filename + '_xy.hdf5')
    data_path_feats, data_path_nns, kmeans_file = None, None,  None
    if instance_cond:
        data_path_feats = os.path.join(data_path,
                                           hdf5_filename + '_feats_%s_%s.hdf5'%
                                           (feature_extractor, backbone_feature_extractor))
        if not compute_nns:
            data_path_nns = os.path.join(data_path,
                                     hdf5_filename + '_feats_%s_%s_nn_k%i.hdf5' %
                                     (feature_extractor,
                                      backbone_feature_extractor,
                                      k_nn))
        if kmeans_subsampled > -1:
            d_name = 'IN' if which_dataset == 'imagenet' else 'COCO'
            kmeans_file = d_name + '_res' + str(
                resolution) + '_rn50_' + feature_extractor + \
                          '_kmeans_k' + str(kmeans_subsampled) + '.npy'


    # Optionally copy the data locally in the cluster.
    if copy_locally:
        tmp_file = os.path.join(tmp_dir, hdf5_filename + '_xy.hdf5')
        print(tmp_file)
        if instance_cond:
            tmp_file_feats = os.path.join(tmp_dir,
                                      hdf5_filename + '_feats_%s_%s.hdf5' %
                                      (feature_extractor,
                                       backbone_feature_extractor))
            print(tmp_file_feats)

        # Only copy locally for the first device in each machine
        if local_rank == 0: #device == 'cuda:0':
            shutil.copy2(data_path_xy, tmp_file)
            if instance_cond:
                shutil.copy2(data_path_feats, tmp_file_feats)
        data_path_xy = tmp_file
        if instance_cond:
            data_path_feats = tmp_file_feats

        # Wait for the main process to copy the data locally
        if ddp:
            dist.barrier()

    # Data transforms
    if augment:
        print('>>>>>>Augmenting images from hdf5 file with flips!')
        transform_list = transforms.RandomHorizontalFlip()
    else:
        transform_list = None

    dataset = dset.ILSVRC_HDF5_feats(root=data_path_xy, root_feats=data_path_feats,
                                     root_nns=data_path_nns,
                                     transform=transform_list,
                                    load_labels= class_cond, load_features=instance_cond,
                                    load_in_mem_images=False, load_in_mem_labels=True,
                                    load_in_mem_feats=load_in_mem_feats, k_nn=k_nn,
                                    which_nn_balance=which_nn_balance,
                                    kmeans_file=kmeans_file,
                                    n_subsampled_data=n_subsampled_data,
                                    feature_augmentation=feature_augmentation,
                                    filter_hd=filter_hd)
    return dataset


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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() + worker_id