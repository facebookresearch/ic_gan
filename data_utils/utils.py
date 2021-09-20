# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# All contributions by Andy Brock:
# Copyright (c) 2019 Andy Brock

# MIT License
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

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from data_utils.resnet import resnet50
import data_utils.datasets_common as dset
from data_utils.cocostuff_dataset import CocoStuff


class CenterCropLongEdge(object):
    """Crops the given PIL Image on the long edge.
    Parameters
    ----------
        size: sequence or int
            Desired output size of the crop. If size is an int instead of sequence like (h, w),
            a square crop (size, size) is made.
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


# Modified to be able to do class-balancing
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

    def __init__(
        self, dataset, num_replicas=None, rank=None, shuffle=True, weights=None
    ):
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
                print("using class balanced!")
                indices = torch.multinomial(
                    self.weights, len(self.dataset), replacement=True, generator=g
                ).tolist()
            else:
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class CheckpointedSampler(torch.utils.data.Sampler):
    """Resumable sample with a random generated initialized with a given seed.

    Arguments
    ---------
        data_source: Dataset
            Dataset to sample from.
        start_itr: int, optional
            Number of iteration to start (or restart) the sampling.
        start_epoch: int, optional
            Number of epoch to start (or restart) the sampling.
        batch_size: int, optional
            Batch size.
        class_balanced: bool, optional
            Sample the data with a class balancing approach.
        custom_distrib_gen: bool, optional
            Use a temperature controlled class balancing.
        samples_per_class: list, optional
            A list of int values that indicate the number of samples per class.
        class_probabilities: list, optional
            A list of float values indicating the probability of a class in the dataset.
        longtail_temperature: float, optional
            Temperature value to smooth the longtail distribution with a softmax function.
        seed: int, optional
            Random seed used.

    """

    def __init__(
        self,
        data_source,
        start_itr=0,
        start_epoch=0,
        batch_size=128,
        class_balanced=False,
        custom_distrib_gen=False,
        samples_per_class=None,
        class_probabilities=None,
        longtail_temperature=1,
        seed=0,
    ):
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.start_itr = start_itr % (len(self.data_source) // batch_size)
        self.start_epoch = start_epoch
        self.batch_size = batch_size
        self.class_balanced = class_balanced
        self.custom_distrib_gen = custom_distrib_gen
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        if self.class_balanced:
            print("Class balanced sampling.")
            self.weights = make_weights_for_balanced_classes(
                samples_per_class,
                self.data_source.labels,
                1000,
                self.custom_distrib_gen,
                longtail_temperature,
                class_probabilities=class_probabilities,
            )
            self.weights = torch.DoubleTensor(self.weights)

        # Resumable data loader
        print(
            "Using the generator ",
            self.start_epoch,
            " times to resume where we left off.",
        )
        # print('Later, we will resume at iteration ', self.start_itr)
        for epoch in range(self.start_epoch):
            self._sample_epoch_perm()

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integeral "
                "value, but got num_samples={}".format(self.num_samples)
            )

    def _sample_epoch_perm(self):
        if self.class_balanced:
            out = [
                torch.multinomial(
                    self.weights,
                    len(self.data_source),
                    replacement=True,
                    generator=self.generator,
                )
            ]
        else:
            out = [torch.randperm(len(self.data_source), generator=self.generator)]
        return out

    def __iter__(self):
        out = self._sample_epoch_perm()
        output = torch.cat(out).tolist()
        return iter(output)

    def __len__(self):
        return len(self.data_source)


def make_weights_for_balanced_classes(
    samples_per_class,
    labels=None,
    nclasses=None,
    custom_distrib_gen=False,
    longtail_temperature=1,
    class_probabilities=None,
):
    """It prepares the sampling weights for the DataLoader.

    Arguments
    ---------
        samples_per_class: list
            A list of int values (size C) that indicate the number of samples per class,
             for all C classes.
        labels: list/ NumPy array/ torch Tensor, optional
            A list of size N that contains a class label for each sample.
        nclasses: int, optional
            Number of classes in the dataset.
        custom_distrib_gen: bool, optional
            Use a temperature controlled class balancing.
        longtail_temperature: float, optional
            Temperature value to smooth the longtail distribution with a softmax function.
        class_probabilities: list
            A list of float values (size C) indicating the probability of a class in the dataset.
        seed: int
            Random seed used.
    Returns
    -------
    If custom_distrib_gen is True, a torch Tensor with size C, where C is the number of classes,
     that contains the sampling weights for each class.
    If custom_distrib_gen is False, a list with size N (dataset size) that contains the sampling
     weights for each individual data sample.

    """
    if custom_distrib_gen:
        # temperature controlled distribution
        print(
            "Temperature controlled distribution for balanced classes! " "Temperature:",
            longtail_temperature,
        )
        class_prob = torch.log(torch.DoubleTensor(class_probabilities))
        weight_per_class = torch.exp(class_prob / longtail_temperature) / torch.sum(
            torch.exp(class_prob / longtail_temperature)
        )
    else:
        count = [0] * nclasses
        for item in labels:
            count[item] += 1
        weight_per_class = [0.0] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            # Standard class balancing
            weight_per_class[i] = N / float(count[i])
    # Convert weighting per class to weighting per example
    weight = [0] * len(labels)
    for idx, val in enumerate(labels):
        # Uniform probability of selecting a sample, given a class
        # p(x|y)p(y)
        weight[idx] = (1 / samples_per_class[val]) * weight_per_class[val]
    return weight


def load_pretrained_feature_extractor(
    pretrained_path="",
    feature_extractor="classification",
    backbone_feature_extractor="resnet50",
):
    """It loads a pre-trained feature extractor.

    Arguments
    ---------
        pretrained_path: str, optional
            Path to the feature extractor's weights.
        feature_extractor: str, optional
            If "classification" a network trained on ImageNet for classification will be used. If
            "selfsupervised", a network trained on ImageNet with self-supervision will be used.
        backbone_feature_extractor: str, optional
            Name of the backbone for the feature extractor. Currently, only ResNet50 is supported.
    Returns
    -------
    A Pytorch network initialized with pre-trained weights.

    """
    if backbone_feature_extractor == "resnet50":
        print("using resnet50 to extract features")
        net = resnet50(
            pretrained=False if pretrained_path != "" else True, classifier_run=False
        ).cuda()
    else:
        raise ValueError("Not implemented for backbones other than ResNet50.")
    if pretrained_path != "":
        print("Loading pretrained weights from: ", pretrained_path)

        # original saved file with DataParallel
        state_dict = torch.load(pretrained_path)
        if not feature_extractor == "selfsupervised":
            state_dict = state_dict["state_dict_best"]["feat_model"]

        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "module." in k:
                name = k[7:]  # remove `module.`
            elif "_feature_blocks." in k:
                name = k.replace("_feature_blocks.", "")
            else:
                name = k
            if name in net.state_dict().keys():
                new_state_dict[name] = v
            else:
                print("key ", name, " not in dict")

        for key in net.state_dict().keys():
            if key not in new_state_dict.keys():
                print("Network key ", key, " not in dict to load")
        if not feature_extractor == "selfsupervised":
            state_dict = torch.load(pretrained_path)["state_dict_best"]["classifier"]
            # create new OrderedDict that does not contain `module.`
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
        # load params
        net.load_state_dict(
            new_state_dict,
            strict=False if feature_extractor == "selfsupervised" else True,
        )
    else:
        print("Using pretrained weights on full ImageNet.")
    return net


def get_dataset_images(
    resolution,
    data_path,
    load_in_mem=False,
    augment=False,
    longtail=False,
    split="train",
    test_part=False,
    which_dataset="imagenet",
    instance_json="",
    stuff_json="",
    **kwargs
):
    """It prepares a dataset that reads the files from a folder.

    Arguments
    ---------
        resolution: int
            Dataset resolution.
        data_path: str
            Path where to find the data.
        load_in_mem: bool, optional
            If True, load all data in memory.
        augment: bool, optional
            If True, use horizontal flips as data augmentation.
        longtail: bool, optional
            If True, use the longtailed version of ImageNet (ImageNet-LT).
        split: str, optional
            Split name to use.
        test_part: bool, optional
            Only used for COCO-Stuff. If True, use the evaluation set instead of the validation set.
        which_dataset: str, optional
            Dataset name.
        instance_json: str, optional
            Path where to find the JSON data for COCO-Stuff instances.
        stuff_json: str, optional
            Path where to find the JSON data for COCO-Stuff stuff.
    Returns
    -------
    A Dataset class.

    """
    # Data transforms
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    if which_dataset not in ["coco"]:
        transform_list = [CenterCropLongEdge(), transforms.Resize(resolution)]
    else:
        transform_list = [transforms.Resize(resolution)]
    transform_list = transforms.Compose(
        transform_list
        + [transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)]
    )
    if augment:
        transform_list = transforms.Compose(
            transform_list + [transforms.RandomHorizontalFlip()]
        )

    if which_dataset not in ["coco"]:
        which_dataset_file = dset.ImageFolder
        dataset_kwargs = {}
    else:
        print("Using coco-stuff dataset class")
        which_dataset_file = CocoStuff
        dataset_kwargs = {
            "image_dir": data_path,
            "instances_json": instance_json,
            "stuff_json": stuff_json,
            "image_size": resolution,
            "iscrowd": True if split == "train" else False,
            "test_part": test_part,
        }
    dataset = which_dataset_file(
        root=data_path,
        transform=transform_list,
        load_in_mem=load_in_mem,
        split=split,
        longtail=longtail,
        **dataset_kwargs
    )
    return dataset


def get_dataset_hdf5(
    resolution,
    data_path,
    augment=False,
    longtail=False,
    local_rank=0,
    copy_locally=False,
    ddp=True,
    tmp_dir="",
    class_cond=True,
    instance_cond=False,
    feature_extractor="classification",
    backbone_feature_extractor="resnext50",
    which_nn_balance="instance_balance",
    which_dataset="imagenet",
    split="train",
    test_part=False,
    kmeans_subsampled=-1,
    n_subsampled_data=-1,
    feature_augmentation=False,
    filter_hd=-1,
    k_nn=50,
    load_in_mem_feats=False,
    compute_nns=False,
    **kwargs
):
    """It prepares a dataset that reads the data from HDF5 files.

    Arguments
    ---------
        resolution: int
            Dataset resolution.
        data_path: str
            Path where to find the data.
        load_in_mem: bool, optional
            If True, load all data in memory.
        augment: bool, optional
            If True, use horizontal flips as data augmentation.
        longtail: bool, optional
            If True, use the longtailed version of ImageNet (ImageNet-LT).
        local_rank: int, optional
            Index indicating the rank of the DistributedDataParallel (DDP) process in the local
             machine. It is set to 0 by default or if DDP is not used.
        copy_locally: bool, optional
            If true, the HDF5 files will be copied locally to the machine.
             Useful if the data is in a server.
        ddp: bool, optional
            If True, use DistributedDataParallel (DDP).
        tmp_dir: str, optional
            Path where to copy the dataset HDF5 files locally.
        class_cond: bool, optional
            If True, the dataset will load the labels of the neighbor real samples.
        instance_cond: bool, optional
            If True, the dataset will load the instance features.
        feature_extractor: str, optional
            If "classification" a network trained on ImageNet for classification will be used. If
            "selfsupervised", a network trained on ImageNet with self-supervision will be used.
        backbone_feature_extractor: str, optional
            Name of the backbone for the feature extractor. Currently, only ResNet50 is supported.
        which_nn_balance: str, optional
            Whether to sample an instance or a neighbor class first. By default,
            ``instance_balance`` is used. Using ``nnclass_balance`` allows class balancing
             to be applied.
        split: str, optional
            Split name to use.
        test_part: bool, optional
            Only used for COCO-Stuff. If True, use the evaluation set instead of the validation set.
        kmeans_subsampled: int, optional
            If other than -1, that number of data points are selected with k-means from the dataset.
            It reduces the amount of available data to train or test the model.
        n_subsampled_data: int, optional
            If other than -1, that number of data points are randomly selected from the dataset.
            It reduces the amount of available data to train or test the model.
        feature_augmentation: bool, optional
            Use the instance features of the flipped ground-truth image instances as
            conditioning, with a 50% probability.
        filter_hd: int, optional
            Only used for COCO-Stuff dataset. If -1, all COCO-Stuff evaluation set is used.
            If 0, only images with seen class combinations are used.
            If 1, only images with unseen class combinations are used.
        k_nn: int, optional
            Size of the neighborhood obtained with the k-NN algorithm.
        load_in_mem_feats: bool, optional
            Load all instance features in memory.
        compute_nns: bool, optional
            If True, compute the nearest neighbors. If False, load them from a file with
            pre-computed neighbors.
    Returns
    -------
    A Dataset class.

    """

    if which_dataset in ["imagenet", "imagenet_lt"]:
        dataset_name_prefix = "ILSVRC"
    elif which_dataset == "coco":
        dataset_name_prefix = "COCO"
    else:
        dataset_name_prefix = which_dataset
    # HDF5 file name
    hdf5_filename = "%s%i%s%s%s" % (
        dataset_name_prefix,
        resolution,
        "" if not longtail else "longtail",
        "_val" if split == "val" else "",
        "_test" if test_part else "",
    )

    # Data paths
    data_path_xy = os.path.join(data_path, hdf5_filename + "_xy.hdf5")
    data_path_feats, data_path_nns, kmeans_file = None, None, None
    if instance_cond:
        data_path_feats = os.path.join(
            data_path,
            hdf5_filename
            + "_feats_%s_%s.hdf5" % (feature_extractor, backbone_feature_extractor),
        )
        if not compute_nns:
            data_path_nns = os.path.join(
                data_path,
                hdf5_filename
                + "_feats_%s_%s_nn_k%i.hdf5"
                % (feature_extractor, backbone_feature_extractor, k_nn),
            )
        # Load a file with indexes of the samples selected with k-means.
        if kmeans_subsampled > -1:
            if which_dataset == "imagenet":
                d_name = "IN"
            elif which_dataset == "coco":
                d_name = "COCO"
            else:
                d_name = which_dataset
            kmeans_file = (
                d_name
                + "_res"
                + str(resolution)
                + "_rn50_"
                + feature_extractor
                + "_kmeans_k"
                + str(kmeans_subsampled)
                + ".npy"
            )
            kmeans_file = os.path.join(data_path, kmeans_file)

    # Optionally copy the data locally in the cluster.
    if copy_locally:
        tmp_file = os.path.join(tmp_dir, hdf5_filename + "_xy.hdf5")
        print(tmp_file)
        if instance_cond:
            tmp_file_feats = os.path.join(
                tmp_dir,
                hdf5_filename
                + "_feats_%s_%s.hdf5" % (feature_extractor, backbone_feature_extractor),
            )
            print(tmp_file_feats)

        # Only copy locally for the first device in each machine
        if local_rank == 0:  # device == 'cuda:0':
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
        transform_list = transforms.RandomHorizontalFlip()
    else:
        transform_list = None

    dataset = dset.ILSVRC_HDF5_feats(
        root=data_path_xy,
        root_feats=data_path_feats,
        root_nns=data_path_nns,
        transform=transform_list,
        load_labels=class_cond,
        load_features=instance_cond,
        load_in_mem_images=False,
        load_in_mem_labels=True,
        load_in_mem_feats=load_in_mem_feats,
        k_nn=k_nn,
        which_nn_balance=which_nn_balance,
        kmeans_file=kmeans_file,
        n_subsampled_data=n_subsampled_data,
        feature_augmentation=feature_augmentation,
        filter_hd=filter_hd,
    )
    return dataset


def filter_by_hd(ood_distance):
    """Pre-select image indexes in COCO-Stuff evaluation set according to its class composition.

    Parameters
    ----------
        ood_distance: int
            Minimum hamming distance (HD) between the set of classes present in the evaluation image
            and all training images.
            If 0, pre-selected images will be the ones that only contain class sets already seen
             during training.
            If other than 0, all other images with unseen class sets will be selected,
             regardless of the hamming distance (HD>0).
    Returns
    -------
        List of pre-selected images.
    """

    image_ids_original = np.load(
        "../coco_stuff_val_indexes/cocostuff_val2_all_idxs.npy", allow_pickle=True
    )
    print("Filtering new ids!")
    odd_image_ids = np.load(
        os.path.join(
            "../coco_stuff_val_indexes", "val2" + "_image_ids_by_hd_75ktraining_im.npy"
        ),
        allow_pickle=True,
    )
    if ood_distance == 0:
        image_ids = odd_image_ids[ood_distance]
    else:
        total_img_ids = []
        for ood_dist in range(1, len(odd_image_ids)):
            total_img_ids += odd_image_ids[ood_dist]
        image_ids = total_img_ids

    allowed_idxs = []
    for i_idx, id in enumerate(image_ids_original):
        if id in image_ids:
            allowed_idxs.append(i_idx)
    allowed_idxs = np.array(allowed_idxs)
    print("Num images after filtering ", len(allowed_idxs))
    return allowed_idxs


def get_dataloader(
    dataset,
    batch_size=64,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    start_itr=0,
    start_epoch=0,
    use_checkpointable_sampler=False,
    use_balanced_sampler=False,
    custom_distrib_gen=False,
    samples_per_class=None,
    class_probabilities=None,
    seed=0,
    longtail_temperature=1,
    rank=0,
    world_size=-1,
    **kwargs
):
    """Get DataLoader to iterate over the dataset.

    Parameters
    ----------
        dataset: Dataset
            Class with the specified dataset characteristics.
        batch_size: int, optional
            Batch size.
        num_workers: int, optional
            Number of workers for the dataloader.
        shuffle: bool, optional
            If True, the data is shuffled. If a sampler is used (use_checkpointable_sampler=True,
            use_balanced_sampler=True or world_size>-1), this parameter is not used.
        pin_memory: bool, optional
            Pin memory in the dataloader.
        drop_last: bool, optional
            Drop last incomplete batch in the dataloader.
        start_itr: int, optional
            Iteration number to resume the sample from. Only used with
             use_checkpointable_sampler=True.
        start_epoch: int, optional
            Epoch number to resume the sample from. Only used with
             use_checkpointable_sampler=True.
        use_checkpointable_sampler: bool, optional
            If True, use the CheckpointedSampler class to resume jobs from the last seen batch
             (deterministic).
        use_balanced_sampler: bool, optional
            If True, balance the data according to a specific class distribution. Use in conjunction
             with ``custom_distrib_gen``, ``samples_per_class``, ``class_probabilities`` and
              ``longtail_temperature``.
        custom_distrib_gen: bool, optional
            Use a temperature controlled class balancing.
        samples_per_class: list, optional
            A list of int values that indicate the number of samples per class.
        class_probabilities: list, optional
            A list of float values indicating the probability of a class in the dataset.
        longtail_temperature: float, optional
            Temperature value to smooth the longtail distribution with a softmax function.
        seed: int, optional
            Random seed used.
        rank: int, optional
            Rank of the current process (if using DistributedDataParallel training).
        world_size: int, optional
            World size (if using DistributedDataParallel training).
    Returns
    -------
        An instance of DataLoader.
    """

    # Prepare loader; the loaders list is for forward compatibility with
    # using validation / test splits.
    # if use_multiepoch_sampler:
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    print("Dropping last batch? ", drop_last)
    # Otherwise, it has issues dividing the batch for accumulations
    # if longtail:
    #   loader_kwargs.update({'drop_last': drop_last})
    if use_checkpointable_sampler:
        print(
            "Using checkpointable sampler from start_itr %d..., using seed %d"
            % (start_itr, seed)
        )

        sampler = CheckpointedSampler(
            dataset,
            start_itr,
            start_epoch,
            batch_size,
            class_balanced=use_balanced_sampler,
            custom_distrib_gen=custom_distrib_gen,
            longtail_temperature=longtail_temperature,
            samples_per_class=samples_per_class,
            class_probabilities=class_probabilities,
            seed=seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            worker_init_fn=seed_worker,
            **loader_kwargs
        )
    else:
        if use_balanced_sampler:
            print("Balancing real data! Custom? ", custom_distrib_gen)
            weights = make_weights_for_balanced_classes(
                samples_per_class,
                dataset.labels,
                1000,
                custom_distrib_gen,
                longtail_temperature,
                class_probabilities=class_probabilities,
            )
            weights = torch.DoubleTensor(weights)
        else:
            weights = None
        if world_size == -1:
            if use_balanced_sampler:
                sampler = torch.utils.data.sampler.WeightedRandomSampler(
                    weights, len(weights)
                )
                shuffle = False
            else:
                sampler = None
        else:
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, weights=weights
            )
            shuffle = False
        print("Loader workers?", loader_kwargs, " with shuffle?", shuffle)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            worker_init_fn=seed_worker if use_checkpointable_sampler else None,
            **loader_kwargs
        )

    return loader


def sample_conditioning_values(
    z_,
    y_,
    ddp=False,
    batch_size=1,
    weights_sampling=None,
    dataset=None,
    constant_conditioning=False,
    class_cond=True,
    instance_cond=False,
    nn_sampling_strategy="instance_balance",
):
    """It samples conditionings from the noise distribution and dataset statistics.

    Arguments
    ---------
        z_: Distribution
            Noise distribution.
        y_: Distribution
            Labels distribution (
        ddp: bool, optional
            If True, use DistributedDataParallel (DDP).
        batch_size: int, optional
            Batch size.
        weights_sampling: NumPy array, optional
            Weights to balance the sampling of the conditionings.
        dataset: Dataset
            Instance of a dataset.
        constant_conditioning: bool, optional
            If True, set all labels to zero.
        class_cond: bool, optional
            If True, the dataset will load the labels of the neighbor real samples.
        instance_cond: bool, optional
            If True, the dataset will load the instance features.
        nn_sampling_strategy: str, optional
            Whether to sample an instance or a neighbor class first. By default,
            ``instance_balance`` is used. Using ``nnclass_balance`` allows class balancing
             to be applied.
    Returns
    -------
        If not using labels (class_cond=False) nor instance features (instance_cond=False),
         return the sampled noise vectors.
        If not using labels (class_cond=False), return the sampled noise vectors and instance
        feature vectors, sampled according to the ``nn_sampling_strategy`` and ``weights_sampling``.
        If using labels (class_cond=True), return the sampled noise vectors, instance feature
         vectors and the neighbor class labels.

    """
    with torch.no_grad():
        z_.sample_()
        if not class_cond and not instance_cond:
            return z_
        elif class_cond and not instance_cond:
            y_.sample_()
            if constant_conditioning:
                return z_, torch.zeros_like(y_)
            else:
                if ddp:
                    return z_, y_
                else:
                    return z_, y_.data.clone()
        else:
            if nn_sampling_strategy == "instance_balance":
                sampling_funct_name = dataset.sample_conditioning_instance_balance
            elif nn_sampling_strategy == "nnclass_balance":
                sampling_funct_name = dataset.sample_conditioning_nnclass_balance

            labels_g, f_g = sampling_funct_name(batch_size, weights_sampling)
            if instance_cond and not class_cond:
                return z_, f_g
            elif instance_cond and class_cond:
                return z_, labels_g, f_g


# Convenience function to prepare a z and y vector
def prepare_z_y(
    G_batch_size,
    dim_z,
    nclasses,
    device="cuda",
    fp16=False,
    z_var=1.0,
    longtail_gen=False,
    custom_distrib=False,
    longtail_temperature=1,
    class_probabilities=None,
):
    """Prepare the noise and label distributions.

    Arguments
    ---------
        G_batch_size: int
            Batch size for the generator.
        dim_z: int
            Noise vector dimensionality.
        nclasses: int
            Number of classes in the dataset
        fp16: bool, optional
            Float16.
        z_var: float, optional
            Variance for the noise normal distribution.
        longtail_gen: bool, optional
            If true, use the longtail distribution for the classes (ImageNet-LT)
        custom_distrib: bool, optional
            If true, use a temperature annealed class distribution.
        longtail_temperature: float, optional
            Temperature value to smooth the longtail distribution with a softmax function.
        class_probabilities: list, optional
            A list of float values indicating the probability of a class in the dataset.

    Returns
    -------
       The noise and class distributions.
    """
    z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
    z_.init_distribution("normal", mean=0, var=z_var)
    #  z_ = z_.to(device, torch.float16 if fp16 else torch.float32)

    if fp16:
        z_ = z_.half()

    y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
    if longtail_gen:
        y_.init_distribution(
            "categorical_longtail",
            num_categories=nclasses,
            class_prob=class_probabilities,
        )
    elif custom_distrib:
        y_.init_distribution(
            "categorical_longtail_temperature",
            num_categories=nclasses,
            temperature=longtail_temperature,
            class_prob=class_probabilities,
        )
    else:
        y_.init_distribution("categorical", num_categories=nclasses)
    # y_ = y_.to(device, torch.int64)
    return z_, y_


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
        if self.dist_type == "normal":
            self.mean, self.var = kwargs["mean"], kwargs["var"]
        elif self.dist_type == "categorical":
            self.num_categories = kwargs["num_categories"]
        elif self.dist_type == "categorical_longtail":
            print("(class conditioning sampler) using longtail distribution")
            self.num_categories = kwargs["num_categories"]
            self.class_prob = torch.DoubleTensor(class_prob)
        elif self.dist_type == "categorical_longtail_temperature":
            print(
                "(class conditioning sampler) Softening the long-tail distribution with temperature ",
                kwargs["temperature"],
            )
            self.num_categories = kwargs["num_categories"]
            self.class_prob = torch.log(torch.DoubleTensor(class_prob))
            self.class_prob = torch.exp(
                self.class_prob / kwargs["temperature"]
            ) / torch.sum(torch.exp(self.class_prob / kwargs["temperature"]))

    def seed_generator(self, seed):
        self.generator.manual_seed(seed)

    def sample_(self):
        if self.dist_type == "normal":
            self.normal_(self.mean, self.var)
        elif self.dist_type == "categorical":
            self.random_(0, self.num_categories)
        elif (
            "categorical_longtail" in self.dist_type
            or "categorical_longtail_temperature" in self.dist_type
        ):
            self.data = torch.multinomial(
                self.class_prob, len(self), replacement=True
            ).to(self.device)
            # return self.variable

    # Silly hack: overwrite the to() method to wrap the new object
    # in a distribution as well
    # def to(self, *args, **kwargs):
    #     new_obj = Distribution(self)
    #     new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
    #     new_obj.data = super().to(*args, **kwargs)
    #     return new_obj


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() + worker_id
