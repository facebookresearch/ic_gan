# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# All contributions by Andy Brock:
# Copyright (c) 2019 Andy Brock
#
# All contributions made by NAVER Corp.:
# Copyright (c) 2020-present NAVER Corp.
#
# MIT license

import sys
import os
import os.path

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from data_utils import utils as data_utils
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import sklearn.metrics
import torch.utils.data as data
try:
    import faiss
    USE_FAISS = 1
except:
    print('Faiss library not found!')
    USE_FAISS = 0
import h5py as h5
import torch

IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm"]


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in tqdm(sorted(os.listdir(dir))):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

      root/dogball/xxx.png
      root/dogball/xxy.png
      root/dogball/xxz.png

      root/cat/123.png
      root/cat/nsdf3.png
      root/cat/asd932_.png

  Parameters
  ----------
      root: string. Root directory path.
      transform: callable, optional. A function/transform that  takes in an PIL image
          and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform: callable, optional. A function/transform that takes in the
          target and transforms it.
      loader: callable, optional. A function to load an image given its path.

   Attributes
   ----------
      classes: list. List of the class names.
      class_to_idx: dict. Dict with items (class_name, class_index).
      imgs: list. List of (image path, class_index) tuples
  """

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=default_loader,
        load_in_mem=False,
        index_filename="imagenet_imgs.npz",
        longtail=False,
        subsampled=False,
        split="train",
        **kwargs
    ):

        classes, class_to_idx = find_classes(root)
        # Load pre-computed image directory walk
        if False:  # os.path.exists(os.path.join(index_filename)):
            print("Loading pre-saved Index file %s..." % index_filename)
            imgs = np.load(os.path.join(index_filename))["imgs"]
        #   If first time, walk the folder directory and save the
        #  results to a pre-computed file.
        else:
            print("Generating  Index file %s..." % index_filename)
            if not longtail:
                imgs = make_dataset(root, class_to_idx)
                if subsampled:
                    # Same number of samples as in ImageNet-LT
                    imgs = random.sample(imgs, 115846)
            else:
                imgs = []
                print("Using long-tail version of the dataset with split ", split, "!")
                with open(
                    "BigGAN_PyTorch/imagenet_lt/ImageNet_LT_" + split + ".txt"
                ) as f:
                    for line in f:
                        imgs.append(
                            (
                                os.path.join(
                                    root, "/".join(line.split()[0].split("/")[1:])
                                ),
                                int(line.split()[1]),
                            )
                        )
            np.savez_compressed(os.path.join(index_filename), **{"imgs": imgs})
        if len(imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 images in subfolders of: " + root + "\n"
                    "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)
                )
            )

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.load_in_mem = load_in_mem

        if self.load_in_mem:
            print("Loading all images into memory...")
            self.data, self.labels = [], []
            for index in tqdm(range(len(self.imgs))):
                path, target = imgs[index][0], imgs[index][1]
                self.data.append(self.transform(self.loader(path)))
                self.labels.append(target)

    def __getitem__(self, index):
        """
    Parameters
    ----------
        index: int. Index

    Returns
    -------
        tuple: (image, target) where target is class_index of the target class.
    """
        if self.load_in_mem:
            img = self.data[index]
            target = self.labels[index]
        else:
            path, target = self.imgs[index]
            img = self.loader(str(path))
            if self.transform is not None:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, int(target), index

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


class ILSVRC_HDF5_feats(data.Dataset):
    """ ILSVRC_HDF5_feats: A dataset to support I/O from an HDF5.

        Parameters
        ----------
            root :str
                Path to the hdf5 file containing images and labels.
            root_feats: str, optional
                Path to the hdf5 file containing the instance features.
            root_nns: str, optional
                Path to the hdf5 file containing the list of nearest neighbors for each instance.
            transform : callable, optional
                A function/transform that  takes in an PIL image and returns a transformed version.
                 E.g, ``transforms.RandomCrop``
            target_transform: callable, optional
                A function/transform that takes in the target and transforms it.
            load_labels: bool, optional
                Return labels for each example.
            load_features: bool, optional
                Return instance features and its neighbors (needed for IC-GAN).
            load_in_mem_images: bool, optional
                Load all images in memory.
            load_in_mem_labels: bool, optional
                Load all labels in memory.
            load_in_mem_feats: bool, optional
                Load all instance features in memory.
            k_nn: int, optional
                Size of the neighborhood obtained with the k-NN algorithm.
            which_nn_balance: str, optional
                Whether to sample an instance or a neighbor class first. By default,
                ``instance_balance`` is used.
                 Using ``nnclass_balance`` allows class balancing to be applied.
            kmeans_file: str, optional
                Path to a file where only the dataset indexes selected with k-means are stored.
                 It reduces the amount of available data to train or test the model.
            n_subsampled_data: int, optional
                If other than -1, that number of data points are randomly selected from the dataset.
                It reduces the amount of available data to train or test the model.
            filter_hd: int, optional
                Only used for COCO-Stuff dataset. If -1, all COCO-Stuff evaluation set is used.
                 If 0, only images with seen class combinations are used.
                 If 1, only images with unseen class combinations are used.
            label_dim: int, optional
                Dimensionality of label embeddings. Useful for the StyleGAN2 backbone code.
            feature_dim: int, optional
                Dimensionality of instance features embeddings. Useful for the StyleGAN2 backbone
                 code.
            feature_augmentation: bool, optional
                Use the instance features of the flipped ground-truth image instances as
                 conditioning, with a 50% probability.
            gpu_knn: bool, optional
                Accelerate k-NN faiss computation with GPUs.
            apply_norm: bool, optional
                Normalize images between [-0.5, 0.5].
            label_onehot: bool, optional
                Return labels as a one hot encoding. Useful for StyleGAN2 backbone code.

        Attributes
        ---------
            root: str
                Path to the hdf5 file containing images and labels.
            root_feats: str
                Path to the hdf5 file containing the instance features.
            root_nns: str
                Path to the hdf5 file containing the list of nearest neighbors for each
                instance.
            transform : callable
                A function/transform that  takes in an PIL image and returns a transformed version.
                E.g, ``transforms.RandomCrop``
            target_transform: callable
                A function/transform that takes in the target and transforms it.
            load_labels: bool
                Return labels for each example.
            load_features: bool
                Return instance features and its neighbors (needed for IC-GAN).
            load_in_mem_images: bool
                Load all images in memory.
            load_in_mem_labels: bool
                Load all labels in memory.
            load_in_mem_feats: bool
                Load all instance features in memory.
            feature_augmentation: bool
                Use the instance features of the flipped ground-truth image instances as conditioning,
                with a 50% probability.
            which_nn_balance: str
                Whether to sample an instance or a neighbor class first. By default,
                ``instance_balance`` is used. Using ``nnclass_balance`` allows class balancing to be
                 applied.
            apply_norm: bool
                Normalize images between [-0.5, 0.5].
            label_onehot: bool
                Return labels as a one hot encoding. Useful for StyleGAN2 backbone code.
            num_imgs: int.
                Number of data points in the dataset.
            data: NumPy array
                Image data, with the shape (num_imgs, w, h, 3), where w: width and h: height.
            labels: NumPy array
                Label data, with the shape (num_imgs, 1).
            feats: NumPy array
                Instance features data, with the shape (num_imgs, 2048).
            sample_nns: list
                List with length ``num_imgs``, that contains a list of the ``k_nn`` neighbor indexes
                 for each instance.
            sample_nn_radius: NumPy array
                Array of size (num_imgs) that stores the distance between each instance and its
                farthest(k-th) neighbor.
            possible_sampling_idxs: list
                List of all effective possible data samples. By default, it is a range(0, num_imgs).
            kmeans_samples: list
                List of indexes for samples selected with k-means algorithm.
            kth_values: NumPy array
                Distances between instances and its k-th neighbor.
        """

    def __init__(
        self,
        root,
        root_feats=None,
        root_nns=None,
        transform=None,
        target_transform=None,
        load_labels=True,
        load_features=True,
        load_in_mem_images=False,
        load_in_mem_labels=False,
        load_in_mem_feats=False,
        k_nn=4,
        which_nn_balance="instance_balance",
        kmeans_file=None,
        n_subsampled_data=-1,
        filter_hd=-1,
        label_dim=0,
        feature_dim=2048,
        feature_augmentation=False,
        gpu_knn=True,
        apply_norm=True,
        label_onehot=False,
        **kwargs
    ):
        self.root = root
        self.root_feats = root_feats
        self.root_nns = root_nns

        self.load_labels = load_labels
        self.load_features = load_features
        self._label_dim = label_dim
        self._feature_dim = feature_dim
        self.label_onehot = label_onehot
        self.feature_augmentation = feature_augmentation

        # Set the transform here
        self.transform = transform
        self.target_transform = target_transform
        # Normalization of images between -0.5 and 0.5 used in BigGAN
        self.apply_norm = apply_norm

        # load the entire dataset into memory?
        self.load_in_mem_images = load_in_mem_images
        self.load_in_mem_labels = load_in_mem_labels
        self.load_in_mem_feats = load_in_mem_feats

        self.which_nn_balance = which_nn_balance

        self.num_imgs = len(h5.File(root, "r")["labels"])

        self.labels, self.feats = None, None
        self.kth_values = None
        # If loading into memory, do so now
        print(
            "Load in mem? Images: %r, Labels: %r, Features: %r."
            % (self.load_in_mem_images, self.load_in_mem_labels, self.load_in_mem_feats)
        )
        if self.load_in_mem_images:
            print("Loading images from %s into memory..." % root)
            with h5.File(root, "r") as f:
                self.data = f["imgs"][:]
        if load_labels and self.load_in_mem_labels:
            print("Loading labels from %s into memory..." % root)
            with h5.File(root, "r") as f:
                self.labels = f["labels"][:]
        if load_features and self.load_in_mem_feats:
            print("Loading features from %s into memory..." % root_feats)
            with h5.File(root_feats, "r") as f:
                self.feats = f["feats"][:]
            # Normalizing features
            print("Normalizing features by their norm")
            self.feats /= np.linalg.norm(self.feats, axis=1, keepdims=True)
            self.feats = torch.from_numpy(self.feats)
            self.feats.share_memory_()

        if load_features:
            if root_nns is None and self.load_in_mem_feats:
                # We compute NNs only if we are loading features and there is no root_nns file.
                self._obtain_nns(k_nn, gpu=gpu_knn, faiss_lib=USE_FAISS)
            elif root_nns is not None:
                # Still loading the NNs indexes!
                print("Loading %s into memory..." % root_nns)
                with h5.File(root_nns, "r") as f:
                    self.sample_nns = f["sample_nns"][:]
                    self.sample_nn_radius = f["sample_nns_radius"][:]
            else:
                raise ValueError(
                    "If no file with pre-computed neighborhoods is provided, "
                    "the features need to be loaded in memory to extract them."
                    " Set the load_in_mem_feats=True."
                )

        # Reducing the number of available samples according to different criteria
        self.possible_sampling_idxs = range(self.num_imgs)
        self.kmeans_samples = None
        if kmeans_file is not None:
            print("Loading file  with just a few centroids (kmeans)... ", kmeans_file)
            self.kmeans_samples = np.load(kmeans_file, allow_pickle=True).item()[
                "center_examples"
            ][:, 0]
            self.possible_sampling_idxs = self.kmeans_samples
        elif n_subsampled_data > -1:
            self.possible_sampling_idxs = np.random.choice(
                np.array(self.possible_sampling_idxs),
                int(n_subsampled_data),
                replace=False,
            )
        elif filter_hd > -1:
            # For COCO_Stuff, we can divide the evaluation set in seen class combinations
            # (filter_hd=0)
            # or unseen class combinations (filter_hd=1)
            allowed_idxs = data_utils.filter_by_hd(filter_hd)
            self.possible_sampling_idxs = allowed_idxs
        # Change the size of the dataset if only a subset of the data is used
        self.possible_sampling_idxs = np.array(self.possible_sampling_idxs)
        self.num_imgs = len(self.possible_sampling_idxs)

        print(
            "All possible conditioning instances are ", len(self.possible_sampling_idxs)
        )

    def __getitem__(self, index):
        """
        Parameters
        ----------
            index: int

        Returns
        -------
            If the dataset loads both features and labels, return 4 elements: neighbor image,
             neighbor class label, instance features and instance radius
            If the dataset loads only features (no labels), return 4 elements: neighbor image,
             instance features, instance radius
            If the dataset loads ony labels (no features), return 2 elements: neighbor image and
            neighbor class label.
            If the dataset does not load features nor labels, return only an image.
        """
        # This only changes the index if possible_sampling_idx contains only a subset of the data
        # (k-means/random sampling or evaluation sets in COCO-Stuff)
        index = self.possible_sampling_idxs[index]
        img = self._get_image(index)
        target = self.get_label(index)
        if self.load_features:
            img_nn, label_nn, feats, radii = self._get_instance_features_and_nn(index)
            img = img_nn
            target = label_nn
        else:
            feats, radii = None, None

        # Apply transform
        img = torch.from_numpy(img)
        if self.apply_norm:
            img = ((img.float() / 255) - 0.5) * 2
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.label_onehot:
            target = int(target)

        if self.load_features and self.load_labels:
            return img, target, feats, radii
        elif self.load_features:
            return img, feats, radii
        elif self.load_labels:
            return img, target
        else:
            return img

    def sample_conditioning_instance_balance(self, batch_size, weights=None):
        """
        It samples a batch size of conditionings.

        First, by first sampling an instance, and then one of the neighbor's class.

        Parameters
        ----------
        batch_size: int
            Number of conditioning to sample.
        weights: NumPy array, optional
            Array of size len(self.possible_sampling_idxs), indicating the weight for each instance,
             used for sampling.

        Returns
        -------
        labels_gen: torch.LongTensor
            Tensor of shape (batch_size, label_dim). Batch of neighbor labels.
        instance_gen: torch.LongTensor
            Tensor of shape (batch_size, label_dim). Batch of instance features.
        """
        # Control instance (center of k-NN) balancing with weights
        # Sampling from p(h)
        if weights is None:
            # np.random.randint is a faster function than np.random.choice.
            # If there is no sampling weights, use this one.
            sel_idxs = np.random.randint(0, len(self.possible_sampling_idxs), size=batch_size)
            sel_idxs = self.possible_sampling_idxs[sel_idxs]
        else:
            sel_idxs = np.random.choice(
                self.possible_sampling_idxs, batch_size, replace=True, p=weights
            )

        # Features from center example
        instance_gen = self.get_instance_features(sel_idxs)
        # Get labels from neighbor
        labels_gen = []
        for idx_ in sel_idxs:
            # Sampling neighbor from p(x_nn, y_nn| h)
            chosen_idx = np.random.choice(self.sample_nns[idx_])
            # Labels from neighbors
            if self.load_labels:
                labels_gen.append(self.get_label(chosen_idx)[np.newaxis, ...])
        if self.load_labels:
            labels_gen = np.concatenate(labels_gen, 0)
            labels_gen = torch.LongTensor(labels_gen)
        else:
            labels_gen = None

        instance_gen = torch.FloatTensor(instance_gen)

        return labels_gen, instance_gen

    def sample_conditioning_nnclass_balance(
        self, batch_size, weights=None, num_classes=1000
    ):
        """
        It samples a batch size of conditionings.

        First, by sampling a class, then an image from this class, and finally an instance feature
        that would have this image as a neighbor in feature space.

        Parameters
        ----------
        batch_size: int
            Number of conditioning to sample.
        weights: NumPy array, optional
            Array of size num_classes, indicating the weight for each instance, used for sampling.
        num_classes: int, optional
            Number of classes in the dataset

        Returns
        -------
        labels_gen: torch.LongTensor
            Tensor of shape (batch_size, label_dim). Batch of neighbor labels.
        instance_gen: torch.LongTensor
            Tensor of shape (batch_size, label_dim). Batch of instance features.
        """
        if weights is not None:
            weights = np.array(weights) / sum(weights)

        # Sampling from p(y)
        chosen_class = np.random.choice(
            range(num_classes), batch_size, replace=True, p=weights
        )
        nn_idxs = []
        for lab_ in chosen_class:
            # Sampling from p(x_nn|y)
            chosen_xnn = np.random.choice((self.labels == lab_).nonzero()[0])
            # Sampling from p(h| x_nn,y)
            nn_idxs.append(np.random.choice(self.sample_nns[chosen_xnn]))

        instance_gen = self.get_instance_features(nn_idxs)

        instance_gen = torch.FloatTensor(instance_gen)
        labels_gen = torch.LongTensor(chosen_class)

        return labels_gen, instance_gen

    def get_label(self, index):
        """Obtain a label as an int or as a one-hot vector."""
        if not self.load_labels:
            if self.label_onehot:
                return np.zeros(self.label_dim, dtype=np.float32).copy()
            else:
                return 0

        if self.load_labels:
            if self.load_in_mem_labels:
                target = self.labels[index]
            else:
                with h5.File(self.root, "r") as f:
                    target = f["labels"][index]
        else:
            target = None
        if self.label_onehot:
            onehot_vec = np.zeros(self.label_dim, dtype=np.float32)
            onehot_vec[target] = 1
            target = onehot_vec.copy()

        return target

    def get_instance_features(self, index):
        """Obtain an instance feature vector."""
        if not self.load_features:
            return np.zeros(self.feature_dim, dtype=np.float32).copy()

        if self.load_in_mem_feats:
            feat = self.feats[index].clone().float()  # .astype('float')
        else:
            with h5.File(self.root_feats, "r") as f:
                if isinstance(index, (int, np.int64)):
                    hflip = np.random.randint(2) == 1
                    if self.feature_augmentation and hflip:
                        feat = f["feats_hflip"][index].astype("float")
                    else:
                        feat = f["feats"][index].astype("float")
                    feat /= np.linalg.norm(feat, keepdims=True)
                else:
                    feat = []
                    for sl_idx in index:
                        hflip = np.random.randint(2) == 1
                        if self.feature_augmentation and hflip:
                            feat.append(
                                f["feats_hflip"][sl_idx].astype("float")[
                                    np.newaxis, ...
                                ]
                            )
                        else:
                            feat.append(
                                f["feats"][sl_idx].astype("float")[np.newaxis, ...]
                            )
                    feat = np.concatenate(feat)
                    feat /= np.linalg.norm(feat, axis=1, keepdims=True)
        return feat

    @property
    def resolution(self):
        with h5.File(self.root, "r") as f:
            sze = list(f["imgs"][0].shape)
        return sze[1]

    @property
    def label_dim(self):
        return self._label_dim

    @property
    def feature_dim(self):
        return self._feature_dim

    def _obtain_nns(self, k_nn=20, faiss_lib=True, feat_sz=2048, gpu=True):
        """
        It obtains the neighborhoods for all instances using the k-NN algorithm.

        Parameters
        ----------
        k_nn: int, optional
            Number of neighbors (k).
        faiss_lib: bool, optional
           If True, use the faiss library implementation of k-NN. If not, use the slower
            implementation of sklearn.
        feat_sz: int, optional
            Feature dimensionality.
        gpu: bool, optional
            If True, leverage GPU resources to speed up computation with the faiss library.

        """
        # K_nn computation takes into account the input sample as the first NN,
        # so we add an extra NN to later remove the input sample.
        k_nn += 1

        self.sample_nns = [[] for _ in range(self.num_imgs)]
        self.sample_nn_radius = np.zeros(self.num_imgs, dtype=float)

        if faiss_lib:
            cpu_index = faiss.IndexFlatL2(feat_sz)
            if gpu:
                gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)  # build the index
                index = gpu_index
            else:
                index = cpu_index
            index.add(self.feats.float().numpy().astype("float32"))
            kth_values, kth_values_arg = index.search(
                self.feats.numpy().astype("float32"), k_nn
            )
            self.kth_values = np.sqrt(kth_values)
            knn_radii = np.sqrt(kth_values[:, -1])

        else:
            dists = sklearn.metrics.pairwise_distances(
                self.feats, self.feats, metric="euclidean", n_jobs=-1
            )
            print("Computed distances.")
            knn_radii, kth_values_arg = self._get_kth_value_accurate(dists, k_nn)
        for i_sample in range(self.num_imgs):
            knns = kth_values_arg[i_sample]
            # Discarding the input sample, also seen as the 0-NN.
            knns = np.delete(knns, np.where(knns == i_sample)[0], 0)
            self.sample_nns[i_sample] = knns.tolist()
            self.sample_nn_radius[i_sample] = knn_radii[i_sample]
        print("Computed NNs.")

    @staticmethod
    def _get_kth_value_accurate(distances, k, axis=-1):
        """ Find k nearest neighbor
        Parameters
        ---------
        distances: NumPy array
            Matrix of size (M, M) of unordered distances.
        k: int
            Neighborhood size
        axis: int

        Returns
        -------
        kth values: NumPy array
            Distances of the k-th nearest neighbor along the designated axis.
        indices: NumPy array
            Array positions in the input matrix indicating all neighbors up until the k-th.

        """
        indices = np.argpartition(distances, k - 1, axis=axis)[..., :k]
        k_smallests = np.take_along_axis(distances, indices, axis=axis)
        kth_values = k_smallests.max(axis=axis)
        return kth_values, indices

    def _get_image(self, index):
        """Obtain an image array."""
        if self.load_in_mem_images:
            img = self.data[index]
        else:
            with h5.File(self.root, "r") as f:
                img = f["imgs"][index]
        return img

    def _get_instance_features_and_nn(self, index):
        """ Builds a quadruplet of neighbor image, its label, conditioning instance features, radii.

        Returns
        ----------
        img_nn: NumPy array
            Neighbor image.
        label_nn: NumPy array
            Neighbor label.
        feats: NumPy array
            Conditioning instance features.
        radii: float
            Distance between conditioning instance and farthest (k-th) neighbor.
        """
        # Standard sampling: Obtain a feature vector for the input index,
        # and image/class label for a neighbor.
        if self.which_nn_balance == "instance_balance":
            idx_h = index
            # If we are only using a selected number of instances (kmeans), re-choose the index
            if self.kmeans_samples is not None:
                index = np.random.choice(self.kmeans_samples)
            idx_nn = np.random.choice(self.sample_nns[index])

        # Reverse sampling, used when we want to perform class balancing (long-tail setup).
        # In class-conditional IC-GAN, the classes are taken from the neighbors.
        # The reverse sampling allows us to control the class balancing by using extra weights
        # in the DataLoader.
        elif self.which_nn_balance == "nnclass_balance":
            idx_h = np.random.choice(self.sample_nns[index])
            idx_nn = index

        # Index selects the instance feature vector
        radii = self.sample_nn_radius[idx_h]

        img_nn = self._get_image(idx_nn)
        label_nn = self.get_label(idx_nn)
        feats = self.get_instance_features(idx_h)

        return img_nn, label_nn, feats, radii

    def __len__(self):
        return self.num_imgs
