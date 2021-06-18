''' Datasets
    This file contains definitions for our CIFAR, ImageFolder, and HDF5 datasets
'''
import os
import os.path
import sys
from PIL import Image
import numpy as np
from tqdm import tqdm, trange
import random
import sklearn.metrics

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url, check_integrity
import torch.utils.data as data
from torch.utils.data import DataLoader
import faiss
import time
         
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


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
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')


def accimage_loader(path):
  import accimage
  try:
    return accimage.Image(path)
  except IOError:
    # Potentially a decoding problem, fall back to PIL.Image
    return pil_loader(path)


def default_loader(path):
  from torchvision import get_image_backend
  if get_image_backend() == 'accimage':
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

  Args:
      root (string): Root directory path.
      transform (callable, optional): A function/transform that  takes in an PIL image
          and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the
          target and transforms it.
      loader (callable, optional): A function to load an image given its path.

   Attributes:
      classes (list): List of the class names.
      class_to_idx (dict): Dict with items (class_name, class_index).
      imgs (list): List of (image path, class_index) tuples
  """

  def __init__(self, root, transform=None, target_transform=None,
               loader=default_loader, load_in_mem=False, 
               index_filename='imagenet_imgs.npz', longtail=False,
               subsampled=False, split='train', **kwargs):

    classes, class_to_idx = find_classes(root)
    # Load pre-computed image directory walk
    if False: #os.path.exists(os.path.join(index_filename)):
      print('Loading pre-saved Index file %s...' % index_filename)
      imgs = np.load(os.path.join(index_filename))['imgs']
 #   If first time, walk the folder directory and save the
  #  results to a pre-computed file.
    else:
      print('Generating  Index file %s...' % index_filename)
      if not longtail:
        imgs = make_dataset(root, class_to_idx)
        if subsampled:
          # Same number of samples as in ImageNet-LT
          imgs = random.sample(imgs, 115846)
      else:
        imgs = []
        print('Using long-tail version of the dataset with split ', split ,'!')
        with open('BigGAN-PyTorch/imagenet_lt/ImageNet_LT_'+split+'.txt') as f:
          for line in f:
            imgs.append((os.path.join(root, '/'.join(line.split()[0].split('/')[1:])),int(line.split()[1])))
      np.savez_compressed(os.path.join(index_filename), **{'imgs' : imgs})
    if len(imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                           "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    self.root = root
    self.imgs = imgs
    self.classes = classes
    self.class_to_idx = class_to_idx
    self.transform = transform
    self.target_transform = target_transform
    self.loader = loader
    self.load_in_mem = load_in_mem
    
    if self.load_in_mem:
      print('Loading all images into memory...')
      self.data, self.labels = [], []
      for index in tqdm(range(len(self.imgs))):
        path, target = imgs[index][0], imgs[index][1]
        self.data.append(self.transform(self.loader(path)))
        self.labels.append(target)
          

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
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
    #print('image is ', img.shape, 'target is ', target, int(target))
    return img, int(target), index

  def __len__(self):
    return len(self.imgs)

  def __repr__(self):
    fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
    fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
    fmt_str += '    Root Location: {}\n'.format(self.root)
    tmp = '    Transforms (if any): '
    fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    tmp = '    Target Transforms (if any): '
    fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    return fmt_str
        

''' ILSVRC_HDF5: A dataset to support I/O from an HDF5 to avoid
    having to load individual images all the time. '''
import h5py as h5
import torch

class ILSVRC_HDF5_feats(data.Dataset):
  def __init__(self, root, root_feats=None, root_nns=None, transform=None,
               target_transform=None, load_labels=True, load_features=True,
               load_in_mem_images=False, load_in_mem_labels=False,
               load_in_mem_feats=False, k_nn=4, which_nn_balance='instance_balance',
               kmeans_file=None, n_subsampled_data=-1, filter_hd=-1,label_dim=0,
               feature_augmentation=False, gpu_knn=True,
               **kwargs):
    self.root = root
    self.root_feats = root_feats
    self.root_nns = root_nns

    self.load_labels = load_labels
    self._label_dim = label_dim
    self.load_features = load_features

    self.feature_augmentation = feature_augmentation

    # self.transform = transform
    self.target_transform = target_transform

    # Set the transform here
    self.transform = transform

    # load the entire dataset into memory?
    self.load_in_mem_images = load_in_mem_images
    self.load_in_mem_labels = load_in_mem_labels
    self.load_in_mem_feats = load_in_mem_feats

    self.which_nn_balance = which_nn_balance
    self.subsampled_data = False

    self.num_imgs = len(h5.File(root, 'r')['labels'])

    # Change the available instances to sample from. Simulating a
    # reduced storage setup where a subset of instances are selected by either
    # uniform sampling or k-means algorithm.

    self.labels, self.feats = None, None

    # self.samples_per_class = np.load('imagenet_lt_samples_per_class.npy',
    #                                  allow_pickle=True)
    #self.sample_conditionings = sample_conditionings
   # print('Sampling conditionings from dataset with dataloader? ', self.sample_conditionings)
  #  self.weights = weights_sampling
    # If loading into memory, do so now
    print('Load in mem? Images: %r, Labels: %r, Features: %r.'%
          (self.load_in_mem_images, self.load_in_mem_labels, self.load_in_mem_feats))
    if self.load_in_mem_images:
      print('Loading images from %s into memory...' % root)
      with h5.File(root, 'r') as f:
        self.data = f['imgs'][:]
    if load_labels and self.load_in_mem_labels:
      print('Loading labels from %s into memory...' % root)
      with h5.File(root, 'r') as f:
        self.labels = f['labels'][:]
    if load_features and self.load_in_mem_feats:
      print('Loading features from %s into memory...' % root_feats)
      with h5.File(root_feats, 'r') as f:
        self.feats = f['feats'][:]
      # Normalizing features
      print('Normalizing features by their norm')
      self.feats /= np.linalg.norm(self.feats,axis=1, keepdims=True)
      self.feats = torch.from_numpy(self.feats)
      self.feats.share_memory_()
      # We compute NNs only if we are loading features and there is no root_nns file.
    if load_features:
      if root_nns is None and self.load_in_mem_feats:
        # obtaining NN samples
        self.obtain_nns(k_nn,gpu=gpu_knn)
      elif root_nns is not None:
        # Still loading the NNs indexes!
        print('Loading %s into memory...' % root_nns)
        with h5.File(root_nns, 'r') as f:
          self.sample_nns = f['sample_nns'][:]
          self.sample_nn_radius = f['sample_nns_radius'][:]
      else:
        raise ValueError('If no file with pre-computed neighborhoods is provided, '
                         'the features need to be loaded in memory to extract them.'
                         ' Set the load_in_mem_feats=True.')



    self.possible_sampling_idxs = range(self.num_imgs)
    if kmeans_file is not None:
      print('Loading file  with just a few centroids (kmeans)... ', kmeans_file)
      self.kmeans_samples = np.load(kmeans_file,
              allow_pickle=True).item()['center_examples'][:, 0]
      self.possible_sampling_idxs = self.kmeans_samples
    elif n_subsampled_data>-1:
      self.possible_sampling_idxs = np.random.choice(np.array(self.possible_sampling_idxs),
                                            int(n_subsampled_data),
                                            replace=False)

    #TODO: clean this, and say it is ony for COCO-stuff
    elif filter_hd>-1:
        self.filter_by_hd('val2', filter_hd, '../')

    else:
      self.kmeans_samples = None

      print('All possible conditioning instances are ',
            len(self.possible_sampling_idxs))

  @property
  def resolution(self):
    with h5.File(self.root, 'r') as f:
      sze = list(f['imgs'][0].shape)
    return sze[1]

  @property
  def label_dim(self):
    return self._label_dim



  def filter_by_hd(self, split, ood_distance, ood_test_path):
    if split in ['val1', 'val2']:
      image_ids_original = np.load('cocostuff_val2_all_idxs.npy', allow_pickle=True)
      odd_image_ids = np.load(os.path.join(ood_test_path, 'stored_vars',
                                           split + '_image_ids_by_hd_75ktraining_im.npy'),
                              allow_pickle=True)  # _cleaned_vocab
      if ood_distance == 0:
        image_ids = odd_image_ids[ood_distance]
      else:
        total_img_ids = []
        for ood_dist in range(1, len(odd_image_ids)):
          total_img_ids += odd_image_ids[ood_dist]
        image_ids = total_img_ids
      print('OOD split with hamming distance ' + str(
        ood_distance) + ' wrt to the training set, has length ',
            len(image_ids))
      allowed_idxs = []

    #  new_sample_nns = []
      for i_idx, id in enumerate(image_ids_original):
        if id in image_ids:
          allowed_idxs.append(i_idx)
         # new_sample_nns.append(self.sample_nns[i_idx])
      allowed_idxs = np.array(allowed_idxs)
      self.possible_sampling_idxs = allowed_idxs

      # self.feats = self.feats[allowed_idxs]
      # self.sample_nns = new_sample_nns
      # self.labels = self.labels[allowed_idxs]
      # self.data = self.data[allowed_idxs]
      # self.num_imgs = len(self.labels)
      print('Num images new ', self.num_imgs)


  def sample_conditioning_instance_balance(self, batch_size, weights=None):
    """
    weights: sampling weights for each of the instances in the dataset.
    """
    # Control instance (center of k-NN) balancing with weights
    # Sampling from p(h)
    if weights is None and len(self.possible_sampling_idxs)==self.num_imgs:
      sel_idxs = np.random.randint(0, self.num_imgs,size=batch_size)
    else:
      sel_idxs = np.random.choice(self.possible_sampling_idxs, batch_size, replace=True, p=weights)

    #Features from center example
    instance_gen = self.get_instance_features(sel_idxs)
    # Get labels from neighbor
    instance_labels = []
    for idx_ in sel_idxs:
      # Sampling neighbor from p(x_nn, y_nn| h)
      chosen_idx = np.random.choice(self.sample_nns[idx_])
      # Labels from neighbors
      if self.load_labels:
        instance_labels.append(self.get_label(chosen_idx)[np.newaxis,...])
    if self.load_labels:
      instance_labels = np.concatenate(instance_labels, 0)
      instance_labels = torch.LongTensor(instance_labels)
    else:
      instance_labels = None

    instance_gen = torch.FloatTensor(instance_gen)

    return instance_labels, instance_gen

  def sample_conditioning_nnclass_balance(self, batch_size, weights=None, num_classes=1000):
    """
    weights: sampling weights for each of the classes in the dataset.
    """
    if weights is not None:
      weights = np.array(weights) / sum(weights)

    # Sampling from p(y)
    chosen_class = np.random.choice(num_classes, batch_size, replace=True, p=weights)
    nn_idxs = []
    for lab_ in chosen_class:
      # Sampling from p(x_nn|y)
      chosen_xnn = np.random.choice((self.labels == lab_).nonzero()[0])
      # Sampling from p(h| x_nn,y)
      nn_idxs.append(np.random.choice(self.sample_nns[chosen_xnn]))

    instance_gen = self.get_instance_features(nn_idxs)

    instance_gen =  torch.FloatTensor(instance_gen)
    labels_gen = torch.LongTensor(chosen_class)

    return labels_gen, instance_gen

  def obtain_nns(self, k_nn=20, faiss_lib=True, feat_sz=2048, gpu=True):
    print('using K=', k_nn)
    # K_nn computation takes into account the input sample as the first NN,
    # so we add an extra NN to later remove the input sample.
    k_nn+=1

    self.sample_nns = [[] for _ in range(self.num_imgs)]
    self.sample_nn_radius = np.zeros(self.num_imgs, dtype=float)

    if faiss_lib:
      ngpus = faiss.get_num_gpus()
      print("number of GPUs:", ngpus)
      cpu_index = faiss.IndexFlatL2(feat_sz)
      if gpu:
        gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
          cpu_index
        )
        index = gpu_index
      else:
        index = cpu_index
      index.add(self.feats.float().numpy().astype('float32'))
      kth_values, kth_values_arg =\
        index.search(self.feats.numpy().astype('float32'), k_nn)
      self.kth_values = np.sqrt(kth_values)
      knn_radii = np.sqrt(kth_values[:,-1])

    else:
      dists = sklearn.metrics.pairwise_distances(self.feats,self.feats,
                                                 metric='euclidean',n_jobs=-1)
      print('Computed distances.')
      knn_radii, kth_values_arg = self.get_kth_value_accurate(
          dists,k_nn)
    for i_sample in range(self.num_imgs):
      knns = kth_values_arg[i_sample]
      #Discarding the input sample, also seen as the 0-NN.
      knns = np.delete(knns, np.where(knns == i_sample)[0], 0)
      self.sample_nns[i_sample] = knns.tolist()
      self.sample_nn_radius[i_sample] = knn_radii[i_sample]
    print('Computed NNs.')

  def get_kth_value_accurate(self,unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k-1, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values, indices


  def get_image(self, index):
    if self.load_in_mem_images:
      img = self.data[index]
    else:
      with h5.File(self.root, 'r') as f:
        img = f['imgs'][index]
    return img

  def get_label(self, index):
    if self.load_labels:
      if self.load_in_mem_labels:
        target = self.labels[index]
      else:
        with h5.File(self.root, 'r') as f:
          target = f['labels'][index]
    else:
      target = None
    return target

  def get_instance_features(self, index):
    if self.load_in_mem_feats:
      feat = self.feats[index].clone().float()  # .astype('float')
    else:
      with h5.File(self.root_feats, 'r') as f:
        if isinstance(index,(int, np.int64)):
          hflip = np.random.randint(2) == 1
          if self.feature_augmentation and hflip:
            feat = f['feats_hflip'][index].astype('float')
          else:
            feat = f['feats'][index].astype('float')
          feat /= np.linalg.norm(feat, keepdims=True)
        else:
          feat = []
          for sl_idx in index:
            hflip = np.random.randint(2) == 1
            if self.feature_augmentation and hflip:
              feat.append(f['feats_hflip'][sl_idx].astype('float')[np.newaxis, ...])
            else:
              feat.append(f['feats'][sl_idx].astype('float')[np.newaxis, ...])
          feat = np.concatenate(feat)
          feat /= np.linalg.norm(feat, axis=1, keepdims=True)
    return feat

  def get_instance_features_and_nn(self, index):
    # Standard sampling: Obtain a feature vector for the input index,
    # and image/class label for a neighbor.
    if self.which_nn_balance == 'instance_balance':
      idx_h = index
      #If we are only using a selected number of instances (kmeans), re-choose the index
      if self.kmeans_samples is not None:
        index = np.random.choice(self.kmeans_samples)
      idx_nn = np.random.choice(self.sample_nns[index])

    # Reverse sampling, used when we want to perform class balancing (long-tail setup).
    # We use the input index as if it were the neigbhor,
    # and obtain its class and image, and then use the feature vector from a sampled neighbor index.
    # In class-conditional IC-GAN, the classes are taken from the neighbors.
    # The reverse sampling allows us to control the class balancing by using extra weights in the DataLoader.
    elif self.which_nn_balance == 'nnclass_balance':
      idx_h = np.random.choice(self.sample_nns[index])
      idx_nn = index
    else:
      raise ValueError('No other sampling method has been defined. '
                       'Choose which_nn_balance in [instance_balance,nnclass_balance].')

    # Index selects the instance feature vector
    radii = self.sample_nn_radius[idx_h]

    img_nn = self.get_image(idx_nn)
    label_nn = self.get_label(idx_nn)
    feats = self.get_instance_features(idx_h)

    return img_nn,label_nn,feats, radii



  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is class_index of the target class.
    """
    # If loaded the entire dataset in RAM, get image from memory

    img = self.get_image(index)
    target = self.get_label(index)
    if self.load_features:
      img_nn,label_nn,feats, radii = self.get_instance_features_and_nn(index)
      img = img_nn
      target = label_nn
    else:
      feats, radii = None, None

    # Apply transform
    img = ((torch.from_numpy(img).float() / 255) - 0.5) * 2
    if self.transform is not None:
      img = self.transform(img)
    if self.target_transform is not None:
      target = self.target_transform(target)

    if self.load_features and self.load_labels:
      return img, int(target), feats, radii
    elif self.load_features:
      return img, feats, radii
    elif self.load_labels:
      return img, int(target)
    else:
      return img

  def __len__(self):
    return self.num_imgs
    # return len(self.f['imgs'])