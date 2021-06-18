## Modified from grid2im code

import json
import os
from collections import defaultdict
import torchvision.transforms as T
from torch.utils.data import Dataset
import PIL
import numpy as np

PREDICATES_VALUES = ['left of', 'right of', 'above', 'below', 'inside',
                     'surrounding']
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

def imagenet_preprocess():
    return T.Normalize(mean=MEAN, std=STD)


class Resize(object):
    def __init__(self, size, interp=PIL.Image.BILINEAR):
        if isinstance(size, tuple):
            H, W = size
            self.size = (W, H)
        else:
            self.size = (size, size)
        self.interp = interp

    def __call__(self, img):
        return img.resize(self.size, self.interp)


class CocoStuff(Dataset):
    def __init__(self, image_dir, instances_json, stuff_json=None,
                 stuff_only=True, image_size=64,
                 normalize_images=True, max_samples=None, min_object_size=0.02,
                 min_objects_per_image=3, max_objects_per_image=8,
                 instance_whitelist=None,
                 stuff_whitelist=None, no__img__=False,
                 test_part=False, split='train', iscrowd=True, mode='train',
                 transform=None, **kwargs):
        """
        A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
        them to scene graphs on the fly.

        Inputs:
        - image_dir: Path to a directory where images are held
        - instances_json: Path to a JSON file giving COCO annotations
        - stuff_json: (optional) Path to a JSON file giving COCO-Stuff annotations
        - stuff_only: (optional, default True) If True then only iterate over
          images which appear in stuff_json; if False then iterate over all images
          in instances_json.
        - image_size: Size (H, W) at which to load images. Default (64, 64).
        - mask_size: Size M for object segmentation masks; default 16.
        - normalize_image: If True then normalize images by subtracting ImageNet
          mean pixel and dividing by ImageNet std pixel.
        - max_samples: If None use all images. Other wise only use images in the
          range [0, max_samples). Default None.
        - min_object_size: Ignore objects whose bounding box takes up less than
          this fraction of the image.
        - min_objects_per_image: Ignore images which have fewer than this many
          object annotations.
        - max_objects_per_image: Ignore images which have more than this many
          object annotations.
        - include_other: If True, include COCO-Stuff annotations which have category
          "other". Default is False, because I found that these were really noisy
          and pretty much impossible for the system to model.
        - instance_whitelist: None means use all instance categories. Otherwise a
          list giving a whitelist of instance category names to use.
        - stuff_whitelist: None means use all stuff categories. Otherwise a list
          giving a whitelist of stuff category names to use.
        """
        super(Dataset, self).__init__()
        if stuff_only and stuff_json is None:
            print('WARNING: Got stuff_only=True but stuff_json=None.')
            print('Falling back to stuff_only=False.')

        self.image_dir = image_dir
        self.max_samples = max_samples
        self.normalize_images = normalize_images
        self.iscrowd = iscrowd
      #  self.transform = transform

        self.left_right_flip = False #True if split == 'train' else False
        self.max_objects_per_image = max_objects_per_image
        self.mode = mode

        if image_size is not None:
            self.set_image_size(image_size)
        print(self.transform)
        self.no__img__ = no__img__

        with open(instances_json, 'r') as f:
            instances_data = json.load(f)

        self.image_id_to_sentences = {}
        stuff_data = None
        if stuff_json is not None and stuff_json != '':
            with open(stuff_json, 'r') as f:
                stuff_data = json.load(f)

        self.image_ids = []
        self.image_id_to_filename = {}
        self.image_id_to_size = {}
        for image_data in instances_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']
            width = image_data['width']
            height = image_data['height']
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename
            self.image_id_to_size[image_id] = (width, height)

        object_idx_to_name = {}
        # Get categories names and ids
        all_instance_categories = self.populate_categories(instances_data,
                                                           object_idx_to_name)
        all_stuff_categories = self.populate_categories(stuff_data,
                                                        object_idx_to_name)

        if instance_whitelist is None:
            instance_whitelist = all_instance_categories
        if stuff_whitelist is None:
            stuff_whitelist = all_stuff_categories

        category_whitelist = set(instance_whitelist) | set(stuff_whitelist)

        # Add object data from instances and stuff
        self.image_id_to_objects = defaultdict(list)
        self.add_object_instances(instances_data, min_object_size,
                                  object_idx_to_name, category_whitelist)
        image_ids_with_stuff = self.add_object_instances(stuff_data,
                                                         min_object_size,
                                                         object_idx_to_name,
                                                         category_whitelist)
        if stuff_only:
            new_image_ids = []
            for image_id in self.image_ids:
                if image_id in image_ids_with_stuff:
                    new_image_ids.append(image_id)
            self.image_ids = new_image_ids

            all_image_ids = set(self.image_id_to_filename.keys())
            image_ids_to_remove = all_image_ids - image_ids_with_stuff
            for image_id in image_ids_to_remove:
                self.image_id_to_filename.pop(image_id, None)
                self.image_id_to_size.pop(image_id, None)
                self.image_id_to_objects.pop(image_id, None)


        # Prune images that have too few or too many objects
        new_image_ids = []
        total_objs = 0
        for image_id in self.image_ids:
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
            if min_objects_per_image <= num_objs <= max_objects_per_image:
                new_image_ids.append(image_id)
        self.image_ids = new_image_ids

        print('TOtal number of images is ', len(self.image_ids))
        # np.save('imgs_ids_val2_cocostuff_deprecated_filtered', self.image_ids)
        # exit()
        if split == 'val':
            if test_part:
                self.image_ids = self.image_ids[1024:]
                print('Len images ', len(self.image_ids))
                np.save('cocostuff_val2_all_idxs', self.image_ids)
             #   exit()
            else:
                print('Entering in val part')
                self.image_ids = self.image_ids[:1024]

    def populate_categories(self, data, object_idx_to_name):
        all_categories = []
        for category_data in data['categories']:
            category_id = category_data['id']
            category_name = category_data['name']
            all_categories.append(category_name)
            object_idx_to_name[category_id] = category_name
        return all_categories

    def add_object_instances(self, data, min_object_size, object_idx_to_name,
                             category_whitelist):
        image_ids_present = set()
        for object_data in data['annotations']:
            image_id = object_data['image_id']
            _, _, w, h = object_data['bbox']
            image_ids_present.add(image_id)
            W, H = self.image_id_to_size[image_id]
            box_area = (w * h) / (W * H)
            box_ok = box_area > min_object_size
            object_name = object_idx_to_name[object_data['category_id']]
            category_ok = object_name in category_whitelist
            other_ok = object_name != 'other'

            condition = box_ok and category_ok and other_ok
            if self.iscrowd:
                condition = condition and (object_data['iscrowd'] != 1)
            if condition:
                self.image_id_to_objects[image_id].append(object_data)
        return image_ids_present

    def set_image_size(self, image_size):
        print('called set_image_size', image_size)
        transform = [Resize(image_size), T.ToTensor()]
        if self.normalize_images:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)
        self.image_size = image_size

    def total_objects(self):
        total_objs = 0
        for i, image_id in enumerate(self.image_ids):
            if self.max_samples and i >= self.max_samples:
                break
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
        return total_objs


    def __len__(self):
        if self.max_samples is None:
            if self.left_right_flip:
                return len(self.image_ids) * 2
            return len(self.image_ids)
        return min(len(self.image_ids), self.max_samples)

    def __getitem__(self, index):
        """
        Get the pixels of an image, and a random synthetic scene graph for that
        image constructed on-the-fly from its COCO object annotations. We assume
        that the image will have height H, width W, C channels; there will be O
        object annotations, each of which will have both a bounding box and a
        segmentation mask of shape (M, M). There will be T triples in the scene
        graph.

        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system
        - masks: LongTensor of shape (O, M, M) giving segmentation masks for
          objects, where 0 is background and 1 is object.
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.
        """
        flip = False
        if self.mode == 'train':
            if index >= len(self.image_ids):
                index = index - len(self.image_ids)
                flip = True

        image_id = self.image_ids[index]

        filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, filename)
        with open(image_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                if flip and self.mode == 'train':
                    image = PIL.ImageOps.mirror(image)
                image = self.transform(image.convert('RGB'))

        return image, int(0), index

