# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Store JPEG images from a hdf5 file, in order to compute FID scores (COCO-Stuff)."""
from argparse import ArgumentParser
import numpy as np
import os
import h5py as h5
from imageio import imwrite as imsave

import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from data_utils.utils import filter_by_hd


def main(args):
    dataset_name_prefix = "COCO"
    test_part = True if args["split"] == "val" else False

    # HDF5 file name
    hdf5_filename = "%s%i%s%s" % (
        dataset_name_prefix,
        args["resolution"],
        "_val" if args["split"] == "val" else "",
        "_test" if test_part else "",
    )
    data_path_xy = os.path.join(args["data_root"], hdf5_filename + "_xy.hdf5")
    # Load data
    print("Loading images %s..." % (data_path_xy))
    with h5.File(data_path_xy, "r") as f:
        imgs = f["imgs"][:]

    # Filter images
    if args["filter_hd"] > -1:
        filtered_idxs = filter_by_hd(args["filter_hd"])
    else:
        filtered_idxs = range(len(imgs))

    # Save images
    counter_i = 0
    for i, im in enumerate(imgs):
        if i in filtered_idxs:
            imsave(
                "%s/%06d.%s" % (args["out_path"], counter_i, "jpg"),
                im.transpose(1, 2, 0),
            )
            counter_i += 1


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Storing ground-truth COCO-Stuff images to compute FID metric."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="Data resolution (default: %(default)s)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Data split (default: %(default)s)",
    )
    parser.add_argument(
        "--filter_hd",
        type=int,
        default=-1,
        help="Hamming distance to filter val test in COCO_Stuff (by default no filtering) (default: %(default)s)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Default location where the hdf5 file is stored (default: %(default)s)",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data",
        help="Default location where the resulting images will be stored (default: %(default)s)",
    )

    args = vars(parser.parse_args())
    main(args)
