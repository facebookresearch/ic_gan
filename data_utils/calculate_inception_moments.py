# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# All contributions by Andy Brock:
# Copyright (c) 2019 Andy Brock
#
# MIT License
""" Calculate Inception Moments
 This script iterates over the dataset and calculates the moments of the 
 activations of the Inception net (needed for FID), and also returns
 the Inception Score of the training data.
 
 Note that if you don't shuffle the data, the IS of true data will be under-
 estimated as it is label-ordered. By default, the data is not shuffled
 so as to reduce non-determinism. """
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import data_utils.utils as utils
import data_utils.inception_utils as inception_utils
from tqdm import tqdm
from argparse import ArgumentParser


def prepare_parser():
    usage = "Calculate and store inception metrics."
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="Which Dataset resolution, out of 64, 128, 256" " (default: %(default)s)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Which Dataset to convert: train, val (default: %(default)s)",
    )
    parser.add_argument(
        "--stratified_moments",
        action="store_true",
        default=False,
        help="Compute moments for FID computation stratifying by many, medium and few-shot classes"
        " (ImageNet-LT)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Default location where data is stored and where dataset hdf5 is found"
        " (default: %(default)s)",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data",
        help="Default location where data in hdf5 format will be stored (default: %(default)s)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Default overall batchsize (default: %(default)s)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Use multiple GPUs (default: %(default)s)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of dataloader workers (default: %(default)s)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
        help="Shuffle the data? (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed to use.")
    parser.add_argument(
        "--load_in_mem",
        action="store_true",
        default=False,
        help="Load all data into memory? (default: %(default)s)",
    )

    parser.add_argument(
        "--which_dataset",
        type=str,
        default="imagenet",
        choices=["imagenet", "imagenet_lt", "coco"],
        help="Dataset choice.",
    )

    return parser


def run(config):
    # Get dataset and loader
    kwargs = {
        "num_workers": config["num_workers"],
        "pin_memory": False,
        "drop_last": False,
        "load_in_mem": config["load_in_mem"],
    }
    if config["which_dataset"] in ["imagenet", "imagenet_lt"]:
        dataset_name_prefix = "ILSVRC"
    elif config["which_dataset"] == "coco":
        dataset_name_prefix = "COCO"
    test_part = False
    if config["which_dataset"] == "coco" and config["split"] == "val":
        test_part = True

    # Using hdf5 filename
    dataset = utils.get_dataset_hdf5(
        config["resolution"],
        data_path=config["data_root"],
        longtail=config["which_dataset"] == "imagenet_lt"
        and config["split"] == "train",
        split=config["split"],
        load_in_mem=config["load_in_mem"],
        which_dataset=config["which_dataset"],
        test_part=test_part,
    )

    loader = utils.get_dataloader(
        dataset, config["batch_size"], shuffle=False, **kwargs
    )

    # Load inception net
    net = inception_utils.load_inception_net(parallel=config["parallel"])
    device = "cuda"

    # Accumulate logits
    pool, logits, labels = [], [], []
    for i, batch in enumerate(tqdm(loader)):
        (x, y) = (batch[0], batch[1])
        x = x.to(device)
        with torch.no_grad():
            pool_val, logits_val = net(x)
            pool += [np.asarray(pool_val.cpu())]
            logits += [np.asarray(F.softmax(logits_val, 1).cpu())]
            labels += [np.asarray(y.cpu())]

    pool, logits, labels = [np.concatenate(item, 0) for item in [pool, logits, labels]]

    print("Calculating inception metrics...")
    IS_mean, IS_std = inception_utils.calculate_inception_score(logits)
    print(
        "Training data from dataset %s has IS of %5.5f +/- %5.5f"
        % (config["which_dataset"], IS_mean, IS_std)
    )
    # Prepare mu and sigma, save to disk. Remove "hdf5" by default
    # (the FID code also knows to strip "hdf5")
    print("Calculating means and covariances...")
    mu, sigma = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
    print("Saving calculated means and covariances to disk...")
    if config["which_dataset"] in ["imagenet", "imagenet_lt"]:
        dataset_name_prefix = "I"
    elif config["which_dataset"] == "coco":
        dataset_name_prefix = "COCO"
    np.savez(
        os.path.join(
            config["out_path"],
            "%s%i_%s%s%s_inception_moments.npz"
            % (
                dataset_name_prefix,
                config["resolution"],
                "longtail"
                if config["which_dataset"] == "imagenet_lt"
                and config["split"] == "train"
                else "",
                "_val" if config["split"] == "val" else "",
                "_test" if test_part else "",
            ),
        ),
        **{"mu": mu, "sigma": sigma}
    )
    # Compute stratified moments for ImageNet-LT dataset
    if config["stratified_moments"]:
        samples_per_class = np.load(
            "BigGAN_PyTorch/imagenet_lt/imagenet_lt_samples_per_class.npy",
            allow_pickle=True,
        )
        for strat_name in ["_many", "_low", "_few"]:
            if strat_name == "_many":
                logits_ = logits[samples_per_class[labels] >= 100]
                pool_ = pool[samples_per_class[labels] >= 100]
            elif strat_name == "_low":
                logits_ = logits[samples_per_class[labels] < 100]
                pool_ = pool[samples_per_class[labels] < 100]
                labels_ = labels[samples_per_class[labels] < 100]
                logits_ = logits_[samples_per_class[labels_] > 20]
                pool_ = pool_[samples_per_class[labels_] > 20]
            elif strat_name == "_few":
                logits_ = logits[samples_per_class[labels] <= 20]
                pool_ = pool[samples_per_class[labels] <= 20]
            print(
                "Calculating inception metrics for strat ",
                strat_name,
                " with number of samples ",
                len(logits_),
                "...",
            )
            IS_mean, IS_std = inception_utils.calculate_inception_score(logits_)
            print(
                "Training data from dataset %s has IS of %5.5f +/- %5.5f"
                % (config["which_dataset"], IS_mean, IS_std)
            )
            # Prepare mu and sigma, save to disk. Remove "hdf5" by default
            # (the FID code also knows to strip "hdf5")
            print("Calculating means and covariances...")
            mu, sigma = np.mean(pool_, axis=0), np.cov(pool_, rowvar=False)
            print("Saving calculated means and covariances to disk...")
            np.savez(
                os.path.join(
                    config["data_root"],
                    "%s%i__val%s_inception_moments.npz"
                    % (dataset_name_prefix, config["resolution"], strat_name),
                ),
                **{"mu": mu, "sigma": sigma}
            )


def main():
    # parse command line
    parser = prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == "__main__":
    main()
