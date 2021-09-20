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
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import numpy as np
import functools

import torch
import torch.nn as nn

import stylegan2_ada_pytorch.dnnlib as dnnlib
import stylegan2_ada_pytorch.legacy as legacy
import BigGAN_PyTorch.utils as biggan_utils
import BigGAN_PyTorch.BigGAN as BigGANModel
import data_utils.utils as data_utils


def get_sampling_funct(
    config,
    generator,
    instance_set="train",
    reference_set="train",
    which_dataset="imagenet",
):
    """It prepares the generation sampling function and the inception moments filename.

    Arguments
    ---------
        config: dict
            Dictionary with configuration parameters.
        generator: torch.nn.module
            Generator network.
        instance_set: str, optional
            If `train`, build a dataset with the training split.
            If `val`, build a dataset with the validation split.
        reference_set: str, optional
            If `train`, use training as a reference to compute metrics.
            If `val`, use validation as a reference to compute metrics.
        which_dataset: str, optional
            Dataset name
    Returns
    -------
    sample_, im_filename, dataset
        sample_: function
            Function to sample generated images from.
        im_filename: str
            Filename where to find the inception moments used to compute the FID metric
             (with Pytorch code).

    """
    # Class labels will follow either a long-tail
    # distribution(if reference==train) or a uniform distribution
    # otherwise).
    if config["longtail"]:
        class_probabilities = np.load(
            "../BigGAN_PyTorch/imagenet_lt/imagenet_lt_class_prob.npy",
            allow_pickle=True,
        )
        samples_per_class = np.load(
            "../BigGAN_PyTorch/imagenet_lt/imagenet_lt_samples_per_class.npy",
            allow_pickle=True,
        )
    else:
        class_probabilities, samples_per_class = None, None

    if (reference_set == "val" and instance_set == "val") and config[
        "which_dataset"
    ] == "coco":
        # using evaluation set
        test_part = True
    else:
        test_part = False

    # Prepare the noise distribution and class distribution
    z_, y_ = data_utils.prepare_z_y(
        config["batch_size"],
        generator.dim_z if config["model_backbone"] == "biggan" else 512,
        config["n_classes"],
        device="cuda",
        fp16=config["G_fp16"],
        longtail_gen=config["longtail"] if reference_set == "train" else False,
        z_var=config["z_var"],
        class_probabilities=class_probabilities,
    )

    # Obtain dataset to sample instances from.
    if config["instance_cond"]:
        dataset = data_utils.get_dataset_hdf5(
            **{
                **config,
                "data_path": config["data_root"],
                "batch_size": config["batch_size"],
                "load_in_mem_feats": config["load_in_mem"],
                "split": instance_set,
                "test_part": test_part,
                "augment": False,
                "ddp": False,
            }
        )

    else:
        dataset = None

    # Weights to sample instances (+classes). By default, weights are None,
    # which means no specific sampling weights will be used (uniform).
    # For long-tail experiments with training as reference distribution,
    # balance the sampling with a long-tail distribution.
    weights_sampling = None
    nn_sampling_strategy = "instance_balance"
    if config["instance_cond"] and config["class_cond"] and config["longtail"]:
        nn_sampling_strategy = "nnclass_balance"
        if reference_set == "val":
            print("Sampling classes uniformly for generator.")
            # Sampling classes uniformly
            weights_sampling = None
        else:
            print("Balancing with weights=samples_per_class (long-tailed).")
            weights_sampling = samples_per_class

    # Prepare conditioning sampling function
    sample_conditioning = functools.partial(
        data_utils.sample_conditioning_values,
        z_=z_,
        y_=y_,
        constant_conditioning=config["constant_conditioning"],
        batch_size=config["batch_size"],
        weights_sampling=weights_sampling,
        dataset=dataset,
        class_cond=config["class_cond"],
        instance_cond=config["instance_cond"],
        nn_sampling_strategy=nn_sampling_strategy,
    )

    # Prepare Sample function for use with inception metrics
    sample_ = functools.partial(
        sample,
        generator,
        sample_conditioning_func=sample_conditioning,
        config=config,
        class_cond=config["class_cond"],
        instance_cond=config["instance_cond"],
        backbone=config["model_backbone"],
        truncation_value=config["z_var"],
    )

    # Get reference statistics to compute FID
    im_prefix = "I" if which_dataset == "imagenet" else "COCO"
    if reference_set == "train":
        im_filename = "%s%i_%s" % (
            im_prefix,
            config["resolution"],
            "" if not config["longtail"] else "longtail",
        )
    else:
        im_filename = "%s%i_%s%s" % (
            im_prefix,
            config["resolution"],
            "_val",
            "_test" if test_part else "",
        )
    print("Using ", im_filename, "for Inception metrics.")
    return sample_, im_filename


def sample(
    generator,
    sample_conditioning_func,
    config,
    class_cond=True,
    instance_cond=False,
    device="cuda",
    backbone="biggan",
    truncation_value=1.0,
):
    """It samples generated images from the model, given the input noise (and conditioning).

    Arguments
    ---------
        generator: torch.nn.module
            Generator network.
        sample_conditioning_func: function
            A function that samples and outputs the conditionings to be fed to the generator.
        config: dict
            Dictionary with configuration parameters.
        class_cond: bool, optional
            If True, use class labels to condition the generator.
        instance_cond: bool, optional
            If True, use instance features to condition the generator.
        device: str, optional
            Device name
        backbone: str, optional
            Name of the backbone architecture to use ("biggan" or "stylegan2").
        truncation_value: float, optional
            Variance for the noise distribution, attributed to the truncation values in BigGAN.
    Returns
    -------
        gen_samples: torch.FloatTensor
            Generated images.
        y_: torch.Tensor
            Sampled class labels. If using BigGAN backbone, y_.shape = [bs],
            if using StyleGAN2 backbone, y_.shape = [bs, c_dim], where `bs` is the batch size
            and `c_dim` is the dimensionality of the class embedding.
        feats_: torch.Tensor
            Sampled instance feature vectors, with shape [bs, h_dim], where `bs` is the batch size
            and `h_dim` is the dimensionality of the instance feature vectors.

    """
    # Sample conditioning
    conditioning = sample_conditioning_func()
    # Send conditionings to proper devices
    with torch.no_grad():
        if not class_cond and not instance_cond:
            z_ = conditioning
            y_, feats_ = None, None
        elif class_cond and not instance_cond:
            z_, y_ = conditioning
            y_ = y_.long()
            y_ = y_.to(device, non_blocking=True)
            feats_ = None
        elif instance_cond and not class_cond:
            z_, feats_ = conditioning
            feats_ = feats_.to(device, non_blocking=True)
            y_ = None
        elif instance_cond and class_cond:
            z_, y_, feats_ = conditioning
            y_, feats_ = (
                y_.to(device, non_blocking=True),
                feats_.to(device, non_blocking=True),
            )
        z_ = z_.to(device, non_blocking=True)

        if backbone == "stylegan2":
            if y_ is None:
                y_ = torch.empty([z_.shape[0], generator.c_dim], device=device)
            else:
                y_ = torch.eye(config["n_classes"], device=device)[y_]
            if feats_ is None:
                feats_ = torch.empty([z_.shape[0], generator.h_dim], device=device)

        # Sample images given the conditionings
        if backbone == "biggan":
            if config["parallel"]:
                gen_samples = nn.parallel.data_parallel(generator, (z_, y_, feats_))
            else:
                gen_samples = generator(z_, y_, feats_)
        elif backbone == "stylegan2":
            gen_samples = generator(
                z=z_,
                c=y_,
                feats=feats_,
                truncation_psi=truncation_value,
                noise_mode="const",
            )
    return gen_samples, y_, feats_


def load_model_inference(config, device="cuda"):
    """It loads the generator network to do inference with and over-rides the configuration file.

    Arguments
    ---------
        config: dict
            Dictionary with configuration parameters.
        device: str, optional
            Device name
    Returns
    -------
        generator: torch.nn.module
            Generator network.
        config: dict
            Overwritten configuration dictionary from the model checkpoint if it exists.

    """
    if config["model_backbone"] == "biggan":
        # Select checkpoint name according to best FID in checkpoint
        best_fid = 1e5
        best_name_final = ""
        for name_best in ["best0", "best1"]:
            try:
                root = "/".join([config["weights_root"], config["experiment_name"]])
                state_dict_loaded = torch.load(
                    "%s/%s.pth"
                    % (root, biggan_utils.join_strings("_", ["state_dict", name_best]))
                )
                print(
                    "For name best ",
                    name_best,
                    " we have an FID: ",
                    state_dict_loaded["best_FID"],
                )
                if state_dict_loaded["best_FID"] < best_fid:
                    best_fid = state_dict_loaded["best_FID"]
                    best_name_final = name_best
            except:
                print("Checkpoint with name ", name_best, " not in folder.")
        config["load_weights"] = best_name_final
        print("Final name selected is ", best_name_final)

        # Prepare state dict, which holds things like epoch # and itr #
        state_dict = {
            "itr": 0,
            "epoch": 0,
            "save_num": 0,
            "save_best_num": 0,
            "best_IS": 0,
            "best_FID": 999999,
            "config": config,
        }
        # Get override some parameters from trained model in experiment config
        biggan_utils.load_weights(
            None,
            None,
            state_dict,
            config["weights_root"],
            config["experiment_name"],
            config["load_weights"],
            None,
            strict=False,
            load_optim=False,
            eval=True,
        )

        # Ignore items which we might want to overwrite from the command line
        for item in state_dict["config"]:
            if item not in [
                "base_root",
                "data_root",
                "load_weights",
                "batch_size",
                "num_workers",
                "weights_root",
                "logs_root",
                "samples_root",
                "eval_reference_set",
                "eval_instance_set",
                "which_dataset",
                "seed",
                "eval_prdc",
                "use_balanced_sampler",
                "custom_distrib",
                "longtail_temperature",
                "longtail_gen",
                "num_inception_images",
                "sample_num_npz",
                "load_in_mem",
                "split",
                "z_var",
                "kmeans_subsampled",
                "filter_hd",
                "n_subsampled_data",
                "feature_augmentation",
            ]:
                if item == "experiment_name" and config["experiment_name"] != "":
                    continue  # Allows to overwride the name of the experiment at test time
                config[item] = state_dict["config"][item]
        # No data augmentation during testing
        config["feature_augmentation"] = False
        config["hflips"] = False
        config["DA"] = False

        experiment_name = (
            config["experiment_name"]
            if config["experiment_name"]
            else biggan_utils.name_from_config(config)
        )
        print("Experiment name is %s" % experiment_name)

        # Next, build the model
        generator = BigGANModel.Generator(**config).to(device)

        # Load weights
        print("Loading weights...")

        # Here is where we deal with the ema--load ema weights or load normal weights
        biggan_utils.load_weights(
            generator if not (config["use_ema"]) else None,
            None,
            state_dict,
            config["weights_root"],
            experiment_name,
            config["load_weights"],
            generator if config["ema"] and config["use_ema"] else None,
            strict=False,
            load_optim=False,
        )

        if config["G_eval_mode"]:
            print("Putting G in eval mode..")
            generator.eval()
        else:
            print("G is in %s mode..." % ("training" if generator.training else "eval"))

    elif config["model_backbone"] == "stylegan2":
        # StyleGAN2 saves the entire network + weights in a pickle. Load it here.
        network_pkl = os.path.join(
            config["base_root"], config["experiment_name"], "best-network-snapshot.pkl"
        )
        print('Loading networks from "%s"...' % network_pkl)
        with dnnlib.util.open_url(network_pkl) as f:
            generator = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore
    return generator, config


def add_backbone_parser(parser):
    parser.add_argument(
        "--model_backbone",
        type=str,
        default="biggan",
        choices=["biggan", "stylegan2"],
        help="Backbone model? " "(default: %(default)s)",
    )
    return parser
