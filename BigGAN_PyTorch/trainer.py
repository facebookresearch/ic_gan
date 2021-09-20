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
import functools
import math
from tqdm import tqdm, trange
import argparse
import time
import subprocess
import re
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import numpy as np

import torch
import torch.nn as nn

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim

# Import my stuff
import data_utils.inception_utils as inception_utils
import utils
import train_fns
from sync_batchnorm import patch_replication_callback
from data_utils import utils as data_utils


def run(config, ddp_setup="slurm", master_node=""):
    config["n_classes"] = 1000  # utils.nclass_dict[self.config['dataset']]
    config["G_activation"] = utils.activation_dict[config["G_nl"]]
    config["D_activation"] = utils.activation_dict[config["D_nl"]]
    config = utils.update_config_roots(config)

    # Prepare root folders if necessary
    utils.prepare_root(config)

    if config["ddp_train"]:
        if ddp_setup == "slurm":
            n_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES"))
            n_gpus_per_node = int(os.environ.get("SLURM_TASKS_PER_NODE").split("(")[0])
            world_size = n_gpus_per_node * n_nodes
            print(
                "Master node is ",
                master_node,
                " World size is ",
                world_size,
                " with ",
                n_gpus_per_node,
                "gpus per node.",
            )
            dist_url = "tcp://"
            dist_url += master_node
            port = 40000
            dist_url += ":" + str(port)
            print("Dist url ", dist_url)
            train(-1, world_size, config, dist_url)
        else:
            world_size = torch.cuda.device_count()
            dist_url = "env://"
            mp.spawn(
                train, args=(world_size, config, dist_url), nprocs=world_size, join=True
            )
    else:
        train(0, -1, config, None)


def train(rank, world_size, config, dist_url):
    print("Rank of this job is ", rank)
    copy_locally = False
    tmp_dir = ""
    if config["ddp_train"]:
        if dist_url == "env://":
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            local_rank = rank
        else:
            rank = int(os.environ.get("SLURM_PROCID"))
            local_rank = int(os.environ.get("SLURM_LOCALID"))
            copy_locally = True
            tmp_dir = "/scratch/slurm_tmpdir/" + str(os.environ.get("SLURM_JOB_ID"))

        print("Before setting process group")
        print(dist_url, rank)
        dist.init_process_group(
            backend="nccl", init_method=dist_url, rank=rank, world_size=world_size
        )
        print("After setting process group")
        device = "cuda:{}".format(local_rank)  # rank % 8)
        print(dist_url, rank, " /Device is ", device)
    else:
        device = "cuda"
        local_rank = "cuda"

    # Update the config dict as necessary
    # This is for convenience, to add settings derived from the user-specified
    # configuration into the config-dict (e.g. inferring the number of classes
    # and size of the images from the dataset, passing in a pytorch object
    # for the activation specified as a string)'

    # Seed RNG
    utils.seed_rng(config["seed"] + rank)

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True
    if config["deterministic_run"]:
        torch.backends.cudnn.deterministic = True

    # Import the model--this line allows us to dynamically select different files.
    model = __import__(config["model"])
    experiment_name = (
        config["experiment_name"]
        if config["experiment_name"]
        else utils.name_from_config(config)
    )
    print("Experiment name is %s" % experiment_name)

    if config["ddp_train"]:
        torch.cuda.set_device(device)
    # Next, build the model
    G = model.Generator(**{**config, "embedded_optimizers": False}).to(device)
    D = model.Discriminator(**{**config, "embedded_optimizers": False}).to(device)

    # If using EMA, prepare it
    if config["ema"]:
        print("Preparing EMA for G with decay of {}".format(config["ema_decay"]))
        G_ema = model.Generator(**{**config, "skip_init": True, "no_optim": True}).to(
            device
        )
        ema = utils.ema(G, G_ema, config["ema_decay"], config["ema_start"])
    else:
        G_ema, ema = None, None

    print(
        "Number of params in G: {} D: {}".format(
            *[sum([p.data.nelement() for p in net.parameters()]) for net in [G, D]]
        )
    )

    # Setup the optimizers
    if config["D_fp16"]:
        print("Using fp16 adam ")
        optim_type = utils.Adam16
    else:
        optim_type = optim.Adam
    optimizer_D = optim_type(
        params=D.parameters(),
        lr=config["D_lr"],
        betas=(config["D_B1"], config["D_B2"]),
        weight_decay=0,
        eps=config["adam_eps"],
    )
    optimizer_G = optim_type(
        params=G.parameters(),
        lr=config["G_lr"],
        betas=(config["G_B1"], config["G_B2"]),
        weight_decay=0,
        eps=config["adam_eps"],
    )

    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {
        "itr": 0,
        "epoch": 0,
        "save_num": 0,
        "save_best_num": 0,
        "best_IS": 0,
        "best_FID": 999999,
        "es_epoch": 0,
        "config": config,
    }

    # FP16?
    if config["G_fp16"]:
        print("Casting G to float16...")
        G = G.half()
        if config["ema"]:
            G_ema = G_ema.half()
    if config["D_fp16"]:
        print("Casting D to fp16...")
        D = D.half()

    ## DDP the models
    if config["ddp_train"]:
        print("before G DDP ")
        G = DDP(
            G,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        print("After G DDP ")
        D = DDP(
            D,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    # If loading from a pre-trained model, load weights
    print("Loading weights...")
    if config["ddp_train"]:
        dist.barrier()
        map_location = device
    else:
        map_location = None

    utils.load_weights(
        G,
        D,
        state_dict,
        config["weights_root"],
        experiment_name,
        config["load_weights"] if config["load_weights"] else None,
        G_ema if config["ema"] else None,
        map_location=map_location,
        embedded_optimizers=False,
        G_optim=optimizer_G,
        D_optim=optimizer_D,
    )

    # wrapper class
    GD = model.G_D(G, D, optimizer_G=optimizer_G, optimizer_D=optimizer_D)

    if config["parallel"] and world_size > -1:
        GD = nn.DataParallel(GD)
        if config["cross_replica"]:
            patch_replication_callback(GD)

    # Prepare loggers for stats; metrics holds test metrics,
    # lmetrics holds any desired training metrics.
    if rank == 0:
        test_metrics_fname = "%s/%s_log.jsonl" % (config["logs_root"], experiment_name)
        train_metrics_fname = "%s/%s" % (config["logs_root"], experiment_name)
        print("Inception Metrics will be saved to {}".format(test_metrics_fname))
        test_log = utils.MetricsLogger(test_metrics_fname, reinitialize=False)
        print("Training Metrics will be saved to {}".format(train_metrics_fname))
        train_log = utils.MyLogger(
            train_metrics_fname, reinitialize=False, logstyle=config["logstyle"]
        )
        # Write metadata
        utils.write_metadata(config["logs_root"], experiment_name, config, state_dict)
    else:
        test_log = None
        train_log = None

    D_batch_size = (
        config["batch_size"] * config["num_D_steps"] * config["num_D_accumulations"]
    )

    if config["longtail"]:
        samples_per_class = np.load(
            "imagenet_lt/imagenet_lt_samples_per_class.npy", allow_pickle=True
        )
        class_probabilities = np.load(
            "imagenet_lt/imagenet_lt_class_prob.npy", allow_pickle=True
        )
    else:
        samples_per_class, class_probabilities = None, None

    train_dataset = data_utils.get_dataset_hdf5(
        **{
            **config,
            "data_path": config["data_root"],
            "batch_size": D_batch_size,
            "augment": config["hflips"],
            "local_rank": local_rank,
            "copy_locally": copy_locally,
            "tmp_dir": tmp_dir,
            "ddp": config["ddp_train"],
        }
    )
    train_loader = data_utils.get_dataloader(
        **{
            **config,
            "dataset": train_dataset,
            "batch_size": config["batch_size"],
            "start_epoch": state_dict["epoch"],
            "start_itr": state_dict["itr"],
            "longtail_temperature": config["longtail_temperature"],
            "samples_per_class": samples_per_class,
            "class_probabilities": class_probabilities,
            "rank": rank,
            "world_size": world_size,
            "shuffle": True,
            "drop_last": True,
        }
    )

    # Prepare inception metrics: FID and IS
    is_moments_prefix = "I" if config["which_dataset"] == "imagenet" else "COCO"

    im_filename = "%s%i_%s" % (
        is_moments_prefix,
        config["resolution"],
        "" if not config["longtail"] else "longtail",
    )
    print("Using ", im_filename, "for Inception metrics.")

    get_inception_metrics = inception_utils.prepare_inception_metrics(
        im_filename,
        samples_per_class,
        config["parallel"],
        config["no_fid"],
        config["data_root"],
        device=device,
    )

    G_batch_size = config["G_batch_size"]

    z_, y_ = data_utils.prepare_z_y(
        G_batch_size,
        G.module.dim_z if config["ddp_train"] else G.dim_z,
        config["n_classes"],
        device=device,
        fp16=config["G_fp16"],
        longtail_gen=config["longtail_gen"],
        custom_distrib=config["custom_distrib_gen"],
        longtail_temperature=config["longtail_temperature"],
        class_probabilities=class_probabilities,
    )

    # Balance instance sampling for ImageNet-LT
    weights_sampling = None
    if (
        config["longtail"]
        and config["use_balanced_sampler"]
        and config["instance_cond"]
    ):
        if config["which_knn_balance"] == "center_balance":
            print(
                "Balancing the instance features." "Using custom temperature distrib?",
                config["custom_distrib_gen"],
                " with temperature",
                config["longtail_temperature"],
            )
            weights_sampling = data_utils.make_weights_for_balanced_classes(
                samples_per_class,
                train_loader.dataset.labels,
                1000,
                config["custom_distrib_gen"],
                config["longtail_temperature"],
                class_probabilities=class_probabilities,
            )
        # Balancing the NN classes (p(y))
        elif config["which_knn_balance"] == "nnclass_balance":
            print(
                "Balancing the class distribution (classes drawn from the neighbors)."
                " Using custom temperature distrib?",
                config["custom_distrib_gen"],
                " with temperature",
                config["longtail_temperature"],
            )
            weights_sampling = torch.exp(
                class_probabilities / config["longtail_temperature"]
            ) / torch.sum(
                torch.exp(class_probabilities / config["longtail_temperature"])
            )

    # Configure conditioning sampling function to train G
    sample_conditioning = functools.partial(
        data_utils.sample_conditioning_values,
        z_=z_,
        y_=y_,
        dataset=train_dataset,
        batch_size=G_batch_size,
        weights_sampling=weights_sampling,
        ddp=config["ddp_train"],
        constant_conditioning=config["constant_conditioning"],
        class_cond=config["class_cond"],
        instance_cond=config["instance_cond"],
        nn_sampling_strategy=config["which_knn_balance"],
    )

    print("G batch size ", G_batch_size)
    # Loaders are loaded, prepare the training function
    train = train_fns.GAN_training_function(
        G,
        D,
        GD,
        ema,
        state_dict,
        config,
        sample_conditioning,
        embedded_optimizers=False,
        device=device,
        batch_size=G_batch_size,
    )

    # Prepare Sample function for use with inception metrics
    sample = functools.partial(
        utils.sample,
        G=(G_ema if config["ema"] and config["use_ema"] else G),
        sample_conditioning_func=sample_conditioning,
        config=config,
        class_cond=config["class_cond"],
        instance_cond=config["instance_cond"],
    )

    print("Beginning training at epoch %d..." % state_dict["epoch"])
    # Train for specified number of epochs, although we mostly track G iterations.
    best_FID_run = state_dict["best_FID"]
    FID = state_dict["best_FID"]

    for epoch in range(state_dict["epoch"], config["num_epochs"]):
        # Set epoch for distributed loader
        if config["ddp_train"]:
            train_loader.sampler.set_epoch(epoch)
        # Initialize seeds at every epoch (useful for conditioning and
        # noise sampling, as well as data order in the sampler)
        if config["deterministic_run"]:
            utils.seed_rng(config["seed"] + rank + state_dict["epoch"])
        # Which progressbar to use? TQDM or my own?
        if config["pbar"] == "mine":
            pbar = utils.progress(
                train_loader,
                displaytype="s1k" if config["use_multiepoch_sampler"] else "eta",
            )
        else:
            pbar = tqdm(train_loader)
        s = time.time()
        print("Before iteration, dataloader length", len(train_loader))
        for i, batch in enumerate(pbar):
            # if i> 5:
            #     break
            in_label, in_feat = None, None
            if config["instance_cond"] and config["class_cond"]:
                x, in_label, in_feat, _ = batch
            elif config["instance_cond"]:
                x, in_feat, _ = batch
            elif config["class_cond"]:
                x, in_label = batch
                if config["constant_conditioning"]:
                    in_label = torch.zeros_like(in_label)
            else:
                x = batch

            x = x.to(device, non_blocking=True)
            if in_label is not None:
                in_label = in_label.to(device, non_blocking=True)
            if in_feat is not None:
                in_feat = in_feat.float().to(device, non_blocking=True)
            # Increment the iteration counter
            state_dict["itr"] += 1
            # Make sure G and D are in training mode, just in case they got set to eval
            # For D, which typically doesn't have BN, this shouldn't matter much.
            G.train()
            D.train()
            if config["ema"]:
                G_ema.train()

            metrics = train(x, in_label, in_feat)
            #   print('After training step ', time.time() - s_stratified)
            #  s_stratified = time.time()
            if rank == 0:
                train_log.log(itr=int(state_dict["itr"]), **metrics)

            # If using my progbar, print metrics.
            if config["pbar"] == "mine" and rank == 0:
                print(
                    ", ".join(
                        ["itr: %d" % state_dict["itr"]]
                        + ["%s : %+4.3f" % (key, metrics[key]) for key in metrics]
                    ),
                    end=" ",
                )
            # Test every specified interval

            print("Iteration time ", time.time() - s)
            s = time.time()
        # Increment epoch counter at end of epoch
        state_dict["epoch"] += 1

        if not (state_dict["epoch"] % config["test_every"]):
            if config["G_eval_mode"]:
                print("Switching G to eval mode...")
                G.eval()
                D.eval()

            # Compute IS and FID using training dataset as reference
            test_time = time.time()
            IS, FID = train_fns.test(
                G,
                D,
                G_ema,
                z_,
                y_,
                state_dict,
                config,
                sample,
                get_inception_metrics,
                experiment_name,
                test_log,
                loader=None,
                embedded_optimizers=False,
                G_optim=optimizer_G,
                D_optim=optimizer_D,
                rank=rank,
            )
            print("Testing took ", time.time() - test_time)

            if 2 * IS < state_dict["best_IS"] and config["stop_when_diverge"]:
                print("Experiment diverged!")
                break
            else:
                print("IS is ", IS, " and 2x best is ", 2 * state_dict["best_IS"])

        if not (state_dict["epoch"] % config["save_every"]) and rank == 0:
            train_fns.save_weights(
                G,
                D,
                G_ema,
                state_dict,
                config,
                experiment_name,
                embedded_optimizers=False,
                G_optim=optimizer_G,
                D_optim=optimizer_D,
            )
        if rank == 0:
            if FID < best_FID_run:
                best_FID_run = FID
                state_dict["es_epoch"] = 0
            else:
                state_dict["es_epoch"] += 1
            if state_dict["es_epoch"] >= config["es_patience"]:
                print("reached Early stopping!")
                return FID
    return FID
