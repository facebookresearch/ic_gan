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
""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

import os
import numpy as np
from tqdm import tqdm, trange
import json

from imageio import imwrite as imsave

# Import my stuff
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import inference.utils as inference_utils
import BigGAN_PyTorch.utils as biggan_utils


class Tester:
    def __init__(self, config):
        self.config = vars(config) if not isinstance(config, dict) else config

    def __call__(self) -> float:
        # Seed RNG
        biggan_utils.seed_rng(self.config["seed"])

        import torch

        # Setup cudnn.benchmark for free speed
        torch.backends.cudnn.benchmark = True

        self.config = biggan_utils.update_config_roots(
            self.config, change_weight_folder=False
        )
        # Prepare root folders if necessary
        biggan_utils.prepare_root(self.config)

        # Load model
        self.G, self.config = inference_utils.load_model_inference(self.config)
        biggan_utils.count_parameters(self.G)

        # Get sampling function and reference statistics for FID
        print("Eval reference set is ", self.config["eval_reference_set"])
        sample, im_reference_filename = inference_utils.get_sampling_funct(
            self.config,
            self.G,
            instance_set=self.config["eval_instance_set"],
            reference_set=self.config["eval_reference_set"],
            which_dataset=self.config["which_dataset"],
        )

        if config["which_dataset"] == "coco":
            image_format = "jpg"
        else:
            image_format = "png"
        if (
            self.config["eval_instance_set"] == "val"
            and config["which_dataset"] == "coco"
        ):
            # using evaluation set
            test_part = True
        else:
            test_part = False
        path_samples = os.path.join(
            self.config["samples_root"],
            self.config["experiment_name"],
            "%s_images_seed%i%s%s%s"
            % (
                config["which_dataset"],
                config["seed"],
                "_test" if test_part else "",
                "_hd" + str(self.config["filter_hd"])
                if self.config["filter_hd"] > -1
                else "",
                ""
                if self.config["kmeans_subsampled"] == -1
                else "_" + str(self.config["kmeans_subsampled"]) + "centers",
            ),
        )

        print("Path samples will be ", path_samples)
        if not os.path.exists(path_samples):
            os.makedirs(path_samples)

        if not os.path.exists(
            os.path.join(self.config["samples_root"], self.config["experiment_name"])
        ):
            os.mkdir(
                os.path.join(
                    self.config["samples_root"], self.config["experiment_name"]
                )
            )
        print(
            "Sampling %d images and saving them with %s format..."
            % (self.config["sample_num_npz"], image_format)
        )
        counter_i = 0
        for i in trange(
            int(
                np.ceil(
                    self.config["sample_num_npz"] / float(self.config["batch_size"])
                )
            )
        ):
            with torch.no_grad():
                images, labels, _ = sample()

                fake_imgs = images.cpu().detach().numpy().transpose(0, 2, 3, 1)
                if self.config["model_backbone"] == "biggan":
                    fake_imgs = fake_imgs * 0.5 + 0.5
                elif self.config["model_backbone"] == "stylegan2":
                    fake_imgs = np.clip((fake_imgs * 127.5 + 128), 0, 255).astype(
                        np.uint8
                    )
                for fake_img in fake_imgs:
                    imsave(
                        "%s/%06d.%s" % (path_samples, counter_i, image_format), fake_img
                    )
                    counter_i += 1
                    if counter_i >= self.config["sample_num_npz"]:
                        break


if __name__ == "__main__":
    parser = biggan_utils.prepare_parser()
    parser = biggan_utils.add_sample_parser(parser)
    parser = inference_utils.add_backbone_parser(parser)
    config = vars(parser.parse_args())
    config["n_classes"] = 1000
    if config["json_config"] != "":
        data = json.load(open(config["json_config"]))
        for key in data.keys():
            if "exp_name" in key:
                config["experiment_name"] = data[key]
            else:
                config[key] = data[key]
    else:
        print("No json file to load configuration from")

    tester = Tester(config)

    tester()
