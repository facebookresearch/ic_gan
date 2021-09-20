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
import numpy as np
from tqdm import trange
import pickle
import json
import sys

print(sys.path[0])
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import data_utils.inception_utils as inception_utils
import data_utils.utils as data_utils
import inference.utils as inference_utils
import BigGAN_PyTorch.utils as biggan_utils

LOCAL = False
try:
    import submitit
except:
    print(
        "No submitit package found! Defaulting to executing the script in the local machine"
    )
    LOCAL = True


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

        # Obtain additional variables for testing
        if self.config["eval_reference_set"] == "val" and self.config["longtail"]:
            stratified_m = True
        else:
            stratified_m = False
        if self.config["longtail"]:
            samples_per_class = np.load(
                "../BigGAN_PyTorch/imagenet_lt/imagenet_lt_samples_per_class.npy",
                allow_pickle=True,
            )
        else:
            samples_per_class = None

        # Prepare inception metrics function
        get_inception_metrics = inception_utils.prepare_inception_metrics(
            im_reference_filename,
            samples_per_class,
            self.config["parallel"],
            self.config["no_fid"],
            self.config["data_root"],
            prdc=self.config["eval_prdc"],
            stratified_fid=stratified_m,
            backbone=self.config["model_backbone"],
        )

        # If computing PRDC, we need a loader to obtain reference Inception features
        if self.config["eval_prdc"]:
            prdc_ref_set = data_utils.get_dataset_hdf5(
                **{
                    **self.config,
                    "data_path": self.config["data_root"],
                    "load_in_mem_feats": self.config["load_in_mem"],
                    "kmeans_subsampled": -1,
                    "test_part": True
                    if self.config["which_dataset"] == "coco"
                    and self.config["eval_reference_set"] == "val"
                    else False,
                    "split": self.config["eval_reference_set"],
                    "ddp": False,
                }
            )
            prdc_loader = data_utils.get_dataloader(
                **{
                    **self.config,
                    "dataset": prdc_ref_set,
                    "batch_size": self.config["batch_size"],
                    "use_checkpointable_sampler": False,
                    "shuffle": True,
                    "drop_last": False,
                }
            )
        else:
            prdc_loader = None

        # Get metrics
        eval_metrics = get_inception_metrics(
            sample,
            num_inception_images=self.config["num_inception_images"],
            num_splits=10,
            prints=False,
            loader_ref=prdc_loader,
            num_pr_images=self.config["num_inception_images"]
            if (
                self.config["longtail"] and self.config["eval_reference_set"] == "train"
            )
            else 10000,
        )
        eval_metrics_dict = dict()
        if self.config["eval_prdc"]:
            IS_mean, IS_std, FID, stratified_FID, prdc_metrics = eval_metrics
        else:
            IS_mean, IS_std, FID, stratified_FID = eval_metrics
        if stratified_m:
            eval_metrics_dict["stratified_FID"] = stratified_FID

        eval_metrics_dict["IS_mean"] = IS_mean
        eval_metrics_dict["IS_std"] = IS_std
        eval_metrics_dict["FID"] = FID
        print(eval_metrics_dict)
        if self.config["eval_prdc"]:
            eval_metrics_dict = {**prdc_metrics, **eval_metrics_dict}

        add_suffix = ""
        if self.config["z_var"] != 1.0:
            add_suffix = "_z_var" + str(self.config["z_var"])
        if not os.path.exists(
            os.path.join(self.config["samples_root"], self.config["experiment_name"])
        ):
            os.mkdir(
                os.path.join(
                    self.config["samples_root"], self.config["experiment_name"]
                )
            )
        # Save metrics in file
        print('Saving metrics in ', os.path.join(
                self.config["samples_root"],
                self.config["experiment_name"]))
        np.save(
            os.path.join(
                self.config["samples_root"],
                self.config["experiment_name"],
                "eval_metrics_reference_"
                + self.config["eval_reference_set"]
                + "_instances_"
                + self.config["eval_instance_set"]
                + "_kmeans"
                + str(self.config["kmeans_subsampled"])
                + "_seed"
                + str(self.config["seed"])
                + add_suffix
                + ".npy",
            ),
            eval_metrics_dict,
        )
        print("Computed metrics:")
        for key, value in eval_metrics_dict.items():
            print(key, ": ", value)

        if self.config["sample_npz"]:
            # Sample a number of images and save them to an NPZ, for use with TF-Inception
            # Lists to hold images and labels for images
            samples_path = os.path.join(
                self.config["samples_root"], self.config["experiment_name"]
            )
            if not os.path.exists(samples_path):
                os.mkdir(samples_path)
            x, y = [], []
            print(
                "Sampling %d images and saving them to npz..."
                % self.config["sample_num_npz"]
            )
            dict_tosave = {}
            for i in trange(
                int(
                    np.ceil(
                        self.config["sample_num_npz"] / float(self.config["batch_size"])
                    )
                )
            ):
                with torch.no_grad():
                    images, labels, _ = sample()
                if self.config["model_backbone"] == "stylegan2":
                    images = torch.clamp((images * 127.5 + 128), 0, 255)
                    images = ((images / 255) - 0.5) * 2

                x += [images.cpu().numpy()]
                if self.config["class_cond"]:
                    y += [labels.cpu().numpy()]
            if self.config["which_dataset"] == "imagenet":
                x = np.concatenate(x, 0)[: self.config["sample_num_npz"]]
                if self.config["class_cond"]:
                    y = np.concatenate(y, 0)[: self.config["sample_num_npz"]]

                np_filename = "%s/samples%s_seed%i.pickle" % (
                    samples_path,
                    "_kmeans" + str(self.config["kmeans_subsampled"])
                    if self.config["kmeans_subsampled"] > -1
                    else "",
                    self.config["seed"],
                )
                print("Saving npy to %s..." % np_filename)
                dict_tosave["x"] = x
                dict_tosave["y"] = y
                file_to_store = open(np_filename, "wb")
                pickle.dump(dict_tosave, file_to_store, protocol=4)
                file_to_store.close()

                if (
                    self.config["longtail"]
                    and self.config["eval_reference_set"] == "val"
                ):
                    print("Additionally storing stratified samples")
                    for strat_name in ["_many", "_low", "_few"]:
                        np_filename = "%s/%s/samples%s_seed%i_strat%s.pickle" % (
                            self.config["samples_root"],
                            self.config["experiment_name"],
                            "_kmeans" + str(self.config["kmeans_subsampled"])
                            if self.config["kmeans_subsampled"] > -1
                            else "",
                            self.config["seed"],
                            strat_name,
                        )
                        print(np_filename)
                        if strat_name == "_many":
                            x_ = x[samples_per_class[y] >= 100]
                            y_ = y[samples_per_class[y] >= 100]
                        elif strat_name == "_low":
                            x_ = x[samples_per_class[y] < 100]
                            y_ = y[samples_per_class[y] < 100]
                            x_ = x_[samples_per_class[y_] > 20]
                            y_ = y_[samples_per_class[y_] > 20]
                        elif strat_name == "_few":
                            x_ = x[samples_per_class[y] <= 20]
                            y_ = y[samples_per_class[y] <= 20]
                        dict_tosave = {}
                        dict_tosave["x"] = x_
                        dict_tosave["y"] = y_
                        file_to_store = open(np_filename, "wb")
                        pickle.dump(dict_tosave, file_to_store, protocol=4)
                        file_to_store.close()


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

    if config["run_setup"] == "local_debug":  # or LOCAL:
        tester()
    else:
        executor = submitit.SlurmExecutor(
            folder=config["slurm_logdir"], max_num_timeout=10
        )
        executor.update_parameters(
            gpus_per_node=1,
            partition=config["partition"],
            cpus_per_task=8,
            mem=128000,
            time=30,
            job_name="testing_" + config["experiment_name"],
        )
        executor.submit(tester)
        import time

        time.sleep(1)
