# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import torch

import data_utils.utils as data_utils
import inference.utils as inference_utils
import BigGAN_PyTorch.utils as biggan_utils
from data_utils.datasets_common import pil_loader
import torchvision.transforms as transforms
import time


def get_data(root_path, model, resolution, which_dataset, visualize_instance_images):
    data_path = os.path.join(root_path, "stored_instances")
    if model == "cc_icgan":
        feature_extractor = "classification"
    else:
        feature_extractor = "selfsupervised"
    filename = "%s_res%i_rn50_%s_kmeans_k1000_instance_features.npy" % (
        which_dataset,
        resolution,
        feature_extractor,
    )
    # Load conditioning instances from files
    data = np.load(os.path.join(data_path, filename), allow_pickle=True).item()

    transform_list = None
    if visualize_instance_images:
        # Transformation used for ImageNet images.
        transform_list = transforms.Compose(
            [data_utils.CenterCropLongEdge(), transforms.Resize(resolution)]
        )
    return data, transform_list


def get_model(exp_name, root_path, backbone, device="cuda"):
    parser = biggan_utils.prepare_parser()
    parser = biggan_utils.add_sample_parser(parser)
    parser = inference_utils.add_backbone_parser(parser)

    args = ["--experiment_name", exp_name]
    args += ["--base_root", root_path]
    args += ["--model_backbone", backbone]

    config = vars(parser.parse_args(args=args))

    # Load model and overwrite configuration parameters if stored in the model
    config = biggan_utils.update_config_roots(config, change_weight_folder=False)
    generator, config = inference_utils.load_model_inference(config, device=device)
    biggan_utils.count_parameters(generator)
    generator.eval()

    return generator


def get_conditionings(test_config, generator, data):
    # Obtain noise vectors
    z = torch.empty(
        test_config["num_imgs_gen"] * test_config["num_conditionings_gen"],
        generator.z_dim if config["model_backbone"] == "stylegan2" else generator.dim_z,
    ).normal_(mean=0, std=test_config["z_var"])

    # Subsampling some instances from the 1000 k-means centers file
    if test_config["num_conditionings_gen"] > 1:
        total_idxs = np.random.choice(
            range(1000), test_config["num_conditionings_gen"], replace=False
        )

    # Obtain features, labels and ground truth image paths
    all_feats, all_img_paths, all_labels = [], [], []
    for counter in range(test_config["num_conditionings_gen"]):
        # Index in 1000 k-means centers file
        if test_config["index"] is not None:
            idx = test_config["index"]
        else:
            idx = total_idxs[counter]
        # Image paths to visualize ground-truth instance
        if test_config["visualize_instance_images"]:
            all_img_paths.append(data["image_path"][idx])
        # Instance features
        all_feats.append(
            torch.FloatTensor(data["instance_features"][idx : idx + 1]).repeat(
                test_config["num_imgs_gen"], 1
            )
        )
        # Obtain labels
        if test_config["swap_target"] is not None:
            # Swap label for a manually specified one
            label_int = test_config["swap_target"]
        else:
            # Use the label associated to the instance feature
            label_int = int(data["labels"][idx])
        # Format labels according to the backbone
        labels = None
        if test_config["model_backbone"] == "stylegan2":
            dim_labels = 1000
            labels = torch.eye(dim_labels)[torch.LongTensor([label_int])].repeat(
                test_config["num_imgs_gen"], 1
            )
        else:
            if test_config["model"] == "cc_icgan":
                labels = torch.LongTensor([label_int]).repeat(
                    test_config["num_imgs_gen"]
                )
        all_labels.append(labels)
    # Concatenate all conditionings
    all_feats = torch.cat(all_feats)
    if all_labels[0] is not None:
        all_labels = torch.cat(all_labels)
    else:
        all_labels = None
    return z, all_feats, all_labels, all_img_paths


def main(test_config):
    suffix = (
        "_nofeataug"
        if test_config["resolution"] == 256
        and test_config["trained_dataset"] == "imagenet"
        else ""
    )
    exp_name = "%s_%s_%s_res%i%s" % (
        test_config["model"],
        test_config["model_backbone"],
        test_config["trained_dataset"],
        test_config["resolution"],
        suffix,
    )
    device = "cuda"
    ### -- Data -- ###
    data, transform_list = get_data(
        test_config["root_path"],
        test_config["model"],
        test_config["resolution"],
        test_config["which_dataset"],
        test_config["visualize_instance_images"],
    )

    ### -- Model -- ###
    generator = get_model(
        exp_name, test_config["root_path"], test_config["model_backbone"], device=device
    )

    ### -- Generate images -- ###
    # Prepare input and conditioning: different noise vector per sample but the same conditioning
    # Sample noise vector
    z, all_feats, all_labels, all_img_paths = get_conditionings(
        test_config, generator, data
    )

    ## Generate the images
    all_generated_images = []
    with torch.no_grad():
        num_batches = 1 + (z.shape[0]) // test_config["batch_size"]
        for i in range(num_batches):
            start = test_config["batch_size"] * i
            end = min(
                test_config["batch_size"] * i + test_config["batch_size"], z.shape[0]
            )
            if all_labels is not None:
                labels_ = all_labels[start:end].to(device)
            else:
                labels_ = None
            gen_img = generator(
                z[start:end].to(device), labels_, all_feats[start:end].to(device)
            )
            if test_config["model_backbone"] == "biggan":
                gen_img = ((gen_img * 0.5 + 0.5) * 255).int()
            elif test_config["model_backbone"] == "stylegan2":
                gen_img = torch.clamp((gen_img * 127.5 + 128), 0, 255).int()
            all_generated_images.append(gen_img.cpu())
    all_generated_images = torch.cat(all_generated_images)
    all_generated_images = all_generated_images.permute(0, 2, 3, 1).numpy()

    big_plot = []
    for i in range(0, test_config["num_conditionings_gen"]):
        row = []
        for j in range(0, test_config["num_imgs_gen"]):
            subplot_idx = (i * test_config["num_imgs_gen"]) + j
            row.append(all_generated_images[subplot_idx])
        row = np.concatenate(row, axis=1)
        big_plot.append(row)
    big_plot = np.concatenate(big_plot, axis=0)

    # (Optional) Show ImageNet ground-truth conditioning instances
    if test_config["visualize_instance_images"]:
        all_gt_imgs = []
        for i in range(0, len(all_img_paths)):
            all_gt_imgs.append(
                np.array(
                    transform_list(
                        pil_loader(
                            os.path.join(test_config["dataset_path"], all_img_paths[i])
                        )
                    )
                ).astype(np.uint8)
            )
        all_gt_imgs = np.concatenate(all_gt_imgs, axis=0)
        white_space = (
            np.ones((all_gt_imgs.shape[0], 20, all_gt_imgs.shape[2])) * 255
        ).astype(np.uint8)
        big_plot = np.concatenate([all_gt_imgs, white_space, big_plot], axis=1)

    plt.figure(
        figsize=(
            5 * test_config["num_imgs_gen"],
            5 * test_config["num_conditionings_gen"],
        )
    )
    plt.imshow(big_plot)
    plt.axis("off")

    fig_path = "%s_Generations_with_InstanceDataset_%s%s%s_zvar%0.2f.png" % (
        exp_name,
        test_config["which_dataset"],
        "_index" + str(test_config["index"])
        if test_config["index"] is not None
        else "",
        "_class_idx" + str(test_config["swap_target"])
        if test_config["swap_target"] is not None
        else "",
        test_config["z_var"],
    )
    plt.savefig(fig_path, dpi=600, bbox_inches="tight", pad_inches=0)

    print("Done! Figure saved as %s" % (fig_path))


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generate and save images using pre-trained models"
    )

    parser.add_argument(
        "--root_path",
        type=str,
        required=True,
        help="Path where pretrained models + instance features have been downloaded.",
    )
    parser.add_argument(
        "--which_dataset",
        type=str,
        default="imagenet",
        choices=["imagenet", "coco"],
        help="Dataset to sample instances from.",
    )
    parser.add_argument(
        "--trained_dataset",
        type=str,
        default="imagenet",
        choices=["imagenet", "coco"],
        help="Dataset in which the model has been trained on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="icgan",
        choices=["icgan", "cc_icgan"],
        help="Model type.",
    )
    parser.add_argument(
        "--model_backbone",
        type=str,
        default="biggan",
        choices=["biggan", "stylegan2"],
        help="Model backbone type.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Resolution to generate images with " "(default: %(default)s)",
    )
    parser.add_argument(
        "--z_var", type=float, default=1.0, help="Noise variance: %(default)s)"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--num_imgs_gen",
        type=int,
        default=5,
        help="Number of images to generate with different noise vectors, "
        "given an input conditioning.",
    )
    parser.add_argument(
        "--num_conditionings_gen",
        type=int,
        default=5,
        help="Number of conditionings to generate with."
        " Use `num_imgs_gen` to control the number of generated samples per conditioning",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Index of the stored instance to use as conditioning [0,1000)."
        " Mutually exclusive with `num_conditionings_gen!=1`",
    )
    parser.add_argument(
        "--swap_target",
        type=int,
        default=None,
        help="For class-conditional IC-GAN, we can choose to swap the target for a different one."
        " If swap_target=None, the original label from the instance is used. "
        "If swap_target is in [0,1000), a specific ImageNet class is used instead.",
    )

    parser.add_argument(
        "--visualize_instance_images",
        action="store_true",
        default=False,
        help="Also visualize the ground-truth image corresponding to the instance conditioning "
        "(requires a path to the ImageNet dataset)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Only needed if visualize_instance_images=True."
        " Folder where to find the dataset ground-truth images.",
    )

    config = vars(parser.parse_args())

    if config["index"] is not None and config["num_conditionings_gen"] != 1:
        raise ValueError(
            "If a specific feature vector (specificed by --index) "
            "wants to be used to sample images from, num_conditionings_gen"
            " needs to be set to 1"
        )
    if config["swap_target"] is not None and config["model"] == "icgan":
        raise ValueError(
            'Cannot specify a class label for IC-GAN! Only use "swap_target" with --model=cc_igan. '
        )
    main(config)
