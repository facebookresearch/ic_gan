# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
import tempfile
import warnings
from pathlib import Path
import nltk
import torch
from torch import nn
import torchvision.transforms as transforms
import numpy as np
import imageio
from PIL import Image as Image_PIL
from scipy.stats import truncnorm
from nltk.corpus import wordnet as wn
import cma
import sklearn.metrics
import cog

sys.path.insert(0, "stylegan2_ada_pytorch")
from pytorch_pretrained_biggan import convert_to_images, utils
import inference.utils as inference_utils
import data_utils.utils as data_utils

NORM_MEAN = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
NORM_STD = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

nltk.download("wordnet")
IND2NAME = {
    index: wn.of2ss("%08dn" % offset).lemma_names()[0]
    for offset, index in utils.IMAGENET.items()
}
NAME2IND = dict([(value, key) for key, value in IND2NAME.items()])

CLASS_NAMES = sorted(list(IND2NAME.values()))


class Predictor(cog.Predictor):
    def setup(self):
        torch.manual_seed(np.random.randint(sys.maxsize))
        warnings.simplefilter("ignore", cma.evolution_strategy.InjectionWarning)
        self.last_gen_model = None
        self.last_feature_extractor = None
        self.model = None
        self.feature_extractor = None
        self.noise_size = 128
        self.batch_size = 4
        self.size = 256

    @cog.input("image", type=Path, help="Input image Instance")
    @cog.input("gen_model", type=str, options=["icgan", "cc_icgan"], default="icgan",
               help='Select type of IC-GAN model. "icgan" is conditioned on the input image; '
                    '"cc_icgan" is conditioned on both input image and a conditional_class')
    @cog.input("conditional_class", type=str, default=None, options=CLASS_NAMES,
               help="Choose conditional class. Only valid for gen_model=cc_icgan")
    @cog.input("num_samples", type=int, default=1, options=[1, 4, 9, 16],
               help="number of samples generated")
    @cog.input("seed", type=int, default=0, help="seed=0 means no seed")
    def predict(self, image, gen_model="icgan", conditional_class=None, num_samples=1, seed=0):
        assert isinstance(seed, int), "seed should be an integer"
        if gen_model == 'cc_icgan':
            assert conditional_class is not None, 'please set conditional_class for cc_icgan'
        num_samples_ranked = num_samples
        experiment_name = (
            "icgan_biggan_imagenet_res256"
            if gen_model == "icgan"
            else "cc_icgan_biggan_imagenet_res256"
        )
        num_samples_total = num_samples * 10
        truncation = 0.7
        if conditional_class is not None:
            class_index = NAME2IND[conditional_class]

        input_image_instance = str(image)

        if gen_model == "icgan":
            class_index = None

        if seed == 0:
            seed = None

        state = None if not seed else np.random.RandomState(seed)
        np.random.seed(seed)

        feature_extractor_name = ("classification" if gen_model == "cc_icgan" else "selfsupervised")

        # Load feature extractor (outlier filtering and optionally input image feature extraction)
        self.feature_extractor, self.last_feature_extractor = load_feature_extractor(
            gen_model, self.last_feature_extractor, self.feature_extractor)
        # Load features
        if input_image_instance not in ["None", "", None]:
            print("Obtaining instance features from input image!")
            input_feature_index = None
            input_image_tensor = preprocess_input_image(input_image_instance, self.size)
            with torch.no_grad():
                input_features, _ = self.feature_extractor(input_image_tensor.cuda())
            input_features /= torch.linalg.norm(input_features, dim=-1, keepdims=True)
        elif input_feature_index is not None:
            print("Selecting an instance from pre-extracted vectors!")
            input_features = np.load(
                "stored_instances/imagenet_res"
                + str(self.size)
                + "_rn50_"
                + feature_extractor_name
                + "_kmeans_k1000_instance_features.npy",
                allow_pickle=True,
            ).item()["instance_features"][input_feature_index: input_feature_index + 1]
        else:
            input_features = None

        # Load generative model
        self.model, self.last_gen_model = load_generative_model(
            gen_model, self.last_gen_model, experiment_name, self.model)
        # Prepare other variables

        replace_to_inplace_relu(self.model)

        # Create noise, instance and class vector
        noise_vector = truncnorm.rvs(
            -2 * truncation,
            2 * truncation,
            size=(num_samples_total, self.noise_size),
            random_state=state,
        ).astype(np.float32)
        noise_vector = torch.tensor(noise_vector, requires_grad=False, device="cuda")
        if input_features is not None:
            instance_vector = torch.tensor(
                input_features, requires_grad=False, device="cuda"
            ).repeat(num_samples_total, 1)
        else:
            instance_vector = None
        if class_index is not None:
            input_label = torch.LongTensor([class_index] * num_samples_total)
        else:
            input_label = None
        if input_feature_index is not None:
            print("Conditioning on instance with index: ", input_feature_index)

        all_outs, all_dists = [], []
        for i_bs in range(num_samples_total // self.batch_size + 1):
            start = i_bs * self.batch_size
            end = min(start + self.batch_size, num_samples_total)
            if start == end:
                break
            out = get_output(
                noise_vector[start:end],
                input_label[start:end] if input_label is not None else None,
                instance_vector[start:end] if instance_vector is not None else None,
                self.model,
                truncation,
                channels=3,
            )

            if instance_vector is not None:
                # Get features from generated images + feature extractor
                out_ = preprocess_generated_image(out)
                with torch.no_grad():
                    out_features, _ = self.feature_extractor(out_.cuda())
                out_features /= torch.linalg.norm(out_features, dim=-1, keepdims=True)
                dists = sklearn.metrics.pairwise_distances(
                    out_features.cpu(),
                    instance_vector[start:end].cpu(),
                    metric="euclidean",
                    n_jobs=-1,
                )
                all_dists.append(np.diagonal(dists))
                all_outs.append(out.detach().cpu())
            del out
        all_outs = torch.cat(all_outs)
        all_dists = np.concatenate(all_dists)

        # Order samples by distance to conditioning feature vector and select only num_samples_ranked images
        selected_idxs = np.argsort(all_dists)[:num_samples_ranked]
        # Create figure
        row_i, col_i, i_im = 0, 0, 0
        all_images_mosaic = np.zeros(
            (
                3,
                self.size * (int(np.sqrt(num_samples_ranked))),
                self.size * (int(np.sqrt(num_samples_ranked))),
            )
        )
        for j in selected_idxs:
            all_images_mosaic[
            :,
            row_i * self.size: row_i * self.size + self.size,
            col_i * self.size: col_i * self.size + self.size,
            ] = all_outs[j]
            if row_i == int(np.sqrt(num_samples_ranked)) - 1:
                row_i = 0
                if col_i == int(np.sqrt(num_samples_ranked)) - 1:
                    col_i = 0
                else:
                    col_i += 1
            else:
                row_i += 1
            i_im += 1

        out_path = Path(tempfile.mkdtemp()) / "out.png"
        save(all_images_mosaic[np.newaxis, ...], str(out_path), torch_format=False)
        return out_path


def replace_to_inplace_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.ReLU(inplace=False))
        else:
            replace_to_inplace_relu(child)


def save(out, name=None, torch_format=True):
    if torch_format:
        with torch.no_grad():
            out = out.cpu().numpy()
    img = convert_to_images(out)[0]
    if name:
        imageio.imwrite(name, np.asarray(img))
    return img


def load_icgan(experiment_name, root_=""):
    root = os.path.join(root_, experiment_name)
    config = torch.load("%s/%s.pth" % (root, "state_dict_best0"))["config"]

    config["weights_root"] = root_
    config["model_backbone"] = "biggan"
    config["experiment_name"] = experiment_name
    G, config = inference_utils.load_model_inference(config)
    G.cuda()
    G.eval()
    return G


def get_output(noise_vector, input_label, input_features, model, truncation, channels):
    # stochastic_truncation = False as how it is set in colab
    noise_vector = noise_vector.clamp(-2 * truncation, 2 * truncation)
    if input_label is not None:
        input_label = torch.LongTensor(input_label)
    else:
        input_label = None

    out = model(
        noise_vector,
        input_label.cuda() if input_label is not None else None,
        input_features.cuda() if input_features is not None else None,
    )

    if channels == 1:
        out = out.mean(dim=1, keepdim=True)
        out = out.repeat(1, 3, 1, 1)
    return out


def load_generative_model(gen_model, last_gen_model, experiment_name, model):
    # Load generative model
    if gen_model != last_gen_model:
        model = load_icgan(experiment_name, root_="./")
        last_gen_model = gen_model
    return model, last_gen_model


def load_feature_extractor(gen_model, last_feature_extractor, feature_extractor):
    # Load feature extractor to obtain instance features
    feat_ext_name = "classification" if gen_model == "cc_icgan" else "selfsupervised"
    if last_feature_extractor != feat_ext_name:
        if feat_ext_name == "classification":
            feat_ext_path = ""
        else:
            feat_ext_path = "swav_pretrained.pth.tar"
        last_feature_extractor = feat_ext_name
        feature_extractor = data_utils.load_pretrained_feature_extractor(
            feat_ext_path, feature_extractor=feat_ext_name
        )
        feature_extractor.eval()
    return feature_extractor, last_feature_extractor


def preprocess_input_image(input_image_path, size):
    pil_image = Image_PIL.open(input_image_path).convert("RGB")
    transform_list = transforms.Compose(
        [
            data_utils.CenterCropLongEdge(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )
    tensor_image = transform_list(pil_image)
    tensor_image = torch.nn.functional.interpolate(
        tensor_image.unsqueeze(0), 224, mode="bicubic", align_corners=True
    )
    return tensor_image


def preprocess_generated_image(image):
    transform_list = transforms.Normalize(NORM_MEAN, NORM_STD)
    image = transform_list(image * 0.5 + 0.5)
    image = torch.nn.functional.interpolate(
        image, 224, mode="bicubic", align_corners=True
    )
    return image
