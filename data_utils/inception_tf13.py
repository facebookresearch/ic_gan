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
#
""" Tensorflow inception score code
Derived from https://github.com/openai/improved-gan
Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
THIS CODE REQUIRES TENSORFLOW 1.3 or EARLIER to run in PARALLEL BATCH MODE 

To use this code, run sample.py on your model with --sample_npz, and then 
pass the experiment name in the --experiment_name.
This code also saves pool3 stats to an npz file for FID calculation
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile
import math
from tqdm import tqdm, trange
from argparse import ArgumentParser

import numpy as np
from six.moves import urllib
import tensorflow as tf
import pickle
import h5py as h5
import json

MODEL_DIR = "../inception_net"
DATA_URL = (
    "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"
)
softmax = None


def prepare_parser():
    usage = "Parser for TF1.3- Inception Score scripts."
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="",
        help="Which experiment" "s samples.npz file to pull and evaluate",
    )
    parser.add_argument(
        "--experiment_root",
        type=str,
        default="samples",
        help="Default location where samples are stored (default: %(default)s)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="Default overall batchsize (default: %(default)s)",
    )
    parser.add_argument(
        "--kmeans_subsampled",
        type=int,
        default=-1,
        help="Reduced number of instances to test with (using this number of centroids).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed (default: %(default)s)"
    )

    ## Ground-truth data arguments ##
    parser.add_argument(
        "--use_ground_truth_data",
        action="store_true",
        default=False,
        help="Use ground truth data to store its reference inception moments",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Default location where data is stored (default: %(default)s)",
    )
    parser.add_argument(
        "--which_dataset",
        type=str,
        default="imagenet",
        choices=["imagenet", "imagenet_lt", "coco"],
        help="Dataset choice.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="Resolution to train with " "(default: %(default)s)",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Data split (default: %(default)s)"
    )
    parser.add_argument(
        "--strat_name",
        type=str,
        default="",
        choices=["", "few", "low", "many"],
        help="Stratified split for FID in ImageNet-LT validation (default: %(default)s)",
    )
    return parser


def run(config):
    assert (
        config["strat_name"] != ""
        and config["which_dataset"] == "imagenet_lt"
        and config["split"] == "val"
    ) or config["strat_name"] == ""
    # Inception with TF1.3 or earlier.
    # Call this function with list of images. Each of elements should be a
    # numpy array with values ranging from 0 to 255.
    def get_inception_score(images, splits=10, normalize=True):
        assert type(images) == list
        assert type(images[0]) == np.ndarray
        assert len(images[0].shape) == 3
        #  assert(np.max(images[0]) > 10)
        #  assert(np.min(images[0]) >= 0.0)
        inps = []
        for img in images:
            if normalize:
                img = np.uint8(255 * (img + 1) / 2.0)
            img = img.astype(np.float32)
            inps.append(np.expand_dims(img, 0))
        bs = config["batch_size"]
        with tf.Session() as sess:
            preds, pools = [], []
            n_batches = int(math.ceil(float(len(inps)) / float(bs)))
            for i in trange(n_batches):
                inp = inps[(i * bs) : min((i + 1) * bs, len(inps))]
                inp = np.concatenate(inp, 0)
                pred, pool = sess.run([softmax, pool3], {"ExpandDims:0": inp})
                preds.append(pred)
                pools.append(pool)
            preds = np.concatenate(preds, 0)
            scores = []
            for i in range(splits):
                part = preds[
                    (i * preds.shape[0] // splits) : (
                        (i + 1) * preds.shape[0] // splits
                    ),
                    :,
                ]
                kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
                kl = np.mean(np.sum(kl, 1))
                scores.append(np.exp(kl))
            return np.mean(scores), np.std(scores), np.squeeze(np.concatenate(pools, 0))

    # Init inception
    def _init_inception():
        global softmax, pool3
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        filename = DATA_URL.split("/")[-1]
        filepath = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(filepath):

            def _progress(count, block_size, total_size):
                sys.stdout.write(
                    "\r>> Downloading %s %.1f%%"
                    % (filename, float(count * block_size) / float(total_size) * 100.0)
                )
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print("Succesfully downloaded", filename, statinfo.st_size, "bytes.")
        tarfile.open(filepath, "r:gz").extractall(MODEL_DIR)
        with tf.gfile.FastGFile(
            os.path.join(MODEL_DIR, "classify_image_graph_def.pb"), "rb"
        ) as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name="")
        # Works with an arbitrary minibatch size.
        with tf.Session() as sess:
            pool3 = sess.graph.get_tensor_by_name("pool_3:0")
            ops = pool3.graph.get_operations()
            for op_idx, op in enumerate(ops):
                for o in op.outputs:
                    shape = o.get_shape()
                    shape = [s.value for s in shape]
                    new_shape = []
                    for j, s in enumerate(shape):
                        if s == 1 and j == 0:
                            new_shape.append(None)
                        else:
                            new_shape.append(s)
                    o.__dict__["_shape_val"] = tf.TensorShape(new_shape)
            w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
            logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
            softmax = tf.nn.softmax(logits)

    # if softmax is None: # No need to functionalize like this.
    _init_inception()

    if config["use_ground_truth_data"]:
        # HDF5 file name
        if config["which_dataset"] in ["imagenet", "imagenet_lt"]:
            dataset_name_prefix = "ILSVRC"
        elif config["which_dataset"] == "coco":
            dataset_name_prefix = "COCO"
        hdf5_filename = "%s%i%s%s%s_xy.hdf5" % (
            dataset_name_prefix,
            config["resolution"],
            "longtail"
            if config["which_dataset"] == "imagenet_lt" and config["split"] == "train"
            else "",
            "_val" if config["split"] == "val" else "",
            "_test"
            if config["split"] == "val" and config["which_dataset"] == "coco"
            else "",
        )
        with h5.File(os.path.join(config["data_root"], hdf5_filename), "r") as f:
            data_imgs = f["imgs"][:]
            data_labels = f["labels"][:]
        ims = data_imgs.transpose(0, 2, 3, 1)

    else:
        if config["strat_name"] != "":
            fname = "%s/%s/samples%s_seed%i_strat_%s.pickle" % (
                config["experiment_root"],
                config["experiment_name"],
                "_kmeans" + str(config["kmeans_subsampled"])
                if config["kmeans_subsampled"] > -1
                else "",
                config["seed"],
                config["strat_name"],
            )
        else:
            fname = "%s/%s/samples%s_seed%i.pickle" % (
                config["experiment_root"],
                config["experiment_name"],
                "_kmeans" + str(config["kmeans_subsampled"])
                if config["kmeans_subsampled"] > -1
                else "",
                config["seed"],
            )
        print("loading %s ..." % fname)
        file_to_read = open(fname, "rb")
        ims = pickle.load(file_to_read)["x"]
        print("loading %s ..." % fname)
        print("number of images saved are ", len(ims))
        file_to_read.close()
        ims = ims.swapaxes(1, 2).swapaxes(2, 3)

    import time

    t0 = time.time()
    inc_mean, inc_std, pool_activations = get_inception_score(
        list(ims), splits=10, normalize=not config["use_ground_truth_data"]
    )
    t1 = time.time()
    print("Saving pool to numpy file for FID calculations...")
    mu = np.mean(pool_activations, axis=0)
    sigma = np.cov(pool_activations, rowvar=False)
    if config["use_ground_truth_data"]:
        np.savez(
            "%s/%s%s_res%i_tf_inception_moments_ground_truth.npz"
            % (
                config["data_root"],
                config["which_dataset"],
                "_val" if config["split"] == "val" else "",
                config["resolution"],
            ),
            **{"mu": mu, "sigma": sigma}
        )
    else:
        np.savez(
            "%s/%s/TF_pool%s_%s.npz"
            % (
                config["experiment_root"],
                config["experiment_name"],
                "_val" if config["split"] == "val" else "",
                "_strat_" + config["strat_name"] if config["strat_name"] != "" else "",
            ),
            **{"mu": mu, "sigma": sigma}
        )
    print(
        "Inception took %3f seconds, score of %3f +/- %3f."
        % (t1 - t0, inc_mean, inc_std)
    )

    # If ground-truth data moments, also compute the moments for stratified FID.
    if (
        config["split"] == "val"
        and config["which_dataset"] == "imagenet_lt"
        and config["use_ground_truth_data"]
    ):
        samples_per_class = np.load(
            "BigGAN_PyTorch/imagenet_lt/imagenet_lt_samples_per_class.npy",
            allow_pickle=True,
        )
        for strat_name in ["_many", "_low", "_few"]:
            if strat_name == "_many":
                pool_ = pool_activations[samples_per_class[data_labels] >= 100]
            elif strat_name == "_low":
                pool_ = pool_activations[samples_per_class[data_labels] < 100]
                labels_ = data_labels[samples_per_class[data_labels] < 100]
                pool_ = pool_[samples_per_class[labels_] > 20]
            elif strat_name == "_few":
                pool_ = pool_activations[samples_per_class[data_labels] <= 20]
            print("Size for strat ", strat_name, " is ", len(pool_))
            mu = np.mean(pool_, axis=0)
            sigma = np.cov(pool_, rowvar=False)

            np.savez(
                "%s/%s%s_res%i_tf_inception_moments%s_ground_truth.npz"
                % (
                    config["data_root"],
                    config["which_dataset"],
                    "_val" if config["split"] == "val" else "",
                    config["resolution"],
                    strat_name,
                ),
                **{"mu": mu, "sigma": sigma}
            )


def main():
    # parse command line and run
    parser = prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == "__main__":
    main()
