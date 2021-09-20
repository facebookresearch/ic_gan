# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Store dataset indexes of datapoints selected by k-means algorithm."""
from argparse import ArgumentParser
import numpy as np
import os
import h5py as h5
import faiss


def main(args):
    if args["which_dataset"] == "imagenet":
        dataset_name_prefix = "ILSVRC"
        im_prefix = "IN"
    elif args["which_dataset"] == "coco":
        dataset_name_prefix = "COCO"
        im_prefix = "COCO"
    else:
        dataset_name_prefix = args["which_dataset"]
        im_prefix = args["which_dataset"]
    # HDF5 filename
    filename = os.path.join(
        args["data_root"],
        "%s%s_feats_%s_%s.hdf5"
        % (
            dataset_name_prefix,
            args["resolution"],
            args["feature_extractor"],
            args["backbone_feature_extractor"],
        ),
    )
    # Load features
    print("Loading features %s..." % (filename))
    with h5.File(filename, "r") as f:
        features = f["feats"][:]
    features = np.array(features)
    # Normalize features
    features /= np.linalg.norm(features, axis=1, keepdims=True)

    feat_dim = 2048
    # k-means
    print("Training k-means with %i centers..." % (args["kmeans_subsampled"]))
    kmeans = faiss.Kmeans(
        feat_dim,
        args["kmeans_subsampled"],
        niter=100,
        verbose=True,
        gpu=args["gpu"],
        min_points_per_centroid=200,
        spherical=False,
    )
    kmeans.train(features.astype(np.float32))

    # Find closest instances to each k-means cluster
    print("Finding closest instances to centers...")
    index = faiss.IndexFlatL2(feat_dim)
    index.add(features.astype(np.float32))
    D, closest_sample = index.search(kmeans.centroids, 1)

    net_str = (
        "rn50"
        if args["backbone_feature_extractor"]
        else args["backbone_feature_extractor"]
    )
    stored_filename = "%s_res%i_%s_%s_kmeans_k%i" % (
        im_prefix,
        args["resolution"],
        net_str,
        args["feature_extractor"],
        args["kmeans_subsampled"],
    )
    np.save(
        os.path.join(args["data_root"], stored_filename),
        {"center_examples": closest_sample},
    )
    print(
        "Instance indexes resulting from a subsampling based on k-means have been saved in file %s!"
        % (stored_filename)
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Storing cluster indexes for k-means-based data subsampling"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="Data resolution (default: %(default)s)",
    )
    parser.add_argument(
        "--which_dataset", type=str, default="imagenet", help="Dataset choice."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Default location where data is stored (default: %(default)s)",
    )
    parser.add_argument(
        "--feature_extractor",
        type=str,
        default="classification",
        choices=["classification", "selfsupervised"],
        help="Choice of feature extractor",
    )
    parser.add_argument(
        "--backbone_feature_extractor",
        type=str,
        default="resnet50",
        choices=["resnet50"],
        help="Choice of feature extractor backbone",
    )
    parser.add_argument(
        "--kmeans_subsampled",
        type=int,
        default=-1,
        help="Number of k-means centers if using subsampled training instances"
             " (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Use faiss with GPUs (default: %(default)s)",
    )
    args = vars(parser.parse_args())
    main(args)
