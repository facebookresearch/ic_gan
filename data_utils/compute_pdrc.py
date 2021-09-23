# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# prdc
# Copyright (c) 2020-present NAVER Corp.
# MIT license

import numpy as np
import sklearn.metrics

__all__ = ["compute_prdc"]


def compute_pairwise_distance(data_x, data_y=None):
    """
    Parameters
    ----------
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns
    -------
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric="euclidean", n_jobs=8
    )
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Parameters
    ----------
        unsorted: numpy.ndarray of any dimensionality.
        k: int
        axis: int
    Returns
    -------
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Parameters
    ----------
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns
    -------
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Parameters
    ----------
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns
    -------
        dict of precision, recall, density, and coverage.
    """

    print(
        "Num real: {} Num fake: {}".format(
            real_features.shape[0], fake_features.shape[0]
        )
    )

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k
    )
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k
    )
    distance_real_fake = compute_pairwise_distance(real_features, fake_features)

    precision = (
        (distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1))
        .any(axis=0)
        .mean()
    )

    recall = (
        (distance_real_fake < np.expand_dims(fake_nearest_neighbour_distances, axis=0))
        .any(axis=1)
        .mean()
    )

    density = (1.0 / float(nearest_k)) * (
        distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
        distance_real_fake.min(axis=1) < real_nearest_neighbour_distances
    ).mean()

    return dict(precision=precision, recall=recall, density=density, coverage=coverage)
