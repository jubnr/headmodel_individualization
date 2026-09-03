#!/usr/bin/env python
"""Picking the scalp proxy points the warp is fitted to.

The warp has one free parameter per principal component and is fitted to a
subsample of the scalp cloud, so *which* points are picked changes the head
that comes out. Sampling this with an unseeded `np.random.choice` made the
whole pipeline non-deterministic: three runs of one scan through identical
code produced scalp meshes differing by a mean of 1.5 mm and up to 5.9 mm,
which is the same order as the total fit error.

Farthest-point sampling removes that. It is deterministic, and it spreads the
subsample over the whole surface instead of over-representing whichever region
the scanner happened to cover densely.
"""
import numpy as np


def farthest_point_sample(points, n, seed=None):
    """Return `n` indices into `points`, greedily maximising mutual distance.

    Deterministic by default: the first point is the one nearest the cloud
    centroid, so no randomness enters at all. Pass an integer `seed` to start
    from a random point instead, which is useful for estimating how much of
    the fit depends on the subsample.
    """
    points = np.asarray(points, dtype=float)
    if n >= len(points):
        return np.arange(len(points))

    if seed is None:
        start = int(np.argmin(np.linalg.norm(
            points - points.mean(axis=0), axis=1)))
    else:
        start = int(np.random.default_rng(seed).integers(len(points)))

    picked = np.empty(n, dtype=int)
    picked[0] = start
    # distance from every point to the nearest point picked so far
    nearest = np.linalg.norm(points - points[start], axis=1)
    for i in range(1, n):
        picked[i] = int(np.argmax(nearest))
        nearest = np.minimum(
            nearest, np.linalg.norm(points - points[picked[i]], axis=1))
    return picked


def subsample(points, n, method='fps', seed=None):
    """Reduce a scalp proxy cloud to `n` points.

    method 'fps'    farthest-point sampling (default, deterministic)
           'random' uniform random draw, reproducible via `seed`
    """
    points = np.asarray(points, dtype=float)
    if n >= len(points):
        return points, np.arange(len(points))
    if method == 'fps':
        idx = farthest_point_sample(points, n, seed=seed)
    elif method == 'random':
        idx = np.random.default_rng(seed).choice(len(points), n,
                                                 replace=False)
    else:
        raise ValueError(f"Unknown sampling method: {method!r}")
    return points[idx], idx


def coverage(points, subset):
    """How well `subset` covers `points`: the worst distance from any point to
    its nearest sampled neighbour, in mm. Lower is better coverage."""
    from scipy.spatial import cKDTree
    return float(cKDTree(np.asarray(subset)).query(np.asarray(points))[0].max())
