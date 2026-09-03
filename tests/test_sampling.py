"""Which scalp points the warp is fitted to decides what head comes out.

Before farthest-point sampling this was an unseeded np.random.choice, and
three runs of one scan through identical code produced scalp meshes differing
by a mean of 1.5 mm and up to 5.9 mm - the same order as the total fit error.
"""
import numpy as np
import pytest

from src.sampling import coverage, farthest_point_sample, subsample


@pytest.fixture
def cloud(sphere):
    """An unevenly sampled head-sized surface: dense on one side, sparse on
    the other, which is what a real scan looks like."""
    verts, _ = sphere(radius=90.0, subdivisions=4)
    dense = verts[verts[:, 0] > 0]
    return np.vstack([verts, dense, dense])


def test_fps_is_deterministic(cloud):
    assert np.array_equal(farthest_point_sample(cloud, 100),
                          farthest_point_sample(cloud, 100))


def test_fps_needs_no_seed_to_repeat(cloud):
    """Determinism must not depend on the caller remembering a seed."""
    first, _ = subsample(cloud, 60)
    second, _ = subsample(cloud, 60)
    assert np.array_equal(first, second)


def test_fps_covers_better_than_a_random_draw(cloud):
    """The point of FPS: it does not over-represent the dense side."""
    fps, _ = subsample(cloud, 100, method='fps')
    worst_random = min(coverage(cloud, subsample(cloud, 100,
                                                 method='random', seed=s)[0])
                       for s in range(5))
    assert coverage(cloud, fps) < worst_random


def test_random_sampling_is_reproducible_given_a_seed(cloud):
    first, _ = subsample(cloud, 50, method='random', seed=7)
    second, _ = subsample(cloud, 50, method='random', seed=7)
    assert np.array_equal(first, second)


def test_different_seeds_give_different_random_draws(cloud):
    first, _ = subsample(cloud, 50, method='random', seed=1)
    second, _ = subsample(cloud, 50, method='random', seed=2)
    assert not np.array_equal(first, second)


def test_asking_for_more_points_than_exist_returns_everything(cloud):
    points, idx = subsample(cloud, len(cloud) + 10)
    assert len(points) == len(cloud)
    assert np.array_equal(idx, np.arange(len(cloud)))


def test_sampled_points_come_from_the_cloud(cloud):
    points, idx = subsample(cloud, 40)
    assert np.array_equal(points, cloud[idx])


def test_unknown_method_is_rejected(cloud):
    with pytest.raises(ValueError, match='Unknown sampling method'):
        subsample(cloud, 10, method='every-other-one')
