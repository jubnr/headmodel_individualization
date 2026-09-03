"""Mesh conventions and shell nesting.

Orientation and nesting are the two things every downstream consumer
(OpenMEEG, FieldTrip, MNE) silently depends on. MNE refuses a BEM whose
normals point inward, and a crossing between shells makes the forward model
meaningless.
"""
import numpy as np
import pytest

from src.pca_warp import (degenerate_shells, enforce_nesting,
                          transfer_displacement)
from src.tri_io import orient_outward, signed_volume


# =====================
# Orientation
# =====================
def test_signed_volume_matches_the_analytic_sphere(sphere):
    verts, tris = sphere(radius=10.0, subdivisions=5)
    # a discretised sphere slightly under-estimates the true volume
    assert signed_volume(verts, tris) == pytest.approx(
        4 / 3 * np.pi * 10.0 ** 3, rel=2e-3)


def test_signed_volume_flips_with_the_winding(sphere):
    verts, tris = sphere()
    assert signed_volume(verts, tris) == pytest.approx(
        -signed_volume(verts, tris[:, ::-1]))


def test_signed_volume_is_origin_independent(sphere):
    verts, tris = sphere(radius=10.0)
    shifted = verts + np.array([500.0, -300.0, 120.0])
    assert signed_volume(verts, tris) == pytest.approx(
        signed_volume(shifted, tris), rel=1e-9)


@pytest.mark.parametrize('flip', [False, True])
def test_orient_outward_always_gives_positive_volume(sphere, flip):
    verts, tris = sphere()
    if flip:
        tris = tris[:, ::-1]
    assert signed_volume(verts, orient_outward(verts, tris)) > 0


def test_orient_outward_leaves_a_correct_mesh_untouched(sphere):
    verts, tris = sphere()
    good = orient_outward(verts, tris)
    assert np.array_equal(good, orient_outward(verts, good))


# =====================
# Degenerate PCA blocks
# =====================
def test_degenerate_shells_spots_an_all_zero_block(nested_shells):
    shells = nested_shells()
    total = sum(len(v[0]) for v in shells.values())
    pcas = np.random.default_rng(0).normal(size=(4, total, 3))
    assert degenerate_shells(pcas, shells) == []

    start = sum(len(shells[s][0]) for s in ['scalp', 'skull', 'csf'])
    pcas[:, start:, :] = 0.0
    assert degenerate_shells(pcas, shells) == ['cortex']


def test_the_shipped_bases_report_what_we_expect(repo):
    """pcas_hartmut ships without cortex variance; pcas does not."""
    import os
    for name, expected in [('pcas', []), ('pcas_hartmut', ['cortex'])]:
        base = os.path.join(repo, 'data', name)
        mean = np.load(os.path.join(base, 'mean_head.npy'),
                       allow_pickle=True).item()
        pcas = np.load(os.path.join(base, 'ALLpcas.npy'),
                       allow_pickle=True)[:16]
        assert degenerate_shells(pcas, mean) == expected


# =====================
# Displacement transfer
# =====================
def test_transfer_displacement_moves_the_shell_by_the_donor_shift(
        nested_shells):
    mean = nested_shells()
    warped = {k: (v[0].copy(), v[1]) for k, v in mean.items()}
    shift = np.array([3.0, -1.0, 2.0])
    warped['csf'] = (mean['csf'][0] + shift, mean['csf'][1])

    pos, tris = transfer_displacement(warped, mean, 'csf', 'cortex')
    assert np.allclose(pos, mean['cortex'][0] + shift)
    assert np.array_equal(tris, mean['cortex'][1])


def test_transfer_displacement_refuses_mismatched_triangulations(
        nested_shells, sphere):
    mean = nested_shells()
    mean['cortex'] = sphere(radius=26.0, subdivisions=2)   # different mesh
    with pytest.raises(ValueError, match='triangulation'):
        transfer_displacement(mean, mean, 'csf', 'cortex')


# =====================
# Nesting
# =====================
def test_enforce_nesting_is_a_no_op_when_shells_are_clear(nested_shells):
    shells = nested_shells()
    _, moved = enforce_nesting(shells, 'csf', 'cortex')
    assert moved == 0


def test_enforce_nesting_pulls_a_protruding_shell_back_inside(sphere):
    """An inner shell deliberately made to burst through the outer one."""
    shells = {'csf': sphere(radius=30.0, subdivisions=3),
              'cortex': sphere(radius=30.0, subdivisions=3)}
    verts = shells['cortex'][0].copy()
    verts[:50] *= 1.4                       # 50 vertices punched outward
    shells['cortex'] = (verts, shells['cortex'][1])

    fixed, moved = enforce_nesting(shells, 'csf', 'cortex', margin=0.5)
    assert moved >= 50
    radii = np.linalg.norm(fixed[0], axis=1)
    assert radii.max() < 30.0, 'a vertex still sticks out of the csf'


def test_enforce_nesting_keeps_the_requested_margin(sphere):
    shells = {'csf': sphere(radius=30.0, subdivisions=3),
              'cortex': sphere(radius=29.9, subdivisions=3)}
    fixed, moved = enforce_nesting(shells, 'csf', 'cortex', margin=1.0)
    assert moved > 0
    # every moved vertex sits ~1 mm inside the csf sphere
    assert np.linalg.norm(fixed[0], axis=1).max() < 29.5
