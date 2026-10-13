#!/usr/bin/env python
"""Individualize an age-matched infant template to a scalp scan.

The PCA bases in data/ were trained on adult heads, and infant heads lie
outside them: a different size, different proportions, and a different
relation between the scalp and the shells beneath it. Instead of a learned
shape space this module starts from the infant MRI template closest in age
(O'Reilly et al. 2021, built on the Richards et al. 2016 database, fetched
through mne.datasets.fetch_infant_template) and deforms it onto the scan:

1. both are put into the CTF frame by their own NAS/LPA/RPA,
2. an affine is fitted to the scan by ICP, which takes care of head size and
   proportions,
3. the remaining scalp mismatch is interpolated by a smoothed thin-plate
   spline and applied to every shell, so the template's scalp-to-skull and
   skull-to-brain distances are carried over rather than invented,
4. shells that come to touch are pulled back apart.

The template ships three closed BEM surfaces. They are named like the outer
three PCAwarp shells, as export_mne already maps them: scalp = outer skin,
skull = outer skull, csf = inner skull.
"""
import os

import numpy as np
import trimesh
from scipy.interpolate import RBFInterpolator

from src.pca_warp import enforce_nesting
from src.sampling import subsample
from src.transform_to_ctf import get_transform, apply_transform
from src.tri_io import orient_outward


AGES = ('2wk', '1mo', '2mo', '3mo', '4.5mo', '6mo', '7.5mo', '9mo', '10.5mo',
        '12mo', '15mo', '18mo', '2yr')
SHELLS = ['scalp', 'skull', 'csf']
SURFACES = {'scalp': 'outer_skin.surf',
            'skull': 'outer_skull.surf',
            'csf': 'inner_skull.surf'}

# The adult pipeline cuts 30 mm above the ear plane, about a fifth of an adult
# LPA-RPA distance. The infant cut keeps that proportion.
CUT_FRACTION = 0.2
# Landmark spacing accepted relative to the template's own.
PLAUSIBLE_FACTOR = (0.75, 1.35)


# =====================
# Template
# =====================
def default_subjects_dir():
    """A folder of its own, not MNE's SUBJECTS_DIR: the templates are large
    and should not land unasked among the user's own subjects."""
    return os.path.join(os.path.expanduser('~'), 'mne_data', 'infant')


def load_template(age, subjects_dir=None):
    """The three BEM shells and NAS/LPA/RPA of one template, in mm.

    Everything is in the template's FreeSurfer surface RAS. Downloads the
    template on first use (a few hundred MB per age).
    """
    if age not in AGES:
        raise ValueError(f'Unknown template age {age!r}; one of '
                         f'{", ".join(AGES)}.')
    import mne
    import nibabel as nib

    subjects_dir = subjects_dir or default_subjects_dir()
    os.makedirs(subjects_dir, exist_ok=True)
    subject = mne.datasets.fetch_infant_template(age,
                                                 subjects_dir=subjects_dir,
                                                 verbose=False)
    subject_dir = os.path.join(subjects_dir, subject)

    bnd = {}
    for shell, fname in SURFACES.items():
        pos, tris = nib.freesurfer.read_geometry(
            os.path.join(subject_dir, 'bem', fname))
        pos = np.asarray(pos, dtype=float)
        bnd[shell] = (pos, orient_outward(pos, tris))

    return {'age': age, 'subject': subject, 'subjects_dir': subjects_dir,
            'subject_dir': subject_dir, 'bnd': bnd,
            'fiducials': read_template_fiducials(subject_dir, subject)}


def read_template_fiducials(subject_dir, subject):
    """NAS, LPA, RPA from bem/<subject>-fiducials.fif, in mm."""
    import mne
    from mne.io.constants import FIFF

    points, frame = mne.io.read_fiducials(
        os.path.join(subject_dir, 'bem', f'{subject}-fiducials.fif'))
    if frame != FIFF.FIFFV_COORD_MRI:
        raise ValueError(f'{subject} fiducials are not in the MRI frame.')
    by_ident = {int(p['ident']): np.asarray(p['r'], dtype=float) * 1000.0
                for p in points}
    return np.array([by_ident[FIFF.FIFFV_POINT_NASION],
                     by_ident[FIFF.FIFFV_POINT_LPA],
                     by_ident[FIFF.FIFFV_POINT_RPA]])


def to_ctf(template):
    """The template shells in its own CTF frame, and the RAS->CTF transform."""
    transform = get_transform(*np.array(template['fiducials'], dtype=float))
    bnd = {shell: (apply_transform(transform, pos), tris)
           for shell, (pos, tris) in template['bnd'].items()}
    return bnd, transform


def landmark_spacing(fiducials):
    nas, lpa, rpa = np.asarray(fiducials, dtype=float)
    return {'lpa-rpa': float(np.linalg.norm(lpa - rpa)),
            'nas-lpa': float(np.linalg.norm(nas - lpa)),
            'nas-rpa': float(np.linalg.norm(nas - rpa))}


def plausible_ranges(fiducials, factor=PLAUSIBLE_FACTOR):
    """Accepted landmark spacing for heads of the template's age, in mm."""
    return {k: (v * factor[0], v * factor[1])
            for k, v in landmark_spacing(fiducials).items()}


def ear_plane_cut(fiducials):
    """Height above the ear plane below which scan points are dropped, mm."""
    return CUT_FRACTION * landmark_spacing(fiducials)['lpa-rpa']


# =====================
# Warp
# =====================
def _mesh(bnd_shell):
    pos, tris = bnd_shell
    return trimesh.Trimesh(np.asarray(pos, dtype=float), np.asarray(tris),
                           process=False)


def apply_affine(matrix, bnd):
    return {shell: (apply_transform(matrix, pos), tris)
            for shell, (pos, tris) in bnd.items()}


def fit_affine(scalp, points, n_iter=50, tol=1e-4):
    """12-parameter affine taking the template scalp onto the points (ICP).

    Each point is paired with its closest point on the current scalp, the
    pair is mapped back onto the untransformed scalp through the inverse of
    the current affine, and the affine is re-solved by least squares. Points
    to surface rather than surface to points, because a scan of an EEG cap
    covers only part of the template.
    Returns the 4x4 matrix and the per-iteration median distance.
    """
    pos, tris = scalp
    points = np.asarray(points, dtype=float)
    matrix = np.eye(4)
    history = []
    hom_points = np.c_[points, np.ones(len(points))]
    for _ in range(n_iter):
        current = _mesh((apply_transform(matrix, pos), tris))
        closest, dist, _ = trimesh.proximity.closest_point(current, points)
        history.append(float(np.median(dist)))
        if len(history) > 1 and history[-2] - history[-1] < tol:
            break
        source = apply_transform(np.linalg.inv(matrix), closest)
        hom_source = np.c_[source, np.ones(len(source))]
        solution, *_ = np.linalg.lstsq(hom_source, hom_points, rcond=None)
        matrix = solution.T
        matrix[3] = [0.0, 0.0, 0.0, 1.0]
    return matrix, history


def affine_scales(matrix):
    """Singular values of the linear part: the stretch along the principal
    axes. Far from 1 means the template is the wrong age for this head."""
    return np.linalg.svd(np.asarray(matrix)[:3, :3], compute_uv=False)


def fit_nonrigid(bnd, points, cut, smoothing=1.0, n_anchors=150,
                 anchor_margin=10.0):
    """Move every shell by a thin-plate spline fitted to the scalp mismatch.

    Control points are the closest scalp points to each scan point, displaced
    onto that scan point. Scalp vertices more than `anchor_margin` mm below
    the cut, where there is no scan data, are pinned in place so the spline
    does not extrapolate across the face and neck. Returns the deformed shells
    and the displacement at the control points.
    """
    points = np.asarray(points, dtype=float)
    closest, _, _ = trimesh.proximity.closest_point(_mesh(bnd['scalp']),
                                                    points)
    shift = points - closest

    scalp_pos = np.asarray(bnd['scalp'][0], dtype=float)
    low = scalp_pos[scalp_pos[:, 2] < cut - anchor_margin]
    anchors, _ = subsample(low, n_anchors) if len(low) else (low, None)

    spline = RBFInterpolator(np.vstack([closest, anchors]),
                             np.vstack([shift, np.zeros_like(anchors)]),
                             kernel='thin_plate_spline', smoothing=smoothing)
    out = {shell: (pos + spline(pos), tris)
           for shell, (pos, tris) in bnd.items()}
    return out, shift


def restore_nesting(bnd, margin=0.5):
    """Pull each shell back inside the one around it. Returns counts moved."""
    moved = {}
    for outer, inner in zip(SHELLS[:-1], SHELLS[1:]):
        bnd[inner], moved[inner] = enforce_nesting(bnd, outer, inner,
                                                   margin=margin)
    return moved


def warp(template_ctf, points_fit, cut, smoothing=1.0):
    """Affine then thin-plate warp of the template shells onto the points.

    All in one CTF frame. Returns the affine-only shells, the final shells and
    a dict describing the fit.
    """
    matrix, history = fit_affine(template_ctf['scalp'], points_fit)
    bnd_affine = apply_affine(matrix, template_ctf)
    bnd, shift = fit_nonrigid(bnd_affine, points_fit, cut,
                              smoothing=smoothing)
    moved = restore_nesting(bnd)
    bnd = {shell: (pos, orient_outward(pos, tris))
           for shell, (pos, tris) in bnd.items()}
    info = {'affine': matrix.tolist(),
            'affine_scales': affine_scales(matrix).tolist(),
            'affine_icp_median_mm': history,
            'spline_shift_median_mm': float(np.median(
                np.linalg.norm(shift, axis=1))),
            'spline_shift_max_mm': float(np.linalg.norm(shift,
                                                        axis=1).max()),
            'nesting_vertices_moved': moved}
    return bnd_affine, bnd, info
