"""The infant path: warping an age-matched template instead of the adult PCA.

The fast tests use spheres, so they need no download. The slow ones fetch the
6-month template (about 600 MB, once) and run the CLI on a scan made from it
by a known deformation, so the true inner shells are known too.
"""
import json
import subprocess
import sys

import numpy as np
import pytest
import trimesh

from src import fiducials as fid
from src import infant_template as infant
from src import qc
from src.transform_to_ctf import apply_transform


def _stretch(sx, sy, sz, shift=(0.0, 0.0, 0.0)):
    matrix = np.diag([sx, sy, sz, 1.0])
    matrix[:3, 3] = shift
    return matrix


def _bump(pos, centre, height=5.0, width=12.0):
    """Push points outward, radially, around `centre`."""
    pos = np.asarray(pos, dtype=float)
    weight = height * np.exp(-np.sum((pos - centre) ** 2, axis=1)
                             / (2 * width ** 2))
    radial = pos / np.linalg.norm(pos, axis=1)[:, None]
    return pos + weight[:, None] * radial


@pytest.fixture
def shells(sphere):
    """Three concentric spheres the size of a 6-month head."""
    return {'scalp': sphere(radius=60.0, subdivisions=4),
            'skull': sphere(radius=55.0, subdivisions=4),
            'csf': sphere(radius=53.0, subdivisions=4)}


# =====================
# Warp
# =====================
def test_affine_recovers_a_known_stretch(shells):
    truth = _stretch(1.15, 0.9, 1.05, shift=(3.0, -2.0, 1.0))
    points = apply_transform(truth, shells['scalp'][0])
    points = points[points[:, 2] > 10]  # a cap, not the whole head
    matrix, history = infant.fit_affine(shells['scalp'], points)
    assert history[-1] < 0.3
    # A sphere is symmetric, so the matrix itself is not unique (A @ R maps
    # it onto the same ellipsoid for any rotation R). The surface is.
    pos, tris = shells['scalp']
    truth_mesh = trimesh.Trimesh(apply_transform(truth, pos), tris,
                                 process=False)
    fitted = apply_transform(matrix, pos)[pos[:, 2] > 20]
    _, dist, _ = trimesh.proximity.closest_point(truth_mesh, fitted)
    assert np.median(dist) < 0.1 and dist.max() < 0.5
    # a cap leaves height and vertical shift partly interchangeable, so the
    # stretch is only known roughly
    assert np.allclose(sorted(infant.affine_scales(matrix)),
                       [0.9, 1.05, 1.15], atol=0.05)


def test_spline_fits_a_bump_and_leaves_the_anchors_alone(shells):
    target = _bump(shells['scalp'][0], centre=[0.0, 0.0, 62.0])
    points = target[target[:, 2] > 12]
    cut = 12.0
    out, _ = infant.fit_nonrigid(shells, points, cut=cut)
    before = qc.scalp_fit(shells, points)['warped']['median_abs']
    after = qc.scalp_fit(out, points)['warped']['median_abs']
    assert after < 0.1 * before, (before, after)

    low = shells['scalp'][0][:, 2] < cut - 10  # the pinned region
    moved = np.linalg.norm(out['scalp'][0][low] - shells['scalp'][0][low],
                           axis=1)
    assert np.median(moved) < 0.1


def test_warped_shells_stay_nested_and_closed(shells):
    target = apply_transform(_stretch(1.1, 0.95, 1.0), shells['scalp'][0])
    target = _bump(target, centre=[0.0, 40.0, 45.0], height=8.0)
    points = target[target[:, 2] > 12]
    _, bnd, info = infant.warp(shells, points, cut=12)
    report, problems = qc.run(bnd, points, template_scalp=shells['scalp'],
                              make_figure=False)
    assert problems == [], problems
    assert set(info['nesting_vertices_moved']) == {'skull', 'csf'}


def test_restore_nesting_pulls_a_crossing_shell_back(shells):
    bnd = dict(shells)
    pos, tris = bnd['skull']
    bnd['skull'] = (pos * 1.12, tris)  # now outside the scalp
    moved = infant.restore_nesting(bnd)
    assert moved['skull'] > 0
    assert qc.nesting(bnd)['skull_in_scalp']['n_crossing'] == 0


# =====================
# Size checks
# =====================
@pytest.fixture
def infant_fiducials():
    """NAS/LPA/RPA of the 6-month template (LPA-RPA 109 mm)."""
    return np.array([[2.26, 66.28, -13.87],
                     [-55.25, -2.52, -34.28],
                     [53.76, -4.52, -32.93]])


def test_infant_spacing_needs_the_scaled_ranges(infant_fiducials):
    with pytest.raises(ValueError, match='Implausible'):
        fid.validate(infant_fiducials)  # adult ranges
    plausible = infant.plausible_ranges(infant_fiducials)
    got = fid.validate(infant_fiducials, plausible=plausible)
    assert got['lpa-rpa'] == pytest.approx(109.0, abs=0.1)


def test_scaled_ranges_still_reject_an_adult(infant_fiducials, fiducials_mm):
    with pytest.raises(ValueError, match='Implausible'):
        fid.validate(fiducials_mm,
                     plausible=infant.plausible_ranges(infant_fiducials))


def test_cut_scales_with_the_head(infant_fiducials):
    # 0.2 x LPA-RPA: 30 mm for a 150 mm adult, about 22 mm here
    assert infant.ear_plane_cut(infant_fiducials) == pytest.approx(21.8,
                                                                   abs=0.1)


def test_an_infant_scan_in_mm_is_left_in_mm(sphere):
    pos, _ = sphere(radius=60.0)
    assert fid.guess_scale_to_mm(pos)[0] == 1.0
    assert fid.guess_scale_to_mm(pos / 1000.0)[0] == 1000.0


def test_a_text_scan_is_not_mistaken_for_its_own_landmarks(tmp_path):
    scan = tmp_path / 'scan.txt'
    np.savetxt(scan, np.random.default_rng(0).normal(size=(50, 3)))
    assert fid.find_fiducial_file(str(scan)) is None
    landmarks = tmp_path / 'fiducials.txt'
    landmarks.write_text('NAS 0 1 0\nLPA -1 0 0\nRPA 1 0 0\n')
    assert fid.find_fiducial_file(str(scan)) == str(landmarks)


def test_unknown_age_is_refused():
    with pytest.raises(ValueError, match='Unknown template age'):
        infant.load_template('5mo')


# =====================
# End to end (downloads the 6-month template)
# =====================
SYNTHETIC = _stretch(1.06, 0.94, 1.03)
ROTATE = np.eye(4)
ROTATE[:2, :2] = [[np.cos(0.35), -np.sin(0.35)], [np.sin(0.35), np.cos(0.35)]]
ROTATE[:3, 3] = [100.0, -40.0, 250.0]


def _deform(pos, scalp):
    """The synthetic infant: a bulge at the back, then stretched and moved
    into an arbitrary digitizer frame. Applied to every shell alike."""
    pos = np.asarray(pos, dtype=float)
    centre = np.asarray(scalp).mean(axis=0)
    out = _bump(pos - centre, centre=np.array([30.0, -50.0, 40.0]) - centre,
                height=4.0, width=25.0) + centre
    return apply_transform(ROTATE @ SYNTHETIC, out)


@pytest.fixture(scope='module')
def template():
    pytest.importorskip('mne')
    return infant.load_template('6mo')


@pytest.mark.slow
def test_cli_recovers_a_deformed_template(repo, tmp_path, template):
    scalp = template['bnd']['scalp']
    mesh = trimesh.Trimesh(*scalp, process=False)
    points, _ = trimesh.sample.sample_surface_even(mesh, 12000, seed=0)
    nas_lpa_rpa = template['fiducials']
    points = np.asarray(points)
    points = points[points[:, 2] > nas_lpa_rpa[1:, 2].mean() + 5]
    points = _deform(points, scalp[0])
    points += np.random.default_rng(0).normal(0, 0.3, points.shape)
    np.savetxt(tmp_path / 'scan.txt', points)
    with open(tmp_path / 'fiducials.txt', 'w') as fh:
        for label, pos in zip(['NAS', 'LPA', 'RPA'],
                              _deform(nas_lpa_rpa, scalp[0])):
            fh.write(f'{label} {pos[0]} {pos[1]} {pos[2]}\n')

    result = subprocess.run(
        [sys.executable, 'infantwarp.py', '-scalp',
         str(tmp_path / 'scan.txt'), '--age', '6mo',
         '--subjects-dir', template['subjects_dir']],
        cwd=repo, capture_output=True, text=True, timeout=1200)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]

    out = tmp_path / 'infantwarp_6mo'
    report = json.loads((out / 'qc.json').read_text())
    assert report['passed'], report['problems']
    assert report['scalp_fit']['warped']['median_abs'] < 1.0

    # the inner shells were never seen; where the scan covers the head they
    # must still land on the truth
    bnd = np.load(out / 'infant_warped_bnd.npy', allow_pickle=True).item()
    for shell in infant.SHELLS:
        pos, tris = template['bnd'][shell]
        truth = trimesh.Trimesh(_deform(pos, scalp[0]), tris, process=False)
        covered = pos[:, 2] > nas_lpa_rpa[1:, 2].mean() + 5
        _, dist, _ = trimesh.proximity.closest_point(truth,
                                                     bnd[shell][0][covered])
        assert np.median(dist) < 1.0, (shell, np.median(dist))

    import mne
    mne.set_log_level('error')
    model = mne.make_bem_model('infantwarp', ico=None,
                               subjects_dir=str(out / 'mne'),
                               conductivity=(0.33, 0.0132, 0.33))
    assert mne.make_bem_solution(model)['solution'].shape[0] > 0

    # the trans must land the digitized-frame scalp on the exported .surf
    import nibabel as nib
    trans = mne.read_trans(str(out / 'mne' / 'infantwarp'
                               / 'digitized2mri-trans.fif'))
    surf, _ = nib.freesurfer.read_geometry(
        str(out / 'mne' / 'infantwarp' / 'bem' / 'outer_skin.surf'))
    moved = mne.transforms.apply_trans(trans, bnd['scalp'][0] / 1000.) * 1000
    assert np.abs(moved - surf).max() < 1e-3
