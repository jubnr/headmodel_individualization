"""End-to-end runs. Opt in with `pytest -m slow`.

The fast tests check the pieces; these check that a real scan goes in and a
usable head model comes out, twice, identically.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.slow

SCAN = os.path.join('data', 'photogrammetry_test_data', 'cutscan.obj')


@pytest.fixture(scope='module')
def warped(repo):
    """Run the warp itself (no exports) on the shipped test scan."""
    sys.path.insert(0, repo)
    import PCAwarp as pipeline
    from src.fiducials import read_fiducial_file, validate
    from src.sampling import subsample
    from src.transform_to_ctf import transform_to_ctf

    scan = os.path.join(repo, SCAN)
    fiducials = read_fiducial_file(os.path.splitext(scan)[0] + '.pp')
    validate(fiducials)
    points = pipeline.load_scalp_file(scan, None, None, None)[0]
    points, fiducials, _ = pipeline.to_millimetres(points, fiducials)
    mean_scalp = np.load(pipeline.MEAN_HEAD,
                         allow_pickle=True).item()['scalp'][0]
    points, transform = transform_to_ctf(points, *fiducials.copy(),
                                         mean_scalp=mean_scalp,
                                         return_transform=True)
    points = points[points[:, 2] > 30]
    sampled, _ = subsample(points, 100)
    bnd, x_p = pipeline.pca_surfacemesh_warping(fiducials, sampled)
    return {'bnd': bnd, 'x_p': x_p, 'scalp_ctf': points,
            'mean_scalp': (mean_scalp,
                           np.load(pipeline.MEAN_HEAD,
                                   allow_pickle=True).item()['scalp'][1])}


def test_the_warp_beats_the_unwarped_template(warped):
    from src import qc
    fit = qc.scalp_fit(warped['bnd'], warped['scalp_ctf'],
                       template_scalp=warped['mean_scalp'])
    assert fit['improvement_ratio'] > 1.5, (
        f"median {fit['warped']['median_abs']:.2f} mm vs template "
        f"{fit['template_baseline']['median_abs']:.2f} mm")


def test_the_output_passes_its_own_quality_gate(warped):
    from src import qc
    report, problems = qc.run(warped['bnd'], warped['scalp_ctf'],
                              template_scalp=warped['mean_scalp'],
                              x_p=warped['x_p'], make_figure=False)
    assert problems == [], problems
    assert report['passed']


def test_two_warps_of_one_scan_are_identical(repo, warped):
    """Before deterministic sampling this differed by up to 5.9 mm."""
    sys.path.insert(0, repo)
    import PCAwarp as pipeline
    from src.sampling import subsample

    sampled, _ = subsample(warped['scalp_ctf'], 100)
    again, x_p = pipeline.pca_surfacemesh_warping(None, sampled)
    for shell in ['scalp', 'skull', 'csf', 'cortex']:
        assert np.array_equal(again[shell][0], warped['bnd'][shell][0]), shell
    assert np.array_equal(x_p, warped['x_p'])


def test_the_cli_produces_a_usable_mne_subject(repo, tmp_path):
    """The whole thing, through the command line, as a user runs it."""
    import shutil
    scan = tmp_path / 'cutscan.obj'
    shutil.copy(os.path.join(repo, SCAN), scan)
    shutil.copy(os.path.join(repo, os.path.splitext(SCAN)[0] + '.pp'),
                tmp_path / 'cutscan.pp')

    result = subprocess.run(
        [sys.executable, 'PCAwarp.py', '-scalp', str(scan)],
        cwd=repo, capture_output=True, text=True, timeout=3600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]

    report = json.loads((tmp_path / 'qc.json').read_text())
    assert report['passed'], report['problems']
    assert (tmp_path / 'qc.png').exists()

    # every shell must now fit inside the exported volume
    for shell, fraction in report['field_of_view_fraction'].items():
        assert fraction == 1.0, f'{shell} is clipped ({fraction:.1%} inside)'

    import mne
    mne.set_log_level('error')
    subjects_dir = str(tmp_path / 'mne')
    model = mne.make_bem_model('pcawarp', ico=None,
                               subjects_dir=subjects_dir,
                               conductivity=(0.3, 0.006, 0.3))
    assert mne.make_bem_solution(model)['solution'].shape[0] > 0

    # the head->mri transform must land the surfaces where the .surf files are
    import nibabel as nib
    trans = mne.read_trans(str(tmp_path / 'mne' / 'pcawarp'
                               / 'ditigized2ras-trans.fif'))
    bnd = np.load(tmp_path / 'pca_warped_bnd.npy', allow_pickle=True).item()
    surf, _ = nib.freesurfer.read_geometry(
        str(tmp_path / 'mne' / 'pcawarp' / 'bem' / 'outer_skin.surf'))
    moved = mne.transforms.apply_trans(trans, bnd['scalp'][0] / 1000.) * 1000
    assert np.abs(moved - surf).max() < 1e-3
