"""The QC gate exists because a run once produced four watertight, properly
nested, entirely plausible surfaces that fitted the scan worse than no warping
at all. These tests check it would now catch that, and the other ways a head
model can be quietly wrong.
"""
import json

import numpy as np
import pytest

from src import qc


def _points_on(radius, n=400, seed=0):
    """Points scattered on a sphere of the given radius."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions * radius


# =====================
# Fit
# =====================
def test_fit_residual_is_near_zero_for_a_matching_scalp(nested_shells):
    shells = nested_shells()
    report = qc.scalp_fit(shells, _points_on(40.0))
    assert report['warped']['median_abs'] < 0.5


def test_improvement_ratio_rewards_beating_the_template(nested_shells,
                                                        sphere):
    shells = nested_shells()                       # scalp radius 40
    template = sphere(radius=48.0)                 # a poor fit
    report = qc.scalp_fit(shells, _points_on(40.0), template_scalp=template)
    assert report['improvement_ratio'] > 5


def test_the_audited_failure_is_caught(nested_shells, sphere):
    """A warp that fits worse than the unwarped template must fail QC, even
    though every surface is watertight and correctly nested."""
    shells = nested_shells()
    scan = _points_on(46.0)                        # the real head
    template = sphere(radius=45.0)                 # closer than the warp
    report = {'scalp_fit': qc.scalp_fit(shells, scan, template_scalp=template),
              'shells': qc.shell_geometry(shells),
              'nesting': qc.nesting(shells)}
    problems = qc.verdict(report)
    assert any('no better than the unwarped template' in p for p in problems)


# =====================
# Geometry
# =====================
def test_geometry_reports_a_healthy_mesh(nested_shells):
    geom = qc.shell_geometry(nested_shells())
    for shell, values in geom.items():
        assert values['watertight'], shell
        assert values['euler_number'] == 2, shell
        assert values['signed_volume_cm3'] > 0, shell


def test_inward_normals_are_flagged(nested_shells):
    shells = nested_shells()
    shells['skull'] = (shells['skull'][0], shells['skull'][1][:, ::-1])
    report = {'shells': qc.shell_geometry(shells), 'nesting': {},
              'scalp_fit': {}}
    assert any('normals point inward' in p for p in qc.verdict(report))


def test_a_hole_in_a_surface_is_flagged(nested_shells):
    shells = nested_shells()
    verts, tris = shells['scalp']
    shells['scalp'] = (verts, tris[:-4])           # punch a hole
    report = {'shells': qc.shell_geometry(shells), 'nesting': {},
              'scalp_fit': {}}
    problems = qc.verdict(report)
    assert any('watertight' in p or 'Euler' in p for p in problems)


# =====================
# Nesting
# =====================
def test_nesting_measures_the_gaps(nested_shells):
    gaps = qc.nesting(nested_shells(radii=(40.0, 34.0, 30.0, 26.0)))
    assert gaps['skull_in_scalp']['min_gap_mm'] == pytest.approx(6.0, abs=0.3)
    assert gaps['skull_in_scalp']['n_crossing'] == 0


def test_a_crossing_is_caught(sphere):
    shells = {'scalp': sphere(radius=40.0), 'skull': sphere(radius=34.0),
              'csf': sphere(radius=30.0), 'cortex': sphere(radius=30.6)}
    report = {'shells': qc.shell_geometry(shells),
              'nesting': qc.nesting(shells), 'scalp_fit': {}}
    problems = qc.verdict(report)
    assert any('cortex_in_csf' in p and 'cross' in p for p in problems)


# =====================
# End to end
# =====================
def test_run_writes_a_report_and_passes_a_good_model(tmp_path,
                                                     nested_shells, sphere):
    shells = nested_shells()
    report, problems = qc.run(shells, _points_on(40.0),
                              template_scalp=sphere(radius=48.0),
                              x_p=np.zeros(16), output_dir=str(tmp_path),
                              provenance={'scalp_file': 'synthetic'},
                              make_figure=False)
    assert problems == []
    assert report['passed']
    written = json.loads((tmp_path / 'qc.json').read_text())
    assert written['provenance']['scalp_file'] == 'synthetic'
    assert 'scalp' in written['shells']


def test_run_reports_problems_without_raising(tmp_path, nested_shells):
    shells = nested_shells()
    shells['scalp'] = (shells['scalp'][0], shells['scalp'][1][:, ::-1])
    report, problems = qc.run(shells, _points_on(40.0),
                              output_dir=str(tmp_path), make_figure=False)
    assert problems and not report['passed']
    assert json.loads((tmp_path / 'qc.json').read_text())['passed'] is False


def test_pc_weights_are_recorded_without_a_bogus_threshold():
    """They are coefficients on unit-norm eigenvectors, not multiples of a
    standard deviation - a value of 78 is an ordinary 2 mm of shape change,
    so flagging anything past 3 would fire on every single run."""
    recorded = qc.pc_weights(np.array([0.5, -1.0, 78.0, 0.1, -4.2]))
    assert recorded['max_abs'] == pytest.approx(78.0)
    assert recorded['weights'][2] == pytest.approx(78.0)
    assert not any('sd' in key.lower() for key in recorded)


def test_displacement_from_template_is_reported_in_mm(nested_shells, sphere):
    """The interpretable counterpart to the raw coefficients."""
    shells = nested_shells()
    template = (shells['scalp'][0] - np.array([0.0, 0.0, 3.0]),
                shells['scalp'][1])
    moved = qc.displacement_from_template(shells, template)
    assert moved['rms_mm'] == pytest.approx(3.0, abs=1e-6)
    assert moved['max_mm'] == pytest.approx(3.0, abs=1e-6)


def test_summary_is_printable(nested_shells, sphere):
    shells = nested_shells()
    report, _ = qc.run(shells, _points_on(40.0),
                       template_scalp=sphere(radius=48.0),
                       x_p=np.zeros(4), make_figure=False)
    text = qc.summary(report)
    assert 'scalp fit' in text and 'cortex' in text
