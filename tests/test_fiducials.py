"""The fiducial path is where the pipeline is most easily fed nonsense.

The regression these guard against: a real run was launched with mesh vertex
indices in place of coordinates, on a scan exported in metres. Both got
through, the warp fitted 16 components to a single point, and the resulting
head was worse than no warping at all.
"""
import json

import numpy as np
import pytest

from src import fiducials as fid


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def _pp(points, names=None):
    rows = []
    for i, (x, y, z) in enumerate(points):
        label = names[i] if names else str(i)
        rows.append(f'  <point active="1" name="{label}" '
                    f'x="{x}" y="{y}" z="{z}"/>')
    return ('<!DOCTYPE PickedPoints>\n<PickedPoints>\n'
            + '\n'.join(rows) + '\n</PickedPoints>\n')


# =====================
# Readers
# =====================
def test_reads_labelled_meshlab_pp(tmp_path, fiducials_mm):
    path = _write(tmp_path, 'scan.pp',
                  _pp(fiducials_mm, ['NAS', 'LPA', 'RPA']))
    assert np.allclose(fid.read_fiducial_file(path), fiducials_mm)


def test_reads_unlabelled_pp_in_picking_order(tmp_path, fiducials_mm):
    """MeshLab names picked points 0/1/2 unless you rename them."""
    path = _write(tmp_path, 'scan.pp', _pp(fiducials_mm))
    assert np.allclose(fid.read_fiducial_file(path), fiducials_mm)


def test_reads_slicer_markups(tmp_path, fiducials_mm):
    data = {'markups': [{'controlPoints': [
        {'label': lab, 'position': list(pos)}
        for lab, pos in zip(['Nz', 'LPA', 'RPA'], fiducials_mm.tolist())]}]}
    path = tmp_path / 'scan.mrk.json'
    path.write_text(json.dumps(data))
    assert np.allclose(fid.read_fiducial_file(str(path)), fiducials_mm)


def test_reads_slicer_fcsv(tmp_path, fiducials_mm):
    rows = ['# Markups fiducial file version = 5.0']
    for i, (lab, pos) in enumerate(zip(['NAS', 'LPA', 'RPA'],
                                       fiducials_mm.tolist())):
        rows.append(f'id_{i},{pos[0]},{pos[1]},{pos[2]},'
                    f'0,0,0,1,1,1,0,{lab},,')
    path = _write(tmp_path, 'scan.fcsv', '\n'.join(rows))
    assert np.allclose(fid.read_fiducial_file(path), fiducials_mm)


@pytest.mark.parametrize('nas_label', ['nasion', 'Nz', 'NAS', 'fidnz'])
def test_label_spellings(tmp_path, fiducials_mm, nas_label):
    lines = [f'{lab} {p[0]} {p[1]} {p[2]}' for lab, p in
             zip([nas_label, 'LPA', 'rpa'], fiducials_mm)]
    path = _write(tmp_path, 'f.txt', '# comment\n' + '\n'.join(lines))
    assert np.allclose(fid.read_fiducial_file(path), fiducials_mm)


def test_missing_landmark_is_reported(tmp_path, fiducials_mm):
    lines = [f'{lab} {p[0]} {p[1]} {p[2]}' for lab, p in
             zip(['NAS', 'LPA'], fiducials_mm[:2])]
    path = _write(tmp_path, 'f.txt', '\n'.join(lines))
    with pytest.raises(ValueError, match='3'):
        fid.read_fiducial_file(path)


# =====================
# Discovery
# =====================
def test_finds_sidecar_named_after_the_scan(tmp_path, fiducials_mm):
    (tmp_path / 'cutscan.obj').write_text('v 0 0 0\n')
    _write(tmp_path, 'cutscan.pp', _pp(fiducials_mm))
    found = fid.find_fiducial_file(str(tmp_path / 'cutscan.obj'))
    assert found is not None and found.endswith('cutscan.pp')


def test_scan_named_sidecar_wins_over_generic(tmp_path, fiducials_mm):
    (tmp_path / 'cutscan.obj').write_text('v 0 0 0\n')
    _write(tmp_path, 'cutscan.pp', _pp(fiducials_mm))
    _write(tmp_path, 'fiducials.txt', 'NAS 1 2 3\nLPA 4 5 6\nRPA 7 8 9\n')
    assert fid.find_fiducial_file(
        str(tmp_path / 'cutscan.obj')).endswith('cutscan.pp')


def test_no_sidecar_returns_none(tmp_path):
    (tmp_path / 'cutscan.obj').write_text('v 0 0 0\n')
    assert fid.find_fiducial_file(str(tmp_path / 'cutscan.obj')) is None


# =====================
# Units
# =====================
@pytest.mark.parametrize('scale,unit', [(1.0, 'mm'), (0.1, 'cm'),
                                        (0.001, 'm')])
def test_unit_is_inferred_from_head_size(sphere, scale, unit):
    verts, _ = sphere(radius=100.0)          # a 200 mm head
    factor, name = fid.guess_scale_to_mm(verts * scale)
    assert name == unit
    assert np.isclose(factor * scale, 1.0)


def test_absurd_extent_is_refused(sphere):
    verts, _ = sphere(radius=100.0)
    with pytest.raises(ValueError, match='not a head'):
        fid.guess_scale_to_mm(verts * 1000)


# =====================
# Vertex indices vs coordinates
# =====================
def test_vertex_indices_are_recognised_and_resolved(sphere):
    """The exact failure mode of the audited run."""
    verts, _ = sphere(radius=0.1, subdivisions=4)   # a scan in metres
    target = 300
    neighbours = np.argsort(
        np.linalg.norm(verts - verts[target], axis=1))[:3]
    picked = fid.as_vertex_indices(neighbours.astype(float), verts)
    assert picked is not None
    assert np.allclose(picked, verts[neighbours].mean(axis=0))


def test_real_coordinates_are_left_alone(sphere):
    verts, _ = sphere(radius=100.0)
    inside = np.array([1.0, 2.0, 3.0])            # whole numbers, but inside
    assert fid.as_vertex_indices(inside, verts) is None


def test_scattered_indices_are_not_treated_as_one_landmark(sphere):
    verts, _ = sphere(radius=100.0, subdivisions=4)
    far_apart = np.array([0.0, 500.0, 1000.0])
    assert fid.as_vertex_indices(far_apart, verts) is None


# =====================
# Validation
# =====================
def test_plausible_fiducials_pass(fiducials_mm):
    spacing = fid.validate(fiducials_mm)
    assert 110 < spacing['lpa-rpa'] < 200


def test_the_audited_bad_fiducials_are_rejected():
    """These are the literal values that produced an unusable head model."""
    bad = np.array([[16918., 16901, 16931],
                    [8091., 8117, 8092],
                    [9708., 9671, 9633]])
    with pytest.raises(ValueError) as excinfo:
        fid.validate(bad, 'the audited run')
    message = str(excinfo.value)
    assert 'lpa-rpa' in message and 'the audited run' in message


def test_metre_scale_fiducials_are_rejected(fiducials_mm):
    with pytest.raises(ValueError, match='Implausible'):
        fid.validate(fiducials_mm / 1000.0)
