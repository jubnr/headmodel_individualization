"""The in-memory VTK path replaced an ASCII-STL round trip through /tmp.

That round trip was 71% of every objective evaluation in the warp, and it ran
thousands of times per fit, leaking a temp directory each time. These tests
pin the two properties that made replacing it safe: the polydata built from
numpy behaves exactly like the one vtkSTLReader produced, and the batched
inside/outside test agrees with the per-point one it replaced.
"""
import numpy as np
import pytest
import vtk

from src.vtk_utils import MESH_DTYPE, bnd2polydata, points_inside


def _via_stl(verts, tris, tmp_path):
    """The old path: trimesh -> ASCII STL on disk -> vtkSTLReader."""
    import trimesh
    path = str(tmp_path / 'mesh.stl')
    trimesh.Trimesh(verts, tris).export(path, file_type='stl_ascii')
    reader = vtk.vtkSTLReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()


def test_polydata_has_the_right_shape(sphere):
    verts, tris = sphere()
    poly = bnd2polydata(verts, tris)
    assert poly.GetNumberOfPoints() == len(verts)
    assert poly.GetNumberOfCells() == len(tris)


def test_polydata_keeps_the_stl_readers_precision():
    """vtkSTLReader emits float32, so the optimizer has always seen float32.
    Building float64 would be more faithful but would move the fit."""
    assert MESH_DTYPE == np.float32


def test_inside_test_matches_the_per_point_api(sphere):
    """points_inside reads the SelectedPoints array instead of calling
    IsInside() once per point."""
    verts, tris = sphere(radius=10.0)
    poly = bnd2polydata(verts, tris)
    query = np.random.default_rng(0).uniform(-15, 15, (2000, 3))

    batched = points_inside(query, poly)

    vtk_points = vtk.vtkPoints()
    for point in query:
        vtk_points.InsertNextPoint(point)
    holder = vtk.vtkPolyData()
    holder.SetPoints(vtk_points)
    select = vtk.vtkSelectEnclosedPoints()
    select.SetInputData(holder)
    select.SetSurfaceData(poly)
    select.Update()
    one_at_a_time = np.array([select.IsInside(i) == 1
                              for i in range(len(query))])

    assert np.array_equal(batched, one_at_a_time)


def test_inside_test_agrees_with_the_analytic_answer(sphere):
    verts, tris = sphere(radius=10.0, subdivisions=5)
    query = np.random.default_rng(1).uniform(-15, 15, (2000, 3))
    # skip points within a discretisation error of the surface
    radii = np.linalg.norm(query, axis=1)
    clear = np.abs(radii - 10.0) > 0.2
    inside = points_inside(query, bnd2polydata(verts, tris))
    assert np.array_equal(inside[clear], radii[clear] < 10.0)


def test_numpy_polydata_matches_the_stl_round_trip(sphere, tmp_path):
    """The property that made dropping the disk round trip safe."""
    verts, tris = sphere(radius=10.0)
    query = np.random.default_rng(2).uniform(-15, 15, (3000, 3))
    assert np.array_equal(
        points_inside(query, bnd2polydata(verts, tris)),
        points_inside(query, _via_stl(verts, tris, tmp_path)))


def test_empty_query_is_handled(sphere):
    verts, tris = sphere()
    result = points_inside(np.zeros((0, 3)), bnd2polydata(verts, tris))
    assert result.shape == (0,) and result.dtype == bool


def test_ray_caster_results_are_unchanged_by_the_rewrite(sphere, tmp_path):
    """The optimizer's objective must not move."""
    import src.pca_pycaster as pycaster
    from src.pca_warp import caster

    verts, tris = sphere(radius=50.0, subdivisions=4)
    origin = np.array([0.0, 0.0, 0.0])
    targets = np.random.default_rng(3).normal(size=(50, 3))
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    targets *= 1000.0

    new = caster(verts, tris)
    old = pycaster.rayCaster(_via_stl(verts, tris, tmp_path))
    for target in targets:
        hits_new = new.castRay(origin, target)
        hits_old = old.castRay(origin, target)
        assert len(hits_new) == len(hits_old)
        if hits_new:
            assert np.array_equal(np.asarray(hits_new),
                                  np.asarray(hits_old))


def test_caster_leaves_no_temp_files(sphere, tmp_path, monkeypatch):
    """The old caster() created a temp directory per objective evaluation and
    removed only the file inside it - thousands of leaked directories a run."""
    import tempfile
    from src.pca_warp import caster

    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    verts, tris = sphere()
    for _ in range(5):
        caster(verts, tris)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('offset', [0.0, 250.0])
def test_inside_test_is_position_independent(sphere, offset):
    verts, tris = sphere(radius=10.0, centre=(offset, offset, offset))
    poly = bnd2polydata(verts, tris)
    centre = np.array([[offset, offset, offset]])
    far = np.array([[offset + 100, offset, offset]])
    assert points_inside(centre, poly)[0]
    assert not points_inside(far, poly)[0]
