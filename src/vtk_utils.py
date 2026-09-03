#!/usr/bin/env python
"""Getting meshes and points in and out of VTK without going via disk.

Both the voxelizer and the warp optimizer used to hand VTK a mesh by writing
an ASCII STL to a temp file and reading it back with vtkSTLReader. In the
optimizer that round trip was 71% of every objective evaluation, and it ran
thousands of times per fit.
"""
import numpy as np
import vtk
from vtk.util import numpy_support as vtk_np

# vtkSTLReader emits float32 points, so every result this pipeline has ever
# produced came from a float32 mesh. Building the polydata at float32 keeps
# the rewrite bit-identical instead of merely equivalent; float64 is ~1e-5 mm
# more faithful to the warped vertices but moves the optimizer's trajectory.
MESH_DTYPE = np.float32


def bnd2polydata(pos, tris, dtype=MESH_DTYPE):
    """vtkPolyData straight from numpy vertices and triangles."""
    points = vtk.vtkPoints()
    points.SetData(vtk_np.numpy_to_vtk(
        np.ascontiguousarray(pos, dtype=dtype), deep=1))

    tris = np.asarray(tris, dtype=np.int64)
    # legacy flat connectivity: [3, i, j, k, 3, i, j, k, ...]
    conn = np.hstack([np.full((len(tris), 1), 3, dtype=np.int64),
                      tris]).ravel()
    cells = vtk.vtkCellArray()
    cells.SetCells(len(tris), vtk_np.numpy_to_vtkIdTypeArray(conn, deep=1))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    return polydata


def points_inside(points, polydata):
    """Which of `points` lie inside the closed surface `polydata`.

    The same vtkSelectEnclosedPoints test as before, but points go in and
    answers come back as arrays rather than one InsertNextPoint / IsInside()
    call each.

    Order matters. The filter picks a random ray direction for points it finds
    ambiguous, so the classification of a handful of borderline points depends
    on the order they are submitted in and on them all arriving in one
    Update(). Callers must keep their traversal order stable and must not
    chunk the batch.
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros(0, dtype=bool)

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtk_np.numpy_to_vtk(points, deep=1))
    query = vtk.vtkPolyData()
    query.SetPoints(vtk_points)

    select = vtk.vtkSelectEnclosedPoints()
    select.SetInputData(query)
    select.SetSurfaceData(polydata)
    select.Update()

    selected = select.GetOutput().GetPointData().GetArray('SelectedPoints')
    return vtk_np.vtk_to_numpy(selected).astype(bool)
