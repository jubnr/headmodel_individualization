#!/usr/bin/env python
import numpy as np
import trimesh
from scipy.optimize import minimize
from numba import jit

import src.pca_pycaster as pycaster
from src.vtk_utils import bnd2polydata


def elec_warp(elecpos, pcas, mean_head, std_dev, regularize=False):
    """Fit the PCA weights so the reconstructed scalp passes through elecpos.

    `regularize` adds a penalty for shells coming closer than 5 mm to each
    other (see `regularizer`). Experimental: the penalty is a raw sum over
    every offending vertex pair and is not weighted against the shape
    distance, so on a head whose shells already intersect it dominates the
    objective and the fit runs away. It is also about 10x slower. Off by
    default; `enforce_nesting` is the cheap post-hoc alternative.
    """
    all_tris = {k: v[1] for k, v in mean_head.items()}
    num_pcas, pca_dim, dim = pcas.shape
    shells = list(mean_head.keys())
    # Determine mean_pnt for surface-line-intersection
    mean_pnt = np.mean(mean_head['cortex'][0], axis=0)
    min_idx, min_dist = shortest_dist(mean_pnt, mean_head['scalp'][0])
    new_z = np.max(mean_head['scalp'][0][:, 2]) - min_dist
    mean_pnt[2] = new_z

    # Minimize shape_distance over all PCs at once
    x0 = np.ones(num_pcas)
    res = minimize(error, x0,
                   args=(pcas, mean_head, std_dev, all_tris, mean_pnt,
                         elecpos, regularize),
                   options={'disp': False, 'eps': 0.5})  # mm
    x_p = res.x

    pos = pca2tri(x_p, pcas, mean_head, std_dev, wise='head')
    reconstructed = {shell: (pos[shell], mean_head[shell][1])
                     for shell in shells}
    return reconstructed, x_p


def caster(pos, tris):
    """Ray caster for a mesh, built straight from the numpy arrays.

    This used to export an ASCII STL to a fresh temp directory and read it
    back with vtkSTLReader on every objective evaluation - 71% of the cost of
    a fit, and one leaked temp directory per evaluation. The mesh is built at
    float32 because that is what vtkSTLReader produced, so the optimizer
    follows exactly the same trajectory as before.
    """
    return pycaster.rayCaster(bnd2polydata(pos, tris))

def shortest_dist(vert_pos, list_of_pos):
    min_idx, min_dist = 0, np.linalg.norm(list_of_pos[0] - vert_pos)
    dist = np.linalg.norm(np.array(list_of_pos) - vert_pos, axis=1)  
    min_idx = np.argmin(dist)
    min_dist = dist[min_idx]
    return min_idx, min_dist


def shape_distance(elecpos, bnd, mean_pnt, return_all=False):
    pos, tris = bnd
    fit = caster(pos, tris)
    proj = elecpos + 1000*(elecpos-mean_pnt)
    intsct = []
    for i, elec in enumerate(elecpos):
        # find intersection with mesh
        intersections = fit.castRay(mean_pnt, proj[i])
        if len(intersections) == 1:
            min_idx = 0
        elif len(intersections) > 1:
            #print('More than one intersection!')
            min_idx, _ = shortest_dist(elec, intersections)
        else:
            #print('Zero intersections!')
            intersections = pos
            min_idx, _ = shortest_dist(elec, intersections)
        intsct.append(intersections[min_idx])
    dist = np.linalg.norm(elecpos - np.array(intsct), axis=1)
    if return_all:
        return dist
    diff = np.mean(dist)
    #diff = np.median(dist)
    return diff


def pca2tri(coeff, pcas, mean_head, std_dev, wise='head'):
    shells = list(mean_head.keys())
    num_pcas, dim_pcas, dim = pcas.shape
    reconstructed = {}
    bndsize = [len(mean_head[shell][0]) for shell in shells]
    mean_bnd = np.zeros((sum(bndsize)*dim))
    for s, shell in enumerate(shells):
        size = bndsize[s]*dim
        start = sum(bndsize[:s])*dim
        mean_bnd[start:start+size] = mean_head[shell][0].flatten()
    reshaped_mean_bnd = []
    reshaped_std_dev = []
    for s, shell in enumerate(shells):
        size = bndsize[s]*dim
        start = sum(bndsize[:s])*dim
        reshaped_mean_bnd.append(
                mean_bnd[start:start+size].reshape(bndsize[s], dim))
        reshaped_std_dev.append(
                std_dev[start:start+size].reshape(bndsize[s], dim))
    if wise == 'bnd':
        for s, shell in enumerate(shells):
            size = bndsize[s]
            start = sum(bndsize[:s])
            X = pcas[:,start:start+size,:].reshape(num_pcas, size*dim).T
            reconstructed[shell] = (X.dot(coeff[shell]) * reshaped_std_dev[s])\
                                   + reshaped_mean_bnd[s]
            reconstructed[shell] = reconstructed[shell].reshape(size, dim)
    elif wise == 'head': 
        X = pcas.reshape((num_pcas, sum(bndsize)*dim)).T 
        reconstructed_all = (X.dot(coeff) * std_dev) + mean_bnd
        for s, shell in enumerate(shells):
            size = bndsize[s]*dim
            start = sum(bndsize[:s])*dim
            reconstructed[shell] = \
                    reconstructed_all[start:start+size].reshape(bndsize[s],
                                                                dim)
    else:
        raise ValueError
    return reconstructed


def error(x_p, pcas, mean_head, std_dev, all_tris, mean_pnt, elecpos,
          regularize=False):
    pos = pca2tri(x_p, pcas, mean_head, std_dev, wise='head')
    fit_bnd = (pos['scalp'], all_tris['scalp'])
    diff = shape_distance(elecpos, fit_bnd, mean_pnt)

    # penalize shells that come too close to each other
    if regularize:
        shells = ['cortex', 'csf', 'skull', 'scalp']
        bndsize = np.array([len(pos[shell]) for shell in shells])
        warped = np.zeros((np.sum(bndsize), 3))
        start = 0
        for s, shell in enumerate(shells):
            warped[start:start + bndsize[s], :] = pos[shell]
            start += bndsize[s]
        diff += regularizer(warped, bndsize)

    return diff


@jit(nopython=True, parallel=True)
def regularizer(warped, bndsize):
    diff = 0
    start = 0
    for s in range(0, len(bndsize)-1):
        pts1 = warped[start:start+bndsize[s]]
        start += bndsize[s]
        pts2 = warped[start:start+bndsize[s+1]]

        # reshaping to be able to calculate the distance matrix
        a_reshaped = pts1.reshape(pts1.shape[0], 1, 3)
        b_reshaped = pts2.reshape(1, pts2.shape[0], 3)

        #calculation of all distances between all points
        norms = np.sqrt(np.sum((a_reshaped - b_reshaped)**2, axis=2)).flatten()

        # penalize too short distances
        #penalty = (0.010-norms[norms < 0.010])
        penalty = (5.0-norms[norms < 5.0]) #5mm
        diff += np.sum(penalty)
    return diff




def degenerate_shells(pcas, mean_head):
    """Shells whose PCA block carries no variance at all.

    Such a shell is returned unchanged by the reconstruction no matter what
    the scalp looks like: `data/pcas_hartmut` ships with an all-zero cortex
    block, so the cortex would silently stay at the template mean.
    """
    shells = list(mean_head.keys())
    bndsize = [len(mean_head[shell][0]) for shell in shells]
    dead = []
    for s, shell in enumerate(shells):
        start = sum(bndsize[:s])
        block = pcas[:, start:start + bndsize[s], :]
        if not np.any(block):
            dead.append(shell)
    return dead


def transfer_displacement(reconstructed, mean_head, src, dst):
    """Move `dst` by the displacement the warp gave `src`.

    Only meaningful for two shells that share a triangulation and a radial
    vertex correspondence, as csf and cortex do in this database (the two
    surfaces agree on vertex direction to within ~4 degrees). It is an
    approximation, not a fit - but it keeps a shell that the PCA cannot
    individualize from being left behind by, and then crossing, the shell
    outside it.
    """
    if not np.array_equal(mean_head[src][1], mean_head[dst][1]):
        raise ValueError(f'{src} and {dst} do not share a triangulation.')
    shift = reconstructed[src][0] - mean_head[src][0]
    return (mean_head[dst][0] + shift, reconstructed[dst][1])


def enforce_nesting(bnd, outer, inner, margin=0.5):
    """Pull `inner` vertices back inside `outer`, keeping `margin` mm of gap.

    The warp fits each shell independently, so two neighbouring surfaces can
    touch or cross where the gap between them is thin to begin with. Vertices
    that offend are moved to the closest point on the outer surface and then
    `margin` inwards along its normal - the smallest correction that restores
    a valid nesting. Returns the corrected shell and how many vertices moved.
    """
    from src.tri_io import signed_volume

    outer_mesh = trimesh.Trimesh(bnd[outer][0], bnd[outer][1], process=False)
    pos = np.array(bnd[inner][0], dtype=float)
    closest, dist, tri_id = trimesh.proximity.closest_point(outer_mesh, pos)
    offending = ~outer_mesh.contains(pos) | (dist < margin)
    if not np.any(offending):
        return bnd[inner], 0

    outward = np.sign(signed_volume(outer_mesh.vertices, outer_mesh.faces))
    normals = outward * outer_mesh.face_normals[tri_id[offending]]
    pos[offending] = closest[offending] - margin * normals
    return (pos, bnd[inner][1]), int(offending.sum())
