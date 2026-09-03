#!/usr/bin/env python
"""
Converts .tri files to .nifti segmentation masks
"""
import nibabel as nib
import numpy as np

from src.vtk_utils import bnd2polydata, points_inside



def voxel_padding(bnds, shape, margin=5):
    """How far the meshes stick out of the template volume, per side.

    The template field of view is fixed, but the warped head is not: a
    neck-extended scalp runs off the inferior edge and used to be silently
    cut flat in the exported masks. Padding is therefore derived from the
    meshes rather than being a constant.
    """
    offset = np.array([int(n / 2.) for n in shape])
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for pos, _tris in bnds:
        voxel = np.asarray(pos, dtype=float) + offset
        lo = np.minimum(lo, voxel.min(axis=0))
        hi = np.maximum(hi, voxel.max(axis=0))
    before = np.maximum(0, np.ceil(-lo).astype(int) + margin)
    after = np.maximum(0, np.ceil(hi - np.array(shape)).astype(int) + margin)
    return tuple((int(b), int(a)) for b, a in zip(before, after))


def tri2nii(bnds, output_dir=None, transform=np.eye(4), t1_fn='template.nii', meshes='all'):

    # Load T1 image
    t1 = nib.load(t1_fn)
    assert all([t1.affine[i,i] == 1.0 for i in range(4)])

    # load meshes

    # Prepare segmentation mask
    data = np.zeros(t1.shape, dtype=np.uint8)

    # Grow the volume to whatever the meshes actually need
    pad_width = voxel_padding(bnds, t1.shape)
    data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)
    pad_before = np.array([p[0] for p in pad_width])

    # Transform points being inside surface meshes into index space for each tissue type
    # Go from out to inside - this makes sure that every voxel has only one label.
    # {1: 'scalp', 2: 'skull', 3: 'csf', 4: 'cortex'}
    for tissue_label in range(1, len(bnds)+1):
        # Work on a copy: the caller keeps using these vertex arrays after
        # tri2nii returns, and the shift below would otherwise leave them in
        # voxel index space.
        tissue_coords = np.array(bnds[tissue_label-1][0], dtype=float)
        ## Transform to voxel space
        # Note that it seems that SimNibs msh2nii just uses t1.shape / 2 as offset
        # bias instead of the actual t1.affine values

        # Note that one should apply the inverse t1.affine to the tissue_coords.
        # This is in our application not necessary, as the t1.affine is for our
        # data just a translation (because our inputs are all in the ACPC
        # coordinate system).

        # Apply translation of t1.affine (half of the shape), then the
        # padding offset so the coordinates index the grown array.
        assert all([-int(ni/2) for ni in t1.shape] == t1.affine[:3,-1])
        for axis in range(3):
            tissue_coords[:, axis] += int(t1.shape[axis] / 2.)
        tissue_coords += pad_before

        # Get complete range of XYZ coords for all nodes for later speed up
        nx, ny, nz = data.shape
        x_min = max(int(np.min(tissue_coords[:,0])), 0)
        x_max = min(int(np.max(tissue_coords[:,0])), nx - 1)
        y_min = max(int(np.min(tissue_coords[:,1])), 0)
        y_max = min(int(np.max(tissue_coords[:,1])), ny - 1)
        z_min = max(int(np.min(tissue_coords[:,2])), 0)
        z_max = min(int(np.max(tissue_coords[:,2])), nz - 1)

        #print('min/max X coordinate: ', x_min,x_max)
        #print('min/max Y coordinate: ', y_min,y_max)
        #print('min/max Z coordinate: ', z_min,z_max)


        # Test only the voxels in the mesh bounding box, as one batch.
        grid = np.meshgrid(np.arange(x_min, x_max + 1),
                           np.arange(y_min, y_max + 1),
                           np.arange(z_min, z_max + 1), indexing='ij')
        points = np.stack([g.ravel() for g in grid], axis=1)

        if points.size == 0:
            continue
        polydata = bnd2polydata(tissue_coords, bnds[tissue_label-1][1])
        inside = points_inside(points, polydata).reshape(grid[0].shape)

        # Fill the data array with the tissue label. Shells are processed
        # outside-in, so inner labels overwrite outer ones.
        box = data[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]
        box[inside] = tissue_label


    ## Save the segmentation mask (as one int-mask or as one binary mask per tissue label)
    # Padding shifts the voxel grid, so the affine has to shift with it.
    padded_affine = t1.affine.copy()
    padded_affine[:3, 3] -= padded_affine[:3, :3] @ pad_before
    affine = transform @ padded_affine

    if meshes == 'all':
        # {1: 'scalp', 2: 'skull', 3: 'csf', 4: 'cortex'}
        for tissue_label in range(1, len(bnds)+1):
            new_data = np.zeros(data.shape, dtype=np.uint8)
            new_data[data == tissue_label] = 1

            #new_img = nilearn.image.new_img_like(t1,new_data,affine=affine)
            # nibabel derives dim from the data shape; the old manual
            # header['dim'] += pad fixup is both unnecessary and wrong now
            # that padding differs per side.
            new_img = nib.nifti1.Nifti1Image(new_data, affine, t1.header)
            if output_dir is not None:
                output_file = output_dir + '/mask_'+str(tissue_label)+'.nii.gz'
            else:
                output_file = '_mask_'+str(tissue_label)+'.nii.gz'
            #print('Output File:', output_file)
            nib.save(new_img, output_file)
    else:
        tissue_color = {1: 0.45, # ~0.4–0.6 (depends on fat content)
                        2: 0.05, # ~0.0–0.1 (very dark, almost no signal)
                        3: 0.1, #CSF ~0.0–0.1 (dark; long T1 → low signal)
                        4: 0.55} #cortex (0.5–0.7 (brighter than cortex)
        new_data = np.zeros(data.shape)
        for tissue_label in range(1, len(bnds)+1):
            new_data[data == tissue_label] = tissue_color[tissue_label]
        #new_img = nilearn.image.new_img_like(t1,new_data,affine=affine)
        new_img = nib.nifti1.Nifti1Image(new_data, affine, t1.header)
        if output_dir is not None:
            output_file = output_dir + '/T1.mgz'
        else:
            output_file = 'bnd_w_mask.nii.gz'
        nib.save(new_img, output_file)



