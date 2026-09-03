#!/usr/bin/env python
import os
import argparse
import numpy as np
from os.path import join as pth

from src.tri_io import write_tri, orient_outward
from src.transform_to_ctf import transform_to_ctf, apply_transform
from src.pca_warp import (shortest_dist, elec_warp, degenerate_shells,
                          transfer_displacement, enforce_nesting)
from src.tri2nii import tri2nii
from src.nii_postprocessing import postprocessing
from src import fiducials as fid_io
from src.sampling import subsample, coverage
from src import qc as qc_mod


BASEDIR = os.path.dirname(os.path.realpath(__file__))


# =====================
# Constants
# =====================
NUM_PCAS = 16  # out of 316
HARTMUT = True  # whether to use Hartmut's PCA-based warping (True) or the
                # original one (False)

# HArtMuT artefact warping. Only runs when HARTMUT is True (the HArtMuT PCAs).
# The artefact sources are defined in the template the chosen HArtMuT model
# lives in, so we warp from that template into the individual head. We start
# from the base HArtMuT NYhead coming from the HArtMuT repo (sibling checkout
# by default), see https://github.com/harmening/HArtMuT. Override the paths if
# yours differ.
HARTMUT_REPO = pth(BASEDIR, '..', 'HArtMuT')
HARTMUT_MODEL = pth(HARTMUT_REPO, 'HArtMuTmodels', 'HArtMuT_NYhead_small.mat')
HARTMUT_TEMPLATE = {'scalp': pth(HARTMUT_REPO, 'individualwarp', 'NYhead',
                                 'scalp.stl'),
                    'skull': pth(HARTMUT_REPO, 'individualwarp', 'NYhead',
                                 'skull.stl')}
HARTMUT_MEANPNT = np.array([0.0, -10.0, 0.0])  # fixed ray origin, model frame, mm
ACPC2CTF = pth(BASEDIR, 'src', 'transform_acpc2ctf_icbm.npy')



# Please do not change the following paths
if HARTMUT:
    pca_dir = 'pcas_hartmut'
else:
    pca_dir = 'pcas'
PCAS = pth(BASEDIR, 'data', pca_dir, 'ALLpcas.npy')
MEAN_HEAD = pth(BASEDIR, 'data', pca_dir, 'mean_head.npy')
STD_DEV = pth(BASEDIR, 'data', pca_dir, 'std_dev.npy')
SHELLS = ['scalp', 'skull', 'csf', 'cortex']

# Which shell a non-individualizable one borrows its displacement from. Only
# valid between shells sharing a triangulation.
FALLBACK_DONOR = {'cortex': 'csf'}



# =====================
# Core Functions
# =====================
def check_hartmut_available():
    """Fail before the warp, not after it, if the HArtMuT checkout is missing.

    The artefact sources live in the HArtMuT repository, which is a separate
    clone this one expects as a sibling directory. Without it the run used to
    get all the way through a multi-minute fit and then die on a missing .mat.
    """
    missing = [path for path in
               [HARTMUT_MODEL, HARTMUT_TEMPLATE['scalp'],
                HARTMUT_TEMPLATE['skull']] if not os.path.isfile(path)]
    if not missing:
        return
    raise FileNotFoundError(
        'HARTMUT is True but the HArtMuT repository was not found. It is a '
        'separate clone:\n\n'
        '    git clone https://github.com/harmening/HArtMuT\n\n'
        f'expected next to this one, at {os.path.abspath(HARTMUT_REPO)}.\n'
        'Missing:\n  ' + '\n  '.join(missing)
        + '\nEither clone it there, point HARTMUT_REPO in PCAwarp.py at your '
          'copy, or set HARTMUT = False to skip the artefact model (the PCA '
          'basis changes too - see the README).')


def pca_surfacemesh_warping(fiducials, optodes, regularize=False):
    """Perform PCA-based surface mesh warping."""
    mean_bnd = np.load(MEAN_HEAD, allow_pickle=True).item()
    std_dev = np.load(STD_DEV, allow_pickle=True)
    pcas = np.load(PCAS, allow_pickle=True)

    if len(pcas) < NUM_PCAS:
        raise ValueError('Too many PCs requested. Check NUM_PCAS.')
    pcas = pcas[:NUM_PCAS]

    mean_pnt = np.mean(mean_bnd['cortex'][0], axis=0)
    _, min_dist = shortest_dist(mean_pnt, mean_bnd['scalp'][0])
    mean_pnt[2] = np.max(mean_bnd['scalp'][0][:, 2]) - min_dist

    bnd_w, x_p = elec_warp(optodes, pcas, mean_bnd, std_dev,
                           regularize=regularize)

    # Shells with an all-zero PCA block come back at the template mean. The
    # HArtMuT basis has this for the cortex; left alone it would be crossed by
    # the csf surface, which does warp. Carry the csf displacement over
    # instead - the two shells share a triangulation and a radial vertex
    # correspondence - and say so, because that cortex is not a fit.
    for shell in degenerate_shells(pcas, mean_bnd):
        donor = FALLBACK_DONOR.get(shell)
        if donor is not None and donor in bnd_w:
            print(f'  {shell}: PCA block in {pca_dir} carries no variance; '
                  f'following the {donor} warp instead (approximation, not a '
                  f'fit to the scalp).')
            bnd_w[shell] = transfer_displacement(bnd_w, mean_bnd, donor, shell)
            bnd_w[shell], moved = enforce_nesting(bnd_w, donor, shell)
            if moved:
                print(f'  {shell}: pulled {moved} vertices back inside '
                      f'{donor}.')
        else:
            print(f'  WARNING: {shell}: PCA block in {pca_dir} carries no '
                  f'variance, this shell stays at the template mean.')

    # The database is wound inwards; every consumer (OpenMEEG, FieldTrip, MNE)
    # wants outward normals.
    bnd_w = {shell: (pos, orient_outward(pos, tris))
             for shell, (pos, tris) in bnd_w.items()}
    return bnd_w, x_p


def load_scalp_file(filepath, nas, lpa, rpa):
    """Load scalp data depending on file format."""
    if filepath.endswith('.npy'):
        return np.load(filepath), nas, lpa, rpa
    if filepath.endswith('.txt'):
        return np.loadtxt(filepath), nas, lpa, rpa

    if filepath.endswith('.bvct'):
        return _load_captrak(filepath)
    if filepath.endswith(('.hsp', '.elp', '.eeg')):
        return _load_polhemus(filepath)
    if filepath.endswith('.elc'):
        return _load_elc(filepath, nas, lpa, rpa)

    return _load_mesh(filepath), nas, lpa, rpa


def _load_captrak(filepath):
    import mne
    captrak = mne.channels.read_dig_captrak(filepath)
    channels = np.array([dig['r'] for dig in captrak.dig if dig['kind'] ==
                         mne.io.constants.FIFF.FIFFV_POINT_EEG])
    scalp_proxies = channels * 1000  # m -> mm
    nas, lpa, rpa = _extract_fiducials(captrak.dig)
    return scalp_proxies, nas, lpa, rpa


def _load_polhemus(filepath):
    import mne
    polhemus = mne.channels.read_dig_polhemus_isotrak(filepath)
    channels = np.array([dig['r'] for dig in polhemus.dig if dig['kind'] ==
                         mne.io.constants.FIFF.FIFFV_POINT_EEG])
    scalp_proxies = channels * 1000  # m -> mm
    nas, lpa, rpa = _extract_fiducials(polhemus.dig)
    return scalp_proxies, nas, lpa, rpa


def _extract_fiducials(dig_points):
    import mne
    nas, lpa, rpa = None, None, None
    for dig in dig_points:
        if dig['kind'] != mne.io.constants.FIFF.FIFFV_POINT_CARDINAL:
            continue
        if dig['ident'] == mne.io.constants.FIFF.FIFFV_POINT_NASION:
            nas = dig['r'] * 1000
        elif dig['ident'] == mne.io.constants.FIFF.FIFFV_POINT_LPA:
            lpa = dig['r'] * 1000
        elif dig['ident'] == mne.io.constants.FIFF.FIFFV_POINT_RPA:
            rpa = dig['r'] * 1000
    return nas, lpa, rpa


def _load_elc(filepath, nas, lpa, rpa):
    coords = []
    with open(filepath, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            label, xyz = line.strip().split(':')
            xyz = np.array(list(map(float, xyz.split())))
            if label.strip() in ['NAS', 'Nz', 'Nasion']:
                nas = xyz
            elif label.strip() in ['LPA', 'Lpa', 'LeftEar']:
                lpa = xyz
            elif label.strip() in ['RPA', 'Rpa', 'RightEar']:
                rpa = xyz
            else:
                coords.append(xyz)
    return np.array(coords), nas, lpa, rpa


def _load_mesh(filepath):
    try:
        import trimesh
        mesh = trimesh.load(filepath, force='mesh')
        return np.array(mesh.vertices)
    except Exception as e:
        raise RuntimeError(f'Could not load scalp file: {e}')


def raw_vertices(filepath):
    """Vertices of a mesh file in file order, or None if it is not a mesh.

    Mesh pickers report indices into this list, so index-shaped fiducials are
    resolved against it rather than against whatever trimesh returns after
    merging and reordering.
    """
    if filepath.lower().endswith('.obj'):
        verts = [[float(v) for v in line.split()[1:4]]
                 for line in open(filepath) if line.startswith('v ')]
        return np.array(verts, dtype=float) if verts else None
    try:
        import trimesh
        mesh = trimesh.load(filepath, force='mesh', process=False)
        return np.asarray(mesh.vertices, dtype=float)
    except Exception:
        return None



# =====================
# Fiducials
# =====================
def resolve_fiducials(args, scalp_path):
    """Work out NAS, LPA and RPA without making the user retype them.

    Order of preference: what was passed on the command line, then an explicit
    -fiducials file, then a landmark file sitting next to the scalp file
    (MeshLab .pp, Slicer .mrk.json/.fcsv, plain text). Values that are really
    vertex indices are recognised and looked up.
    """
    verts = raw_vertices(scalp_path)
    given = {'nas': args.nas, 'lpa': args.lpa, 'rpa': args.rpa}

    if all(v is not None for v in given.values()):
        out = {}
        for name, value in given.items():
            picked = fid_io.as_vertex_indices(value, verts)
            if picked is not None:
                print(f'  {name.upper()}: {[int(v) for v in value]} are vertex '
                      f'indices, not coordinates - using the picked vertices at '
                      f'{np.round(picked, 2).tolist()}.')
                out[name] = picked
            else:
                out[name] = np.asarray(value, dtype=float)
        return np.array([out['nas'], out['lpa'], out['rpa']]), 'command line'

    if args.fiducials:
        if not os.path.isfile(args.fiducials):
            raise FileNotFoundError(f'No such fiducial file: {args.fiducials}')
        print(f'  reading landmarks from {args.fiducials}')
        return fid_io.read_fiducial_file(args.fiducials), args.fiducials

    found = fid_io.find_fiducial_file(scalp_path)
    if found:
        print(f'  found landmarks next to the scalp file: {found}')
        return fid_io.read_fiducial_file(found), found

    raise ValueError(
        'No fiducials. Give them with -nas/-lpa/-rpa, point -fiducials at a '
        'landmark file, or drop one next to the scalp file - a MeshLab .pp, a '
        'Slicer .mrk.json/.fcsv, or a text file with one "LABEL x y z" line '
        f'per landmark, named e.g. '
        f'{os.path.splitext(os.path.basename(scalp_path))[0]}.pp.')


def to_millimetres(scalp, fiducials):
    """Put the scalp proxy and its landmarks into mm, whatever they arrived in.

    The PCA database is in mm; a scan exported in metres would otherwise be
    warped to as if it were a 24 mm head.
    """
    scale, unit = fid_io.guess_scale_to_mm(scalp)
    if scale != 1.0:
        print(f'  scalp proxy looks like {unit}, scaling by {scale:g} to mm')
    return scalp * scale, np.asarray(fiducials, dtype=float) * scale, scale



# =====================
# Export Functions
# =====================
def export_openmeeg(bnd_w, output_dir):
    for shell in SHELLS:
        write_tri(bnd_w[shell][0], bnd_w[shell][1],
                  pth(output_dir, f'pca_warped_{shell}.tri'))
        try:
            import trimesh
            mesh = trimesh.Trimesh(bnd_w[shell][0], bnd_w[shell][1])
            mesh.export(pth(output_dir, f'pca_warped_{shell}.stl'),
                        file_type='stl_ascii')
        except ImportError:
            pass


def export_fieldtrip(bnd_w, output_dir, coordsys='digitized', eye_pos=None,
                     fiducials=None, hartmut=None, fname='pcawarp_bnd.mat'):
    """Write the surfaces as bnd.mat to be used by fieldtrip's dipolefitting.

    Standard FieldTrip constructs only: a 4-element mesh array `bnd` (scalp,
    skull, csf, cortex), a parallel `tissue` cell array, an optional `eye`
    struct with one eye's candidate positions for artefact-aware dipole fitting
    a la HArtMuT, and an optional `fiducials` struct (Nz, LPA, RPA, in the same
    digitized frame) so the head can be transformed to a known coordinate system.
    The `coordsys` field is set to 'digitized' by default,

    When HARTMUT is True, `hartmut` carries the warped artefact model (positions,
    source model dict), which is written alongside as the individual HArtMuT
    artefact model.
    """
    try:
        import scipy.io as sio
    except ImportError:
        print('scipy not installed, skipping .mat export.')
        return

    dtype = np.dtype([('pos', 'O'), ('tri', 'O'), ('unit', 'O'),
                      ('coordsys', 'O')])
    bnd = np.empty((1, len(SHELLS)), dtype=dtype)
    for i, shell in enumerate(SHELLS):
        pos, tri = bnd_w[shell]
        bnd[0, i]['pos'] = np.asarray(pos, dtype=float)
        bnd[0, i]['tri'] = np.asarray(tri, dtype=float) + 1  # MATLAB uses 1-based indexing
        bnd[0, i]['unit'] = 'mm'
        bnd[0, i]['coordsys'] = coordsys

    out = {'bnd': bnd, 'tissue': np.array(SHELLS, dtype=object)}
    if eye_pos is not None and len(eye_pos):
        out['eye'] = {'pos': np.asarray(eye_pos, dtype=float)}
    if fiducials is not None and len(fiducials):
        out['fiducials'] = {'pos': np.asarray(fiducials, dtype=float),
                            'label': np.array(['Nz', 'LPA', 'RPA'], dtype=object)}
    sio.savemat(pth(output_dir, fname), out)

    if HARTMUT and hartmut is not None:
        new_pos, model = hartmut
        export_hartmut(new_pos, model, output_dir)


def export_hartmut(new_pos, model, output_dir, fname='pcawarp_hartmut.mat'):
    """Save the warped HArtMuT artefact model, mirroring the Julia individualwarp
    output: same orientation, labels and unit, only the positions are warped."""
    import scipy.io as sio
    artefactmodel = {'pos': np.asarray(new_pos, dtype=float),
                     'orientation': model['orientation'],
                     'labels': model['labels'],
                     'unit': model['unit']}
    sio.savemat(pth(output_dir, fname),
                {'HArtMuT': {'artefactmodel': artefactmodel}})


def warp_artefacts_ctf(bnd_w_ctf):
    """Warp the HArtMuT artefact sources into the individual head, in CTF.

    bnd_w_ctf holds the warped individual surfaces in CTF (before the
    back-transform to the input frame). Returns the warped artefact positions,
    the one-eye candidate positions, and the source model dict, all in CTF.
    """
    import scipy.io as sio
    import trimesh
    from src.hartmut_warp import warp_hartmut

    acpc2ctf = np.load(ACPC2CTF, allow_pickle=True).astype(float)
    acpc2ctf[:3, 3] *= 1000.0  # m -> mm

    # source template (HArtMuT model frame) into CTF
    src_head = {}
    for shell in ('scalp', 'skull'):
        mesh = trimesh.load(HARTMUT_TEMPLATE[shell])
        verts = np.asarray(mesh.vertices, dtype=float)
        src_head[shell] = (apply_transform(acpc2ctf, verts),
                           np.asarray(mesh.faces))
    tgt_head = {shell: bnd_w_ctf[shell] for shell in ('scalp', 'skull')}

    H = sio.loadmat(HARTMUT_MODEL, struct_as_record=False,
                    squeeze_me=True)['HArtMuT']
    am = H.artefactmodel
    model = {'pos': np.asarray(am.pos, dtype=float),
             'orientation': am.orientation, 'labels': am.labels,
             'unit': am.unit}
    pos_ctf = apply_transform(acpc2ctf, model['pos'])
    mean_pnt = apply_transform(acpc2ctf, HARTMUT_MEANPNT[None, :])[0]

    new_pos_ctf, eye_ctf = warp_hartmut(pos_ctf, model['labels'],
                                        src_head, tgt_head, mean_pnt)
    return new_pos_ctf, eye_ctf, model


def export_npy(bnd_w, transform, output_dir):
    np.save(pth(output_dir, 'pca_warped_bnd.npy'), bnd_w)
    np.save(pth(output_dir, 'pca_warped_bnd_transform.npy'), transform)


def export_cedalion(bnd_w, transform, fiducials, output_dir):
    print('Start cedalion export...')
    # Transform again into ctf and then into RAS (this is necessary for tri2nii)
    # Build the RAS copy locally: bnd_w belongs to the caller, which still
    # needs the digitized-frame vertices for the MNE export afterwards.
    ras2ctf = np.load(ACPC2CTF, allow_pickle=True).astype(float)
    ras2ctf[:3,3] *= 1000 # m -> mm
    ctf2ras = np.linalg.pinv(ras2ctf)
    dig2ras = ctf2ras @ transform  # first to ctf, then ctf2ras
    bnds = [(apply_transform(dig2ras, bnd_w[shell][0]), bnd_w[shell][1])
            for shell in SHELLS]
    cedalion_output_dir = pth(output_dir, 'cedalion')
    os.makedirs(cedalion_output_dir, exist_ok=True)

    # Create segmentation masks
    """
    back_transform = np.linalg.pinv(transform) @ ras2ctf # first back to ctf,
                                                         # then in phtgrammetry
                                                         # coordinate system
    """
    back_transform = np.eye(4) #for RAS
    tri2nii(bnds, output_dir=cedalion_output_dir, transform=back_transform,
            t1_fn=pth(BASEDIR, 'src', 'template.nii'), meshes='all')

    # Postprocessing and clean up
    postprocessing(cedalion_output_dir, num_tissues=4)
    for i in range(5):
        for fn in [f'mask_{i}.nii', f'mask_{i}.nii.gz']:
            if os.path.exists(pth(cedalion_output_dir, fn)):
                os.remove(pth(cedalion_output_dir, fn))

    # Save ditigized2ras transform (photogrammetry to output)
    from_crs, to_crs = "digitized", "ras"
    ditigized2ras = ctf2ras @ transform # first to ctf, then ctf2ras
    fiducials_ras = apply_transform(ditigized2ras, fiducials)
    try:
        import xarray as xr
    except ImportError:
        print('xarray not installed, skipping ditigized2ras export.')
    else:
        ditigized2ras = xr.DataArray(ditigized2ras, dims=[to_crs, from_crs])
        ditigized2ras.to_netcdf(pth(cedalion_output_dir, "t_ditigized2ras.nc"))
  
    # Save landmarks
    import json
    landmarks = [{"id": i,
                  "label": label,
                  "position": list(pos),
                  "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                  }
                 for i, (label, pos) in enumerate(zip(['Nz', 'LPA', 'RPA'],
                                                      fiducials_ras))]
    data_dict = {"@schema": str("https://raw.githubusercontent.com/slicer/"
                              "slicer/master/Modules/Loadable/Markups/"
                              "Resources/Schema/markups-schema-v1.0.3.json"),
                 "markups": [{
                     "type": "Fiducial",
                     "coordinateSystem": to_crs,
                     "coordinateUnits": "mm", #landmark.units,
                     "controlPoints": landmarks,
                     }]}
    json.dump(data_dict, open(pth(cedalion_output_dir, "landmarks.mrk.json"),
                              "w"), indent=2)


def _to_lia(data, affine):
    """Reorient a volume to the LIA voxel order FreeSurfer .mgz files use."""
    from nibabel.orientations import (io_orientation, axcodes2ornt,
                                      ornt_transform, apply_orientation,
                                      inv_ornt_aff)
    xf = ornt_transform(io_orientation(affine), axcodes2ornt(('L', 'I', 'A')))
    return (apply_orientation(data, xf),
            affine @ inv_ornt_aff(xf, data.shape))


def export_mne(bnd_w, transform, output_dir):
    """Write an MNE subject directory: bem/*.surf, mri/T1.mgz and a trans.

    MNE reads bem/*.surf as FreeSurfer surface RAS in mm and expects the
    head->mri transform in metres, so the surfaces are moved out of the
    digitized frame into the surface RAS of the T1 we just wrote, and the
    transform is scaled on the way out.
    """
    print('Start python-MNE export...')
    try:
        import nibabel as nib
        import mne
        from mne.transforms import Transform
    except ImportError:
        print('mne or nibabel not installed, skipping MNE export.')
        return

    mne_output_dir = pth(output_dir, 'mne', 'pcawarp')
    os.makedirs(pth(mne_output_dir, 'bem'), exist_ok=True)
    os.makedirs(pth(mne_output_dir, 'mri'), exist_ok=True)
    mne_names = {'scalp': 'outer_skin.surf',
                 'skull': 'outer_skull.surf',
                 'csf': 'inner_skull.surf',
                 'cortex': 'inner_csf.surf'}

    # Create a fake T1.mgz file from the cedalion output (for MNE plotting)
    data = nib.load(pth(output_dir, 'cedalion', 'mask_skin.nii')).get_fdata()
    new_data = np.zeros(data.shape)
    tissue_color = {'skin': 0.45, # ~0.4–0.6 (depends on fat content)
                    'bone': 0.05, # ~0.0–0.1 (very dark, almost no signal)
                    'csf': 0.1, #CSF ~0.0–0.1 (dark; long T1 -> low signal)
                    'cortex': 0.55} #cortex (0.5–0.7 (wm brighter than gm)
    for lab, tissue in enumerate(['skin', 'bone', 'csf', 'cortex']):
        mask = nib.load(pth(output_dir, 'cedalion', f"mask_{tissue}.nii"))
        new_data[mask.get_fdata() == 1] = tissue_color[tissue]
    # FreeSurfer volumes are stored LIA, and that is what nibabel and MNE
    # assume when they derive the surface RAS ("tkreg") frame from an .mgz.
    # The masks come out of tri2nii in RAS order, so reorient before saving -
    # left as RAS, get_vox2ras_tkr() would describe a mirrored head.
    new_data, t1_affine = _to_lia(new_data, mask.affine)
    t1_fn = pth(mne_output_dir, 'mri', 'T1.mgz')
    nib.save(nib.freesurfer.mghformat.MGHImage(
        new_data.astype(np.float32), t1_affine), t1_fn)

    # digitized -> the surface RAS of that volume
    ras2ctf = np.load(ACPC2CTF, allow_pickle=True).astype(float)
    ras2ctf[:3,3] *= 1000 # m -> mm
    ctf2ras = np.linalg.pinv(ras2ctf)
    t1 = nib.load(t1_fn)
    ras2tkr = t1.header.get_vox2ras_tkr() @ np.linalg.inv(t1.affine)
    ditigized2tkr = ras2tkr @ ctf2ras @ transform

    for shell in SHELLS:
        pos = apply_transform(ditigized2tkr, bnd_w[shell][0])
        nib.freesurfer.io.write_geometry(pth(mne_output_dir, 'bem',
                                             mne_names[shell]),
                                         pos,
                                         orient_outward(pos, bnd_w[shell][1]))

    # Transform object for coregistration. MNE works in metres, so only the
    # translation column changes; the rotation is scale free.
    trans_m = ditigized2tkr.copy()
    trans_m[:3, 3] /= 1000.
    trans = Transform('head', 'mri', trans_m)
    mne.write_trans(pth(mne_output_dir, 'ditigized2ras-trans.fif'), trans,
                    overwrite=True)



# =====================
# Main Workflow
# =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-scalp', type=str, required=True,
                        help='Path to scalp proxy file. Can be a .npy- or a \
                              .txt-file or a surface mesh of typical mesh \
                              formats like .stl, .obj, .ply')
    parser.add_argument('-fiducials', type=str, default=None,
                        help='Path to a landmark file holding NAS, LPA and \
                              RPA: MeshLab .pp, 3D Slicer .mrk.json/.fcsv, or \
                              a text file with one "LABEL x y z" line per \
                              landmark. Omit it and a landmark file sitting \
                              next to the scalp file is picked up \
                              automatically.')
    parser.add_argument('-nas', type=float, nargs=3, default=None,
                        help='Nasion (NAS) fiducial coordinates. Optional; \
                              overrides the landmark file.')
    parser.add_argument('-lpa', type=float, nargs=3, default=None,
                        help='Left preauricular (LPA) fiducial coordinates.')
    parser.add_argument('-rpa', type=float, nargs=3, default=None,
                        help='Right preauricular (RPA) fiducial coordinates.')
    parser.add_argument('--n-points', type=int, default=100,
                        help='Number of scalp proxy points the warp is \
                              fitted to (default 100).')
    parser.add_argument('--sampling', choices=['fps', 'random'],
                        default='fps',
                        help='How to pick those points. "fps" \
                              (farthest-point, the default) is deterministic \
                              and spreads them evenly; "random" reproduces \
                              the old behaviour and needs --seed to be \
                              reproducible at all.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for --sampling random. Unused by fps, \
                              which is deterministic without one.')
    parser.add_argument('--no-qc-gate', action='store_true',
                        help='Export the head model even if the quality \
                              checks fail. qc.json is written either way.')
    parser.add_argument('--regularize', action='store_true',
                        help='EXPERIMENTAL. Penalize shells coming closer \
                              than 5 mm during the fit. Roughly 10x slower, \
                              and the penalty is unweighted against the shape \
                              distance, so it can dominate and blow the fit \
                              up. Off by default.')
    args = parser.parse_args()

    if HARTMUT:
        check_hartmut_available()

    scalp, nas, lpa, rpa = load_scalp_file(args.scalp, args.nas, args.lpa,
                                           args.rpa)
    print('Locating fiducials...')
    if nas is not None and lpa is not None and rpa is not None:
        # the scalp file carried them itself (CapTrak, Polhemus, .elc)
        args.nas, args.lpa, args.rpa = nas, lpa, rpa
    fiducials, fid_source = resolve_fiducials(args, args.scalp)

    scalp, fiducials, unit_scale = to_millimetres(scalp, fiducials)
    spacing = fid_io.validate(fiducials, fid_source)
    print('  NAS %s\n  LPA %s\n  RPA %s' % tuple(
        np.round(f, 2).tolist() for f in fiducials))
    print('  LPA-RPA %.1f mm, NAS-LPA %.1f mm, NAS-RPA %.1f mm'
          % (spacing['lpa-rpa'], spacing['nas-lpa'], spacing['nas-rpa']))

    print('Transforming into CTF coordinate system...')
    mean_scalp = np.load(MEAN_HEAD, allow_pickle=True).item()['scalp'][0]
    scalp, transform = transform_to_ctf(scalp, *fiducials.copy(),
                                        mean_scalp=mean_scalp,
                                        return_transform=True)

    print('Cut scalp proxy points above the ears...')
    CUT = 30#mm
    scalp = scalp[scalp[:, 2] > CUT]
    if len(scalp) < 20:
        raise ValueError(
            f'Only {len(scalp)} scalp proxy points survive the cut {CUT} mm '
            'above the ear plane. The warp has 16 free parameters and needs a '
            'point cloud spread over the upper head, so it would fit noise. '
            'Check that the fiducials belong to this scalp file.')
    spread = float(np.linalg.norm(scalp.max(axis=0) - scalp.min(axis=0)))
    if spread < 100.0:
        raise ValueError(
            f'The scalp proxy spans only {spread:.1f} mm after transforming to '
            'CTF, which is far too small for a head. Check the units and the '
            'fiducials of the scalp file.')
    scalp_full = scalp  # kept for the QC fit residual
    print(f'Reduce scalp proxy to {args.n_points} points '
          f'({args.sampling})...')
    scalp, _ = subsample(scalp, args.n_points, method=args.sampling,
                         seed=args.seed)
    print(f'  worst gap from any scan point to a sampled one: '
          f'{coverage(scalp_full, scalp):.1f} mm')

    print('Performing PCA warping...')
    bnd_w_ctf, pc_weights = pca_surfacemesh_warping(
        fiducials, scalp, regularize=args.regularize)

    output_dir = os.path.dirname(args.scalp)
    inv = np.linalg.pinv(transform)  # CTF -> input frame

    # Measure the warp before spending two minutes exporting it. A run that
    # does not beat the unwarped template has not worked, however plausible
    # the meshes look.
    print('Quality control...')
    qc_report, qc_problems = qc_mod.run(
        bnd_w_ctf, scalp_full, template_scalp=(mean_scalp,
                                               np.load(MEAN_HEAD,
                                                       allow_pickle=True
                                                       ).item()['scalp'][1]),
        x_p=pc_weights, output_dir=output_dir, basedir=BASEDIR,
        provenance={'scalp_file': os.path.abspath(args.scalp),
                    'fiducials': np.asarray(fiducials).tolist(),
                    'fiducial_source': fid_source,
                    'unit_scale_to_mm': unit_scale,
                    'n_points': int(args.n_points),
                    'sampling': args.sampling,
                    'seed': args.seed,
                    'num_pcas': NUM_PCAS,
                    'hartmut': HARTMUT,
                    'regularize': bool(args.regularize)})
    print(qc_mod.summary(qc_report))
    if qc_problems:
        print('\n  QC FAILED:')
        for problem in qc_problems:
            print(f'    - {problem}')
        print(f'  See {pth(output_dir, "qc.json")} and qc.png.')
        if not args.no_qc_gate:
            raise SystemExit('Refusing to export a head model that failed QC. '
                             'Pass --no-qc-gate to export it anyway.')
        print('  --no-qc-gate given, exporting anyway.')
    else:
        print('  QC passed.')

    # Warp the HArtMuT artefact sources (muscle, eyes) into the individual head.
    # Only for the HArtMuT PCAs, the artefact model lives in their template.
    eye_pos = None
    eye_ctf_export = None
    hartmut_ctf = None
    if HARTMUT:
        print('Warping HArtMuT artefact sources into the individual head...')
        new_pos_ctf, eye_ctf, model = warp_artefacts_ctf(bnd_w_ctf)
        eye_pos = apply_transform(inv, eye_ctf) if len(eye_ctf) else None
        eye_ctf_export = eye_ctf if len(eye_ctf) else None
        # the artefact model is exported once, alongside the CTF head
        hartmut_ctf = (new_pos_ctf, model)

    print('Back-transform warped boundaries...')
    bnd_w = {}
    for shell in SHELLS:
        bnd_w[shell] = (apply_transform(inv, bnd_w_ctf[shell][0]),
                        bnd_w_ctf[shell][1])

    # Exports
    print(f"Export (.tri, .stl, .mat, .npy) to {output_dir}/...")
    export_openmeeg(bnd_w, output_dir)
    # export the FieldTrip head in three frames: the input (digitized) frame, CTF, and MNI. CTF and
    # MNI are recognized coordinate systems, so HArtMuT eye fitting works on those two files without
    # any further coordinate transform, and a standard cap aligns with the MNI head
    fiducials_ctf = apply_transform(transform, fiducials)
    ras2ctf = np.load(ACPC2CTF, allow_pickle=True).astype(float)
    ras2ctf[:3, 3] *= 1000  # m -> mm
    ctf2ras = np.linalg.pinv(ras2ctf)
    bnd_w_mni = {shell: (apply_transform(ctf2ras, bnd_w_ctf[shell][0]), bnd_w_ctf[shell][1])
                 for shell in SHELLS}
    eye_mni = apply_transform(ctf2ras, eye_ctf_export) if eye_ctf_export is not None else None
    fiducials_mni = apply_transform(ctf2ras, fiducials_ctf)
    export_fieldtrip(bnd_w, output_dir, coordsys='digitized', eye_pos=eye_pos,
                     fiducials=fiducials, fname='pcawarp_bnd_digitized.mat')
    export_fieldtrip(bnd_w_ctf, output_dir, coordsys='ctf', eye_pos=eye_ctf_export,
                     fiducials=fiducials_ctf, hartmut=hartmut_ctf, fname='pcawarp_bnd_ctf.mat')
    export_fieldtrip(bnd_w_mni, output_dir, coordsys='mni', eye_pos=eye_mni,
                     fiducials=fiducials_mni, fname='pcawarp_bnd_mni.mat')
    export_npy(bnd_w, transform, output_dir)
    export_cedalion(bnd_w, transform, fiducials, output_dir)
    export_mne(bnd_w, transform, output_dir)

    # The mask volume only exists once cedalion has run, so the field-of-view
    # check is folded into qc.json afterwards.
    ras2ctf_fov = np.load(ACPC2CTF, allow_pickle=True).astype(float)
    ras2ctf_fov[:3, 3] *= 1000
    bnd_ras = {shell: (apply_transform(np.linalg.pinv(ras2ctf_fov),
                                       bnd_w_ctf[shell][0]),
                       bnd_w_ctf[shell][1]) for shell in SHELLS}
    fov = qc_mod.field_of_view(bnd_ras, pth(output_dir, 'cedalion',
                                            'mask_skin.nii'))
    if fov is not None:
        qc_report['field_of_view_fraction'] = fov
        worst_shell, worst = min(fov.items(), key=lambda kv: kv[1])
        if worst < 0.999:
            print(f'  NOTE: {100 * (1 - worst):.0f}% of the {worst_shell} '
                  f'falls outside the exported mask volume and is clipped '
                  f'from the .nii masks and T1.mgz.')
        import json as _json
        with open(pth(output_dir, 'qc.json'), 'w') as fid:
            _json.dump(qc_report, fid, indent=2)



if __name__ == '__main__':
    main()

