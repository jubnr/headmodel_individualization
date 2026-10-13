#!/usr/bin/env python
"""Infant head models: warp an age-matched MRI template onto a scalp scan.

The PCA bases PCAwarp uses were trained on adults, and an infant head lies
outside them. This entry point instead deforms the infant template of the
given age (O'Reilly et al. 2021) onto the scan, see src/infant_template.py.
Input handling, quality control and the exporters are shared with PCAwarp.
"""
import argparse
import os
import shutil
from os.path import join as pth

import numpy as np

from PCAwarp import (BASEDIR, load_scalp_file, resolve_fiducials,
                     to_millimetres, export_openmeeg, export_fieldtrip,
                     export_npy)
from src import fiducials as fid_io
from src import infant_template as infant
from src import qc as qc_mod
from src.sampling import subsample, coverage
from src.transform_to_ctf import transform_to_ctf, apply_transform
from src.tri_io import orient_outward


PREFIX = 'infant_warped'


def export_mne_infant(bnd_ctf, template, template_transform, transform,
                      output_dir):
    """Write an MNE subject directory: bem/*.surf, mri/T1.mgz and a trans.

    The surfaces are put back into the template's surface RAS, so the
    template T1 copied alongside lies roughly under them. That T1 is the
    template's, not this infant's - it is there for orientation in plots only.
    """
    print('Start python-MNE export...')
    try:
        import nibabel as nib
        import mne
        from mne.transforms import Transform
    except ImportError:
        print('mne or nibabel not installed, skipping MNE export.')
        return

    mne_output_dir = pth(output_dir, 'mne', 'infantwarp')
    os.makedirs(pth(mne_output_dir, 'bem'), exist_ok=True)
    os.makedirs(pth(mne_output_dir, 'mri'), exist_ok=True)

    ctf2mri = np.linalg.inv(template_transform)
    for shell, fname in infant.SURFACES.items():
        pos = apply_transform(ctf2mri, bnd_ctf[shell][0])
        nib.freesurfer.io.write_geometry(pth(mne_output_dir, 'bem', fname),
                                         pos,
                                         orient_outward(pos,
                                                        bnd_ctf[shell][1]))
    shutil.copy(pth(template['subject_dir'], 'mri', 'T1.mgz'),
                pth(mne_output_dir, 'mri', 'T1.mgz'))

    # digitized -> CTF -> template surface RAS, in metres for MNE
    trans_m = ctf2mri @ transform
    trans_m[:3, 3] /= 1000.
    mne.write_trans(pth(mne_output_dir, 'digitized2mri-trans.fif'),
                    Transform('head', 'mri', trans_m), overwrite=True)
    print(f'  mri/T1.mgz is the {template["age"]} template, not this '
          'infant: use it for orientation only.')


def main():
    parser = argparse.ArgumentParser(
        description='Individualized 3-shell BEM head model for an infant, '
                    'from photogrammetry or digitized electrode positions.')
    parser.add_argument('-scalp', type=str, required=True,
                        help='Path to scalp proxy file, any format PCAwarp '
                             'reads.')
    parser.add_argument('--age', required=True, choices=infant.AGES,
                        help='Template age. Use the corrected age for infants '
                             'born preterm.')
    parser.add_argument('--subjects-dir', default=None,
                        help='Where the infant templates are downloaded to '
                             '(default ~/mne_data/infant).')
    parser.add_argument('-fiducials', type=str, default=None,
                        help='Landmark file holding NAS, LPA and RPA. Omit it '
                             'and one next to the scalp file is picked up.')
    parser.add_argument('-nas', type=float, nargs=3, default=None,
                        help='Nasion coordinates. Overrides the landmark file.')
    parser.add_argument('-lpa', type=float, nargs=3, default=None,
                        help='Left preauricular coordinates.')
    parser.add_argument('-rpa', type=float, nargs=3, default=None,
                        help='Right preauricular coordinates.')
    parser.add_argument('--n-points', type=int, default=100,
                        help='Number of scalp proxy points the warp is fitted '
                             'to (default 100, farthest-point sampled).')
    parser.add_argument('--smoothing', type=float, default=1.0,
                        help='Thin-plate spline smoothing. 0 interpolates the '
                             'sampled points exactly, noise included; larger '
                             'values give a smoother, less exact scalp '
                             '(default 1).')
    parser.add_argument('--no-qc-gate', action='store_true',
                        help='Export the head model even if the quality '
                             'checks fail. qc.json is written either way.')
    args = parser.parse_args()

    print(f'Loading the {args.age} infant template...')
    template = infant.load_template(args.age, args.subjects_dir)
    template_ctf, template_transform = infant.to_ctf(template)
    print(f'  {template["subject"]} from {template["subjects_dir"]}')

    scalp, nas, lpa, rpa = load_scalp_file(args.scalp, args.nas, args.lpa,
                                           args.rpa)
    print('Locating fiducials...')
    if nas is not None and lpa is not None and rpa is not None:
        args.nas, args.lpa, args.rpa = nas, lpa, rpa
    fiducials, fid_source = resolve_fiducials(args, args.scalp)

    scalp, fiducials, unit_scale = to_millimetres(scalp, fiducials)
    spacing = fid_io.validate(
        fiducials, fid_source,
        plausible=infant.plausible_ranges(template['fiducials']))
    expected = infant.landmark_spacing(template['fiducials'])
    print('  LPA-RPA %.1f mm, NAS-LPA %.1f mm, NAS-RPA %.1f mm'
          % (spacing['lpa-rpa'], spacing['nas-lpa'], spacing['nas-rpa']))
    print('  template: LPA-RPA %.1f mm, NAS-LPA %.1f mm, NAS-RPA %.1f mm'
          % (expected['lpa-rpa'], expected['nas-lpa'], expected['nas-rpa']))

    print('Transforming into CTF coordinate system...')
    scalp, transform = transform_to_ctf(scalp, *fiducials.copy(),
                                        mean_scalp=template_ctf['scalp'][0],
                                        return_transform=True)

    cut = infant.ear_plane_cut(template['fiducials'])
    print(f'Cut scalp proxy points {cut:.0f} mm above the ears...')
    scalp = scalp[scalp[:, 2] > cut]
    if len(scalp) < 20:
        raise ValueError(
            f'Only {len(scalp)} scalp proxy points survive the cut {cut:.0f} '
            'mm above the ear plane, too few to fit a warp to. Check that the '
            'fiducials belong to this scalp file.')
    template_top = template_ctf['scalp'][0]
    template_top = template_top[template_top[:, 2] > cut]
    template_spread = float(np.linalg.norm(template_top.max(axis=0)
                                           - template_top.min(axis=0)))
    spread = float(np.linalg.norm(scalp.max(axis=0) - scalp.min(axis=0)))
    if spread < 0.6 * template_spread:
        raise ValueError(
            f'The scalp proxy spans only {spread:.1f} mm after transforming '
            f'to CTF, against {template_spread:.1f} mm for the {args.age} '
            'template. Check the units, the fiducials and the template age.')
    scalp_full = scalp
    print(f'Reduce scalp proxy to {args.n_points} points (fps)...')
    scalp, _ = subsample(scalp, args.n_points)
    print(f'  worst gap from any scan point to a sampled one: '
          f'{coverage(scalp_full, scalp):.1f} mm')

    print('Warping the template...')
    bnd_affine, bnd_ctf, fit = infant.warp(template_ctf, scalp, cut,
                                           smoothing=args.smoothing)
    scales = fit['affine_scales']
    print('  affine stretch along principal axes: '
          + ', '.join(f'{s:.2f}' for s in scales))
    if min(scales) < 0.8 or max(scales) > 1.25:
        print(f'  WARNING: the {args.age} template had to be stretched by '
              'more than 20% to fit; check the age, the units and the '
              'fiducials.')
    fit['affine_scalp_fit'] = qc_mod.scalp_fit(bnd_affine,
                                               scalp_full)['warped']
    print(f'  affine stage alone: median '
          f'{fit["affine_scalp_fit"]["median_abs"]:.2f} mm')
    for shell, moved in fit['nesting_vertices_moved'].items():
        if moved:
            print(f'  {shell}: pulled {moved} vertices back inside the shell '
                  'around it.')

    output_dir = pth(os.path.dirname(args.scalp), f'infantwarp_{args.age}')
    os.makedirs(output_dir, exist_ok=True)

    print('Quality control...')
    qc_report, qc_problems = qc_mod.run(
        bnd_ctf, scalp_full, template_scalp=template_ctf['scalp'],
        output_dir=output_dir, basedir=BASEDIR,
        provenance={'method': 'infant template warp',
                    'template': template['subject'],
                    'template_age': args.age,
                    'scalp_file': os.path.abspath(args.scalp),
                    'fiducials': np.asarray(fiducials).tolist(),
                    'fiducial_source': fid_source,
                    'unit_scale_to_mm': unit_scale,
                    'cut_mm': cut,
                    'n_points': int(args.n_points),
                    'smoothing': args.smoothing,
                    'fit': fit})
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

    print('Back-transform warped boundaries...')
    inv = np.linalg.pinv(transform)
    bnd = {shell: (apply_transform(inv, pos), tris)
           for shell, (pos, tris) in bnd_ctf.items()}

    print(f'Export (.tri, .stl, .mat, .npy, MNE) to {output_dir}/...')
    export_openmeeg(bnd, output_dir, shells=infant.SHELLS, prefix=PREFIX)
    export_fieldtrip(bnd, output_dir, coordsys='digitized',
                     fiducials=fiducials, shells=infant.SHELLS,
                     fname=f'{PREFIX}_bnd_digitized.mat')
    export_fieldtrip(bnd_ctf, output_dir, coordsys='ctf',
                     fiducials=apply_transform(transform, fiducials),
                     shells=infant.SHELLS, fname=f'{PREFIX}_bnd_ctf.mat')
    export_npy(bnd, transform, output_dir, prefix=PREFIX)
    export_mne_infant(bnd_ctf, template, template_transform, transform,
                      output_dir)
    print('  Not exported for infants: the MNI head and the cedalion masks, '
          'which both rest on an adult template.')


if __name__ == '__main__':
    main()
