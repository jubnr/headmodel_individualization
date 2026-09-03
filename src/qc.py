#!/usr/bin/env python
"""Quality control for a PCAwarp run.

A warp can fail silently. The failure that prompted this module produced four
watertight, properly nested, entirely plausible-looking surfaces that were
*worse than not warping at all* — the fiducials had been given as vertex
indices, so the fit had one usable proxy point. Nothing in the pipeline
noticed, it exited 0, and the problem surfaced only because someone measured
the result by hand a week later.

So every run now measures its own output and says so. The single most useful
number is the scalp fit residual against the full input cloud, compared with
the same residual for the unwarped template: a warp that does not beat the
population mean has not worked, whatever the meshes look like.
"""
import json
import subprocess
from os.path import join as pth

import numpy as np
import trimesh

from src.tri_io import signed_volume

# Ordered outside-in; nesting is checked between consecutive pairs.
SHELL_ORDER = ['scalp', 'skull', 'csf', 'cortex']


def _mesh(bnd_shell):
    verts, tris = bnd_shell
    return trimesh.Trimesh(np.asarray(verts, dtype=float),
                           np.asarray(tris), process=False)


def _signed_distance(mesh, points):
    """Distance from each point to the surface, negative inside."""
    _, dist, _ = trimesh.proximity.closest_point(mesh, points)
    return np.where(mesh.contains(points), -dist, dist)


def _stats(values):
    values = np.asarray(values, dtype=float)
    absolute = np.abs(values)
    return {'mean_abs': float(absolute.mean()),
            'median_abs': float(np.median(absolute)),
            'rms': float(np.sqrt((values ** 2).mean())),
            'p95_abs': float(np.percentile(absolute, 95)),
            'max_abs': float(absolute.max()),
            'signed_mean': float(values.mean()),
            'frac_inside': float((values < 0).mean())}


def _git_sha(basedir):
    try:
        return subprocess.run(['git', '-C', basedir, 'rev-parse', 'HEAD'],
                              capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:
        return None


# =====================
# Individual checks
# =====================
def scalp_fit(bnd, scalp_points, template_scalp=None):
    """Residual between the fitted scalp and the cloud it was fitted to.

    `template_scalp` is the unwarped mean head in the same frame. Including it
    turns an unanchored number into a verdict: the warp is supposed to beat it.
    """
    out = {'n_points': int(len(scalp_points)),
           'warped': _stats(_signed_distance(_mesh(bnd['scalp']),
                                             scalp_points))}
    if template_scalp is not None:
        out['template_baseline'] = _stats(
            _signed_distance(_mesh(template_scalp), scalp_points))
        out['improvement_ratio'] = (out['template_baseline']['median_abs']
                                    / max(out['warped']['median_abs'], 1e-9))
    return out


def shell_geometry(bnd):
    """Topology and size of each surface."""
    out = {}
    for shell in SHELL_ORDER:
        if shell not in bnd:
            continue
        mesh = _mesh(bnd[shell])
        areas = mesh.area_faces
        out[shell] = {
            'n_vertices': int(len(mesh.vertices)),
            'n_triangles': int(len(mesh.faces)),
            'watertight': bool(mesh.is_watertight),
            'winding_consistent': bool(mesh.is_winding_consistent),
            'euler_number': int(mesh.euler_number),
            'signed_volume_cm3': signed_volume(*bnd[shell]) / 1000.0,
            'area_cm2': float(mesh.area) / 100.0,
            'min_triangle_area_mm2': float(areas.min()),
        }
    return out


def nesting(bnd, pairs=None):
    """Gap between each shell and the one enclosing it."""
    if pairs is None:
        pairs = list(zip(SHELL_ORDER[1:], SHELL_ORDER[:-1]))  # inner, outer
    out = {}
    for inner, outer in pairs:
        if inner not in bnd or outer not in bnd:
            continue
        gap = -_signed_distance(_mesh(bnd[outer]),
                                np.asarray(bnd[inner][0], dtype=float))
        out[f'{inner}_in_{outer}'] = {
            'min_gap_mm': float(gap.min()),
            'median_gap_mm': float(np.median(gap)),
            'n_crossing': int((gap <= 0).sum()),
        }
    return out


def field_of_view(bnd, mask_path):
    """Fraction of each shell that fits inside the exported mask volume.

    The template volume is a fixed size; a neck-extended scalp can run off the
    bottom of it, and the exported T1 then shows a head with the neck cut flat.
    """
    try:
        import nibabel as nib
        img = nib.load(mask_path)
    except Exception:
        return None
    inv = np.linalg.inv(img.affine)
    shape = np.array(img.shape)
    out = {}
    for shell in SHELL_ORDER:
        if shell not in bnd:
            continue
        ijk = nib.affines.apply_affine(inv,
                                       np.asarray(bnd[shell][0], dtype=float))
        inside = np.all((ijk >= 0) & (ijk < shape), axis=1)
        out[shell] = float(inside.mean())
    return out


def pc_weights(x_p):
    """The fitted PCA coefficients.

    These are coefficients on unit-norm eigenvectors over all ~24 000 mesh
    coordinates, not multiples of a standard deviation: a coefficient of 78
    moves the surface by about 2 mm RMS. They are recorded so a fit can be
    reproduced or compared, but they are not interpretable on their own -
    `displacement_from_template` is the number to read.
    """
    x_p = np.asarray(x_p, dtype=float)
    return {'weights': x_p.tolist(), 'max_abs': float(np.abs(x_p).max())}


def displacement_from_template(bnd, template, shell='scalp'):
    """How far the warp moved a shell away from the population mean, in mm.

    Same triangulation on both sides, so this is a straight per-vertex
    distance. It is the physically meaningful counterpart to the raw PCA
    coefficients.
    """
    warped = np.asarray(bnd[shell][0], dtype=float)
    mean = np.asarray(template[0], dtype=float)
    if warped.shape != mean.shape:
        return None
    distance = np.linalg.norm(warped - mean, axis=1)
    return {'shell': shell,
            'rms_mm': float(np.sqrt((distance ** 2).mean())),
            'median_mm': float(np.median(distance)),
            'max_mm': float(distance.max())}


# =====================
# Verdict
# =====================
def verdict(report):
    """Hard failures that should stop a run being used. Returns a list of
    human-readable problems; empty means the run is usable."""
    problems = []
    for shell, geom in report.get('shells', {}).items():
        if not geom['watertight']:
            problems.append(f'{shell} surface is not watertight')
        if geom['euler_number'] != 2:
            problems.append(f'{shell} surface has Euler number '
                            f'{geom["euler_number"]}, expected 2 (holes or '
                            f'handles)')
        if geom['signed_volume_cm3'] <= 0:
            problems.append(f'{shell} normals point inward')
    for pair, gap in report.get('nesting', {}).items():
        if gap['n_crossing']:
            problems.append(f'{pair}: {gap["n_crossing"]} vertices cross, '
                            f'worst {gap["min_gap_mm"]:.2f} mm')
    fit = report.get('scalp_fit', {})
    if 'template_baseline' in fit:
        if fit['warped']['median_abs'] >= fit['template_baseline']['median_abs']:
            problems.append(
                'the warped scalp fits the scan no better than the unwarped '
                f'template ({fit["warped"]["median_abs"]:.2f} mm vs '
                f'{fit["template_baseline"]["median_abs"]:.2f} mm) - the fit '
                'did not work')
    return problems


# =====================
# Entry point
# =====================
def run(bnd, scalp_points, template_scalp=None, x_p=None, mask_path=None,
        provenance=None, output_dir=None, basedir=None, make_figure=True):
    """Measure a finished warp and write qc.json (+ qc.png) next to it.

    All meshes and points must be in one common frame (PCAwarp uses CTF).
    Returns (report, problems).
    """
    report = {
        'scalp_fit': scalp_fit(bnd, scalp_points, template_scalp),
        'shells': shell_geometry(bnd),
        'nesting': nesting(bnd),
    }
    if x_p is not None:
        report['pc_weights'] = pc_weights(x_p)
    if template_scalp is not None:
        moved = displacement_from_template(bnd, template_scalp)
        if moved is not None:
            report['displacement_from_template'] = moved
    if mask_path is not None:
        fov = field_of_view(bnd, mask_path)
        if fov is not None:
            report['field_of_view_fraction'] = fov
    report['provenance'] = dict(provenance or {})
    if basedir:
        report['provenance']['git_sha'] = _git_sha(basedir)

    problems = verdict(report)
    report['problems'] = problems
    report['passed'] = not problems

    if output_dir is not None:
        with open(pth(output_dir, 'qc.json'), 'w') as fid:
            json.dump(report, fid, indent=2)
        if make_figure:
            try:
                figure(bnd, scalp_points, report, template_scalp,
                       pth(output_dir, 'qc.png'))
            except Exception as exc:  # a missing backend must not fail a run
                print(f'  (qc figure skipped: {type(exc).__name__}: {exc})')
    return report, problems


def summary(report):
    """A compact stdout table. Printed on every run, pass or fail."""
    lines = []
    fit = report['scalp_fit']
    w = fit['warped']
    lines.append(f"  scalp fit vs {fit['n_points']} scan points: "
                 f"median {w['median_abs']:.2f} mm, p95 {w['p95_abs']:.2f} mm,"
                 f" max {w['max_abs']:.2f} mm")
    if 'template_baseline' in fit:
        b = fit['template_baseline']
        lines.append(f"    unwarped template baseline:          "
                     f"median {b['median_abs']:.2f} mm, p95 "
                     f"{b['p95_abs']:.2f} mm   "
                     f"({fit['improvement_ratio']:.1f}x better)")
    lines.append(f"  {'shell':8s}{'volume':>11s}{'watertight':>12s}"
                 f"{'euler':>7s}{'gap to outer':>15s}")
    for shell in SHELL_ORDER:
        if shell not in report['shells']:
            continue
        g = report['shells'][shell]
        gap = ''
        for pair, val in report['nesting'].items():
            if pair.startswith(shell + '_in_'):
                gap = f"{val['min_gap_mm']:.2f} mm"
        lines.append(f"  {shell:8s}{g['signed_volume_cm3']:9.1f} cm3"
                     f"{str(g['watertight']):>12s}{g['euler_number']:>7d}"
                     f"{gap:>15s}")
    moved = report.get('displacement_from_template')
    if moved:
        lines.append(f"  warp moved the scalp {moved['rms_mm']:.1f} mm RMS "
                     f"({moved['max_mm']:.1f} mm max) from the mean head")
    if 'field_of_view_fraction' in report:
        worst = min(report['field_of_view_fraction'].items(),
                    key=lambda kv: kv[1])
        lines.append(f"  inside the exported volume: {100*worst[1]:.0f}% of "
                     f"the {worst[0]} (worst shell)")
    return '\n'.join(lines)


def figure(bnd, scalp_points, report, template_scalp, path):
    """One page: fit residuals, shell outlines, and where the error sits."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colours = {'scalp': 'tab:red', 'skull': 'tab:orange',
               'csf': 'tab:cyan', 'cortex': 'tab:purple'}
    fig = plt.figure(figsize=(15, 8.5))
    grid = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.26)

    # A. residual histogram, with the template as the thing to beat
    ax = fig.add_subplot(grid[0, 0])
    warped = _signed_distance(_mesh(bnd['scalp']), scalp_points)
    ax.hist(warped, bins=80, histtype='step', lw=1.9, color='tab:green',
            label=f"warped  (median "
                  f"{report['scalp_fit']['warped']['median_abs']:.2f} mm)")
    if template_scalp is not None:
        base = _signed_distance(_mesh(template_scalp), scalp_points)
        ax.hist(base, bins=80, histtype='step', lw=1.6, color='0.55',
                label=f"template (median "
                      f"{report['scalp_fit']['template_baseline']['median_abs']:.2f} mm)")
    ax.axvline(0, c='k', lw=0.8, ls='--')
    ax.set_xlabel('signed scan-to-scalp distance [mm]')
    ax.set_ylabel('# scan points')
    ax.legend(fontsize=8)
    ax.set_title('A  Scalp fit', fontsize=11, loc='left')

    # B, C. shell outlines through the middle of the head
    for col, (cut, a, b, title, xlabel, ylabel) in enumerate([
            (1, 0, 2, 'B  sagittal', 'x  P->A [mm]', 'z  I->S [mm]'),
            (0, 1, 2, 'C  coronal', 'y  R->L [mm]', 'z  I->S [mm]')]):
        ax = fig.add_subplot(grid[0, 1 + col])
        sel = np.abs(scalp_points[:, cut]) < 10
        ax.plot(scalp_points[sel, a], scalp_points[sel, b], '.', ms=1.4,
                c='tab:blue', label='scan')
        for shell in SHELL_ORDER:
            if shell not in bnd:
                continue
            pos = np.asarray(bnd[shell][0])
            sel = np.abs(pos[:, cut]) < 10
            if sel.sum() < 8:
                continue
            q = pos[sel]
            centre = q[:, [a, b]].mean(axis=0)
            order = np.argsort(np.arctan2(q[:, b] - centre[1],
                                          q[:, a] - centre[0]))
            ax.plot(np.r_[q[order, a], q[order[0], a]],
                    np.r_[q[order, b], q[order[0], b]],
                    c=colours[shell], lw=1.6, label=shell)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11, loc='left')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if col == 0:
            ax.legend(fontsize=7, loc='lower left')

    # D. where on the head the error sits
    ax = fig.add_subplot(grid[1, 0])
    sc = ax.scatter(scalp_points[:, 0], scalp_points[:, 1],
                    c=np.abs(warped), s=2, cmap='inferno',
                    vmin=0, vmax=max(10.0,
                                     report['scalp_fit']['warped']['p95_abs']))
    plt.colorbar(sc, ax=ax, label='|error| [mm]')
    ax.set_aspect('equal')
    ax.set_title('D  Error over the scalp (top view)', fontsize=11, loc='left')
    ax.set_xlabel('x  P->A [mm]')
    ax.set_ylabel('y  R->L [mm]')

    # E. layer thickness
    ax = fig.add_subplot(grid[1, 1])
    labels, data = [], []
    for pair, val in report['nesting'].items():
        inner, outer = pair.split('_in_')
        gap = -_signed_distance(_mesh(bnd[outer]),
                                np.asarray(bnd[inner][0], dtype=float))
        data.append(gap)
        labels.append(f'{outer}\nto {inner}')
    if data:
        # matplotlib renamed boxplot's `labels` to `tick_labels` in 3.9
        try:
            box = ax.boxplot(data, tick_labels=labels, showfliers=False,
                             patch_artist=True)
        except TypeError:
            box = ax.boxplot(data, labels=labels, showfliers=False,
                             patch_artist=True)
        for patch in box['boxes']:
            patch.set_facecolor('0.85')
    ax.axhline(0, c='tab:red', lw=1.0, ls='--')
    ax.set_ylabel('layer thickness [mm]')
    ax.set_title('E  Nesting (below the red line = crossing)', fontsize=11,
                 loc='left')
    ax.tick_params(axis='x', labelsize=8)

    # F. the fitted PCA weights
    ax = fig.add_subplot(grid[1, 2])
    if 'pc_weights' in report:
        w = np.asarray(report['pc_weights']['weights'])
        ax.bar(np.arange(1, len(w) + 1), w, color='tab:blue')
        ax.axhline(0, c='k', lw=0.8)
        ax.set_xlabel('principal component')
        ax.set_ylabel('coefficient')
    moved = report.get('displacement_from_template')
    title = 'F  Fitted PC coefficients'
    if moved:
        title += (f"  -  scalp moved {moved['rms_mm']:.1f} mm RMS, "
                  f"{moved['max_mm']:.1f} mm max from the mean head")
    ax.set_title(title, fontsize=9.5, loc='left')

    status = 'PASSED' if report['passed'] else 'FAILED'
    fig.suptitle(f'PCAwarp QC  -  {status}', fontsize=13,
                 color='tab:green' if report['passed'] else 'tab:red')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return path
