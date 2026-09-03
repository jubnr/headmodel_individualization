#!/usr/bin/env python
"""Click NAS, LPA and RPA on a head scan and write the landmark file.

MeshLab's PickPoints is fiddly - points only reach disk when you remember to
press Save, and its depth-buffer picking misbehaves under some compositors.
This does the one job PCAwarp needs, writes the same MeshLab .pp format, and
puts it where the pipeline finds it on its own.

    python tools/pick_fiducials.py path/to/cutscan.obj

Left-click the three landmarks in order: nasion, left pre-auricular, right
pre-auricular. Press u to undo the last one, r to start over, q when done.
Coordinates are written in the file's own units, so a scan in metres stays in
metres - PCAwarp rescales it.
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fiducials import PLAUSIBLE_MM, guess_scale_to_mm  # noqa: E402

ORDER = [('NAS', 'nasion - the dip between the eyes, above the nose bridge'),
         ('LPA', 'left pre-auricular - the notch just in front of the LEFT ear'),
         ('RPA', 'right pre-auricular - the same notch on the RIGHT ear')]


def write_pp(path, picked, mesh_name):
    """MeshLab PickedPoints XML, so the file is interchangeable with MeshLab."""
    root = ET.Element('PickedPoints')
    doc = ET.SubElement(root, 'DocumentData')
    ET.SubElement(doc, 'DataFileName', {'name': mesh_name})
    for (label, _), point in zip(ORDER, picked):
        ET.SubElement(root, 'point', {
            'active': '1', 'name': label,
            'x': repr(float(point[0])), 'y': repr(float(point[1])),
            'z': repr(float(point[2]))})
    ET.ElementTree(root).write(path, encoding='UTF-8', xml_declaration=True)


def report(picked, scale):
    """Distances in mm, so an obvious mis-click is visible immediately."""
    nas, lpa, rpa = [np.asarray(p, dtype=float) * scale for p in picked]
    spacing = {'lpa-rpa': np.linalg.norm(lpa - rpa),
               'nas-lpa': np.linalg.norm(nas - lpa),
               'nas-rpa': np.linalg.norm(nas - rpa)}
    lines = []
    for name, value in spacing.items():
        low, high = PLAUSIBLE_MM[name]
        ok = 'ok' if low <= value <= high else f'SUSPECT (expect {low:.0f}-{high:.0f})'
        lines.append(f'  {name}: {value:6.1f} mm   {ok}')
    return '\n'.join(lines)


def set_banner(actor, text):
    """pyvista returns a CornerAnnotation for corner positions and a plain
    text actor otherwise; they take the text differently."""
    if hasattr(actor, 'SetText'):
        actor.SetText(0, text)
    else:
        actor.SetInput(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse
                                     .RawDescriptionHelpFormatter)
    parser.add_argument('scalp', help='the head scan (.obj, .ply, .stl, ...)')
    parser.add_argument('-o', '--out', default=None,
                        help='where to write the landmarks '
                             '(default: <scan>.pp, which PCAwarp finds itself)')
    args = parser.parse_args()

    import pyvista as pv

    out = args.out or os.path.splitext(args.scalp)[0] + '.pp'
    mesh = pv.read(args.scalp)
    scale, unit = guess_scale_to_mm(np.asarray(mesh.points))
    print(f'{args.scalp}: {mesh.n_points} points, looks like {unit}')

    picked = []
    plotter = pv.Plotter(window_size=(1100, 850))
    plotter.add_mesh(mesh, color='linen', smooth_shading=True)
    banner = plotter.add_text('', position='upper_left', font_size=11)
    labels = []

    def prompt():
        if len(picked) < len(ORDER):
            name, hint = ORDER[len(picked)]
            set_banner(banner, f'Click {name}\n{hint}\n\n'
                               f'u = undo   r = reset   q = done')
        else:
            set_banner(banner, 'All three picked.\n' + report(picked, scale)
                       + '\n\nq = save and quit   u = undo   r = reset')
        plotter.render()

    def draw():
        for actor in labels:
            plotter.remove_actor(actor, render=False)
        labels.clear()
        if picked:
            names = [ORDER[i][0] for i in range(len(picked))]
            labels.append(plotter.add_point_labels(
                np.array(picked), names, point_size=18, font_size=20,
                point_color='crimson', text_color='crimson',
                always_visible=True, shape=None, render=False))
        prompt()

    def on_pick(point, *_):
        if len(picked) >= len(ORDER):
            print('  already have three - press u to undo or r to reset')
            return
        picked.append(np.asarray(point, dtype=float))
        print(f'  {ORDER[len(picked) - 1][0]} = '
              f'{np.round(picked[-1], 6).tolist()}')
        draw()

    def undo():
        if picked:
            print(f'  undo {ORDER[len(picked) - 1][0]}')
            picked.pop()
            draw()

    def reset():
        picked.clear()
        print('  reset')
        draw()

    plotter.enable_surface_point_picking(callback=on_pick, show_point=False,
                                         left_clicking=True,
                                         show_message=False)
    plotter.add_key_event('u', undo)
    plotter.add_key_event('r', reset)
    draw()
    print('\nLeft-click the three landmarks in order. u = undo, r = reset, '
          'q = done.\n')
    plotter.show()

    if len(picked) != 3:
        print(f'\nOnly {len(picked)} of 3 landmarks picked - nothing written.')
        return 1

    write_pp(out, picked, os.path.basename(args.scalp))
    print(f'\nWrote {out}')
    print(report(picked, scale))
    print(f'\nNow run:\n  python PCAwarp.py -scalp {args.scalp}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
