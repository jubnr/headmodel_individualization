#!/usr/bin/env python
"""Finding, reading and sanity-checking the NAS/LPA/RPA landmarks.

The three fiducials define the CTF frame the PCA database lives in, so a
mistake here silently invalidates the whole warp. This module therefore does
three things:

* it looks for the landmarks next to the scalp file instead of asking the user
  to retype them (MeshLab .pp, Slicer .mrk.json/.fcsv, plain text),
* it accepts vertex indices as well as coordinates, because most mesh pickers
  show both and it is easy to copy the wrong one,
* it checks the result against the geometry of a human head and refuses
  implausible landmarks rather than warping to them.
"""
import json
import os
import re
import xml.etree.ElementTree as ET

import numpy as np


# Label spellings seen in the wild, lowercased and stripped of separators.
LABELS = {
    'nas': ('nas', 'nasion', 'nz', 'n', 'fidnz', 'nose'),
    'lpa': ('lpa', 'l', 'left', 'leftear', 'lefttragus', 'lpaleft', 'fidt9',
            't9', 'al', 'earl'),
    'rpa': ('rpa', 'r', 'right', 'rightear', 'righttragus', 'rparight',
            'fidt10', 't10', 'ar', 'earr'),
}
ORDER = ('nas', 'lpa', 'rpa')

# Plausible landmark geometry of an adult head, in mm. Used to catch unit
# errors and index-instead-of-coordinate mix-ups.
PLAUSIBLE_MM = {'lpa-rpa': (110.0, 200.0),
                'nas-lpa': (85.0, 175.0),
                'nas-rpa': (85.0, 175.0)}

FIDUCIAL_EXTS = ('.pp', '.mrk.json', '.fcsv', '.json', '.txt', '.csv', '.tsv')


def _norm(label):
    return re.sub(r'[^a-z0-9]', '', str(label).lower())


def _match(label):
    """Map a free-form label onto 'nas' / 'lpa' / 'rpa', or None."""
    key = _norm(label)
    for name, spellings in LABELS.items():
        if key in spellings:
            return name
    return None


# =====================
# Readers
# =====================
def _read_pp(filepath):
    """MeshLab PickPoints. Points are often named '0', '1', '2', in which case
    we fall back to picking order: NAS, LPA, RPA."""
    root = ET.parse(filepath).getroot()
    named, ordered = {}, []
    for pnt in root.iter('point'):
        if pnt.get('active') == '0':
            continue
        xyz = [float(pnt.get(ax)) for ax in ('x', 'y', 'z')]
        ordered.append(xyz)
        name = _match(pnt.get('name', ''))
        if name is not None:
            named[name] = xyz
    return named if len(named) == 3 else _by_order(ordered, filepath)


def _read_mrk_json(filepath):
    """3D Slicer markups, the format PCAwarp itself writes as landmarks.mrk.json."""
    with open(filepath) as fid:
        data = json.load(fid)
    named, ordered = {}, []
    for markup in data.get('markups', [{}]):
        for pnt in markup.get('controlPoints', []):
            xyz = [float(v) for v in pnt['position']]
            ordered.append(xyz)
            name = _match(pnt.get('label', ''))
            if name is not None:
                named[name] = xyz
    return named if len(named) == 3 else _by_order(ordered, filepath)


def _read_json(filepath):
    """Plain {"nas": [x, y, z], ...} or a Slicer markups file."""
    with open(filepath) as fid:
        data = json.load(fid)
    if isinstance(data, dict) and 'markups' in data:
        return _read_mrk_json(filepath)
    named, ordered = {}, []
    if isinstance(data, dict):
        for label, xyz in data.items():
            name = _match(label)
            if name is not None and len(xyz) == 3:
                named[name] = [float(v) for v in xyz]
    elif isinstance(data, list):
        ordered = [[float(v) for v in xyz] for xyz in data if len(xyz) == 3]
    return named if len(named) == 3 else _by_order(ordered, filepath)


def _read_fcsv(filepath):
    """3D Slicer legacy fiducial csv: id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,..."""
    named, ordered = {}, []
    for line in open(filepath):
        if line.startswith('#') or not line.strip():
            continue
        cols = line.strip().split(',')
        if len(cols) < 4:
            continue
        xyz = [float(v) for v in cols[1:4]]
        ordered.append(xyz)
        label = cols[11] if len(cols) > 11 else cols[0]
        name = _match(label)
        if name is not None:
            named[name] = xyz
    return named if len(named) == 3 else _by_order(ordered, filepath)


def _read_text(filepath):
    """'LABEL x y z' per line (any of space/comma/tab/colon), or three bare rows."""
    named, ordered = {}, []
    for line in open(filepath):
        line = line.strip()
        if not line or line.startswith(('#', '//')):
            continue
        parts = [p for p in re.split(r'[\s,;:]+', line) if p]
        nums = []
        for part in parts:
            try:
                nums.append(float(part))
            except ValueError:
                pass
        if len(nums) < 3:
            continue
        xyz = nums[-3:]
        ordered.append(xyz)
        if len(parts) > 3:
            name = _match(parts[0])
            if name is not None:
                named[name] = xyz
    return named if len(named) == 3 else _by_order(ordered, filepath)


def _by_order(ordered, filepath):
    if len(ordered) != 3:
        raise ValueError(
            f'{os.path.basename(filepath)} holds {len(ordered)} unlabelled '
            'points; expected exactly 3 (NAS, LPA, RPA in that order) or '
            'labels naming them.')
    print(f'  {os.path.basename(filepath)}: points are unlabelled, reading '
          'them in picking order as NAS, LPA, RPA.')
    return dict(zip(ORDER, ordered))


READERS = {'.pp': _read_pp, '.mrk.json': _read_mrk_json, '.fcsv': _read_fcsv,
           '.json': _read_json, '.txt': _read_text, '.csv': _read_text,
           '.tsv': _read_text}


def read_fiducial_file(filepath):
    """Read NAS/LPA/RPA from a landmark file. Returns a (3, 3) array."""
    lower = filepath.lower()
    ext = next((e for e in ('.mrk.json',) if lower.endswith(e)),
               os.path.splitext(lower)[1])
    if ext not in READERS:
        raise ValueError(f'Unsupported fiducial file format: {ext}')
    named = READERS[ext](filepath)
    missing = [n for n in ORDER if n not in named]
    if missing:
        raise ValueError(f'{os.path.basename(filepath)} is missing '
                         f'{", ".join(m.upper() for m in missing)}.')
    return np.array([named[n] for n in ORDER], dtype=float)


# =====================
# Discovery
# =====================
def find_fiducial_file(scalp_path):
    """Look for a landmark file sitting next to the scalp file.

    Tried in order: <stem>.pp / <stem>.mrk.json / ..., then <stem> with a
    '_fiducials' or '_landmarks' suffix, then the well-known generic names,
    then a lone .pp anywhere in that directory.
    """
    folder = os.path.dirname(os.path.abspath(scalp_path)) or '.'
    stem = os.path.splitext(os.path.basename(scalp_path))[0]

    candidates = [stem + ext for ext in FIDUCIAL_EXTS]
    for suffix in ('_fiducials', '_landmarks', '-fiducials', '-landmarks'):
        candidates += [stem + suffix + ext for ext in FIDUCIAL_EXTS]
    candidates += ['fiducials' + ext for ext in FIDUCIAL_EXTS]
    candidates += ['landmarks' + ext for ext in FIDUCIAL_EXTS]
    candidates += ['picked_points.pp', 'landmarks.mrk.json']

    for name in candidates:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path

    loose = sorted(f for f in os.listdir(folder) if f.lower().endswith('.pp'))
    return os.path.join(folder, loose[0]) if len(loose) == 1 else None


# =====================
# Units and sanity
# =====================
def guess_scale_to_mm(points):
    """Return the factor turning `points` into millimetres.

    A human head is roughly 200-350 mm across, so the bounding-box diagonal
    of a head scan places the unit unambiguously as long as we only consider
    metres, centimetres and millimetres.
    """
    points = np.asarray(points, dtype=float)
    diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if diag == 0.0:
        return 1.0, 'mm'
    if diag < 3.0:
        return 1000.0, 'm'
    if diag < 60.0:
        return 10.0, 'cm'
    if diag > 3000.0:
        raise ValueError(
            f'Scalp proxy spans {diag:.0f} units, which is not a head in any '
            'sane unit. Check the file.')
    return 1.0, 'mm'


def as_vertex_indices(value, vertices):
    """Interpret a fiducial argument as vertex indices, if that is what it is.

    Mesh pickers show a vertex index next to the coordinates and the two are
    easy to confuse. Three whole numbers that all index the mesh, all point at
    vertices within a few mm of each other, and lie outside the mesh itself,
    are indices - the picked point is their centroid.
    """
    value = np.asarray(value, dtype=float)
    if vertices is None or value.shape != (3,):
        return None
    if not np.all(value == np.round(value)) or np.any(value < 0):
        return None
    idx = value.astype(int)
    if np.any(idx >= len(vertices)):
        return None
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    if np.all((value >= lo) & (value <= hi)):
        return None  # a valid coordinate inside the mesh; leave it alone
    picked = vertices[idx]
    if np.linalg.norm(picked - picked.mean(axis=0), axis=1).max() > 5.0:
        return None  # not three picks of the same landmark
    return picked.mean(axis=0)


def validate(fiducials, source=''):
    """Raise if the three landmarks cannot belong to one human head (mm)."""
    nas, lpa, rpa = np.asarray(fiducials, dtype=float)
    got = {'lpa-rpa': np.linalg.norm(lpa - rpa),
           'nas-lpa': np.linalg.norm(nas - lpa),
           'nas-rpa': np.linalg.norm(nas - rpa)}
    bad = [f'{k} = {v:.1f} mm (expected {PLAUSIBLE_MM[k][0]:.0f}-'
           f'{PLAUSIBLE_MM[k][1]:.0f} mm)'
           for k, v in got.items()
           if not PLAUSIBLE_MM[k][0] <= v <= PLAUSIBLE_MM[k][1]]
    if bad:
        raise ValueError(
            'Implausible fiducials' + (f' from {source}' if source else '')
            + ':\n  ' + '\n  '.join(bad)
            + '\nNAS/LPA/RPA must be coordinates in the same frame and unit as '
              'the scalp file. Vertex indices, pixel positions and mixed units '
              'all land here.')
    return got
