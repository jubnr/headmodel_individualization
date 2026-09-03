import os
import sys

import numpy as np
import pytest
import trimesh

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


@pytest.fixture(scope='session')
def repo():
    return REPO


@pytest.fixture
def sphere():
    """A closed, watertight mesh of known volume."""
    def _make(radius=10.0, subdivisions=3, centre=(0.0, 0.0, 0.0)):
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions,
                                          radius=radius)
        return (np.asarray(mesh.vertices) + np.asarray(centre),
                np.asarray(mesh.faces))
    return _make


@pytest.fixture
def nested_shells(sphere):
    """Four concentric spheres, outermost first, like a head model."""
    def _make(radii=(40.0, 34.0, 30.0, 26.0)):
        names = ['scalp', 'skull', 'csf', 'cortex']
        return {name: sphere(radius=r) for name, r in zip(names, radii)}
    return _make


@pytest.fixture
def fiducials_mm():
    """NAS/LPA/RPA with realistic spacing (149 / 131 / 127 mm)."""
    return np.array([[-95.843333, 500.173333, -36.273333],
                     [7.503333, 466.943333, 36.503333],
                     [-4.170000, 454.283333, -111.406667]])
