from amcheck import __version__

from amcheck import check_altermagnetism_orbit
from amcheck import is_altermagnet

import numpy as np


def test_version():
    assert __version__ == '0.1.0'


def test_check_altermagnetism_orbit():
    symops = [(np.array([[-1,  0,  0], [0, -1,  0], [0,  0, -1]], dtype=int),
               np.array([2.e-06, 2.e-06, 0.e+00]))]
    positions = np.array([[0.156793, 0.313586, 0.75],
                          [0.843207, 0.686414, 0.25],
                          [0.686414, 0.843207, 0.75],
                          [0.313586, 0.156793, 0.25],
                          [0.156793, 0.843207, 0.75],
                          [0.843207, 0.156793, 0.25]])

    spins = ['u', 'd', 'u', 'd', 'u', 'd']
    afm = check_altermagnetism_orbit(symops, positions, spins)
    assert afm == False

    spins = ['u', 'u', 'u', 'd', 'd', 'd']
    am = check_altermagnetism_orbit(symops, positions, spins)
    assert am == True


def test_is_altermagnet():
    symops = [(np.array([[-1,  0,  0], [0, -1,  0], [0,  0, -1]], dtype=int),
               np.array([2.e-06, 2.e-06, 0.e+00]))]
    positions = np.array([[0.156793, 0.313586, 0.75],
                          [0.843207, 0.686414, 0.25],
                          [0.686414, 0.843207, 0.75],
                          [0.313586, 0.156793, 0.25],
                          [0.156793, 0.843207, 0.75],
                          [0.843207, 0.156793, 0.25],
                          [0.333333, 0.666667, 0.25],
                          [0.666667, 0.333333, 0.75]])
    equiv_atoms = np.array([0, 0, 0, 0, 0, 0, 6, 6])
    chem_symbols = ['Mn', 'Mn', 'Mn', 'Mn', 'Mn', 'Mn', 'Fe', 'Fe']

    spins = ['u', 'd', 'u', 'd', 'u', 'd', 'u', 'd']
    afm = is_altermagnet(symops, positions, equiv_atoms, chem_symbols,
                         spins)
    assert afm == False

    spins = ['u', 'u', 'u', 'd', 'd', 'd', 'u', 'd']
    am = is_altermagnet(symops, positions, equiv_atoms, chem_symbols,
                        spins)
    assert am == True
