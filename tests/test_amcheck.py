from amcheck import __version__

from amcheck import check_altermagnetism_orbit
from amcheck import is_altermagnet

import numpy as np


def test_version():
    assert __version__ == '0.1.0'


def test_check_altermagnetism_orbit():
    symops = [(np.array([[ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.0])),
              (np.array([[-1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.0])),
              (np.array([[ 1, -1,  0], [ 1,  0,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[-1,  1,  0], [-1,  0,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[ 0, -1,  0], [ 1, -1,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.0])),
              (np.array([[ 0,  1,  0], [-1,  1,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.0])),
              (np.array([[-1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[ 1,  0,  0], [ 0,  1,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[-1,  1,  0], [-1,  0,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.0])),
              (np.array([[ 1, -1,  0], [ 1,  0,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.0])),
              (np.array([[ 0,  1,  0], [-1,  1,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[ 0, -1,  0], [ 1, -1,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[ 0, -1,  0], [-1,  0,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[ 0,  1,  0], [ 1,  0,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[-1,  0,  0], [-1,  1,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.0])),
              (np.array([[ 1,  0,  0], [ 1, -1,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.0])),
              (np.array([[-1,  1,  0], [ 0,  1,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[ 1, -1,  0], [ 0, -1,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[ 0,  1,  0], [ 1,  0,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.0])),
              (np.array([[ 0, -1,  0], [-1,  0,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.0])),
              (np.array([[ 1,  0,  0], [ 1, -1,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[-1,  0,  0], [-1,  1,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[ 1, -1,  0], [ 0, -1,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.0])),
              (np.array([[-1,  1,  0], [ 0,  1,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.0]))]
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
    symops = [(np.array([[ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.0])),
              (np.array([[-1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.0])),
              (np.array([[ 1, -1,  0], [ 1,  0,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[-1,  1,  0], [-1,  0,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[ 0, -1,  0], [ 1, -1,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.0])),
              (np.array([[ 0,  1,  0], [-1,  1,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.0])),
              (np.array([[-1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[ 1,  0,  0], [ 0,  1,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[-1,  1,  0], [-1,  0,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.0])),
              (np.array([[ 1, -1,  0], [ 1,  0,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.0])),
              (np.array([[ 0,  1,  0], [-1,  1,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[ 0, -1,  0], [ 1, -1,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[ 0, -1,  0], [-1,  0,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[ 0,  1,  0], [ 1,  0,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[-1,  0,  0], [-1,  1,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.0])),
              (np.array([[ 1,  0,  0], [ 1, -1,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.0])),
              (np.array([[-1,  1,  0], [ 0,  1,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[ 1, -1,  0], [ 0, -1,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[ 0,  1,  0], [ 1,  0,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.0])),
              (np.array([[ 0, -1,  0], [-1,  0,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.0])),
              (np.array([[ 1,  0,  0], [ 1, -1,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[-1,  0,  0], [-1,  1,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.5])),
              (np.array([[ 1, -1,  0], [ 0, -1,  0], [ 0,  0, -1]], dtype=int), np.array([0.0, 0.0, 0.0])),
              (np.array([[-1,  1,  0], [ 0,  1,  0], [ 0,  0,  1]], dtype=int), np.array([0.0, 0.0, 0.0]))]

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

def test_non_primitive():
    symops = [(np.array([[ 1,  0,  0,], [ 0,  1,  0], [ 0,  0,  1]], dtype=int), np.array([0.0,  0.0,  0.00])),
              (np.array([[-1,  0,  0,], [ 0, -1,  0], [ 0,  0, -1]], dtype=int), np.array([0.0,  0.0,  0.00])), 
              (np.array([[-1,  0,  0,], [ 0, -1,  0], [ 0,  0,  1]], dtype=int), np.array([0.0,  0.0,  0.00])),
              (np.array([[ 1,  0,  0,], [ 0,  1,  0], [ 0,  0, -1]], dtype=int), np.array([0.0,  0.0,  0.00])),
              (np.array([[ 1,  0,  0,], [ 0, -1,  0], [ 0,  0, -1]], dtype=int), np.array([0.5,  0.5,  0.25])),
              (np.array([[-1,  0,  0,], [ 0,  1,  0], [ 0,  0,  1]], dtype=int), np.array([0.5,  0.5,  0.25])),
              (np.array([[-1,  0,  0,], [ 0,  1,  0], [ 0,  0, -1]], dtype=int), np.array([0.5,  0.5,  0.25])),
              (np.array([[ 1,  0,  0,], [ 0, -1,  0], [ 0,  0,  1]], dtype=int), np.array([0.5,  0.5,  0.25])),
              (np.array([[ 1,  0,  0,], [ 0,  1,  0], [ 0,  0,  1]], dtype=int), np.array([0.0,  0.0,  0.50])),
              (np.array([[-1,  0,  0,], [ 0, -1,  0], [ 0,  0, -1]], dtype=int), np.array([0.0,  0.0,  0.50])),
              (np.array([[-1,  0,  0,], [ 0, -1,  0], [ 0,  0,  1]], dtype=int), np.array([0.0,  0.0,  0.50])),
              (np.array([[ 1,  0,  0,], [ 0,  1,  0], [ 0,  0, -1]], dtype=int), np.array([0.0,  0.0,  0.50])),
              (np.array([[ 1,  0,  0,], [ 0, -1,  0], [ 0,  0, -1]], dtype=int), np.array([0.5,  0.5,  0.75])),
              (np.array([[-1,  0,  0,], [ 0,  1,  0], [ 0,  0,  1]], dtype=int), np.array([0.5,  0.5,  0.75])),
              (np.array([[-1,  0,  0,], [ 0,  1,  0], [ 0,  0, -1]], dtype=int), np.array([0.5,  0.5,  0.75])),
              (np.array([[ 1,  0,  0,], [ 0, -1,  0], [ 0,  0,  1]], dtype=int), np.array([0.5,  0.5,  0.75]))]

    positions = np.array([[0.00000000, 0.00000000, 0.00],
                          [0.00000000, 0.00000000, 0.50],
                          [0.50000000, 0.50000000, 0.25],
                          [0.50000000, 0.50000000, 0.75],
                          [0.18850000, 0.35609999, 0.00],
                          [0.18850000, 0.35609999, 0.50],
                          [0.81150001, 0.64389998, 0.00],
                          [0.81150001, 0.64389998, 0.50],
                          [0.31150001, 0.85610002, 0.25],
                          [0.31150001, 0.85610002, 0.75],
                          [0.68849999, 0.14390001, 0.25],
                          [0.68849999, 0.14390001, 0.75]])

    equiv_atoms  = [0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4]
    chem_symbols = ["Fe", "Fe", "Fe", "Fe", "Sb", "Sb", "Sb", "Sb", "Sb", "Sb", "Sb", "Sb"]

    spins = ['u', 'u', 'd', 'd', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n']
    am = is_altermagnet(symops, positions, equiv_atoms, chem_symbols,
                        spins)
    assert am == True

    spins = ['u', 'd', 'u', 'd', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n']
    afm = is_altermagnet(symops, positions, equiv_atoms, chem_symbols,
                        spins)

    assert afm == False

    spins = ['u', 'd', 'd', 'u', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n']
    afm = is_altermagnet(symops, positions, equiv_atoms, chem_symbols,
                        spins)

    assert afm == False
