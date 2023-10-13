from amcheck import __version__

from amcheck import check_altermagnetism_orbit
from amcheck import is_altermagnet
from amcheck import label_matrix
from amcheck import symmetrized_conductivity_tensor

from math import pi, cos, sin
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


def test_NiAs_structure():
    symops = [(np.array([[ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]], dtype=int), np.array( [0., 0., 0.])),
              (np.array([[-1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]], dtype=int), np.array( [0., 0., 0.])),
              (np.array([[ 1, -1,  0], [ 1,  0,  0], [ 0,  0,  1]], dtype=int), np.array( [0., 0., 0.5])),
              (np.array([[-1,  1,  0], [-1,  0,  0], [ 0,  0, -1]], dtype=int), np.array( [0., 0., 0.5])),
              (np.array([[ 0, -1,  0], [ 1, -1,  0], [ 0,  0,  1]], dtype=int), np.array( [0., 0., 0.])),
              (np.array([[ 0,  1,  0], [-1,  1,  0], [ 0,  0, -1]], dtype=int), np.array( [0., 0., 0.])),
              (np.array([[-1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]], dtype=int), np.array( [0., 0., 0.5])),
              (np.array([[ 1,  0,  0], [ 0,  1,  0], [ 0,  0, -1]], dtype=int), np.array( [0., 0., 0.5])),
              (np.array([[-1,  1,  0], [-1,  0,  0], [ 0,  0,  1]], dtype=int), np.array( [0., 0., 0.])),
              (np.array([[ 1, -1,  0], [ 1,  0,  0], [ 0,  0, -1]], dtype=int), np.array( [0., 0., 0.])),
              (np.array([[ 0,  1,  0], [-1,  1,  0], [ 0,  0,  1]], dtype=int), np.array( [0., 0., 0.5])),
              (np.array([[ 0, -1,  0], [ 1, -1,  0], [ 0,  0, -1]], dtype=int), np.array( [0., 0., 0.5])),
              (np.array([[ 0, -1,  0], [-1,  0,  0], [ 0,  0, -1]], dtype=int), np.array( [0., 0., 0.5])),
              (np.array([[ 0,  1,  0], [ 1,  0,  0], [ 0,  0,  1]], dtype=int), np.array( [0., 0., 0.5])),
              (np.array([[-1,  0,  0], [-1,  1,  0], [ 0,  0, -1]], dtype=int), np.array( [0., 0., 0.])),
              (np.array([[ 1,  0,  0], [ 1, -1,  0], [ 0,  0,  1]], dtype=int), np.array( [0., 0., 0.])),
              (np.array([[-1,  1,  0], [ 0,  1,  0], [ 0,  0, -1]], dtype=int), np.array( [0., 0., 0.5])),
              (np.array([[ 1, -1,  0], [ 0, -1,  0], [ 0,  0,  1]], dtype=int), np.array( [0., 0., 0.5])),
              (np.array([[ 0,  1,  0], [ 1,  0,  0], [ 0,  0, -1]], dtype=int), np.array( [0., 0., 0.])),
              (np.array([[ 0, -1,  0], [-1,  0,  0], [ 0,  0,  1]], dtype=int), np.array( [0., 0., 0.])),
              (np.array([[ 1,  0,  0], [ 1, -1,  0], [ 0,  0, -1]], dtype=int), np.array( [0., 0., 0.5])),
              (np.array([[-1,  0,  0], [-1,  1,  0], [ 0,  0,  1]], dtype=int), np.array( [0., 0., 0.5])),
              (np.array([[ 1, -1,  0], [ 0, -1,  0], [ 0,  0, -1]], dtype=int), np.array( [0., 0., 0.])),
              (np.array([[-1,  1,  0], [ 0,  1,  0], [ 0,  0,  1]], dtype=int), np.array( [0., 0., 0.]))]

    positions = np.array([[0.00, 0.00, 0.00],
                          [0.00, 0.00, 0.50],
                          [1/3., 2/3., 0.25],
                          [2/3., 1/3., 0.75]])

    equiv_atoms  = [0, 0, 1, 1]
    chem_symbols = ["Ni", "Ni", "As", "As"]

    # high-pressure FeO: Fe at As positions, O at Ni positions => afm
    spins = ["n", "n", "u", "d"]
    hp_FeO = is_altermagnet(symops, positions, equiv_atoms, chem_symbols,
                        spins)

    assert hp_FeO == False

    # MnTe: Mn at Ni positions, Te at As positions => am
    spins = ["u", "d", "n", "n"]
    MnTe = is_altermagnet(symops, positions, equiv_atoms, chem_symbols,
                        spins)

    assert MnTe == True


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


def test_label_matrix():
    # 1.1.1
    S = label_matrix(np.array([[11,12,13],[21,22,23],[31,32,33]]))
    ethalon = np.array([["σxx","σxy","σxz"], ["σyx","σyy","σyz"], ["σzx","σzy","σzz"]])
    assert (S==ethalon).all()

    # 1.2.2
    S = label_matrix(np.array([[11,12,13],[12,22,23],[13,23,33]]))
    ethalon = np.array([["σxx","σxy","σxz"], ["σxy","σyy","σyz"], ["σxz","σyz","σzz"]])
    assert (S==ethalon).all()

    # 3.1.6
    S = label_matrix(np.array([[11,0,13], [0,22,0], [31,0,33]]))
    ethalon = np.array([["σxx","0","σxz"], ["0","σyy","0"], ["σzx","0","σzz"]])
    assert (S==ethalon).all()

    # 3.3.8
    S = label_matrix(np.array([[ 11, 12,13], [-12, 22,23], [ 13,-23,33]]))
    ethalon = np.array([["σxx", "σxy","σxz"], ["-σxy", "σyy","σyz"], ["σxz","-σyz","σzz"]])
    assert (S==ethalon).all()

    # 4.2.10
    S = label_matrix(np.array([[11,0,13], [0,22,0], [13,0,33]]))
    ethalon = np.array([["σxx","0","σxz"], ["0","σyy","0"], ["σxz","0","σzz"]])
    assert (S==ethalon).all()

    # 6.1.17
    S = label_matrix(np.array([[11,0,0], [0,22,0], [0,0,33]]))
    ethalon = np.array([["σxx","0","0"], ["0","σyy","0"], ["0","0","σzz"]])
    assert (S==ethalon).all()

    # 6.3.19
    S = label_matrix(np.array([[11,12,0], [-12,22,0], [0,0,33]]))
    ethalon = np.array([["σxx","σxy","0"], ["-σxy","σyy","0"], ["0","0","σzz"]])
    assert (S==ethalon).all()

    # 7.3.22
    S = label_matrix(np.array([[11,0,13], [0,22,0], [-13,0,33]]))
    ethalon = np.array([["σxx","0","σxz"], ["0","σyy","0"], ["-σxz","0","σzz"]])
    assert (S==ethalon).all()

    # 9.2.30
    S = label_matrix(np.array([[11,0,0], [0,11,0], [0,0,33]]))
    ethalon = np.array([["σxx","0","0"], ["0","σxx","0"], ["0","0","σzz"]])
    assert (S==ethalon).all()

    # 28.1.107
    S = label_matrix(np.array([[11,0,0], [0,11,0], [0,0,11]]))
    ethalon = np.array([["σxx","0","0"], ["0","σxx","0"], ["0","0","σxx"]])
    assert (S==ethalon).all()

def test_symmetrized_conductivity_tensor():
    # 1.1.1
    rotations = [np.array([[1,0,0],[0,1,0],[0,0,1]])]
    time_reversals = [False]
    S = label_matrix(symmetrized_conductivity_tensor(rotations, time_reversals))
    ethalon = np.array([["σxx","σxy","σxz"], ["σyx","σyy","σyz"], ["σzx","σzy","σzz"]])
    assert (S==ethalon).all()

    # 1.2.2
    rotations = [np.array([[1,0,0],[0,1,0],[0,0,1]]),
                 np.array([[1,0,0],[0,1,0],[0,0,1]])]
    time_reversals = [False, True]
    S = label_matrix(symmetrized_conductivity_tensor(rotations, time_reversals))
    ethalon = np.array([["σxx","σxy","σxz"], ["σxy","σyy","σyz"], ["σxz","σyz","σzz"]], dtype='<U4')
    assert (S==ethalon).all()

    # 3.1.6
    rotations = [np.array([[ 1,0,0],[0,1,0],[0,0, 1]]),
                 np.array([[-1,0,0],[0,1,0],[0,0,-1]])]
    time_reversals = [False, False]
    S = label_matrix(symmetrized_conductivity_tensor(rotations, time_reversals))
    ethalon = np.array([["σxx","0","σxz"], ["0","σyy","0"], ["σzx","0","σzz"]])
    assert (S==ethalon).all()

    # 3.3.8
    rotations = [np.array([[ 1,0,0],[0,1,0],[0,0,1]]),
                 np.array([[-1,0,0],[0,1,0],[0,0,-1]])]
    time_reversals = [False, True]
    S = label_matrix(symmetrized_conductivity_tensor(rotations, time_reversals))
    ethalon = np.array([["σxx", "σxy","σxz"], ["-σxy", "σyy","σyz"], ["σxz","-σyz","σzz"]])
    assert (S==ethalon).all()

    # 4.2.10
    rotations = [np.array([[ 1, 0, 0],[ 0, 1, 0],[ 0, 0, 1]]),
                 np.array([[ 1, 0, 0],[ 0,-1, 0],[ 0, 0, 1]]),
                 np.array([[ 1, 0, 0],[ 0, 1, 0],[ 0, 0, 1]]),
                 np.array([[ 1, 0, 0],[ 0,-1, 0],[ 0, 0, 1]])]
    time_reversals = [False, False, True, True]
    S = label_matrix(symmetrized_conductivity_tensor(rotations, time_reversals))
    ethalon = np.array([["σxx","0","σxz"], ["0","σyy","0"], ["σxz","0","σzz"]])
    assert (S==ethalon).all()

    # 6.1.17
    rotations = [np.array([[ 1, 0, 0],[ 0, 1, 0],[ 0, 0, 1]]),
                 np.array([[ 1, 0, 0],[ 0,-1, 0],[ 0, 0,-1]]),
                 np.array([[-1, 0, 0],[ 0, 1, 0],[ 0, 0,-1]]),
                 np.array([[-1, 0, 0],[ 0,-1, 0],[ 0, 0, 1]])]
    time_reversals = [False, False, False, False]
    S = label_matrix(symmetrized_conductivity_tensor(rotations, time_reversals))
    ethalon = np.array([["σxx","0","0"], ["0","σyy","0"], ["0","0","σzz"]])
    assert (S==ethalon).all()

    # 6.3.19
    rotations = [np.array([[ 1, 0, 0],[ 0, 1, 0],[ 0, 0, 1]]),
                 np.array([[-1, 0, 0],[ 0,-1, 0],[ 0, 0, 1]]),
                 np.array([[ 1, 0, 0],[ 0,-1, 0],[ 0, 0,-1]]),
                 np.array([[-1, 0, 0],[ 0, 1, 0],[ 0, 0,-1]])]
    time_reversals = [False, False, True, True]
    S = label_matrix(symmetrized_conductivity_tensor(rotations, time_reversals))
    ethalon = np.array([["σxx","σxy","0"], ["-σxy","σyy","0"], ["0","0","σzz"]])
    assert (S==ethalon).all()

    # 7.3.22
    rotations = [np.array([[ 1, 0, 0],[ 0, 1, 0],[ 0, 0, 1]]),
                 np.array([[ 1, 0, 0],[ 0,-1, 0],[ 0, 0, 1]]),
                 np.array([[-1, 0, 0],[ 0,-1, 0],[ 0, 0, 1]]),
                 np.array([[-1, 0, 0],[ 0, 1, 0],[ 0, 0, 1]])]
    time_reversals = [False, False, True, True]
    S = label_matrix(symmetrized_conductivity_tensor(rotations, time_reversals))
    ethalon = np.array([["σxx","0","σxz"], ["0","σyy","0"], ["-σxz","0","σzz"]])
    assert (S==ethalon).all()

    # 9.2.30
    rotations = [np.array([[ 1, 0, 0],[ 0, 1, 0],[ 0, 0, 1]]),
                 np.array([[-1, 0, 0],[ 0,-1, 0],[ 0, 0, 1]]),
                 np.array([[ 0,-1, 0],[ 1, 0, 0],[ 0, 0, 1]]),
                 np.array([[ 0, 1, 0],[-1, 0, 0],[ 0, 0, 1]]),
                 np.array([[ 1, 0, 0],[ 0, 1, 0],[ 0, 0, 1]]),
                 np.array([[-1, 0, 0],[ 0,-1, 0],[ 0, 0, 1]]),
                 np.array([[ 0,-1, 0],[ 1, 0, 0],[ 0, 0, 1]]),
                 np.array([[ 0, 1, 0],[-1, 0, 0],[ 0, 0, 1]])]
    time_reversals = [False, False, False, False, True, True, True, True]
    S = label_matrix(symmetrized_conductivity_tensor(rotations, time_reversals))
    ethalon = np.array([["σxx","0","0"], ["0","σxx","0"], ["0","0","σzz"]])
    assert (S==ethalon).all()

    # 28.1.107
    rotations = [np.array([[ 1, 0, 0],[ 0, 1, 0],[ 0, 0, 1]]),
                 np.array([[ 1, 0, 0],[ 0,-1, 0],[ 0, 0,-1]]), # 2x
                 np.array([[-1, 0, 0],[ 0, 1, 0],[ 0, 0,-1]]), # 2y
                 np.array([[-1, 0, 0],[ 0,-1, 0],[ 0, 0, 1]]), # 2z
                 np.array([[ 0, 1, 0],[ 0, 0, 1],[ 1, 0, 0]]),
                 np.array([[ 0, 1, 0],[ 0, 0,-1],[-1, 0, 0]]),
                 np.array([[ 0,-1, 0],[ 0, 0, 1],[-1, 0, 0]]),
                 np.array([[ 0,-1, 0],[ 0, 0,-1],[ 1, 0, 0]]),
                 np.array([[ 0, 0, 1],[ 1, 0, 0],[ 0, 1, 0]]),
                 np.array([[ 0, 0, 1],[-1, 0, 0],[ 0,-1, 0]]),
                 np.array([[ 0, 0,-1],[ 1, 0, 0],[ 0,-1, 0]]),
                 np.array([[ 0, 0,-1],[-1, 0, 0],[ 0, 1, 0]])]
    time_reversals = [False]*12
    S = label_matrix(symmetrized_conductivity_tensor(rotations, time_reversals))
    ethalon = np.array([["σxx","0","0"], ["0","σxx","0"], ["0","0","σxx"]])
    assert (S==ethalon).all()

    # 16.1.60
    a = 2*pi/3
    rotations = [np.array([[ 1, 0, 0],[ 0, 1, 0],[ 0, 0, 1]]),
                 np.array([[cos( a), -sin( a), 0],[sin( a), cos( a), 0],[0,0,1]]),
                 np.array([[cos(-a), -sin(-a), 0],[sin(-a), cos(-a), 0],[0,0,1]])]
    time_reversals = [False]*3
    S = label_matrix(symmetrized_conductivity_tensor(rotations, time_reversals))
    ethalon = np.array([["σxx","σxy","0"], ["-σxy","σxx","0"], ["0","0","σzz"]])
    assert (S==ethalon).all()

    # 16.1.61
    rotations = [np.array([[ 1, 0, 0],[ 0, 1, 0],[ 0, 0, 1]]),
                 np.array([[cos( a), -sin( a), 0],[sin( a), cos( a), 0],[0,0,1]]),
                 np.array([[cos(-a), -sin(-a), 0],[sin(-a), cos(-a), 0],[0,0,1]]),
                 np.array([[ 1, 0, 0],[ 0, 1, 0],[ 0, 0, 1]]),
                 np.array([[cos( a), -sin( a), 0],[sin( a), cos( a), 0],[0,0,1]]),
                 np.array([[cos(-a), -sin(-a), 0],[sin(-a), cos(-a), 0],[0,0,1]])]
    time_reversals = [False]*3+[True]*3
    S = label_matrix(symmetrized_conductivity_tensor(rotations, time_reversals))
    ethalon = np.array([["σxx","0","0"], ["0","σxx","0"], ["0","0","σzz"]])
    assert (S==ethalon).all()
