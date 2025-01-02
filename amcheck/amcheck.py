# ********************************************************************************
# Copyright 2023-2024 Andriy Smolyanyuk, Libor Smejkal, Igor Mazin
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ********************************************************************************

# ********************************************************************************
# Checks if a given structure is an altermagnet by checking that no pair of
# atoms with opposite spins consists of atoms related by inversion or translation.
# ********************************************************************************

import argparse
import os
import sys

import numpy as np
from diophantine import lllhermite

import ase.io
import ase.io.vasp
from ase import Atoms
import spglib

import amcheck

DEFAULT_TOLERANCE = 1e-3  # tolerance for numeric operations


def eprint(*args, **kwargs):
    """ Print to stderr."""
    print(*args, file=sys.stderr, **kwargs)


def bring_in_cell(r, tol=DEFAULT_TOLERANCE):
    """
    Brings an atomic position, defined in the fractional coordinate system,
    to the origin unit cell.

    Parameters
    ----------
    r : array
        Array containing scaled (fractional) positions of an atom.
    tol : float
        A tolerance parameter for numerical computations.
    """
    r = np.mod(r, 1)
    # a possible issue with numbers close to unity, if the distances are later
    # computed: mod(0.99999,1)=0.99999, but for the practical applications it
    # is reasonable to move it to the right of the origin instead
    r[np.isclose(np.ones(r.shape), r, atol=tol)] = 1.0 - r[np.isclose(np.ones(r.shape), r, atol=tol)]
    return r


def check_altermagnetism_orbit(symops, positions, spins, tol=DEFAULT_TOLERANCE,
                               verbose=False, silent=True):
    """
    Check if a given Wyckoff orbit with a given spin pattern is altermagnetic.

    Parameters
    ----------
    symops : list of tuples
        List of tuples containing symmetry operations [(R0, t0), ..., Rk, tk)],
        where Ri is 3x3 matrix and ti is a 3-element vector.
    positions : array of arrays
        Array of arrays containing scaled (fractional) positions of atoms in
        a given Wyckoff orbit, i.e. [[0,0,0], [0.5,0.5,0.5]].
    spins : list of strings
        List of string objects denoting the spin designation of each atom.
        Possible values are: "u", "U" for spin-up, "d", "D" for spin-down and
        "n", "N" to mark a non-magnetic atom.
    tol : float
        A tolerance parameter for numerical computations.
    verbose : Bool
        Print some additional information during the execution.
    silent : Bool
        Suppress any print instructions if True.

    Returns
    -------
    is_altermagnet : Bool
        Returns True if the orbit with a given spin configuration is
        altermagnetic or False otherwise.

    Raises
    ------
    ValueError
        If the number of atomic positions is not equal to the number of spins.

    ValueError
        If number of up and down spins is not the same (spins should be
        compensated, otherwise it's not an altermagnet).
    """

    # if orbit has multiplicity 1, it cannot be altermagnetic
    if len(positions) == 1: return False

    if len(positions) != len(spins):
        raise ValueError("The number of atomic positions should be the same as \
the number of spin designations: got {} and {} instead!".format(len(positions),
                                                            len(spins)))

    # "normalize" spin designations to lowercase for easier bookkeeping
    spins = [s.lower() for s in spins]

    # for a given spin pattern, determine antisymmetry operations among
    # the space group operations
    magn_symops_filter = np.full(len(symops), True)
    for i in range(len(positions)):
        # we want to check only pairs of up-down spins
        if not (spins[i] == "u" or spins[i] == "d"):
            continue

        for (si,symop) in enumerate(symops):
            symop_is_present = False
            R, t = symop

            for j in range(len(positions)):
                # we want to check only pairs of up-down spins
                if not ((spins[i] == "u" and spins[j] == "d") or
                        (spins[i] == "d" and spins[j] == "u")):
                    continue

                # check if the up-down pair is related by some symmetry
                dp = np.dot(R, positions[i]) + t - positions[j]
                dp = bring_in_cell(dp, tol)

                if np.linalg.norm(dp) < tol:
                    symop_is_present |= True
                    break

            magn_symops_filter[si] &= symop_is_present


    magn_symops = [s for (s,is_in_list) in zip(symops, magn_symops_filter)
                    if is_in_list]

    if not magn_symops:
        if not silent and verbose:
            print("Up and down sublattices are not symmetry-related: the material is Luttinger ferrimagnet!")
        return False

    is_altermagnet = False

    N_magnetic_atoms = 2*spins.count("u")

    is_in_sym_related_pair = np.zeros(len(positions))
    is_in_IT_related_pair  = np.zeros(len(positions))
    # Check if inversion symmetry is located in the midpoint of the pair of
    # magnetic atoms with opposite spin directions or if they are related by
    # translation
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            # we want to check only pairs of up-down spins
            if not ((spins[i] == "u" and spins[j] == "d") or
                    (spins[i] == "d" and spins[j] == "u")):
                continue

            midpoint = (positions[i] + positions[j])/2

            for symop in magn_symops:
                R, t = symop

                # check if the up-down pair is related by some symmetry
                dp = np.dot(R, positions[i]) + t - positions[j]
                dp = bring_in_cell(dp, tol)

                if np.linalg.norm(dp) < tol:
                    is_in_sym_related_pair[i] = 1
                    is_in_sym_related_pair[j] = 1

                # if symop is inversion
                if (abs(np.trace(R)+3) < tol):
                    midpoint_prime = np.dot(symop[0], midpoint) + symop[1]
                    midpoint_prime -= midpoint

                    # bring it back to the unit cell
                    midpoint_prime = bring_in_cell(midpoint_prime, tol)

                    if np.linalg.norm(midpoint_prime) < tol:
                        # the inversion is positioned at the midpoint:
                        # mark atoms i and j as the ones belonging to the pair
                        # of opposite spins with an inversion point at its
                        # midpoint
                        is_in_IT_related_pair[i] = 1
                        is_in_IT_related_pair[j] = 1
                        if not silent and verbose:
                            print("Atoms {} and {} are related by inversion (midpoint {}).".format(i+1,
                                                                                                   j+1, midpoint))

                # if symop is translation
                if (abs(np.trace(R)-3) < tol and np.linalg.norm(t) > tol):
                    dp = positions[i] + t - positions[j]
                    dp = bring_in_cell(dp, tol)

                    if np.linalg.norm(dp) < tol:
                        # atoms i and j are related by translation
                        is_in_IT_related_pair[i] = 1
                        is_in_IT_related_pair[j] = 1
                        if not silent and verbose:
                            print("Atoms {} and {} are related by translation {}.".format(
                                i+1, j+1, t))

    if not silent and verbose:
        print("Atoms related by inversion/translation (1-yes, 0-no):", is_in_IT_related_pair)
    if not silent and verbose:
        print("Atoms related by some symmetry (1-yes, 0-no):", is_in_sym_related_pair)

    is_Luttinger_ferrimagnet = abs(np.sum(is_in_sym_related_pair)-N_magnetic_atoms) > tol
    if not silent and verbose:
        if is_Luttinger_ferrimagnet:
            print("Up and down sublattices are not related by symmetry: the material is Luttinger ferrimagnet!")

    # This orbit of magnetic atoms will produce an AF if all its atoms belong
    # to some pair of opposite spins that either has an inversion at its
    # midpoint or are related by a translation.
    # Otherwise it's an altermagnet.
    is_altermagnet = abs(np.sum(is_in_IT_related_pair)-N_magnetic_atoms) > tol
    is_altermagnet &= not is_Luttinger_ferrimagnet

    return is_altermagnet


def is_altermagnet(symops, atom_positions, equiv_atoms, chemical_symbols, spins,
                   tol=DEFAULT_TOLERANCE, verbose=False, silent=True):
    """ 
    Check if a given structure is altermagnetic.

    Parameters
    ----------
    symops : list of tuples
        List of tuples containing symmetry operations [(R0, t0), ..., Rk, tk)],
        where Ri is 3x3 matrix and ti is a 3-element vector.
    atom_positions : array of arrays
        Array of arrays containing scaled (fractional) positions of atoms in
        a given structure, i.e. [[0,0,0], [0.5,0.5,0.5]].
    equiv_atoms : array of int
        Array representing how atoms in the structure are split into groups
        of symmetry equivalent atoms (Wyckoff orbit), i.e. [0 0 0 0 0 0 6 6]
        means that there are two Wyckoff orbits consisting of atoms of id
        0 and 6.
    chemical_symbols : list of strings
        List of chemical symbols designated to each atom, i.e.
        ['Mn', 'Mn', 'Mn', 'Mn', 'Mn', 'Mn', 'Fe', 'Fe'].
    spins : list of strings
        List of string objects denoting the spin designation of each atom.
        Possible values are: "u", "U" for spin-up, "d", "D" for spin-down and
        "n", "N" to mark a non-magnetic atom.
    tol : float
        A tolerance parameter for numerical computations.
    verbose : Bool
        Print some additional information during the execution.
    silent : Bool
        Suppress any print instructions if True.

    Returns
    -------
    altermagnet : Bool
        Returns True if the structure with a given spin configuration is
        altermagnetic or False otherwise.

    Raises
    ------
    ValueError
        If number of up and down spins is not the same (spins should be
        compensated, otherwise it's not an altermagnet).
    RuntimeError
        If, for some reason, no Wyckoff orbit was checked for altermagnetism.
        This situation is possible when all atoms were marked as non-magnetic.
        Although the non-magnetic structure is not an altermagnet, we still want
        to notify the user to check for possible mistakes in the input.
    """

    altermagnet = False
    check_was_performed = False
    all_orbits_multiplicity_one = True
    for u in np.unique(equiv_atoms):
        atom_ids = np.where(equiv_atoms == u)[0]
        orbit_positions = atom_positions[atom_ids]

        if not silent and verbose:
            print()
            print("Orbit of {} atoms:".format(chemical_symbols[atom_ids[0]]))

        all_orbits_multiplicity_one = all_orbits_multiplicity_one \
                                      and (len(orbit_positions) == 1)
        if len(orbit_positions) == 1:
            if not silent:
                print("Only one atom in the orbit: skipping.")
            continue

        orbit_spins = [spins[i] for i in atom_ids]

        # skip if the Wyckoff orbit consists of non-magnetic atoms
        if all(s == "n" for s in orbit_spins):
            if not silent:
                print(
                "Group of non-magnetic atoms ({}): skipping.".format(chemical_symbols[u]))
            continue

        # a sanity check: the number of up and down spins should be the same.
        # Otherwise it's not an antiferromagnet/altermagnet
        N_u = orbit_spins.count("u")
        N_d = orbit_spins.count("d")
        if N_u != N_d:
            raise ValueError("The number of up spins should be the same as the number of down spins: " +
                             "got {} up and {} down spins!".format(N_u, N_d))

        # a sanity check: if there is something wrong with input data in the
        # way that we never reach the following line, we need to report it
        check_was_performed = True
        is_orbit_altermagnetic = check_altermagnetism_orbit(symops,
                                                            orbit_positions, orbit_spins, tol, verbose, silent)
        altermagnet |= is_orbit_altermagnetic
        if not silent and verbose:
            print("Altermagnetic orbit ({})?".format(chemical_symbols[u]),
                  is_orbit_altermagnetic)

    if not check_was_performed:
        if all_orbits_multiplicity_one:
            altermagnet = False
            if not silent:
                print("Note: in this structure, all orbits have multiplicity one.\n\
This material can only be a Luttinger ferrimagnet.")
        else:
            raise RuntimeError("Something is wrong with the description of magnetic atoms!\n\
Have you provided a non-magnetic/ferromagnetic material?")

    return altermagnet


def input_spins(num_atoms):
    """
    Read a list of spin designations for a given Wyckoff orbit from stdin.

    Parameters
    ----------
    num_atoms : int
        Number of atoms in the given Wyckoff orbit.

    Returns
    -------
    spins : list of strings
        List of string objects denoting the spin designation of each atom.
        Possible values are: "u", "U" for spin-up, "d", "D" for spin-down and
        "n", "N" to mark a non-magnetic atom.
        "nn" or "NN" can be used to mark entire Wyckoff orbit as non-magnetic.

    Raises
    ------
    ValueError
        If the number of spins from input is not the same as num_atoms.

    ValueError
        If the number of up and down spins is not the same: for an altermagnet,
        spins should be compensated.
    """

    print("Type spin (u, U, d, D, n, N, nn or NN) for each of them (space separated):")
    spins = input().split()
    # "normalize" spin designations to lowercase for easier bookkeeping
    spins = [s.lower() for s in spins]

    # empty line or "nn" in input marks all atoms in the orbit as nonmagnetic
    if len(spins) < 1 or spins[0] == 'nn':
        return ["n"]*num_atoms

    if len(spins) != num_atoms:
        raise ValueError(
            "Wrong number of spins was given: got {} instead of {}!".format(len(spins), num_atoms))

    if not all(s in ["u", "d", "n"] for s in spins):
        raise ValueError("Use u, U, d, D, n or N for spin designation!")

    N_u = spins.count("u")
    N_d = spins.count("d")
    if N_u != N_d:
        raise ValueError("The number of up spins should be the same as the number of down spins: " +
                         "got {} up and {} down spins!".format(N_u, N_d))

    # all atoms in the orbit are nonmagnetic
    if N_u == 0:
        return ["n"]*num_atoms

    return spins

def label_matrix(m, tol=1e-3):
    """
    Obtain a symbolic representation of a given numeric matrix.

    The goal is to label the matrix entries, i.e., the conductivity tensor,
    in the way that the number of labels is the same as the number
    of independent components.
    For example, the matrix [[1,0,0],[0,1,0],[0,0,2]] is represented by
    [["xx", 0, 0],[0, "xx", 0], [0, 0, "zz"]].

    Parameters
    ----------
    m : 3x3 matrix
        A 3x3 matrix to find a symbolic representation for.
    tol : float
        A tolerance parameter for numerical computations.
    Returns
    -------
    s : 3x3 matrix of strings
        A symbolic matrix with labels obtained from a given matrix.
    """

    dictionary = "xx", "yy", "zz", "yz", "xz", "xy", "zy", "zx", "yx"
    ids = [(0,0), (1,1), (2,2), (1,2), (0,2), (0,1), (2,1), (2,0), (1,0)]

    s = np.array([["0", "0", "0"],["0", "0", "0"],["0", "0", "0"]], dtype='<U4')
    s[0, 0] = "xx"

    for i in range(9):
        for j in range(i+1):
            if abs(m[ids[i]]) > tol:
                if abs(m[ids[i]] - m[ids[j]]) < tol:
                    s[ids[i]] = s[ids[j]]
                    break
                elif abs(abs(m[ids[i]]) - abs(m[ids[j]])) < tol:
                    s[ids[i]] = "-"+s[ids[j]]
                    break
                else:
                    s[ids[i]] = dictionary[i]
            else:
                s[ids[i]] = "0"
    return s


def symmetrized_conductivity_tensor(rotations, time_reversals):
    """
    Return a symmetrized conductivity tensor w.r.t given symmetries.

    Parameters
    ----------
    rotations : list of 3x3 matrices
        List of 3x3 matrices that contains symmetry operations.
    time_reversals : list of booleans
        Each entry defines if the rotation in `rotations` list is a symmetry
        or antisymmetry operation.

    Returns
    -------
    S : 3x3 matrix
        A symmetrized conductivity tensor.
        Entries are numbers, use `label_matrix` function to get symbolic
        representation.
    """
    # a seed matrix to be symmetrized: Qa*diagm(s)*Qb, where s = [s1, s2, s3]
    # some distinct random singular values, A = Qa*Ra and B = Qb*Rb are
    # QR decompositions of random matrices
    seed   = np.array([[ 0.18848,   -0.52625,    0.047702],
                       [  0.403317, -0.112371, -0.0564825],
                       [ -0.352134,  0.350489,  0.0854533]])
    seed_T = np.transpose(seed)

    S = np.zeros((3,3))
    for (R, T) in zip(rotations, time_reversals):
        if T:
            S += np.linalg.inv(R) @ seed_T @ R
        else:
            S += np.linalg.inv(R) @ seed   @ R

    return S


def main(args):
    """ Run altermagnet/antiferromagnet structure analysis interactively. """
    if args.verbose:
        print('spglib version:', spglib.__version__)

    for filename in args.file:
        try:
            print("="*80)
            print("Processing:", filename)
            print("-"*80)
            atoms = ase.io.read(filename)
            cell = atoms.get_cell(complete=True)[:]

            # get the space group number
            spglib_cell = (
                atoms.cell, atoms.get_scaled_positions(), atoms.numbers)
            sg = spglib.get_spacegroup(spglib_cell, symprec=args.symprec)
            sg_no = int(sg[sg.find('(') + 1:sg.find(')')])
            print("Spacegroup: {}".format(sg))

            # if the space group number is present in the input file (i.e. in cif),
            # check if it is consistent with the one we just got from the spglib
            if "spacegroup" in atoms.info:
                if sg_no != atoms.info["spacegroup"].no:
                    print("WARNING: the space group from the input is different from the spglib \
analysis: {} (spglib) vs {} (input)!".format(sg_no, atoms.info['spacegroup'].no))

            primitive = spglib.standardize_cell((atoms.cell, atoms.get_scaled_positions(),
                                                 atoms.numbers),
                                                to_primitive=True, no_idealize=True,
                                                symprec=args.symprec)
            prim_cell, prim_pos, prim_num = primitive

            # The given unit cell might be non-primitive, and if this is the
            # case, we will ask the user: shall we keep using it, or shall we
            # use a primitive one instead?
            if abs(np.linalg.det(atoms.cell) - np.linalg.det(prim_cell)) > args.tol:
                symmetry = spglib.get_symmetry(primitive, symprec=args.symprec)
                rotations = symmetry['rotations']
                translations = symmetry['translations']
                N_ops_prim = len(translations)

                print("There is a primitive unit cell!")
                print("Do you want to use it instead? (Y/n)")
                answer = input()
                if answer.lower() != "n":
                    print("The primitive unit cell will be used.")
                    atoms = Atoms(prim_num, cell=prim_cell,
                                  scaled_positions=prim_pos)
                    equiv_atoms = symmetry['equivalent_atoms']
                else:
                    print("The original non-primitive unit cell will be used.")

                    symmetry_dataset = spglib.get_symmetry_dataset(
                        spglib_cell, symprec=args.symprec)

                    if args.verbose:
                        print(
                            "Atoms mapping from primitive to non-primitive unit cell:")
                        for i in range(len(prim_num)):
                            atoms_mapped = np.where(
                                symmetry_dataset['mapping_to_primitive'] == i)[0]
                            print("{}->{}".format(i+1, atoms_mapped+1))

                    equiv_atoms = symmetry_dataset['crystallographic_orbits']

                    # S = T*P, where S is a unit cell of a supercell and P of
                    # a primitive cell
                    T = np.rint(np.dot(cell, np.linalg.inv(prim_cell)))

                    det_T = np.linalg.det(T)
                    det_ratios = np.linalg.det(cell)/np.linalg.det(prim_cell)
                    assert np.isclose(det_T, det_ratios),\
                    "Sanity check: the determinant of transformation is not equal to \
original cell/primitive cell ratio: got {} instead of {}!".format(det_T, det_ratios)

                    if args.verbose:
                        print(
                            "Transformation from primitive to non-primitive cell, T:")
                        print(T)

                    # All possible supercells can be grouped by det(T) and
                    # within each group, the amount of possible distinct
                    # supercells is finite.
                    # All of them can be enumerated using the Hermite Normal
                    # Form, H.
                    H, _, _ = lllhermite(T)
                    H = np.array(H, dtype=int)
                    if args.verbose:
                        print("HNF of T:")
                        print(H)

                    # By knowing the HNF we can determine the direction and
                    # multiplicity of fractional translations and transform
                    # them into the basis of the original supercell
                    tau = [np.mod([i, j, k] @ prim_cell @ np.linalg.inv(cell), 1)
                           for i in range(H[0, 0]) for j in range(H[1, 1]) for k in range(H[2, 2])]

                    # The final collection of symmetry operations is a copy of
                    # the original operations augmented by the new translations:
                    # (R,t) = (R0,t0)*{(E,0) + (E,t1) + ... + (E,tN)}
                    # However, original fractional translations should also be
                    # transformed to the basis of a new cell.
                    N = int(np.rint(np.linalg.det(H)))
                    rotations = np.tile(rotations, (N, 1, 1))
                    translations = np.tile(translations, (N, 1))
                    for (i, t) in enumerate(tau):
                        for j in range(i*N_ops_prim, (i+1)*N_ops_prim):
                            translations[j] = np.mod(translations[j] @ prim_cell @ np.linalg.inv(cell) + t, 1)

            # The original unit cell is primitive
            else:
                symmetry = spglib.get_symmetry(
                    spglib_cell, symprec=args.symprec)
                rotations = symmetry['rotations']
                translations = symmetry['translations']
                equiv_atoms = symmetry['equivalent_atoms']

            if args.verbose:
                print("Number of symmetry operations: ",
                      len(rotations), len(translations))

            symops = [(r,t) for (r, t) in zip(rotations, translations)]

            if args.verbose:
                print("Symmetry operations:")
                for (i, (r, t)) in enumerate(symops):
                    sym_type = ""
                    if (abs(np.trace(r)+3) < args.tol):
                        sym_type = "inversion"
                    if (abs(np.trace(r)-3) < args.tol and np.linalg.norm(t) > args.tol):
                        sym_type = "translation"

                    print("{}: {}".format(i+1, sym_type))
                    print(r)
                    print(t)

            # for convenience, we will create an auxiliary file that the user can
            # use to assign spins while examining the file in some visualizer
            aux_filename = filename+"_amcheck.vasp"
            print()
            print("Writing the used structure to auxiliary file: check {}.".format(
                aux_filename))
            ase.io.vasp.write_vasp(aux_filename, atoms, direct=True)

            # get spins from user's input
            chemical_symbols = atoms.get_chemical_symbols()
            spins = ['n' for i in range(len(chemical_symbols))]
            for u in np.unique(equiv_atoms):
                atom_ids = np.where(equiv_atoms == u)[0]
                positions = atoms.get_scaled_positions()[atom_ids]
                print()
                print("Orbit of {} atoms at positions:".format(
                    chemical_symbols[atom_ids[0]]))
                for (i, j, p) in zip(atom_ids, range(1, len(atom_ids)+1), positions):
                    print(i+1, "({})".format(j), p)

                if len(positions) == 1:
                    print("Only one atom in the orbit: skipping.")
                    continue

                orbit_spins = input_spins(len(positions))
                for i in range(len(orbit_spins)):
                    spins[atom_ids[i]] = orbit_spins[i]

            is_am = is_altermagnet(symops, atoms.get_scaled_positions(),
                                   equiv_atoms, chemical_symbols, spins,
                                   args.tol, args.verbose, False)

            print()
            print("Altermagnet?", is_am)

        except ase.io.formats.UnknownFileTypeError as error:
            eprint("[ERROR] " + "ASE: unknown file type ({})".format(str(error)))

        except Exception as error:
            eprint("[ERROR] " + str(error))


def main_ahc_type(args):
    """
    Determine the type (up to an equivalency) of Anomalous Hall Coefficient 
    for a given material.
    Raises
    ------
    ValueError
        If the provided magnetic moment is not represented by three numbers.

    RuntimeError
        If spglib.get_magnetic_symmetry_dataset return None, i.e. not able to
        determine the magnetic symmetry of a given structure.
    """

    for filename in args.file:
        try:
            print("="*80)
            print("Processing:", filename)
            print("-"*80)

            atoms = ase.io.read(filename)
            positions = atoms.get_scaled_positions()

            print("List of atoms:")
            for (i, (pos, label)) in enumerate(zip(positions,
                                                   atoms.get_chemical_symbols())):
                print(label, pos)

            magnetic_moments = [[0, 0, 0] for i in range(len(positions))]
            print()
            print("Type magnetic moments for each atom in Cartesian coordinates\n\
('mx my mz' or empty line for non-magnetic atom):")
            for i in range(len(positions)):
                mm = input()
                mm = list(map(float, mm.split()))

                # if the user provides an empty line as input, the atom is
                # non-magnetic: no need to do anything
                if not mm:
                    continue

                if len(mm) != 3:
                    raise ValueError("Three numbers for magnetic moment \
definition were expected!")
                magnetic_moments[i] = mm

            print("Assigned magnetic moments:")
            print(magnetic_moments)
            print()

            spglib_cell = (atoms.cell, atoms.get_scaled_positions(),
                           atoms.numbers, np.array(magnetic_moments))
            MSG = spglib.get_magnetic_symmetry_dataset(spglib_cell,
                                                       symprec=args.symprec, mag_symprec=args.mag_symprec)
            if MSG is None:
                raise RuntimeError("spglib is not able to determine the magnetic symmetry!\n\
Check if the crystal structure and magnetic pattern are reasonable.\n\
Additionally, try different tolerance values.")

            print("Magnetic Space Group:",
                  spglib.get_magnetic_spacegroup_type(MSG['uni_number']))
            print()
            symmetries = spglib.get_magnetic_symmetry(spglib_cell, symprec=args.symprec,
                    angle_tolerance=-1.0, mag_symprec=args.mag_symprec,
                    is_axial=True, with_time_reversal=True)
            if symmetries is None:
                raise RuntimeError("spglib is not able to determine the magnetic symmetry!\n\
Check if the crystal structure and magnetic pattern are reasonable.\n\
Additionally, try different tolerance values.")

            # The given unit cell might have some uncertainties; thus, for the
            # transformation matrix T, we'd like to take the "idealized" unit
            # cell, which conforms to the obtained space group.

            # A caveat: refine_cell function might lead to using the
            # conventional instead of the original primitive cell: that is not
            # what we want.
            # Thus, we will explicitly use the standardize_cell function to
            # get the "idealized" unit cell, but we need to be careful to
            # continue using the same type of the unit cell (primitive or
            # conventional) as we had originally.
            primitive = spglib.standardize_cell(spglib_cell, to_primitive=True,
                                                no_idealize=True,
                                                symprec=args.symprec)
            prim_cell, prim_pos, prim_num = primitive
            is_primitive = abs(np.linalg.det(atoms.cell) - np.linalg.det(prim_cell)) < args.tol

            refined_cell, _, _ = spglib.standardize_cell(spglib_cell,
                                                         to_primitive=is_primitive,
                                                         no_idealize=False,
                                                         symprec=args.symprec,
                                                         angle_tolerance=-1.0)

            refined_cell = np.rint(atoms.cell @ np.linalg.inv(refined_cell)) @ refined_cell

            T = np.transpose(refined_cell)
            rotations = [T @ R @ np.linalg.inv(T) for R in symmetries['rotations']]
            time_reversals = symmetries['time_reversals']
            if args.verbose:
                print("Symmetry operations:")
                np.set_printoptions(precision=3)
                for (i, (r, t)) in enumerate(zip(rotations, time_reversals)):
                    sym_type = "Time reversal: {}".format(t)
                    print("{}: {}".format(i+1, sym_type))
                    with np.printoptions(precision=3, suppress=True):
                        print(r)
                print()

            S = symmetrized_conductivity_tensor(rotations, time_reversals)
            print("Conductivity tensor:")
            if args.verbose:
                with np.printoptions(precision=7, suppress=True):
                    print(S)
            print(label_matrix(S))
            print()

            Sa = label_matrix((S - np.transpose(S))/2)
            print("The antisymmetric part of the conductivity tensor (Anomalous Hall Effect):")
            if args.verbose:
                with np.printoptions(precision=7, suppress=True):
                    print((S-np.transpose(S))/2)
            print(Sa)
            print()

            print("Hall vector:")
            print([Sa[2,1], Sa[0,2], Sa[1,0]])

        except ase.io.formats.UnknownFileTypeError as error:
            eprint("[ERROR] " + "ASE: unknown file type ({})".format(str(error)))

        except Exception as error:
            eprint("[ERROR] " + str(error))

def cli(args=None):
    if not args:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog='amcheck',
        description='A tool to check if a given material is an altermagnet.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=amcheck.__version__))
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="verbosely list the information during the execution")
    parser.add_argument('file', nargs='+',
                        help="name of the structure file to analyze")

    parser.add_argument('-s', '--symprec', default=DEFAULT_TOLERANCE, type=float,
                        help="tolerance spglib uses during the symmetry analysis")
    parser.add_argument('-ms', '--mag_symprec', default=-1.0, type=float,
                        help="tolerance for magnetic moments spglib uses during the magnetic symmetry analysis")
    parser.add_argument('-t', '--tol', '--tolerance', default=DEFAULT_TOLERANCE, type=float,
                        help="tolerance for internal numerical checks")

    parser.add_argument('--ahc', action='store_true',
                        help="determine the possible form of Anomalous Hall Coefficient")

    args = parser.parse_args()

    if args.ahc:
        main_ahc_type(args)
    else:
        main(args)

if __name__ == "__main__":
    cli()
