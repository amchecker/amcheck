#!/usr/bin/env python3
# ********************************************************************************
# Copyright 2023 Andriy Smolyanyuk, Olivia Taivo, Libor Smejkal, Igor Mazin
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


DEFAULT_TOLERANCE = 1e-3  # tolerance for numeric operations


def eprint(*args, **kwargs):
    """ Print to stderr."""
    print(*args, file=sys.stderr, **kwargs)


def check_altermagnetism_orbit(symops, positions, spins, tol=DEFAULT_TOLERANCE,
                               verbose=False):
    """
    Check if a given Wyckoff orbit with a given spins is altermagnetic.

    Parameters
    ----------
    symops : list of tuples
        List of tuples that contains symmetry operations [(R0, t0), ..., Rk, tk)],
        where Ri is 3x3 matrix and ti is 3-elements vector.
    positions : array of arrays
        Array of arrays containing scaled (fractional) positions of atoms in
        a given Wyckoff orbit, i.e. [[0,0,0], [0.5,0.5,0.5]].
    spins : list of strings
        List of string objects denoting spin designation of each atom.
        Possible values are: "u", "U" for spin-up, "d", "D" for spin-down and
        "n", "N" to mark a non-magnetic atom.
    tol : float
        A tolerance parameter for numerical computations.
    verbose : Bool
        Print some additional information during the execution.

    Returns
    -------
    is_altermagnet : Bool
        Returns True if the orbit with a given spin configuration is
        altermagnetic or False otherwise.

    Raises
    ------
    ValueError
        If len(positions) != len(spins)
    """

    if len(positions) != len(spins):
        raise ValueError("Number of atomic positions should be the same as \
number of spin designation: got {} and {} instead!".format(len(positions),
                                                           len(spins)))

    # if symops list is empty, the spacegroup is definitely centrosymmetric,
    # thus the orbit is altermagnetic (this check is relevant if only
    # "altermagnetic" symmetry operations are provided)
    if not symops:
        return True

    is_altermagnet = False

    is_in_sym_related_pair = np.zeros(len(positions))
    # Check if inversion symmetry is present in midpoint of the pair of
    # magnetic atoms with opposite spin directions or if they are related by
    # translation
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            # we want to check only pairs of up-down spins
            if not ((spins[i] == "u" and spins[j] == "d") or
                    (spins[i] == "d" and spins[j] == "u")):
                continue

            midpoint = (positions[i] + positions[j])/2

            for symop in symops:
                R, t = symop
                # if symop is inversion
                if (abs(np.trace(R)+3) < tol):
                    midpoint_prime = np.dot(symop[0], midpoint) + symop[1]
                    midpoint_prime -= midpoint

                    # bring it back to the unit cell
                    midpoint_prime = np.mod(midpoint_prime, 1)

                    # a possible issue with numbers close to one
                    for k in range(3):
                        if abs(1.0-midpoint_prime[k]) < tol:
                            midpoint_prime[k] = 0.0

                    if np.linalg.norm(midpoint_prime) < tol:
                        # the inversion is positioned at the midpoint:
                        # mark atoms i and j as the ones belonging to the pair
                        # of opposite spins that has an inversion point at its
                        # midpoint
                        is_in_sym_related_pair[i] = 1
                        is_in_sym_related_pair[j] = 1
                        if verbose:
                            print("Atoms {} and {} are related by inversion (midpoint {}).".format(i+1,
                                                                                                   j+1, midpoint))

                # if symop is translation
                if (abs(np.trace(R)-3) < tol and np.linalg.norm(t) > tol):
                    dp = np.mod(positions[i] + t - positions[j], 1)
                    # a possible issue with numbers close to one
                    for k in range(3):
                        if abs(1.0-dp[k]) < tol:
                            dp[k] = 0.0

                    if np.linalg.norm(dp) < tol:
                        # atoms i and j are related by translation
                        is_in_sym_related_pair[i] = 1
                        is_in_sym_related_pair[j] = 1
                        if verbose:
                            print("Atoms {} and {} are related by translation {}.".format(
                                i+1, j+1, t))

    if verbose:
        print("Symmetry related atoms (1-yes, 0-no):", is_in_sym_related_pair)
    # This orbit of magnetic atoms will produce an AF if all its atoms belong
    # to some pair of opposite spins that either has an inversion at its
    # midpoint or are related by a translation.
    # Otherwise it's an altermagnet.
    is_altermagnet = abs(np.sum(is_in_sym_related_pair)-len(positions)) > tol

    return is_altermagnet


def is_altermagnet(symops, atom_positions, equiv_atoms, chemical_symbols, spins,
                   tol=DEFAULT_TOLERANCE, verbose=False):
    """ 
    Check if a given structure is altermagnetic.

    Parameters
    ----------
    symops : list of tuples
        List of tuples that contains symmetry operations [(R0, t0), ..., Rk, tk)],
        where Ri is 3x3 matrix and ti is 3-elements vector.
    atom_positions : array of arrays
        Array of arrays containing scaled (fractional) positions of atoms in
        a given structure, i.e. [[0,0,0], [0.5,0.5,0.5]].
    equiv_atoms : array of int
        Array that represents how atoms in the structure are split into group
        of symmetry equivalent atoms (Wyckoff orbit), i.e. [0 0 0 0 0 0 6 6]
        means that there are two Wyckoff orbits consisting with atoms of id
        0 and 6.
    chemical_symbols : list of strings
        List of chemical symbols designated to each atom, i.e.
        ['Mn', 'Mn', 'Mn', 'Mn', 'Mn', 'Mn', 'Fe', 'Fe'].
    spins : list of strings
        List of string objects denoting spin designation of each atom.
        Possible values are: "u", "U" for spin-up, "d", "D" for spin-down and
        "n", "N" to mark a non-magnetic atom.
    tol : float
        A tolerance parameter for numerical computations.
    verbose : Bool
        Print some additional information during the execution.

    Returns
    -------
    altermagnet : Bool
        Returns True if the orbit with a given spin configuration is

    Raises
    ------
    ValueError
        If number of up and down spins is not the same (spins should be
        compensated, otherwise it's not an altermagnet).
    RuntimeError
        If for some reason no Wyckoff orbit was checked for altermagnetism.
        This situation is possible when all atoms were marked as non-magnetic.
        Although non-magnetic structure is not an altermagnet, we still want
        to notify user to check for possible mistakes in the input.
    """

    # if symops list is empty, the spacegroup is definitely centrosymmetric,
    # thus the orbit is altermagnetic
    if not symops:
        return True

    altermagnet = False
    check_was_performed = False
    for u in np.unique(equiv_atoms):
        atom_ids = np.where(equiv_atoms == u)[0]
        orbit_positions = atom_positions[atom_ids]

        if verbose:
            print()
            print("Orbit of {} atoms:".format(chemical_symbols[atom_ids[0]]))

        if len(orbit_positions) == 1:
            print("Only one atom in the orbit: skipping.")
            continue

        orbit_spins = [spins[i] for i in atom_ids]

        # skip if the Wyckoff orbit consists of non-magnetic atoms
        if all(s == "n" for s in orbit_spins):
            print(
                "Group of non-magnetic atoms ({}): skipping.".format(chemical_symbols[u]))
            continue

        # a sanity check: number of up and down spins should be the same,
        # otherwise it's not a antiferromagnet/altermagnet
        N_u = orbit_spins.count("u")
        N_d = orbit_spins.count("d")
        if N_u != N_d:
            raise ValueError("Number of up spins should be the same as down spins: " +
                             "got {} up and {} down spins!".format(N_u, N_d))

        # a sanity check: in case if there is something wrong with input data
        # in the way that we never reach the next line, we need to report it
        check_was_performed = True
        is_orbit_altermagnetic = check_altermagnetism_orbit(symops,
                                                            orbit_positions, orbit_spins, tol, verbose)
        altermagnet |= is_orbit_altermagnetic
        if verbose:
            print("Altermagnetic orbit ({})?".format(chemical_symbols[u]),
                  is_orbit_altermagnetic)

    if not check_was_performed:
        raise RuntimeError("Something is wrong with the description of magnetic atoms!\n\
Have you provided a non-magnetic/ferromagnetic material?")

    return altermagnet


def input_spins(num_atoms):
    """
    Read list of spin designations for a given Wyckoff orbit from stdin.

    Parameters
    ----------
    num_atoms : int
        Number of atoms in the given Wyckoff orbit.

    Returns
    -------
    spins : list of strings
        List of string objects denoting spin designation of each atom.
        Possible values are: "u", "U" for spin-up, "d", "D" for spin-down and
        "n", "N" to mark a non-magnetic atom.
        "nn", "NN" can be used to mark entire Wyckoff orbit as non-magnetic

    Raises
    ------
    ValueError
        If the number of spins from input is not the same as num_atoms.
        If the number of up and down spins is not the same: for an altermagnet
        spins should be compensated.
    """

    print("Type spin (u, U, d, D, n, N, nn or NN) for each of them (space separated):")
    spins = input().split()
    # "normalize" spin designations to lowercase for an easier bookkeeping
    spins = [s.lower() for s in spins]

    # empty line or "nn" to mark all atoms in the orbit as nonmagnetic
    if len(spins) < 1 or spins[0] == 'nn':
        return ["n"]*num_atoms

    if len(spins) != num_atoms:
        raise ValueError(
            "Wrong number of spins: got {} instead of {}!".format(len(spins), num_atoms))

    if not all(s in ["u", "d", "n"] for s in spins):
        raise ValueError("Use u, U, d, D, n or N for spin designation!")

    N_u = spins.count("u")
    N_d = spins.count("d")
    if N_u != N_d:
        raise ValueError("Number of up spins should be the same as down spins: " +
                         "got {} up and {} down spins!".format(N_u, N_d))

    # all atoms in the orbit are nonmagnetic
    if N_u == 0:
        return ["n"]*num_atoms

    return spins


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

            # get space group number
            spglib_cell = (
                atoms.cell, atoms.get_scaled_positions(), atoms.numbers)
            sg = spglib.get_spacegroup(spglib_cell, symprec=args.symprec)
            sg_no = int(sg[sg.find('(') + 1:sg.find(')')])
            print("Spacegroup: {}".format(sg))

            # if space group number is present in the input file (i.e. in cif),
            # check if it is consistent with the one we just got from the spglib
            if "spacegroup" in atoms.info:
                if sg_no != atoms.info["spacegroup"].no:
                    print("WARNING: space group from the input is different from the spglib \
analysis: {} (spglib) vs {} (input)!".format(sg_no, atoms.info['spacegroup'].no))

            primitive = spglib.find_primitive((atoms.cell, atoms.get_scaled_positions(), atoms.numbers),
                                              symprec=args.symprec)
            prim_cell, prim_pos, prim_num = primitive

            # The given unit cell might be non-primitive and if it is the case
            # we will ask user: shall we keep using it or shall we use a
            # primitive one instead?
            if abs(np.linalg.det(atoms.cell) - np.linalg.det(prim_cell)) > args.tol:
                symmetry = spglib.get_symmetry(primitive, symprec=args.symprec)
                rotations = symmetry['rotations']
                translations = symmetry['translations']
                N_ops_prim = len(translations)

                print("There is a primitive unit cell!")
                print("Do you want to use it instead? (Y/n)")
                answer = input()
                if answer.lower() != "n":
                    print("Primitive unit cell will be used.")
                    atoms = Atoms(prim_num, cell=prim_cell,
                                  scaled_positions=prim_pos)
                    equiv_atoms = symmetry['equivalent_atoms']
                else:
                    print("Original non-primitive unit cell will be used.")

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
                    if args.verbose:
                        print(
                            "Transformation from primitive to non-primitive cell, T:")
                        print(T)

                    # All possible supercells can be grouped by det(T) and
                    # within each group the amount of possible distinct
                    # supercell is finite.
                    # All of them can be enumerated using the Hermite Normal
                    # Form, H.
                    H, _, _ = lllhermite(T)
                    H = np.matrix(H, dtype=int)
                    if args.verbose:
                        print("HNF of T:")
                        print(H)

                    # By knowing the HNF we can determine the direction and
                    # multiplicity of fractional translations and transform
                    # them into the basis of the original supercell
                    tau = [np.mod([i, j, k] @ prim_cell @ np.linalg.inv(cell), 1)
                           for i in range(H[0, 0]) for j in range(H[1, 1]) for k in range(H[2, 2])]

                    # The final collection of symmetry operators is a copy of
                    # original augmented by the new translations:
                    # (R,t) = (R0,t0)*{(E,0) + (E,t1) + ... + (E,tN)}
                    N = int(np.rint(np.linalg.det(H)))
                    rotations = np.tile(rotations, (N, 1, 1))
                    translations = np.tile(translations, (N, 1))
                    for (i, t) in enumerate(tau):
                        for j in range(i*N_ops_prim, (i+1)*N_ops_prim):
                            translations[j] = np.mod(translations[j] + t, 1)

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

            symops = []
            for (r, t) in zip(rotations, translations):
                # if there is an inversion or pure translation
                if (abs(np.trace(r)+3) < args.tol) or \
                   (abs(np.trace(r)-3) < args.tol and np.linalg.norm(t) > args.tol):
                    symops.append((r, t))

            if args.verbose:
                print("Relevant symmetry operations:")
                for (i, (r, t)) in enumerate(symops):
                    sym_type = ""
                    if (abs(np.trace(r)+3) < args.tol):
                        sym_type = "inversion"
                    if (abs(np.trace(r)-3) < args.tol and np.linalg.norm(t) > args.tol):
                        sym_type = "translation"

                    print("{}: {}".format(i+1, sym_type))
                    print(r)
                    print(t)

            # for convenience we will create an auxiliary file that user can
            # use to assign spins while examining the file in some visualizer
            aux_filename = filename+"_amcheck.vasp"
            print()
            print("Writing the used structure to auxiliary file: check {}.".format(
                aux_filename))
            ase.io.vasp.write_vasp(aux_filename, atoms, direct=True)

            # get spins as an input from user
            chemical_symbols = atoms.get_chemical_symbols()
            spins = []
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
                spins.append(orbit_spins)

            spins = [s for sublist in spins for s in sublist]  # flatten
            is_am = is_altermagnet(symops, atoms.get_scaled_positions(),
                                   equiv_atoms, chemical_symbols, spins,
                                   args.tol, args.verbose)

            print()
            print("Altermagnet?", is_am)

        except Exception as error:
            eprint("[ERROR] " + str(error))


def main_ahc_type(args):
    """
    Determine the type (up to an equivalency) of Anomalous Hall Coefficient 
    for a given structure.
    """

    import ahc_data

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
            print("Type magnetic moments for each atom ('mx my mz' or empty line \
for non-magnetic atom):")
            for i in range(len(positions)):
                mm = input()
                mm = list(map(float, mm.split()))

                # if user provides an empty line as input, the atom is
                # non-magnetic: no need to do anything
                if not mm:
                    continue

                if len(mm) != 3:
                    raise ValueError("Expected 3 numbers for magnetic moment \
definition!")
                magnetic_moments[i] = mm

            print("Magnetic moments assigned:")
            print(magnetic_moments)
            print()

            spglib_cell = (atoms.cell, atoms.get_scaled_positions(),
                           atoms.numbers, np.array(magnetic_moments))
            MSG = spglib.get_magnetic_symmetry_dataset(spglib_cell,
                                                       symprec=args.symprec, mag_symprec=args.mag_symprec)

            print("Magnetic Space Group:",
                  spglib.get_magnetic_spacegroup_type(MSG['uni_number']))
            print()
            print(ahc_data.ahc_types[ahc_data.get_ahc_type(MSG['uni_number'])])
        except Exception as error:
            eprint("[ERROR] " + str(error))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='amcheck.py',
        description='Checks if a given structure is an altermagnet.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="verbosely list the information during the execution")
    parser.add_argument('file', nargs='+',
                        help="name of the structure file to analyze")

    parser.add_argument('-s', '--symprec', default=DEFAULT_TOLERANCE, type=float,
                        help="tolerance used by spglib to do the symmetry analysis")
    parser.add_argument('-ms', '--mag_symprec', default=-1.0, type=float,
                        help="tolerance for magnetic moments used by spglib for magnetic search")
    parser.add_argument('-t', '--tol', '--tolerance', default=DEFAULT_TOLERANCE, type=float,
                        help="tolerance for numerical checks inside of amcheck.py")

    parser.add_argument('--ahc_type', action='store_true',
                        help="Determine the possible form of Anomalous Hall Coefficient")

    args = parser.parse_args()

    if args.ahc_type:
        main_ahc_type(args)
    else:
        main(args)
