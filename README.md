# amcheck
`amcheck` is the program/library to check, based on the symmetry arguments,
if a given structure is an altermagnet or not.

To describe the material of interest user is supposed to provide a crystal
structure and a magnetic pattern.
Note, that it is implicitly assumed that the net magnetic moment is
zero, i.e. if user types in a ferromagnet, the program will return
an error.
The underlying idea is that some pre-classification was already done
and the user wants to figure out if the given material is an altermagnet or not,
thus an antiferromagnet.

## Installation
The code is written in `python` and has the following libraries as its 
dependencies: `ase`, `spglib` and `diophantine`.
Which one can install using `pip`:
```
pip install ase spglib diophantine
```

## Usage example
```
$ amcheck.py Mn3Fe.cif Mn3Fe.cif
==========================================================
Processing: Mn3Fe.cif
----------------------------------------------------------
Spacegroup: P6_3/mmc (194)
Writing the used structure to auxiliary file:
check Mn3Fe.cif_amcheck.vasp.
Orbit of Mn atoms at positions:
1 (1) [0.156793 0.313586 0.75    ]
2 (2) [0.843207 0.686414 0.25    ]
3 (3) [0.686414 0.843207 0.75    ]
4 (4) [0.313586 0.156793 0.25    ]
5 (5) [0.156793 0.843207 0.75    ]
6 (6) [0.843207 0.156793 0.25    ]
Type spin (u, U, d, D, n, N, nn or NN) for each of them (space
separated):
u d u d u d
Orbit of Fe atoms at positions:
7 (1) [0.333333 0.666667 0.25    ]
8 (2) [0.666667 0.333333 0.75    ]
Type spin (u, U, d, D, n, N, nn or NN) for each of them (space
separated):
u d
Altermagnet? False
==========================================================
Processing: Mn3Fe.cif
----------------------------------------------------------
Spacegroup: P6_3/mmc (194)
Writing the used structure to auxiliary file:
check Mn3Fe.cif_amcheck.vasp.
Orbit of Mn atoms at positions:
1 (1) [0.156793 0.313586 0.75    ]
2 (2) [0.843207 0.686414 0.25    ]
3 (3) [0.686414 0.843207 0.75    ]
4 (4) [0.313586 0.156793 0.25    ]
5 (5) [0.156793 0.843207 0.75    ]
6 (6) [0.843207 0.156793 0.25    ]
Type spin (u, U, d, D, n, N, nn or NN) for each of them (space
separated):
u u u d d d
Orbit of Fe atoms at positions:
7 (1) [0.333333 0.666667 0.25    ]
8 (2) [0.666667 0.333333 0.75    ]
Type spin (u, U, d, D, n, N, nn or NN) for each of them (space
separated):
u d
Altermagnet? True
```
