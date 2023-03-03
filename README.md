# amcheck (WIP)
`amcheck` is the program/library to check, based on the symmetry arguments,
if a given material is an altermagnet or not.

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
dependencies: `ase`, `spglib` and `diophantine`, that can be installed using `pip`:
```
pip install ase spglib diophantine
```

## Usage
To use it as a command line tool, one provides one or more structure files
(the code will internally loops over all listed files) and, when prompted,
types in spin designation for each atom: 'u' or 'U' for spin-up, 'd' or 'D'
for spin-down and 'n' or 'N' if atom is non-magnetic.
All atoms will be grouped into sets of symmetry-related atoms (orbits) and user
will need to provide spin designations per such a group.
To mark entire group as non-magnetic one can use 'nn' or 'NN' designation.
Note that here we treat spins as pseudoscalars (up and down, black and white),
not as pseudovectors and, thus, no spacial anisotropy for spins is assumed.

## Examples
### Checking if the material is altermagnetic
```
$ amcheck.py Mn3Fe.cif Mn3Fe.cif
================================================================================
Processing: examples/Mn3Fe.cif
--------------------------------------------------------------------------------
Spacegroup: P6_3/mmc (194)

Writing the used structure to auxiliary file: check examples/Mn3Fe.cif_amcheck.vasp.

Orbit of Mn atoms at positions:
1 (1) [0.156793 0.313586 0.75    ]
2 (2) [0.843207 0.686414 0.25    ]
3 (3) [0.686414 0.843207 0.75    ]
4 (4) [0.313586 0.156793 0.25    ]
5 (5) [0.156793 0.843207 0.75    ]
6 (6) [0.843207 0.156793 0.25    ]
Type spin (u, U, d, D, n, N, nn or NN) for each of them (space separated):
u d u d u d

Orbit of Fe atoms at positions:
7 (1) [0.333333 0.666667 0.25    ]
8 (2) [0.666667 0.333333 0.75    ]
Type spin (u, U, d, D, n, N, nn or NN) for each of them (space separated):
u d

Altermagnet? False
================================================================================
Processing: examples/Mn3Fe.cif
--------------------------------------------------------------------------------
Spacegroup: P6_3/mmc (194)

Writing the used structure to auxiliary file: check examples/Mn3Fe.cif_amcheck.vasp.

Orbit of Mn atoms at positions:
1 (1) [0.156793 0.313586 0.75    ]
2 (2) [0.843207 0.686414 0.25    ]
3 (3) [0.686414 0.843207 0.75    ]
4 (4) [0.313586 0.156793 0.25    ]
5 (5) [0.156793 0.843207 0.75    ]
6 (6) [0.843207 0.156793 0.25    ]
Type spin (u, U, d, D, n, N, nn or NN) for each of them (space separated):
u u u d d d

Orbit of Fe atoms at positions:
7 (1) [0.333333 0.666667 0.25    ]
8 (2) [0.666667 0.333333 0.75    ]
Type spin (u, U, d, D, n, N, nn or NN) for each of them (space separated):
u d

Altermagnet? True
```


### Using as a library
Here is a code snippet providing an example on how to use the `amcheck` as a
library:
```
from amcheck import is_altermagnet

symops = [(np.array([[-1,  0,  0], [0, -1,  0], [0,  0, -1]],
           dtype=int),
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
spins = ['u', 'u', 'u', 'd', 'd', 'd', 'u', 'd']
is_altermagnet(symops, positions, equiv_atoms, chem_symbols, spins)
```


### Determining the form of Anomalous Hall coefficient
```
$ amcheck.py FeSb2.cif
================================================================================
Processing: FeSb2.cif
--------------------------------------------------------------------------------
List of atoms:
Fe [0. 0. 0.]
Fe [0.5 0.5 0.5]
Sb [0.1885 0.3561 0.    ]
Sb [0.8115 0.6439 0.    ]
Sb [0.3115 0.8561 0.5   ]
Sb [0.6885 0.1439 0.5   ]

Type magnetic moments for each atom ('mx my mz' or empty line for non-magnetic atom):
1 0 0
-1 0 0
0 0 0
0 0 0
0 0 0
0 0 0
Magnetic moments assigned:
[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

Magnetic Space Group: {'uni_number': 496, 'litvin_number': 476, 'bns_number': '58.398', 'og_number': '58.6.476', 'number': 58, 'type': 3}

 σxx  σxy   0
-σxy  σyy   0
  0    0   σzz
```


## Contributors
Andriy Smolyanyuk[1], Olivia Taiwo[2], Libor Šmejkal[3] and Igor I. Mazin[4]

[1] Technische Universität Wien, Vienna, Austria

[2] Princeton University, Princeton, USA

[3] Johannes Gutenberg Universität Mainz, Mainz, Germany

[4] George Mason University, Fairfax, USA

## How to cite
The preparation of the manuscript describing the underlying ideas is currently
in process.
