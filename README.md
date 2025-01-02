# amcheck
`amcheck` is the program (and a library) to check, based on the symmetry arguments,
whether a given material is an altermagnet.

Altermagnet is a collinear magnet with symmetry-enforced zero net magnetization,
where the opposite spin sublattices are coupled by symmetry operations that are not inversions or translations.

The user is supposed to provide a crystal structure and a magnetic pattern to describe the material of interest.
It is implicitly assumed that the net magnetic moment is zero, i.e. if user types in a ferromagnet, the program will return an error.
The underlying idea is that some pre-classification was already done
and the user wants to figure out if the given material is an altermagnet or not, thus an antiferromagnet.

## Installation
The code is written in `python` and can be installed using `pip`:
```
pip install amcheck
```
It has the following packages among its dependencies: `ase`, `spglib` and `diophantine`.

Depending on your OS and the setup of the command line tool used, one might need to make adjustments
if the `amcheck` executable is not found (i.e., if one encounters `amcheck: command not found` or similar error).
Please consult the
[Installing to the User Site](https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-to-the-user-site)
chapter of the Python Packaging User Guide to see how to fix the issue.

## Installation in a virtual environment
Alternatively, it might be more convenient to install the `amcheck` package in its own virtual environment:
in this case, there is no interference between various packages, especially if they depend on concurring versions of the same library.

Here, we briefly describe how to set up a virtual environment using `venv` tool (examples are Linux-specific).
Create an environment with
```
python -m venv py-amcheck
```
Activate it using:
```
source py-amcheck/bin/activate
```
Install the `amcheck` package using `pip` when the virtual environment is used for the first time:
```
pip install amcheck
```

After the package is installed, the `amcheck` command and the `python` module are available in `py-amcheck` environment.
Use `deactivate` to exit the environment and return to the basic shell.

For more details on the virtual environments, see the following [link](https://docs.python.org/3/tutorial/venv.html).

## Usage
To use it as a command line tool, one provides one or more structure files
(the code will internally loop over all listed files) and, when prompted,
types in spin designation for each atom: 'u' or 'U' for spin-up, 'd' or 'D'
for spin-down and 'n' or 'N' if the atom is non-magnetic.
All atoms will be grouped into sets of symmetry-related atoms (orbits),
and the user will need to provide spin designations per such a group.
To mark the entire group as non-magnetic, one can use the 'nn' or 'NN' designation.
Note that here, we treat spins as pseudoscalars (up and down, black and white),
not as pseudovectors, and thus, no spatial anisotropy for spins is assumed.

## Examples
### Checking if a given material is altermagnetic
```
$ amcheck FeO.vasp MnTe.vasp
==========================================================
Processing: FeO.vasp
----------------------------------------------------------
Spacegroup: P6_3/mmc (194)
Writing the used structure to auxiliary file:
check FeO.vasp_amcheck.vasp.

Orbit of Fe atoms at positions:
1 (1) [0.33333334 0.66666669 0.25      ]
2 (2) [0.66666663 0.33333331 0.75      ]
Type spin (u, U, d, D, n, N, nn or NN) for each of them (space
separated):
u d

Orbit of O atoms at positions:
3 (1) [0. 0. 0.]
4 (2) [0.  0.  0.5]
Type spin (u, U, d, D, n, N, nn or NN) for each of them (space
separated):
n n
Group of non-magnetic atoms (O): skipping.

Altermagnet? False
==========================================================
Processing: MnTe.vasp
----------------------------------------------------------
Spacegroup: P6_3/mmc (194)
Writing the used structure to auxiliary file:
check MnTe.vasp_amcheck.vasp.

Orbit of Mn atoms at positions:
1 (1) [0. 0. 0.]
2 (2) [0.  0.  0.5]
Type spin (u, U, d, D, n, N, nn or NN) for each of them (space
separated):
u d

Orbit of Te atoms at positions:
3 (1) [0.33333334 0.66666669 0.25      ]
4 (2) [0.66666663 0.33333331 0.75      ]
Type spin (u, U, d, D, n, N, nn or NN) for each of them (space
separated):
nn
Group of non-magnetic atoms (Te): skipping.

Altermagnet? True
```


### Using as a library
Here is a code snippet providing an example on how to use the `amcheck` as a
library:
```python
import numpy as np
from amcheck import is_altermagnet

symmetry_operations = [(np.array([[-1,  0,  0],
                                  [ 0, -1,  0],
                                  [ 0,  0, -1]],
                                 dtype=int),
                       np.array([0.0, 0.0, 0.0])),
                       # for compactness reasons,
                       # other symmetry operations are omitted
                       # from this example
                       ]

# positions of atoms in NiAs structure: ["Ni", "Ni", "As", "As"]
positions = np.array([[0.00, 0.00, 0.00],
                      [0.00, 0.00, 0.50],
                      [1/3., 2/3., 0.25],
                      [2/3., 1/3., 0.75]])

equiv_atoms  = [0, 0, 1, 1]

# high-pressure FeO: Fe at As positions, O at Ni positions => afm
chem_symbols = ["O", "O", "Fe", "Fe"]
spins = ["n", "n", "u", "d"]
print(is_altermagnet(symmetry_operations, positions, equiv_atoms,
                     chem_symbols, spins))

# MnTe: Mn at Ni positions, Te at As positions => am
chem_symbols = ["Mn", "Mn", "Te", "Te"]
spins = ["u", "d", "n", "n"]
print(is_altermagnet(symmetry_operations, positions, equiv_atoms,
                     chem_symbols, spins))
```


### Determining the form of Anomalous Hall coefficient
```
$ amcheck --ahc RuO2.vasp
==========================================================
Processing: RuO2.vasp
----------------------------------------------------------
List of atoms:
Ru [0. 0. 0.]
Ru [0.5 0.5 0.5]
O [0.30557999 0.30557999 0.        ]
O [0.19442001 0.80558002 0.5       ]
O [0.80558002 0.19442001 0.5       ]
O [0.69441998 0.69441998 0.        ]

Type magnetic moments for each atom in Cartesian coordinates
('mx my mz' or empty line for non-magnetic atom):
 1  1  0
-1 -1  0
 0  0  0
 0  0  0
 0  0  0
 0  0  0

Assigned magnetic moments:
[[1.0, 1.0, 0.0], [-1.0, -1.0, 0.0], [0, 0, 0], [0, 0, 0],
[0, 0, 0], [0, 0, 0]] 

Magnetic Space Group: {'uni_number': 584,
'litvin_number': 550, 'bns_number': '65.486',
'og_number': '65.6.550', 'number': 65, 'type': 3}

Conductivity tensor:
[[ 'xx'  'xy' '-yz']
 [ 'xy'  'xx'  'yz']
 [ 'yz' '-yz'  'zz']]

The antisymmetric part of the conductivity tensor
 (Anomalous Hall Effect):
[['0'   '0' '-yz']
 ['0'   '0'  'yz']
 ['yz' '-yz' '0']]

Hall vector:
['-yz', '-yz', '0']
```


## Contributors
Andriy Smolyanyuk[1], Libor Šmejkal[2] and Igor I. Mazin[3]

[1] Institute of Solid State Physics, TU Wien, 1040 Vienna, Austria

[2] Johannes Gutenberg Universität Mainz, Mainz, Germany

[3] George Mason University, Fairfax, USA

## How to cite
If you're using the `amcheck` package, please cite the manuscript describing the underlying ideas:
[A tool to check whether a symmetry-compensated collinear magnetic material is antiferro- or altermagnetic](https://scipost.org/SciPostPhysCodeb.30).

```bibtex
@Article{10.21468/SciPostPhysCodeb.30,
	title={{A tool to check whether a symmetry-compensated collinear magnetic material is antiferro- or altermagnetic}},
	author={Andriy Smolyanyuk and Libor \v{S}mejkal and Igor I. Mazin},
	journal={SciPost Phys. Codebases},
	pages={30},
	year={2024},
	publisher={SciPost},
	doi={10.21468/SciPostPhysCodeb.30},
	url={https://scipost.org/10.21468/SciPostPhysCodeb.30},
}

@Article{10.21468/SciPostPhysCodeb.30-r1.0,
	title={{Codebase release r1.0 for amcheck}},
	author={Andriy Smolyanyuk and Libor \v{S}mejkal and Igor I. Mazin},
	journal={SciPost Phys. Codebases},
	pages={30-r1.0},
	year={2024},
	publisher={SciPost},
	doi={10.21468/SciPostPhysCodeb.30-r1.0},
	url={https://scipost.org/10.21468/SciPostPhysCodeb.30-r1.0},
}
```
