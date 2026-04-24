"""
VASP ZVAL valence electron counts per element.

Values match the VASP PBE PAW potentials used to generate the
Materials Project and GNoME datasets (same as ELECTRAFY paper).
Sourced from ELECTRA codebase and extended for periodic-materials elements.
"""

import torch
from ase.data import atomic_numbers

# Element symbol -> ZVAL (VASP PAW-PBE valence electron count)
VASP_ZVAL: dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 2,
    "B": 3,
    "C": 4,
    "N": 5,
    "O": 6,
    "F": 7,
    "Ne": 8,
    "Na": 7,
    "Mg": 2,
    "Al": 3,
    "Si": 4,
    "P": 5,
    "S": 6,
    "Cl": 7,
    "Ar": 8,
    "K": 9,
    "Ca": 10,
    "Sc": 11,
    "Ti": 12,
    "V": 13,
    "Cr": 12,
    "Mn": 13,
    "Fe": 8,
    "Co": 9,
    "Ni": 16,
    "Cu": 17,
    "Zn": 12,
    "Ga": 13,
    "Ge": 14,
    "As": 5,
    "Se": 6,
    "Br": 7,
    "Kr": 8,
    "Rb": 9,
    "Sr": 10,
    "Y": 11,
    "Zr": 10,
    "Nb": 13,
    "Mo": 14,
    "Tc": 13,
    "Ru": 14,
    "Rh": 15,
    "Pd": 10,
    "Ag": 11,
    "Cd": 12,
    "In": 13,
    "Sn": 4,
    "Sb": 5,
    "Te": 6,
    "I": 7,
    "Xe": 8,
    "Cs": 9,
    "Ba": 10,
    "La": 11,
    "Ce": 12,
    "Pr": 11,
    "Nd": 11,
    "Pm": 11,
    "Sm": 11,
    "Eu": 8,
    "Gd": 9,
    "Tb": 9,
    "Dy": 9,
    "Ho": 9,
    "Er": 9,
    "Tm": 9,
    "Yb": 8,
    "Lu": 9,
    "Hf": 10,
    "Ta": 11,
    "W": 14,
    "Re": 13,
    "Os": 8,
    "Ir": 9,
    "Pt": 10,
    "Au": 11,
    "Hg": 12,
    "Tl": 13,
    "Pb": 14,
    "Bi": 15,
    "Po": 16,
    "At": 7,
    "Rn": 8,
    "Fr": 1,
    "Ra": 2,
    "Ac": 11,
    "Th": 12,
    "Pa": 13,
    "U": 14,
    "Np": 15,
    "Pu": 16,
    "Am": 17,
    "Cm": 18,
}

# Maximum ZVAL across all elements - determines channel count C = M * MAX_ZVAL
MAX_ZVAL: int = max(VASP_ZVAL.values())  # 18 (Cm)


def build_zval_lookup(max_atomic_number: int = 100) -> torch.Tensor:
    """
    Build a lookup tensor: zval_lookup[atomic_number] = ZVAL.

    :param max_atomic_number: Largest atomic number to include.
    :return: Integer tensor of shape (max_atomic_number + 1,).
    """
    from ase.data import chemical_symbols

    table = torch.zeros(max_atomic_number + 1, dtype=torch.long)
    for z in range(1, max_atomic_number + 1):
        sym = chemical_symbols[z]
        table[z] = VASP_ZVAL.get(sym, 0)
    return table


# Pre-built lookup table (atomic number -> ZVAL)
ZVAL_LOOKUP: torch.Tensor = build_zval_lookup()
