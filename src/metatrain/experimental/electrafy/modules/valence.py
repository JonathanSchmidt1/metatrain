"""
VASP ZVAL valence electron counts per element.

Values match the VASP PBE PAW potentials used to generate the
Materials Project and GNoME datasets (same as ELECTRAFY paper).

The paper's Appendix B.3 prescribes (Z, v_a) admissible-set embeddings: for
each element Z there is a small admissible set {v(0)(Z), v(1)(Z), ...} and
each (Z, v) pair gets a distinct embedding. We expose:

  - ``VASP_ZVAL`` and ``ZVAL_LOOKUP``: canonical default ZVAL per element
    (matches what VASP/MP write into the CHGCAR for each material).
  - ``ADMISSIBLE_ZVALS``: per-element admissible-set table for the
    pseudopotentials documented for VASP PAW-PBE (e.g. W:6, W:14;
    Mn:7, Mn:13; Cu:11, Cu:17; ...).
  - :func:`build_zv_index_lookup`: builds a ``(max_Z+1, max_zval+1) -> int``
    embedding index table consumable as the embedding-table input.
"""

from typing import Dict, List

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
    Build a lookup tensor: zval_lookup[atomic_number] = canonical ZVAL.
    """
    from ase.data import chemical_symbols

    table = torch.zeros(max_atomic_number + 1, dtype=torch.long)
    for z in range(1, max_atomic_number + 1):
        sym = chemical_symbols[z]
        table[z] = VASP_ZVAL.get(sym, 0)
    return table


# Pre-built lookup table (atomic number -> canonical ZVAL).
ZVAL_LOOKUP: torch.Tensor = build_zval_lookup()


# Admissible (Z, v_a) sets — VASP PAW-PBE pseudopotentials with multiple
# valence options. Lists are ordered with the canonical default first
# (matches ``VASP_ZVAL``). Single-valence elements are omitted; clients
# should fall back to ``[VASP_ZVAL[sym]]`` in that case.
#
# Sourced from the VASP PAW-PBE 5.4 pseudopotential collection (the family
# used by Materials Project for charge-density calcs).
ADMISSIBLE_ZVALS: Dict[str, List[int]] = {
    "Li": [3, 1],          # Li, Li_sv (3e is canonical for MP)
    "Be": [2, 4],          # Be, Be_sv
    "Na": [7, 1, 9],       # Na_pv, Na, Na_sv (MP uses 7)
    "K": [9, 7],           # K_sv, K_pv
    "Ca": [10, 8],         # Ca_sv, Ca_pv
    "Sc": [11, 9, 3],      # Sc_sv, Sc, Sc_pv (MP varies)
    "Ti": [12, 10, 4],     # Ti_sv, Ti_pv, Ti
    "V": [13, 11, 5],      # V_sv, V_pv, V
    "Cr": [12, 14, 6],     # Cr_pv (12), Cr_sv (14), Cr (6)
    "Mn": [13, 15, 7],     # Mn_pv (13), Mn_sv (15), Mn (7)
    "Fe": [8, 14, 16],     # Fe (8), Fe_pv (14), Fe_sv (16)
    "Co": [9, 15, 17],     # Co (9), Co_pv, Co_sv
    "Ni": [16, 10, 18],    # Ni_pv (16), Ni (10), Ni_sv
    "Cu": [17, 11, 19],    # Cu_pv (17), Cu (11), Cu_sv
    "Zn": [12, 20],        # Zn (12), Zn_sv
    "Rb": [9, 7],          # Rb_sv, Rb_pv
    "Sr": [10, 8],         # Sr_sv, Sr_pv
    "Y": [11, 9, 3],
    "Zr": [10, 12, 4],     # Zr_sv (12), Zr_pv (10), Zr (4)
    "Nb": [13, 11, 5],
    "Mo": [14, 12, 6],
    "Tc": [13, 15, 7],
    "Ru": [14, 16, 8],
    "Rh": [15, 9],
    "Pd": [10, 16],
    "Ag": [11, 17, 19],
    "Cd": [12, 20],
    "Cs": [9, 1],
    "Ba": [10, 8],
    "Hf": [10, 12, 4],
    "Ta": [11, 13, 5],
    "W": [14, 6, 12],      # W_pv (14), W (6), W_sv (12)
    "Re": [13, 7, 15],
    "Os": [8, 14, 16],
    "Ir": [9, 15],
    "Pt": [10, 16],
    "Au": [11, 17, 19],
}


# Default admissible set for an element: first canonical, then single-element
# fallback. Used when building the (Z, v) embedding index table.
def _admissible_for(symbol: str) -> List[int]:
    if symbol in ADMISSIBLE_ZVALS:
        return ADMISSIBLE_ZVALS[symbol]
    if symbol in VASP_ZVAL:
        return [VASP_ZVAL[symbol]]
    return []


def build_zv_index_lookup(
    max_atomic_number: int = 100,
) -> "ZVIndex":
    """
    Build a (Z, v) -> embedding-index lookup.

    Returns a :class:`ZVIndex` that maps ``(atomic_number, ZVAL)`` to a unique
    integer embedding index in ``[0, num_embeddings)``. The first slot per
    element is the canonical default and is what callers should use when no
    explicit ZVAL annotation is available.
    """
    from ase.data import chemical_symbols

    pair_to_idx: Dict[tuple, int] = {}
    canonical_for_z: Dict[int, int] = {}  # z -> embedding index of default
    next_idx = 0
    for z in range(1, max_atomic_number + 1):
        sym = chemical_symbols[z]
        admissible = _admissible_for(sym)
        if not admissible:
            continue
        canonical_for_z[z] = next_idx  # canonical = first in admissible list
        for v in admissible:
            pair_to_idx[(z, int(v))] = next_idx
            next_idx += 1

    # Dense lookup table for fast forward-time indexing.
    max_z = max(canonical_for_z) if canonical_for_z else max_atomic_number
    max_v = max(v for (_, v) in pair_to_idx) if pair_to_idx else 0
    table = torch.full((max_z + 1, max_v + 1), -1, dtype=torch.long)
    for (z, v), idx in pair_to_idx.items():
        table[z, v] = idx
    canonical_table = torch.full((max_z + 1,), -1, dtype=torch.long)
    for z, idx in canonical_for_z.items():
        canonical_table[z] = idx

    return ZVIndex(
        table=table,
        canonical_for_z=canonical_table,
        num_embeddings=next_idx,
        pair_to_idx=pair_to_idx,
    )


class ZVIndex:
    """Container for the (Z, v) -> embedding-index lookup."""

    def __init__(
        self,
        table: torch.Tensor,
        canonical_for_z: torch.Tensor,
        num_embeddings: int,
        pair_to_idx: Dict[tuple, int],
    ) -> None:
        self.table = table                     # (max_z+1, max_v+1) long
        self.canonical_for_z = canonical_for_z  # (max_z+1,) long
        self.num_embeddings = num_embeddings
        self.pair_to_idx = pair_to_idx

    def index_for(self, z: int, v: int) -> int:
        idx = int(self.table[z, v].item())
        if idx < 0:
            raise KeyError(f"({z}, {v}) not in admissible (Z, v) set")
        return idx

    def canonical_index_for(self, z: int) -> int:
        idx = int(self.canonical_for_z[z].item())
        if idx < 0:
            raise KeyError(f"no canonical (Z, v) for Z={z}")
        return idx
