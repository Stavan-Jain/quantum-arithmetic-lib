from typing import List
from .gates import Circuit


def compare_const(circ: Circuit, x_bits: List[int], N: int, flag_qubit: int):
    circ.add("CMP_CONST", *x_bits, flag_qubit, N=N)


def sub_const_if_flag(
    circ: Circuit,
    x_bits: List[int],
    N: int,
    flag_qubit: int,
    carry_anc: int,
):
    circ.add("SUB_CONST_IF", *x_bits, flag_qubit, N=N)
