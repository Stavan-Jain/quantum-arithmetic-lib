"""Quantum comparator circuits built from ripple-carry subtractors."""

from __future__ import annotations

from typing import Sequence

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit

QubitLike = Qubit | int


def apply_less_than(
    circ: QuantumCircuit,
    a_qubits: Sequence[QubitLike],
    b_qubits: Sequence[QubitLike],
    work_qubits: Sequence[QubitLike],
    carry_qubits: Sequence[QubitLike],
    flag_qubit: QubitLike,
) -> None:
    """
    Compute the predicate (a < b) into ``flag_qubit`` while restoring all inputs.

    Implementation strategy:
      1. Copy ``b`` into a work register and form the two's complement (~b + 1).
      2. Run the forward pass of an in-place ripple adder on ``a`` and the work
         register, which computes carries c[i+1].
      3. Copy the final carry bit into the flag (carry=0 indicates borrow).
      4. Reverse the adder and clean up temporary registers.

    Args:
        circ: QuantumCircuit under construction.
        a_qubits: Qubit indices holding the first operand (little-endian).
        b_qubits: Qubit indices holding the second operand (little-endian).
        work_qubits: Scratch space (n qubits) initialized to |0...0⟩.
        carry_qubits: Carry register (n+1 qubits) initialized to |0...0⟩.
        flag_qubit: Single qubit initialized to |0⟩ where the result is stored.
    """
    n = len(a_qubits)
    if len(b_qubits) != n:
        raise ValueError("a and b must have the same length")
    if len(work_qubits) != n:
        raise ValueError("work register must have length n")
    if len(carry_qubits) != n + 1:
        raise ValueError("carry register must have length n+1")

    # Copy b into work
    for i in range(n):
        circ.cx(b_qubits[i], work_qubits[i])

    # work = ~b
    for i in range(n):
        circ.x(work_qubits[i])

    # Add +1 by flipping the carry-in
    circ.x(carry_qubits[0])

    # Forward pass of ripple adder (build carries)
    for i in range(n):
        circ.ccx(a_qubits[i], work_qubits[i], carry_qubits[i + 1])
        circ.ccx(a_qubits[i], carry_qubits[i], carry_qubits[i + 1])
        circ.ccx(work_qubits[i], carry_qubits[i], carry_qubits[i + 1])

    # Carry-out is 1 iff a >= b; invert to obtain (a < b)
    circ.cx(carry_qubits[n], flag_qubit)
    circ.x(flag_qubit)

    # Reverse the forward pass to clean carries
    for i in reversed(range(n)):
        circ.ccx(work_qubits[i], carry_qubits[i], carry_qubits[i + 1])
        circ.ccx(a_qubits[i], carry_qubits[i], carry_qubits[i + 1])
        circ.ccx(a_qubits[i], work_qubits[i], carry_qubits[i + 1])

    # Reset carry-in
    circ.x(carry_qubits[0])

    # Uncompute work register back to |0...0>
    for i in range(n):
        circ.x(work_qubits[i])
    for i in range(n):
        circ.cx(b_qubits[i], work_qubits[i])


def less_than_circuit(n_bits: int) -> QuantumCircuit:
    """Standalone circuit producing |a>|b>|lt> with lt=1 iff a < b."""
    a = QuantumRegister(n_bits, "a")
    b = QuantumRegister(n_bits, "b")
    work = QuantumRegister(n_bits, "work")
    carry = QuantumRegister(n_bits + 1, "carry")
    flag = QuantumRegister(1, "lt")

    qc = QuantumCircuit(a, b, work, carry, flag)
    apply_less_than(
        qc,
        list(a),
        list(b),
        list(work),
        list(carry),
        flag[0],
    )
    return qc
