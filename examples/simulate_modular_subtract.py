"""Simulate the modular quantum-classical subtractor from `qarith/operators.py`."""

from __future__ import annotations

import argparse
from typing import Dict

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qarith.operators import modular_subtract_quantum_classical


def _load_value(circ: QuantumCircuit, qubits: list[int], value: int) -> None:
    """Store `value` on `qubits` (little-endian)."""
    if value >= 1 << len(qubits):
        raise ValueError(f"value {value} does not fit in {len(qubits)} bits")
    for bit in range(len(qubits)):
        if (value >> bit) & 1:
            circ.x(qubits[bit])


def simulate_modular_subtract(
    n_bits: int, initial_value: int, subtract_value: int, modulus: int, shots: int
) -> tuple[QuantumCircuit, Dict[str, int]]:
    """Construct the subtraction circuit and sample the result register."""
    total_qubits = n_bits + (n_bits + 2)
    circ = QuantumCircuit(total_qubits)

    a_qubits = list(range(n_bits))
    aux_qubits = list(range(n_bits, total_qubits))

    _load_value(circ, a_qubits, initial_value)
    modular_subtract_quantum_classical(
        circ, a_qubits, subtract_value, modulus, aux_qubits
    )

    result_register = aux_qubits[1 : n_bits + 1]
    state = Statevector.from_instruction(circ)
    counts = state.sample_counts(shots=shots, qargs=result_register)
    return circ, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate modular_subtract_quantum_classical"
    )
    parser.add_argument("--n-bits", type=int, default=3, help="register width")
    parser.add_argument("--initial", type=int, default=3, help="value stored in |a⟩")
    parser.add_argument(
        "--subtract", type=int, default=5, help="classical value to subtract"
    )
    parser.add_argument("--modulus", type=int, default=7, help="modulus N")
    parser.add_argument("--shots", type=int, default=1024, help="samples to draw")
    args = parser.parse_args()

    circ, counts = simulate_modular_subtract(
        n_bits=args.n_bits,
        initial_value=args.initial,
        subtract_value=args.subtract,
        modulus=args.modulus,
        shots=args.shots,
    )

    expected = (args.initial - args.subtract) % args.modulus
    print("Circuit:")
    print(circ.draw(fold=-1))
    print()
    print(f"Result register sample counts (little-endian bitstrings):")
    for bitstring, freq in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {bitstring}: {freq}")
    print(f"Expected result: {expected}")


if __name__ == "__main__":
    main()
