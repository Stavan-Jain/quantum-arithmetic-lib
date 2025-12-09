"""Simulate the inplace modular adder defined in `qarith/operators.py`.

Run this script to prepare a classical input value on the adder register,
apply the `modular_addin_circuit`, and sample the resulting distribution of
the data qubits to verify that the operation adds `k` modulo `N`.
"""

from __future__ import annotations

import argparse
from typing import Dict

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Allow running without installing the package by adding repo root to sys.path
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qarith.operators import modular_addin_circuit


def _load_value(circ: QuantumCircuit, value: int, n_bits: int) -> None:
    """Encode `value` onto the adder register (qubits 1..n_bits, little-endian)."""
    if value >= 1 << n_bits:
        raise ValueError(f"value {value} does not fit in {n_bits} bits")
    for bit in range(n_bits):
        if (value >> bit) & 1:
            circ.x(1 + bit)


def simulate_inplace_modular_add(
    n_bits: int, initial_value: int, add_value: int, modulus: int, shots: int
) -> tuple[QuantumCircuit, Dict[str, int]]:
    """Build the modular-addition circuit and sample the data qubits."""
    circ = QuantumCircuit(n_bits + 2)
    _load_value(circ, initial_value, n_bits)

    addin = modular_addin_circuit(n_bits, add_value, modulus)
    circ.compose(addin, range(n_bits + 2), inplace=True)

    # Use a statevector to sample the data register (qubits 1..n_bits).
    data_qubits = list(range(1, n_bits + 1))
    state = Statevector.from_instruction(circ)
    counts = state.sample_counts(shots=shots, qargs=data_qubits)
    return circ, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate the inplace modular adder from qarith.operators"
    )
    parser.add_argument("--n-bits", type=int, default=3, help="number of data qubits")
    parser.add_argument(
        "--initial",
        type=int,
        default=3,
        help="initial value encoded on the adder register",
    )
    parser.add_argument(
        "--add",
        type=int,
        default=5,
        help="classical integer added by modular_addin_circuit",
    )
    parser.add_argument("--modulus", type=int, default=7, help="arithmetic modulus N")
    parser.add_argument("--shots", type=int, default=1024, help="samples to draw")
    args = parser.parse_args()

    circ, counts = simulate_inplace_modular_add(
        n_bits=args.n_bits,
        initial_value=args.initial,
        add_value=args.add,
        modulus=args.modulus,
        shots=args.shots,
    )

    expected = (args.initial + args.add) % args.modulus
    print("Circuit:")
    print(circ.draw(fold=-1))
    print()
    print(
        f"Data qubits (1..{args.n_bits}) sample counts "
        "(bitstrings are little-endian):"
    )
    for bitstring, freq in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {bitstring}: {freq}")
    print(f"Expected value on the data register: {expected}")


if __name__ == "__main__":
    main()
