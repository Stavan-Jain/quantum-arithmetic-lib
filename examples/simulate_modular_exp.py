"""Simulate the modular quantum-classical exponentiation from `qarith/operators.py`.

This simulates a^x mod N where:
- a is a classical constant (base)
- x is a quantum register (exponent)
- N is the modulus

The result is computed into a target register initialized to |1⟩.
"""

from __future__ import annotations

import argparse
from typing import Dict
from math import gcd

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qarith.operators import modular_exponentiate_quantum_classical


def _load_value(circ: QuantumCircuit, qubits: list[int], value: int) -> None:
    """Store `value` on `qubits` (little-endian)."""
    if value >= 1 << len(qubits):
        raise ValueError(f"value {value} does not fit in {len(qubits)} bits")
    for bit in range(len(qubits)):
        if (value >> bit) & 1:
            circ.x(qubits[bit])


def simulate_modular_exp(
    n_bits: int,
    base: int,
    exp_value: int,
    modulus: int,
    shots: int,
) -> tuple[QuantumCircuit, Dict[str, int]]:
    """Construct the exponentiation circuit and sample the result register.

    Args:
        n_bits: Number of bits for exponent and result registers
        base: Classical base value (a)
        exp_value: Value to load into exponent register (x)
        modulus: Modulus N
        shots: Number of samples to draw

    Returns:
        Tuple of (circuit, sample_counts)
    """
    # Layout: [exp_qubits (n_bits)][target_qubits (n_bits)][aux_qubits (n_bits+2)]
    total_qubits = n_bits + n_bits + (n_bits + 2)
    circ = QuantumCircuit(total_qubits)

    exp_qubits = list(range(n_bits))
    target_qubits = list(range(n_bits, 2 * n_bits))
    aux_qubits = list(range(2 * n_bits, total_qubits))

    # Load exponent value into exponent register
    _load_value(circ, exp_qubits, exp_value)

    # Initialize target register to |1⟩ (required for exponentiation)
    if n_bits > 0:
        circ.x(target_qubits[0])  # LSB = 1

    # Apply modular exponentiation
    modular_exponentiate_quantum_classical(
        circ, exp_qubits, base, modulus, target_qubits, aux_qubits
    )

    state = Statevector.from_instruction(circ)
    counts = state.sample_counts(shots=shots, qargs=target_qubits)
    return circ, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate modular_exponentiate_quantum_classical (a^x mod N)"
    )
    parser.add_argument("--n-bits", type=int, default=3, help="register width")
    parser.add_argument("--base", type=int, default=3, help="classical base value (a)")
    parser.add_argument(
        "--exp",
        type=int,
        default=2,
        help="value to load into exponent register (x)",
    )
    parser.add_argument("--modulus", type=int, default=7, help="modulus N")
    parser.add_argument("--shots", type=int, default=1024, help="samples to draw")
    args = parser.parse_args()

    # Check coprimality
    if gcd(args.base, args.modulus) != 1:
        print(
            f"Warning: base {args.base} and modulus {args.modulus} are not coprime. "
            f"Exponentiation may not be well-defined."
        )

    if args.exp >= 1 << args.n_bits:
        raise ValueError(
            f"Exponent value {args.exp} does not fit in {args.n_bits} bits"
        )

    circ, counts = simulate_modular_exp(
        n_bits=args.n_bits,
        base=args.base,
        exp_value=args.exp,
        modulus=args.modulus,
        shots=args.shots,
    )

    expected = pow(args.base, args.exp, args.modulus)
    print("Circuit:")
    print(circ.draw(fold=-1))
    print()
    print("Result register sample counts (little-endian bitstrings):")
    for bitstring, freq in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {bitstring}: {freq}")
    print(
        f"Expected result: {expected} (i.e., {args.base}^{args.exp} mod {args.modulus})"
    )


if __name__ == "__main__":
    main()
