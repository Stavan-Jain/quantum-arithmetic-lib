"""Demo for the |a⟩ < |b⟩ comparator circuit in ``qarith.comparator``."""

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

from qarith.comparator import apply_less_than


def _load_value(circ: QuantumCircuit, qubits: list[int], value: int) -> None:
    if value >= 1 << len(qubits):
        raise ValueError(f"value {value} does not fit in {len(qubits)} bits")
    for i in range(len(qubits)):
        if (value >> i) & 1:
            circ.x(qubits[i])


def simulate_comparator(
    n_bits: int, a_value: int, b_value: int, shots: int
) -> tuple[QuantumCircuit, Dict[str, int]]:
    total_qubits = (2 * n_bits) + n_bits + (n_bits + 1) + 1
    circ = QuantumCircuit(total_qubits)

    a_qubits = list(range(n_bits))
    b_qubits = list(range(n_bits, 2 * n_bits))
    work_qubits = list(range(2 * n_bits, 3 * n_bits))
    carry_qubits = list(range(3 * n_bits, 4 * n_bits + 1))
    flag_qubit = 4 * n_bits + 1

    _load_value(circ, a_qubits, a_value)
    _load_value(circ, b_qubits, b_value)

    apply_less_than(circ, a_qubits, b_qubits, work_qubits, carry_qubits, flag_qubit)

    state = Statevector.from_instruction(circ)
    counts = state.sample_counts(shots=shots, qargs=[flag_qubit])
    return circ, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a < b comparator circuit")
    parser.add_argument("--n-bits", type=int, default=3, help="register width")
    parser.add_argument("--a", type=int, default=2, help="value of a")
    parser.add_argument("--b", type=int, default=5, help="value of b")
    parser.add_argument("--shots", type=int, default=1024, help="statevector samples")
    args = parser.parse_args()

    circ, counts = simulate_comparator(
        n_bits=args.n_bits, a_value=args.a, b_value=args.b, shots=args.shots
    )

    expected = int(args.a < args.b)
    print("Circuit:")
    print(circ.draw(fold=-1))
    print()
    print("Flag qubit sample counts (lt flag):")
    for bitstring, freq in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {bitstring}: {freq}")
    print(f"Expected flag value: {expected}")


if __name__ == "__main__":
    main()
