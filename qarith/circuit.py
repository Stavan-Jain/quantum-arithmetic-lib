from qiskit import QuantumCircuit
from typing import List
from dataclasses import dataclass


@dataclass(frozen=True)
class QubitAllocation:
    """Tracks qubit allocation for circuit compilation.
    This class tracks the allocation of data and ancilla qubits for a circuit.
    Args:
       data_qubits: Number of data qubits (starting from 0)
       ancilla_qubits: List of ancilla qubit indices
       expr_data_qubits: List of actual qubit indices used by the expression (optional)
                        If provided, these are the qubits that belong to the expression itself
    """

    data_qubits: int
    ancilla_qubits: List[int]
    expr_data_qubits: List[int] | None = None  # Actual qubit indices for the expression

    def total_qubits(self) -> int:
        """Total number of qubits needed."""
        if not self.ancilla_qubits:
            return self.data_qubits
        return max(self.data_qubits, max(self.ancilla_qubits) + 1)

    def allocate_ancillas(self, count: int) -> tuple["QubitAllocation", List[int]]:
        """Allocate new ancilla qubits and return updated allocation and new indices.

        Args:
            count: Number of ancilla qubits to allocate

        Returns:
            Tuple of (updated QubitAllocation, list of new ancilla qubit indices)
        """
        start = self.total_qubits()
        new_ancillas = list(range(start, start + count))
        return (
            QubitAllocation(
                self.data_qubits,
                self.ancilla_qubits + new_ancillas,
                self.expr_data_qubits,
            ),
            new_ancillas,
        )


class Circuit:
    """Wrapper around Qiskit QuantumCircuit that also exposes the underlying circuit.

    This allows direct access to Qiskit operations (e.g., from adders) while
    maintaining a clean abstraction for higher-level IR compilation.
    """

    def __init__(
        self,
        n_qubits: int,
        allocation: QubitAllocation | None = None,
        expr_data_qubits: List[int] | None = None,
    ):
        self.n_qubits = n_qubits
        self.qc = QuantumCircuit(n_qubits)

        # Initialize allocation if not provided
        if allocation is None:
            # Default: all qubits are data qubits
            self.allocation = QubitAllocation(
                data_qubits=n_qubits,
                ancilla_qubits=[],
                expr_data_qubits=expr_data_qubits,
            )
        else:
            # Update allocation with expr_data_qubits if provided
            if expr_data_qubits is not None:
                self.allocation = QubitAllocation(
                    data_qubits=allocation.data_qubits,
                    ancilla_qubits=allocation.ancilla_qubits,
                    expr_data_qubits=expr_data_qubits,
                )
            else:
                self.allocation = allocation

    def add(self, name: str, *qs: int, **params):
        if name == "X":
            self.qc.x(qs[0])
        elif name == "CX":
            self.qc.cx(qs[0], qs[1])
        elif name == "CCX":
            self.qc.ccx(qs[0], qs[1], qs[2])
        elif name in (
            "CMUL_CONST_MOD",
            "CMUL_MOD",
            "CPOW_MOD",
            "CDIV_MOD",
            "CMUL_CONST",
            "CPOW_CONST",
            "CDIV_CONST",
            "CPOW_CONST_MOD",
            "CDIV_CONST_MOD",
            "COMMENT",
        ):
            # keep as barriers/labels so you can still see structure
            self.qc.barrier(*[q for q in qs if q >= 0])
        else:
            raise ValueError(f"Unknown gate name: {name}")

    def __str__(self) -> str:
        return str(self.qc.draw())
