from qiskit import QuantumCircuit
from qiskit import QuantumRegister
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

        # Create QuantumCircuit with named QuantumRegisters for better visualization
        # Group qubits into logical registers: data, ancilla, result
        data_qubits_count = self.allocation.data_qubits
        ancilla_qubits_count = (
            len(self.allocation.ancilla_qubits) if self.allocation.ancilla_qubits else 0
        )

        # Create registers with meaningful names
        registers = []
        if data_qubits_count > 0:
            data_reg = QuantumRegister(data_qubits_count, "data")
            registers.append(data_reg)

        if ancilla_qubits_count > 0:
            ancilla_reg = QuantumRegister(ancilla_qubits_count, "ancilla")
            registers.append(ancilla_reg)

        # If we have more qubits than data + ancilla, add a working register
        working_qubits = n_qubits - data_qubits_count - ancilla_qubits_count
        if working_qubits > 0:
            work_reg = QuantumRegister(working_qubits, "work")
            registers.append(work_reg)

        # Create circuit with named registers
        if registers:
            self.qc = QuantumCircuit(*registers)
        else:
            # Fallback if no registers (shouldn't happen, but be safe)
            self.qc = QuantumCircuit(n_qubits)

    def add(self, name: str, *qs: int, **params):
        if name == "X":
            self.qc.x(qs[0])
        elif name == "CX":
            self.qc.cx(qs[0], qs[1])
        elif name == "CCX":
            self.qc.ccx(qs[0], qs[1], qs[2])
        else:
            raise ValueError(f"Unknown gate name: {name}")

    def __str__(self) -> str:
        return str(self.qc.draw())

    def save_png(self, filename: str) -> None:
        self.qc.draw(output="mpl").savefig(filename, format="png")

    def decompose(self, reps: int = 1, **kwargs):
        """Decompose composite gates in the circuit.

        This method calls the decompose method of the underlying Qiskit QuantumCircuit
        to expand composite gates (like Addin, Multout) into their basic gate components.

        Args:
            reps: Number of times to decompose (default: 1)
            **kwargs: Additional arguments passed to Qiskit's decompose method

        Returns:
            A new Circuit instance with decomposed gates
        """
        decomposed_qc = self.qc.decompose(reps=reps, **kwargs)
        # Create a new Circuit instance with the decomposed circuit
        new_circ = Circuit(
            self.n_qubits, self.allocation, self.allocation.expr_data_qubits
        )
        new_circ.qc = decomposed_qc
        return new_circ
