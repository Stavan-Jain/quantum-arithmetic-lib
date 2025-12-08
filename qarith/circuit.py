from qiskit import QuantumCircuit
from typing import List
from dataclasses import dataclass


@dataclass(frozen=True)
class QubitAllocation:
    """Tracks qubit allocation for circuit compilation.

    Similar to cobble's QubitAllocation, this tracks:
    - data_qubits: Number of data qubits (starting from 0)
    - ancilla_qubits: List of ancilla qubit indices
    """

    data_qubits: int  # Number of data qubits (starting from 0)
    ancilla_qubits: List[int]  # List of ancilla qubit indices

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
            QubitAllocation(self.data_qubits, self.ancilla_qubits + new_ancillas),
            new_ancillas,
        )


class Circuit:
    """Wrapper around Qiskit QuantumCircuit that also exposes the underlying circuit.

    This allows direct access to Qiskit operations (e.g., from adders) while
    maintaining a clean abstraction for higher-level IR compilation.

    Now includes qubit allocation tracking similar to cobble.
    """

    def __init__(self, n_qubits: int, allocation: QubitAllocation | None = None):
        self.n_qubits = n_qubits
        self.qc = QuantumCircuit(n_qubits)

        # Initialize allocation if not provided
        if allocation is None:
            # Default: all qubits are data qubits
            self.allocation = QubitAllocation(data_qubits=n_qubits, ancilla_qubits=[])
        else:
            self.allocation = allocation

    def add(self, name: str, *qs: int, **params):
        if name == "X":
            self.qc.x(qs[0])
        elif name == "CX":
            self.qc.cx(qs[0], qs[1])
        elif name == "CCX":
            self.qc.ccx(qs[0], qs[1], qs[2])
        elif name == "CMP_CONST":
            # Compare register with constant N, set flag if register >= N
            # qs: [x_bits..., flag_qubit], params: N
            N = params.get("N")
            if N is None:
                raise ValueError("CMP_CONST requires N parameter")
            x_bits = list(qs[:-1])  # All qubits except the last one
            flag_qubit = qs[-1]
            self._compare_const(x_bits, N, flag_qubit)
        elif name == "SUB_CONST_IF":
            # Subtract constant N from register if flag is set
            # qs: [x_bits..., flag_qubit], params: N, carry_anc
            N = params.get("N")
            carry_anc = params.get("carry_anc", 0)
            if N is None:
                raise ValueError("SUB_CONST_IF requires N parameter")
            x_bits = list(qs[:-1])  # All qubits except the last one
            flag_qubit = qs[-1]
            self._sub_const_if_flag(x_bits, N, flag_qubit, carry_anc)
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

    def _compare_const(self, x_bits: List[int], N: int, flag_qubit: int):
        """
        Compare quantum register with constant N.
        Sets flag_qubit to 1 if register value >= N, else 0.

        For small values, we use a simple bit-by-bit comparison.
        """
        n = len(x_bits)
        max_value = (1 << n) - 1

        if N > max_value:
            # Register can never be >= N, so flag stays 0
            return

        if N == 0:
            # Always >= 0, so flag is always 1
            self.qc.x(flag_qubit)
            return

        # Placeholder: use a barrier to mark where comparison should happen
        # In a real implementation, this would be a proper quantum comparator
        self.qc.barrier(*x_bits, flag_qubit)

    def _sub_const_if_flag(
        self, x_bits: List[int], N: int, flag_qubit: int, carry_anc: int
    ):
        """
        Subtract constant N from register if flag_qubit is set.
        Uses controlled subtraction.
        """
        n = len(x_bits)

        # Two's complement subtraction: x - N = x + (~N + 1)
        complement = (1 << n) - N  # 2^n - N

        # Flip bits for complement (controlled by flag)
        for i, bit in enumerate(x_bits):
            if not ((complement >> i) & 1):
                # This bit should be 0 in complement, so we need to flip it
                self.qc.cx(flag_qubit, bit)

        # Add 1 (with carry-in controlled by flag)
        if n > 0:
            # Add 1 to LSB if flag is set
            self.qc.cx(flag_qubit, x_bits[0])

            # Propagate carry
            for i in range(1, n):
                # If previous bit was 1 and we're adding 1, carry propagates
                self.qc.ccx(flag_qubit, x_bits[i - 1], x_bits[i])

        # Restore the complement bits (uncompute)
        for i, bit in enumerate(x_bits):
            if not ((complement >> i) & 1):
                self.qc.cx(flag_qubit, bit)

    def __str__(self) -> str:
        return str(self.qc.draw())
