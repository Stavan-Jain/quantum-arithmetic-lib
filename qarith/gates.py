from qiskit import QuantumCircuit
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .register import QRegister


class Circuit:
    """Wrapper around Qiskit QuantumCircuit that also exposes the underlying circuit.

    This allows direct access to Qiskit operations (e.g., from adders) while
    maintaining a clean abstraction for higher-level IR compilation.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qc = QuantumCircuit(n_qubits)

    @property
    def quantum_circuit(self) -> QuantumCircuit:
        """Expose the underlying Qiskit QuantumCircuit for direct manipulation."""
        return self.qc

    def add(self, name: str, *qs: int, **params):
        if name == "X":
            self.qc.x(qs[0])
        elif name == "CX":
            self.qc.cx(qs[0], qs[1])
        elif name == "CCX":
            self.qc.ccx(qs[0], qs[1], qs[2])
        elif name in (
            "CMP_CONST",
            "SUB_CONST_IF",
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

    def pretty_print(self):
        print(self.qc.draw())  # text diagram; use .draw('mpl') in a notebook

    def measure(self, registers: List["QRegister"] = None):
        """
        Add measurement operations to the circuit.

        Args:
            registers: List of QRegister objects to measure. If None, measures all qubits.
        """
        if registers is None:
            # Measure all qubits (this automatically creates classical bits)
            self.qc.measure_all()
        else:
            # Measure specific registers
            # First, ensure we have enough classical bits
            total_bits_needed = sum(reg.n_bits for reg in registers)
            current_cbits = self.qc.num_clbits

            # Add classical bits if needed
            if total_bits_needed > current_cbits:
                from qiskit import ClassicalRegister

                cr = ClassicalRegister(total_bits_needed - current_cbits)
                self.qc.add_register(cr)

            # Now measure each register
            cbit_offset = 0
            for reg in registers:
                bits = reg.bits()
                for i, q in enumerate(bits):
                    c = cbit_offset + i
                    self.qc.measure(q, c)
                cbit_offset += len(bits)
