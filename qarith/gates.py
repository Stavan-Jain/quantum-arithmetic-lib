from qiskit import QuantumCircuit


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
        elif name in ("CMP_CONST", "SUB_CONST_IF", "CMUL_CONST_MOD", "COMMENT"):
            # keep as barriers/labels so you can still see structure
            self.qc.barrier(*[q for q in qs if q >= 0])
        else:
            raise ValueError(f"Unknown gate name: {name}")

    def pretty_print(self):
        print(self.qc.draw())  # text diagram; use .draw('mpl') in a notebook


# shorthands for primitive gates


def x(circ: Circuit, t: int):
    circ.add("X", t)


def cx(circ: Circuit, c: int, t: int):
    circ.add("CX", c, t)


def ccx(circ: Circuit, c1: int, c2: int, t: int):
    circ.add("CCX", c1, c2, t)
