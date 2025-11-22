from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

from qarith.types import AdderResources
from qarith.resources.helpers import getGateCount
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator

def inplace_ripple_adder_resources(n: int) -> AdderResources:
    """
    Resource estimate for the in-place ripple-carry adder.

    Model (per bit):
        - 6 Toffoli (ccx) gates
        - 2 CNOT (cx) gates

    Ancillas:
        - n + 1 carry ancillas total (c[0..n])

    Overall:
        - toffoli_count = 6n
        - cnot_count = 2n
        - ancilla_count = n + 1
    """
    toffoli_per_bit = 6
    cnot_per_bit = 2

    return AdderResources(
        n_bits=n,
        toffoli_count=toffoli_per_bit * n,
        cnot_count=cnot_per_bit * n,
        ancilla_count=n + 1,
    )

def inplace_ripple_adder(circ: QuantumCircuit, x, y, c):
    """
    In-place ripple-carry adder.

    Logical effect (assuming c[0] = 0 and c[1..n] = 0):
        |x⟩ |y⟩ |0...0⟩  ↦  |x⟩ |x + y (mod 2^n)⟩ |0...0⟩

    Args:
        circ: QuantumCircuit to which the adder gates are added.
        x:    sequence of n qubits for the first addend (LSB at index 0).
        y:    sequence of n qubits for the second addend; will be overwritten with the sum.
        c:    sequence of n+1 qubits for carries; c[0] is the global carry-in.
              All c[i] are assumed to start in |0⟩ and are returned to |0⟩.
    """
    n = len(x)
    assert len(y) == n, "x and y must have the same length"
    assert len(c) == n + 1, "carry register must have length n+1"

    # ---------- Forward pass: compute carries into c[1..n] ----------
    # For each bit i, compute:
    #   c[i+1] ^= (x[i] & y[i]) ⊕ (x[i] & c[i]) ⊕ (y[i] & c[i])
    for i in range(n):
        circ.ccx(x[i], y[i], c[i + 1])
        circ.ccx(x[i], c[i], c[i + 1])
        circ.ccx(y[i], c[i], c[i + 1])

    # ---------- Backward pass: uncompute carries, write sum into y ----------
    # Process bits from MSB down to LSB.
    for i in reversed(range(n)):
        # Uncompute c[i+1] (reverse the forward-pass carry logic).
        circ.ccx(y[i], c[i], c[i + 1])
        circ.ccx(x[i], c[i], c[i + 1])
        circ.ccx(x[i], y[i], c[i + 1])

        # Now write the sum bit into y[i]:
        #   y[i] ^= x[i]
        #   y[i] ^= c[i]
        circ.cx(x[i], y[i])
        circ.cx(c[i], y[i])

def inplace_ripple_adder_circuit(n: int) -> QuantumCircuit:
    """
    Build an n-bit in-place ripple-carry adder circuit.

    Registers:
        x (n): input addend, preserved
        y (n): input addend, overwritten with x + y (mod 2^n)
        c (n+1): carry ancillas, all returned to |0⟩

    Initial state should be:
        |x⟩_x |y⟩_y |0...0⟩_c
    Final state:
        |x⟩_x |x + y (mod 2^n)⟩_y |0...0⟩_c
    """
    x = QuantumRegister(n, "x")
    y = QuantumRegister(n, "y")
    c = QuantumRegister(n + 1, "c")  # c[0] is cin, c[1..n] are carries

    circ = QuantumCircuit(x, y, c)
    inplace_ripple_adder(circ, x, y, c)
    return circ

def run(n):
    circ = inplace_ripple_adder_circuit(n)
    print(circ)
    resources = inplace_ripple_adder_resources(n)
    print("\nResource estimate (model):", resources)

    # Count actual gates in the constructed circuit
    num_ccx, num_cx = getGateCount(circ)
    print(f"Actual gate counts in circuit: ccx={num_ccx}, cx={num_cx}")

if __name__ == "__main__":
    run(2)
