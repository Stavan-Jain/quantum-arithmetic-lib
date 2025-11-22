from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
        
from dataclasses import dataclass
from qarith.types import AdderResources
from qarith.resources.helpers import getGateCount
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator

def naive_ripple_adder_resources(n: int) -> AdderResources:
    """Return a rough resource estimate for the naive ripple-carry adder.

    Model:
      - Each bit uses ~3 Toffolis and ~3 CNOTs in its full-adder cell.
      - We allocate one carry ancilla per bit (in the `cout` register).
      - We add one ancilla bit for cin.
    """
    toffoli_per_bit = 3
    cnot_per_bit = 3
    ancilla_per_bit = 1  # one carry ancilla in `cout`

    return AdderResources(
        n_bits=n,
        toffoli_count=toffoli_per_bit * n,
        cnot_count=cnot_per_bit * n,
        ancilla_count=ancilla_per_bit * n + 1,
    )

def naive_full_adder_cell(
    circ: QuantumCircuit,
    x_qubit,
    y_qubit,
    cin_qubit,
    cout_qubit,
    s_qubit,
):
    """One-bit reversible full-adder cell.

    Conceptual behavior:
      - Inputs: x, y, cin, s=0, cout=0
      - Outputs:
          s    <- x ⊕ y ⊕ cin
          cout <- (x & y) ⊕ (x & cin) ⊕ (y & cin)

    We keep x, y, and cin unchanged. This is *not* an in-place adder;
    the sum is stored in a separate register `s`, and the carry is stored
    in `cout`.
    """
    # Sum bit into s: s = x ⊕ y ⊕ cin
    circ.cx(x_qubit, s_qubit)
    circ.cx(y_qubit, s_qubit)
    circ.cx(cin_qubit, s_qubit)

    # Carry-out into cout:
    # cout = (x & y) ⊕ (x & cin) ⊕ (y & cin)
    circ.ccx(x_qubit, y_qubit, cout_qubit)
    circ.ccx(x_qubit, cin_qubit, cout_qubit)
    circ.ccx(y_qubit, cin_qubit, cout_qubit)

def naive_ripple_adder_circuit(n: int) -> QuantumCircuit:
    """Construct a naive n-bit ripple-carry adder circuit (out-of-place).

    Registers (all size n except `cin`):
      - x:   input bits of the first addend
      - y:   input bits of the second addend (unchanged here)
      - cin: 1-qubit global carry-in (assumed initialized to |0>)
      - cout: n-qubit carry register (one ancilla per bit)
      - s:   n-qubit sum register (assumed initialized to |0...0>)

    Conceptual semantics:
      |x>|y>|0>_cin|0...0>_cout|0...0>_s
        → |x>|y>|0>_cin|carries>_cout|x+y>_s  (mod 2^n)

    This function does *not* perform any uncomputation of carries and
    is not optimized. It is intended as a first, transparent prototype
    of a ripple-carry quantum adder.
    """
    # Define registers
    x = QuantumRegister(n, "x")
    y = QuantumRegister(n, "y")
    cin = QuantumRegister(1, "cin")
    cout = QuantumRegister(n, "cout")
    s = QuantumRegister(n, "s")

    circ = QuantumCircuit(x, y, cin, cout, s)

    # For bit i:
    #   carry-in is cin[0] for i=0, otherwise cout[i-1]
    #   carry-out is cout[i]
    #   sum is written into s[i]
    for i in range(n):
        cin_qubit = cin[0] if i == 0 else cout[i - 1]
        cout_qubit = cout[i]
        s_qubit = s[i]

        naive_full_adder_cell(
            circ,
            x_qubit=x[i],
            y_qubit=y[i],
            cin_qubit=cin_qubit,
            cout_qubit=cout_qubit,
            s_qubit=s_qubit,
        )

    return circ

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
    circ = naive_ripple_adder_circuit(n)
    print(circ)
    resources = naive_ripple_adder_resources(n)
    print("\nResource estimate (model):", resources)

    # Count actual gates in the constructed circuit
    num_ccx, num_cx = getGateCount(circ)
    print(f"Actual gate counts in circuit: ccx={num_ccx}, cx={num_cx}")

if __name__ == "__main__":
    run(2)