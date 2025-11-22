from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

from qiskit import QuantumCircuit, QuantumRegister

from qarith.adders.base import Adder
from qarith.types import AdderResources

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

class NaiveRippleAdder(Adder):
    """Out-of-place ripple adder built from naive full-adder cells."""

    def build_circuit(self) -> QuantumCircuit:
        return naive_ripple_adder_circuit(self.n_bits)

    def resource_estimate(self) -> AdderResources:
        return naive_ripple_adder_resources(self.n_bits)


def run(n: int) -> None:
    adder = NaiveRippleAdder(n)
    profile = adder.characterize()
    print(profile.circuit)
    print("\nResource estimate (model):", profile.resource_estimate)
    print(
        "Actual gate counts in circuit: "
        f"ccx={profile.ccx_count}, cx={profile.cx_count}",
    )

if __name__ == "__main__":
    run(2)
