"""Test that ecc_add can be compiled to a circuit."""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qarith.expr import Var
from qarith.qtype import Mod
from tests.ecc import ecc_add


def test_ecc_add_compiles():
    """Test that ecc_add can be compiled to a circuit."""
    modulus = 11
    x1 = Var("x1", Mod(modulus))
    y1 = Var("y1", Mod(modulus))
    x2 = Var("x2", Mod(modulus))
    y2 = Var("y2", Mod(modulus))

    # Compile ecc_add
    x3_expr, y3_expr = ecc_add(x1, y1, x2, y2)

    print("Compiling x3 expression...")
    x3_circuit = x3_expr.circuit()
    print(f"x3 circuit: {x3_circuit.n_qubits} qubits")
    print(f"x3 allocation: {x3_circuit.allocation}")

    print("\nCompiling y3 expression...")
    y3_circuit = y3_expr.circuit()
    print(f"y3 circuit: {y3_circuit.n_qubits} qubits")
    print(f"y3 allocation: {y3_circuit.allocation}")

    print("\n✓ ecc_add compiled successfully!")


if __name__ == "__main__":
    test_ecc_add_compiles()
