# test.py
from qarith.compile import CompilationContext, _compile_with_context
from qarith.expr import Var
from qarith.qtype import Mod

x1 = Var("x1", Mod(11))
x2 = Var("x2", Mod(11))
y1 = Var("y1", Mod(11))
y2 = Var("y2", Mod(11))

compound = x1 + x2 * y1 - y2

ctx = CompilationContext()
circ = _compile_with_context(compound, ctx)

print("Expression to qubits mapping:")
for expr, qubits in ctx.expr_to_qubits.items():
    print(f"  {expr}: qubits {qubits}")

print(
    f"\nAllocation: data={circ.allocation.data_qubits}, ancilla={circ.allocation.ancilla_qubits}"
)
print(f"Total qubits in circuit: {circ.n_qubits}")
