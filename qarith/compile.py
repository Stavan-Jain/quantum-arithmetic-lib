"""
Compiler for QMARITH: compiles AST statements to quantum circuits.

This module compiles the AST-based representation (from expr.py) into
quantum circuits using the gate-level operations.
"""

from typing import List
from .adders.inplace_adder import inplace_ripple_adder
from .gates import Circuit
from .prog import Program
from .register import QRegister
from .expr import TypedExpr, BinOp, Var, Lit
from .qtype import Mod, is_optional


def _bits(reg: QRegister) -> List[int]:
    """Get the qubit indices for a quantum register."""
    return reg.bits()


def _compare_const(circ: Circuit, x_bits: List[int], N: int, flag_qubit: int):
    """Compare a quantum register with a constant (for modular reduction)."""
    circ.add("CMP_CONST", *x_bits, flag_qubit, N=N)


def _sub_const_if_flag(
    circ: Circuit,
    x_bits: List[int],
    N: int,
    flag_qubit: int,
    carry_anc: int,
):
    """Subtract a constant from a register if a flag qubit is set (for modular reduction)."""
    circ.add("SUB_CONST_IF", *x_bits, flag_qubit, N=N)


def _compile_expr(
    expr: TypedExpr,
    context: Program,
    circ: Circuit,
    carry_anc: int,
    flag: int,
    target_reg: QRegister,
) -> None:
    """
    Compile a typed expression into circuit operations.

    For now, we only support in-place operations where the result
    is stored in target_reg. This handles the common case of
    assignments like `x = x + y`.
    """
    if isinstance(expr.expr, BinOp):
        _compile_binop(expr, context, circ, carry_anc, flag, target_reg)
    elif isinstance(expr.expr, Var):
        # Variable reference: copy source to target
        src_reg = context.lookup_var(expr.expr.name)
        _copy_register(circ, src_reg, target_reg)
    elif isinstance(expr.expr, Lit):
        # Literal: initialize target with constant
        _init_constant(circ, target_reg, expr.expr.value)
    else:
        raise NotImplementedError(f"Cannot compile expression: {expr.expr}")


def _compile_binop(
    expr: TypedExpr,
    context: Program,
    circ: Circuit,
    carry_anc: int,
    flag: int,
    target_reg: QRegister,
) -> None:
    """Compile a binary operation."""
    if not isinstance(expr.expr, BinOp):
        raise TypeError("Expected BinOp")

    op = expr.expr.op
    left = expr.expr.left
    right = expr.expr.right

    # For in-place operations like `x = x + y`, we assume:
    # - left is the target variable (Var)
    # - right is the source (Var or Lit)
    # - result goes to target_reg

    if isinstance(left, Var):
        # Get the source register
        src_reg = context.lookup_var(left.name)

        # Copy source to target first (for cases like `x = y + z`)
        if left.name != target_reg.name:
            _copy_register(circ, src_reg, target_reg)

        # Now handle the operation
        if isinstance(right, Var):
            # Binary op with two variables: x = x + y
            right_reg = context.lookup_var(right.name)
            if op == "+":
                _compile_mod_add(circ, target_reg, right_reg, carry_anc, flag)
            elif op == "-":
                _compile_mod_sub(circ, target_reg, right_reg, carry_anc, flag)
            elif op == "*":
                _compile_mod_mul(circ, target_reg, right_reg, carry_anc, flag)
            elif op == "**":
                _compile_mod_pow(circ, target_reg, right_reg, carry_anc, flag)
            elif op == "//":
                _compile_mod_div(circ, target_reg, right_reg, carry_anc, flag)
            else:
                raise NotImplementedError(f"Binary operation {op} not yet implemented")
        elif isinstance(right, Lit):
            # Binary op with constant: x = x + 5
            if op == "+":
                _compile_add_const(circ, target_reg, right.value, carry_anc, flag)
            elif op == "-":
                _compile_sub_const(circ, target_reg, right.value, carry_anc, flag)
            elif op == "*":
                _compile_mul_const(circ, target_reg, right.value, carry_anc, flag)
            elif op == "**":
                _compile_pow_const(circ, target_reg, right.value, carry_anc, flag)
            elif op == "//":
                _compile_div_const(circ, target_reg, right.value, carry_anc, flag)
            else:
                raise NotImplementedError(
                    f"Binary operation {op} with constant not yet implemented"
                )
        else:
            raise NotImplementedError(f"Cannot compile right operand: {right}")
    else:
        raise NotImplementedError(f"Cannot compile left operand: {left}")


def _copy_register(circ: Circuit, src: QRegister, dst: QRegister):
    """Copy a quantum register to another (using CNOT gates)."""
    if src.n_bits != dst.n_bits:
        raise ValueError(
            f"Cannot copy register: bit width mismatch ({src.n_bits} vs {dst.n_bits})"
        )
    src_bits = _bits(src)
    dst_bits = _bits(dst)
    qc = circ.quantum_circuit
    for s, d in zip(src_bits, dst_bits):
        qc.cx(s, d)


def _init_constant(circ: Circuit, reg: QRegister, value: int):
    """Initialize a register with a constant value."""
    bits = _bits(reg)
    qc = circ.quantum_circuit
    # Set bits to represent the value
    for i, bit in enumerate(bits):
        if (value >> i) & 1:
            qc.x(bit)


def _compile_add_const(
    circ: Circuit, reg: QRegister, const: int, carry_anc: int, flag: int
):
    """Add a constant to a register."""
    # For now, use a simplified approach: add constant using the adder
    # In a full implementation, we'd optimize constant addition
    bits = _bits(reg)
    modulus = _get_modulus_from_type(reg.ty)

    # Create a temporary register for the constant
    # For simplicity, we'll use a different approach: add constant directly
    # This is a placeholder - full implementation would use optimized constant addition
    if modulus and const < modulus:
        # Use modular addition logic
        # For now, we'll use the symbolic gate
        _compare_const(circ, bits, modulus, flag)
        # Add constant using CNOT gates (simplified)
        for i, bit in enumerate(bits):
            if (const >> i) & 1:
                circ.add("X", bit)
    else:
        # Integer addition
        for i, bit in enumerate(bits):
            if (const >> i) & 1:
                circ.add("X", bit)


def _compile_sub_const(
    circ: Circuit, reg: QRegister, const: int, carry_anc: int, flag: int
):
    """Subtract a constant from a register."""
    # Similar to _compile_add_const, but for subtraction
    bits = _bits(reg)
    modulus = _get_modulus_from_type(reg.ty)

    if modulus:
        # Modular subtraction: subtract constant, handle underflow
        # This is simplified - full implementation would be more complex
        for i, bit in enumerate(bits):
            if (const >> i) & 1:
                circ.add("X", bit)
    else:
        # Integer subtraction
        for i, bit in enumerate(bits):
            if (const >> i) & 1:
                circ.add("X", bit)


def _get_modulus_from_type(ty) -> int | None:
    """Extract modulus from a type."""
    if isinstance(ty, Mod):
        return ty.N
    elif is_optional(ty) and isinstance(ty.inner, Mod):
        return ty.inner.N
    return None


def _compile_mod_mul(
    circ: Circuit, dst: QRegister, src: QRegister, carry_anc: int, flag: int
):
    """
    Compile modular multiplication: dst = dst * src (mod N).

    For now, uses a symbolic gate. A full implementation would use
    modular multiplication circuits.
    """
    dst_bits = _bits(dst)
    src_bits = _bits(src)
    modulus = _get_modulus_from_type(dst.ty)

    if modulus is None:
        raise ValueError("Cannot perform modular multiplication: no modulus in type")

    # Use symbolic gate for now
    circ.add("CMUL_MOD", *dst_bits, *src_bits, N=modulus)


def _compile_mod_pow(
    circ: Circuit, dst: QRegister, src: QRegister, carry_anc: int, flag: int
):
    """
    Compile modular exponentiation: dst = dst ** src (mod N).

    For now, uses a symbolic gate. A full implementation would use
    modular exponentiation circuits.
    """
    dst_bits = _bits(dst)
    src_bits = _bits(src)
    modulus = _get_modulus_from_type(dst.ty)

    if modulus is None:
        raise ValueError("Cannot perform modular exponentiation: no modulus in type")

    # Use symbolic gate for now
    circ.add("CPOW_MOD", *dst_bits, *src_bits, N=modulus)


def _compile_mod_div(
    circ: Circuit, dst: QRegister, src: QRegister, carry_anc: int, flag: int
):
    """
    Compile modular division: dst = dst // src (mod N).

    For now, uses a symbolic gate. A full implementation would use
    modular division (multiplication by inverse) circuits.
    """
    dst_bits = _bits(dst)
    src_bits = _bits(src)
    modulus = _get_modulus_from_type(dst.ty)

    if modulus is None:
        raise ValueError("Cannot perform modular division: no modulus in type")

    # Use symbolic gate for now
    circ.add("CDIV_MOD", *dst_bits, *src_bits, N=modulus)


def _compile_mul_const(
    circ: Circuit, reg: QRegister, const: int, carry_anc: int, flag: int
):
    """Multiply a register by a constant."""
    bits = _bits(reg)
    modulus = _get_modulus_from_type(reg.ty)

    if modulus:
        const = const % modulus
        # Use symbolic gate for now
        circ.add("CMUL_CONST_MOD", *bits, const=const, N=modulus)
    else:
        # Integer multiplication - use symbolic gate
        circ.add("CMUL_CONST", *bits, const=const)


def _compile_pow_const(
    circ: Circuit, reg: QRegister, const: int, carry_anc: int, flag: int
):
    """Raise a register to a constant power."""
    bits = _bits(reg)
    modulus = _get_modulus_from_type(reg.ty)

    if modulus:
        # Use symbolic gate for now
        circ.add("CPOW_CONST_MOD", *bits, const=const, N=modulus)
    else:
        # Integer exponentiation - use symbolic gate
        circ.add("CPOW_CONST", *bits, const=const)


def _compile_div_const(
    circ: Circuit, reg: QRegister, const: int, carry_anc: int, flag: int
):
    """Divide a register by a constant."""
    bits = _bits(reg)
    modulus = _get_modulus_from_type(reg.ty)

    if modulus:
        # Use symbolic gate for now (division = multiplication by inverse)
        circ.add("CDIV_CONST_MOD", *bits, const=const, N=modulus)
    else:
        # Integer division - use symbolic gate
        circ.add("CDIV_CONST", *bits, const=const)


def _compile_mod_add(
    circ: Circuit, dst: QRegister, src: QRegister, carry_anc: int, flag: int
):
    """
    Compile modular addition: dst = dst + src (mod N).

    Uses the in-place ripple-carry adder from qarith.adders.
    """
    dst_bits = _bits(dst)
    src_bits = _bits(src)
    n = len(dst_bits)
    modulus = _get_modulus_from_type(dst.ty)

    if modulus is None:
        raise ValueError("Cannot perform modular addition: no modulus in type")

    # Allocate carry ancillas
    max_carry_needed = n + 1
    if carry_anc + max_carry_needed > circ.n_qubits:
        raise ValueError(
            f"Insufficient ancilla qubits: need {max_carry_needed} carry qubits "
            f"starting from {carry_anc}, but circuit only has {circ.n_qubits} qubits"
        )
    carry_reg = list(range(carry_anc, carry_anc + n + 1))
    qc = circ.quantum_circuit
    inplace_ripple_adder(qc, dst_bits, src_bits, carry_reg)

    if modulus < (1 << n):
        _compare_const(circ, dst_bits, modulus, flag)
        _sub_const_if_flag(circ, dst_bits, modulus, flag, carry_anc)


def _compile_mod_sub(
    circ: Circuit, dst: QRegister, src: QRegister, carry_anc: int, flag: int
):
    """
    Compile modular subtraction: dst = dst - src (mod N).

    Uses two's complement subtraction with the in-place ripple-carry adder.
    """
    dst_bits = _bits(dst)
    src_bits = _bits(src)
    n = len(dst_bits)
    modulus = _get_modulus_from_type(dst.ty)

    if modulus is None:
        raise ValueError("Cannot perform modular subtraction: no modulus in type")

    # Allocate carry ancillas
    max_carry_needed = n + 1
    if carry_anc + max_carry_needed > circ.n_qubits:
        raise ValueError(
            f"Insufficient ancilla qubits: need {max_carry_needed} carry qubits "
            f"starting from {carry_anc}, but circuit only has {circ.n_qubits} qubits"
        )
    carry_reg = list(range(carry_anc, carry_anc + n + 1))

    qc = circ.quantum_circuit

    # Step 1: Negate src (bitwise NOT) for two's complement
    for bit in src_bits:
        qc.x(bit)

    # Step 2: Use in-place ripple adder with carry-in=1
    qc.x(carry_reg[0])
    inplace_ripple_adder(qc, dst_bits, src_bits, carry_reg)
    qc.x(carry_reg[0])  # Restore carry-in to 0

    # Step 3: Restore src
    for bit in src_bits:
        qc.x(bit)

    # Step 4: Modular reduction if needed
    if modulus < (1 << n):
        _compare_const(circ, dst_bits, modulus, flag)
        _sub_const_if_flag(circ, dst_bits, modulus, flag, carry_anc)


def compile_expr(
    expr: TypedExpr,
    target_var: str,
    prog: Program,
) -> Circuit:
    """
    Compile a typed expression to a quantum circuit.

    Args:
        expr: The typed expression to compile
        target_var: The target variable name (must be declared in prog)
        prog: The program context with variable declarations

    Returns:
        A quantum circuit implementing the expression
    """
    circ = Circuit(n_qubits=prog.n_qubits)
    carry_anc = prog.carry_ancilla_start
    flag_qubit = prog.flag_qubit_index

    # Get the target register
    target_reg = prog.lookup_var(target_var)
    # Compile the expression
    _compile_expr(expr, prog, circ, carry_anc, flag_qubit, target_reg)

    return circ


def _compile_program(prog: Program) -> Circuit:
    """Compile all operations in a program into a quantum circuit."""
    circ = Circuit(n_qubits=prog.n_qubits)
    carry_anc = prog.carry_ancilla_start
    flag_qubit = prog.flag_qubit_index

    for expr, target_var in prog._operations:
        # Get the target register
        target_reg = prog.lookup_var(target_var)
        # Compile the expression
        _compile_expr(expr, prog, circ, carry_anc, flag_qubit, target_reg)

    return circ
