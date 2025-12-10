from typing import Dict, List
from dataclasses import dataclass, field
from .expr import (
    Var,
    Const,
    Add,
    Sub,
    Mul,
    Pow,
    Div,
    Sqrt,
    Inv,
    Log,
    Indexed,
    Join,
    Expr,
)
from .circuit import Circuit, QubitAllocation
from .qtype import Mod, Maybe, QType
from . import operators


@dataclass
class CompilationContext:
    """Tracks compiled expressions and their qubit assignments for CSE."""

    expr_to_qubits: Dict[Expr, List[int]] = field(default_factory=dict)
    expr_to_circuit: Dict[Expr, Circuit] = field(default_factory=dict)
    next_data_qubit: int = 0
    allocation: QubitAllocation = field(default_factory=lambda: QubitAllocation(0, []))

    def get_or_allocate_qubits(self, expr: Expr) -> List[int]:
        """Get qubit indices for an expression, allocating if necessary."""
        if expr in self.expr_to_qubits:
            return self.expr_to_qubits[expr]

        # Allocate new qubits tightly
        n_bits = expr.qtype().n_bits
        qubits = list(range(self.next_data_qubit, self.next_data_qubit + n_bits))
        self.expr_to_qubits[expr] = qubits
        self.next_data_qubit += n_bits

        # Update allocation
        max_qubit = max(qubits) if qubits else 0
        new_data_qubits = max(self.allocation.data_qubits, max_qubit + 1)
        self.allocation = QubitAllocation(
            data_qubits=new_data_qubits, ancilla_qubits=self.allocation.ancilla_qubits
        )

        return qubits

    def get_or_compile(self, expr: "Expr") -> tuple[list[int], "Circuit"]:
        """Get qubits and compiled circuit for an expression."""
        from .expr import Var, Const

        # Check if already compiled
        if expr in self.expr_to_circuit:
            qubits = self.expr_to_qubits[expr]
            return qubits, self.expr_to_circuit[expr]

        # For leaf nodes (Var, Const), allocate qubits first
        if isinstance(expr, (Var, Const)):
            qubits = self.get_or_allocate_qubits(expr)

        # Compile the expression
        circ = _compile_with_context(expr, self)

        # Get qubits (should have been set by compilation)
        qubits = self.expr_to_qubits.get(expr)
        if qubits is None:
            raise RuntimeError(
                f"Compilation of {expr} did not set qubit allocation. "
                f"This is a bug in the compiler."
            )

        self.expr_to_circuit[expr] = circ
        return qubits, circ


def _compile_with_context(expr: Expr, ctx: CompilationContext) -> Circuit:
    """Compile an expression using the given context."""
    if isinstance(expr, Var):
        return _compile_basic(expr, ctx)
    elif isinstance(expr, Const):
        return _compile_const(expr, ctx)
    elif isinstance(expr, Add):
        return _compile_add(expr, ctx)
    elif isinstance(expr, Sub):
        return _compile_sub(expr, ctx)
    elif isinstance(expr, Mul):
        return _compile_mul(expr, ctx)
    elif isinstance(expr, Pow):
        return _compile_pow(expr, ctx)
    elif isinstance(expr, Div):
        return _compile_floordiv(expr, ctx)
    elif isinstance(expr, Sqrt):
        return _compile_sqrt(expr, ctx)
    elif isinstance(expr, Inv):
        return _compile_inv(expr, ctx)
    elif isinstance(expr, Log):
        return _compile_log(expr, ctx)
    elif isinstance(expr, Indexed):
        return _compile_indexed(expr, ctx)
    elif isinstance(expr, Join):
        return _compile_join(expr, ctx)
    else:
        raise NotImplementedError(
            f"Cannot compile expression type: {type(expr).__name__}"
        )


def _compile_basic(expr: Var, ctx: CompilationContext) -> Circuit:
    qubits = ctx.get_or_allocate_qubits(expr)
    max_qubit = max(qubits) if qubits else -1
    allocation = QubitAllocation(
        data_qubits=max_qubit + 1, ancilla_qubits=[], expr_data_qubits=qubits
    )
    total_qubits = max_qubit + 1 if qubits else 0
    return Circuit(n_qubits=total_qubits, allocation=allocation)


def _compile_const(expr: Const, ctx: CompilationContext) -> Circuit:
    qubits = ctx.get_or_allocate_qubits(expr)
    max_qubit = max(qubits) if qubits else -1
    allocation = QubitAllocation(
        data_qubits=max_qubit + 1, ancilla_qubits=[], expr_data_qubits=qubits
    )
    total_qubits = max_qubit + 1 if qubits else 0
    circ = Circuit(n_qubits=total_qubits, allocation=allocation)
    _init_constant(circ, qubits, expr.value)
    return circ


def _compile_add(expr: Add, ctx: CompilationContext) -> Circuit:
    left_qubits, left_circ = ctx.get_or_compile(expr.left)
    right_qubits, right_circ = ctx.get_or_compile(expr.right)

    n_bits = expr.qtype().n_bits
    ty = expr.qtype()
    left_type = expr.left.qtype()
    right_type = expr.right.qtype()

    left_inner_qubits, left_error_flag = _extract_maybe_components(
        left_qubits, left_type
    )
    right_inner_qubits, right_error_flag = _extract_maybe_components(
        right_qubits, right_type
    )
    modulus, _ = _get_modulus_and_inner_n_bits(ty)

    # 1. Allocate Result Qubits Tightly Packed
    result_start = ctx.next_data_qubit
    result_qubits = list(range(result_start, result_start + n_bits))
    ctx.next_data_qubit += n_bits

    # 2. Determine Ancilla Placement
    current_max_data = max(
        max(left_qubits) if left_qubits else 0,
        max(right_qubits) if right_qubits else 0,
        max(result_qubits),
    )
    ancilla_start = current_max_data + 1

    left_anc_len = len(left_circ.allocation.ancilla_qubits)
    right_anc_len = len(right_circ.allocation.ancilla_qubits)
    max_child_ancillas = max(left_anc_len, right_anc_len)

    left_inner_set = set(left_inner_qubits)
    right_inner_set = set(right_inner_qubits)
    need_temp_space = bool(left_inner_set & right_inner_set) and (
        left_inner_qubits != right_inner_qubits
    )

    # 3. Alloc workspace (2 pads)
    work_start = ancilla_start + max_child_ancillas
    work_qubits = list(range(work_start, work_start + 2))

    # Temp space for overlapping inputs
    temp_space_len = n_bits if need_temp_space else 0
    temp_start = work_start + 2

    total_ancillas = max_child_ancillas + 2 + temp_space_len
    total_qubits = ancilla_start + total_ancillas

    merged_allocation = QubitAllocation(
        data_qubits=current_max_data + 1,
        ancilla_qubits=list(range(ancilla_start, ancilla_start + total_ancillas)),
        expr_data_qubits=result_qubits,
    )
    merged_circ = Circuit(n_qubits=total_qubits, allocation=merged_allocation)

    # Remap Children
    left_ancilla_target = list(range(ancilla_start, ancilla_start + left_anc_len))
    left_remapped = _remap_circuit(left_circ, left_ancilla_target, left_qubits)
    _compose_circuit(merged_circ, left_remapped)

    right_ancilla_target = list(range(ancilla_start, ancilla_start + right_anc_len))

    if need_temp_space:
        if left_inner_qubits == right_inner_qubits:
            temp_inner_qubits = left_inner_qubits
        else:
            temp_inner_qubits = list(
                range(temp_start, temp_start + len(left_inner_qubits))
            )
            right_remapped = _remap_circuit(
                right_circ, right_ancilla_target, right_qubits
            )
            _compose_circuit(merged_circ, right_remapped)
            _copy_qubits(merged_circ, right_inner_qubits, temp_inner_qubits)
            right_inner_qubits = temp_inner_qubits
    else:
        right_remapped = _remap_circuit(right_circ, right_ancilla_target, right_qubits)
        _compose_circuit(merged_circ, right_remapped)

    # Arithmetic
    from .expr import Const

    left_is_const = isinstance(expr.left, Const)
    right_is_const = isinstance(expr.right, Const)

    inner_n_bits = len(left_inner_qubits)
    result_inner_qubits = result_qubits[:inner_n_bits]
    result_error_flag_qubit = (
        result_qubits[inner_n_bits] if isinstance(ty, Maybe) else None
    )

    # FIXED: Pad the result to align with adder's expectations
    # [Pad0] + [Result (n)] + [Pad1]
    target_qubits = [work_qubits[0]] + result_inner_qubits + [work_qubits[1]]

    if left_is_const and right_is_const:
        res_val = (expr.left.value + expr.right.value) % modulus
        _init_constant(merged_circ, result_inner_qubits, res_val)
    elif left_is_const:
        operators.modular_add_quantum_classical(
            merged_circ.qc, right_inner_qubits, expr.left.value, modulus, target_qubits
        )
    elif right_is_const:
        operators.modular_add_quantum_classical(
            merged_circ.qc, left_inner_qubits, expr.right.value, modulus, target_qubits
        )
    else:
        operators.modular_add_quantum_quantum(
            merged_circ.qc,
            left_inner_qubits,
            right_inner_qubits,
            modulus,
            target_qubits,
        )

    if isinstance(ty, Maybe) and result_error_flag_qubit is not None:
        _combine_error_flags(
            merged_circ, left_error_flag, right_error_flag, result_error_flag_qubit
        )

    ctx.expr_to_qubits[expr] = result_qubits
    ctx.allocation = merged_allocation
    return merged_circ


def _compile_sub(expr: Sub, ctx: CompilationContext) -> Circuit:
    left_qubits, left_circ = ctx.get_or_compile(expr.left)
    right_qubits, right_circ = ctx.get_or_compile(expr.right)

    n_bits = expr.qtype().n_bits
    ty = expr.qtype()
    left_type = expr.left.qtype()
    right_type = expr.right.qtype()
    left_inner_qubits, left_error_flag = _extract_maybe_components(
        left_qubits, left_type
    )
    right_inner_qubits, right_error_flag = _extract_maybe_components(
        right_qubits, right_type
    )
    modulus, inner_n_bits = _get_modulus_and_inner_n_bits(ty)

    result_start = ctx.next_data_qubit
    result_qubits = list(range(result_start, result_start + n_bits))
    ctx.next_data_qubit += n_bits

    current_max_data = max(
        max(left_qubits) if left_qubits else 0,
        max(right_qubits) if right_qubits else 0,
        max(result_qubits),
    )
    ancilla_start = current_max_data + 1

    left_anc_len = len(left_circ.allocation.ancilla_qubits)
    right_anc_len = len(right_circ.allocation.ancilla_qubits)
    max_child_ancillas = max(left_anc_len, right_anc_len)

    left_inner_set = set(left_inner_qubits)
    right_inner_set = set(right_inner_qubits)
    need_temp_space = bool(left_inner_set & right_inner_set) and (
        left_inner_qubits != right_inner_qubits
    )

    work_start = ancilla_start + max_child_ancillas
    work_qubits = list(range(work_start, work_start + 2))

    temp_space_len = inner_n_bits if need_temp_space else 0
    temp_start = work_start + 2

    total_ancillas = max_child_ancillas + 2 + temp_space_len
    total_qubits = ancilla_start + total_ancillas

    merged_allocation = QubitAllocation(
        data_qubits=current_max_data + 1,
        ancilla_qubits=list(range(ancilla_start, ancilla_start + total_ancillas)),
        expr_data_qubits=result_qubits,
    )
    merged_circ = Circuit(n_qubits=total_qubits, allocation=merged_allocation)

    left_ancilla_target = list(range(ancilla_start, ancilla_start + left_anc_len))
    left_remapped = _remap_circuit(left_circ, left_ancilla_target, left_qubits)
    _compose_circuit(merged_circ, left_remapped)

    right_ancilla_target = list(range(ancilla_start, ancilla_start + right_anc_len))

    temp_inner_qubits: List[int] = []
    if need_temp_space:
        temp_inner_qubits = list(range(temp_start, temp_start + inner_n_bits))
        right_remapped = _remap_circuit(right_circ, right_ancilla_target, right_qubits)
        _compose_circuit(merged_circ, right_remapped)
        _copy_qubits(merged_circ, right_inner_qubits, temp_inner_qubits)
        right_inner_qubits = temp_inner_qubits
    else:
        right_remapped = _remap_circuit(right_circ, right_ancilla_target, right_qubits)
        _compose_circuit(merged_circ, right_remapped)

    from .expr import Const

    result_inner_qubits = result_qubits[:inner_n_bits]
    # FIXED: Pad result
    target_qubits = [work_qubits[0]] + result_inner_qubits + [work_qubits[1]]
    result_error_flag_qubit = (
        result_qubits[inner_n_bits] if isinstance(ty, Maybe) else None
    )

    left_is_const = isinstance(expr.left, Const)
    right_is_const = isinstance(expr.right, Const)

    if left_is_const and right_is_const:
        result_value = (expr.left.value - expr.right.value) % modulus
        _init_constant(merged_circ, result_inner_qubits, result_value)
    elif left_is_const:
        _init_constant(merged_circ, result_inner_qubits, expr.left.value)
        # Manual subtraction using Addin gates on target
        for i in range(inner_n_bits):
            tempk = 2 ** (inner_n_bits - (i + 1))
            neg_tempk = (modulus - tempk) % modulus
            addin = (
                operators.modular_addin_circuit(inner_n_bits, neg_tempk, modulus)
                .to_gate()
                .control(1)
            )
            addin.name = f"Addin(-{tempk}, N)"
            merged_circ.qc.append(
                addin, [right_inner_qubits[inner_n_bits - (i + 1)]] + target_qubits[:]
            )
    elif right_is_const:
        operators.modular_subtract_quantum_classical(
            merged_circ.qc, left_inner_qubits, expr.right.value, modulus, target_qubits
        )
    else:
        operators.modular_subtract_quantum_quantum(
            merged_circ.qc,
            left_inner_qubits,
            right_inner_qubits,
            modulus,
            target_qubits,
        )

    if isinstance(ty, Maybe) and result_error_flag_qubit is not None:
        _combine_error_flags(
            merged_circ, left_error_flag, right_error_flag, result_error_flag_qubit
        )

    ctx.expr_to_qubits[expr] = result_qubits
    ctx.allocation = merged_allocation
    return merged_circ


def _compile_mul(expr: Mul, ctx: CompilationContext) -> Circuit:
    left_qubits, left_circ = ctx.get_or_compile(expr.left)
    right_qubits, right_circ = ctx.get_or_compile(expr.right)

    n_bits = expr.qtype().n_bits
    ty = expr.qtype()
    left_type = expr.left.qtype()
    right_type = expr.right.qtype()
    left_inner_qubits, left_error_flag = _extract_maybe_components(
        left_qubits, left_type
    )
    right_inner_qubits, right_error_flag = _extract_maybe_components(
        right_qubits, right_type
    )
    modulus, inner_n_bits = _get_modulus_and_inner_n_bits(ty)

    result_start = ctx.next_data_qubit
    result_qubits = list(range(result_start, result_start + n_bits))
    ctx.next_data_qubit += n_bits

    current_max_data = max(
        max(left_qubits) if left_qubits else 0,
        max(right_qubits) if right_qubits else 0,
        max(result_qubits),
    )
    ancilla_start = current_max_data + 1

    left_anc_len = len(left_circ.allocation.ancilla_qubits)
    right_anc_len = len(right_circ.allocation.ancilla_qubits)
    max_child_ancillas = max(left_anc_len, right_anc_len)

    left_inner_set = set(left_inner_qubits)
    right_inner_set = set(right_inner_qubits)
    need_temp_space = bool(left_inner_set & right_inner_set) and (
        left_inner_qubits != right_inner_qubits
    )

    work_start = ancilla_start + max_child_ancillas
    work_qubits = list(range(work_start, work_start + 2))

    temp_space_len = inner_n_bits if need_temp_space else 0
    temp_start = work_start + 2

    total_ancillas = max_child_ancillas + 2 + temp_space_len
    total_qubits = ancilla_start + total_ancillas

    merged_allocation = QubitAllocation(
        data_qubits=current_max_data + 1,
        ancilla_qubits=list(range(ancilla_start, ancilla_start + total_ancillas)),
        expr_data_qubits=result_qubits,
    )
    merged_circ = Circuit(n_qubits=total_qubits, allocation=merged_allocation)

    # Remap
    left_ancilla_target = list(range(ancilla_start, ancilla_start + left_anc_len))
    left_remapped = _remap_circuit(left_circ, left_ancilla_target, left_qubits)
    _compose_circuit(merged_circ, left_remapped)

    right_ancilla_target = list(range(ancilla_start, ancilla_start + right_anc_len))

    if need_temp_space:
        if left_inner_qubits == right_inner_qubits:
            temp_inner_qubits = left_inner_qubits
        else:
            temp_inner_qubits = list(
                range(temp_start, temp_start + len(left_inner_qubits))
            )
            right_remapped = _remap_circuit(
                right_circ, right_ancilla_target, right_qubits
            )
            _compose_circuit(merged_circ, right_remapped)
            _copy_qubits(merged_circ, right_inner_qubits, temp_inner_qubits)
            right_inner_qubits = temp_inner_qubits
    else:
        right_remapped = _remap_circuit(right_circ, right_ancilla_target, right_qubits)
        _compose_circuit(merged_circ, right_remapped)

    from .expr import Const

    result_inner_qubits = result_qubits[:inner_n_bits]
    # FIXED: Pad result
    target_qubits = [work_qubits[0]] + result_inner_qubits + [work_qubits[1]]
    result_error_flag_qubit = (
        result_qubits[inner_n_bits] if isinstance(ty, Maybe) else None
    )

    left_is_const = isinstance(expr.left, Const)
    right_is_const = isinstance(expr.right, Const)

    if left_is_const and right_is_const:
        result_value = (expr.left.value * expr.right.value) % modulus
        _init_constant(merged_circ, result_inner_qubits, result_value)
    elif left_is_const:
        operators.modular_multiply_quantum_classical(
            merged_circ.qc, right_inner_qubits, expr.left.value, modulus, target_qubits
        )
    elif right_is_const:
        operators.modular_multiply_quantum_classical(
            merged_circ.qc, left_inner_qubits, expr.right.value, modulus, target_qubits
        )
    else:
        operators.modular_multiply_quantum_quantum(
            merged_circ.qc,
            left_inner_qubits,
            right_inner_qubits,
            modulus,
            target_qubits,
        )

    if isinstance(ty, Maybe) and result_error_flag_qubit is not None:
        _combine_error_flags(
            merged_circ, left_error_flag, right_error_flag, result_error_flag_qubit
        )

    ctx.expr_to_qubits[expr] = result_qubits
    ctx.allocation = merged_allocation
    return merged_circ


def _compile_pow(expr: Pow, ctx: CompilationContext) -> Circuit:
    from .expr import Const

    base_is_const = isinstance(expr.base, Const)
    exp_is_const = isinstance(expr.exponent, (int, Const))
    ty = expr.qtype()
    modulus, inner_n_bits = _get_modulus_and_inner_n_bits(ty)

    if isinstance(expr.exponent, int):
        exp_value = expr.exponent
    elif isinstance(expr.exponent, Const):
        exp_value = expr.exponent.value
    else:
        exp_value = None

    if isinstance(expr.base, Const):
        base_value = expr.base.value
    else:
        base_value = None

    if base_is_const and exp_is_const:
        result_qubits = ctx.get_or_allocate_qubits(expr)
        result_value = pow(base_value, exp_value, modulus)
        max_qubit = max(result_qubits) if result_qubits else -1
        allocation = QubitAllocation(
            data_qubits=max_qubit + 1 if result_qubits else 0,
            ancilla_qubits=[],
            expr_data_qubits=result_qubits,
        )
        total_qubits = max_qubit + 1 if result_qubits else 0
        circ = Circuit(n_qubits=total_qubits, allocation=allocation)
        _init_constant(circ, result_qubits, result_value)
        ctx.expr_to_qubits[expr] = result_qubits
        return circ

    if base_is_const and not exp_is_const:
        return _compile_pow_classical_base_quantum_exp(
            expr, ctx, base_value, modulus, inner_n_bits
        )

    if not base_is_const and exp_is_const:
        return _compile_pow_quantum_base_classical_exp(
            expr, ctx, exp_value, modulus, inner_n_bits
        )

    raise NotImplementedError("Quantum-quantum exponentiation not supported")


def _compile_pow_classical_base_quantum_exp(
    expr: Pow, ctx: CompilationContext, base_value: int, modulus: int, inner_n_bits: int
) -> Circuit:
    from math import gcd

    if gcd(base_value, modulus) != 1:
        raise ValueError("Base and modulus must be coprime")

    exp_qubits, exp_circ = ctx.get_or_compile(expr.exponent)
    exp_type = expr.exponent.qtype()
    exp_inner_qubits, exp_error_flag = _extract_maybe_components(exp_qubits, exp_type)
    n = len(exp_inner_qubits)
    ty = expr.qtype()
    n_bits = ty.n_bits
    from .operators import precompute_powers

    powers = precompute_powers(base_value, n, modulus)

    result_start = ctx.next_data_qubit
    result_qubits = list(range(result_start, result_start + n_bits))
    ctx.next_data_qubit += n_bits

    current_max_data = max(max(exp_qubits) if exp_qubits else 0, max(result_qubits))
    ancilla_start = current_max_data + 1

    max_exp_ancillas = len(exp_circ.allocation.ancilla_qubits)
    aux_start = ancilla_start + max_exp_ancillas
    aux_qubits = list(range(aux_start, aux_start + inner_n_bits + 2))

    total_ancillas = max_exp_ancillas + (inner_n_bits + 2)
    total_qubits = ancilla_start + total_ancillas

    merged_allocation = QubitAllocation(
        data_qubits=current_max_data + 1,
        ancilla_qubits=list(range(ancilla_start, ancilla_start + total_ancillas)),
        expr_data_qubits=result_qubits,
    )
    merged_circ = Circuit(n_qubits=total_qubits, allocation=merged_allocation)

    exp_ancilla_target = list(range(ancilla_start, ancilla_start + max_exp_ancillas))
    exp_remapped = _remap_circuit(exp_circ, exp_ancilla_target, exp_qubits)
    _compose_circuit(merged_circ, exp_remapped)

    result_inner_qubits = result_qubits[:inner_n_bits]
    result_error_flag_qubit = (
        result_qubits[inner_n_bits] if isinstance(ty, Maybe) else None
    )

    if inner_n_bits > 0:
        merged_circ.qc.x(result_inner_qubits[0])

    from .operators import _apply_controlled_inplace_multiply

    for i in range(n):
        power = powers[i]
        _apply_controlled_inplace_multiply(
            merged_circ.qc,
            ctrl_qubit=exp_inner_qubits[i],
            target_qubits=result_inner_qubits,
            k=power,
            N=modulus,
            aux_qubits=aux_qubits,
        )

    if isinstance(ty, Maybe) and result_error_flag_qubit is not None:
        if exp_error_flag is not None:
            merged_circ.qc.cx(exp_error_flag, result_error_flag_qubit)

    ctx.expr_to_qubits[expr] = result_qubits
    ctx.allocation = merged_allocation
    return merged_circ


def _compile_pow_quantum_base_classical_exp(
    expr: "Pow",
    ctx: "CompilationContext",
    exp_value: int,
    modulus: int,
    inner_n_bits: int,
) -> "Circuit":
    base_qubits, base_circ = ctx.get_or_compile(expr.base)
    base_type = expr.base.qtype()
    base_inner_qubits, base_error_flag = _extract_maybe_components(
        base_qubits, base_type
    )
    ty = expr.qtype()
    n_bits = ty.n_bits

    if exp_value == 0:
        result_qubits = ctx.get_or_allocate_qubits(expr)
        allocation = QubitAllocation(
            data_qubits=ctx.next_data_qubit,
            ancilla_qubits=[],
            expr_data_qubits=result_qubits,
        )
        circ = Circuit(n_qubits=ctx.next_data_qubit, allocation=allocation)
        _init_constant(circ, result_qubits[:inner_n_bits], 1)
        ctx.expr_to_qubits[expr] = result_qubits
        return circ

    if exp_value == 1:
        result_qubits = ctx.get_or_allocate_qubits(expr)

        current_max_data = max(max(base_qubits), max(result_qubits))
        ancilla_start = current_max_data + 1
        anc_len = len(base_circ.allocation.ancilla_qubits)

        allocation = QubitAllocation(
            data_qubits=current_max_data + 1,
            ancilla_qubits=list(range(ancilla_start, ancilla_start + anc_len)),
            expr_data_qubits=result_qubits,
        )
        circ = Circuit(n_qubits=ancilla_start + anc_len, allocation=allocation)

        anc_target = list(range(ancilla_start, ancilla_start + anc_len))
        base_remapped = _remap_circuit(base_circ, anc_target, base_qubits)
        _compose_circuit(circ, base_remapped)
        _copy_qubits(circ, base_inner_qubits, result_qubits[:inner_n_bits])

        ctx.expr_to_qubits[expr] = result_qubits
        return circ

    max_base_ancillas = len(base_circ.allocation.ancilla_qubits)

    power_regs = []
    for i in range(exp_value):
        if i == exp_value - 1:
            reg = list(range(ctx.next_data_qubit, ctx.next_data_qubit + n_bits))
            ctx.next_data_qubit += n_bits
            ctx.expr_to_qubits[expr] = reg
            power_regs.append(reg[:inner_n_bits])
            result_qubits = reg
        else:
            reg = list(range(ctx.next_data_qubit, ctx.next_data_qubit + inner_n_bits))
            ctx.next_data_qubit += inner_n_bits
            power_regs.append(reg)

    current_max_data = ctx.next_data_qubit - 1
    ancilla_start = current_max_data + 1

    aux_start = ancilla_start + max_base_ancillas
    aux_qubits = list(range(aux_start, aux_start + inner_n_bits + 2))
    aux_result = aux_qubits[1 : inner_n_bits + 1]

    total_ancillas = max_base_ancillas + (inner_n_bits + 2)
    total_qubits = ancilla_start + total_ancillas

    merged_allocation = QubitAllocation(
        data_qubits=current_max_data + 1,
        ancilla_qubits=list(range(ancilla_start, ancilla_start + total_ancillas)),
        expr_data_qubits=result_qubits,
    )
    merged_circ = Circuit(n_qubits=total_qubits, allocation=merged_allocation)

    anc_target = list(range(ancilla_start, ancilla_start + max_base_ancillas))
    base_remapped = _remap_circuit(base_circ, anc_target, base_qubits)
    _compose_circuit(merged_circ, base_remapped)

    _copy_qubits(merged_circ, base_inner_qubits, power_regs[0])

    for i in range(1, exp_value):
        operators.modular_multiply_quantum_quantum(
            merged_circ.qc,
            power_regs[i - 1],
            base_inner_qubits,
            modulus,
            aux_qubits,
        )
        _copy_qubits(merged_circ, aux_result, power_regs[i])
        _copy_qubits(merged_circ, power_regs[i], aux_result)

    result_inner_qubits = result_qubits[:inner_n_bits]
    result_error_flag_qubit = (
        result_qubits[inner_n_bits] if isinstance(ty, Maybe) else None
    )

    if isinstance(ty, Maybe) and result_error_flag_qubit is not None:
        if base_error_flag is not None:
            merged_circ.qc.cx(base_error_flag, result_error_flag_qubit)

    ctx.allocation = merged_allocation
    return merged_circ


def _compile_inv(expr: Inv, ctx: CompilationContext) -> Circuit:
    operand_qubits, operand_circ = ctx.get_or_compile(expr.expr)
    n_bits = expr.qtype().n_bits
    ty = expr.qtype()
    operand_type = expr.expr.qtype()
    operand_inner_qubits, operand_error_flag = _extract_maybe_components(
        operand_qubits, operand_type
    )
    inner_n_bits = len(operand_inner_qubits)

    result_start = ctx.next_data_qubit
    result_qubits = list(range(result_start, result_start + n_bits))
    ctx.next_data_qubit += n_bits

    current_max_data = max(
        max(operand_qubits) if operand_qubits else 0, max(result_qubits)
    )
    ancilla_start = current_max_data + 1

    max_ancillas = len(operand_circ.allocation.ancilla_qubits)
    total_qubits = ancilla_start + max_ancillas

    merged_allocation = QubitAllocation(
        data_qubits=current_max_data + 1,
        ancilla_qubits=list(range(ancilla_start, ancilla_start + max_ancillas)),
        expr_data_qubits=result_qubits,
    )
    merged_circ = Circuit(n_qubits=total_qubits, allocation=merged_allocation)

    anc_target = list(range(ancilla_start, ancilla_start + max_ancillas))
    operand_remapped = _remap_circuit(operand_circ, anc_target, operand_qubits)
    _compose_circuit(merged_circ, operand_remapped)

    result_inner_qubits = result_qubits[:inner_n_bits]
    _copy_qubits(merged_circ, operand_inner_qubits, result_inner_qubits)

    result_error_flag_qubit = (
        result_qubits[inner_n_bits] if isinstance(ty, Maybe) else None
    )
    if isinstance(ty, Maybe) and result_error_flag_qubit is not None:
        if operand_error_flag is not None:
            merged_circ.qc.cx(operand_error_flag, result_error_flag_qubit)

    ctx.expr_to_qubits[expr] = result_qubits
    ctx.allocation = merged_allocation
    return merged_circ


def _compile_floordiv(expr: Div, ctx: CompilationContext) -> Circuit:
    qubits = ctx.get_or_allocate_qubits(expr)
    n_bits = expr.qtype().n_bits
    allocation = QubitAllocation(
        data_qubits=n_bits, ancilla_qubits=[], expr_data_qubits=qubits
    )
    return Circuit(n_qubits=n_bits, allocation=allocation)


def _compile_sqrt(expr: Sqrt, ctx: CompilationContext) -> Circuit:
    qubits = ctx.get_or_allocate_qubits(expr)
    n_bits = expr.qtype().n_bits
    allocation = QubitAllocation(
        data_qubits=n_bits, ancilla_qubits=[], expr_data_qubits=qubits
    )
    return Circuit(n_qubits=n_bits, allocation=allocation)


def _compile_log(expr: Log, ctx: CompilationContext) -> Circuit:
    qubits = ctx.get_or_allocate_qubits(expr)
    n_bits = expr.qtype().n_bits
    allocation = QubitAllocation(
        data_qubits=n_bits, ancilla_qubits=[], expr_data_qubits=qubits
    )
    return Circuit(n_qubits=n_bits, allocation=allocation)


def _compile_indexed(expr: Indexed, ctx: CompilationContext) -> Circuit:
    qubits = ctx.get_or_allocate_qubits(expr)
    n_bits = expr.qtype().n_bits
    allocation = QubitAllocation(
        data_qubits=n_bits, ancilla_qubits=[], expr_data_qubits=qubits
    )
    return Circuit(n_qubits=n_bits, allocation=allocation)


def _compile_join(expr: Join, ctx: CompilationContext) -> Circuit:
    qubits = ctx.get_or_allocate_qubits(expr)
    n_bits = expr.qtype().n_bits
    allocation = QubitAllocation(
        data_qubits=n_bits, ancilla_qubits=[], expr_data_qubits=qubits
    )
    return Circuit(n_qubits=n_bits, allocation=allocation)


def _combine_error_flags(
    circ: Circuit,
    left_error_flag: int | None,
    right_error_flag: int | None,
    result_error_flag: int,
):
    if left_error_flag is not None and right_error_flag is not None:
        circ.qc.cx(left_error_flag, result_error_flag)
        circ.qc.cx(right_error_flag, result_error_flag)
    elif left_error_flag is not None:
        circ.qc.cx(left_error_flag, result_error_flag)
    elif right_error_flag is not None:
        circ.qc.cx(right_error_flag, result_error_flag)


def _extract_maybe_components(
    qubits: List[int], qtype: QType
) -> tuple[List[int], int | None]:
    if isinstance(qtype, Mod):
        return qubits, None
    elif isinstance(qtype, Maybe):
        inner_qubits = qubits[:-1]
        error_flag = qubits[-1]
        inner_inner, _ = _extract_maybe_components(inner_qubits, qtype.inner_type)
        return inner_inner, error_flag
    else:
        raise ValueError(f"Cannot extract Maybe components from type {qtype}")


def _unwrap_maybe_type(ty: QType) -> tuple[Mod, int]:
    if isinstance(ty, Mod):
        return ty, 0
    elif isinstance(ty, Maybe):
        inner_mod, depth = _unwrap_maybe_type(ty.inner_type)
        return inner_mod, depth + 1
    else:
        raise ValueError(f"Cannot unwrap type {ty}")


def _get_modulus_and_inner_n_bits(ty: QType) -> tuple[int, int]:
    inner_mod, _ = _unwrap_maybe_type(ty)
    return inner_mod.N, inner_mod.n_bits


# def _remap_circuit(
#     circ: Circuit, ancilla_target: List[int], data_qubits: List[int]
# ) -> Circuit:
#     qubit_map: Dict[int, int] = {}
#     used_qubits = set()
#     for instruction in circ.qc.data:
#         for q in instruction.qubits:
#             used_qubits.add(circ.qc.find_bit(q).index)

#     if circ.allocation.expr_data_qubits is not None:
#         expr_data_qubits = circ.allocation.expr_data_qubits
#         if len(expr_data_qubits) != len(data_qubits):
#             raise ValueError(
#                 f"Expression uses {len(expr_data_qubits)} data qubits but {len(data_qubits)} target qubits provided."
#             )
#         for old_q, new_q in zip(expr_data_qubits, data_qubits):
#             qubit_map[old_q] = new_q
#     else:
#         num_data = min(circ.allocation.data_qubits, len(data_qubits))
#         for i in range(num_data):
#             qubit_map[i] = data_qubits[i]

#     ancilla_qubits_used = sorted(circ.allocation.ancilla_qubits)
#     if len(ancilla_qubits_used) > len(ancilla_target):
#         raise ValueError(
#             f"Circuit uses {len(ancilla_qubits_used)} ancilla qubits but only {len(ancilla_target)} target qubits provided."
#         )
#     for old_ancilla, new_ancilla in zip(ancilla_qubits_used, ancilla_target):
#         qubit_map[old_ancilla] = new_ancilla

#     unmapped_used = used_qubits - set(qubit_map.keys())
#     if unmapped_used:
#         all_target_qubits = data_qubits + ancilla_target
#         next_free_qubit = max(all_target_qubits) + 1 if all_target_qubits else 0
#         for old_q in sorted(unmapped_used):
#             qubit_map[old_q] = next_free_qubit
#             next_free_qubit += 1

#     all_target_qubits = data_qubits + ancilla_target
#     max_qubit = max(all_target_qubits) if all_target_qubits else 0
#     if qubit_map:
#         max_qubit = max(max_qubit, max(qubit_map.values()))
#     new_n_qubits = max_qubit + 1

#     new_allocation = QubitAllocation(
#         data_qubits=max(data_qubits) + 1 if data_qubits else 0,
#         ancilla_qubits=ancilla_target,
#         expr_data_qubits=data_qubits,
#     )
#     new_circ = Circuit(n_qubits=new_n_qubits, allocation=new_allocation)

#     for instruction in circ.qc.data:
#         remapped_qubits = []
#         for q in instruction.qubits:
#             old_index = circ.qc.find_bit(q).index
#             if old_index not in qubit_map:
#                 raise ValueError(f"Qubit {old_index} not in qubit_map.")
#             new_index = qubit_map[old_index]
#             remapped_qubits.append(new_index)
#         new_circ.qc.append(instruction.operation, remapped_qubits, instruction.clbits)

#     return new_circ


def _remap_circuit(
    circ: Circuit, ancilla_target: List[int], data_qubits: List[int]
) -> Circuit:
    qubit_map: Dict[int, int] = {}

    # 1. Map ALL Global Data Qubits 1:1 (Identity)
    # Since data qubits are allocated globally, they stay where they are.
    # We map everything from 0 up to the circuit's data_qubits boundary.
    for i in range(circ.allocation.data_qubits):
        qubit_map[i] = i

    # 2. Validation (Optional but good for debugging)
    # Ensure the target 'data_qubits' (result of expr) matches the identity map we just made
    if circ.allocation.expr_data_qubits is not None:
        expr_data = circ.allocation.expr_data_qubits
        if len(expr_data) != len(data_qubits):
            raise ValueError("Mismatch in result qubit counts")
        for old, new in zip(expr_data, data_qubits):
            if old != new:
                # This should theoretically never happen if Context works correctly
                raise ValueError(
                    f"CRITICAL: Context mismatch. Data qubit {old} trying to move to {new}"
                )

    # 3. Map Ancilla Qubits to the new Target locations
    ancilla_qubits_used = sorted(circ.allocation.ancilla_qubits)
    if len(ancilla_qubits_used) > len(ancilla_target):
        raise ValueError(
            f"Circuit uses {len(ancilla_qubits_used)} ancilla qubits but only {len(ancilla_target)} provided."
        )
    for old_ancilla, new_ancilla in zip(ancilla_qubits_used, ancilla_target):
        qubit_map[old_ancilla] = new_ancilla

    # 4. Handle "Unmapped" (True internal scratch qubits that aren't inputs or ancillas)
    # Calculate used qubits dynamically
    used_qubits = set()
    for instruction in circ.qc.data:
        for q in instruction.qubits:
            used_qubits.add(circ.qc.find_bit(q).index)

    unmapped_used = used_qubits - set(qubit_map.keys())

    if unmapped_used:
        # These need to be moved to safe space after all known data/ancillas
        all_target_qubits = list(range(circ.allocation.data_qubits)) + ancilla_target
        next_free_qubit = max(all_target_qubits) + 1 if all_target_qubits else 0
        for old_q in sorted(unmapped_used):
            qubit_map[old_q] = next_free_qubit
            next_free_qubit += 1

    # 5. Construct New Circuit
    # Determine size
    max_mapped = max(qubit_map.values()) if qubit_map else 0
    new_n_qubits = max_mapped + 1

    new_allocation = QubitAllocation(
        data_qubits=circ.allocation.data_qubits,  # Data count stays same
        ancilla_qubits=ancilla_target,
        expr_data_qubits=data_qubits,
    )
    new_circ = Circuit(n_qubits=new_n_qubits, allocation=new_allocation)

    # 6. Copy Gates
    for instruction in circ.qc.data:
        remapped_qubits = []
        for q in instruction.qubits:
            old_index = circ.qc.find_bit(q).index
            if old_index not in qubit_map:
                # This catches gates applied to qubits we didn't expect
                raise ValueError(f"Qubit {old_index} used in gate but not mapped.")
            new_index = qubit_map[old_index]
            remapped_qubits.append(new_index)
        new_circ.qc.append(instruction.operation, remapped_qubits, instruction.clbits)

    return new_circ


def _compose_circuit(target: Circuit, source: Circuit):
    needed_qubits = max(
        target.n_qubits,
        max(
            (
                source.qc.find_bit(q).index
                for inst in source.qc.data
                for q in inst.qubits
            ),
            default=0,
        )
        + 1,
    )
    if needed_qubits > target.n_qubits:
        from qiskit import QuantumCircuit

        new_qc = QuantumCircuit(needed_qubits)
        for inst in target.qc.data:
            new_qc.append(inst.operation, inst.qubits, inst.clbits)
        target.qc = new_qc
        target.n_qubits = needed_qubits

    for instruction in source.qc.data:
        mapped_qubits = [
            target.qc.qubits[source.qc.find_bit(q).index] for q in instruction.qubits
        ]
        target.qc.append(instruction.operation, mapped_qubits, instruction.clbits)


def _copy_qubits(circ: Circuit, src_qubits: List[int], dst_qubits: List[int]):
    """Copy qubit values from source to destination using CNOT."""
    assert len(src_qubits) == len(dst_qubits), (
        "Source and destination must have same length"
    )
    for src, dst in zip(src_qubits, dst_qubits):
        circ.qc.cx(src, dst)


def _init_constant(circ: Circuit, qubits: List[int], value: int):
    for i, bit in enumerate(qubits):
        if (value >> i) & 1:
            circ.add("X", bit)
