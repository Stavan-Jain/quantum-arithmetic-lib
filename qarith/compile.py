"""
Compiler for QMARITH: compiles expressions to quantum circuits.

This module compiles expressions directly to quantum circuits without
requiring variable management. Parameters are provided as Basic expressions.

The compilation uses a context to track compiled expressions and ensure
common subexpressions share qubits, minimizing input qubit requirements.
"""

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
    """Tracks compiled expressions and their qubit assignments for CSE.

    This context ensures that common subexpressions are compiled only once
    and share the same qubit assignments, minimizing input qubit requirements.
    """

    expr_to_qubits: Dict[Expr, List[int]] = field(default_factory=dict)
    expr_to_circuit: Dict[Expr, Circuit] = field(default_factory=dict)
    next_data_qubit: int = 0
    allocation: QubitAllocation = field(default_factory=lambda: QubitAllocation(0, []))

    def get_or_allocate_qubits(self, expr: Expr) -> List[int]:
        """Get qubit indices for an expression, allocating if necessary.

        If the expression has been seen before, returns its existing qubit assignment.
        Otherwise, allocates new qubits and records the mapping.

        Args:
            expr: Expression to get qubits for

        Returns:
            List of qubit indices for this expression's data qubits
        """
        if expr in self.expr_to_qubits:
            return self.expr_to_qubits[expr]

        # Allocate new qubits
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

    # def get_or_compile(self, expr: Expr) -> tuple[List[int], Circuit]:
    #     """Get qubits and compiled circuit for an expression.

    #     If already compiled, returns cached qubits and circuit.
    #     Otherwise, compiles the expression and caches the result.

    #     Args:
    #         expr: Expression to compile

    #     Returns:
    #         Tuple of (qubit_indices, compiled_circuit)
    #     """
    #     # Get or allocate qubits (handles CSE)
    #     qubits = self.get_or_allocate_qubits(expr)

    #     # Check if already compiled
    #     if expr in self.expr_to_circuit:
    #         return qubits, self.expr_to_circuit[expr]

    #     # Compile the expression
    #     circ = _compile_with_context(expr, self)
    #     self.expr_to_circuit[expr] = circ
    #     return qubits, circ

    def get_or_compile(self, expr: "Expr") -> tuple[list[int], "Circuit"]:
        """Get qubits and compiled circuit for an expression.

        If already compiled, returns cached qubits and circuit.
        Otherwise, compiles the expression and caches the result.

        FIXED: Don't pre-allocate qubits for compound expressions.
        Only allocate for leaf nodes (Var, Const). Compound expressions
        allocate their result qubits AFTER compiling children.

        Args:
            expr: Expression to compile

        Returns:
            Tuple of (qubit_indices, compiled_circuit)
        """
        from .expr import Var, Const

        # Check if already compiled
        if expr in self.expr_to_circuit:
            qubits = self.expr_to_qubits[expr]
            return qubits, self.expr_to_circuit[expr]

        # For leaf nodes (Var, Const), allocate qubits first
        # For compound expressions, the compile function will allocate
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
    """Compile an expression using the given context.

    This is the internal dispatcher that routes to the appropriate
    compilation function based on expression type.
    """
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
    """Compile a Basic expression to a circuit.

    Basic expressions represent input parameters/oracles. They are allocated
    qubits but don't generate gates (the actual circuit is provided externally).
    """
    qubits = ctx.get_or_allocate_qubits(expr)
    max_qubit = max(qubits) if qubits else -1
    allocation = QubitAllocation(
        data_qubits=max_qubit + 1, ancilla_qubits=[], expr_data_qubits=qubits
    )
    # Circuit size only needs to accommodate this expression's qubits
    # The actual circuit will be expanded when composed with other circuits
    total_qubits = max_qubit + 1 if qubits else 0
    return Circuit(n_qubits=total_qubits, allocation=allocation)


def _compile_const(expr: Const, ctx: CompilationContext) -> Circuit:
    """Compile a Const expression to a circuit.

    Constants are initialized to their value using X gates.
    """
    qubits = ctx.get_or_allocate_qubits(expr)
    max_qubit = max(qubits) if qubits else -1
    allocation = QubitAllocation(
        data_qubits=max_qubit + 1, ancilla_qubits=[], expr_data_qubits=qubits
    )
    # Circuit size only needs to accommodate this expression's qubits
    # The actual circuit will be expanded when composed with other circuits
    total_qubits = max_qubit + 1 if qubits else 0
    circ = Circuit(n_qubits=total_qubits, allocation=allocation)

    # Initialize constant value
    _init_constant(circ, qubits, expr.value)

    return circ


def _compile_add(expr: Add, ctx: CompilationContext) -> Circuit:
    """Compile an Add expression to a circuit using modular arithmetic.

    Strategy:
    1. Get or compile left and right operands (CSE handled automatically)
    2. Extract modulus from type
    3. Use modular operators from operators.py
    4. Handle qubit allocation and circuit merging
    5. Handle Maybe type error flags if present
    """
    # Get qubits and circuits for operands (CSE: if they share subexpressions,
    # those subexpressions will have the same qubit assignments)
    left_qubits, left_circ = ctx.get_or_compile(expr.left)
    right_qubits, right_circ = ctx.get_or_compile(expr.right)

    n_bits = expr.qtype().n_bits
    ty = expr.qtype()
    left_type = expr.left.qtype()
    right_type = expr.right.qtype()

    # Extract inner qubits and error flags for Maybe types
    left_inner_qubits, left_error_flag = _extract_maybe_components(
        left_qubits, left_type
    )
    right_inner_qubits, right_error_flag = _extract_maybe_components(
        right_qubits, right_type
    )

    # Extract modulus (all arithmetic is modular in this language)
    modulus, _ = _get_modulus_and_inner_n_bits(ty)

    # Determine total qubits needed
    # Use inner qubits for overlap detection (error flags don't participate in operations)
    max_data_qubit = max(
        max(left_inner_qubits) if left_inner_qubits else 0,
        max(right_inner_qubits) if right_inner_qubits else 0,
        n_bits - 1,  # Result needs n_bits
    )

    # Ancillas: max of both circuits' ancillas
    max_ancillas = max(
        len(left_circ.allocation.ancilla_qubits),
        len(right_circ.allocation.ancilla_qubits),
    )

    # Check if we need temporary space for overlapping operands
    # Only check inner qubits (error flags are separate)
    left_inner_set = set(left_inner_qubits)
    right_inner_set = set(right_inner_qubits)
    need_temp_space = bool(left_inner_set & right_inner_set) and (
        left_inner_qubits != right_inner_qubits
    )

    # We need space for:
    # - result_qubits: n_bits qubits
    # - aux_qubits: n_bits + 2 qubits (for modular operations)
    # - temp_qubits: n_bits qubits (if needed for overlapping operands)
    # Total additional space needed: n_bits + (n_bits + 2) + (n_bits if need_temp_space else 0)

    # Allocate result qubits starting after max_ancillas
    result_start = max_data_qubit + 1 + max_ancillas
    result_qubits = list(range(result_start, result_start + n_bits))
    aux_start = result_start + n_bits
    aux_qubits = list(range(aux_start, aux_start + n_bits + 2))

    # Total qubits needed
    temp_space = n_bits if need_temp_space else 0
    total_qubits = max_data_qubit + 1 + max_ancillas + 2 * n_bits + 2 + temp_space
    total_ancillas = max_ancillas + 2 * n_bits + 2 + temp_space

    # Create merged circuit
    merged_allocation = QubitAllocation(
        data_qubits=max_data_qubit + 1,
        ancilla_qubits=list(
            range(max_data_qubit + 1, max_data_qubit + 1 + total_ancillas)
        ),
        expr_data_qubits=result_qubits,  # Expression's result qubits
    )
    merged_circ = Circuit(n_qubits=total_qubits, allocation=merged_allocation)

    # Remap and compose left circuit
    left_ancilla_target = list(
        range(max_data_qubit + 1, max_data_qubit + 1 + max_ancillas)
    )
    left_remapped = _remap_circuit(left_circ, left_ancilla_target, left_qubits)
    _compose_circuit(merged_circ, left_remapped)

    # Remap and compose right circuit
    right_ancilla_target = list(
        range(max_data_qubit + 1, max_data_qubit + 1 + max_ancillas)
    )

    if left_inner_set & right_inner_set:
        # They share qubits - need temporary space
        if left_inner_qubits == right_inner_qubits:
            # Same expression: x + x = 2x
            temp_inner_qubits = left_inner_qubits
        else:
            # Different but overlapping: need temporary space
            # Allocate temp space after aux_qubits
            inner_n_bits = len(left_inner_qubits)
            temp_start = aux_start + n_bits + 2
            temp_inner_qubits = list(range(temp_start, temp_start + inner_n_bits))
            # Copy right inner qubits to temp
            right_remapped = _remap_circuit(
                right_circ, right_ancilla_target, right_qubits
            )
            _compose_circuit(merged_circ, right_remapped)
            _copy_qubits(merged_circ, right_inner_qubits, temp_inner_qubits)
            right_inner_qubits = temp_inner_qubits
    else:
        # No overlap: remap right to its assigned qubits
        right_remapped = _remap_circuit(right_circ, right_ancilla_target, right_qubits)
        _compose_circuit(merged_circ, right_remapped)

    # Check if operands are constants
    from .expr import Const

    left_is_const = isinstance(expr.left, Const)
    right_is_const = isinstance(expr.right, Const)

    # Initialize aux_qubits to zero (required by modular_add functions)
    # aux_qubits[0] is control, aux_qubits[1:n_bits+1] will contain result
    # Note: In quantum circuits, qubits are initialized to |0⟩ by default,
    # but we ensure they are explicitly zero by not applying any gates before use

    # Extract inner n_bits for the actual Mod type (excluding Maybe flag)
    inner_n_bits = len(left_inner_qubits)
    result_inner_qubits = result_qubits[:inner_n_bits]
    result_error_flag_qubit = (
        result_qubits[inner_n_bits] if isinstance(ty, Maybe) else None
    )

    if left_is_const and right_is_const:
        # Both constants: initialize result with left, then add right
        _init_constant(merged_circ, result_inner_qubits, expr.left.value)
        # Use quantum-classical to add right constant
        operators.modular_add_quantum_classical(
            merged_circ.qc, result_inner_qubits, expr.right.value, modulus, aux_qubits
        )
        # Result is in aux_qubits[1:inner_n_bits+1]
        _copy_qubits(merged_circ, aux_qubits[1 : inner_n_bits + 1], result_inner_qubits)
    elif left_is_const:
        # Left is constant: copy right to result, then add left constant
        _copy_qubits(merged_circ, right_inner_qubits, result_inner_qubits)
        operators.modular_add_quantum_classical(
            merged_circ.qc, result_inner_qubits, expr.left.value, modulus, aux_qubits
        )
        # Result is in aux_qubits[1:inner_n_bits+1]
        _copy_qubits(merged_circ, aux_qubits[1 : inner_n_bits + 1], result_inner_qubits)
    elif right_is_const:
        # Right is constant: copy left to result, then add right constant
        _copy_qubits(merged_circ, left_inner_qubits, result_inner_qubits)
        operators.modular_add_quantum_classical(
            merged_circ.qc, result_inner_qubits, expr.right.value, modulus, aux_qubits
        )
        # Result is in aux_qubits[1:inner_n_bits+1]
        _copy_qubits(merged_circ, aux_qubits[1 : inner_n_bits + 1], result_inner_qubits)
    else:
        # Both quantum: copy left to result, then add right quantum
        _copy_qubits(merged_circ, left_inner_qubits, result_inner_qubits)
        operators.modular_add_quantum_quantum(
            merged_circ.qc, result_inner_qubits, right_inner_qubits, modulus, aux_qubits
        )
        # Result is in aux_qubits[1:inner_n_bits+1]
        _copy_qubits(merged_circ, aux_qubits[1 : inner_n_bits + 1], result_inner_qubits)

    # Handle error flags if result is Maybe type
    if isinstance(ty, Maybe) and result_error_flag_qubit is not None:
        _combine_error_flags(
            merged_circ, left_error_flag, right_error_flag, result_error_flag_qubit
        )

    # Update context to use result_qubits as the output
    ctx.expr_to_qubits[expr] = result_qubits
    ctx.allocation = merged_allocation
    return merged_circ


def _compile_sub(expr: Sub, ctx: CompilationContext) -> Circuit:
    """Compile a Sub expression to a circuit using modular arithmetic.

    Strategy: Use modular subtraction operators from operators.py
    """
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
    if (
        len(left_inner_qubits) != inner_n_bits
        or len(right_inner_qubits) != inner_n_bits
    ):
        raise ValueError(
            f"Sub operands must have inner width {inner_n_bits}, "
            f"got left {len(left_inner_qubits)}, right {len(right_inner_qubits)}"
        )

    # Overlap detection (inner qubits only)
    left_inner_set = set(left_inner_qubits)
    right_inner_set = set(right_inner_qubits)
    need_temp_space = bool(left_inner_set & right_inner_set) and (
        left_inner_qubits != right_inner_qubits
    )

    # Layout:
    # [operand data ...][result (n_bits)][child ancillas][aux (inner_n_bits+2)][temp (optional inner_n_bits)]
    max_data_qubit_operands = max(
        max(left_inner_qubits) if left_inner_qubits else 0,
        max(right_inner_qubits) if right_inner_qubits else 0,
        n_bits - 1,
    )

    max_child_ancillas = max(
        len(left_circ.allocation.ancilla_qubits),
        len(right_circ.allocation.ancilla_qubits),
    )

    result_start = max_data_qubit_operands + 1
    result_qubits = list(range(result_start, result_start + n_bits))
    result_inner_qubits = result_qubits[:inner_n_bits]
    result_error_flag_qubit = (
        result_qubits[inner_n_bits] if isinstance(ty, Maybe) else None
    )

    child_ancilla_start = result_start + n_bits
    child_ancilla_qubits = list(
        range(child_ancilla_start, child_ancilla_start + max_child_ancillas)
    )

    aux_start = child_ancilla_start + max_child_ancillas
    aux_qubits = list(range(aux_start, aux_start + inner_n_bits + 2))

    temp_inner_qubits: List[int] = []
    if need_temp_space:
        temp_start = aux_start + inner_n_bits + 2
        temp_inner_qubits = list(range(temp_start, temp_start + inner_n_bits))

    # Calculate total qubits needed (including working qubits)
    # Layout: [operand data ...][result (n_bits)][child ancillas][aux (inner_n_bits+2)][temp (optional inner_n_bits)]
    temp_space = inner_n_bits if need_temp_space else 0
    total_qubits = (
        max_data_qubit_operands
        + 1  # operand data qubits
        + n_bits  # result qubits
        + max_child_ancillas  # child ancilla qubits
        + inner_n_bits
        + 2  # aux qubits
        + temp_space  # temp qubits (if needed)
    )
    data_qubits_count = max(result_qubits) + 1

    # For the allocation, we need to include ALL ancillas for correct state vector size,
    # but distinguish between those that need postselection vs working qubits.
    # We'll store the working qubits info in a way the simulator can understand.
    all_ancilla_qubits = child_ancilla_qubits + aux_qubits
    if need_temp_space and temp_inner_qubits:
        all_ancilla_qubits += temp_inner_qubits

    merged_allocation = QubitAllocation(
        data_qubits=data_qubits_count,
        ancilla_qubits=child_ancilla_qubits,  # Only ancillas needing postselection
        expr_data_qubits=result_qubits,  # Expression's result qubits
    )
    merged_circ = Circuit(n_qubits=total_qubits, allocation=merged_allocation)

    # Remap and compose left circuit
    left_ancilla_target = child_ancilla_qubits[
        : len(left_circ.allocation.ancilla_qubits)
    ]
    left_remapped = _remap_circuit(left_circ, left_ancilla_target, left_qubits)
    _compose_circuit(merged_circ, left_remapped)

    # Remap and compose right circuit
    right_ancilla_target = child_ancilla_qubits[
        : len(right_circ.allocation.ancilla_qubits)
    ]
    right_remapped = _remap_circuit(right_circ, right_ancilla_target, right_qubits)
    _compose_circuit(merged_circ, right_remapped)

    # If overlapping, copy right into temp buffer
    if need_temp_space and temp_inner_qubits:
        _copy_qubits(merged_circ, right_inner_qubits, temp_inner_qubits)
        right_inner_qubits = temp_inner_qubits

    left_is_const = isinstance(expr.left, Const)
    right_is_const = isinstance(expr.right, Const)

    if left_is_const and right_is_const:
        # Both constants: compute (left.value - right.value) mod N directly
        result_value = (expr.left.value - expr.right.value) % modulus
        _init_constant(merged_circ, result_inner_qubits, result_value)
    elif left_is_const:
        # Left is constant: initialize result with left value, then subtract right
        _init_constant(merged_circ, result_inner_qubits, expr.left.value)
        operators.modular_subtract_quantum_quantum(
            merged_circ.qc, result_inner_qubits, right_inner_qubits, modulus, aux_qubits
        )
        # Result is in aux_qubits[1:inner_n_bits+1], copy it back
        _copy_qubits(merged_circ, aux_qubits[1 : inner_n_bits + 1], result_inner_qubits)
    elif right_is_const:
        # Right is constant: copy left to result, then subtract right constant
        _copy_qubits(merged_circ, left_inner_qubits, result_inner_qubits)
        operators.modular_subtract_quantum_classical(
            merged_circ.qc, result_inner_qubits, expr.right.value, modulus, aux_qubits
        )
        # Result is in aux_qubits[1:inner_n_bits+1], copy it back
        _copy_qubits(merged_circ, aux_qubits[1 : inner_n_bits + 1], result_inner_qubits)
    else:
        # Both quantum: implement a - b as a + (N - b) using working addition
        # This is mathematically correct: a - b ≡ a + (N - b) (mod N)

        # Use the exact same pattern as modular_add_quantum_quantum
        # but replace each 2^i with (N - 2^i) for the right operand

        # Apply controlled modular additions for each bit of a (left operand)
        for i in range(inner_n_bits):
            tempk = 2 ** (inner_n_bits - (i + 1))
            addin = (
                operators.modular_addin_circuit(inner_n_bits, tempk, modulus)
                .to_gate()
                .control(1)
            )
            addin.name = f"Addin({tempk}, N)"
            merged_circ.qc.append(
                addin, [left_inner_qubits[inner_n_bits - (i + 1)]] + aux_qubits[:]
            )

        # Apply controlled modular additions for each bit of b, but with (N - 2^i)
        for i in range(inner_n_bits):
            tempk = 2 ** (inner_n_bits - (i + 1))
            neg_tempk = (modulus - tempk) % modulus  # This is N - 2^i
            addin = (
                operators.modular_addin_circuit(inner_n_bits, neg_tempk, modulus)
                .to_gate()
                .control(1)
            )
            addin.name = f"Addin(-{tempk}, N)"
            merged_circ.qc.append(
                addin, [right_inner_qubits[inner_n_bits - (i + 1)]] + aux_qubits[:]
            )

        # Result is in aux_qubits[1:inner_n_bits+1]
        _copy_qubits(merged_circ, aux_qubits[1 : inner_n_bits + 1], result_inner_qubits)

    if isinstance(ty, Maybe) and result_error_flag_qubit is not None:
        _combine_error_flags(
            merged_circ, left_error_flag, right_error_flag, result_error_flag_qubit
        )

    ctx.expr_to_qubits[expr] = result_qubits
    ctx.allocation = merged_allocation
    return merged_circ


def _compile_mul(expr: Mul, ctx: CompilationContext) -> Circuit:
    """Compile a Mul expression to a circuit using modular arithmetic."""
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

    if len(left_inner_qubits) != len(right_inner_qubits):
        raise ValueError(
            f"Operands must have same inner Mod type size for multiplication: "
            f"left has {len(left_inner_qubits)} qubits, right has {len(right_inner_qubits)} qubits, "
            f"expected {inner_n_bits} qubits"
        )

    max_data_qubit = max(
        max(left_inner_qubits) if left_inner_qubits else 0,
        max(right_inner_qubits) if right_inner_qubits else 0,
        n_bits - 1,  # Result needs n_bits
    )

    max_ancillas = max(
        len(left_circ.allocation.ancilla_qubits),
        len(right_circ.allocation.ancilla_qubits),
    )

    left_inner_set = set(left_inner_qubits)
    right_inner_set = set(right_inner_qubits)
    need_temp_space = bool(left_inner_set & right_inner_set) and (
        left_inner_qubits != right_inner_qubits
    )

    result_start = max_data_qubit + 1 + max_ancillas
    result_qubits = list(range(result_start, result_start + n_bits))

    aux_start = result_start + n_bits
    aux_qubits = list(range(aux_start, aux_start + inner_n_bits + 2))
    inner_result_qubits = aux_qubits[1 : inner_n_bits + 1]

    temp_space = inner_n_bits if need_temp_space else 0
    total_qubits = (
        max_data_qubit + 1 + max_ancillas + n_bits + inner_n_bits + 2 + temp_space
    )
    total_ancillas = max_ancillas + n_bits + inner_n_bits + 2 + temp_space

    merged_allocation = QubitAllocation(
        data_qubits=max_data_qubit + 1,
        ancilla_qubits=list(
            range(max_data_qubit + 1, max_data_qubit + 1 + total_ancillas)
        ),
        expr_data_qubits=result_qubits,  # Expression's result qubits
    )
    merged_circ = Circuit(n_qubits=total_qubits, allocation=merged_allocation)

    left_ancilla_target = list(
        range(max_data_qubit + 1, max_data_qubit + 1 + max_ancillas)
    )
    left_remapped = _remap_circuit(left_circ, left_ancilla_target, left_qubits)
    _compose_circuit(merged_circ, left_remapped)

    right_ancilla_target = list(
        range(max_data_qubit + 1, max_data_qubit + 1 + max_ancillas)
    )

    if left_inner_set & right_inner_set:
        if left_inner_qubits == right_inner_qubits:
            temp_inner_qubits = left_inner_qubits
        else:
            temp_start = aux_start + inner_n_bits + 2
            temp_inner_qubits = list(range(temp_start, temp_start + inner_n_bits))
            right_remapped = _remap_circuit(
                right_circ, right_ancilla_target, right_qubits
            )
            _compose_circuit(merged_circ, right_remapped)
            _copy_qubits(merged_circ, right_inner_qubits, temp_inner_qubits)
            right_inner_qubits = temp_inner_qubits
    else:
        right_remapped = _remap_circuit(right_circ, right_ancilla_target, right_qubits)
        _compose_circuit(merged_circ, right_remapped)

    left_is_const = isinstance(expr.left, Const)
    right_is_const = isinstance(expr.right, Const)

    # Extract result inner qubits and error flag
    result_inner_qubits = result_qubits[:inner_n_bits]
    result_error_flag_qubit = (
        result_qubits[inner_n_bits] if isinstance(ty, Maybe) else None
    )

    if left_is_const and right_is_const:
        result_value = (expr.left.value * expr.right.value) % modulus
        _init_constant(merged_circ, result_inner_qubits, result_value)
    elif left_is_const:
        _init_constant(merged_circ, inner_result_qubits, 0)
        operators.modular_multiply_quantum_classical(
            merged_circ.qc, right_inner_qubits, expr.left.value, modulus, aux_qubits
        )
        _copy_qubits(merged_circ, inner_result_qubits, result_inner_qubits)
    elif right_is_const:
        _init_constant(merged_circ, inner_result_qubits, 0)
        operators.modular_multiply_quantum_classical(
            merged_circ.qc, left_inner_qubits, expr.right.value, modulus, aux_qubits
        )
        # Copy result back to result_qubits (from inner_result_qubits)
        _copy_qubits(merged_circ, inner_result_qubits, result_inner_qubits)
    else:
        _init_constant(merged_circ, inner_result_qubits, 0)
        operators.modular_multiply_quantum_quantum(
            merged_circ.qc, left_inner_qubits, right_inner_qubits, modulus, aux_qubits
        )
        _copy_qubits(merged_circ, inner_result_qubits, result_inner_qubits)

    # Handle error flags if result is Maybe type
    if isinstance(ty, Maybe) and result_error_flag_qubit is not None:
        _combine_error_flags(
            merged_circ, left_error_flag, right_error_flag, result_error_flag_qubit
        )

    # Update context to use result_qubits as the output
    ctx.expr_to_qubits[expr] = result_qubits
    ctx.allocation = merged_allocation
    return merged_circ


def _compile_pow(expr: Pow, ctx: CompilationContext) -> Circuit:
    """
    Compile a Pow expression to a circuit using modular arithmetic.

    Handles three cases:
    1. Both base and exponent are constants: compute classically
    2. Base is quantum, exponent is classical (x^a): repeated squaring with quantum mult
    3. Base is classical, exponent is quantum (a^x): Shor-style with precomputed powers

    Note: x^y where both are quantum is not supported (too expensive).
    """
    from .expr import Const

    # Determine what's constant vs quantum
    base_is_const = isinstance(expr.base, Const)
    exp_is_const = isinstance(expr.exponent, (int, Const))

    # Get type information
    ty = expr.qtype()
    modulus, inner_n_bits = _get_modulus_and_inner_n_bits(ty)

    # Get exponent value if constant
    if isinstance(expr.exponent, int):
        exp_value = expr.exponent
    elif isinstance(expr.exponent, Const):
        exp_value = expr.exponent.value
    else:
        exp_value = None

    # Get base value if constant
    if isinstance(expr.base, Const):
        base_value = expr.base.value
    else:
        base_value = None

    # Case 1: Both constant - compute classically
    if base_is_const and exp_is_const:
        result_qubits = ctx.get_or_allocate_qubits(expr)
        result_value = pow(base_value, exp_value, modulus)

        max_qubit = max(result_qubits) if result_qubits else -1
        allocation = QubitAllocation(
            data_qubits=max_qubit + 1 if result_qubits else 0,
            ancilla_qubits=[],
            expr_data_qubits=result_qubits,
        )
        # Circuit size only needs to accommodate this expression's qubits
        total_qubits = max_qubit + 1 if result_qubits else 0
        circ = Circuit(n_qubits=total_qubits, allocation=allocation)
        _init_constant(circ, result_qubits, result_value)

        ctx.expr_to_qubits[expr] = result_qubits
        return circ

    # Case 2: a^x - Classical base, quantum exponent (Shor's algorithm style)
    if base_is_const and not exp_is_const:
        return _compile_pow_classical_base_quantum_exp(
            expr, ctx, base_value, modulus, inner_n_bits
        )

    # Case 3: x^a - Quantum base, classical exponent
    if not base_is_const and exp_is_const:
        return _compile_pow_quantum_base_classical_exp(
            expr, ctx, exp_value, modulus, inner_n_bits
        )

    raise NotImplementedError(
        "Quantum-quantum exponentiation (x^y where both x and y are in superposition) "
        "is not supported due to exponential resource requirements. "
        "Consider using a^x (classical base) or x^a (classical exponent) instead."
    )


def _compile_pow_classical_base_quantum_exp(
    expr: Pow,
    ctx: CompilationContext,
    base_value: int,
    modulus: int,
    inner_n_bits: int,
) -> Circuit:
    """
    Compile a^x where a is classical and x is quantum.

    This is the Shor's algorithm style exponentiation.
    Uses precomputed powers: a^1, a^2, a^4, ..., a^(2^{n-1}) mod N

    For each bit x_i of the exponent:
        if x_i = 1: result *= a^(2^i) mod N
    """
    from math import gcd

    # Check coprimality (required for modular exponentiation to be well-defined)
    if gcd(base_value, modulus) != 1:
        raise ValueError(
            f"Base {base_value} and modulus {modulus} must be coprime for modular exponentiation"
        )

    # Get qubits for exponent
    exp_qubits, exp_circ = ctx.get_or_compile(expr.exponent)
    exp_type = expr.exponent.qtype()
    exp_inner_qubits, exp_error_flag = _extract_maybe_components(exp_qubits, exp_type)

    n = len(exp_inner_qubits)
    ty = expr.qtype()
    n_bits = ty.n_bits

    # Precompute powers: a^(2^i) mod N
    from .operators import precompute_powers

    powers = precompute_powers(base_value, n, modulus)

    # Allocate qubits
    max_data_qubit = max(exp_inner_qubits) if exp_inner_qubits else 0
    max_exp_ancillas = len(exp_circ.allocation.ancilla_qubits)

    # Layout: [exp_data][result (n_bits)][exp_ancillas][aux (inner_n_bits+2)]
    result_start = max_data_qubit + 1
    result_qubits = list(range(result_start, result_start + n_bits))
    result_inner_qubits = result_qubits[:inner_n_bits]
    result_error_flag_qubit = (
        result_qubits[inner_n_bits] if isinstance(ty, Maybe) else None
    )

    exp_ancilla_start = result_start + n_bits
    exp_ancilla_qubits = list(
        range(exp_ancilla_start, exp_ancilla_start + max_exp_ancillas)
    )

    aux_start = exp_ancilla_start + max_exp_ancillas
    aux_qubits = list(range(aux_start, aux_start + inner_n_bits + 2))

    total_qubits = aux_start + inner_n_bits + 2

    # Create circuit
    merged_allocation = QubitAllocation(
        data_qubits=result_start + n_bits,
        ancilla_qubits=exp_ancilla_qubits + aux_qubits,
        expr_data_qubits=result_qubits,
    )
    merged_circ = Circuit(n_qubits=total_qubits, allocation=merged_allocation)

    # Compose exponent circuit
    exp_remapped = _remap_circuit(
        exp_circ, exp_ancilla_qubits[:max_exp_ancillas], exp_qubits
    )
    _compose_circuit(merged_circ, exp_remapped)

    # Initialize result to |1⟩
    if inner_n_bits > 0:
        merged_circ.qc.x(result_inner_qubits[0])  # LSB = 1

    # For each bit of exponent, apply controlled modular multiplication
    from .operators import _apply_controlled_inplace_multiply

    for i in range(n):
        power = powers[i]  # a^(2^i) mod N

        # Controlled in-place multiplication: if exp_inner_qubits[i] = 1, result *= power
        _apply_controlled_inplace_multiply(
            merged_circ.qc,
            ctrl_qubit=exp_inner_qubits[i],
            target_qubits=result_inner_qubits,
            k=power,
            N=modulus,
            aux_qubits=aux_qubits,
        )

    # Handle error flags
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
    """
    Compile x^a where x is quantum and a is classical.

    Uses separate registers for each power to avoid aux register clearing issues.
    Space: O(a * n) qubits for intermediate powers.

    Key insight: modular_multiply_quantum_quantum ADDS the result to aux (aux += a*b),
    so aux must be |0⟩ before each multiplication. We maintain this by:
    1. Copying result from aux to destination power register
    2. Clearing aux by XORing with destination (same values cancel to 0)
    """
    from .circuit import Circuit, QubitAllocation
    from .qtype import Maybe
    from . import operators

    # Get qubits for base
    base_qubits, base_circ = ctx.get_or_compile(expr.base)
    base_type = expr.base.qtype()
    base_inner_qubits, base_error_flag = _extract_maybe_components(
        base_qubits, base_type
    )

    ty = expr.qtype()
    n_bits = ty.n_bits

    # Special case: x^0 = 1
    if exp_value == 0:
        result_qubits = ctx.get_or_allocate_qubits(expr)
        max_qubit = max(result_qubits) if result_qubits else -1
        allocation = QubitAllocation(
            data_qubits=max_qubit + 1 if result_qubits else 0,
            ancilla_qubits=[],
            expr_data_qubits=result_qubits,
        )
        total = max(ctx.allocation.total_qubits(), max(result_qubits) + 1)
        circ = Circuit(n_qubits=total, allocation=allocation)
        _init_constant(circ, result_qubits[:inner_n_bits], 1)
        ctx.expr_to_qubits[expr] = result_qubits
        return circ

    # Special case: x^1 = x
    if exp_value == 1:
        result_qubits = ctx.get_or_allocate_qubits(expr)
        max_data = max(max(base_inner_qubits), max(result_qubits))
        allocation = QubitAllocation(
            data_qubits=max_data + 1,
            ancilla_qubits=list(base_circ.allocation.ancilla_qubits),
            expr_data_qubits=result_qubits,
        )
        circ = Circuit(
            n_qubits=max_data + 1 + len(allocation.ancilla_qubits),
            allocation=allocation,
        )
        _compose_circuit(circ, base_circ)
        _copy_qubits(circ, base_inner_qubits, result_qubits[:inner_n_bits])
        ctx.expr_to_qubits[expr] = result_qubits
        return circ

    # For exp_value >= 2: allocate separate registers for each power
    max_data_qubit = max(base_inner_qubits) if base_inner_qubits else 0
    max_base_ancillas = len(base_circ.allocation.ancilla_qubits)

    # Allocate power registers for x^1, x^2, x^3, ..., x^a
    power_regs = []
    next_start = max_data_qubit + 1

    for i in range(exp_value):  # x^1, x^2, ..., x^a
        reg = list(range(next_start, next_start + inner_n_bits))
        power_regs.append(reg)
        next_start += inner_n_bits

    # Result qubits
    result_start = next_start
    result_qubits = list(range(result_start, result_start + n_bits))
    result_inner_qubits = result_qubits[:inner_n_bits]
    result_error_flag_qubit = (
        result_qubits[inner_n_bits] if isinstance(ty, Maybe) else None
    )

    base_ancilla_start = result_start + n_bits
    base_ancilla_qubits = list(
        range(base_ancilla_start, base_ancilla_start + max_base_ancillas)
    )

    aux_start = base_ancilla_start + max_base_ancillas
    aux_qubits = list(range(aux_start, aux_start + inner_n_bits + 2))
    aux_result = aux_qubits[1 : inner_n_bits + 1]

    total_qubits = aux_start + inner_n_bits + 2

    merged_allocation = QubitAllocation(
        data_qubits=result_start + n_bits,
        ancilla_qubits=base_ancilla_qubits,  # Only base ancillas for postselection
        expr_data_qubits=result_qubits,
    )
    merged_circ = Circuit(n_qubits=total_qubits, allocation=merged_allocation)

    # Compose base circuit
    if max_base_ancillas > 0:
        base_remapped = _remap_circuit(
            base_circ, base_ancilla_qubits[:max_base_ancillas], base_qubits
        )
    else:
        base_remapped = _remap_circuit(base_circ, [], base_qubits)
    _compose_circuit(merged_circ, base_remapped)
    _copy_qubits(merged_circ, base_inner_qubits, power_regs[0])

    # Compute x^2, x^3, ..., x^a
    # x^(i+1) = x^i * x
    for i in range(1, exp_value):
        # Multiply: aux_result = power_regs[i-1] * base
        operators.modular_multiply_quantum_quantum(
            merged_circ.qc,
            power_regs[i - 1],  # x^i
            base_inner_qubits,  # x
            modulus,
            aux_qubits,  # result goes to aux_qubits[1:n+1]
        )
        _copy_qubits(merged_circ, aux_result, power_regs[i])

        # Clear aux_result for next iteration by XORing with power_regs[i]
        # aux_result = power_regs[i] XOR power_regs[i] = 0
        _copy_qubits(merged_circ, power_regs[i], aux_result)

    # Final result is in power_regs[exp_value - 1] = x^a
    # Copy to result_qubits
    _copy_qubits(merged_circ, power_regs[exp_value - 1], result_inner_qubits)

    # Handle error flags if result is Maybe type
    if isinstance(ty, Maybe) and result_error_flag_qubit is not None:
        if base_error_flag is not None:
            merged_circ.qc.cx(base_error_flag, result_error_flag_qubit)

    ctx.expr_to_qubits[expr] = result_qubits
    ctx.allocation = merged_allocation
    return merged_circ


def _compile_floordiv(expr: Div, ctx: CompilationContext) -> Circuit:
    """Compile a Div expression to a circuit."""
    # TODO: Implement division
    qubits = ctx.get_or_allocate_qubits(expr)
    n_bits = expr.qtype().n_bits
    allocation = QubitAllocation(
        data_qubits=n_bits, ancilla_qubits=[], expr_data_qubits=qubits
    )
    return Circuit(n_qubits=n_bits, allocation=allocation)


def _compile_sqrt(expr: Sqrt, ctx: CompilationContext) -> Circuit:
    """Compile a Sqrt expression to a circuit."""
    # TODO: Implement square root
    qubits = ctx.get_or_allocate_qubits(expr)
    n_bits = expr.qtype().n_bits
    allocation = QubitAllocation(
        data_qubits=n_bits, ancilla_qubits=[], expr_data_qubits=qubits
    )
    return Circuit(n_qubits=n_bits, allocation=allocation)


def _compile_inv(expr: Inv, ctx: CompilationContext) -> Circuit:
    """Compile an Inv expression to a circuit.

    This is a black box implementation - the actual modular inverse computation
    is not implemented, but the circuit structure is set up correctly for Maybe types.
    """
    # Get qubits and circuit for operand
    operand_qubits, operand_circ = ctx.get_or_compile(expr.expr)

    n_bits = expr.qtype().n_bits
    ty = expr.qtype()
    operand_type = expr.expr.qtype()

    # Extract inner qubits and error flag from operand
    operand_inner_qubits, operand_error_flag = _extract_maybe_components(
        operand_qubits, operand_type
    )

    # Result is Maybe type, so it has an error flag
    inner_n_bits = len(operand_inner_qubits)

    # Determine total qubits needed
    max_data_qubit = max(
        max(operand_inner_qubits) if operand_inner_qubits else 0,
        n_bits - 1,  # Result needs n_bits
    )

    # Ancillas from operand circuit
    max_ancillas = len(operand_circ.allocation.ancilla_qubits)

    # Allocate result qubits
    result_start = max_data_qubit + 1 + max_ancillas
    result_qubits = list(range(result_start, result_start + n_bits))
    result_inner_qubits = result_qubits[:inner_n_bits]
    result_error_flag_qubit = (
        result_qubits[inner_n_bits] if isinstance(ty, Maybe) else None
    )

    # Create merged circuit
    merged_allocation = QubitAllocation(
        data_qubits=max_data_qubit + 1,
        ancilla_qubits=list(
            range(max_data_qubit + 1, max_data_qubit + 1 + max_ancillas)
        ),
        expr_data_qubits=result_qubits,  # Expression's result qubits
    )
    merged_circ = Circuit(n_qubits=result_start + n_bits, allocation=merged_allocation)

    # Remap and compose operand circuit
    operand_ancilla_target = list(
        range(max_data_qubit + 1, max_data_qubit + 1 + max_ancillas)
    )
    operand_remapped = _remap_circuit(
        operand_circ, operand_ancilla_target, operand_qubits
    )
    _compose_circuit(merged_circ, operand_remapped)

    # Black box: For now, just copy the operand to result
    # In a real implementation, this would compute the modular inverse
    # For now, we'll just copy the input (this is a placeholder)
    _copy_qubits(merged_circ, operand_inner_qubits, result_inner_qubits)

    # Copy error flag from operand to result (if both are Maybe types)
    if isinstance(ty, Maybe) and result_error_flag_qubit is not None:
        if operand_error_flag is not None:
            merged_circ.qc.cx(operand_error_flag, result_error_flag_qubit)

    # Update context
    ctx.expr_to_qubits[expr] = result_qubits
    ctx.allocation = merged_allocation
    return merged_circ


def _compile_log(expr: Log, ctx: CompilationContext) -> Circuit:
    """Compile a Log expression to a circuit."""
    # TODO: Implement logarithm
    qubits = ctx.get_or_allocate_qubits(expr)
    n_bits = expr.qtype().n_bits
    allocation = QubitAllocation(
        data_qubits=n_bits, ancilla_qubits=[], expr_data_qubits=qubits
    )
    return Circuit(n_qubits=n_bits, allocation=allocation)


def _compile_indexed(expr: Indexed, ctx: CompilationContext) -> Circuit:
    """Compile an Indexed expression to a circuit."""
    # TODO: Implement indexed operation
    qubits = ctx.get_or_allocate_qubits(expr)
    n_bits = expr.qtype().n_bits
    allocation = QubitAllocation(
        data_qubits=n_bits, ancilla_qubits=[], expr_data_qubits=qubits
    )
    return Circuit(n_qubits=n_bits, allocation=allocation)


def _compile_join(expr: Join, ctx: CompilationContext) -> Circuit:
    """Compile a Join expression to a circuit."""
    # TODO: Implement join operation
    qubits = ctx.get_or_allocate_qubits(expr)
    n_bits = expr.qtype().n_bits
    allocation = QubitAllocation(
        data_qubits=n_bits, ancilla_qubits=[], expr_data_qubits=qubits
    )
    return Circuit(n_qubits=n_bits, allocation=allocation)


# Helper functions


def _combine_error_flags(
    circ: Circuit,
    left_error_flag: int | None,
    right_error_flag: int | None,
    result_error_flag: int,
):
    """Combine error flags from two operands using logical OR.

    If either operand has an error (flag = 1), the result should also have an error.
    This implements the semantics from Figure 4(f) where error flags are combined.

    Args:
        circ: Circuit to add gates to
        left_error_flag: Error flag qubit from left operand (None if not Maybe type)
        right_error_flag: Error flag qubit from right operand (None if not Maybe type)
        result_error_flag: Error flag qubit for result (must be initialized to |0⟩)
    """
    # Initialize result error flag to 0 (should already be |0⟩, but ensure it)
    # Then combine: result = left OR right
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
    """Extract inner qubits and error flag from a Maybe type.

    Args:
        qubits: List of qubit indices for the expression
        qtype: Type of the expression

    Returns:
        Tuple of (inner_qubits, error_flag_qubit)
        - If qtype is Mod: returns (qubits, None)
        - If qtype is Maybe: returns (inner_qubits, error_flag_qubit)
    """
    if isinstance(qtype, Mod):
        return qubits, None
    elif isinstance(qtype, Maybe):
        # Last qubit is the error flag
        inner_qubits = qubits[:-1]
        error_flag = qubits[-1]
        # Recursively extract from inner type
        inner_inner, _ = _extract_maybe_components(inner_qubits, qtype.inner_type)
        return inner_inner, error_flag
    else:
        raise ValueError(f"Cannot extract Maybe components from type {qtype}")


def _unwrap_maybe_type(ty: QType) -> tuple[Mod, int]:
    """
    Unwrap Maybe types to get the inner Mod type and the number of Maybe wrappers.

    Args:
        ty: A QType (Mod or Maybe)

    Returns:
        Tuple of (inner Mod type, number of Maybe wrappers)

    Examples:
        _unwrap_maybe_type(Mod(11)) -> (Mod(11), 0)
        _unwrap_maybe_type(Maybe(Mod(11))) -> (Mod(11), 1)
        _unwrap_maybe_type(Maybe(Maybe(Mod(11)))) -> (Mod(11), 2)
    """
    if isinstance(ty, Mod):
        return ty, 0
    elif isinstance(ty, Maybe):
        inner_mod, depth = _unwrap_maybe_type(ty.inner_type)
        return inner_mod, depth + 1
    else:
        raise ValueError(f"Cannot unwrap type {ty}")


def _get_modulus_and_inner_n_bits(ty: QType) -> tuple[int, int]:
    """
    Extract modulus and inner n_bits from a QType (handling Maybe wrappers).

    Args:
        ty: A QType (Mod or Maybe)

    Returns:
        Tuple of (modulus, inner_n_bits)
    """
    inner_mod, _ = _unwrap_maybe_type(ty)
    return inner_mod.N, inner_mod.n_bits


def _remap_circuit(
    circ: Circuit, ancilla_target: List[int], data_qubits: List[int]
) -> Circuit:
    """Remap a circuit's qubits to new positions.

    Args:
        circ: Circuit to remap
        ancilla_target: Target qubit indices for ancillas
        data_qubits: Target qubit indices for data qubits (expression's qubits)

    Returns:
        New circuit with remapped qubits
    """
    qubit_map: Dict[int, int] = {}

    # Find all qubits actually used in the circuit
    used_qubits = set()
    for instruction in circ.qc.data:
        for q in instruction.qubits:
            used_qubits.add(circ.qc.find_bit(q).index)

    # Map expression's data qubits to target data qubits
    if circ.allocation.expr_data_qubits is not None:
        expr_data_qubits = circ.allocation.expr_data_qubits
        if len(expr_data_qubits) != len(data_qubits):
            raise ValueError(
                f"Expression uses {len(expr_data_qubits)} data qubits but {len(data_qubits)} target qubits provided. "
                f"Expression qubits: {expr_data_qubits}, target: {data_qubits}"
            )
        # Map expression's data qubits to target data qubits
        for old_q, new_q in zip(expr_data_qubits, data_qubits):
            qubit_map[old_q] = new_q
    else:
        # Fallback: assume data qubits start from 0
        num_data = min(circ.allocation.data_qubits, len(data_qubits))
        for i in range(num_data):
            qubit_map[i] = data_qubits[i]

    # Map ancilla qubits
    ancilla_qubits_used = sorted(circ.allocation.ancilla_qubits)
    if len(ancilla_qubits_used) > len(ancilla_target):
        raise ValueError(
            f"Circuit uses {len(ancilla_qubits_used)} ancilla qubits but only {len(ancilla_target)} target qubits provided. "
            f"Used ancilla qubits: {ancilla_qubits_used}, target: {ancilla_target}"
        )
    for old_ancilla, new_ancilla in zip(ancilla_qubits_used, ancilla_target):
        qubit_map[old_ancilla] = new_ancilla

    # Map any remaining used qubits that aren't in expr_data_qubits or ancilla_qubits
    # These are likely intermediate results or temporary qubits
    # We need to allocate new qubits for them that don't conflict with targets
    unmapped_used = used_qubits - set(qubit_map.keys())
    if unmapped_used:
        # Allocate new qubits starting after all target qubits
        all_target_qubits = data_qubits + ancilla_target
        next_free_qubit = max(all_target_qubits) + 1 if all_target_qubits else 0
        for old_q in sorted(unmapped_used):
            qubit_map[old_q] = next_free_qubit
            next_free_qubit += 1

    # Create new circuit with remapped gates
    all_target_qubits = data_qubits + ancilla_target
    # Include any newly allocated qubits for unmapped qubits
    max_qubit = max(all_target_qubits) if all_target_qubits else 0
    if qubit_map:
        max_qubit = max(max_qubit, max(qubit_map.values()))
    new_n_qubits = max_qubit + 1

    # Create new allocation with remapped qubits
    new_allocation = QubitAllocation(
        data_qubits=max(data_qubits) + 1 if data_qubits else 0,
        ancilla_qubits=ancilla_target,
        expr_data_qubits=data_qubits,  # Expression's qubits are now at target positions
    )
    new_circ = Circuit(n_qubits=new_n_qubits, allocation=new_allocation)

    # Copy and remap all gates
    for instruction in circ.qc.data:
        remapped_qubits = []
        for q in instruction.qubits:
            old_index = circ.qc.find_bit(q).index
            # All qubits should be mapped now
            if old_index not in qubit_map:
                raise ValueError(
                    f"Qubit {old_index} not in qubit_map. "
                    f"Used qubits: {used_qubits}, qubit_map keys: {set(qubit_map.keys())}"
                )
            new_index = qubit_map[old_index]
            remapped_qubits.append(new_index)

        # Check for duplicates before appending
        if len(remapped_qubits) != len(set(remapped_qubits)):
            raise ValueError(
                f"Duplicate qubit indices in remapping: {remapped_qubits}, "
                f"qubit_map: {qubit_map}, old_indices: {[circ.qc.find_bit(q).index for q in instruction.qubits]}"
            )

        new_circ.qc.append(instruction.operation, remapped_qubits, instruction.clbits)

    return new_circ


def _compose_circuit(target: Circuit, source: Circuit):
    """Compose source circuit into target circuit.

    Args:
        target: Circuit to compose into
        source: Circuit to compose from
    """
    # Ensure target has enough qubits
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
    """Copy qubit values from source to destination.

    Args:
        circ: Circuit to add gates to
        src_qubits: Source qubit indices
        dst_qubits: Destination qubit indices
    """
    assert len(src_qubits) == len(dst_qubits), (
        "Source and destination must have same length"
    )
    for src, dst in zip(src_qubits, dst_qubits):
        circ.qc.cx(src, dst)


def _init_constant(circ: Circuit, qubits: List[int], value: int):
    """Initialize qubits with a constant value."""
    for i, bit in enumerate(qubits):
        if (value >> i) & 1:
            circ.add("X", bit)
