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
    Basic,
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

    def get_or_compile(self, expr: Expr) -> tuple[List[int], Circuit]:
        """Get qubits and compiled circuit for an expression.

        If already compiled, returns cached qubits and circuit.
        Otherwise, compiles the expression and caches the result.

        Args:
            expr: Expression to compile

        Returns:
            Tuple of (qubit_indices, compiled_circuit)
        """
        # Get or allocate qubits (handles CSE)
        qubits = self.get_or_allocate_qubits(expr)

        # Check if already compiled
        if expr in self.expr_to_circuit:
            return qubits, self.expr_to_circuit[expr]

        # Compile the expression
        circ = _compile_with_context(expr, self)
        self.expr_to_circuit[expr] = circ
        return qubits, circ


def _compile_with_context(expr: Expr, ctx: CompilationContext) -> Circuit:
    """Compile an expression using the given context.

    This is the internal dispatcher that routes to the appropriate
    compilation function based on expression type.
    """
    if isinstance(expr, Basic):
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


def _compile_basic(expr: Basic, ctx: CompilationContext) -> Circuit:
    """Compile a Basic expression to a circuit.

    Basic expressions represent input parameters/oracles. They are allocated
    qubits but don't generate gates (the actual circuit is provided externally).
    """
    qubits = ctx.get_or_allocate_qubits(expr)
    max_qubit = max(qubits) if qubits else 0
    allocation = QubitAllocation(data_qubits=max_qubit + 1, ancilla_qubits=[])
    # Circuit size needs to accommodate all qubits allocated so far
    total_qubits = ctx.allocation.total_qubits()
    return Circuit(n_qubits=total_qubits, allocation=allocation)


def _compile_const(expr: Const, ctx: CompilationContext) -> Circuit:
    """Compile a Const expression to a circuit.

    Constants are initialized to their value using X gates.
    """
    qubits = ctx.get_or_allocate_qubits(expr)
    max_qubit = max(qubits) if qubits else 0
    allocation = QubitAllocation(data_qubits=max_qubit + 1, ancilla_qubits=[])
    total_qubits = ctx.allocation.total_qubits()
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
    """
    # Get qubits and circuits for operands (CSE: if they share subexpressions,
    # those subexpressions will have the same qubit assignments)
    left_qubits, left_circ = ctx.get_or_compile(expr.left)
    right_qubits, right_circ = ctx.get_or_compile(expr.right)

    n_bits = expr.qtype().n_bits
    ty = expr.qtype()

    # Extract modulus (all arithmetic is modular in this language)
    modulus, _ = _get_modulus_and_inner_n_bits(ty)

    # Determine total qubits needed
    max_data_qubit = max(
        max(left_qubits) if left_qubits else 0,
        max(right_qubits) if right_qubits else 0,
        n_bits - 1,  # Result needs n_bits
    )

    # Ancillas: max of both circuits' ancillas
    max_ancillas = max(
        len(left_circ.allocation.ancilla_qubits),
        len(right_circ.allocation.ancilla_qubits),
    )

    # Check if we need temporary space for overlapping operands
    left_qubits_set = set(left_qubits)
    right_qubits_set = set(right_qubits)
    need_temp_space = bool(left_qubits_set & right_qubits_set) and (
        left_qubits != right_qubits
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

    if left_qubits_set & right_qubits_set:
        # They share qubits - need temporary space
        if left_qubits == right_qubits:
            # Same expression: x + x = 2x
            temp_qubits = left_qubits
        else:
            # Different but overlapping: need temporary space
            # Allocate temp space after aux_qubits
            temp_start = aux_start + n_bits + 2
            temp_qubits = list(range(temp_start, temp_start + n_bits))
            # Copy right to temp
            right_remapped = _remap_circuit(
                right_circ, right_ancilla_target, right_qubits
            )
            _compose_circuit(merged_circ, right_remapped)
            _copy_qubits(merged_circ, right_qubits, temp_qubits)
            right_qubits = temp_qubits
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

    if left_is_const and right_is_const:
        # Both constants: initialize result with left, then add right
        _init_constant(merged_circ, result_qubits, expr.left.value)
        # Use quantum-classical to add right constant
        operators.modular_add_quantum_classical(
            merged_circ.qc, result_qubits, expr.right.value, modulus, aux_qubits
        )
        # Result is in aux_qubits[1:n_bits+1]
        _copy_qubits(merged_circ, aux_qubits[1 : n_bits + 1], result_qubits)
    elif left_is_const:
        # Left is constant: copy right to result, then add left constant
        _copy_qubits(merged_circ, right_qubits, result_qubits)
        operators.modular_add_quantum_classical(
            merged_circ.qc, result_qubits, expr.left.value, modulus, aux_qubits
        )
        # Result is in aux_qubits[1:n_bits+1]
        _copy_qubits(merged_circ, aux_qubits[1 : n_bits + 1], result_qubits)
    elif right_is_const:
        # Right is constant: copy left to result, then add right constant
        _copy_qubits(merged_circ, left_qubits, result_qubits)
        operators.modular_add_quantum_classical(
            merged_circ.qc, result_qubits, expr.right.value, modulus, aux_qubits
        )
        # Result is in aux_qubits[1:n_bits+1]
        _copy_qubits(merged_circ, aux_qubits[1 : n_bits + 1], result_qubits)
    else:
        # Both quantum: copy left to result, then add right quantum
        _copy_qubits(merged_circ, left_qubits, result_qubits)
        operators.modular_add_quantum_quantum(
            merged_circ.qc, result_qubits, right_qubits, modulus, aux_qubits
        )
        # Result is in aux_qubits[1:n_bits+1]
        _copy_qubits(merged_circ, aux_qubits[1 : n_bits + 1], result_qubits)

    # Update context to use result_qubits as the output
    ctx.expr_to_qubits[expr] = result_qubits
    ctx.allocation = merged_allocation
    return merged_circ


def _compile_sub(expr: Sub, ctx: CompilationContext) -> Circuit:
    """Compile a Sub expression to a circuit using modular arithmetic.

    Strategy: Use modular subtraction operators from operators.py
    """
    # Get qubits and circuits for operands
    left_qubits, left_circ = ctx.get_or_compile(expr.left)
    right_qubits, right_circ = ctx.get_or_compile(expr.right)

    n_bits = expr.qtype().n_bits
    ty = expr.qtype()

    # Extract modulus
    modulus, _ = _get_modulus_and_inner_n_bits(ty)

    # Determine total qubits needed
    max_data_qubit = max(
        max(left_qubits) if left_qubits else 0,
        max(right_qubits) if right_qubits else 0,
        n_bits - 1,  # Result needs n_bits
    )

    # Ancillas: max of both circuits' ancillas
    max_ancillas = max(
        len(left_circ.allocation.ancilla_qubits),
        len(right_circ.allocation.ancilla_qubits),
    )

    # Check if we need temporary space for overlapping operands
    left_qubits_set = set(left_qubits)
    right_qubits_set = set(right_qubits)
    need_temp_space = bool(left_qubits_set & right_qubits_set) and (
        left_qubits != right_qubits
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

    if left_qubits_set & right_qubits_set:
        # They share qubits - need temporary space
        if left_qubits == right_qubits:
            # Same expression: x - x = 0
            temp_qubits = left_qubits
        else:
            # Different but overlapping: need temporary space
            temp_start = aux_start + n_bits + 2
            temp_qubits = list(range(temp_start, temp_start + n_bits))
            # Copy right to temp
            right_remapped = _remap_circuit(
                right_circ, right_ancilla_target, right_qubits
            )
            _compose_circuit(merged_circ, right_remapped)
            _copy_qubits(merged_circ, right_qubits, temp_qubits)
            right_qubits = temp_qubits
    else:
        # No overlap: remap right to its assigned qubits
        right_remapped = _remap_circuit(right_circ, right_ancilla_target, right_qubits)
        _compose_circuit(merged_circ, right_remapped)

    # Check if operands are constants
    from .expr import Const

    left_is_const = isinstance(expr.left, Const)
    right_is_const = isinstance(expr.right, Const)

    # Copy left operand to result
    _copy_qubits(merged_circ, left_qubits, result_qubits)

    if left_is_const and right_is_const:
        # Both constants: subtract right from left
        operators.modular_subtract_quantum_classical(
            merged_circ.qc, result_qubits, expr.right.value, modulus, aux_qubits
        )
        # Result is in aux_qubits[1:n_bits+1]
        _copy_qubits(merged_circ, aux_qubits[1 : n_bits + 1], result_qubits)
    elif left_is_const:
        # Left is constant: compute left - right
        # First copy left to result, then subtract right
        _init_constant(merged_circ, result_qubits, expr.left.value)
        operators.modular_subtract_quantum_quantum(
            merged_circ.qc, result_qubits, right_qubits, modulus, aux_qubits
        )
        # Result is in aux_qubits[1:n_bits+1]
        _copy_qubits(merged_circ, aux_qubits[1 : n_bits + 1], result_qubits)
    elif right_is_const:
        # Right is constant: subtract right from left
        operators.modular_subtract_quantum_classical(
            merged_circ.qc, result_qubits, expr.right.value, modulus, aux_qubits
        )
        # Result is in aux_qubits[1:n_bits+1]
        _copy_qubits(merged_circ, aux_qubits[1 : n_bits + 1], result_qubits)
    else:
        # Both quantum: use quantum-quantum subtraction
        operators.modular_subtract_quantum_quantum(
            merged_circ.qc, result_qubits, right_qubits, modulus, aux_qubits
        )
        # Result is in aux_qubits[1:n_bits+1]
        _copy_qubits(merged_circ, aux_qubits[1 : n_bits + 1], result_qubits)

    # Update context to use result_qubits as the output
    ctx.expr_to_qubits[expr] = result_qubits
    ctx.allocation = merged_allocation
    return merged_circ


def _compile_mul(expr: Mul, ctx: CompilationContext) -> Circuit:
    """Compile a Mul expression to a circuit using modular arithmetic."""
    # Get qubits and circuits for operands
    left_qubits, left_circ = ctx.get_or_compile(expr.left)
    right_qubits, right_circ = ctx.get_or_compile(expr.right)

    n_bits = expr.qtype().n_bits
    ty = expr.qtype()

    # Extract modulus and inner n_bits
    modulus, inner_n_bits = _get_modulus_and_inner_n_bits(ty)

    # Extract inner qubits from Maybe types (remove flag bits)
    left_qubits = _extract_inner_qubits(left_qubits, expr.left.qtype())
    right_qubits = _extract_inner_qubits(right_qubits, expr.right.qtype())

    # Verify that inner qubits have the same length
    if len(left_qubits) != len(right_qubits):
        raise ValueError(
            f"Operands must have same inner Mod type size for multiplication: "
            f"left has {len(left_qubits)} qubits, right has {len(right_qubits)} qubits, "
            f"expected {inner_n_bits} qubits"
        )

    # Determine total qubits needed
    max_data_qubit = max(
        max(left_qubits) if left_qubits else 0,
        max(right_qubits) if right_qubits else 0,
        n_bits - 1,  # Result needs n_bits
    )

    # Ancillas: max of both circuits' ancillas
    max_ancillas = max(
        len(left_circ.allocation.ancilla_qubits),
        len(right_circ.allocation.ancilla_qubits),
    )

    # Check if we need temporary space for overlapping operands
    left_qubits_set = set(left_qubits)
    right_qubits_set = set(right_qubits)
    need_temp_space = bool(left_qubits_set & right_qubits_set) and (
        left_qubits != right_qubits
    )

    # Allocate aux qubits for modular multiplication
    # Use inner_n_bits for the actual Mod type (not n_bits which may include Maybe flag)
    aux_start = max_data_qubit + 1 + max_ancillas
    aux_qubits = list(range(aux_start, aux_start + inner_n_bits + 2))
    # Result will be in aux_qubits[1:inner_n_bits+1] for the inner Mod type
    # But we need to map it back to result_qubits which may include Maybe flag
    inner_result_qubits = aux_qubits[1 : inner_n_bits + 1]

    # Allocate result qubits (may include Maybe flag bits)
    result_start = aux_start + inner_n_bits + 2
    result_qubits = list(range(result_start, result_start + n_bits))

    # Total qubits needed
    temp_space = inner_n_bits if need_temp_space else 0
    total_qubits = (
        max_data_qubit + 1 + max_ancillas + inner_n_bits + 2 + n_bits + temp_space
    )
    total_ancillas = max_ancillas + inner_n_bits + 2 + n_bits + temp_space

    # Create merged circuit
    merged_allocation = QubitAllocation(
        data_qubits=max_data_qubit + 1,
        ancilla_qubits=list(
            range(max_data_qubit + 1, max_data_qubit + 1 + total_ancillas)
        ),
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

    if left_qubits_set & right_qubits_set:
        if left_qubits == right_qubits:
            temp_qubits = left_qubits
        else:
            temp_start = result_start + n_bits
            temp_qubits = list(range(temp_start, temp_start + inner_n_bits))
            right_remapped = _remap_circuit(
                right_circ, right_ancilla_target, right_qubits
            )
            _compose_circuit(merged_circ, right_remapped)
            _copy_qubits(merged_circ, right_qubits, temp_qubits)
            right_qubits = temp_qubits
    else:
        right_remapped = _remap_circuit(right_circ, right_ancilla_target, right_qubits)
        _compose_circuit(merged_circ, right_remapped)

    # Check if operands are constants
    from .expr import Const

    left_is_const = isinstance(expr.left, Const)
    right_is_const = isinstance(expr.right, Const)

    if left_is_const and right_is_const:
        # Both constants: result is (left * right) mod N
        result_value = (expr.left.value * expr.right.value) % modulus
        _init_constant(merged_circ, result_qubits, result_value)
    elif left_is_const:
        # Left is constant: compute result = left_const * right
        # modular_multiply_quantum_classical: |a⟩ |b⟩ → |a⟩ |b + k*a mod N⟩
        # We want: result = k * right, so initialize aux_qubits[1:n_bits+1] to 0
        # Then use right_qubits as control, k=left_const
        # Result will be in aux_qubits[1:n_bits+1]
        operators.modular_multiply_quantum_classical(
            merged_circ.qc, right_qubits, expr.left.value, modulus, aux_qubits
        )
        # Copy result back to result_qubits (from inner_result_qubits)
        _copy_qubits(merged_circ, inner_result_qubits, result_qubits[:inner_n_bits])
    elif right_is_const:
        # Right is constant: compute result = left * right_const
        # Use left_qubits as control, k=right_const
        operators.modular_multiply_quantum_classical(
            merged_circ.qc, left_qubits, expr.right.value, modulus, aux_qubits
        )
        # Copy result back to result_qubits (from inner_result_qubits)
        _copy_qubits(merged_circ, inner_result_qubits, result_qubits[:inner_n_bits])
    else:
        # Both quantum: use quantum-quantum multiplication
        # modular_multiply_quantum_quantum: |a⟩ |b⟩ |0⟩ → |a⟩ |b⟩ |(a * b) mod N⟩
        # We want: result = left * right
        # So we use left_qubits as a, right_qubits as b, aux_qubits for result
        # Initialize aux_qubits[1:n_bits+1] to 0 (already done)
        operators.modular_multiply_quantum_quantum(
            merged_circ.qc, left_qubits, right_qubits, modulus, aux_qubits
        )
        # Result is in inner_result_qubits
        # Copy to result_qubits (only the inner Mod part, Maybe flag bits remain 0)
        _copy_qubits(merged_circ, inner_result_qubits, result_qubits[:inner_n_bits])

    # Update context to use result_qubits as the output
    ctx.expr_to_qubits[expr] = result_qubits
    ctx.allocation = merged_allocation
    return merged_circ


def _compile_pow(expr: Pow, ctx: CompilationContext) -> Circuit:
    """Compile a Pow expression to a circuit using modular arithmetic."""
    # Get qubits and circuits for operands
    base_qubits, base_circ = ctx.get_or_compile(expr.base)

    # Handle exponent: it can be int or Expr
    if isinstance(expr.exponent, int):
        # Convert int exponent to Const expression
        from .expr import Const

        modulus, _ = _get_modulus_and_inner_n_bits(expr.base.qtype())
        exp_expr = Const(expr.exponent, modulus=modulus)
        exp_qubits, exp_circ = ctx.get_or_compile(exp_expr)
    else:
        exp_qubits, exp_circ = ctx.get_or_compile(expr.exponent)

    n_bits = expr.qtype().n_bits
    ty = expr.qtype()

    # Extract modulus
    modulus, _ = _get_modulus_and_inner_n_bits(ty)

    # Determine total qubits needed
    max_data_qubit = max(
        max(base_qubits) if base_qubits else 0,
        max(exp_qubits) if exp_qubits else 0,
        n_bits - 1,  # Result needs n_bits
    )

    # Ancillas: max of both circuits' ancillas + modular exponentiation ancillas
    max_ancillas = max(
        len(base_circ.allocation.ancilla_qubits),
        len(exp_circ.allocation.ancilla_qubits),
    )
    modular_ancillas = n_bits + 2  # For modular operations

    total_ancillas = max_ancillas + modular_ancillas
    total_qubits = max_data_qubit + 1 + total_ancillas

    # Create merged circuit
    merged_allocation = QubitAllocation(
        data_qubits=max_data_qubit + 1,
        ancilla_qubits=list(
            range(max_data_qubit + 1, max_data_qubit + 1 + total_ancillas)
        ),
    )
    merged_circ = Circuit(n_qubits=total_qubits, allocation=merged_allocation)

    # Remap and compose base circuit
    base_ancilla_target = list(
        range(max_data_qubit + 1, max_data_qubit + 1 + max_ancillas)
    )
    base_remapped = _remap_circuit(base_circ, base_ancilla_target, base_qubits)
    _compose_circuit(merged_circ, base_remapped)

    # Remap and compose exponent circuit
    exp_ancilla_target = list(
        range(max_data_qubit + 1, max_data_qubit + 1 + max_ancillas)
    )
    exp_remapped = _remap_circuit(exp_circ, exp_ancilla_target, exp_qubits)
    _compose_circuit(merged_circ, exp_remapped)

    # Allocate aux qubits for modular exponentiation (n + 2 qubits needed)
    aux_start = max_data_qubit + 1 + max_ancillas
    aux_qubits = list(range(aux_start, aux_start + n_bits + 2))
    # Result will be in aux_qubits[1:n_bits+1] (initialized to |1⟩)
    result_qubits = aux_qubits[1 : n_bits + 1]

    # Initialize result to |1⟩ (required for exponentiation)
    if n_bits > 0:
        merged_circ.qc.x(result_qubits[0])  # Set LSB to 1

    # Check if operands are constants
    from .expr import Const

    base_is_const = isinstance(expr.base, Const)
    exp_is_const = isinstance(expr.exponent, (int, Const))

    if base_is_const and exp_is_const:
        # Both constants: result is (base^exp) mod N
        exp_value = (
            expr.exponent if isinstance(expr.exponent, int) else expr.exponent.value
        )
        result_value = pow(expr.base.value, exp_value, modulus)
        # Reset result qubits
        if n_bits > 0:
            merged_circ.qc.x(result_qubits[0])  # Reset LSB
        _init_constant(merged_circ, result_qubits, result_value)
    elif base_is_const:
        # Base is constant: use quantum-classical exponentiation
        # This requires controlled IMQCM_cir which is not yet fully implemented
        raise NotImplementedError(
            "Modular exponentiation with quantum exponent is not yet implemented"
        )
    elif exp_is_const:
        # Exponent is constant: use quantum-classical exponentiation
        # This requires controlled IMQCM_cir which is not yet fully implemented
        raise NotImplementedError(
            "Modular exponentiation with quantum base is not yet implemented"
        )
    else:
        # Both quantum: use quantum-quantum exponentiation
        # This is not yet implemented in operators.py
        raise NotImplementedError(
            "Quantum-quantum modular exponentiation is not yet implemented"
        )

    # Update context to use result_qubits as the output
    ctx.expr_to_qubits[expr] = result_qubits
    ctx.allocation = merged_allocation
    return merged_circ


def _compile_floordiv(expr: Div, ctx: CompilationContext) -> Circuit:
    """Compile a Div expression to a circuit."""
    # TODO: Implement division
    n_bits = expr.qtype().n_bits
    allocation = QubitAllocation(data_qubits=n_bits, ancilla_qubits=[])
    return Circuit(n_qubits=n_bits, allocation=allocation)


def _compile_sqrt(expr: Sqrt, ctx: CompilationContext) -> Circuit:
    """Compile a Sqrt expression to a circuit."""
    # TODO: Implement square root
    n_bits = expr.qtype().n_bits
    allocation = QubitAllocation(data_qubits=n_bits, ancilla_qubits=[])
    return Circuit(n_qubits=n_bits, allocation=allocation)


def _compile_inv(expr: Inv, ctx: CompilationContext) -> Circuit:
    """Compile an Inv expression to a circuit."""
    # TODO: Implement modular inverse
    n_bits = expr.qtype().n_bits
    allocation = QubitAllocation(data_qubits=n_bits, ancilla_qubits=[])
    return Circuit(n_qubits=n_bits, allocation=allocation)


def _compile_log(expr: Log, ctx: CompilationContext) -> Circuit:
    """Compile a Log expression to a circuit."""
    # TODO: Implement logarithm
    n_bits = expr.qtype().n_bits
    allocation = QubitAllocation(data_qubits=n_bits, ancilla_qubits=[])
    return Circuit(n_qubits=n_bits, allocation=allocation)


def _compile_indexed(expr: Indexed, ctx: CompilationContext) -> Circuit:
    """Compile an Indexed expression to a circuit."""
    # TODO: Implement indexed operation
    n_bits = expr.qtype().n_bits
    allocation = QubitAllocation(data_qubits=n_bits, ancilla_qubits=[])
    return Circuit(n_qubits=n_bits, allocation=allocation)


def _compile_join(expr: Join, ctx: CompilationContext) -> Circuit:
    """Compile a Join expression to a circuit."""
    # TODO: Implement join operation
    n_bits = expr.qtype().n_bits
    allocation = QubitAllocation(data_qubits=n_bits, ancilla_qubits=[])
    return Circuit(n_qubits=n_bits, allocation=allocation)


# Helper functions


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


def _extract_inner_qubits(qubits: List[int], expr_type: QType) -> List[int]:
    """
    Extract the inner Mod type qubits from a Maybe type qubit list.

    For Maybe types, the last qubit is the flag bit. This function removes
    all flag bits to get the inner Mod type qubits.

    Args:
        qubits: List of qubit indices for the expression
        expr_type: The type of the expression

    Returns:
        List of qubit indices for the inner Mod type (without flag bits)

    Examples:
        _extract_inner_qubits([0, 1, 2, 3], Mod(11)) -> [0, 1, 2, 3]
        _extract_inner_qubits([0, 1, 2, 3, 4], Maybe(Mod(11))) -> [0, 1, 2, 3]
    """
    if isinstance(expr_type, Mod):
        return qubits
    elif isinstance(expr_type, Maybe):
        # Remove the last qubit (flag bit) and recurse
        inner_qubits = qubits[:-1]
        return _extract_inner_qubits(inner_qubits, expr_type.inner_type)
    else:
        raise ValueError(f"Cannot extract qubits from type {expr_type}")


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
        data_qubits: Target qubit indices for data qubits

    Returns:
        New circuit with remapped qubits
    """
    qubit_map: Dict[int, int] = {}

    # Map data qubits
    for i, old_qubit in enumerate(range(circ.allocation.data_qubits)):
        if i < len(data_qubits):
            qubit_map[old_qubit] = data_qubits[i]

    # Map ancilla qubits
    for i, old_ancilla in enumerate(circ.allocation.ancilla_qubits):
        if i < len(ancilla_target):
            qubit_map[old_ancilla] = ancilla_target[i]

    # Create new circuit with remapped gates
    new_n_qubits = max(qubit_map.values()) + 1 if qubit_map else 0
    new_circ = Circuit(n_qubits=new_n_qubits, allocation=circ.allocation)

    # Copy and remap all gates
    for instruction in circ.qc.data:
        remapped_qubits = [
            qubit_map.get(circ.qc.find_bit(q).index, circ.qc.find_bit(q).index)
            for q in instruction.qubits
        ]
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

    # Compose gates - map source qubits to target qubits
    for instruction in source.qc.data:
        # Map source qubits to target qubits by index
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


def _compare_const(circ: Circuit, x_bits: List[int], N: int, flag_qubit: int):
    """Compare a quantum register with a constant."""
    circ.add("CMP_CONST", *x_bits, flag_qubit, N=N)


def _sub_const_if_flag(
    circ: Circuit,
    x_bits: List[int],
    N: int,
    flag_qubit: int,
    carry_anc: int,
):
    """Subtract a constant from a register if a flag qubit is set."""
    circ.add("SUB_CONST_IF", *x_bits, flag_qubit, N=N, carry_anc=carry_anc)
