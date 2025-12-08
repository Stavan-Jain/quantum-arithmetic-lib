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
from .qtype import Mod


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
    """Compile an Add expression to a circuit.

    Strategy:
    1. Get or compile left and right operands (CSE handled automatically)
    2. Determine qubit layout for the operation
    3. Merge circuits with proper qubit remapping
    4. Apply inplace_ripple_adder
    5. Handle modular reduction if needed
    """
    # Get qubits and circuits for operands (CSE: if they share subexpressions,
    # those subexpressions will have the same qubit assignments)
    left_qubits, left_circ = ctx.get_or_compile(expr.left)
    right_qubits, right_circ = ctx.get_or_compile(expr.right)

    n_bits = expr.qtype().n_bits

    # Determine total qubits needed
    # Data: need space for both operands (they may overlap if sharing subexpressions)
    max_data_qubit = max(
        max(left_qubits) if left_qubits else 0,
        max(right_qubits) if right_qubits else 0,
        n_bits - 1,  # Result needs n_bits
    )

    # Ancillas: max of both circuits' ancillas + adder ancillas + flag
    max_ancillas = max(
        len(left_circ.allocation.ancilla_qubits),
        len(right_circ.allocation.ancilla_qubits),
    )
    adder_carry_ancillas = n_bits + 1
    flag_ancilla = 1

    total_ancillas = max_ancillas + adder_carry_ancillas + flag_ancilla
    total_qubits = max_data_qubit + 1 + total_ancillas

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
    # For addition, we need right operand in a separate space initially
    # But if left and right share subexpressions, they already use same qubits
    # So we need to handle the case where they overlap
    right_ancilla_target = list(
        range(max_data_qubit + 1, max_data_qubit + 1 + max_ancillas)
    )

    # Check if left and right share qubits (common subexpressions)
    left_qubits_set = set(left_qubits)
    right_qubits_set = set(right_qubits)

    if left_qubits_set & right_qubits_set:
        # They share qubits - this means they have common subexpressions
        # For addition, we need to copy right to a temporary location first
        # For now, we'll use a simplified approach: assume they can be added in-place
        # if they're the same, or use a temporary if different
        if left_qubits == right_qubits:
            # Same expression: x + x = 2x (can optimize later)
            temp_qubits = left_qubits
        else:
            # Different but overlapping: need temporary space
            temp_start = max_data_qubit + 1 + max_ancillas
            temp_qubits = list(range(temp_start, temp_start + n_bits))
            # Copy right to temp
            _copy_qubits(merged_circ, right_qubits, temp_qubits)
            right_qubits = temp_qubits
    else:
        # No overlap: remap right to its assigned qubits
        right_remapped = _remap_circuit(right_circ, right_ancilla_target, right_qubits)
        _compose_circuit(merged_circ, right_remapped)

    # Apply inplace ripple adder
    # Result will be in left_qubits (in-place addition)
    carry_start = max_data_qubit + 1 + max_ancillas
    carry_qubits = list(range(carry_start, carry_start + adder_carry_ancillas))

    from .adders.inplace_adder import inplace_ripple_adder

    inplace_ripple_adder(
        merged_circ.qc,
        left_qubits,  # First operand (will contain result)
        right_qubits,  # Second operand
        carry_qubits,  # Carry register
    )

    # Apply modular reduction if needed
    ty = expr.qtype()
    if isinstance(ty, Mod):
        modulus = ty.N
        if modulus < (1 << n_bits):
            flag_qubit = carry_start + adder_carry_ancillas
            _compare_const(merged_circ, left_qubits, modulus, flag_qubit)
            _sub_const_if_flag(
                merged_circ, left_qubits, modulus, flag_qubit, carry_qubits[0]
            )

    # Update context allocation
    ctx.allocation = merged_allocation

    return merged_circ


def _compile_sub(expr: Sub, ctx: CompilationContext) -> Circuit:
    """Compile a Sub expression to a circuit."""
    # Similar to Add, but using subtraction logic
    left_qubits, left_circ = ctx.get_or_compile(expr.left)
    right_qubits, right_circ = ctx.get_or_compile(expr.right)

    n_bits = expr.qtype().n_bits
    # TODO: Implement subtraction similar to addition
    # For now, placeholder
    allocation = QubitAllocation(data_qubits=n_bits, ancilla_qubits=[])
    return Circuit(n_qubits=n_bits, allocation=allocation)


def _compile_mul(expr: Mul, ctx: CompilationContext) -> Circuit:
    """Compile a Mul expression to a circuit."""
    # TODO: Implement multiplication
    n_bits = expr.qtype().n_bits
    allocation = QubitAllocation(data_qubits=n_bits, ancilla_qubits=[])
    return Circuit(n_qubits=n_bits, allocation=allocation)


def _compile_pow(expr: Pow, ctx: CompilationContext) -> Circuit:
    """Compile a Pow expression to a circuit."""
    # TODO: Implement exponentiation
    n_bits = expr.qtype().n_bits
    allocation = QubitAllocation(data_qubits=n_bits, ancilla_qubits=[])
    return Circuit(n_qubits=n_bits, allocation=allocation)


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
        remapped_qubits = [qubit_map.get(q.index, q.index) for q in instruction.qubits]
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
        max((q.index for inst in source.qc.data for q in inst.qubits), default=0) + 1,
    )
    if needed_qubits > target.n_qubits:
        from qiskit import QuantumCircuit

        new_qc = QuantumCircuit(needed_qubits)
        for inst in target.qc.data:
            new_qc.append(inst.operation, inst.qubits, inst.clbits)
        target.qc = new_qc
        target.n_qubits = needed_qubits

    # Compose gates
    for instruction in source.qc.data:
        target.qc.append(instruction.operation, instruction.qubits, instruction.clbits)


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
