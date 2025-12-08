"""
Simulator for quantum arithmetic circuits.

This module provides functionality to run compiled quantum circuits,
handling input initialization, ancilla postselection, and result extraction.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from typing import Dict, Union, Tuple
from .expr import Expr, Basic
from .compile import CompilationContext, _compile_with_context
from .qtype import QType, Mod, Maybe


def run(
    expr: Expr,
    inputs: Dict[str, int],
    postselect_ancillas: bool = True,
) -> Union[int, Tuple[int, bool], np.ndarray]:
    """
    Run a quantum arithmetic expression with given inputs.

    Args:
        expr: The expression to evaluate
        inputs: Dictionary mapping Basic expression names to their integer values
        postselect_ancillas: If True, postselect ancilla qubits to |0⟩ (default: True)

    Returns:
        - For Mod types: integer result value
        - For Maybe types: tuple of (value, success_flag) where success_flag indicates
          whether the operation succeeded (flag bit = 0 means success)
        - For complex expressions: numpy array of the final state vector

    Examples:
        >>> x1 = Basic("x1", Mod(11))
        >>> x2 = Basic("x2", Mod(11))
        >>> sum_expr = x1 + x2
        >>> result = run(sum_expr, {"x1": 5, "x2": 3})
        >>> print(result)  # 8 (since (5 + 3) mod 11 = 8)
    """
    # Compile the expression to get circuit and context
    ctx = CompilationContext()
    circuit = _compile_with_context(expr, ctx)

    # Get qubit allocation
    allocation = circuit.allocation
    data_qubits = allocation.data_qubits
    ancilla_qubits = allocation.ancilla_qubits
    total_qubits = allocation.total_qubits()

    # Initialize input state
    initial_state = _initialize_inputs(expr, inputs, ctx, data_qubits, ancilla_qubits)

    # Run the circuit
    final_state_full = _execute_circuit(circuit.qc, initial_state, total_qubits)

    # Extract result qubits before postselection (they might be outside data_qubits range)
    result_qubits = ctx.expr_to_qubits.get(expr)

    if result_qubits is None:
        # If expression wasn't in context, it might be a simple Basic or Const
        # Postselect if requested and return
        if postselect_ancillas and ancilla_qubits:
            final_state = _postselect_ancillas(
                final_state_full, data_qubits, ancilla_qubits, total_qubits
            )
            return final_state
        return final_state_full

    # Check if result qubits are within data qubits range
    max_result_qubit = max(result_qubits) if result_qubits else 0
    result_in_data_range = max_result_qubit < data_qubits

    # Postselect ancillas if requested
    if postselect_ancillas and ancilla_qubits:
        # Check if any result qubits overlap with ancilla qubits
        result_qubits_set = set(result_qubits)
        ancilla_qubits_set = set(ancilla_qubits)
        result_overlaps_ancillas = bool(result_qubits_set & ancilla_qubits_set)

        if result_in_data_range and not result_overlaps_ancillas:
            # Result is in data qubits and doesn't overlap with ancillas, can postselect
            final_state = _postselect_ancillas(
                final_state_full, data_qubits, ancilla_qubits, total_qubits, None
            )
            return _extract_result(
                final_state, expr.qtype(), result_qubits, total_qubits, True
            )
        elif not result_overlaps_ancillas:
            # Result qubits are outside data range but don't overlap with ancillas
            # Postselect ancillas while preserving result qubits
            final_state = _postselect_ancillas(
                final_state_full,
                data_qubits,
                ancilla_qubits,
                total_qubits,
                result_qubits,
            )
            # Extract from postselected state (which now includes result qubits)
            max_result_qubit = max(result_qubits) if result_qubits else 0
            output_qubits = max(data_qubits, max_result_qubit + 1)
            return _extract_result(
                final_state, expr.qtype(), result_qubits, output_qubits, False
            )
        else:
            # Result qubits are outside data range or overlap with ancillas
            # Need to postselect while preserving result qubits
            # Also need to preserve aux_qubits[1:n_bits+1] which contain the result
            # before it's copied to result_qubits
            n_bits = len(result_qubits)
            aux_start = max(result_qubits) + 1
            aux_qubits = list(range(aux_start, aux_start + n_bits + 2))
            aux_result_qubits = aux_qubits[1 : n_bits + 1]  # Result is stored here
            preserve_qubits_set = result_qubits_set | set(aux_result_qubits)

            # Create a new ancilla list excluding result qubits and aux_result_qubits
            ancillas_to_postselect = [
                a for a in ancilla_qubits if a not in preserve_qubits_set
            ]
            if ancillas_to_postselect:
                # Postselect only non-result ancillas
                # Extract result from aux_qubits[1:n_bits+1] where the result is actually stored
                # (before it's copied to result_qubits)
                return _extract_result_with_postselection(
                    final_state_full,
                    expr.qtype(),
                    aux_result_qubits,  # Use aux_result_qubits instead of result_qubits
                    ancillas_to_postselect,
                    total_qubits,
                )
            else:
                # All ancillas are result qubits, no postselection needed
                return _extract_result(
                    final_state_full, expr.qtype(), result_qubits, total_qubits, False
                )
    else:
        # No postselection
        return _extract_result(
            final_state_full, expr.qtype(), result_qubits, total_qubits, False
        )


def _initialize_inputs(
    expr: Expr,
    inputs: Dict[str, int],
    ctx: CompilationContext,
    data_qubits: int,
    ancilla_qubits: list[int],
) -> np.ndarray:
    """
    Initialize the quantum state with input values.

    Args:
        expr: The expression being evaluated
        inputs: Dictionary mapping Basic names to values
        ctx: Compilation context containing qubit mappings
        data_qubits: Number of data qubits
        ancilla_qubits: List of ancilla qubit indices
        total_qubits: Total number of qubits

    Returns:
        Initial state vector as a numpy array
    """
    total_qubits = (
        max(data_qubits, max(ancilla_qubits) + 1) if ancilla_qubits else data_qubits
    )

    # Collect all Basic expressions and their values
    basic_values = {}  # Map from Basic expr to (qubits, value, qtype)

    def collect_basic(basic_expr: Basic):
        if basic_expr.name not in inputs:
            raise ValueError(
                f"Missing input value for Basic expression '{basic_expr.name}'. "
                f"Provided inputs: {list(inputs.keys())}"
            )

        value = inputs[basic_expr.name]
        qubits = ctx.expr_to_qubits.get(basic_expr)

        if qubits is None:
            # Basic wasn't compiled yet, need to allocate it
            qubits = ctx.get_or_allocate_qubits(basic_expr)

        basic_values[basic_expr] = (qubits, value, basic_expr.qtype())

    # Traverse expression tree to find all Basic inputs
    _traverse_and_initialize(expr, collect_basic, ctx)

    # Build initial state: find the basis state where all Basic qubits are set correctly
    initial_state = np.zeros(2**total_qubits, dtype=complex)
    target_basis_idx = 0

    for basic_expr, (qubits, value, qtype) in basic_values.items():
        # Set bits for this Basic expression
        # Qiskit uses little-endian: qubit 0 is LSB, so index = sum(bit_i * 2^i)
        if isinstance(qtype, Mod):
            encoded_value = value % qtype.N
            for i, qubit_idx in enumerate(qubits):
                bit_value = (encoded_value >> i) & 1
                if bit_value:
                    target_basis_idx |= 1 << qubit_idx
        elif isinstance(qtype, Maybe):
            inner_type = qtype.inner_type
            if isinstance(inner_type, Mod):
                encoded_value = value % inner_type.N
                inner_qubits = qubits[:-1]
                for i, qubit_idx in enumerate(inner_qubits):
                    bit_value = (encoded_value >> i) & 1
                    if bit_value:
                        target_basis_idx |= 1 << qubit_idx
                # Flag bit stays 0 (already set)
    initial_state[target_basis_idx] = 1.0
    return initial_state


def _traverse_and_initialize(expr: Expr, init_func, ctx: CompilationContext):
    """Traverse expression tree and collect/initialize Basic expressions."""
    if isinstance(expr, Basic):
        init_func(expr)
    elif hasattr(expr, "left") and hasattr(expr, "right"):
        # Binary operations
        _traverse_and_initialize(expr.left, init_func, ctx)
        _traverse_and_initialize(expr.right, init_func, ctx)
    elif hasattr(expr, "base") and hasattr(expr, "exponent"):
        # Pow operation
        _traverse_and_initialize(expr.base, init_func, ctx)
        if isinstance(expr.exponent, Expr):
            _traverse_and_initialize(expr.exponent, init_func, ctx)
    elif hasattr(expr, "expr"):
        # Unary operations (Sqrt, Inv, etc.)
        _traverse_and_initialize(expr.expr, init_func, ctx)
    # Const expressions don't need initialization (they're compiled into gates)


def _execute_circuit(
    qc: QuantumCircuit, initial_state: np.ndarray, total_qubits: int
) -> np.ndarray:
    """
    Execute a quantum circuit on an initial state.

    Args:
        qc: Qiskit QuantumCircuit
        initial_state: Initial state vector
        total_qubits: Total number of qubits

    Returns:
        Final state vector after applying the circuit
    """
    # Create Statevector from initial state
    statevector = Statevector(initial_state)

    # Evolve by the circuit
    statevector = statevector.evolve(qc)

    return statevector.data


def _postselect_ancillas(
    final_state: np.ndarray,
    data_qubits: int,
    ancilla_qubits: list[int],
    total_qubits: int,
    preserve_qubits: list[int] | None = None,
) -> np.ndarray:
    """
    Postselect ancilla qubits to |0⟩.

    Args:
        final_state: Full final state vector
        data_qubits: Number of data qubits
        ancilla_qubits: List of ancilla qubit indices to postselect
        total_qubits: Total number of qubits
        preserve_qubits: Optional list of qubit indices to preserve (e.g., result qubits)

    Returns:
        Postselected state vector. If preserve_qubits is None, returns data qubits only.
        Otherwise, returns state with data qubits + preserved qubits.
    """
    if not ancilla_qubits:
        # No ancillas to postselect
        return final_state

    # Determine which qubits to preserve
    preserve_set = set(preserve_qubits) if preserve_qubits else set()
    # Ancillas to actually postselect (exclude preserved qubits)
    ancillas_to_postselect = [a for a in ancilla_qubits if a not in preserve_set]

    if not ancillas_to_postselect:
        # All ancillas are preserved, no postselection
        return final_state

    # Calculate output dimension
    if preserve_qubits:
        # Include preserved qubits in output
        max_preserved = max(preserve_qubits) if preserve_qubits else -1
        output_qubits = max(data_qubits, max_preserved + 1)
    else:
        # Only data qubits
        output_qubits = data_qubits

    output_dim = 2**output_qubits
    postselected = np.zeros(output_dim, dtype=complex)

    # Iterate over all basis states in the full state
    for full_idx in range(len(final_state)):
        # Check if all ancillas to postselect are 0 in this basis state
        all_ancillas_zero = True
        for ancilla_idx in ancillas_to_postselect:
            # Qiskit uses little-endian: qubit_idx corresponds to bit qubit_idx in the index
            bit_value = (full_idx >> ancilla_idx) & 1
            if bit_value != 0:
                all_ancillas_zero = False
                break

        if all_ancillas_zero:
            # Extract output qubit values
            # Qiskit uses little-endian: qubit i corresponds to bit i in the index
            output_idx = 0
            for i in range(output_qubits):
                bit_value = (full_idx >> i) & 1
                output_idx |= bit_value << i

            # Add amplitude to postselected state
            postselected[output_idx] += final_state[full_idx]

    # Normalize
    norm = np.linalg.norm(postselected)
    if norm < 1e-15:
        raise ValueError(
            "Postselection resulted in zero state. "
            "This may indicate an error in the circuit or invalid inputs."
        )

    return postselected / norm


def _extract_result_with_postselection(
    final_state_full: np.ndarray,
    qtype: QType,
    result_qubits: list[int],
    ancillas_to_postselect: list[int],
    total_qubits: int,
) -> Union[int, Tuple[int, bool]]:
    """
    Extract result while postselecting ancillas that don't overlap with result qubits.

    This handles the case where result qubits are outside the data qubits range.
    """
    # Find the most probable basis state where ancillas are 0
    max_prob = 0.0
    best_idx = 0

    for idx in range(len(final_state_full)):
        # Check if all ancillas to postselect are 0
        all_ancillas_zero = True
        for ancilla_idx in ancillas_to_postselect:
            # Qiskit uses little-endian: qubit_idx corresponds to bit qubit_idx in the index
            bit_value = (idx >> ancilla_idx) & 1
            if bit_value != 0:
                all_ancillas_zero = False
                break

        if all_ancillas_zero:
            prob = abs(final_state_full[idx]) ** 2
            if prob > max_prob:
                max_prob = prob
                best_idx = idx

    # Extract result from the best basis state
    if isinstance(qtype, Mod):
        value = 0
        for i, qubit_idx in enumerate(result_qubits):
            # Qiskit uses little-endian: qubit_idx corresponds to bit qubit_idx in the index
            bit_value = (best_idx >> qubit_idx) & 1
            value |= bit_value << i
        return value % qtype.N
    elif isinstance(qtype, Maybe):
        inner_type = qtype.inner_type
        if isinstance(inner_type, Mod):
            inner_qubits = result_qubits[:-1]
            flag_qubit = result_qubits[-1]

            inner_value = 0
            for i, qubit_idx in enumerate(inner_qubits):
                # Qiskit uses little-endian: qubit_idx corresponds to bit qubit_idx in the index
                bit_value = (best_idx >> qubit_idx) & 1
                inner_value |= bit_value << i
            inner_value = inner_value % inner_type.N

            flag_value = (best_idx >> flag_qubit) & 1
            success = flag_value == 0
            return (inner_value, success)
        else:
            raise NotImplementedError(
                f"Maybe type with inner type {inner_type} not supported"
            )
    else:
        raise ValueError(f"Unsupported result type: {qtype}")


def _extract_result(
    final_state: np.ndarray,
    qtype: QType,
    result_qubits: list[int],
    total_qubits: int,
    is_postselected: bool,
) -> Union[int, Tuple[int, bool]]:
    """
    Extract the result value from the final state vector.

    Args:
        final_state: Final state vector (may be postselected, so dimension = 2^data_qubits)
        qtype: Type of the result expression
        result_qubits: List of qubit indices for the result
        total_qubits: Total number of qubits in the original circuit
        is_postselected: Whether the state has been postselected

    Returns:
        - For Mod types: integer value
        - For Maybe types: tuple of (value, success_flag)
    """

    if isinstance(qtype, Mod):
        # Extract value from result qubits
        value = _extract_value_from_qubits(
            final_state, result_qubits, qtype.n_bits, total_qubits, is_postselected
        )
        return value % qtype.N

    elif isinstance(qtype, Maybe):
        # Extract inner value and flag bit
        inner_type = qtype.inner_type
        inner_qubits = result_qubits[:-1]
        flag_qubit = result_qubits[-1]

        if isinstance(inner_type, Mod):
            inner_value = _extract_value_from_qubits(
                final_state,
                inner_qubits,
                inner_type.n_bits,
                total_qubits,
                is_postselected,
            )
            inner_value = inner_value % inner_type.N

            # Extract flag bit
            flag_value = _extract_flag_bit(
                final_state, flag_qubit, total_qubits, is_postselected
            )

            # Flag = 0 means success
            success = flag_value == 0
            return (inner_value, success)
        else:
            raise NotImplementedError(
                f"Maybe type with inner type {inner_type} not supported"
            )

    else:
        raise ValueError(f"Unsupported result type: {qtype}")


def _extract_value_from_qubits(
    state: np.ndarray,
    qubits: list[int],
    n_bits: int,
    total_qubits: int,
    is_postselected: bool,
) -> int:
    """
    Extract integer value from qubits in the state vector.

    For postselected states, we assume qubits are in positions 0 to state_dim-1.
    """
    # Find the most probable basis state
    state_dim = len(state)
    state_qubits = int(np.log2(state_dim))

    prob_state = np.argmax(np.abs(state) ** 2)
    value = 0

    for i, qubit_idx in enumerate(qubits):
        if qubit_idx < state_qubits:
            # Qubit is in the state vector
            # Qiskit uses little-endian: qubit_idx corresponds to bit qubit_idx in the index
            bit_value = (prob_state >> qubit_idx) & 1
            value |= bit_value << i
        # If qubit_idx >= state_qubits, it's outside the state vector and should be 0

    return value


def _extract_flag_bit(
    state: np.ndarray,
    flag_qubit: int,
    total_qubits: int,
    is_postselected: bool,
) -> int:
    """
    Extract flag bit value from the state vector.

    For postselected states, if flag is an ancilla it should be 0, but if it's
    part of the data qubits (Maybe type), we extract it normally.
    """
    if is_postselected:
        # After postselection
        data_qubits = int(np.log2(len(state)))
        if flag_qubit < data_qubits:
            # Flag is part of data qubits (Maybe type)
            # Qiskit uses little-endian: qubit_idx corresponds to bit qubit_idx in the index
            prob_state = np.argmax(np.abs(state) ** 2)
            return (prob_state >> flag_qubit) & 1
        else:
            # Flag is an ancilla, should be 0 after postselection
            return 0
    else:
        # Full state
        # Qiskit uses little-endian: qubit_idx corresponds to bit qubit_idx in the index
        prob_idx = np.argmax(np.abs(state) ** 2)
        return (prob_idx >> flag_qubit) & 1
