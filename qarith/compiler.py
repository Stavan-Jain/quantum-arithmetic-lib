from typing import List
from .adders.inplace_adder import inplace_ripple_adder
from .gates import Circuit
from .ir import Program, Block, ModAdd, ModSub, ModExp, QInt
from .arithmetic import compare_const, sub_const_if_flag


def _bits(q: QInt) -> List[int]:
    return q.bits()


def _compile_program(prog: Program) -> Circuit:
    circ = Circuit(n_qubits=prog.n_qubits)
    carry_anc = prog.carry_ancilla_start
    flag_qubit = prog.flag_qubit_index

    for op in prog.ops:
        if isinstance(op, ModAdd):
            _compile_mod_add(circ, op, carry_anc, flag_qubit)
        elif isinstance(op, ModSub):
            _compile_mod_sub(circ, op, carry_anc, flag_qubit)
        elif isinstance(op, ModExp):
            _compile_mod_exp(circ, op, carry_anc, flag_qubit)
        else:
            raise TypeError(f"Unknown op: {op}")
    return circ


def compile(block: Block) -> Circuit:
    prog = block.program
    return _compile_program(prog)


def _compile_mod_add(circ: Circuit, op: ModAdd, carry_anc: int, flag: int):
    """
    Compile ModAdd using the in-place ripple-carry adder from qarith.adders.

    Strategy:
      1. Use inplace_ripple_adder to compute dst = dst + src (mod 2^n)
      2. If modulus < 2^n, check if dst >= modulus and conditionally subtract modulus

    This leverages the high-quality adder implementation while handling modular
    arithmetic correctly.

    Note: This assumes the circuit has enough ancilla qubits allocated.
    The in-place adder needs n+1 carry qubits, which should be allocated
    starting from carry_anc. For simplicity, we reuse the same ancilla space
    for all operations (they are uncomputed after each operation).
    """
    dst_bits = _bits(op.dst)
    src_bits = _bits(op.src)
    n = len(dst_bits)
    modulus = op.modulus

    # Allocate carry ancillas: we need n+1 qubits for the in-place adder
    # We'll use a contiguous block starting from carry_anc.
    # Note: This assumes enough qubits are available. In practice, we might
    # need to allocate more ancillas in Program.n_qubits based on max bit width.
    max_carry_needed = n + 1
    if carry_anc + max_carry_needed > circ.n_qubits:
        raise ValueError(
            f"Insufficient ancilla qubits: need {max_carry_needed} carry qubits "
            f"starting from {carry_anc}, but circuit only has {circ.n_qubits} qubits"
        )
    carry_reg = list(range(carry_anc, carry_anc + n + 1))

    # Step 1: Use in-place ripple adder to compute dst = dst + src (mod 2^n)
    # inplace_ripple_adder expects Qiskit qubit indices, not Circuit abstraction
    qc = circ.quantum_circuit
    inplace_ripple_adder(qc, dst_bits, src_bits, carry_reg)

    # Step 2: If modulus < 2^n, we need modular reduction
    # For now, we use the symbolic comparison/subtract gates from arithmetic.py
    # In a full implementation, these would be expanded into actual circuits.
    if modulus < (1 << n):
        compare_const(circ, dst_bits, modulus, flag)
        sub_const_if_flag(circ, dst_bits, modulus, flag, carry_anc)


# TODO: this is unfinished.
def _compile_mod_sub(circ: Circuit, op: ModSub, carry_anc: int, flag: int):
    """
    Compile ModSub using the in-place ripple-carry adder from qarith.adders.

    Strategy for dst = dst - src (mod N):
      1. Compute dst = dst - src (mod 2^n) using two's complement:
         - Negate src (bitwise NOT)
         - Use inplace_ripple_adder with carry-in=1 to compute dst = dst + (~src + 1) = dst - src
      2. If modulus < 2^n, handle modular reduction (similar to ModAdd)

    This leverages the high-quality adder implementation for subtraction as well.
    """
    dst_bits = _bits(op.dst)
    src_bits = _bits(op.src)
    n = len(dst_bits)
    modulus = op.modulus

    # Allocate carry ancillas: we need n+1 qubits for the in-place adder
    max_carry_needed = n + 1
    if carry_anc + max_carry_needed > circ.n_qubits:
        raise ValueError(
            f"Insufficient ancilla qubits: need {max_carry_needed} carry qubits "
            f"starting from {carry_anc}, but circuit only has {circ.n_qubits} qubits"
        )
    carry_reg = list(range(carry_anc, carry_anc + n + 1))

    qc = circ.quantum_circuit

    # Step 1: Negate src (bitwise NOT) for two's complement subtraction
    # We'll compute dst = dst + (~src + 1) = dst - src (mod 2^n)
    # First, flip all bits of src
    for bit in src_bits:
        qc.x(bit)

    # Step 2: Use in-place ripple adder with carry-in=1 to compute dst = dst + (~src + 1)
    # Set carry-in (c[0]) to 1 for the +1 in two's complement
    qc.x(carry_reg[0])
    inplace_ripple_adder(qc, dst_bits, src_bits, carry_reg)
    qc.x(carry_reg[0])  # Restore carry-in to 0

    # Step 3: Restore src by flipping bits back
    for bit in src_bits:
        qc.x(bit)

    # Step 4: If modulus < 2^n, we need modular reduction
    # For subtraction, if result < 0 (which becomes >= 2^n in unsigned), we add modulus
    # This is more complex than addition; for now we use symbolic gates
    if modulus < (1 << n):
        # Note: Full modular subtraction reduction would need to check for underflow
        # and add modulus if needed. This is left as a symbolic operation for now.
        from .arithmetic import compare_const, sub_const_if_flag

        # Simplified: we assume the result is in [0, 2^n) and handle reduction
        # A full implementation would need to detect underflow and add modulus
        compare_const(circ, dst_bits, modulus, flag)
        sub_const_if_flag(circ, dst_bits, modulus, flag, carry_anc)


# TODO: this is unfinished.
def _compile_mod_exp(circ: Circuit, op: ModExp, carry: int, flag: int):
    """
    dst <- dst * base^exp  (mod N)
    Tiny, symbolic implementation using repeated modular multiplies
    by precomputed constants; actual multiply is a single high-level gate.

    We don't expand into Toffoli-level here to keep it small.
    """
    dst_bits = _bits(op.dst)
    exp_bits = _bits(op.exp)
    base = op.base % op.modulus
    N = op.modulus

    power = base
    for i, qb in enumerate(exp_bits):
        # controlled modular multiplication by 'power'
        circ.add(
            "CMUL_CONST_MOD",
            qb,
            *dst_bits,
            ctrl_index=qb,
            const=power,
            N=N,
        )
        # next square
        power = (power * power) % N
