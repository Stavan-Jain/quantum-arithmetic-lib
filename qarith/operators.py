"""
Modular arithmetic operations for quantum circuits.

This module implements modular addition, subtraction, multiplication, and
exponentiation using QFT-based approaches as described in the notebook.
"""

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
import numpy as np
from typing import List
from math import gcd


def mod_inverse(k: int, N: int) -> int:
    """
    Compute modular inverse of k modulo N using extended Euclidean algorithm.

    Args:
        k: Integer to find inverse of
        N: Modulus

    Returns:
        Modular inverse k^(-1) mod N

    Raises:
        ValueError: If k is 0 or not invertible modulo N
    """
    if k == 0:
        raise ValueError("0 has no modular inverse")
    t, new_t = 0, 1
    r, new_r = N, k % N
    while new_r != 0:
        quotient = r // new_r
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r
    if r > 1:
        raise ValueError(f"{k} is not invertible modulo {N}")
    if t < 0:
        t = t + N
    return t


def precompute_powers(a: int, n_bits: int, N: int) -> List[int]:
    """
    Precompute a^(2^i) mod N for i = 0, 1, ..., n_bits-1.

    This is used for modular exponentiation where the base is classical.
    The powers are computed using repeated squaring.

    Args:
        a: Base (classical)
        n_bits: Number of bits in exponent
        N: Modulus

    Returns:
        List where powers[i] = a^(2^i) mod N
    """
    powers = []
    current = a % N
    for i in range(n_bits):
        powers.append(current)
        current = (current * current) % N  # Square for next power
    return powers


def modular_addin_circuit(n: int, k: int, N: int) -> QuantumCircuit:
    """
    Inplace modular quantum-classical adder circuit.

    Computes: |a⟩ |0⟩ |0⟩ → |a⟩ |(a + k) mod N⟩ |aux⟩

    This is the fundamental building block for modular arithmetic operations.
    Based on QFT-based addition with modular reduction.

    Args:
        n: Number of bits for the register
        k: Classical constant to add
        N: Modulus

    Returns:
        QuantumCircuit implementing the modular addition
    """
    sumbit = n + 1
    qc = QuantumCircuit(n + 2, name=f"Addin({k}, {N})")
    sumN = QuantumCircuit(n + 1, name="Sum(N)")
    aux_q2 = n + 1

    # Part 1: QFT, rotations for Sum(k) and Sum(-N), QFT inverse, CNOT
    qft1 = QFTGate(n + 1)
    qft1.do_swaps = True
    qc.append(qft1, range(1, n + 2))

    for i in range(n + 1):
        angle = (2 * np.pi * k) / (2 ** (sumbit - i))
        qc.p(angle, i + 1)

    for i in range(n + 1):
        angle = (2 * np.pi * (-N)) / (2 ** (sumbit - i))
        qc.p(angle, i + 1)

    qft1_inv = QFTGate(n + 1)
    qft1_inv.do_swaps = True
    qc.append(qft1_inv.inverse(), range(1, n + 2))
    qc.cx(aux_q2, 0)

    # Part 2: QFT, controlled Sum(N), Sum(-k), QFT inverse
    qft2 = QFTGate(n + 1)
    qft2.do_swaps = True
    qc.append(qft2, range(1, n + 2))

    for i in range(n + 1):
        angle = (2 * np.pi * N) / (2 ** (sumbit - i))
        sumN.p(angle, i)

    qc.append(sumN.to_gate().control(1), [0] + list(range(1, n + 2)))

    for i in range(n + 1):
        angle = (2 * np.pi * (-k)) / (2 ** (sumbit - i))
        qc.p(angle, i + 1)

    qft2_inv = QFTGate(n + 1)
    qft2_inv.do_swaps = True
    qc.append(qft2_inv.inverse(), range(1, n + 2))

    # Part 3: CNOT with control state 0, QFT, Sum(k), QFT inverse
    qc.cx(aux_q2, 0, ctrl_state=0)

    qft3 = QFTGate(n + 1)
    qft3.do_swaps = True
    qc.append(qft3, range(1, n + 2))

    for i in range(n + 1):
        angle = (2 * np.pi * k) / (2 ** (sumbit - i))
        qc.p(angle, i + 1)

    qft3_inv = QFTGate(n + 1)
    qft3_inv.do_swaps = True
    qc.append(qft3_inv.inverse(), range(1, n + 2))

    return qc


def modular_add_quantum_classical(
    circ: QuantumCircuit, a_qubits: List[int], k: int, N: int, aux_qubits: List[int]
) -> None:
    """
    Outplace modular quantum-classical adder.

    Computes: |a⟩ |0⟩ → |a⟩ |(a + k) mod N⟩

    Args:
        circ: Quantum circuit to add gates to
        a_qubits: Qubit indices for register a (n qubits)
        k: Classical constant to add
        N: Modulus
        aux_qubits: Auxiliary qubits (n + 2 qubits needed)
    """
    n = len(a_qubits)
    assert len(aux_qubits) >= n + 2, f"Need at least {n + 2} auxiliary qubits"

    # Apply controlled modular additions for each bit
    for i in range(n):
        tempk = 2 ** (n - (i + 1))
        addin_circ = modular_addin_circuit(n, tempk, N)
        addin = addin_circ.to_gate()
        addin.name = f"Addin({tempk}, {N})"
        addin_controlled = addin.control(1)
        addin_controlled.name = f"C-Addin({tempk}, {N})"
        # Control from a_qubits[i], target aux_qubits
        control_qubit = a_qubits[n - (i + 1)]
        circ.append(addin_controlled, [control_qubit] + aux_qubits[:])

    # Final addition with k
    addin_kN_circ = modular_addin_circuit(n, k, N)
    addin_kN = addin_kN_circ.to_gate()
    addin_kN.name = f"Sum({k}, {N})"
    circ.append(addin_kN, aux_qubits[:])


def modular_add_quantum_quantum(
    circ: QuantumCircuit,
    a_qubits: List[int],
    b_qubits: List[int],
    N: int,
    aux_qubits: List[int],
) -> None:
    """
    Outplace modular quantum-quantum adder.

    Computes: |a⟩ |b⟩ |0⟩ → |a⟩ |b⟩ |(a + b) mod N⟩

    Args:
        circ: Quantum circuit to add gates to
        a_qubits: Qubit indices for register a (n qubits)
        b_qubits: Qubit indices for register b (n qubits)
        N: Modulus
        aux_qubits: Auxiliary qubits (n + 2 qubits needed)
    """
    n = len(a_qubits)
    assert len(b_qubits) == n, "a and b must have same length"
    assert len(aux_qubits) >= n + 2, f"Need at least {n + 2} auxiliary qubits"

    # Apply controlled modular additions for each bit of a
    for i in range(n):
        tempk = 2 ** (n - (i + 1))
        addin_circ = modular_addin_circuit(n, tempk, N)
        addin = addin_circ.to_gate()
        addin.name = f"Addin({tempk}, {N})"
        addin_controlled = addin.control(1)
        addin_controlled.name = f"C-Addin({tempk}, {N})"
        circ.append(addin_controlled, [a_qubits[n - (i + 1)]] + aux_qubits[:])

    # Apply controlled modular additions for each bit of b
    for i in range(n):
        tempk = 2 ** (n - (i + 1))
        addin_circ = modular_addin_circuit(n, tempk, N)
        addin = addin_circ.to_gate()
        addin.name = f"Addin({tempk}, {N})"
        addin_controlled = addin.control(1)
        addin_controlled.name = f"C-Addin({tempk}, {N})"
        circ.append(addin_controlled, [b_qubits[n - (i + 1)]] + aux_qubits[:])


def modular_subtract_quantum_classical(
    circ: QuantumCircuit, a_qubits: List[int], k: int, N: int, aux_qubits: List[int]
) -> None:
    """
    Modular quantum-classical subtractor.

    Computes: |a⟩ |0⟩ → |a⟩ |(a - k) mod N⟩

    Uses: a - k mod N = a + (N - k) mod N

    Args:
        circ: Quantum circuit to add gates to
        a_qubits: Qubit indices for register a (n qubits)
        k: Classical constant to subtract
        N: Modulus
        aux_qubits: Auxiliary qubits (n + 2 qubits needed)
    """
    # Compute (N - k) mod N
    k_neg = (N - k) % N
    modular_add_quantum_classical(circ, a_qubits, k_neg, N, aux_qubits)


def modular_subtract_quantum_quantum(
    circ: QuantumCircuit,
    a_qubits: List[int],
    b_qubits: List[int],
    N: int,
    aux_qubits: List[int],
) -> None:
    """
    Modular quantum-quantum subtractor.

    Computes: |a⟩ |b⟩ |0⟩ → |a⟩ |b⟩ |(a - b) mod N⟩

    Uses: a - b mod N = a + (N - b) mod N

    Strategy:
    1. First copy a to aux_qubits (compute a + 0 mod N)
    2. Then for each bit i of b, if b[i] = 1, subtract 2^i by adding (N - 2^i) mod N

    Args:
        circ: Quantum circuit to add gates to
        a_qubits: Qubit indices for register a (n qubits)
        b_qubits: Qubit indices for register b (n qubits)
        N: Modulus
        aux_qubits: Auxiliary qubits (n + 2 qubits needed)
    """
    n = len(a_qubits)
    assert len(b_qubits) == n, "a and b must have same length"
    assert len(aux_qubits) >= n + 2, f"Need at least {n + 2} auxiliary qubits"

    # First copy a to aux_qubits by computing a + 0 mod N
    # This uses modular_add_quantum_classical with k=0
    modular_add_quantum_classical(circ, a_qubits, 0, N, aux_qubits)

    # Now subtract b: for each bit i of b, if b[i] = 1, subtract 2^i
    # We do this by adding (N - 2^i) mod N
    for i in range(n):
        power_of_2 = 2 ** (n - (i + 1))  # 2^(n-i-1) for MSB to LSB
        tempk = (N - power_of_2) % N
        addin_circ = modular_addin_circuit(n, tempk, N)
        addin = addin_circ.to_gate()
        addin.name = f"Addin({tempk}, {N})"
        addin_controlled = addin.control(1)
        addin_controlled.name = f"C-Sub({power_of_2}, {N})"
        control_qubit = b_qubits[n - (i + 1)]
        circ.append(addin_controlled, [control_qubit] + aux_qubits[:])


def modular_multiply_quantum_classical(
    circ: QuantumCircuit, a_qubits: List[int], k: int, N: int, aux_qubits: List[int]
) -> None:
    """
    Outplace modular quantum-classical multiplier (multout_cir).

    Computes: |a⟩ |b⟩ → |a⟩ |b + k*a mod N⟩

    From notebook: For each bit i of a, if a[i] = 1, add k * 2^i mod N to b

    Args:
        circ: Quantum circuit to add gates to
        a_qubits: Qubit indices for register a (n qubits)
        k: Classical constant to multiply by
        N: Modulus
        aux_qubits: Auxiliary qubits (n + 2 qubits needed), contains b initially
    """
    n = len(a_qubits)
    assert len(aux_qubits) >= n + 2, f"Need at least {n + 2} auxiliary qubits"

    # For each bit position i of a, if a[i] = 1, add k * 2^i mod N to aux_qubits
    # From notebook: multout_cir uses a_qubits as control, aux_qubits as target
    # The control qubit is a_qubits[n - (i + 1)] (MSB to LSB)

    for i in range(n):
        tempk = ((2 ** (n - (i + 1))) * k) % N
        addin_circ = modular_addin_circuit(n, tempk, N)
        addin = addin_circ.to_gate()
        addin.name = f"Addin({tempk}, {N})"
        addin_controlled = addin.control(1)
        addin_controlled.name = f"C-Multout({tempk}, {N})"
        # Control from a_qubits[n - (i + 1)], target aux_qubits
        # From notebook: control qubit is totl_qubits - i
        control_qubit = a_qubits[n - (i + 1)]
        circ.append(addin_controlled, [control_qubit] + aux_qubits[:])


def modular_multiply_quantum_quantum(
    circ: QuantumCircuit,
    a_qubits: List[int],
    b_qubits: List[int],
    N: int,
    aux_qubits: List[int],
) -> None:
    """
    Outplace modular quantum-quantum multiplier (OMQQM_cir).

    Computes: |a⟩ |b⟩ |0⟩ → |a⟩ |b⟩ |(a * b) mod N⟩

    Based on notebook: For each bit i of a, if a[i] = 1, add 2^(n-i-1) * b mod N to aux.
    This effectively computes a * b mod N.

    Args:
        circ: Quantum circuit to add gates to
        a_qubits: Qubit indices for register a (n qubits)
        b_qubits: Qubit indices for register b (n qubits)
        N: Modulus
        aux_qubits: Auxiliary qubits (n + 2 qubits needed), result will be in aux_qubits[1:n+1]
    """
    n = len(a_qubits)
    assert len(b_qubits) == n, "a and b must have same length"
    assert len(aux_qubits) >= n + 2, f"Need at least {n + 2} auxiliary qubits"
    # Build multout_cir circuit for each bit position
    for i in range(n):
        tempk = 2 ** (n - (i + 1))  # 2^(n-i-1) for MSB to LSB
        # multout_cir needs: n qubits for control (b_qubits) + (n + 2) qubits for aux
        total_subcirc_qubits = n + (n + 2)
        multout_subcirc = QuantumCircuit(
            total_subcirc_qubits, name=f"Multout({tempk}, {N})"
        )
        # Map: first n qubits are b_qubits, next n+2 are aux_qubits
        b_mapped = list(range(n))
        aux_mapped = list(range(n, n + n + 2))
        modular_multiply_quantum_classical(
            multout_subcirc, b_mapped, tempk, N, aux_mapped
        )

        # Make it controlled by a_qubits[i]
        # Controlled gate needs: 1 control qubit + total_subcirc_qubits
        multout_gate = multout_subcirc.to_gate()
        multout_gate.name = f"Multout({tempk}, {N})"
        multout_gate_controlled = multout_gate.control(1)
        multout_gate_controlled.name = f"C-Multout({tempk}, {N})"

        # Apply: control from a_qubits[i], then b_qubits, then aux_qubits
        # The gate expects: [control] + [b_qubits (n)] + [aux_qubits (n+2)]
        control_qubit = a_qubits[n - (i + 1)]  # MSB first

        # Ensure aux_qubits has exactly n+2 qubits
        if len(aux_qubits) != n + 2:
            raise ValueError(
                f"aux_qubits must have exactly {n + 2} qubits, got {len(aux_qubits)}"
            )

        circ.append(multout_gate_controlled, [control_qubit] + b_qubits + aux_qubits)


def modular_multin_quantum_classical(
    circ: QuantumCircuit, a_qubits: List[int], k: int, N: int, aux_qubits: List[int]
) -> None:
    """
    Inplace modular quantum-classical multiplier (IMQCM_cir).

    Computes: |a⟩ |0⟩ → |(k*a) mod N⟩ |a⟩

    From notebook: Uses multout_cir, then swap, then inverse multout_cir

    Args:
        circ: Quantum circuit to add gates to
        a_qubits: Qubit indices for register a (n qubits)
        k: Classical constant to multiply by
        N: Modulus
        aux_qubits: Auxiliary qubits (n + 2 qubits needed)
    """
    n = len(a_qubits)
    assert len(aux_qubits) >= n + 2, f"Need at least {n + 2} auxiliary qubits"

    # Apply multout_cir: |a⟩ |aux⟩ → |a⟩ |aux + k*a mod N⟩
    modular_multiply_quantum_classical(circ, a_qubits, k, N, aux_qubits)

    # Swap a_qubits and aux_qubits[1:n+1] (the result part)
    for i in range(n):
        circ.swap(a_qubits[i], aux_qubits[i + 1])

    # Apply inverse multout_cir with k^(-1) mod N
    try:
        k_inv = mod_inverse(k, N)
        modular_multiply_quantum_classical(
            circ,
            aux_qubits[1 : n + 1],
            k_inv,
            N,
            a_qubits + [aux_qubits[0]] + aux_qubits[n + 1 :],
        )
    except ValueError:
        raise ValueError(
            f"{k} is not invertible modulo {N}, cannot compute inplace multiplication"
        )


def modular_multin_quantum_quantum(
    circ: QuantumCircuit,
    a_qubits: List[int],
    b_qubits: List[int],
    N: int,
    aux_qubits: List[int],
) -> None:
    """
    Inplace modular quantum-quantum multiplier for squaring.

    Computes: |a⟩ |a⟩ |0⟩ → |a^2 mod N⟩ |a⟩ |0⟩

    This is a specialized version for squaring (a == b) that avoids duplicate qubit issues.
    For squaring, we directly compute a^2 using controlled additions, similar to
    modular_multiply_quantum_classical but adapted for quantum-quantum case.

    Args:
        circ: Quantum circuit to add gates to
        a_qubits: Qubit indices for register a (n qubits)
        b_qubits: Qubit indices for register b (n qubits, should be same as a_qubits for squaring)
        N: Modulus
        aux_qubits: Auxiliary qubits (n + 2 qubits needed)
    """
    n = len(a_qubits)
    assert len(b_qubits) == n, "a and b must have same length"
    assert len(aux_qubits) >= n + 2, f"Need at least {n + 2} auxiliary qubits"

    # Check if this is squaring (a == b)
    a_set = set(a_qubits)
    b_set = set(b_qubits)
    is_squaring = a_set == b_set

    if not is_squaring:
        raise ValueError(
            "modular_multin_quantum_quantum is currently only implemented for squaring (a == b)"
        )
    for i in range(n):
        power_of_2 = 2 ** (n - (i + 1))  # 2^(n-i-1)
        control_qubit = a_qubits[n - (i + 1)]  # MSB first
        for j in range(n):
            bit_value = 2 ** (n - (j + 1))  # 2^(n-j-1)
            add_value = (
                power_of_2 * bit_value
            ) % N  # 2^(n-i-1) * 2^(n-j-1) = 2^(2n-i-j-2)

            target_qubit = a_qubits[n - (j + 1)]

            # Check if control_qubit and target_qubit are the same (i == j)
            addin_circ = modular_addin_circuit(n, add_value, N)
            addin = addin_circ.to_gate()
            addin.name = f"Addin({add_value}, {N})"
            if control_qubit == target_qubit:
                # Use single control gate when i == j
                addin_controlled = addin.control(1)
                addin_controlled.name = f"C-Addin({add_value}, {N})"
                circ.append(addin_controlled, [control_qubit] + aux_qubits)
            else:
                # Use double control gate when i != j
                addin_controlled = addin.control(2)
                addin_controlled.name = f"CC-Addin({add_value}, {N})"
                circ.append(
                    addin_controlled, [control_qubit, target_qubit] + aux_qubits
                )

    # Now aux_qubits[1:n+1] contains a^2 mod N
    # Swap a_qubits and aux_qubits[1:n+1] to get result in a_qubits
    result_in_aux = aux_qubits[1 : n + 1]
    for i in range(n):
        circ.swap(a_qubits[i], result_in_aux[i])

    # Note: b_qubits (which is the same as a_qubits for squaring) now contains the original a
    # The aux register contains a^2, which we've swapped into a_qubits


def _apply_controlled_inplace_multiply(
    qc: QuantumCircuit,
    ctrl_qubit: int,
    target_qubits: List[int],
    k: int,
    N: int,
    aux_qubits: List[int],
) -> None:
    """
    Apply controlled in-place modular multiplication.

    If ctrl_qubit = 1: target *= k mod N
    If ctrl_qubit = 0: target unchanged

    Uses the IMQCM approach:
    1. Controlled outplace multiplication: aux = k * target mod N
    2. Controlled swap: swap target and aux
    3. Controlled inverse: uncompute using k^(-1)

    Args:
        qc: Quantum circuit to add gates to
        ctrl_qubit: Control qubit index
        target_qubits: Qubit indices for target register (n qubits)
        k: Multiplication constant
        N: Modulus
        aux_qubits: Auxiliary qubits (n + 2 qubits needed)
    """
    n = len(target_qubits)
    assert len(aux_qubits) >= n + 2, f"Need at least {n + 2} auxiliary qubits"
    result_in_aux = aux_qubits[1 : n + 1]

    # Compute k^(-1) mod N
    k_inv = mod_inverse(k, N)

    # Step 1: Controlled outplace multiplication
    # For each bit i of target, if ctrl AND target[i], add k * 2^i to aux
    for i in range(n):
        tempk = ((2 ** (n - (i + 1))) * k) % N
        addin_circ = modular_addin_circuit(n, tempk, N)
        addin = addin_circ.to_gate()
        addin.name = f"Addin({tempk}, {N})"
        addin_controlled = addin.control(2)
        addin_controlled.name = f"CC-Addin({tempk}, {N})"
        qc.append(
            addin_controlled, [ctrl_qubit, target_qubits[n - (i + 1)]] + aux_qubits
        )

    # Step 2: Controlled swap
    for i in range(n):
        qc.cswap(ctrl_qubit, target_qubits[i], result_in_aux[i])

    # Step 3: Controlled inverse multiplication
    # Subtract k_inv * (new target value) from aux to uncompute
    for i in range(n):
        tempk = ((2 ** (n - (i + 1))) * k_inv) % N
        neg_tempk = (N - tempk) % N
        addin_circ = modular_addin_circuit(n, neg_tempk, N)
        addin = addin_circ.to_gate()
        addin.name = f"Addin({neg_tempk}, {N})"
        addin_controlled = addin.control(2)
        addin_controlled.name = f"CC-Addin(-{tempk}, {N})"
        qc.append(
            addin_controlled, [ctrl_qubit, target_qubits[n - (i + 1)]] + aux_qubits
        )


def modular_exponentiate_quantum_classical(
    circ: QuantumCircuit,
    x_qubits: List[int],
    a: int,
    N: int,
    target_qubits: List[int],
    aux_qubits: List[int],
) -> None:
    """
    Outplace modular quantum-classical exponentiation (OMEO_cir).

    Computes: |x⟩ |1⟩ |0⟩ → |x⟩ |(a^x) mod N⟩ |0⟩

    This is the Shor's algorithm style exponentiation where the base is classical
    and the exponent is quantum. All powers a^(2^i) are precomputed classically.

    Algorithm:
    1. Precompute a^(2^i) mod N for all i
    2. For each bit x_i of x:
       - If x_i = 1, multiply target by a^(2^i) mod N using controlled in-place multiplication

    Args:
        circ: Quantum circuit to add gates to
        x_qubits: Qubit indices for exponent register (n qubits, quantum)
        a: Base (classical constant)
        N: Modulus
        target_qubits: Qubit indices for result register (n qubits), initialized to |1⟩
        aux_qubits: Auxiliary qubits (n + 2 qubits needed)
    """
    n = len(x_qubits)
    assert len(target_qubits) == n, "target must have same length as x"
    assert len(aux_qubits) >= n + 2, f"Need at least {n + 2} auxiliary qubits"

    # Check that a and N are coprime (required for modular exponentiation)
    if gcd(a, N) != 1:
        raise ValueError(
            f"Base {a} and modulus {N} must be coprime for modular exponentiation"
        )

    # Precompute all powers: a^1, a^2, a^4, ..., a^(2^{n-1}) mod N
    powers = precompute_powers(a, n, N)

    # For each bit of the exponent, apply controlled modular multiplication
    # x_i controls multiplication by a^(2^i)
    for i in range(n):
        power = powers[i]  # a^(2^i) mod N

        # Apply controlled in-place multiplication: if x_qubits[i] = 1, result *= power
        _apply_controlled_inplace_multiply(
            circ,
            ctrl_qubit=x_qubits[i],
            target_qubits=target_qubits,
            k=power,
            N=N,
            aux_qubits=aux_qubits,
        )
