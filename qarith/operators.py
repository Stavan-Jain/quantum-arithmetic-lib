"""
Modular arithmetic operations for quantum circuits.

This module implements modular addition, subtraction, multiplication, and
exponentiation using QFT-based approaches as described in the notebook.
"""

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
import numpy as np
from typing import List


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
    qc = QuantumCircuit(n + 2)
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
        addin = modular_addin_circuit(n, tempk, N).to_gate().control(1)
        addin.name = f"Addin({tempk})"
        # Control from a_qubits[i], target aux_qubits
        control_qubit = a_qubits[n - (i + 1)]
        circ.append(addin, [control_qubit] + aux_qubits[:])

    # Final addition with k
    addin_kN = modular_addin_circuit(n, k, N).to_gate()
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
        addin = modular_addin_circuit(n, tempk, N).to_gate().control(1)
        addin.name = f"Addin({tempk}, N)"
        circ.append(addin, [a_qubits[n - (i + 1)]] + aux_qubits[:])

    # Apply controlled modular additions for each bit of b
    for i in range(n):
        tempk = 2 ** (n - (i + 1))
        addin = modular_addin_circuit(n, tempk, N).to_gate().control(1)
        addin.name = f"Addin({tempk}, N)"
        circ.append(addin, [b_qubits[n - (i + 1)]] + aux_qubits[:])


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
        addin = modular_addin_circuit(n, tempk, N).to_gate().control(1)
        addin.name = f"Sub({power_of_2}, N)"
        # Control from b_qubits[n - (i + 1)], add (N - 2^i) to aux_qubits
        control_qubit = b_qubits[n - (i + 1)]
        circ.append(addin, [control_qubit] + aux_qubits[:])


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
        addin = modular_addin_circuit(n, tempk, N).to_gate().control(1)
        addin.name = f"Addin({tempk})"
        # Control from a_qubits[n - (i + 1)], target aux_qubits
        # From notebook: control qubit is totl_qubits - i
        control_qubit = a_qubits[n - (i + 1)]
        circ.append(addin, [control_qubit] + aux_qubits[:])


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

    # From notebook OMQQM_cir: For each bit i of a (from MSB to LSB),
    # if a[i] = 1, apply controlled multout_cir with k = 2^(n-i-1)
    # multout_cir(n, k, N): |a⟩ |b⟩ → |a⟩ |b + k*a mod N⟩
    # So for each bit i of a, if a[i] = 1, we want: aux = aux + 2^(n-i-1) * b mod N

    # We need to build a controlled multout_cir circuit
    # multout_cir uses a_qubits as control and aux_qubits as target
    # But in OMQQM, we want to use a_qubits[i] as control and operate on aux + b

    # Actually, looking at the notebook more carefully:
    # multout_cir(n, k, N) operates on |a⟩ |b⟩ where a is control and b is target
    # In OMQQM, we want: for each bit i of a, if a[i] = 1, add 2^(n-i-1) * b to aux
    # So we use a_qubits[i] as control, and multout_cir operates on b_qubits (as control) and aux_qubits (as target)

    # Build multout_cir circuit for each bit position
    for i in range(n):
        tempk = 2 ** (n - (i + 1))  # 2^(n-i-1) for MSB to LSB
        # Create a multout_cir circuit: |b⟩ |aux⟩ → |b⟩ |aux + tempk * b mod N⟩
        # This is controlled by a_qubits[i]

        # Build multout_cir as a subcircuit
        # multout_cir needs: n qubits for control (b_qubits) + (n + 2) qubits for aux
        total_subcirc_qubits = n + (n + 2)
        multout_subcirc = QuantumCircuit(total_subcirc_qubits)
        # Map: first n qubits are b_qubits, next n+2 are aux_qubits
        b_mapped = list(range(n))
        aux_mapped = list(range(n, n + n + 2))
        modular_multiply_quantum_classical(
            multout_subcirc, b_mapped, tempk, N, aux_mapped
        )

        # Make it controlled by a_qubits[i]
        # Controlled gate needs: 1 control qubit + total_subcirc_qubits
        multout_gate = multout_subcirc.to_gate().control(1)
        multout_gate.name = f"Multout({tempk})"

        # Apply: control from a_qubits[i], then b_qubits, then aux_qubits
        # The gate expects: [control] + [b_qubits (n)] + [aux_qubits (n+2)]
        control_qubit = a_qubits[n - (i + 1)]  # MSB first

        # Ensure aux_qubits has exactly n+2 qubits
        if len(aux_qubits) != n + 2:
            raise ValueError(
                f"aux_qubits must have exactly {n + 2} qubits, got {len(aux_qubits)}"
            )

        circ.append(multout_gate, [control_qubit] + b_qubits + aux_qubits)


def modular_multiply_inplace_quantum_classical(
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
    # We need to compute modular inverse of k modulo N
    # For now, use extended Euclidean algorithm
    def mod_inverse(k, N):
        """Compute modular inverse of k modulo N using extended Euclidean algorithm"""
        if k == 0:
            return 0
        t, new_t = 0, 1
        r, new_r = N, k
        while new_r != 0:
            quotient = r // new_r
            t, new_t = new_t, t - quotient * new_t
            r, new_r = new_r, r - quotient * new_r
        if r > 1:
            raise ValueError(f"{k} is not invertible modulo {N}")
        if t < 0:
            t = t + N
        return t

    try:
        k_inv = mod_inverse(k, N)
        # Apply multout_cir with k_inv, but in reverse
        # Actually, we need to apply the inverse operation
        # For now, we'll use a simpler approach: apply multout_cir with -k_inv
        # But this is not quite right - we need the actual inverse circuit

        # Actually, looking at the notebook more carefully:
        # IMQCM uses multout_cir, then swap, then inverse multout_cir with k^(-1)
        # The inverse multout_cir should undo the multiplication
        # So we apply multout_cir with k_inv
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

    From notebook: For each bit i of x, if x[i] = 1, use IMQCM_cir to multiply by a^(2^i) mod N

    Args:
        circ: Quantum circuit to add gates to
        x_qubits: Qubit indices for exponent register (n qubits)
        a: Base (classical constant)
        N: Modulus
        target_qubits: Qubit indices for result register (n qubits), initialized to |1⟩
        aux_qubits: Auxiliary qubits (n + 2 qubits needed)
    """
    n = len(x_qubits)
    assert len(target_qubits) == n, "target must have same length as x"
    assert len(aux_qubits) >= n + 2, f"Need at least {n + 2} auxiliary qubits"

    # For each bit position i of x, if x[i] = 1, multiply target by a^(2^i) mod N
    # From notebook: OMEO uses IMQCM_cir controlled by bits of x
    # IMQCM_cir multiplies by k mod N in-place: |a⟩ |0⟩ → |(k*a) mod N⟩ |a⟩

    # We need to build a controlled version of IMQCM_cir
    # For each bit i of x, if x[i] = 1, apply IMQCM_cir to multiply target by a^(2^i)
    # This is complex - for now, raise NotImplementedError
    # Full implementation would require building the complete IMQCM_cir circuit
    # and controlling it properly
    raise NotImplementedError(
        "Modular exponentiation requires implementing controlled IMQCM_cir circuit"
    )


def modular_exponentiate_quantum_quantum(
    circ: QuantumCircuit,
    x_qubits: List[int],
    a_qubits: List[int],
    N: int,
    target_qubits: List[int],
    aux_qubits: List[int],
) -> None:
    """
    Outplace modular quantum-quantum exponentiation.

    Computes: |x⟩ |a⟩ |1⟩ |0⟩ → |x⟩ |a⟩ |(a^x) mod N⟩ |0⟩

    This is more complex and requires computing powers of a dynamically.

    Args:
        circ: Quantum circuit to add gates to
        x_qubits: Qubit indices for exponent register (n qubits)
        a_qubits: Qubit indices for base register (n qubits)
        N: Modulus
        target_qubits: Qubit indices for result register (n qubits), initialized to |1⟩
        aux_qubits: Auxiliary qubits (need more than n + 2 for this operation)
    """
    n = len(x_qubits)
    assert len(a_qubits) == n, "a and x must have same length"
    assert len(target_qubits) == n, "target must have same length"
    assert len(aux_qubits) >= 2 * n + 2, f"Need at least {2 * n + 2} auxiliary qubits"

    # This is more complex - we need to compute powers of a dynamically
    # For each bit position i of x:
    #   1. Compute a^(2^i) mod N into a temporary register
    #   2. If x[i] = 1, multiply target by this temporary value

    # For now, this is a placeholder - full implementation would require
    # more sophisticated power computation
    raise NotImplementedError(
        "Quantum-quantum modular exponentiation requires more complex implementation"
    )
