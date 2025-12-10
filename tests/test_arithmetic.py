"""
Test suite for quantum arithmetic operations.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qarith.expr import Var, Const
from qarith.qtype import Mod
from qarith import run


def test_addition():
    """Test modular addition."""
    print("Testing Addition")

    modulus = 11
    x1 = Var("x1", Mod(modulus))
    x2 = Var("x2", Mod(modulus))

    test_cases = [
        (5, 3),
        (5, 7),
        (10, 2),
        (0, 5),
    ]

    for a, b in test_cases:
        expected = (a + b) % modulus
        expr = x1 + x2
        result = run(expr, {"x1": a, "x2": b})
        status = "✓" if result == expected else "✗"
        print(f"{status} ({a} + {b}) mod {modulus} = {result} (expected: {expected})")
        assert result == expected, (
            f"Addition failed: ({a} + {b}) mod {modulus} = {result}, expected {expected}"
        )

    print()


def test_subtraction():
    """Test modular subtraction."""
    print("Testing Subtraction")

    modulus = 11
    x1 = Var("x1", Mod(modulus))
    x2 = Var("x2", Mod(modulus))

    test_cases = [
        (5, 3),
        (5, 6),
        (2, 5),
        (0, 3),
    ]

    for a, b in test_cases:
        expected = (a - b) % modulus
        expr = x1 - x2
        result = run(expr, {"x1": a, "x2": b})
        status = "✓" if result == expected else "✗"
        print(f"{status} ({a} - {b}) mod {modulus} = {result} (expected: {expected})")
        assert result == expected, (
            f"Subtraction failed: ({a} - {b}) mod {modulus} = {result}, expected {expected}"
        )

    print()


def test_multiplication():
    """Test modular multiplication."""
    print("Testing Multiplication")
    modulus = 5
    x1 = Var("x1", Mod(modulus))
    x2 = Var("x2", Mod(modulus))

    test_cases = [
        (2, 3),
        (3, 2),
        (4, 2),
        (0, 5),
        (1, 4),
    ]
    for a, b in test_cases:
        expected = (a * b) % modulus
        expr = x1 * x2
        result = run(expr, {"x1": a, "x2": b})
        status = "✓" if result == expected else "✗"
        print(f"{status} ({a} * {b}) mod {modulus} = {result} (expected: {expected})")
        assert result == expected, (
            f"Multiplication failed: ({a} * {b}) mod {modulus} = {result}, expected {expected}"
        )

    print()


def test_quantum_base_classic_exp():
    """Test quantum base, classic exponent exponentiation."""
    print("Testing Exponentiation")
    modulus = 4
    x1 = Var("x1", Mod(modulus))
    test_cases = [
        (2, 3),
        (1, 2),
        (3, 2),
    ]

    for base, exp in test_cases:
        expected = (base**exp) % modulus
        expr = x1**exp
        result = run(expr, {"x1": base})
        status = "✓" if result == expected else "✗"
        print(
            f"{status} ({base} ** {exp}) mod {modulus} = {result} (expected: {expected})"
        )
        assert result == expected, (
            f"Exponentiation failed: ({base} ** {exp}) mod {modulus} = {result}, expected {expected}"
        )
    print()


def test_mixed_operations():
    """Test mixed arithmetic operations."""
    print("=" * 60)
    print("Testing Mixed Operations")
    print("=" * 60)

    modulus = 11
    x1 = Var("x1", Mod(modulus))
    x2 = Var("x2", Mod(modulus))
    x3 = Var("x3", Mod(modulus))

    # Test: (x1 + x2) * x3
    test_cases = [
        (2, 3, 4, 9),  # (2 + 3) * 4 = 5 * 4 = 20 mod 11 = 9
        (5, 1, 3, 7),  # (5 + 1) * 3 = 6 * 3 = 18 mod 11 = 7
    ]

    for a, b, c, expected in test_cases:
        expr = (x1 + x2) * x3
        result = run(expr, {"x1": a, "x2": b, "x3": c})
        status = "✓" if result == expected else "✗"
        print(
            f"{status} (({a} + {b}) * {c}) mod {modulus} = {result} (expected: {expected})"
        )
        assert result == expected, (
            f"Mixed operation failed: (({a} + {b}) * {c}) mod {modulus} = {result}, expected {expected}"
        )

    # Test: x1 * x2 + x3
    test_cases = [
        (2, 3, 4, 10),  # 2 * 3 + 4 = 6 + 4 = 10 mod 11
        (5, 2, 1, 0),  # 5 * 2 + 1 = 10 + 1 = 11 mod 11 = 0
    ]

    for a, b, c, expected in test_cases:
        expr = x1 * x2 + x3
        result = run(expr, {"x1": a, "x2": b, "x3": c})
        status = "✓" if result == expected else "✗"
        print(
            f"{status} ({a} * {b} + {c}) mod {modulus} = {result} (expected: {expected})"
        )
        assert result == expected, (
            f"Mixed operation failed: ({a} * {b} + {c}) mod {modulus} = {result}, expected {expected}"
        )

    print()


# a = Const(3, modulus=5)
# x = Var(“x”, Mod(16))
# z = a ** x


def test_quantum_exponent_classic_base():
    """Test quantum exponent, classic base exponentiation."""
    print("Testing Exponentiation")
    modulus = 5
    x1 = Var("x1", Mod(modulus))
    test_cases = [
        (2, 3),
        (1, 2),
        (3, 2),
    ]

    for base, exp in test_cases:
        expected = (base**exp) % modulus
        expr = Const(base, modulus=modulus) ** x1
        result = run(expr, {"x1": exp})
        status = "✓" if result == expected else "✗"
        print(
            f"{status} ({base} ** {exp}) mod {modulus} = {result} (expected: {expected})"
        )
        assert result == expected, (
            f"Exponentiation failed: ({base} ** {exp}) mod {modulus} = {result}, expected {expected}"
        )
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Quantum Arithmetic Test Suite")
    print("=" * 60 + "\n")

    try:
        test_quantum_exponent_classic_base()
        test_quantum_base_classic_exp()
        test_addition()
        test_subtraction()
        test_multiplication()
        test_mixed_operations()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        raise
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
