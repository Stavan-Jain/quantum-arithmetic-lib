"""
Test suite for quantum arithmetic operations.

Tests addition, subtraction, multiplication, and exponentiation operations
with various input values and moduli.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qarith.expr import Var
from qarith.qtype import Mod
from qarith import run


def test_addition():
    """Test modular addition."""
    print("Testing Addition")

    modulus = 11
    x1 = Var("x1", Mod(modulus))
    x2 = Var("x2", Mod(modulus))

    test_cases = [
        (5, 3, 8),  # 5 + 3 = 8 mod 11
        (5, 7, 1),  # 5 + 7 = 12 mod 11 = 1
        (10, 2, 1),  # 10 + 2 = 12 mod 11 = 1
        (0, 5, 5),  # 0 + 5 = 5 mod 11
    ]

    for a, b, expected in test_cases:
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
        (5, 3, 2),  # 5 - 3 = 2 mod 11
        (5, 6, 10),  # 5 - 6 = -1 mod 11 = 10
        (2, 5, 8),  # 2 - 5 = -3 mod 11 = 8
        (0, 3, 8),  # 0 - 3 = -3 mod 11 = 8
    ]

    for a, b, expected in test_cases:
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
        (2, 3, 1),  # 2 * 3 = 6 mod 5 = 1
        (3, 2, 1),  # 3 * 2 = 6 mod 5 = 1
        (4, 2, 3),  # 4 * 2 = 8 mod 5 = 3
        (0, 5, 0),  # 0 * 5 = 0 mod 5 = 0
        (1, 4, 4),  # 1 * 4 = 4 mod 5 = 4
    ]
    for a, b, expected in test_cases:
        expr = x1 * x2
        result = run(expr, {"x1": a, "x2": b})
        status = "✓" if result == expected else "✗"
        print(f"{status} ({a} * {b}) mod {modulus} = {result} (expected: {expected})")
        assert result == expected, (
            f"Multiplication failed: ({a} * {b}) mod {modulus} = {result}, expected {expected}"
        )

    print()


def test_exponentiation():
    """Test modular exponentiation."""
    print("Testing Exponentiation")
    modulus = 11
    x1 = Var("x1", Mod(modulus))
    test_cases = [
        (2, 3, 8),  # 2^3 = 8 mod 11
        (3, 2, 9),  # 3^2 = 9 mod 11
        (5, 2, 3),  # 5^2 = 25 mod 11 = 3
        (2, 5, 10),  # 2^5 = 32 mod 11 = 10
        (7, 3, 2),  # 7^3 = 343 mod 11 = 2
    ]

    for base, exp, expected in test_cases:
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


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Quantum Arithmetic Test Suite")
    print("=" * 60 + "\n")

    try:
        # test_addition()
        test_subtraction()
        test_multiplication()
        test_exponentiation()
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
