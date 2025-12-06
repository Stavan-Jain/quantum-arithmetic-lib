"""
Type system for QMARITH: Quantum Modular Arithmetic DSL.

This module implements the type system as defined in the language specification,
including:
- Int: integer type
- Mod[N]: modular integer type with modulus N
- Optional[τ]: optional type (τ?)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import (
    Union,
    Optional,
)


@dataclass(frozen=True)
class Int:
    """Integer type for quantum integers."""

    def __repr__(self) -> str:
        return "Int"


@dataclass(frozen=True)
class Mod:
    """Modular integer type with a fixed modulus N."""

    N: int

    def __repr__(self) -> str:
        return f"[{self.N}]"

    def __post_init__(self):
        if self.N <= 0:
            raise ValueError(f"Modulus must be positive, got {self.N}")


@dataclass(frozen=True)
class Opt:
    """Optional type: represents a value of type T or None.

    In the DSL syntax, this is written as τ? (e.g., [N]?).
    """

    inner: QType

    def __repr__(self) -> str:
        return f"{self.inner}?"


# Union type for all possible quantum types
QType = Union[Int, Mod, Opt]


def is_optional(ty: QType) -> bool:
    """Check if a type is optional."""
    return isinstance(ty, Opt)


def unwrap_optional(ty: QType) -> QType:
    """Unwrap an optional type, returning the inner type."""
    if isinstance(ty, Opt):
        return ty.inner
    return ty


def check_bin_mod(e1_type: QType, e2_type: QType) -> Optional[Mod]:
    """
    Type check binary operations (+, -, *, **) on [N] types.

    Rule T-BinMod:
        Γ ⊢ e₁ : [N]
        Γ ⊢ e₂ : [N]
        ──────────────────
        Γ ⊢ e₁ ⊕ e₂ : [N]

    Returns the result type if valid, None otherwise.
    """
    if isinstance(e1_type, Mod) and isinstance(e2_type, Mod):
        if e1_type.N == e2_type.N:
            return e1_type
        else:
            raise TypeError(
                f"Binary operation on Mod types requires same modulus: "
                f"got Mod[{e1_type.N}] and Mod[{e2_type.N}]"
            )
    return None


def check_bin_int(e1_type: QType, e2_type: QType) -> Optional[Int]:
    """
    Type check binary operations (+, -, *, **, //) on Int types.

    Rule T-Bin:
        Γ ⊢ e₁ : Int
        Γ ⊢ e₂ : Int
        ──────────────────
        Γ ⊢ e₁ ⊕ e₂ : Int

    Returns the result type if valid, None otherwise.
    """
    if isinstance(e1_type, Int) and isinstance(e2_type, Int):
        return Int()
    return None


def check_inv(e_type: QType) -> Optional[Opt]:
    """
    Type check modular inverse operation.

    Rule T-Inv:
        Γ ⊢ e : [N]
        ──────────────────
        Γ ⊢ inv e : [N]?

    Returns the result type if valid, None otherwise.
    """
    if isinstance(e_type, Mod):
        return Opt(e_type)
    return None


def check_ord(e1_type: QType, e2_type: QType) -> Optional[Opt]:
    """
    Type check order operation.

    Rule T-Ord:
        Γ ⊢ e₁ : [N]
        Γ ⊢ e₂ : [N]
        ──────────────────
        Γ ⊢ ord e₁ e₂ : Int?

    Returns the result type if valid, None otherwise.
    """
    if isinstance(e1_type, Mod) and isinstance(e2_type, Mod):
        if e1_type.N == e2_type.N:
            return Opt(Int())
    return None


def check_sqrt(e_type: QType) -> Optional[Opt]:
    """
    Type check square root operation.

    Rule T-Sqrt:
        Γ ⊢ e : [N]
        ──────────────────
        Γ ⊢ sqrt e : [N]?

    Returns the result type if valid, None otherwise.
    """
    if isinstance(e_type, Mod):
        return Opt(e_type)
    return None


def check_binary_op(op: str, e1_type: QType, e2_type: QType) -> Optional[QType]:
    """
    Type check a binary operation based on the operator.

    Supports:
    - +, -, *, **: work on both Int and [N]
    - //: only works on Int

    Returns the result type if valid, raises TypeError otherwise.
    """
    # Integer division only works on Int
    if op == "//":
        result = check_bin_int(e1_type, e2_type)
        if result is not None:
            return result
        raise TypeError(
            f"Integer division (//) requires Int types, got {e1_type} and {e2_type}"
        )

    # Other binary ops: try Mod first, then Int
    result = check_bin_mod(e1_type, e2_type)
    if result is not None:
        return result

    result = check_bin_int(e1_type, e2_type)
    if result is not None:
        return result

    raise TypeError(
        f"Binary operation '{op}' requires compatible types "
        f"(both Int or both Mod[N] with same N), "
        f"got {e1_type} and {e2_type}"
    )


def type_to_string(ty: QType) -> str:
    """Convert a type to its string representation."""
    return repr(ty)
