"""
Type system for QMARITH: Quantum Modular Arithmetic DSL.

Types in QMARITH:
- Base types: [N] for modulus N ∈ ℕ
- Maybe types: Maybe(T) for optional type T (T | None)

For type [N], the bit-length is n = ⌈log₂(N)⌉.
For type Maybe(T), the bit-length is T.n_bits + 1 (extra bit for None flag).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Union
import math


@dataclass(frozen=True)
class Mod:
    """Modular integer type with a fixed modulus N: [N]

    The bit-length is n = ⌈log₂(N)⌉.
    """

    N: int

    def __repr__(self) -> str:
        return f"[{self.N}]"

    def __post_init__(self):
        if self.N <= 0:
            raise ValueError(f"Modulus must be positive, got {self.N}")

    @property
    def n_bits(self) -> int:
        """Return the number of bits needed to represent this type: ⌈log₂(N)⌉"""
        if self.N == 1:
            return 1
        return math.ceil(math.log2(self.N))


# TODO: not sure how to do this. Use Maybe monad for now.
# @dataclass(frozen=True)
# class Sum:
#     inner_types: tuple[QType, ...]

#     def __repr__(self) -> str:
#         return f"Sum({self.inner_types})"

#     def __post_init__(self):
#         if not all(isinstance(ty, QType) for ty in self.inner_types):
#             raise ValueError(f"Sum inner types must be QTypes, got {self.inner_types}")

#     @property
#     def n_bits(self) -> int:
#         return max(ty.n_bits for ty in self.inner_types)

#     @classmethod
#     def of(cls, *types: QType) -> Sum:
#         return cls(types)


@dataclass(frozen=True)
class Maybe:
    """Maybe type: T | None, where T is a QType.

    For reversibility, Maybe(T) represents an optional value of type T.
    The bit-length is T.n_bits + 1 (extra bit for None flag).
    """

    inner_type: QType

    def __repr__(self) -> str:
        return f"Maybe({self.inner_type})"

    def __post_init__(self):
        if not isinstance(self.inner_type, (Mod, Maybe)):
            raise ValueError(
                f"Maybe inner type must be a QType, got {type(self.inner_type)}"
            )

    @property
    def n_bits(self) -> int:
        """Return the number of bits needed: inner_type.n_bits + 1 (for None flag)"""
        return self.inner_type.n_bits + 1


# Union type for all possible quantum types
QType = Union[Mod, Maybe]


def _check_binary_mod(e1_type: QType, e2_type: QType) -> Mod:
    """
    Internal helper: Check that both types are Mod with same modulus.

    Returns the result Mod type if valid, raises TypeError otherwise.
    """
    if isinstance(e1_type, Mod) and isinstance(e2_type, Mod):
        if e1_type.N == e2_type.N:
            return e1_type
        else:
            raise TypeError(
                f"Binary operation on Mod types requires same modulus: "
                f"got Mod[{e1_type.N}] and Mod[{e2_type.N}]"
            )
    elif isinstance(e1_type, Maybe) and isinstance(e2_type, Maybe):
        return Maybe(_check_binary_mod(e1_type.inner_type, e2_type.inner_type))
    elif isinstance(e1_type, Maybe) and isinstance(e2_type, Mod):
        return Maybe(_check_binary_mod(e1_type.inner_type, e2_type))
    elif isinstance(e1_type, Mod) and isinstance(e2_type, Maybe):
        return Maybe(_check_binary_mod(e1_type, e2_type.inner_type))
    raise TypeError(
        f"Binary operation requires Mod types with same modulus, "
        f"got {e1_type} and {e2_type}"
    )


def check_add(e1_type: QType, e2_type: QType) -> QType:
    return _check_binary_mod(e1_type, e2_type)


def check_sub(e1_type: QType, e2_type: QType) -> QType:
    return _check_binary_mod(e1_type, e2_type)


def check_mul(e1_type: QType, e2_type: QType) -> QType:
    return _check_binary_mod(e1_type, e2_type)


def check_pow(e1_type: QType, e2_type: QType | int) -> QType:
    if isinstance(e2_type, int):
        return e1_type
    return _check_binary_mod(e1_type, e2_type)


def check_floordiv(e1_type: QType, e2_type: QType) -> QType:
    return _check_binary_mod(e1_type, e2_type)
