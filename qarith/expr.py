from __future__ import annotations
from dataclasses import dataclass
from abc import ABC
from .qtype import QType


class Expr(ABC):
    """Base class for all expressions."""

    pass


@dataclass(frozen=True)
class Var(Expr):
    """Variable reference: x"""

    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Lit(Expr):
    """Literal value: v"""

    value: int

    def __repr__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class BinOp(Expr):
    """Binary operation: e₁ ⊕ e₂"""

    op: str  # +, -, *, //, **
    left: Expr
    right: Expr

    def __repr__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass(frozen=True)
class UnaryOp(Expr):
    """Unary operation: op e"""

    op: str  # sqrt, inv
    expr: Expr

    def __repr__(self) -> str:
        return f"{self.op}({self.expr})"


@dataclass(frozen=True)
class Ord(Expr):
    """Order operation: ord e₁ e₂"""

    base: Expr
    modulus: Expr

    def __repr__(self) -> str:
        return f"ord({self.base}, {self.modulus})"


@dataclass(frozen=True)
class TypedExpr:
    """Expression with type information."""

    expr: Expr
    ty: QType

    def __repr__(self) -> str:
        return f"{self.expr}: {self.ty}"
