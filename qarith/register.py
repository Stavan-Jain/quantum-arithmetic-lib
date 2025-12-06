from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

from .expr import TypedExpr, BinOp, Var, Lit
from .qtype import QType

if TYPE_CHECKING:
    from .prog import Program


@dataclass
class QRegister:
    """
    Quantum register representing a quantum integer.

    Tracks the qubit allocation for a quantum variable.
    Supports operator overloading for natural code: a = a + b
    """

    name: str
    n_bits: int
    offset: int  # starting qubit index
    ty: QType
    _program: "Program" = field(default=None, repr=False)  # Forward reference

    def bits(self) -> List[int]:
        """Return the list of qubit indices for this register."""
        return [self.offset + i for i in range(self.n_bits)]

    def __repr__(self) -> str:
        return f"QRegister({self.name}, {self.ty}, bits={self.n_bits}, offset={self.offset})"

    def _check_compatible(self, other: QRegister) -> None:
        """Type check: ensure types are compatible for binary operations."""
        from .qtype import check_binary_op

        try:
            check_binary_op("+", self.ty, other.ty)
        except TypeError as e:
            raise TypeError(f"Type mismatch: {e}")

    def __add__(self, other):
        """Support: self + other"""
        if isinstance(other, QRegister):
            self._check_compatible(other)
            return TypedExpr(BinOp("+", Var(self.name), Var(other.name)), self.ty)
        elif isinstance(other, int):
            return TypedExpr(BinOp("+", Var(self.name), Lit(other)), self.ty)
        else:
            raise TypeError(f"Cannot add {type(other)} to QRegister")

    def __radd__(self, other):
        """Support: other + self"""
        if isinstance(other, int):
            return TypedExpr(BinOp("+", Lit(other), Var(self.name)), self.ty)
        return NotImplemented

    def __sub__(self, other):
        """Support: self - other"""
        if isinstance(other, QRegister):
            self._check_compatible(other)
            return TypedExpr(BinOp("-", Var(self.name), Var(other.name)), self.ty)
        elif isinstance(other, int):
            return TypedExpr(BinOp("-", Var(self.name), Lit(other)), self.ty)
        else:
            raise TypeError(f"Cannot subtract {type(other)} from QRegister")

    def __rsub__(self, other):
        """Support: other - self"""
        if isinstance(other, int):
            return TypedExpr(BinOp("-", Lit(other), Var(self.name)), self.ty)
        return NotImplemented

    def __mul__(self, other):
        """Support: self * other"""
        if isinstance(other, QRegister):
            self._check_compatible(other)
            return TypedExpr(BinOp("*", Var(self.name), Var(other.name)), self.ty)
        elif isinstance(other, int):
            return TypedExpr(BinOp("*", Var(self.name), Lit(other)), self.ty)
        else:
            raise TypeError(f"Cannot multiply QRegister by {type(other)}")

    def __pow__(self, other):
        """Support: self ** other"""
        if isinstance(other, QRegister):
            self._check_compatible(other)
            return TypedExpr(BinOp("**", Var(self.name), Var(other.name)), self.ty)
        elif isinstance(other, int):
            return TypedExpr(BinOp("**", Var(self.name), Lit(other)), self.ty)
        else:
            raise TypeError(f"Cannot raise QRegister to {type(other)}")

    def __floordiv__(self, other):
        """Support: self // other"""
        if isinstance(other, QRegister):
            self._check_compatible(other)
            return TypedExpr(BinOp("//", Var(self.name), Var(other.name)), self.ty)
        elif isinstance(other, int):
            return TypedExpr(BinOp("//", Var(self.name), Lit(other)), self.ty)
        else:
            raise TypeError(f"Cannot divide QRegister by {type(other)}")

    def __iadd__(self, other):
        """Support: self += other (in-place addition)"""
        expr = self + other
        # Add operation to program (will be compiled later)
        if self._program is not None:
            self._program.add_operation(expr, self.name)
        return self

    def __isub__(self, other):
        """Support: self -= other (in-place subtraction)"""
        expr = self - other
        # Add operation to program (will be compiled later)
        if self._program is not None:
            self._program.add_operation(expr, self.name)
        return self

    def __imul__(self, other):
        """Support: self *= other (in-place multiplication)"""
        expr = self * other
        # Add operation to program (will be compiled later)
        if self._program is not None:
            self._program.add_operation(expr, self.name)
        return self

    def __ipow__(self, other):
        """Support: self **= other (in-place exponentiation)"""
        expr = self**other
        # Add operation to program (will be compiled later)
        if self._program is not None:
            self._program.add_operation(expr, self.name)
        return self

    def __ifloordiv__(self, other):
        """Support: self //= other (in-place floor division)"""
        expr = self // other
        # Add operation to program (will be compiled later)
        if self._program is not None:
            self._program.add_operation(expr, self.name)
        return self

    def assign(self, value):
        """
        Assign a value to this register.

        Supports:
        - a.assign(0) for constant initialization
        - a.assign(a + b) for expression assignment

        Note: For Pythonic syntax like 'a = 0', you can use a wrapper.
        This method provides explicit assignment.
        """
        if self._program is None:
            raise RuntimeError("Cannot assign: QRegister is not attached to a Program")
        if isinstance(value, int):
            from .expr import Lit

            const_expr = TypedExpr(Lit(value), self.ty)
            self._program.add_operation(const_expr, self.name)
        elif isinstance(value, TypedExpr):
            self._program.add_operation(value, self.name)
        else:
            raise TypeError(f"Cannot assign {type(value)} to QRegister")
        return self
