from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing_extensions import override
from typing import TYPE_CHECKING

from .qtype import QType, Mod

if TYPE_CHECKING:
    from .circuit import Circuit
else:
    Circuit = object  # Placeholder for runtime


class Expr(ABC):
    """Base class for all expressions."""

    @abstractmethod
    def qtype(self) -> QType:
        """Return the type of this expression. Raises TypeCheckError if ill-typed."""

    @abstractmethod
    def optimize(self) -> Expr:
        """Return an optimized version of this expression."""

    @abstractmethod
    def circuit(self) -> Circuit:
        """Compile this expression to a quantum circuit."""

    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of this expression."""

    # Operator overloading for natural syntax
    def __add__(self, other: Expr) -> Add:
        """A + B = Add(A, B)"""
        return Add(self, other)

    def __sub__(self, other: Expr) -> Sub:
        """A - B = Sub(A, B)"""
        return Sub(self, other)

    def __mul__(self, other: Expr) -> Mul:
        """A * B = Mul(A, B)"""
        return Mul(self, other)

    def __pow__(self, other: Expr) -> Pow:
        """A ** B = Pow(A, B)"""
        return Pow(self, other)

    def __floordiv__(self, other: Expr) -> Div:
        """A // B = FloorDiv(A, B)"""
        return Div(self, other)

    def __neg__(self) -> Sub:
        """-A = Sub(Const(0), A)"""
        try:
            ty = self.qtype()
            from .qtype import Mod

            if isinstance(ty, Mod):
                return Sub(Const(0, modulus=ty.N), self)
        except Exception as e:
            print(f"Error in __neg__: {e}")
            pass
        return Sub(Const(0), self)

    def __radd__(self, other: Expr) -> Add:
        """scalar + A = Add(Const(scalar), A)"""
        return Add(other, self)

    def __rsub__(self, other: Expr) -> Sub:
        """scalar - A = Sub(Const(scalar), A)"""
        return Sub(other, self)

    def __rmul__(self, other: Expr) -> Mul:
        """scalar * A = Mul(Const(scalar), A)"""
        return Mul(other, self)

    def __rpow__(self, other: Expr) -> Pow:
        """scalar ** A = Pow(Const(scalar), A)"""
        return Pow(other, self)

    def __rfloordiv__(self, other: Expr) -> Div:
        """scalar // A = FloorDiv(Const(scalar), A)"""
        return Div(other, self)

    @classmethod
    def log(cls, base: Expr, arg: Expr) -> Log:
        return Log(base, arg)

    @classmethod
    def sqrt(cls, expr: Expr) -> Sqrt:
        return Sqrt(expr)

    @classmethod
    def inv(cls, expr: Expr) -> Inv:
        return Inv(expr)

    @classmethod
    def indexed(cls, N: int, index: Expr) -> Indexed:
        return Indexed(N, index)

    @classmethod
    def join(cls, left: Expr, right: Expr) -> Join:
        return Join(left, right)

    @classmethod
    def const(cls, value: int, modulus: int | None = None) -> Const:
        return Const(value, modulus)


# Basic expression classes
@dataclass(frozen=True)
class Basic(Expr):
    """Black-box oracle for block encoding a matrix (parameter).

    Similar to cobble's Basic class, this represents a parameter/oracle
    that can be used in expressions. The actual quantum circuit implementation
    is provided when compiling.

    Attributes:
        name: Name of the black-box oracle (e.g., "A", "B", "X", "H")
        qtype_: The quantum type of this expression
    """

    name: str
    qtype_: QType

    @override
    def qtype(self) -> QType:
        return self.qtype_

    @override
    def optimize(self) -> Expr:
        return self

    @override
    def circuit(self) -> Circuit:
        from .compile import CompilationContext, _compile_with_context

        # Create context at top level - this ensures CSE works across the entire expression tree
        ctx = CompilationContext()
        return _compile_with_context(self, ctx)

    @override
    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Const(Expr):
    """Literal value: v

    Note: Constants need a modulus to have a type. Use Basic with a constant
    value instead, or provide the modulus explicitly.
    """

    value: int
    modulus: int | None = None  # Modulus for the constant

    @override
    def qtype(self) -> QType:
        if self.modulus is None:
            raise ValueError(
                f"Const({self.value}) requires a modulus. "
                f"Either provide modulus parameter or use Basic expression."
            )
        return Mod(self.modulus)

    @override
    def optimize(self) -> Expr:
        return self

    @override
    def circuit(self) -> Circuit:
        from .compile import CompilationContext, _compile_with_context

        # Create context at top level - this ensures CSE works across the entire expression tree
        ctx = CompilationContext()
        return _compile_with_context(self, ctx)

    @override
    def __str__(self) -> str:
        return str(self.value)


# Binary operation classes
@dataclass(frozen=True)
class Add(Expr):
    """Addition: e₁ + e₂"""

    left: Expr
    right: Expr
    _result_type: QType | None = None

    def __post_init__(self):
        """Check type compatibility and compute result type at initialization."""
        from .qtype import check_add

        result_type = check_add(self.left.qtype(), self.right.qtype())
        # Use object.__setattr__ because dataclass is frozen
        object.__setattr__(self, "_result_type", result_type)

    @override
    def qtype(self) -> QType:
        if self._result_type is None:
            raise RuntimeError("Add expression was not properly initialized")
        return self._result_type

    @override
    def optimize(self) -> Expr:
        # Dummy implementation - just optimize operands
        return Add(self.left.optimize(), self.right.optimize())

    @override
    def circuit(self) -> Circuit:
        from .compile import CompilationContext, _compile_with_context

        # Create context at top level - this ensures CSE works across the entire expression tree
        ctx = CompilationContext()
        return _compile_with_context(self, ctx)

    @override
    def __str__(self) -> str:
        return f"({self.left} + {self.right})"


@dataclass(frozen=True)
class Sub(Expr):
    """Subtraction: e₁ - e₂"""

    left: Expr
    right: Expr
    _result_type: QType | None = None

    def __post_init__(self):
        """Check type compatibility and compute result type at initialization."""
        from .qtype import check_sub

        result_type = check_sub(self.left.qtype(), self.right.qtype())
        object.__setattr__(self, "_result_type", result_type)

    @override
    def qtype(self) -> QType:
        if self._result_type is None:
            raise RuntimeError("Sub expression was not properly initialized")
        return self._result_type

    @override
    def optimize(self) -> Expr:
        return Sub(self.left.optimize(), self.right.optimize())

    @override
    def circuit(self) -> Circuit:
        from .compile import CompilationContext, _compile_with_context

        # Create context at top level - this ensures CSE works across the entire expression tree
        ctx = CompilationContext()
        return _compile_with_context(self, ctx)

    @override
    def __str__(self) -> str:
        return f"({self.left} - {self.right})"


@dataclass(frozen=True)
class Mul(Expr):
    """Multiplication: e₁ * e₂"""

    left: Expr
    right: Expr
    _result_type: QType | None = None

    def __post_init__(self):
        """Check type compatibility and compute result type at initialization."""
        from .qtype import check_mul

        result_type = check_mul(self.left.qtype(), self.right.qtype())
        object.__setattr__(self, "_result_type", result_type)

    @override
    def qtype(self) -> QType:
        if self._result_type is None:
            raise RuntimeError("Mul expression was not properly initialized")
        return self._result_type

    @override
    def optimize(self) -> Expr:
        # Dummy implementation - just optimize operands
        return Mul(self.left.optimize(), self.right.optimize())

    @override
    def circuit(self) -> Circuit:
        from .compile import CompilationContext, _compile_with_context

        # Create context at top level - this ensures CSE works across the entire expression tree
        ctx = CompilationContext()
        return _compile_with_context(self, ctx)

    @override
    def __str__(self) -> str:
        return f"({self.left} * {self.right})"


@dataclass(frozen=True)
class Pow(Expr):
    """Exponentiation: e₁ ** e₂"""

    base: Expr
    exponent: Expr
    _result_type: QType | None = None

    def __post_init__(self):
        """Check type compatibility and compute result type at initialization."""
        from .qtype import check_pow

        if isinstance(self.exponent, int):
            result_type = check_pow(self.base.qtype(), self.exponent)
        else:
            result_type = check_pow(self.base.qtype(), self.exponent.qtype())
        object.__setattr__(self, "_result_type", result_type)

    @override
    def qtype(self) -> QType:
        if self._result_type is None:
            raise RuntimeError("Pow expression was not properly initialized")
        return self._result_type

    @override
    def optimize(self) -> Expr:
        # Dummy implementation - just optimize operands
        return Pow(self.base.optimize(), self.exponent.optimize())

    @override
    def circuit(self) -> Circuit:
        from .compile import CompilationContext, _compile_with_context

        # Create context at top level - this ensures CSE works across the entire expression tree
        ctx = CompilationContext()
        return _compile_with_context(self, ctx)

    @override
    def __str__(self) -> str:
        return f"({self.base} ** {self.exponent})"


@dataclass(frozen=True)
class Div(Expr):
    """Floor division: e₁ // e₂"""

    left: Expr
    right: Expr
    _result_type: QType | None = None

    def __post_init__(self):
        """Check type compatibility and compute result type at initialization."""
        from .qtype import check_floordiv

        result_type = check_floordiv(self.left.qtype(), self.right.qtype())
        object.__setattr__(self, "_result_type", result_type)

    @override
    def qtype(self) -> QType:
        if self._result_type is None:
            raise RuntimeError("FloorDiv expression was not properly initialized")
        return self._result_type

    @override
    def optimize(self) -> Expr:
        # Dummy implementation - just optimize operands
        return Div(self.left.optimize(), self.right.optimize())

    @override
    def circuit(self) -> Circuit:
        from .compile import CompilationContext, _compile_with_context

        # Create context at top level - this ensures CSE works across the entire expression tree
        ctx = CompilationContext()
        return _compile_with_context(self, ctx)

    @override
    def __str__(self) -> str:
        return f"({self.left} // {self.right})"


# Unary operation classes
@dataclass(frozen=True)
class Sqrt(Expr):
    """Square root: sqrt e"""

    expr: Expr

    @override
    def qtype(self) -> QType:
        from .qtype import Maybe

        return Maybe(self.expr.qtype())

    @override
    def optimize(self) -> Expr:
        # Dummy implementation - just optimize operand
        return Sqrt(self.expr.optimize())

    @override
    def circuit(self) -> Circuit:
        from .compile import CompilationContext, _compile_with_context

        # Create context at top level - this ensures CSE works across the entire expression tree
        ctx = CompilationContext()
        return _compile_with_context(self, ctx)

    @override
    def __str__(self) -> str:
        return f"sqrt({self.expr})"


@dataclass(frozen=True)
class Inv(Expr):
    """Modular inverse: inv e"""

    expr: Expr

    @override
    def qtype(self) -> QType:
        from .qtype import Maybe

        return Maybe(self.expr.qtype())

    @override
    def optimize(self) -> Expr:
        # Dummy implementation - just optimize operand
        return Inv(self.expr.optimize())

    @override
    def circuit(self) -> Circuit:
        from .compile import CompilationContext, _compile_with_context

        # Create context at top level - this ensures CSE works across the entire expression tree
        ctx = CompilationContext()
        return _compile_with_context(self, ctx)

    @override
    def __str__(self) -> str:
        return f"inv({self.expr})"


@dataclass(frozen=True)
class Log(Expr):
    """Logarithm: log e"""

    base: Expr
    arg: Expr

    @override
    def qtype(self) -> QType:
        # TODO: Define type checking for log operation
        # For now, log of [N] might return [N] or need a different type
        # This is a placeholder - log operation needs proper type definition
        raise NotImplementedError("Log operation type checking not yet defined")

    @override
    def optimize(self) -> Expr:
        # Dummy implementation - just optimize operand
        return Log(self.base.optimize(), self.arg.optimize())

    @override
    def circuit(self) -> Circuit:
        from .compile import CompilationContext, _compile_with_context

        # Create context at top level - this ensures CSE works across the entire expression tree
        ctx = CompilationContext()
        return _compile_with_context(self, ctx)

    @override
    def __str__(self) -> str:
        return f"log({self.base}, {self.arg})"


# Special expression classes
@dataclass(frozen=True)
class Indexed(Expr):
    """Natural number indexed: N[n]"""

    N: int
    index: Expr

    @override
    def qtype(self) -> QType:
        # TODO: Define type checking for indexed operation
        # N[n] - need to determine the type based on N and index
        raise NotImplementedError("Indexed operation type checking not yet defined")

    @override
    def optimize(self) -> Expr:
        # Dummy implementation - just optimize index
        return Indexed(self.N, self.index.optimize())

    @override
    def circuit(self) -> Circuit:
        from .compile import CompilationContext, _compile_with_context

        # Create context at top level - this ensures CSE works across the entire expression tree
        ctx = CompilationContext()
        return _compile_with_context(self, ctx)

    @override
    def __str__(self) -> str:
        return f"{self.N}[{self.index}]"


@dataclass(frozen=True)
class Join(Expr):
    """Join operation: e₁ ⋈ e₂ (placeholder)"""

    left: Expr
    right: Expr

    @override
    def qtype(self) -> QType:
        # Dummy implementation - Join operation type needs to be defined
        raise NotImplementedError("Join operation type checking not yet defined")

    @override
    def optimize(self) -> Expr:
        # Dummy implementation
        return Join(self.left.optimize(), self.right.optimize())

    @override
    def circuit(self) -> Circuit:
        from .compile import CompilationContext, _compile_with_context

        # Create context at top level - this ensures CSE works across the entire expression tree
        ctx = CompilationContext()
        return _compile_with_context(self, ctx)

    @override
    def __str__(self) -> str:
        return f"({self.left} ⋈ {self.right})"
