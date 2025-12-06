from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional as PyOptional
import math

from .register import QRegister
from .qtype import QType, Int, Mod, is_optional


@dataclass
class Context:
    """
    Context for type checking and quantum register allocation.

    Manages:
    - Variable type bindings (Γ)
    - Quantum register allocations
    - Modulus information for Mod<N> types
    """

    # Type bindings: var_name -> QType
    type_env: Dict[str, QType] = field(default_factory=dict)

    # Quantum register bindings: var_name -> QRegister
    registers: Dict[str, QRegister] = field(default_factory=dict)

    # Next available qubit index
    _next_qubit: int = 0

    def declare(self, name: str, ty: QType) -> QRegister:
        """
        Declare a new variable with the given type and allocate a quantum register.

        Returns the allocated QRegister.
        """
        if name in self.type_env:
            raise ValueError(f"Variable {name} already declared")

        # Calculate number of bits needed
        if isinstance(ty, Int):
            # For Int, we need to determine bit width from context
            # For now, use a default or require explicit specification
            n_bits = 8  # default, could be configurable
        elif isinstance(ty, Mod):
            n_bits = math.ceil(math.log2(ty.N))
        elif is_optional(ty):
            # For optional types, use the inner type
            inner_ty = ty.inner
            if isinstance(inner_ty, Mod):
                n_bits = math.ceil(math.log2(inner_ty.N))
            else:
                n_bits = 8  # default for Int?
        else:
            raise TypeError(f"Cannot allocate register for type {ty}")

        # Allocate register
        reg = QRegister(
            name=name,
            n_bits=n_bits,
            offset=self._next_qubit,
            ty=ty,
        )
        self._next_qubit += n_bits

        # Update bindings
        self.type_env[name] = ty
        self.registers[name] = reg

        return reg

    def lookup_type(self, name: str) -> QType:
        """Look up the type of a variable."""
        if name not in self.type_env:
            raise ValueError(f"Variable {name} not declared")
        return self.type_env[name]

    def lookup_register(self, name: str) -> QRegister:
        """Look up the quantum register for a variable."""
        if name not in self.registers:
            raise ValueError(f"Variable {name} not declared")
        return self.registers[name]

    def get_modulus(self) -> PyOptional[int]:
        """
        Get the modulus from the context.

        If all variables are Mod<N> with the same N, return N.
        Otherwise return None.
        """
        moduli = set()
        for ty in self.type_env.values():
            if isinstance(ty, Mod):
                moduli.add(ty.N)
            elif is_optional(ty) and isinstance(ty.inner, Mod):
                moduli.add(ty.inner.N)

        if len(moduli) == 1:
            return moduli.pop()
        return None

    @property
    def n_qubits(self) -> int:
        """Total number of qubits allocated so far."""
        return self._next_qubit

    def max_bit_width(self) -> int:
        """Find the maximum bit width used in any register."""
        if not self.registers:
            return 0
        return max(reg.n_bits for reg in self.registers.values())
