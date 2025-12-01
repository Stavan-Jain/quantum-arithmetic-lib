from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Union
import math


@dataclass
class QInt:
    """
    Quantum integer backed by a contiguous block of qubits.

    A QInt never stores the modulus itself; it only knows which Block it
    belongs to. All arithmetic is implicitly modulo that Block's modulus,
    so user code can write:

        block = Block(modulus=N)
        a = block.new_int("a")
        b = block.new_int("b")
        a = a + b  # appends a ModAdd to block.program.ops
        b = b - a  # appends a ModSub to block.program.ops
    """

    name: str
    n_bits: int
    offset: int  # starting qubit index
    _block: Block

    def bits(self) -> List[int]:
        return [self.offset + i for i in range(self.n_bits)]

    @property
    def modulus(self) -> int:
        """Return the arithmetic modulus for this integer."""
        return self._block.modulus

    # --- small DSL for in-place modular arithmetic on this QInt ---

    def _check_compatible(self, other: QInt) -> None:
        if self._block is not other._block:
            raise ValueError("Cannot combine QInts from different Blocks")

    def __add__(self, other: QInt) -> QInt:
        """Sugar for: self = self + other (mod modulus)."""
        self._check_compatible(other)
        # Record a modular addition in the owning Block's Program.
        self._block.program.add_mod(dst=self, src=other)
        return self

    def __sub__(self, other: QInt) -> QInt:
        """Sugar for: self = self - other (mod modulus)."""
        self._check_compatible(other)
        self._block.program.sub_mod(dst=self, src=other)
        return self


@dataclass
class ModAdd:
    """Modular addition: dst = dst + src (mod dst.modulus).

    Since all arithmetic is modular, the modulus is stored in the QInt objects
    themselves, not duplicated here.
    """

    dst: QInt
    src: QInt

    @property
    def modulus(self) -> int:
        """Get the modulus from dst (src.modulus should be the same)."""
        return self.dst.modulus


@dataclass
class ModSub:
    """Modular subtraction: dst = dst - src (mod dst.modulus).

    Since all arithmetic is modular, the modulus is stored in the QInt objects
    themselves, not duplicated here.
    """

    dst: QInt
    src: QInt

    @property
    def modulus(self) -> int:
        """Get the modulus from dst (src.modulus should be the same)."""
        return self.dst.modulus


@dataclass
class ModExp:
    dst: QInt  # accumulator: dst <- dst * base^exp mod N
    base: int  # classical base
    exp: QInt  # exponent register (quantum)
    modulus: int


Op = Union[ModAdd, ModSub, ModExp]


@dataclass
class Program:
    qints: List[QInt] = field(default_factory=list)
    ops: List[Op] = field(default_factory=list)
    _next_qubit: int = 0

    def _new_qint(self, name: str, n_bits: int, block: Block) -> QInt:
        """Allocate a new QInt within the global Program, owned by `block`."""
        qi = QInt(
            name=name,
            n_bits=n_bits,
            offset=self._next_qubit,
            _block=block,
        )
        self._next_qubit += n_bits
        self.qints.append(qi)
        return qi

    def add_mod(self, dst: QInt, src: QInt):
        """Add src to dst modulo the Block modulus.

        We assume both QInts belong to the same Block; this is enforced by
        QInt._check_compatible at the DSL layer.
        """
        self.ops.append(ModAdd(dst, src))

    def sub_mod(self, dst: QInt, src: QInt):
        """Subtract src from dst modulo the Block modulus."""
        self.ops.append(ModSub(dst, src))

    def exp_mod(self, dst: QInt, base: int, exp: QInt, modulus: int):
        self.ops.append(ModExp(dst, base, exp, modulus))

    def _max_bit_width(self) -> int:
        """Find the maximum bit width used in any arithmetic operation."""
        max_bit_width = 0
        for op in self.ops:
            if isinstance(op, (ModAdd, ModSub)):
                max_bit_width = max(max_bit_width, op.dst.n_bits, op.src.n_bits)
            elif isinstance(op, ModExp):
                max_bit_width = max(max_bit_width, op.dst.n_bits, op.exp.n_bits)
        return max_bit_width

    @property
    def n_qubits(self) -> int:
        """
        Calculate total qubits needed, including ancillas for arithmetic operations.

        The in-place ripple-carry adder needs max_bit_width + 1 carry qubits
        for each addition/subtraction operation. We allocate enough ancillas
        to handle the largest operation in the program, plus one flag qubit.
        """
        max_bit_width = self._max_bit_width()
        # Allocate: max_bit_width + 1 carry qubits + 1 flag qubit
        # If no operations yet, default to 2 ancillas (minimal)
        ancilla_count = (max_bit_width + 1) + 1 if max_bit_width > 0 else 2
        return self._next_qubit + ancilla_count

    @property
    def carry_ancilla_start(self) -> int:
        """Starting index for carry ancilla qubits."""
        return self._next_qubit

    @property
    def flag_qubit_index(self) -> int:
        """Index of the flag qubit (used for modular reduction comparisons)."""
        max_bit_width = self._max_bit_width()
        carry_count = (max_bit_width + 1) if max_bit_width > 0 else 1
        return self._next_qubit + carry_count


@dataclass
class Block:
    """A block/module of arithmetic over a fixed modulus N.

    This is a thin wrapper over Program that bakes in a single modulus, so
    that user code does not need to keep passing N around:

        block = Block(modulus=15)
        a = block.new_int("a")
        b = block.new_int("b")
        e = block.new_int("e")
        a = a + b
        b = b - a
        block.exp(a, base=7, exp=e)
    """

    modulus: int
    program: Program = field(default_factory=Program)

    def new_int(self, name: str) -> QInt:
        """Create a new QInt in this Block, using the Block's modulus."""
        n_bits = math.ceil(math.log2(self.modulus))
        return self.program._new_qint(name=name, n_bits=n_bits, block=self)

    def exp(self, dst: QInt, base: int, exp: QInt):
        """dst <- dst * base^exp (mod N) within this block."""
        self.program.exp_mod(dst=dst, base=base, exp=exp, modulus=self.modulus)
