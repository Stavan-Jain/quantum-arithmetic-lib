"""Helpers for running adder circuits through an Aer simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Tuple

from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit import QuantumRegister
from qiskit_aer import AerSimulator

from qarith.adders.base import Adder
from qarith.adders.inplace_adder import InplaceRippleAdder
from qarith.adders.ripple_adder import NaiveRippleAdder


_AER_SIMULATOR = AerSimulator()


@dataclass(frozen=True)
class AdderSimulationResult:
    """Structured results returned by ``simulate_adder_sum``."""

    adder: str
    n_bits: int
    shots: int
    x_val: int
    y_val: int
    sum_value: int
    sum_bits: str
    carry_out: int | None
    raw_bitstring: str
    counts: Dict[str, int]


AdderFactory = type[Adder]
MeasureHook = Callable[[QuantumCircuit, int], None]
ParseHook = Callable[[str], Tuple[int, str, int | None, str]]


_ADDER_FACTORIES: Mapping[str, AdderFactory] = {
    "naive": NaiveRippleAdder,
    "naive_ripple": NaiveRippleAdder,
    "inplace": InplaceRippleAdder,
    "inplace_ripple": InplaceRippleAdder,
}


def simulate_adder_sum(
    adder: str,
    x_val: int,
    y_val: int,
    *,
    n_bits: int | None = None,
    shots: int = 1024,
) -> AdderSimulationResult:
    """Simulate an adder circuit on prepared inputs and return the measurement record.

    Args:
        adder: Which implementation to run (``naive`` or ``inplace`` aliases supported).
        x_val: Value encoded into the ``x`` register (LSB stored at index 0).
        y_val: Value encoded into the ``y`` register before applying the adder.
        n_bits: Optional override for the circuit size; defaults to the minimum number
            of bits that can represent both inputs.
        shots: Number of simulator shots passed to Aer (defaults to 1024).

    Returns:
        ``AdderSimulationResult`` capturing the measured sum, optional carry-out, and
        the raw shot counts from the simulator.
    """

    adder_key = adder.lower()
    adder_cls = _ADDER_FACTORIES.get(adder_key)
    if adder_cls is None:
        valid = ", ".join(sorted(set(_ADDER_FACTORIES.keys())))
        msg = f"Unknown adder '{adder}'. Expected one of: {valid}"
        raise ValueError(msg)

    if n_bits is None:
        n_bits = max(x_val.bit_length(), y_val.bit_length(), 1)

    _ensure_non_negative(x_val, "x_val")
    _ensure_non_negative(y_val, "y_val")
    _ensure_fits_in_bits(x_val, n_bits, "x_val")
    _ensure_fits_in_bits(y_val, n_bits, "y_val")

    adder_instance = adder_cls(n_bits)
    base = adder_instance.build_circuit()
    prepared = _prepare_inputs(base, x_val, y_val, n_bits)

    measure_fn, parse_fn = _build_hooks(adder_key)
    measure_fn(prepared, n_bits)

    compiled = transpile(prepared, _AER_SIMULATOR)
    job = _AER_SIMULATOR.run(compiled, shots=shots)
    result = job.result()
    counts = result.get_counts()

    winning = max(counts, key=counts.get)
    bitstring = winning.replace(" ", "")
    sum_value, sum_bits, carry, raw_bits = parse_fn(bitstring)

    return AdderSimulationResult(
        adder=adder_key,
        n_bits=n_bits,
        shots=shots,
        x_val=x_val,
        y_val=y_val,
        sum_value=sum_value,
        sum_bits=sum_bits,
        carry_out=carry,
        raw_bitstring=raw_bits,
        counts=counts,
    )


def _prepare_inputs(base_circ: QuantumCircuit, x_val: int, y_val: int, n_bits: int) -> QuantumCircuit:
    """Prepare |x_val>|y_val> before executing the provided circuit."""

    prep = QuantumCircuit(*base_circ.qregs)
    x_reg = _get_register(prep, "x")
    y_reg = _get_register(prep, "y")

    for i in range(n_bits):
        if (x_val >> i) & 1:
            prep.x(x_reg[i])
        if (y_val >> i) & 1:
            prep.x(y_reg[i])

    composed = prep.compose(base_circ)
    assert composed is not None
    return composed


def _build_hooks(adder_key: str) -> Tuple[MeasureHook, ParseHook]:
    if adder_key in {"naive", "naive_ripple"}:
        return _measure_naive, _parse_naive
    if adder_key in {"inplace", "inplace_ripple"}:
        return _measure_inplace, _parse_inplace
    msg = f"No measurement hooks registered for adder '{adder_key}'"
    raise ValueError(msg)


def _measure_naive(circ: QuantumCircuit, n_bits: int) -> None:
    meas = ClassicalRegister(n_bits + 1, "meas")
    circ.add_register(meas)
    s_reg = _get_register(circ, "s")
    cout_reg = _get_register(circ, "cout")

    for i in range(n_bits):
        circ.measure(s_reg[i], meas[i])
    circ.measure(cout_reg[n_bits - 1], meas[n_bits])


def _parse_naive(bitstring: str) -> Tuple[int, str, int, str]:
    carry_bit = bitstring[0] if bitstring else "0"
    sum_bits = bitstring[1:] if len(bitstring) > 1 else ""
    sum_val = int(sum_bits, 2) if sum_bits else 0
    return sum_val, sum_bits, int(carry_bit, 2), bitstring


def _measure_inplace(circ: QuantumCircuit, n_bits: int) -> None:
    meas = ClassicalRegister(n_bits, "meas")
    circ.add_register(meas)
    y_reg = _get_register(circ, "y")

    for i in range(n_bits):
        circ.measure(y_reg[i], meas[i])


def _parse_inplace(bitstring: str) -> Tuple[int, str, None, str]:
    sum_val = int(bitstring, 2) if bitstring else 0
    return sum_val, bitstring, None, bitstring


def _get_register(circ: QuantumCircuit, name: str) -> QuantumRegister:
    for reg in circ.qregs:
        if reg.name == name:
            return reg
    raise ValueError(f"Quantum register '{name}' not present in circuit.")


def _ensure_non_negative(value: int, label: str) -> None:
    if value < 0:
        raise ValueError(f"{label} must be non-negative, got {value}.")


def _ensure_fits_in_bits(value: int, n_bits: int, label: str) -> None:
    max_value = (1 << n_bits) - 1
    if value > max_value:
        raise ValueError(
            f"{label}={value} does not fit in {n_bits} bits (max {max_value}).",
        )
