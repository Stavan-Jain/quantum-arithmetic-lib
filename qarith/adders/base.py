"""Common abstractions shared by adder implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from qiskit import QuantumCircuit

from qarith.resources.helpers import getGateCount
from qarith.adders.types import AdderResources


@dataclass
class AdderProfile:
    """Summary of an adder implementation for a fixed problem size."""

    circuit: QuantumCircuit
    resource_estimate: AdderResources
    ccx_count: int
    cx_count: int


class Adder(ABC):
    """Abstract base class for concrete adders."""

    def __init__(self, n_bits: int) -> None:
        if n_bits <= 0:
            msg = "n_bits must be positive"
            raise ValueError(msg)
        self.n_bits = n_bits

    @abstractmethod
    def build_circuit(self) -> QuantumCircuit:
        """Return a circuit implementing the adder."""

    @abstractmethod
    def resource_estimate(self) -> AdderResources:
        """Return a coarse model of expected resources."""

    def characterize(self) -> AdderProfile:
        """Build the circuit and compute both model and actual resources."""

        circuit = self.build_circuit()
        ccx_count, cx_count = getGateCount(circuit)
        resources = self.resource_estimate()
        return AdderProfile(
            circuit=circuit,
            resource_estimate=resources,
            ccx_count=ccx_count,
            cx_count=cx_count,
        )

    def describe(self) -> str:
        """Return a concise textual summary of the adder."""

        profile = self.characterize()
        resource_summary = profile.resource_estimate
        counts_summary = f"ccx={profile.ccx_count}, cx={profile.cx_count}"
        return (
            f"{self.__class__.__name__}(n_bits={self.n_bits})\n"
            f"  model resources: {resource_summary}\n"
            f"  actual gates: {counts_summary}"
        )
