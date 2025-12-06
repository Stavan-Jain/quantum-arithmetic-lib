from dataclasses import dataclass

@dataclass
class AdderResources:
    """Simple container for resource estimates of an n-bit adder."""
    n_bits: int
    toffoli_count: int
    cnot_count: int
    ancilla_count: int


