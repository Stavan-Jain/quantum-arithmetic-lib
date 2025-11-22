# Quantum Arithmetic Library

Experiments for building and characterizing quantum
arithmetic operators with Qiskit. The library currently focuses on ripple-carry adders and
offers both out-of-place and in-place variants together with resource
estimators and helper utilities.

## Features
- Abstract `Adder` base class that standardizes circuit synthesis and resource
  bookkeeping through `characterize()` and `describe()` helpers.
- `NaiveRippleAdder` builds a ripple-carry circuit that writes the
  sum and carries into dedicated registers.
- `InplaceRippleAdder` performs the addition in-place, returning the carry
  ancillae to `|0⟩`.
- `qarith.resources.helpers.getGateCount()` counts `ccx` and `cx` operations in
  any `QuantumCircuit`, making it easy to correlate resource models with the
  actual circuit.

## Repository layout
```
quantum-arithmetic-lib/
├── qarith/
│   ├── adders/
│   │   ├── base.py              # Shared Adder API and AdderProfile dataclass
│   │   ├── inplace_adder.py     # In-place ripple-carry adder implementation
│   │   └── ripple_adder.py      # Naive out-of-place ripple-carry adder
│   ├── resources/helpers.py     # Gate counting helper
│   └── types.py                 # AdderResources dataclass
```

## Getting started
```bash
python -m venv .venv
source .venv/bin/activate
pip install qiskit
```

## Running the sample adders
Both adders expose a small `run()` helper and can be executed as modules. These
commands build the corresponding circuit for `n=2` (or any `n` you pass to the
`run()` function) and print the modelled and measured gate counts.

```bash
# Prints the entire circuit plus model vs. actual gate counts
python -m qarith.adders.ripple_adder

# Prints resource model and ccx/cx histogram for the in-place adder
python -m qarith.adders.inplace_adder
```

## Programmatic usage

Each `Adder` subclass is responsible for `build_circuit()` and
`resource_estimate()`, so extending the library is as simple as deriving from
`Adder` and returning the proper dataclass.

