from dataclasses import dataclass, field
from typing import List, Dict
from .context import Context
from .register import QRegister
from .qtype import QType
from .expr import TypedExpr
from .gates import Circuit


@dataclass
class Program:
    """
    A program manages the context for quantum register allocation.

    This is a built-in DSL for Qiskit, so we only need to manage
    variable declarations and register allocation.
    """

    context: Context = field(default_factory=Context)
    _var_counter: int = field(default=0, init=False)
    _operations: List[tuple] = field(
        default_factory=list, init=False, repr=False
    )  # List of (expr, target_var) tuples

    def declare_var(self, ty: QType) -> QRegister:
        """
        Declare a new variable with the given type and return its register.

        The variable name is automatically generated.
        """
        var_name = f"var_{self._var_counter}"
        self._var_counter += 1
        reg = self.context.declare(var_name, ty)
        reg._program = self
        return reg

    def __setitem__(self, key: QRegister, value):
        """
        Support assignment syntax: prog[a] = 0 or prog[a] = a + b

        This allows: prog[a] = 0  instead of a.assign(0)
        """
        key.assign(value)

    def lookup_var(self, name: str) -> QRegister:
        """Look up a variable's register."""
        return self.context.lookup_register(name)

    @property
    def n_qubits(self) -> int:
        """
        Calculate total qubits needed, including ancillas for arithmetic operations.

        The in-place ripple-carry adder needs max_bit_width + 1 carry qubits
        for each addition/subtraction operation. We allocate enough ancillas
        to handle the largest operation in the program, plus one flag qubit.
        """
        max_bit_width = self.context.max_bit_width()
        # Allocate: max_bit_width + 1 carry qubits + 1 flag qubit
        # If no operations yet, default to 2 ancillas (minimal)
        ancilla_count = (max_bit_width + 1) + 1 if max_bit_width > 0 else 2
        return self.context.n_qubits + ancilla_count

    @property
    def carry_ancilla_start(self) -> int:
        """Starting index for carry ancilla qubits."""
        return self.context.n_qubits

    @property
    def flag_qubit_index(self) -> int:
        """Index of the flag qubit (used for modular reduction comparisons)."""
        max_bit_width = self.context.max_bit_width()
        carry_count = (max_bit_width + 1) if max_bit_width > 0 else 1
        return self.context.n_qubits + carry_count

    def add_operation(self, expr, target_var: str):
        """
        Add an operation to the program.

        The operation will be compiled when compile() is called.
        """
        if not isinstance(expr, TypedExpr):
            raise TypeError(f"Expected TypedExpr, got {type(expr)}")
        self._operations.append((expr, target_var))

    def compile(self, expr=None, target_var: str = None) -> Circuit:
        """
        Compile the program to a quantum circuit.

        If expr and target_var are provided, compile a single expression.
        Otherwise, compile all operations in the program.

        Returns:
            A quantum circuit implementing the program or expression.
        """
        # Import here to avoid circular import
        from .compile import compile_expr, _compile_program

        if expr is not None and target_var is not None:
            if not isinstance(expr, TypedExpr):
                raise TypeError(f"Expected TypedExpr, got {type(expr)}")
            self.add_operation(expr, target_var)
            return compile_expr(expr, target_var, self)
        else:
            return _compile_program(self)

    # TODO: fix this.
    def run(self, shots: int = 1024, backend=None) -> Dict[str, int]:
        """
        Compile, run the circuit, and return the measurement results.

        Args:
            shots: Number of measurement shots to run
            backend: Qiskit backend to use. If None, uses Aer's statevector_simulator
                     for exact results (shots=1) or qasm_simulator for multiple shots.

        Returns:
            Dictionary mapping variable names to their measured values.
            For multiple shots, returns the most frequent result.
        """
        # Debug: print operations before compiling
        print(f"Debug: Operations in program: {len(self._operations)}")
        for i, (expr, target_var) in enumerate(self._operations):
            print(f"  {i}: {target_var} = {expr}")

        # Compile the circuit
        circ = self.compile()
        registers_to_measure = list(self.context.registers.values())
        circ.measure(registers_to_measure)
        if backend is None:
            try:
                from qiskit_aer import Aer

                if shots == 1:
                    # Use statevector for exact result
                    backend = Aer.get_backend("statevector_simulator")
                else:
                    backend = Aer.get_backend("qasm_simulator")
            except ImportError:
                try:
                    from qiskit import Aer

                    if shots == 1:
                        backend = Aer.get_backend("statevector_simulator")
                    else:
                        backend = Aer.get_backend("qasm_simulator")
                except ImportError:
                    raise ImportError(
                        "Qiskit Aer is required for running circuits. "
                        "Install it with: pip install qiskit-aer"
                    )

        # Run the circuit
        # Use backend.run() instead of execute() (new Qiskit API)
        if shots == 1 and "statevector" in str(backend):
            # Use statevector for exact result
            job = backend.run(circ.quantum_circuit, shots=1)
            result = job.result()
            statevector = result.get_statevector()

            # Extract results from statevector
            # Find the state with highest probability
            max_prob = 0
            max_state = 0
            for i, amplitude in enumerate(statevector):
                prob = abs(amplitude) ** 2
                if prob > max_prob:
                    max_prob = prob
                    max_state = i

            # Parse the state into variable values
            results = {}
            bit_offset = 0
            for var_name, reg in sorted(self.context.registers.items()):
                # Extract bits for this register from the state
                value = 0
                for i in range(reg.n_bits):
                    if (max_state >> (bit_offset + i)) & 1:
                        value |= 1 << i
                results[var_name] = value
                bit_offset += reg.n_bits

            return results
        else:
            # Use qasm simulator for multiple shots
            job = backend.run(circ.quantum_circuit, shots=shots)
            result = job.result()
            counts = result.get_counts()

            # Find the most frequent result
            if not counts:
                raise RuntimeError("No measurement results obtained")

            most_frequent = max(counts.items(), key=lambda x: x[1])
            bit_string = most_frequent[0]

            # Debug: print counts to see what we're getting
            print(f"Debug: counts = {counts}")
            print(
                f"Debug: most frequent bit string = {bit_string} (length={len(bit_string)})"
            )

            # Get registers in sorted order (by name) to match measurement order
            sorted_regs = sorted(self.context.registers.items())
            total_data_bits = sum(reg.n_bits for _, reg in sorted_regs)
            print(
                f"Debug: Total data bits = {total_data_bits}, registers = {[(name, reg.n_bits) for name, reg in sorted_regs]}"
            )

            # Parse the bit string into variable values
            # Qiskit returns bit strings with MSB first (leftmost is MSB)
            # The measurement order: we measure qubits in order (qubit 0, 1, 2, ...)
            # In the bit string: bit_string[0] is MSB (highest qubit), bit_string[-1] is LSB (qubit 0)
            # But we only measured data qubits, so the bit string should only contain data bits
            # Reverse to get LSB first (qubit 0 first)
            results = {}

            # Reverse the bit string to get LSB first (qubit 0 is first)
            # Only use the first total_data_bits (in case we measured ancillas too)
            if len(bit_string) > total_data_bits:
                # If we measured more bits than data bits, take only the data bits
                # The data bits are the first total_data_bits in the reversed string
                reversed_bits = bit_string[::-1][:total_data_bits]
            else:
                reversed_bits = bit_string[::-1]

            print(f"Debug: Reversed bits (LSB first) = {reversed_bits}")

            bit_offset = 0
            for var_name, reg in sorted_regs:
                # Extract bits for this register
                # reg.bits() returns [offset, offset+1, ..., offset+n_bits-1]
                # These are measured in order, so bit_offset corresponds to the first bit
                value = 0
                for i in range(reg.n_bits):
                    bit_index = bit_offset + i
                    if bit_index < len(reversed_bits):
                        if reversed_bits[bit_index] == "1":
                            value |= 1 << i  # bit i is the i-th bit (LSB is i=0)
                print(
                    f"Debug: {var_name} = {value} (bits {reg.bits()}, extracted from positions {bit_offset} to {bit_offset + reg.n_bits - 1})"
                )
                results[var_name] = value
                bit_offset += reg.n_bits

            return results
