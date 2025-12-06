from qarith.prog import Program
from qarith.qtype import Mod


def main():
    # Create a program context
    prog = Program()

    # Declare variables naturally - no need to specify names!
    a = prog.declare_var(Mod(11))
    b = prog.declare_var(Mod(11))

    # Initialize variables using assign() method
    # Note: Due to Python limitations, we can't intercept 'a = 0' directly
    # Use one of these alternatives:
    a.assign(1)  # Method 1: explicit assign
    prog[b] = 1  # Method 2: using program indexing
    a.assign(a + b)

    # # In-place operators automatically add operations
    # a += b  # Addition
    # a -= b  # Subtraction
    # a *= b  # Multiplication
    # a **= b  # Exponentiation
    # a //= b  # Floor division

    # Compile and run the circuit
    print("Compiling circuit...")
    circ = prog.compile()
    print("Circuit (all operations):")
    print("Total qubits:", circ.n_qubits)
    circ.pretty_print()

    # Run the circuit and get results
    print("\nRunning circuit...")
    results = prog.run(shots=1024)
    print("Results:", results)
    print(f"Final value of a: {results.get('var_0', 'N/A')}")
    print(f"Final value of b: {results.get('var_1', 'N/A')}")


if __name__ == "__main__":
    main()
