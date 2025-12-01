from qarith.compiler import compile
from qarith.ir import Block


def main():
    block = Block(modulus=101)
    a = block.new_int("a")
    b = block.new_int("b")

    a += b
    b = b - a
    circ = compile(block)
    print("Total qubits:", circ.n_qubits)
    circ.pretty_print()


if __name__ == "__main__":
    main()
