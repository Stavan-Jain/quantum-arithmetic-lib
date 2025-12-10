from qarith.expr import Var
from qarith.qtype import Mod
from qarith import run
from tests.ecc import ecc_add


def main():
    x1 = Var("x1", Mod(11))
    y1 = Var("y1", Mod(11))
    x2 = Var("x2", Mod(11))
    y2 = Var("y2", Mod(11))

    sum = x1 + x2
    print("sum expression: ", sum)
    print("Circuit:")
    print(sum.circuit())

    diff = y1 - y2
    print("diff expression: ", diff)
    print("Circuit:")
    print(diff.circuit())

    times = x1 * x2
    print("times expression: ", times)
    print("Circuit:")
    print(times.circuit())

    pow = 3**x2
    print("pow expression: ", pow)
    print("Circuit:")
    print(pow.circuit())

    compound = x1 + x2 * y1 - y2
    print("compound expression: ", compound)
    print("Circuit:")
    print(compound.circuit())
    compound.circuit().save_png("compound.png")


if __name__ == "__main__":
    main()
