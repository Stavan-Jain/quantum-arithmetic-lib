from qarith.expr import Var
from qarith.qtype import Mod
from tests.ecc import ecc_add


def main():
    x1 = Var("x1", Mod(11))
    y1 = Var("y1", Mod(11))
    x2 = Var("x2", Mod(11))
    y2 = Var("y2", Mod(11))

    sum = x1 + x2
    print("sum expression: ", sum)
    print("Circuit:")
    sum_circ = sum.circuit()
    print(sum_circ)
    print(sum_circ.decompose())

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
    # compound.circuit().save_png("compound.png")

    x3, y3 = ecc_add(x1, y1, x2, y2)
    print("x3 expression: ", x3)
    print("y3 expression: ", y3)
    print("Circuit:")
    print(x3.circuit())
    print(y3.circuit())


if __name__ == "__main__":
    main()
