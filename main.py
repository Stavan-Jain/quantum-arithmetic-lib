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

    # result = run(sum, {"x1": 5, "x2": 7})
    # print(f"Result: {result} (expected: 1, since (5 + 7) mod 11 = 1)")

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

    # result = run(diff, {"x1": 5, "x2": 6})
    # print(f"Result: {result} (expected: 10, since (5 - 6) mod 11 = 10)")

    # x3_expr, y3_expr = ecc_add(x1, y1, x2, y2)
    # print("x3 expression: ", x3_expr)
    # print("y3 expression: ", y3_expr)
    # print("x3 Circuit:")
    # print(x3_expr.circuit())
    # print("y3 Circuit:")
    # print(y3_expr.circuit())


if __name__ == "__main__":
    main()
