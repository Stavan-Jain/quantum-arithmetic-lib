from qarith.expr import Basic
from qarith.qtype import Mod
from qarith import run
from tests.ecc import ecc_add
from tests.simple import simple_addition, simple_subtraction


def main():
    x1 = Basic("x1", Mod(11))
    y1 = Basic("y1", Mod(11))
    x2 = Basic("x2", Mod(11))
    y2 = Basic("y2", Mod(11))

    sum_expr = simple_addition(x1, x2)
    print("sum expression: ", sum_expr)
    # print("Circuit:")
    # print(sum_expr.circuit())

    result = run(sum_expr, {"x1": 5, "x2": 7})
    print(f"Result: {result} (expected: 1, since (5 + 7) mod 11 = 1)")

    diff_expr = simple_subtraction(x1, x2)
    print("diff expression: ", diff_expr)
    # print("Circuit:")
    # print(diff_expr.circuit())

    result = run(diff_expr, {"x1": 5, "x2": 6})
    print(f"Result: {result} (expected: 10, since (5 - 6) mod 11 = 10)")

    # x3_expr, y3_expr = ecc_add(x1, y1, x2, y2)
    # print("x3 expression: ", x3_expr)
    # print("y3 expression: ", y3_expr)
    # print("x3 Circuit:")
    # print(x3_expr.circuit())
    # print("y3 Circuit:")
    # print(y3_expr.circuit())


if __name__ == "__main__":
    main()
