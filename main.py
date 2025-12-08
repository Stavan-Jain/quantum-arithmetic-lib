from qarith.expr import Basic
from qarith.qtype import Mod
from tests.ecc import ecc_add
from tests.simple import simple_addition


def main():
    # Create Basic expressions as inputs
    x1 = Basic("x1", Mod(11))
    y1 = Basic("y1", Mod(11))
    x2 = Basic("x2", Mod(11))
    y2 = Basic("y2", Mod(11))

    x3 = simple_addition(x1, x2)
    print("x3: ", x3)
    print(x3.circuit())


if __name__ == "__main__":
    main()
