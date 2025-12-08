from qarith.expr import Inv, Expr


def ecc_add(x1: Expr, y1: Expr, x2: Expr, y2: Expr) -> (Expr, Expr):
    k = (y1 - y2) * Inv(x1 - x2)
    x3 = k**2 - x1 - x2
    y3 = k * (x1 - x3) - y1
    return (x3, y3)


def ecc_double(a: Expr, x: Expr, y: Expr) -> (Expr, Expr):
    s = (3 * x**2 + a) * Inv(2 * y)
    x3 = s**2 - 2 * x
    y3 = s * (x - x3) - y
    return (x3, y3)
