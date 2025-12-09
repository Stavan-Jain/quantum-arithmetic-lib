namespace ECC {

    newtype ECPoint = (
        x : Int,
        y : Int,
        isInfinity : Bool
    );

    // ----------------------------------------
    // Utilities
    // ----------------------------------------

    function Mod(a : Int, p : Int) : Int {
        let r = a % p;
        if (r < 0) { r + p } else { r }
    }

    function ModAdd(a : Int, b : Int, p : Int) : Int {
        Mod(a + b, p)
    }

    function ModSub(a : Int, b : Int, p : Int) : Int {
        Mod(a - b, p)
    }

    function ModMul(a : Int, b : Int, p : Int) : Int {
        Mod(a * b, p)
    }

    // ----------------------------------------
    // Recursive extended Euclidean algorithm
    // (Q# 1.0 has no while loops)
    // ----------------------------------------

    function Egcd(a : Int, b : Int) : (Int, Int, Int) {
        if (b == 0) {
            (a, 1, 0)
        } else {
            let (g, x1, y1) = Egcd(b, a % b);
            let x = y1;
            let y = x1 - (a / b) * y1;
            (g, x, y)
        }
    }

    function ModInv(a : Int, p : Int) : Int {
        let (g, x, _) = Egcd(Mod(a, p), p);
        if (g != 1) {
            fail "ModInv: number is not invertible modulo p.";
        }
        Mod(x, p)
    }

    // ----------------------------------------
    // Point printing
    // ----------------------------------------

    function ShowPoint(P : ECPoint) : String {
        let x1 = P::x;
        let y1 = P::y;
        let inf = P::isInfinity;
        if (inf) {
            "ECPoint(Infinity)"
        } else {
            $"ECPoint(x={x1}, y={y1})"
        }
    }

    // ----------------------------------------
    // Elliptic curve addition
    // ----------------------------------------

    function ECAdd(p : Int, aCurve : Int, P : ECPoint, Q : ECPoint) : ECPoint {

        let x1 = P::x;
        let y1 = P::y;
        let inf1 = P::isInfinity;

        let x2 = Q::x;
        let y2 = Q::y;
        let inf2 = Q::isInfinity;

        // handle infinity cases
        if (inf1) { return Q; }
        if (inf2) { return P; }

        // P + (-P) = ∞
        if (x1 == x2 and ModAdd(y1, y2, p) == 0) {
            return ECPoint(0, 0, true);
        }

        // Compute slope λ
        let lambda = if (x1 == x2 and y1 == y2) {
            // doubling
            let numer = Mod(3 * x1 * x1 + aCurve, p);
            let denom = ModMul(2, y1, p);
            ModMul(numer, ModInv(denom, p), p)
        } else {
            // addition
            let numer = ModSub(y2, y1, p);
            let denom = ModSub(x2, x1, p);
            ModMul(numer, ModInv(denom, p), p)
        };

        let x3 = ModSub(ModSub(Mod(lambda * lambda, p), x1, p), x2, p);
        let y3 = ModSub(ModMul(lambda, ModSub(x1, x3, p), p), y1, p);

        ECPoint(x3, y3, false)
    }

    // ----------------------------------------
    // Test entry point
    // ----------------------------------------

    @EntryPoint()
    operation RunECCTest() : Unit {
        let p = 97;
        let a = 2;

        let P = ECPoint(3, 6, false);
        let Q = ECPoint(80, 10, false);

        let R = ECAdd(p, a, P, Q);

        Message($"P + Q = {ShowPoint(R)}");
    }
}
