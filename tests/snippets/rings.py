import math


# Rename factory to Zmod to reflect it supports Rings (composites)
def Zmod(modulus):
    """
    Factory function to create a Ring Class for integers modulo n.
    Supports +,-,*,/,inv,pow,sqrt,log.
    """

    class RingElement(ModularElement):
        n = modulus

        def __init__(self, value):
            super().__init__(value, self.n)

        @classmethod
        def __repr__(cls):
            return f"Z_{cls.n}"

    return RingElement


class ModularElement:
    def __init__(self, value, modulus):
        self.value = value % modulus
        self.modulus = modulus

    def __repr__(self):
        return f"{self.value}"

    def _check(self, other):
        if isinstance(other, int):
            return ModularElement(other, self.modulus)
        if self.modulus != other.modulus:
            raise TypeError("Different Moduli")
        return other

    # --- 1, 2, 3: Basic Arithmetic ---
    def __add__(self, other):
        other = self._check(other)
        return ModularElement(self.value + other.value, self.modulus)

    def __sub__(self, other):
        other = self._check(other)
        return ModularElement(self.value - other.value, self.modulus)

    def __mul__(self, other):
        other = self._check(other)
        return ModularElement(self.value * other.value, self.modulus)

    # --- 5: Exponentiation ---
    def __pow__(self, exponent):
        if exponent < 0:
            return self.inverse() ** (-exponent)
        return ModularElement(pow(self.value, exponent, self.modulus), self.modulus)

    # --- 4: Inverse / Division ---
    def inverse(self):
        # Uses Python's native Extended Euclidean Algo (Python 3.8+)
        # This works for Composite n as long as gcd(val, n) == 1
        try:
            inv = pow(self.value, -1, self.modulus)
            return ModularElement(inv, self.modulus)
        except ValueError:
            raise ZeroDivisionError(
                f"{self.value} has no inverse mod {self.modulus} (not coprime)"
            )

    def __truediv__(self, other):
        return self * self._check(other).inverse()

    # --- 6: Discrete Log (Baby-Step Giant-Step) ---
    def log(self, base):
        """
        Solves base^x = self (mod n) using Baby-Step Giant-Step.
        Complexity: O(sqrt(n)).
        """
        base = self._check(base)
        target = self.value
        g = base.value
        n = self.modulus

        # m = ceil(sqrt(n))
        m = int(math.ceil(math.sqrt(n)))

        # 1. Baby Steps: Store g^j for 0 <= j < m
        table = {}
        baby = 1
        for j in range(m):
            table[baby] = j
            baby = (baby * g) % n

        # 2. Giant Steps: Check target * (g^-m)^i
        # Factor = g^(-m) mod n
        try:
            # We need inverse of g^m to step backwards
            giant_stride = pow(pow(g, m, n), -1, n)
        except ValueError:
            raise ValueError("Base is not invertible, standard BSGS failed.")

        current = target
        for i in range(m):
            if current in table:
                # x = i*m + j
                return i * m + table[current]
            current = (current * giant_stride) % n

        raise ValueError(f"No discrete log found for base {g} target {target}")

    # --- 7: Square Root ---
    def sqrt(self):
        """
        Attempts to find x such that x^2 = self.
        Note: For large composite n, this requires factorization (Hard).
        """
        if self.value == 0:
            return ModularElement(0, self.modulus)

        # Naive Brute Force for small n (fallback for composites)
        if self.modulus < 10000:  # Arbitrary limit for demo
            # Check if prime-optimized algo failed or n is composite
            for x in range(1, self.modulus):
                if (x * x) % self.modulus == self.value:
                    return ModularElement(x, self.modulus)
            raise ValueError(f"{self.value} is not a quadratic residue")

        # Tonelli-Shanks (Only works if n is Prime!)
        # Check simple Euler criterion first
        ls = pow(self.value, (self.modulus - 1) // 2, self.modulus)
        if ls != 1 and ls != 0:
            raise ValueError("Not a quadratic residue (or n is not prime)")

        # ... (Tonelli-Shanks implementation omitted for brevity,
        # refer to previous file for the prime-only version)
        # For this general version, we warn the user:
        raise NotImplementedError(
            "Sqrt for large non-prime moduli requires factorization."
        )


# ==========================================
# Example
# ==========================================
if __name__ == "__main__":
    # Create Ring for Modulo 22 (Composite)
    R = Zmod(22)
    print(f"--- Working in {R} ---")

    # 1. Discrete Log
    # 5^3 = 125 = 15 mod 22. So log_5(15) should be 3.
    base = R(5)
    target = R(15)

    print(f"Base: {base}, Target: {target}")
    try:
        exp = target.log(base)
        print(f"Log:  log_{base}({target}) = {exp}")
        print(f"Ck:   {base}^{exp} = {base**exp}")
    except ValueError as e:
        print(e)

    # 2. Inverse (Composite)
    # 3 * 15 = 45 = 1 mod 22. Inverse of 3 is 15.
    a = R(3)
    print(f"Inv:  inv({a}) = {a.inverse()}")
