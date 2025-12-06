from sage.all import *

# ==========================================
# SageMath Modular Arithmetic DSL Demo
# ==========================================
# To run this, you need SageMath installed.
# Online runner: https://sagecell.sagemath.org/

def demonstrate_modular_dsl():
    # 1. Define the Field (The "World")
    # We choose a prime p. 
    # p = 17 is small for reading, but this works for massive primes too.
    p = 17
    F = GF(p)
    print(f"--- Environment: Integers Modulo {p} ---")

    # 2. Define Elements
    # In this DSL, 'a' and 'b' are not just numbers, they are "Field Elements"
    a = F(7)
    b = F(3)
    
    print(f"a = {a}")
    print(f"b = {b}")

    # 3. Basic Arithmetic (+, -, *)
    # Operations automatically wrap around p
    add_res = a + b
    sub_res = a - b
    mul_res = a * b
    
    print(f"\n[Addition]       {a} + {b} = {add_res}")
    print(f"[Subtraction]    {a} - {b} = {sub_res}")
    print(f"[Multiplication] {a} * {b} = {mul_res}")

    # 4. Division (Modular Inverse)
    # Division is technically multiplication by the inverse: a * b^(-1)
    # Sage handles finding the inverse automatically.
    div_res = a / b
    print(f"\n[Division]       {a} / {b} = {div_res}")
    
    # Verification of division:
    # If (a / b) = x, then x * b should equal a
    check = div_res * b
    print(f"   (Check: {div_res} * {b} = {check})")

    # 5. Square Roots (Quadratic Residue)
    # Not every number has a square root in a finite field.
    # We pick a number we know is a square: 4^2 = 16.
    square_val = F(16)
    
    # Sage uses the Tonelli-Shanks algorithm internally
    try:
        root = square_val.sqrt()
        print(f"\n[Square Root]    sqrt({square_val}) = {root}")
        print(f"   (Check: {root}^2 = {root^2})")
        
        # Note: There are usually two roots (x and -x). 
        # Sage returns one. The other is -root.
        print(f"   (Other root: -{root} = {-root})")
    except ValueError:
        print(f"\n[Square Root]    {square_val} is not a quadratic residue (no integer root exists).")

    # 6. Handling Non-Residues
    # Let's try to find the square root of a number that doesn't have one (e.g., 3 mod 17)
    non_residue = F(3)
    try:
        print(f"\n[Attempt Sqrt]   sqrt({non_residue})...")
        print(non_residue.sqrt())
    except Exception as e:
        print(f"   Error: {e}")
        print("   (This is normal: 3 has no square root mod 17)")

if __name__ == "__main__":
    demonstrate_modular_dsl()