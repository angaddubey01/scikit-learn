"""
Mathematical validation of the fowlkes_mallows_score fix.

This script demonstrates that:
np.sqrt(tk / pk) * np.sqrt(tk / qk) === tk / np.sqrt(pk * qk)

The advantage of the first form is that it avoids potential integer overflow
when pk and qk are very large.
"""

import numpy as np

def original_formula(tk, pk, qk):
    """Original formula used in fowlkes_mallows_score."""
    return tk / np.sqrt(pk * qk) if tk != 0 else 0

def fixed_formula(tk, pk, qk):
    """Fixed formula that avoids integer overflow."""
    return np.sqrt(tk / pk) * np.sqrt(tk / qk) if tk != 0 else 0

# Test with normal values
print("Testing with normal values:")
tk, pk, qk = 100, 200, 400
print(f"Original formula: {original_formula(tk, pk, qk)}")
print(f"Fixed formula: {fixed_formula(tk, pk, qk)}")
print(f"Difference: {abs(original_formula(tk, pk, qk) - fixed_formula(tk, pk, qk))}")
print()

# Test with large values that could cause overflow in original formula
print("Testing with large values that could cause overflow:")
# These values would cause pk * qk to exceed int32 limit but are safely handled by the new formula
tk, pk, qk = 10**7, 10**16, 10**16  
print(f"Fixed formula: {fixed_formula(tk, pk, qk)}")
print("Note: Original formula would likely cause overflow with these values")

# Mathematical proof of equivalence
print("\nMathematical proof of equivalence:")
print("tk / np.sqrt(pk * qk)")
print("= tk / (sqrt(pk) * sqrt(qk))")
print("= (tk / sqrt(pk)) / sqrt(qk)")
print("= (tk / sqrt(pk)) * (1/sqrt(qk))")
print("= (tk / sqrt(pk)) * sqrt(1/qk)")
print("= sqrt(tk^2 / pk) * sqrt(1/qk)")
print("= sqrt(tk^2 / (pk * qk))")
print("= sqrt(tk / pk) * sqrt(tk / qk)")