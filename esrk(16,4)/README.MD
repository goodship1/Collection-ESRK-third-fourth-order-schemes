# ESRK(16,4) — Extended-Stability Runge–Kutta Scheme

**Order:** 4  
**Stages:** 16  
**Type:** Explicit RK (strictly lower-triangular A)  
**Family:** ESRK — Extended-Stability Runge–Kutta  
**Design goal:** Maximize the real-axis stability interval while retaining 4th-order accuracy.

---

## 📂 Contents

This folder provides the coefficients and stability data for the **ESRK(16,4)** scheme in machine-readable formats:

- `A.csv` — Butcher tableau \(A\) (16×16, strictly lower-triangular)  
- `b.csv` — Weights vector \(b\) (length 16)  
- `coeffs.json` — Metadata, stability radius, and stability polynomial coefficients  

---

## 🔑 Scheme Summary

| Property | Value |
|----------|-------|
| Name | ESRK(16,4) |
| Family | Extended-Stability Runge–Kutta (ESRK) |
| Order | 4 |
| Stages | 16 |
| Type | Explicit RK (lower-triangular \(A\)) |
| Stability polynomial degree | 16 |
| Real-axis stability radius \(R_{\mathbb{R}}\) | **≈ 13.9170474646** |

---

## 📈 Stability Function

The stability function is defined as:

\[
R(z) = 1 + z\, b^\top (I - zA)^{-1}\mathbf{1},
\]

where \(\mathbf{1}\) is the all-ones vector.  
Because \(A\) is strictly lower-triangular, the solve is a simple forward substitution.

In `coeffs.json`, we also provide the **degree-16 polynomial form**:

\[
R(z) = \sum_{k=0}^{16} c_k z^k,
\]

with coefficients given in both ascending (`[c0, c1, ..., c16]`) and descending order.

---

## ⚙️ Usage Examples

### Python

```python
import numpy as np, json

# Load coefficients
A = np.loadtxt("A.csv", delimiter=",")
b = np.loadtxt("b.csv", delimiter=",")

# Verify explicitness
assert np.allclose(A, np.tril(A, -1))

# Evaluate stability function from tableau
def R_from_Ab(z, A, b):
    s = len(b)
    y = np.zeros(s, dtype=complex)
    for i in range(s):
        y[i] = 1 + z * np.dot(A[i,:i], y[:i])
    return 1 + z * np.dot(b, y)

# Evaluate stability function from JSON polynomial
with open("coeffs.json") as f:
    coeffs = json.load(f)["stability"]["stability_poly"]["coeffs_ascending"]

def R_from_poly(z, coeffs):
    val = 0.0 + 0.0j
    for c in reversed(coeffs):
        val = val*z + c
    return val
