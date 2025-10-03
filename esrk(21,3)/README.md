# ESRK(15,3) — Extended-Stability Runge–Kutta Scheme

**Order:** 3  
**Stages:** 21 
**Type:** Explicit RK (strictly lower-triangular A)  
**Family:** ESRK — Extended-Stability Runge–Kutta  
**Design goal:** Maximize the real-axis stability interval while retaining 3rd-order accuracy.

---

## 📂 Contents

This folder provides the coefficients and stability data for the **ESRK(16,4)** scheme in machine-readable formats:

- `A.csv` — Butcher tableau \(A\) (21×21, strictly lower-triangular)  
- `b.csv` — Weights vector \(b\) (length 21)  
- `coeffs.json` — Metadata, stability radius, and stability polynomial coefficients  

---

## 🔑 Scheme Summary

| Property | Value |
|----------|-------|
| Name | ESRK(15,4) |
| Family | Extended-Stability Runge–Kutta (ESRK) |
| Order | 3 |
| Stages | 21 |
| Type | Explicit RK (lower-triangular \(A\)) |
| Stability polynomial degree | 21 |
| Real-axis stability radius ) | **≈ 141.49,** |

---

## 📈 Stability Function

### Stability boundary in the complex plane (|R(z)| = 1)
![ESRK(21,3): Stability boundary](third_15(5).png)

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




