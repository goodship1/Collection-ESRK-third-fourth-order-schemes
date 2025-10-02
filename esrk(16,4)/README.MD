# ESRK(16,4) — Extended-Stability Runge–Kutta Scheme

**Order:** 4  **Stages:** 16  **Type:** Explicit RK (strictly lower-triangular \(A\))  
**Family:** ESRK — Extended-Stability Runge–Kutta  
**Design goal:** Maximize the real-axis stability interval while retaining 4th-order accuracy.

This folder provides the coefficients and stability data for the **ESRK(16,4)** scheme in machine-readable formats (CSV and JSON) together with usage examples.

---


ESRK16-4/
├── A.csv # Butcher tableau (16×16), strictly lower-triangular
├── b.csv # weights vector (length 16)
└── coeffs.json # metadata + stability radius + polynomial for R(z)
