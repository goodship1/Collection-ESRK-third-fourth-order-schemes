#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brusselator with ESRK(16,4) + embedded(3), **hard-coded and safety-hardened**.

- Coefficients are baked in below (from your CSVs).
- Prints sanity (order-3 for both, one linear basis of order-4 for main).
- Computes real-axis stability radii R4 (main), R3 (embed).
- Adaptive integration with:
  * non-finite stage guard,
  * Jacobian-based linear stability cap: h <= 0.9 * min(R4, R3) / rho(J(y)),
  * calmer step growth (facmax=2).
- Fixed-step convergence (~4th order).

Outputs to ./esrk_embedded_out_safe_hc/ :
  stability_realaxis.png
  brusselator_adaptive_timeseries.png
  convergence_fixedstep.png
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------- Hard-coded coefficients ----------------------
A = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.29795063269635103, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.089284076424934397, 0.52202693303334102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.089284076424934397, 0.089284076424934397, 0.14434974635228001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.089284076424934397, 0.089284076424934397, 0.089284076424934397, -0.00037195629573239002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, -0.12411747366216, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.19280013115096101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, -0.0072120168886084897, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.38549687402306099, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.24819285595992099, 0, 0, 0, 0, 0, 0, 0],
    [0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, -0.29585495006398099, -4.2537189111117502e-05, 0, 0, 0, 0, 0, 0],
    [0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, -0.29585495006398099, 0.16301716951297901, 0.13837104421540999, 0, 0, 0, 0, 0],
    [0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, -0.29585495006398099, 0.16301716951297901, -0.081982432554952203, 0.40310809047621399, 0, 0, 0, 0],
    [0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, -0.29585495006398099, 0.16301716951297901, -0.081982432554952203, 0.546008221888163, 0.125164780662438, 0, 0, 0],
    [0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, -0.29585495006398099, 0.16301716951297901, -0.081982432554952203, 0.546008221888163, -0.042284432961144001, -0.0057986271050176402, 0, 0],
    [0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, -0.29585495006398099, 0.16301716951297901, -0.081982432554952203, 0.546008221888163, -0.042284432961144001, 0.46743119776808101, 0.50203613164768501, 0]
], dtype=float)
b_main = np.array([ 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, 0.089284076424934397, -0.29585495006398077, 0.16301716951297909, -0.081982432554952231, 0.54600822188816311, -0.042284432961144008, 0.46743119776808084, -0.45495020324594998, -0.015657181742671309 ], dtype=float)
b_emb  = np.array([ 0.089284076436289703, 0.089284076391799805, 0.089284076484897182, 0.089284076399890389, 0.089284076416699026, 0.089284076413932281, 0.089284076441101257, 0.089284076433956486, -0.29585495021342056, 0.16301716952727804, -0.081982432538810199, 0.54600822191166198, -0.042284432939687817, 0.46743119779250508, -0.45495020322171581, -0.01565718173637691 ], dtype=float)
S = b_main.size

# ---------------------- Options ----------------------
# Adaptive run
T_ADAPT   = 10.0
Y0        = np.array([1.2, 3.1], float)
ATOL      = 1e-9
RTOL      = 1e-6
H0        = 1e-3

# Convergence run
T_CONV    = 5.0
HS        = 2.0 ** (-np.arange(5, 10))   # 1/32 .. 1/512

outdir = Path("./esrk_embedded_out_safe_hc")
outdir.mkdir(parents=True, exist_ok=True)

# ---------------------- Helpers ----------------------
def c_nodes(A):
    return np.array([A[i, :i].sum() for i in range(A.shape[0])], float)

def Ac_vec(A, c):
    return np.array([np.dot(A[i, :i], c[:i]) for i in range(A.shape[0])], float)

def A_times(A, v):
    return np.array([np.dot(A[i, :i], v[:i]) for i in range(A.shape[0])], float)

def make_M3(A):
    c  = c_nodes(A)
    Ac = Ac_vec(A, c)
    M3 = np.vstack([np.ones_like(c), c, c**2, Ac])
    d3 = np.array([1.0, 0.5, 1/3, 1/6], float)
    return M3, d3

def make_M4_linear(A):
    # one convenient linear 4th-order basis for fixed A
    e  = np.ones(A.shape[0])
    c  = c_nodes(A); c2 = c*c; c3 = c2*c
    Ac = Ac_vec(A, c)
    Ac2 = A_times(A, c2)   # A(c^2)
    Acc = Ac*c             # (Ac) ∘ c
    A2c = A_times(A, Ac)   # A^2 c
    M4 = np.vstack([e, c, c2, Ac, c3, Ac2, Acc, A2c])
    d4 = np.array([1.0, 1/2, 1/3, 1/6, 1/4, 1/8, 1/12, 1/24], float)
    return M4, d4

def phi_from_Ab(A, b):
    A = np.asarray(A, float); b = np.asarray(b, float); s = b.size
    def phi(z):
        z = np.asarray(z, dtype=complex)
        Y = np.zeros(z.shape + (s,), dtype=complex)
        for i in range(s):
            acc = 0.0 if i == 0 else np.tensordot(A[i, :i], Y[..., :i], axes=(0, -1))
            Y[..., i] = 1.0 + z * acc
        return 1.0 + z * np.tensordot(b, Y, axes=(0, -1))
    return phi

def radius_real(phi, xmax=200.0, N=60000):
    xs = -np.linspace(0.0, xmax, N)
    vals = np.abs(phi(xs))
    inside = (vals <= 1.0)
    if not inside.any(): return 0.0
    k = np.where(inside)[0][-1]
    if k+1 >= len(xs): return float(-xs[k])
    a, c = xs[k+1], xs[k]
    for _ in range(50):
        m = 0.5*(a+c)
        if np.abs(phi(m)) <= 1.0: c = m
        else: a = m
    return float(-c)

# Jacobian of Brusselator and spectral radius
def jac_brusselator(y, Ap=1.0, Bp=3.0):
    x, yv = y
    return np.array([[2*x*yv - (Bp + 1.0), x*x],
                     [Bp - 2*x*yv,        -x*x]], float)

def spec_radius(J):
    w = np.linalg.eigvals(J)
    return float(np.max(np.abs(w)))

# ---------------------- Sanity & radii ----------------------
M3, d3 = make_M3(A); M4, d4 = make_M4_linear(A)
r3_main = np.linalg.norm(M3 @ b_main - d3, ord=np.inf)
r3_emb  = np.linalg.norm(M3 @ b_emb  - d3, ord=np.inf)
r4_main = np.linalg.norm(M4 @ b_main - d4, ord=np.inf)

phi4 = phi_from_Ab(A, b_main); R4 = radius_real(phi4)
phi3 = phi_from_Ab(A, b_emb);  R3 = radius_real(phi3)
print("=== ESRK(16) coefficients (hard-coded) ===")
print(f"S = {S}")
print(f"||M3 b_main - d3||_inf = {r3_main:.3e}")
print(f"||M3 b_emb  - d3||_inf = {r3_emb:.3e}")
print(f"(One linear basis) ||M4 b_main - d4||_inf = {r4_main:.3e}")
print(f"R4(main) ≈ {R4:.3f}, R3(embed) ≈ {R3:.3f}, ratio ≈ {(R3/R4 if R4>0 else float('nan')):.3f}")

# Stability plot
xs = np.linspace(0.0, max(R4, R3)*1.2 if max(R4, R3) > 0 else 120.0, 4000)
plt.figure(figsize=(7.0,4.3))
plt.plot(xs, np.abs(phi4(-xs)), label="main (order 4)")
plt.plot(xs, np.abs(phi3(-xs)), "--", label="embedded (order 3)")
plt.axhline(1.0, lw=1, alpha=0.6)
plt.axvline(R4, ls=":", label=f"R4≈{R4:.2f}")
plt.axvline(R3, ls=":", label=f"R3≈{R3:.2f}")
plt.xlabel("x (z=-x)"); plt.ylabel("|φ(-x)|")
plt.title("Real-axis stability")
plt.grid(True, ls="--", alpha=0.5); plt.legend()
plt.tight_layout(); plt.savefig(outdir/"stability_realaxis.png", dpi=160); plt.close()

# ---------------------- Embedded step ----------------------
def rk_embedded_step(f, t, y, h, A, b_main, b_emb, c):
    S = len(b_main); n = len(y)
    K = np.zeros((S, n), dtype=float)
    for i in range(S):
        yi = y.copy()
        if i > 0:
            yi += h * (A[i, :i] @ K[:i])
        ti = t + c[i] * h
        fi = f(ti, yi)
        if not np.all(np.isfinite(fi)):
            return None, None, None
        K[i] = fi
    y_main = y + h * (b_main @ K)
    y_emb  = y + h * (b_emb  @ K)
    return y_main, y_emb, K

# ---------------------- Stability-aware adaptive integrator ----------------------
def integrate_adaptive(f, t0, y0, t1, A, b_main, b_emb,
                       atol=1e-9, rtol=1e-6, h0=1e-3, h_min=1e-14,
                       safety=0.9, order_main=4, facmin=0.2, facmax=2.0,
                       R_lin_cap=None, Ap=1.0, Bp=3.0, max_steps=1_000_000):
    c = c_nodes(A)
    t = float(t0); y = np.array(y0, float)
    h = float(h0)
    ts = [t]; ys = [y.copy()]
    steps = 0
    eps = 1e-14
    while t < t1 - 1e-16 and steps < max_steps:
        steps += 1
        # Jacobian-based cap
        J = jac_brusselator(y, Ap, Bp)
        rho = spec_radius(J)
        if R_lin_cap is not None and rho > eps:
            h_cap = R_lin_cap / rho
            if h > h_cap: h = max(h_min, 0.9*h_cap)
        if t + h > t1: h = t1 - t

        y1, y1e, K = rk_embedded_step(f, t, y, h, A, b_main, b_emb, c)
        if y1 is None:
            h = max(h_min, h * facmin)
            if h <= h_min:
                raise RuntimeError("Stage evaluation non-finite; h reached h_min.")
            continue

        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y1))
        err = np.max(np.abs(y1 - y1e) / scale)
        fac = facmax if err == 0 else safety * (1.0 / err)**(1.0/(order_main+1))
        fac = min(facmax, max(facmin, fac))

        if np.isfinite(err) and err <= 1.0 and np.all(np.isfinite(y1)):
            t += h; y = y1
            ts.append(t); ys.append(y.copy())
            h *= fac
        else:
            h = max(h_min, h * facmin)
            if h <= h_min:
                raise RuntimeError("Step size underflow or non-finite state.")
    return np.array(ts), np.vstack(ys)

# Brusselator RHS
def brusselator(Ap=1.0, Bp=3.0):
    def f(t, y):
        x, yv = y
        dx = Ap + x*x*yv - (Bp + 1.0) * x
        dy = Bp * x - x*x*yv
        return np.array([dx, dy], float)
    return f

# ---------------------- Run adaptive demo ----------------------
R_lin_cap = 0.9 * min(R4, R3)
f = brusselator(1.0, 3.0)
ts, ys = integrate_adaptive(f, 0.0, Y0, T_ADAPT, A, b_main, b_emb,
                            atol=ATOL, rtol=RTOL, h0=H0,
                            R_lin_cap=R_lin_cap, Ap=1.0, Bp=3.0)

plt.figure(figsize=(7.0,4.2))
plt.plot(ts, ys[:,0], label="x(t)")
plt.plot(ts, ys[:,1], label="y(t)")
plt.xlabel("t"); plt.ylabel("state"); plt.legend()
plt.title("Brusselator — ESRK(4) adaptive (stability-aware, hard-coded)")
plt.tight_layout(); plt.savefig(outdir/"brusselator_adaptive_timeseries.png", dpi=160); plt.close()
print(f"Adaptive steps taken: {len(ts)-1}. Final y(T) = {ys[-1]}")

# ---------------------- Fixed-step 4th-order convergence ----------------------
def rk_fixed(f, t0, y0, t1, h, A, b_main):
    c = c_nodes(A)
    t = float(t0); y = np.array(y0, float)
    while t < t1 - 1e-16:
        if t + h > t1: h = t1 - t
        S = len(b_main); n = len(y)
        K = np.zeros((S, n), dtype=float)
        for i in range(S):
            yi = y.copy()
            if i > 0: yi += h * (A[i, :i] @ K[:i])
            ti = t + c[i] * h
            K[i] = f(ti, yi)
        y = y + h * (b_main @ K)
        t += h
    return y

# Reference at T_CONV
h_ref = HS[-1] / 4.0
nref = int(math.ceil(T_CONV / h_ref))
y = Y0.copy(); t = 0.0
for _ in range(nref):
    y = rk_fixed(f, t, y, t + T_CONV / nref, T_CONV / nref, A, b_main)
    t += T_CONV / nref
y_ref = y.copy()

errs = []
for h in HS:
    yT = rk_fixed(f, 0.0, Y0.copy(), T_CONV, h, A, b_main)
    errs.append(np.linalg.norm(yT - y_ref, ord=np.inf))
errs = np.array(errs)

plt.figure(figsize=(6.6,4.3))
plt.loglog(HS, errs, marker="o", label="error at T")
plt.loglog(HS, HS**4 * (errs[0]/(HS[0]**4)), "--", label="O(h^4)")
plt.gca().invert_xaxis()
plt.xlabel("h"); plt.ylabel("∞-norm error")
plt.title("Fixed-step convergence (~4th order)")
plt.grid(True, ls="--", alpha=0.5); plt.legend()
plt.tight_layout(); plt.savefig(outdir/"convergence_fixedstep.png", dpi=160); plt.close()
orders = np.log(errs[:-1]/errs[1:]) / np.log(HS[:-1]/HS[1:])
print("Observed local orders:", orders, "  mean≈", orders.mean())


print("Saved:")
print(" •", outdir/"stability_realaxis.png")
print(" •", outdir/"brusselator_adaptive_timeseries.png")
print(" •", outdir/"convergence_fixedstep.png")

