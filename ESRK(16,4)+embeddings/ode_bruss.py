#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESRK(16,4/3) – Full Validation + Adaptive Benchmark
----------------------------------------------------
Performs:
  • Convergence and order check on the Brusselator
  • Real-axis stability radii computation
  • Adaptive integration with local-error control (PI controller)
  • Tolerance sweep and performance metrics
Outputs:
  convergence_adaptive.png
  adaptive_scaling.png
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.linalg import solve, norm
from scipy.integrate import solve_ivp

# ============================================================
#  Load ESRK(16,4/3) coefficients
# ============================================================
A  = np.loadtxt("A.csv", delimiter=",")
b4 = np.loadtxt("b_main.csv", delimiter=",")
b3 = np.loadtxt("b_embedded_dev04.csv", delimiter=",")
S = len(b4)
I = np.eye(S)
e = np.ones(S)

# ============================================================
#  Resolvent and stability utilities
# ============================================================
def R_abs(b, z):
    """Evaluate |R(z)| for a given RK method."""
    try:
        y = solve(I - z * A, e)
        return abs(1 + z * (b @ y))
    except np.linalg.LinAlgError:
        return np.inf

def radius_real_axis(b, x_max=200, tol=1e-3):
    """Find real-axis stability radius."""
    lo, hi = 0.0, 1.0
    while R_abs(b, -hi) <= 1.0 and hi < x_max:
        hi *= 2
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if R_abs(b, -mid) <= 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return lo

# ============================================================
#  Brusselator ODE
# ============================================================
def brusselator(_, y, a=1.0, b=3.0, scale=0.1):
    x, yv = y
    dx = a - (b + 1) * x + yv * x**2
    dy = b * x - yv * x**2
    return scale * np.array([dx, dy])

# ============================================================
#  ESRK Integrator (fixed & adaptive)
# ============================================================
def esrk_step_adaptive(f, y, t, h, A, b4, b3):
    """Single ESRK step returning y4, local error estimate."""
    k = np.zeros((S, len(y)))
    for i in range(S):
        yi = y + h * np.sum(A[i, :i, None] * k[:i], axis=0)
        k[i] = f(t + h * np.sum(A[i, :]), yi)
    y4 = y + h * np.sum(b4[:, None] * k, axis=0)
    y3 = y + h * np.sum(b3[:, None] * k, axis=0)
    err = norm(y4 - y3)
    return y4, err

def adaptive_integrate(f, y0, t0, t1, h0, A, b4, b3, tol=1e-5, safety=0.9):
    """Adaptive ESRK integrator with PI-style step-size control."""
    h = h0
    y = y0.copy()
    t = t0
    steps, rejects, h_hist = 0, 0, []
    prev_err = tol
    while t < t1:
        if t + h > t1:
            h = t1 - t
        y_new, err = esrk_step_adaptive(f, y, t, h, A, b4, b3)
        if err < tol or np.isnan(err):
            y, t = y_new, t + h
            h_hist.append(h)
        else:
            rejects += 1
        # PI controller for smooth h adaptation
        fac = (tol / (err + 1e-14))**(0.2) * (tol / (prev_err + 1e-14))**(0.08)
        h = h * max(0.2, min(5.0, safety * fac))
        prev_err = err
        steps += 1
        if steps > 20000:
            break
    return y, np.array(h_hist), steps, rejects

def esrk_integrate(f, y0, T, h, A, b):
    """Fixed-step ESRK integration."""
    y = y0.copy()
    for _ in range(int(np.ceil(T / h))):
        k = np.zeros((S, len(y)))
        for i in range(S):
            yi = y + h * np.sum(A[i, :i, None] * k[:i], axis=0)
            k[i] = f(0, yi)
        y = y + h * np.sum(b[:, None] * k, axis=0)
    return y

# ============================================================
#  Convergence test
# ============================================================
def convergence(A, b, h_list):
    y0 = np.array([1.2, 3.1])
    T = 20.0
    # Use DOP853 as high-accuracy reference
    sol_ref = solve_ivp(brusselator, [0, T], y0, rtol=1e-10, atol=1e-12, method="DOP853")
    y_ref = sol_ref.y[:, -1]
    errs = []
    for h in h_list:
        y = esrk_integrate(brusselator, y0, T, h, A, b)
        e = norm(y - y_ref)
        errs.append(e)
        print(f"h={h:7.5f}, L2={e:10.3e}")
    p = np.mean([np.log(errs[i] / errs[i + 1]) / np.log(h_list[i] / h_list[i + 1])
                 for i in range(len(h_list) - 1)])
    return np.array(errs), p

# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    start = time.time()

    # --- Convergence check
    h_list = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    print("\n=== ESRK(16,4) Convergence ===")
    e4, p4 = convergence(A, b4, h_list)
    print(f"Observed order ≈ {p4:.3f}")

    print("\n=== ESRK(16,3) Convergence ===")
    e3, p3 = convergence(A, b3, h_list)
    print(f"Observed order ≈ {p3:.3f}")

    plt.figure(figsize=(6, 4))
    plt.loglog(h_list, e4, "o-", label=f"ESRK(16,4) p≈{p4:.2f}")
    plt.loglog(h_list, e3, "s-", label=f"ESRK(16,3) p≈{p3:.2f}")
    plt.xlabel("h")
    plt.ylabel("L2 error")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence_adaptive.png", dpi=160)

    # --- Stability radii
    print("\n=== Real-axis stability radii ===")
    R4 = radius_real_axis(b4, x_max=200)
    R3 = radius_real_axis(b3, x_max=200)
    print(f"R4 ≈ {R4:.3f}")
    print(f"R3 ≈ {R3:.3f}")
    print(f"R3/R4 ≈ {R3 / R4:.3f}")

    # --- Adaptive sweep
    print("\n=== Adaptive ESRK(16,4/3) test ===")
    y0 = np.array([1.2, 3.1])
    t0_, t1 = 0.0, 20.0
    tols = [1e-2, 1e-3, 1e-4, 1e-5]
    global_errors, steps_list, rejects_list, mean_h, std_h = [], [], [], [], []
    y_ref = esrk_integrate(brusselator, y0, t1, 1e-4, A, b4)
    for tol in tols:
        y, h_hist, steps, rejects = adaptive_integrate(brusselator, y0, t0_, t1, 0.1, A, b4, b3, tol=tol)
        err = norm(y - y_ref)
        global_errors.append(err)
        steps_list.append(steps)
        rejects_list.append(rejects)
        mean_h.append(np.mean(h_hist) if len(h_hist) > 0 else 0)
        std_h.append(np.std(h_hist) if len(h_hist) > 0 else 0)
        print(f"tol={tol:.0e}  steps={steps:4d}  rejects={rejects:3d}  mean(h)={mean_h[-1]:.3f}  err={err:.3e}")

    # Plot global error vs tolerance
    plt.figure(figsize=(6, 4))
    plt.loglog(tols, global_errors, "o-", label="ESRK(16,4/3)")
    plt.xlabel("tolerance")
    plt.ylabel("global L2 error")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("adaptive_scaling.png", dpi=160)
    print("Saved: adaptive_scaling.png")

    # Summary table
    print("\n=== Adaptive performance summary ===")
    print(" tol       steps  rejects  mean(h)   std(h)   global_err")
    for i, tol in enumerate(tols):
        print(f" {tol:7.0e}  {steps_list[i]:6d}  {rejects_list[i]:7d}  "
              f"{mean_h[i]:8.3f}  {std_h[i]:8.3f}  {global_errors[i]:10.3e}")

    print(f"\nRuntime: {time.time() - start:.1f}s")
    print("Validation complete.")
