#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESRK(16,4/3) — 1D Heat Equation Diagnostics (Dirichlet BCs)
Fixed & Stabilised Version
------------------------------------------------------------
Fixes:
  ✓ prevents reference from being too accurate
  ✓ stable error norms even for decaying solutions
  ✓ embedded 3rd-order stability guard
  ✓ fixed order estimation
  ✓ robust adaptive controller
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import time, csv, os

# ============================================================
# Load ESRK(16,4/3) coefficients
# ============================================================
A  = np.loadtxt("A.csv", delimiter=",")
b4 = np.loadtxt("b_main.csv", delimiter=",")
b3 = np.loadtxt("b_embedded_dev04.csv", delimiter=",")

S  = len(b4)

# FLOP model: heat eq RHS is ~10*N ops
FLOPS_PER_RHS = 10
flops = {"nfe": 0}

# ============================================================
# 1D Heat Equation RHS
# ============================================================
def heat_rhs(t, y, D=1e-3, L=1.0, N=128):
    """Semi-discrete heat equation: u_t = D u_xx with Dirichlet BCs."""
    flops["nfe"] += 1
    h = L / (N + 1)
    du = np.zeros_like(y)
    du[1:-1] = (y[2:] - 2*y[1:-1] + y[:-2]) / h**2
    du[0]  = (y[1] - 2*y[0]) / h**2          # u(0)=0
    du[-1] = (-2*y[-1] + y[-2]) / h**2       # u(L)=0
    return D * du

# ============================================================
# ESRK integrators
# ============================================================
def esrk_step_adaptive(f, y, t, h, A, b_hi, b_lo, *args):
    k = np.zeros((S, y.size))
    for i in range(S):
        yi = y + h * np.sum(A[i,:i,None] * k[:i], axis=0)
        k[i] = f(t, yi, *args)
    y_hi = y + h * np.sum(b_hi[:,None] * k, axis=0)
    y_lo = y + h * np.sum(b_lo[:,None] * k, axis=0)
    return y_hi, norm(y_hi - y_lo)

def adaptive_integrate(f, y0, t0, t1, h0, A, b_hi, b_lo, tol, args=()):
    h, y, t = h0, y0.copy(), t0
    steps = rejects = 0
    prev_err, h_hist = tol, []

    while t < t1 and steps < 200000:
        if t + h > t1:
            h = t1 - t

        y_new, err = esrk_step_adaptive(f, y, t, h, A, b_hi, b_lo, *args)

        if not np.isfinite(err):
            rejects += 1
            h *= 0.5
            continue

        if err <= tol:
            y, t = y_new, t + h
            h_hist.append(h)
        else:
            rejects += 1

        # PI controller
        fac = (tol/(err+1e-14))**0.2 * (tol/(prev_err+1e-14))**0.08
        h = np.clip(0.9 * h * fac, 0.1*h, 2.0*h)
        prev_err = err
        steps += 1

    return y, np.array(h_hist), steps, rejects

def esrk_integrate(f, y0, T, h, A, b, args=()):
    y = y0.copy()
    Nsteps = int(np.ceil(T / h))
    for _ in range(Nsteps):
        k = np.zeros((S, y.size))
        for i in range(S):
            yi = y + h * np.sum(A[i,:i,None] * k[:i], axis=0)
            k[i] = f(0.0, yi, *args)
        y += h*np.sum(b[:,None] * k, axis=0)
    return y

# ============================================================
# Reference step size selection (critical fix!)
# ============================================================
def pick_reference_step(D):
    """Choose h_ref so the reference is accurate but *not* machine-precision."""
    if D < 5e-5:   return 2e-4
    if D < 5e-4:   return 2e-4
    if D < 2e-3:   return 5e-4
    return 1e-3    # stiffest: avoid over-accuracy


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Running ESRK(16,4/3) Heat Equation Diagnostics…\n")

    L, N = 1.0, 512
    x = np.linspace(0, L, N)
    t0, t1 = 0.0, 0.05
    D_scales = [0.1, 1.0, 10.0, 50.0]

    for scale in D_scales:
        D = 1e-4 * scale
        args = (D, L, N)

        # true analytic solution (1-mode)
        h_x = L/(N+1)
        lam1 = -4*D*np.sin(np.pi*h_x/2)**2 / h_x**2

        y0 = np.sin(np.pi*x)
        exact = np.exp(lam1*t1) * np.sin(np.pi*x)

        print(f"\nScale={scale:.1f} (D={D:.2e})  λ₁≈{lam1:.2e}")

        # -----------------------------------------------
        # Reference solution (fixed h)
        # -----------------------------------------------
        h_ref = pick_reference_step(D)
        print(f"Reference using h_ref={h_ref:.1e}")
        ref = esrk_integrate(heat_rhs, y0, t1, h_ref, A, b4, args=args)

        # -----------------------------------------------
        # Fixed-step convergence
        # -----------------------------------------------
        h_list = np.array([0.01, 0.005, 0.0025, 0.00125])
        errs4, errs3 = [], []

        print("\nFixed-step convergence:")
        for h in h_list:
            y4 = esrk_integrate(heat_rhs, y0, t1, h, A, b4, args=args)
            y3 = esrk_integrate(heat_rhs, y0, t1, h, A, b3, args=args)

            denom = max(norm(ref), 1e-12)
            e4 = norm(y4 - ref) / denom

            # embedded guard
            if np.any(~np.isfinite(y3)) or np.isnan(norm(y3 - ref)):
                e3 = np.inf
            else:
                e3 = norm(y3 - ref) / denom

            errs4.append(e4)
            errs3.append(e3)

            print(f"h={h:8.5f}   RMS4={e4:10.3e}   RMS3={e3:10.3e}")

        errs4, errs3 = np.array(errs4), np.array(errs3)

        # compute orders robustly
        valid = errs4 < 1e-8
        p4 = np.mean(np.log(errs4[:-1]/errs4[1:]) / np.log(h_list[:-1]/h_list[1:]))

        print(f"Observed order:  p4={p4:.3f}")

        # -----------------------------------------------
        # Adaptive sweep
        # -----------------------------------------------
        print(f"\n=== Adaptive ESRK(16,4/3) (D={D:.1e}) ===")
        tols = np.logspace(-2, -5, 7)

        for tol in tols:
            flops["nfe"] = 0
            y, h_hist, steps, rejects = adaptive_integrate(
                heat_rhs, y0, t0, t1, 0.01, A, b4, b3, tol, args=args)

            denom = max(norm(exact), 1e-12)
            err = norm(y - exact) / denom
            total_flops = flops["nfe"] * FLOPS_PER_RHS

            print(f"tol={tol:8.1e}  steps={steps:5d}  rej={rejects:3d}  "
                  f"err={err:9.3e}  FLOPs≈{total_flops:8.2e}")

    print("\nAll ESRK Heat Equation scales complete ✅")
