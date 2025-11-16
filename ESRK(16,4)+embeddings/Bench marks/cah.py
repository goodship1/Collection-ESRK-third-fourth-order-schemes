#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import csv, time

# ============================================================
# Load ESRK(16,4/3) coefficients
# ============================================================
A  = np.loadtxt("A.csv", delimiter=",")
b4 = np.loadtxt("b_main.csv", delimiter=",")
b3 = np.loadtxt("b_embedded_dev04.csv", delimiter=",")
S  = len(b4)                # number of stages = 16

# ============================================================
# PDE: FitzHugh–Nagumo (1D Reaction–Diffusion)
# ============================================================
def lap(z, dx):
    out = np.zeros_like(z)
    out[1:-1] = (z[2:] - 2*z[1:-1] + z[:-2]) / dx**2
    out[0]    = (z[1] - 2*z[0] + z[-1]) / dx**2
    out[-1]   = (z[0] - 2*z[-1] + z[-2]) / dx**2
    return out

def fhn_rhs(t, y, Du, eps, gamma, a, dx):
    N = len(y)//2
    u, v = y[:N], y[N:]
    fu = Du * lap(u, dx) + u*(u-a)*(1-u) - v
    fv = eps * (u - gamma*v)
    return np.concatenate([fu, fv])

# ============================================================
# ESRK Integrators
# ============================================================
def esrk_step(f, y, t, h, A, b, args):
    k = np.zeros((S, len(y)))
    for i in range(S):
        yi = y + h * np.sum(A[i,:i,None] * k[:i], axis=0)
        k[i] = f(t + h*np.sum(A[i,:]), yi, *args)
    return y + h * np.sum(b[:,None] * k, axis=0)

def esrk_step_adaptive(f, y, t, h, A, b_hi, b_lo, args):
    k = np.zeros((S, len(y)))
    for i in range(S):
        yi = y + h * np.sum(A[i,:i,None] * k[:i], axis=0)
        k[i] = f(t + h*np.sum(A[i,:]), yi, *args)

    y_hi = y + h*np.sum(b_hi[:,None] * k, axis=0)
    y_lo = y + h*np.sum(b_lo[:,None] * k, axis=0)
    err  = norm(y_hi - y_lo) / np.sqrt(len(y))
    return y_hi, err

def esrk_integrate(f, y0, t0, t1, h, A, b, args):
    t, y = t0, y0.copy()
    while t < t1:
        if t + h > t1:
            h = t1 - t
        y = esrk_step(f, y, t, h, A, b, args)
        t += h
    return y

def adaptive_esrk(f, y0, t0, t1, h0, A, b_hi, b_lo, tol, args):
    t, y, h = t0, y0.copy(), h0
    safety, kP, kI = 0.9, 0.2, 0.08
    steps = rejects = 0
    prev_err = tol
    h_hist = []

    while t < t1:
        if t + h > t1:
            h = t1 - t

        y_new, err = esrk_step_adaptive(f, y, t, h, A, b_hi, b_lo, args)

        if err <= tol or np.isnan(err):
            y = y_new
            t += h
            h_hist.append(h)
        else:
            rejects += 1

        fac = (tol/(err+1e-14))**kP * (tol/(prev_err+1e-14))**kI
        h = float(np.clip(h * safety * fac, 0.2*h, 4.0*h))
        prev_err = err
        steps += 1

        if steps > 20000:
            break

    return y, np.array(h_hist), steps, rejects

def compute_reference(f, y0, t0, t1, A, b4, args):
    """Very fine fixed step for reference."""
    h_ref = 5e-5
    return esrk_integrate(f, y0, t0, t1, h_ref, A, b4, args)

# ============================================================
# Main Script
# ============================================================
print("\nFitzHugh–Nagumo PDE — ESRK(16,4/3) Full Diagnostics\n")

# Grid - less stiff
N = 200
L = 1.0
x = np.linspace(0, L, N)
dx = L / N

# PDE parameters
Du = 0.005
eps = 0.05
gamma = 2.0
a = 0.1

# Initial condition
u0 = 0.5*(1 + np.tanh((0.5-x)*10))
v0 = np.zeros_like(x)
y0 = np.concatenate([u0, v0])

t0, t1 = 0.0, 0.1

# ------------------------------------------------------------
# Reference
# ------------------------------------------------------------
print("Computing reference solution (fixed h=5e-5)…")
y_ref = compute_reference(
    fhn_rhs, y0, t0, t1, A, b4,
    (Du, eps, gamma, a, dx)
)
print(f"Reference computed. Max |u|={np.max(np.abs(y_ref[:N])):.3f}")

# ------------------------------------------------------------
# FOURTH-ORDER METHOD
# ------------------------------------------------------------
print("\n=== ESRK(16,4) Fourth-Order Method Convergence ===")
h_list = np.array([0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625])
errors4 = []

for h in h_list:
    y = esrk_integrate(fhn_rhs, y0, t0, t1, h, A, b4,
                       (Du, eps, gamma, a, dx))
    if np.any(np.isnan(y)):
        print(f"h={h:9.6f}  [UNSTABLE - skipped]")
        continue
    e = norm(y - y_ref) / np.sqrt(len(y))
    errors4.append(e)
    print(f"h={h:9.6f}  RMS={e:.3e}")

if len(errors4) > 1:
    orders4 = [np.log(errors4[i]/errors4[i+1]) /
               np.log(h_list[i]/h_list[i+1])
               for i in range(len(errors4)-1)]
    mean_order4 = np.mean(orders4)
else:
    mean_order4 = np.nan

print(f"Observed order (4th) ≈ {mean_order4:.3f}")

# ------------------------------------------------------------
# THIRD-ORDER METHOD
# ------------------------------------------------------------
print("\n=== ESRK(16,3) Third-Order Embedded Method Convergence ===")
errors3 = []
h_list_3 = []

for h in h_list:
    y = esrk_integrate(fhn_rhs, y0, t0, t1, h, A, b3,
                       (Du, eps, gamma, a, dx))
    if np.any(np.isnan(y)):
        print(f"h={h:9.6f}  [UNSTABLE - skipped]")
        continue
    e = norm(y - y_ref) / np.sqrt(len(y))
    errors3.append(e)
    h_list_3.append(h)
    print(f"h={h:9.6f}  RMS={e:.3e}")

if len(errors3) > 1:
    orders3 = [np.log(errors3[i]/errors3[i+1]) /
               np.log(h_list_3[i]/h_list_3[i+1])
               for i in range(len(errors3)-1)]
    mean_order3 = np.mean(orders3)
else:
    mean_order3 = np.nan

print(f"Observed order (3rd) ≈ {mean_order3:.3f}")

# ------------------------------------------------------------
# ADAPTIVE ESRK + FLOPs
# ------------------------------------------------------------
print("\n=== Adaptive ESRK(16,4/3) ===")

tols = np.logspace(-2, -5, 6)

# FLOP model A
F_RHS = 20 * N         # FLOPs per RHS evaluation
F_per_step = S * F_RHS # 16 stages × RHS FLOPs

for tol in tols:
    y, h_hist, steps, rej = adaptive_esrk(
        fhn_rhs, y0, t0, t1,
        h0=0.001,
        A=A, b_hi=b4, b_lo=b3,
        tol=tol,
        args=(Du, eps, gamma, a, dx)
    )

    err = norm(y - y_ref) / np.sqrt(len(y))
    mean_h = np.mean(h_hist) if len(h_hist) > 0 else np.nan

    # FLOP count
    FLOPs_total = steps * F_per_step

    print(f"tol={tol:7.1e}  steps={steps:5d}  rej={rej:3d}  "
          f"err={err:.3e}  mean(h)={mean_h:.3e}  FLOPs≈{FLOPs_total:.2e}")

# ------------------------------------------------------------
# DONE
# ------------------------------------------------------------
print("\nREADY.\n")
plt.show()
