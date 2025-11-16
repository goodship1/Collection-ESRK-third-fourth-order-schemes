#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESRK(16,4/3) — 1D Viscous Burgers' Equation (Flux Form)
-------------------------------------------------------
Stabilized conservative-form implementation with periodic BCs.
Diagnostics:
  - Fixed-step convergence (4th vs 3rd)
  - Adaptive tolerance sweep
  - FLOPs tracking + CSV and plots
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import csv, time

# ============================================================
# Load ESRK coefficients
# ============================================================
A  = np.loadtxt("A.csv", delimiter=",")
b4 = np.loadtxt("b_main.csv", delimiter=",")
b3 = np.loadtxt("b_embedded_dev04.csv", delimiter=",")
S  = len(b4)
FLOPS_PER_RHS = 10
flops = {"nfe": 0}

# ============================================================
# 1️⃣ Burgers' PDE RHS (1D periodic, flux form)
# ============================================================
def burgers_rhs(t, y, nu=1e-3, L=1.0, N=128):
    """Semi-discrete viscous Burgers' equation in conservative flux form."""
    flops["nfe"] += 1
    h = L / N
    up = np.roll(y, -1)
    um = np.roll(y, 1)

    # Conservative form: du/dt = -1/2 * d(u^2)/dx + nu * u_xx
    flux = 0.5 * (up**2 - um**2) / (2*h)
    uxx = (up - 2*y + um) / h**2
    du = -flux + nu * uxx
    return du

# ============================================================
# 2️⃣ ESRK Integrators
# ============================================================
def esrk_step_adaptive(f, y, t, h, A, b_hi, b_lo, *args):
    k = np.zeros((S, y.size))
    for i in range(S):
        yi = y + h * np.sum(A[i,:i,None]*k[:i], axis=0)
        k[i] = f(t, yi, *args)
    y_hi = y + h * np.sum(b_hi[:,None]*k, axis=0)
    y_lo = y + h * np.sum(b_lo[:,None]*k, axis=0)
    return y_hi, norm(y_hi - y_lo)

def adaptive_integrate(f, y0, t0, t1, h0, A, b_hi, b_lo, tol, args=()):
    h, y, t = h0, y0.copy(), t0
    steps = rejects = 0
    prev_err, h_hist = tol, []
    while t < t1 and steps < 100000:
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
        fac = (tol / (err + 1e-14))**0.2 * (tol / (prev_err + 1e-14))**0.08
        h = np.clip(0.9 * h * fac, 0.1 * h, 2.0 * h)
        prev_err = err
        steps += 1
    return y, np.array(h_hist), steps, rejects

def esrk_integrate(f, y0, T, h, A, b, args=()):
    y = y0.copy()
    Nsteps = int(np.ceil(T / h))
    for _ in range(Nsteps):
        k = np.zeros((S, y.size))
        for i in range(S):
            yi = y + h * np.sum(A[i,:i,None]*k[:i], axis=0)
            k[i] = f(0.0, yi, *args)
        y += h * np.sum(b[:,None]*k, axis=0)
    return y

# ============================================================
# 3️⃣ Main driver
# ============================================================
if __name__ == "__main__":
    start = time.time()
    print("Running ESRK(16,4/3) — 1D Viscous Burgers’ Equation (Flux Form)...")

    # Parameters
    L, N = 1.0, 128
    x = np.linspace(0, L, N, endpoint=False)
    t0, t1 = 0.0, 0.25
    nus = [5e-3, 1e-3, 5e-4]  # mild-to-moderate stiffness levels

    for nu in nus:
        args = (nu, L, N)
        y0 = np.sin(2*np.pi*x)  # smooth periodic initial condition

        print(f"\nν = {nu:.1e}")

        # Reference integration (tiny dt)
        ref = esrk_integrate(burgers_rhs, y0, t1, 1e-4, A, b4, args=args)

        # ------------------------------------------------------------
        # Fixed-step convergence
        # ------------------------------------------------------------
        h_list = np.array([0.01, 0.005, 0.0025, 0.00125])
        errs4, errs3 = [], []
        for h in h_list:
            y4 = esrk_integrate(burgers_rhs, y0, t1, h, A, b4, args=args)
            y3 = esrk_integrate(burgers_rhs, y0, t1, h, A, b3, args=args)
            e4 = norm(y4 - ref) / norm(ref)
            e3 = norm(y3 - ref) / norm(ref)
            errs4.append(e4)
            errs3.append(e3)
            print(f"h={h:7.5f}, RMS(4th)={e4:10.3e}, RMS(3rd)={e3:10.3e}")

        errs4, errs3 = np.array(errs4), np.array(errs3)
        p4 = np.mean(np.log(errs4[:-1]/errs4[1:]) / np.log(h_list[:-1]/h_list[1:]))
        p3 = np.mean(np.log(errs3[:-1]/errs3[1:]) / np.log(h_list[:-1]/h_list[1:]))
        print(f"Observed order ≈ p4={p4:.3f}, p3={p3:.3f}")

        plt.figure()
        plt.loglog(h_list, errs4, "o-", label=f"ESRK(16,4) ν={nu}")
        plt.loglog(h_list, errs3, "s--", label=f"ESRK(16,3)")
        plt.xlabel("Δt"); plt.ylabel("RMS Error")
        plt.grid(True, which="both", ls=":")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"esrk_burgers_convergence_nu{nu:.0e}.png", dpi=160)

        # ------------------------------------------------------------
        # Adaptive tolerance sweep
        # ------------------------------------------------------------
        tols = np.logspace(-2, -5, 7)
        errs_ad, steps_ad, rej_ad, flops_ad = [], [], [], []
        print(f"\n=== Adaptive ESRK(16,4/3) Burgers (ν={nu:.1e}) ===")
        for tol in tols:
            flops["nfe"] = 0
            y, h_hist, steps, rejects = adaptive_integrate(
                burgers_rhs, y0, t0, t1, 0.01, A, b4, b3, tol, args=args)
            err = norm(y - ref) / norm(ref)
            total_flops = flops["nfe"] * FLOPS_PER_RHS
            errs_ad.append(err)
            steps_ad.append(steps)
            rej_ad.append(rejects)
            flops_ad.append(total_flops)
            print(f"tol={tol:8.1e}  steps={steps:5d}  rej={rejects:3d}  "
                  f"err={err:10.3e}  FLOPs≈{total_flops:7.2e}")

        # Save plots + CSV
        plt.figure()
        plt.loglog(tols, errs_ad, "o-", label=f"ν={nu}")
        plt.xlabel("Tolerance"); plt.ylabel("RMS Error")
        plt.grid(True, which="both", ls=":"); plt.legend()
        plt.savefig(f"esrk_burgers_error_vs_tol_nu{nu:.0e}.png", dpi=160)

        plt.figure()
        plt.loglog(flops_ad, errs_ad, "o-", label=f"ν={nu}")
        plt.xlabel("Estimated FLOPs"); plt.ylabel("RMS Error")
        plt.grid(True, which="both", ls=":"); plt.legend()
        plt.savefig(f"esrk_burgers_error_vs_flops_nu{nu:.0e}.png", dpi=160)

        with open(f"esrk_burgers_adaptive_nu{nu:.0e}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tol", "steps", "rejects", "global_err", "flops_est"])
            for i in range(len(tols)):
                w.writerow([f"{tols[i]:.1e}", steps_ad[i], rej_ad[i],
                            f"{errs_ad[i]:.4e}", f"{flops_ad[i]:.4e}"])

        print(f"ν={nu:.1e} complete. Runtime so far: {time.time()-start:.1f}s")

    print("\nAll ESRK Burgers’ Equation tests complete ✅")
