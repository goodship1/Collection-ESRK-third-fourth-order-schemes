#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESRK(16,4/3) — Schnakenberg PDE Stiffness Sweep
-----------------------------------------------
Like-for-like comparison with ROCK4 (Julia version).
Diagnostics:
  - Fixed-step convergence (4th vs 3rd)
  - Adaptive tolerance sweep
  - Multi-scale stiffness analysis (0.1–100×)
  - FLOPs tracking + CSV/plots per scale
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, solve
import csv, time, os

# ============================================================
# Load ESRK(16,4/3) coefficients
# ============================================================
A  = np.loadtxt("A.csv", delimiter=",")
b4 = np.loadtxt("b_main.csv", delimiter=",")
b3 = np.loadtxt("b_embedded_dev04.csv", delimiter=",")
S  = len(b4)
I  = np.eye(S)
e1 = np.ones(S)
FLOPS_PER_RHS = 10
flops = {"nfe": 0}

# ============================================================
# Schnakenberg PDE RHS (1D, periodic BCs)
# ============================================================
def schnak_rhs(t, y, a=0.2, b=1.3, Du=1e-4, Dv=5e-5, L=1.0, N=64):
    """Semi-discrete Schnakenberg reaction–diffusion system."""
    flops["nfe"] += 1
    h = L / N
    u, v = y[:N], y[N:]

    def lap(z):
        zxx = np.zeros_like(z)
        zxx[1:-1] = (z[2:] - 2*z[1:-1] + z[:-2]) / h**2
        zxx[0] = (z[1] - 2*z[0] + z[-1]) / h**2
        zxx[-1] = (z[0] - 2*z[-1] + z[-2]) / h**2
        return zxx

    du = Du * lap(u) + a - u + u*u*v
    dv = Dv * lap(v) + b - u*u*v
    return np.concatenate([du, dv])

# ============================================================
# ESRK Integrators
# ============================================================
def esrk_step_adaptive(f, y, t, h, A, b_hi, b_lo, *args):
    k = np.zeros((S, y.size))
    for i in range(S):
        yi = y + h * np.sum(A[i, :i, None] * k[:i], axis=0)
        k[i] = f(t, yi, *args)
    y_hi = y + h * np.sum(b_hi[:, None] * k, axis=0)
    y_lo = y + h * np.sum(b_lo[:, None] * k, axis=0)
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
        fac = (tol / (err + 1e-14))**0.2 * (tol / (prev_err + 1e-14))**0.08
        h = np.clip(0.9 * h * fac, 0.1 * h, 2 * h)
        prev_err = err
        steps += 1
    return y, np.array(h_hist), steps, rejects


def esrk_integrate(f, y0, T, h, A, b, args=()):
    y = y0.copy()
    Nsteps = int(np.ceil(T / h))
    for _ in range(Nsteps):
        k = np.zeros((S, y.size))
        for i in range(S):
            yi = y + h * np.sum(A[i, :i, None] * k[:i], axis=0)
            k[i] = f(0.0, yi, *args)
        y += h * np.sum(b[:, None] * k, axis=0)
    return y

# ============================================================
# Main PDE diagnostics
# ============================================================
if __name__ == "__main__":
    start = time.time()

    # Base parameters
    L, N = 1.0, 64
    a, b = 0.2, 1.3
    Du_base, Dv_base = 1e-4, 5e-5
    x = np.linspace(0, L, N)
    t0, t1 = 0.0, 5.0
    scales = [0.1, 1.0, 10.0, 100.0]

    print(f"Running ESRK(16,4/3) Schnakenberg PDE across scales: {scales}")

    for scale in scales:
        print("\n" + "=" * 35)
        print(f"Scale={scale:.1f}  (Du={Du_base*scale:.2e}, Dv={Dv_base*scale:.2e})")
        print("=" * 35)

        Du, Dv = Du_base * scale, Dv_base * scale
        u0 = a + b + 0.05 * np.sin(2 * np.pi * x / L)
        v0 = b / (a + b)**2 + 0.05 * np.cos(2 * np.pi * x / L)
        y0 = np.concatenate([u0, v0])
        args = (a, b, Du, Dv, L, N)

        # Reference integration
        ref = esrk_integrate(schnak_rhs, y0, t1, 1e-3, A, b4, args=args)

        # ------------------------------------------------------------
        # Fixed-step convergence
        # ------------------------------------------------------------
        h_list = np.array([0.05, 0.025, 0.0125, 0.00625])
        errs4, errs3 = [], []
        for h in h_list:
            y4 = esrk_integrate(schnak_rhs, y0, t1, h, A, b4, args=args)
            y3 = esrk_integrate(schnak_rhs, y0, t1, h, A, b3, args=args)
            e4 = norm(y4 - ref) / np.sqrt(2 * N)
            e3 = norm(y3 - ref) / np.sqrt(2 * N)
            errs4.append(e4)
            errs3.append(e3)
            print(f"h={h:7.5f}, RMS(4th)={e4:10.3e}, RMS(3rd)={e3:10.3e}")

        errs4, errs3 = np.array(errs4), np.array(errs3)
        p4 = np.mean(np.log(errs4[:-1]/errs4[1:]) / np.log(h_list[:-1]/h_list[1:]))
        p3 = np.mean(np.log(errs3[:-1]/errs3[1:]) / np.log(h_list[:-1]/h_list[1:]))
        print(f"Observed order ≈ p4={p4:.3f}, p3={p3:.3f}")

        plt.figure()
        plt.loglog(h_list, errs4, "o-", label=f"ESRK(16,4) p≈{p4:.2f}")
        plt.loglog(h_list, errs3, "s-", label=f"ESRK(16,3) p≈{p3:.2f}")
        plt.xlabel("Δt")
        plt.ylabel("RMS Error")
        plt.grid(True, which="both", ls=":")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"esrk_schnak_convergence_scale{scale}.png", dpi=160)

        # ------------------------------------------------------------
        # Adaptive tolerance sweep
        # ------------------------------------------------------------
        tols = np.logspace(-2, -5, 7)
        global_errs, steps_list, rejects_list, flop_tol, mean_h = [], [], [], [], []
        print("\n=== Adaptive ESRK(16,4/3) Schnakenberg (scale={:.1f}) ===".format(scale))

        for tol in tols:
            flops["nfe"] = 0
            y, h_hist, steps, rejects = adaptive_integrate(
                schnak_rhs, y0, t0, t1, 0.02, A, b4, b3, tol, args=args)
            err = norm(y - ref) / np.sqrt(2 * N)
            total_flops = flops["nfe"] * FLOPS_PER_RHS
            global_errs.append(err)
            steps_list.append(steps)
            rejects_list.append(rejects)
            flop_tol.append(total_flops)
            mean_h.append(np.mean(h_hist) if len(h_hist) else 0.0)
            print(f"tol={tol:8.1e}  steps={steps:5d}  rej={rejects:3d}  "
                  f"err={err:10.3e}  FLOPs≈{total_flops:7.2e}")

        # ------------------------------------------------------------
        # Plots + CSV
        # ------------------------------------------------------------
        plt.figure()
        plt.loglog(tols, global_errs, "o-", label=f"ESRK scale={scale}")
        plt.xlabel("Tolerance")
        plt.ylabel("RMS Error")
        plt.grid(True, which="both", ls=":")
        plt.legend()
        plt.savefig(f"esrk_schnak_error_vs_tol_scale{scale}.png", dpi=160)

        plt.figure()
        plt.loglog(flop_tol, global_errs, "o-", label=f"ESRK scale={scale}")
        plt.xlabel("Estimated FLOPs")
        plt.ylabel("RMS Error")
        plt.grid(True, which="both", ls=":")
        plt.legend()
        plt.savefig(f"esrk_schnak_error_vs_flops_scale{scale}.png", dpi=160)

        with open(f"esrk_schnak_adaptive_scale{scale}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tol", "steps", "rejects", "global_err", "flops_est", "mean_h"])
            for i in range(len(tols)):
                w.writerow([
                    f"{tols[i]:.1e}", steps_list[i], rejects_list[i],
                    f"{global_errs[i]:.4e}", f"{flop_tol[i]:.4e}", f"{mean_h[i]:.4e}"
                ])

        print(f"\nSaved CSV and plots for scale={scale} (runtime so far: {time.time()-start:.1f}s)")

    print("\nAll ESRK Schnakenberg stiffness scales complete ✅")
