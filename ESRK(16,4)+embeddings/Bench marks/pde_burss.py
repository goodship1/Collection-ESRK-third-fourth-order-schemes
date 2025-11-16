#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESRK(16,4/3) — PDE Brusselator Multi-Scale Diagnostic
-----------------------------------------------------
Like-for-like comparison with ROCK4 (Julia).
Per-scale diagnostics:
- Fixed-step convergence (4th vs 3rd order)
- Adaptive tolerance sweep
- FLOP accounting
- CSV + plots per scale
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, solve
import csv, time, os

# ============================================================
# Load ESRK(16,4/3) coefficients
# ============================================================
A  = np.loadtxt("A.csv", delimiter=",")
b4 = np.loadtxt("b_main.csv", delimiter=",")             # 4th-order main
b3 = np.loadtxt("b_embedded_dev04.csv", delimiter=",")   # 3rd-order embedded
S  = len(b4)
I  = np.eye(S)
e1 = np.ones(S)

# ============================================================
# PDE RHS (1D Brusselator, periodic BCs)
# ============================================================
def brusselator_pde_rhs(t, y, a=1.0, b=3.0, Du=5e-4, Dv=2.5e-4, L=1.0, N=64):
    flops["nfe"] += 1
    h = L / N
    u, v = y[:N], y[N:]

    def lap(z):
        zxx = np.zeros_like(z)
        zxx[1:-1] = (z[2:] - 2*z[1:-1] + z[:-2]) / h**2
        zxx[0]    = (z[1] - 2*z[0] + z[-1]) / h**2
        zxx[-1]   = (z[0] - 2*z[-1] + z[-2]) / h**2
        return zxx

    uxx, vxx = lap(u), lap(v)
    du = Du * uxx + a - (b + 1) * u + u*u*v
    dv = Dv * vxx + b*u - u*u*v

    if not np.all(np.isfinite(du)) or not np.all(np.isfinite(dv)):
        du[:] = 0; dv[:] = 0
    return np.concatenate([du, dv])

# ============================================================
# FLOP counter and constants
# ============================================================
FLOPS_PER_RHS = 10
flops = {"nfe": 0}

# ============================================================
# ESRK Integrators
# ============================================================
def esrk_step_adaptive(f, y, t, h, A, b_hi, b_lo, *args):
    """One adaptive ESRK step."""
    k = np.zeros((S, y.size))
    for i in range(S):
        yi = y + h * np.sum(A[i, :i, None] * k[:i], axis=0)
        k[i] = f(t + h * np.sum(A[i, :]), yi, *args)
    y_hi = y + h * np.sum(b_hi[:, None] * k, axis=0)
    y_lo = y + h * np.sum(b_lo[:, None] * k, axis=0)
    err = norm(y_hi - y_lo)
    return y_hi, err


def adaptive_integrate(f, y0, t0, t1, h0, A, b_hi, b_lo, tol, args=(),
                       safety=0.9, kP=0.2, kI=0.08, h_min=1e-6, h_max=0.1):
    """Adaptive ESRK(16,4/3) integration."""
    h, y, t = h0, y0.copy(), t0
    steps = rejects = 0
    prev_err = tol
    h_hist = []

    while t < t1:
        if t + h > t1:
            h = t1 - t

        y_new, err = esrk_step_adaptive(f, y, t, h, A, b_hi, b_lo, *args)

        if not np.isfinite(err):
            rejects += 1
            h = max(h_min, h * 0.5)
            continue

        if err <= tol or err == 0.0:
            y, t = y_new, t + h
            h_hist.append(h)
        else:
            rejects += 1

        fac = (tol / (err + 1e-14))**kP * (tol / (prev_err + 1e-14))**kI
        h = np.clip(h * safety * fac, 0.1*h, 2.0*h)
        h = min(max(h, h_min), h_max)
        prev_err = err
        steps += 1

        if steps > 200000:
            break

    return y, np.array(h_hist), steps, rejects


def esrk_integrate(f, y0, T, h, A, b, args=()):
    """Fixed-step ESRK integrator."""
    y = y0.copy()
    Nsteps = int(np.ceil(T / h))
    for _ in range(Nsteps):
        k = np.zeros((S, y.size))
        for i in range(S):
            yi = y + h * np.sum(A[i, :i, None] * k[:i], axis=0)
            k[i] = f(0.0, yi, *args)
        y = y + h * np.sum(b[:, None] * k, axis=0)
    return y

# ============================================================
# Stability radius (for reference)
# ============================================================
def R_abs(b, z):
    try:
        y = solve(I - z * A, e1)
        return abs(1 + z * (b @ y))
    except np.linalg.LinAlgError:
        return np.inf

def radius_real_axis(b, x_max=200, tol=1e-3):
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
# Main Multi-Scale Diagnostics
# ============================================================
if __name__ == "__main__":
    L, N = 1.0, 64
    a, b = 1.0, 3.0
    Du_base, Dv_base = 5e-4, 2.5e-4
    scales = [0.1, 1.0, 10.0, 100.0]
    x = np.linspace(0, L, N)
    u0 = 1.0 + 0.1 * np.sin(2*np.pi*x/L)
    v0 = 3.0 + 0.1 * np.cos(2*np.pi*x/L)
    y0 = np.concatenate([u0, v0])
    t0, t1 = 0.0, 5.0

    print(f"Running ESRK(16,4/3) PDE diagnostics across scales: {scales}")

    for scale in scales:
        start = time.time()
        Du, Dv = Du_base * scale, Dv_base * scale
        args = (a, b, Du, Dv, L, N)
        print("\n===================================")
        print(f"Scale={scale:.1f}  (Du={Du:.2e}, Dv={Dv:.2e})")
        print("===================================")

        # Reference high-accuracy
        ref = esrk_integrate(brusselator_pde_rhs, y0, t1, 1e-3, A, b4, args=args)

        # --------------------------------------------------------
        # Fixed-step convergence (4th & 3rd)
        # --------------------------------------------------------
        h_list = np.array([0.05, 0.025, 0.0125, 0.00625])
        errs4, errs3 = [], []
        for h in h_list:
            y4 = esrk_integrate(brusselator_pde_rhs, y0, t1, h, A, b4, args=args)
            y3 = esrk_integrate(brusselator_pde_rhs, y0, t1, h, A, b3, args=args)
            e4 = norm(y4 - ref) / np.sqrt(2*N)
            e3 = norm(y3 - ref) / np.sqrt(2*N)
            errs4.append(e4); errs3.append(e3)
            print(f"h={h:7.5f}, RMS(4th)={e4:10.3e}, RMS(3rd)={e3:10.3e}")

        p4 = np.mean([np.log(errs4[i]/errs4[i+1])/np.log(h_list[i]/h_list[i+1])
                      for i in range(len(errs4)-1)])
        p3 = np.mean([np.log(errs3[i]/errs3[i+1])/np.log(h_list[i]/h_list[i+1])
                      for i in range(len(errs3)-1)])
        print(f"Observed order ≈ p4={p4:.3f}, p3={p3:.3f}")

        plt.figure()
        plt.loglog(h_list, errs4, "o-", label=f"ESRK(16,4) p≈{p4:.2f}")
        plt.loglog(h_list, errs3, "s--", label=f"ESRK(16,3) p≈{p3:.2f}")
        plt.xlabel("Δt"); plt.ylabel("RMS Error")
        plt.legend(); plt.grid(True, which="both", ls=":")
        plt.tight_layout()
        plt.savefig(f"esrk_pde_convergence_scale_{scale}.png", dpi=160)

        # --------------------------------------------------------
        # Adaptive tolerance sweep
        # --------------------------------------------------------
        tols = np.logspace(-2, -5, num=7)
        global_errs, steps_list, rejects_list, mean_h, std_h, flop_tol = [], [], [], [], [], []

        for tol in tols:
            flops["nfe"] = 0
            y, h_hist, steps, rejects = adaptive_integrate(
                brusselator_pde_rhs, y0, t0, t1, 0.02, A, b4, b3, tol, args=args)
            err = norm(y - ref) / np.sqrt(2*N)
            total_flops = flops["nfe"] * FLOPS_PER_RHS
            global_errs.append(err)
            steps_list.append(steps)
            rejects_list.append(rejects)
            mean_h.append(np.mean(h_hist) if len(h_hist) else 0)
            std_h.append(np.std(h_hist) if len(h_hist) else 0)
            flop_tol.append(total_flops)
            print(f"tol={tol:8.1e} steps={steps:5d} rej={rejects:3d} "
                  f"err={err:10.3e} FLOPs≈{total_flops:7.2e}")

        # --------------------------------------------------------
        # Plots
        # --------------------------------------------------------
        plt.figure(); plt.loglog(tols, global_errs, "o-")
        plt.xlabel("Tolerance"); plt.ylabel("RMS Error")
        plt.grid(True, which="both", ls=":")
        plt.title(f"ESRK(16,4/3) Error vs Tolerance (scale={scale})")
        plt.savefig(f"esrk_pde_error_vs_tol_scale_{scale}.png", dpi=160)

        plt.figure(); plt.loglog(flop_tol, global_errs, "o-")
        plt.xlabel("Estimated FLOPs"); plt.ylabel("RMS Error")
        plt.grid(True, which="both", ls=":")
        plt.title(f"ESRK(16,4/3) Error vs FLOPs (scale={scale})")
        plt.savefig(f"esrk_pde_error_vs_flops_scale_{scale}.png", dpi=160)

        # --------------------------------------------------------
        # CSV export
        # --------------------------------------------------------
        csv_name = f"esrk_pde_adaptive_summary_scale_{scale}.csv"
        with open(csv_name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tol","steps","rejects","mean_h","std_h","global_err","flops_est","scale"])
            for i in range(len(tols)):
                w.writerow([
                    f"{tols[i]:.1e}", steps_list[i], rejects_list[i],
                    f"{mean_h[i]:.4e}", f"{std_h[i]:.4e}",
                    f"{global_errs[i]:.4e}", f"{flop_tol[i]:.4e}", scale
                ])

        print(f"Saved CSV and plots for scale={scale} (runtime={time.time()-start:.1f}s)")
    print("\nAll ESRK diagnostic scales complete ✅")
