#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brusselator ODE benchmark: ESRK(15,3) vs SSPRK3(s=n^2)

- Compares your ESRK(15,3) to high-stage third-order SSPRK3(s) (s = 9,16,25,36).
- Builds a common Δt grid, computes final-time error vs a tight reference,
  and plots: error vs Δt, pairwise EOC, cost vs Δt, and work–precision.

Requires: numpy, matplotlib, nodepy (for SSPRK3 generator). Install: pip install nodepy
"""

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

np.seterr(over='raise', invalid='raise', divide='raise', under='warn')

# ----------------------- Your ESRK(15,3) coefficients ------------------------
A_esrk = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0243586417803786, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0258303808904268, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0667956303329210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0140960387721938, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0412105997557866, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0149469583607297, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.414086419082813, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00395908281378477, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.480561088337756, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.319660987317690, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0.00668808071535874, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0, 0.0374638233561973, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0, 0.422645975498266, 0.439499983548480, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0, 0.422645975498266, 0.0327614907498598, 0.367805790222090, 0],
], dtype=float)

b_esrk = np.array([
    0.035898932499408134, 0.035898932499408134, 0.035898932499408134,
    0.035898932499408134, 0.035898932499408134, 0.035898932499408134,
    0.035898932499408134, 0.035898932499408134, 0.006612457947210495,
    0.21674686949693006, 0.0, 0.42264597549826616,
    0.03276149074985981, 0.0330623263939421, 0.0009799086295048407
], dtype=float)

assert np.allclose(A_esrk, np.tril(A_esrk, -1)), "A (ESRK) must be strictly lower-triangular."

# ------------------------------ NodePy SSPRK3(s) ------------------------------
try:
    from nodepy import runge_kutta_method as rk
except ImportError as e:
    raise SystemExit("This script needs NodePy for SSPRK3(s). Install: pip install nodepy") from e

ssprk3_stage_list = [9, 16, 25, 36]   # NodePy requires s = n^2
SSP_methods = [rk.SSPRK3(s) for s in ssprk3_stage_list]

# --------------------------- Brusselator ODE RHS ------------------------------
def brusselator_rhs(t, y, A=1.0, B=3.0):
    u, v = y
    du = A + u*u*v - (B+1.0)*u
    dv = B*u - u*u*v
    return np.array([du, dv], dtype=float)

# ------------------------- Generic explicit RK stepper ------------------------
def rk_stepper(A, b):
    A = np.asarray(A, float); b = np.asarray(b, float); s = len(b)
    c = np.array([A[i,:i].sum() for i in range(s)], float)
    def advance(f, t_span, y0, h):
        t0, tf = t_span
        t = float(t0); y = np.array(y0, dtype=float)
        steps = int(np.ceil((tf - t0)/h))
        for _ in range(steps):
            if t + h > tf + 1e-14: h = tf - t
            if h <= 0: break
            k = [np.zeros_like(y) for _ in range(s)]
            for i in range(s):
                yi = y.copy()
                for j in range(i):
                    aij = A[i,j]
                    if aij != 0.0:
                        yi += h*aij*k[j]
                k[i] = f(t + c[i]*h, yi)
            for i in range(s):
                bi = b[i]
                if bi != 0.0:
                    y += h*bi*k[i]
            t += h
            if not np.all(np.isfinite(y)) or np.linalg.norm(y) > 1e12:
                raise FloatingPointError("solution blow-up")
        return y
    return advance

# ------------------------------ Experiment setup ------------------------------
# ODE parameters and ICs
Apar, Bpar = 1.0, 3.0
y0 = np.array([1.2, 3.1], dtype=float)
T  = 20.0

# Methods list: ESRK + SSP family
methods = [('ESRK(15,3)', A_esrk, b_esrk)]
for m in SSP_methods:
    methods.append((m.name, m.A, m.b))

# Common Δt grid (non-stiff ODE → simple, conservative choice)
dt_max = 0.2      # coarse but safe for these methods/parameters
M = 18
exps = np.linspace(0.0, 6.0, M)   # about 2^6 span
dts = dt_max * (2.0 ** (-exps))
dt_min = dts[-1]

# Reference (3rd order → /8 is typical; we’ll go tighter /12)
advance_esrk = rk_stepper(A_esrk, b_esrk)
dt_ref = dt_min / 12.0
t0 = perf_counter()
y_ref_T = advance_esrk(lambda t,y: brusselator_rhs(t,y,Apar,Bpar), (0.0, T), y0, dt_ref)
print(f"[ref] ESRK(15,3) dt_ref={dt_ref:.3e}, build time={perf_counter()-t0:.3f}s")

# ------------------------------ Run & measure --------------------------------
def l2_err(u, v): return float(np.sqrt(np.sum((u-v)**2)))

def pairwise_eoc(errors, h):
    e = np.asarray(errors, float); h = np.asarray(h, float)
    p = np.full(len(e)-1, np.nan)
    for i in range(len(e)-1):
        if np.isfinite(e[i]) and np.isfinite(e[i+1]) and e[i] > 0 and e[i+1] > 0:
            p[i] = np.log(e[i+1]/e[i]) / np.log(h[i+1]/h[i])
    return p

results = {}
for name, A, b in methods:
    adv = rk_stepper(A, b)
    errs = np.full(M, np.nan); times = np.full(M, np.nan)
    for i, h in enumerate(dts):
        try:
            t1 = perf_counter()
            yT = adv(lambda t,y: brusselator_rhs(t,y,Apar,Bpar), (0.0, T), y0, h)
            times[i] = perf_counter() - t1
            errs[i] = l2_err(yT, y_ref_T)
        except FloatingPointError:
            pass
    p_pair = pairwise_eoc(errs, dts)
    i0, i1 = 2, M-2
    idx = np.arange(M)
    mask = (idx >= i0) & (idx <= i1) & np.isfinite(errs) & (errs > 0)
    if mask.sum() >= 2:
        p_ls, _ = np.polyfit(np.log(dts[mask]), np.log(errs[mask]), 1)
    else:
        p_ls = np.nan
    results[name] = dict(errs=errs, times=times, p_pair=p_pair, p_ls=p_ls)
    print(f"{name:>12s}: LS order≈{p_ls:.3f}, median err={np.nanmedian(errs):.3e}, "
          f"median time={np.nanmedian(times):.4f}s")

# ------------------------------- Plots ---------------------------------------
# Error vs dt
plt.figure(figsize=(7,5))
for name, res in results.items():
    e = res['errs']; ok = np.isfinite(e) & (e>0)
    plt.loglog(dts[ok], e[ok], 'o-', label=name)
# 3rd-order guide (normalize at first valid ESRK point)
e0 = results['ESRK(15,3)']['errs']; ok0 = np.isfinite(e0) & (e0>0)
if ok0.any():
    j = np.where(ok0)[0][0]
    base = e0[j]/(dts[j]**3)
    plt.loglog(dts, base*(dts**3), '--', label='O(Δt^3) guide')
plt.gca().invert_xaxis()
plt.xlabel('Δt'); plt.ylabel('L2 error at T')
plt.title('Brusselator ODE: ESRK(15,3) vs SSPRK3(s=n²)')
plt.grid(True, which='both', ls='--'); plt.legend(); plt.tight_layout()

# Pairwise EOC
plt.figure(figsize=(7,4))
for name, res in results.items():
    p = res['p_pair']; ok = np.isfinite(p)
    plt.plot(dts[:-1][ok], p[ok], '.-', label=name)
plt.axhline(3, linestyle='--', label='3')
plt.gca().set_xscale('log'); plt.gca().invert_xaxis()
plt.xlabel('Δt'); plt.ylabel('Estimated order p')
plt.title('Local pairwise orders'); plt.grid(True, which='both', ls='--'); plt.legend(); plt.tight_layout()

# Cost vs dt
plt.figure(figsize=(7,4))
for name, res in results.items():
    tms = res['times']; ok = np.isfinite(tms) & (tms>0)
    plt.loglog(dts[ok], tms[ok], 'o-', label=name)
plt.gca().invert_xaxis()
plt.xlabel('Δt'); plt.ylabel('Wall time per solve [s]')
plt.title('Cost vs step size'); plt.grid(True, which='both', ls='--'); plt.legend(); plt.tight_layout()

# Work–precision (error vs wall time)
plt.figure(figsize=(7,5))
for name, res in results.items():
    e = res['errs']; t = res['times']
    ok = np.isfinite(e) & (e>0) & np.isfinite(t) & (t>0)
    plt.loglog(t[ok], e[ok], 'o-', label=name)
plt.xlabel('Wall time per solve [s]'); plt.ylabel('L2 error at T')
plt.title('Work–precision: Brusselator ODE'); plt.grid(True, which='both', ls='--')
plt.legend(); plt.tight_layout()

plt.show()
