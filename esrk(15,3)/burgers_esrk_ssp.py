#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Third-order benchmark: ESRK(15,3) vs NodePy SSPRK3(s) with s=n^2
- Robust to NodePy versions where stability_function() returns (num, den) tuple.
- We build phi(z) ourselves from (A,b): phi(z) = 1 + z * b^T (I - zA)^{-1} 1.
"""

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

# ----------------------- Your ESRK(15,3) coefficients ------------------------
A_esrk = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0.0243586417803786,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0258303808904268,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0667956303329210,0,0,0,0,0,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0140960387721938,0,0,0,0,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0412105997557866,0,0,0,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0149469583607297,0,0,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.414086419082813,0,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.00395908281378477,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.480561088337756,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.00661245794721050,0.319660987317690,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.00661245794721050,0.216746869496930,0.00668808071535874,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.00661245794721050,0.216746869496930,0,0.0374638233561973,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.00661245794721050,0.216746869496930,0,0.422645975498266,0.439499983548480,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.00661245794721050,0.216746869496930,0,0.422645975498266,0.0327614907498598,0.367805790222090,0],
], dtype=float)

b_esrk = np.array([
    0.035898932499408134, 0.035898932499408134, 0.035898932499408134,
    0.035898932499408134, 0.035898932499408134, 0.035898932499408134,
    0.035898932499408134, 0.035898932499408134, 0.006612457947210495,
    0.21674686949693006, 0.0, 0.42264597549826616,
    0.03276149074985981, 0.0330623263939421, 0.0009799086295048407
], dtype=float)

assert np.allclose(A_esrk, np.tril(A_esrk, -1)), "A (ESRK) must be strictly lower triangular."

# ------------------------------ NodePy methods -------------------------------
try:
    from nodepy import runge_kutta_method as rk
except ImportError as e:
    raise SystemExit("Needs NodePy. Install with: pip install nodepy") from e

# Build SSPRK3(s) for s=n^2
ssprk3_stage_list = [9, 16, 25, 36]
SSP_methods = [rk.SSPRK3(s) for s in ssprk3_stage_list]

# --------------------- Stability function φ(z) from (A,b) --------------------
def phi_from_Ab(A, b):
    """
    Explicit RK stability function:
    φ(z) = 1 + z * b^T (I - zA)^{-1} 1
    Implemented via forward substitution since A is strictly lower triangular.
    Vectorized over z (numpy array ok).
    """
    A = np.asarray(A, float); b = np.asarray(b, float); s = len(b)
    def phi(z):
        z = np.asarray(z, dtype=complex)
        y = np.zeros(z.shape + (s,), dtype=complex)
        for i in range(s):
            acc = 0.0 if i == 0 else np.tensordot(A[i,:i], y[..., :i], axes=(0, -1))
            y[..., i] = 1.0 + z*acc
        by = np.tensordot(b, y, axes=(0, -1))
        return 1.0 + z*by
    return phi

def real_axis_radius_from_phi(phi, start=80.0, max_to=8000.0):
    """
    Largest R such that |φ(-x)|<=1 for all x in [0,R] (approx).
    Scan & bisect on the negative real axis.
    """
    scan_to = start
    last_inside = None
    while scan_to <= max_to:
        xs = -np.linspace(0.0, scan_to, int(max(2000, scan_to*80))+1)
        vals = np.abs(phi(xs))
        inside = vals <= 1.0
        if inside.any():
            k_last = np.where(inside)[0][-1]
            last_inside = xs[k_last]
            if k_last+1 < len(xs):
                z_out = xs[k_last+1]
            else:
                scan_to *= 2
                continue
            a, c = z_out, last_inside
            for _ in range(100):
                m = 0.5*(a+c)
                if np.abs(phi(m)) <= 1.0: c = m
                else: a = m
                if np.abs(c-a) <= 1e-12*max(1.0, np.abs(c)): break
            return float(-c)
        scan_to *= 2
    if last_inside is not None:
        return float(-last_inside)
    return 0.0

# Radii for ESRK and each SSPRK3(s) using our φ-from-(A,b)
phi_esrk = phi_from_Ab(A_esrk, b_esrk)
R_esrk = real_axis_radius_from_phi(phi_esrk)

radii = {'ESRK(15,3)': R_esrk}
for m in SSP_methods:
    radii[m.name] = real_axis_radius_from_phi(phi_from_Ab(m.A, m.b))

print("\nApproximate real-axis stability radii:")
for name, Rr in radii.items():
    print(f"  {name:>12s}: {Rr:9.3f}")

# ---------------------- Smooth Burgers (periodic, spectral) ------------------
class Burgers1D:
    """
    u_t + u u_x = nu u_xx on [0, 2π], periodic.
    Pseudo-spectral with 2/3 de-aliasing.
    """
    def __init__(self, N=128, nu=0.02):
        self.N = int(N); self.nu = float(nu)
        self.L = 2*np.pi
        self.x  = np.linspace(0.0, self.L, self.N, endpoint=False)
        self.dx = self.L/self.N
        self.k  = np.fft.fftfreq(self.N, d=self.dx/(2*np.pi))
        self.ik = 1j*self.k
        self.k2 = (self.k*self.k).astype(float)
        idx = np.fft.fftfreq(self.N, d=1.0/self.N)
        cutoff = int(np.floor((2/3)*(self.N/2)))
        self.dealias = (np.abs(idx) <= cutoff)
    def rhs(self, t, u):
        uhat = np.fft.fft(u)
        diff_hat   = -self.nu * self.k2 * uhat
        u2hat      = np.fft.fft(u*u) * self.dealias
        nonlin_hat = -0.5 * (self.ik * u2hat)
        return np.fft.ifft(diff_hat + nonlin_hat).real

def rk_stepper(A, b):
    A = np.asarray(A, float); b = np.asarray(b, float); s = len(b)
    c = np.array([A[i,:i].sum() for i in range(s)], float)
    def advance(f, t_span, y0, h):
        t0, tf = t_span
        t = float(t0); y = y0.copy()
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

# Methods list
methods = [('ESRK(15,3)', A_esrk, b_esrk)]
for m in SSP_methods:
    methods.append((m.name, m.A, m.b))

# Problem setup
N, nu, T = 128, 0.02, 0.4
burgers = Burgers1D(N=N, nu=nu)
x = burgers.x
u0 = (np.sin(9*x) + 0.5*np.sin(5*x)).astype(float)

# Common Δt grid (diffusion-limited by min radius)
kmax = np.max(np.abs(burgers.k))
R_min = min(radii.values())
dt_ceiling = R_min / (nu*(kmax**2) + 1e-30)
dt_max = 0.7 * dt_ceiling
M = 18; exps = np.linspace(0.0, 6.0, M)
dts = dt_max * (2.0 ** (-exps))
dt_min = dts[-1]

# Reference (3rd order): dt_ref = dt_min / 8
advance_esrk = rk_stepper(A_esrk, b_esrk)
dt_ref = dt_min / 8.0
t0 = perf_counter()
u_ref_T = advance_esrk(burgers.rhs, (0.0, T), u0, dt_ref)
print(f"\nReference with ESRK(15,3): dt_ref={dt_ref:.3e} in {perf_counter()-t0:.3f}s")

# Runs
def l2_grid_error(u, v):
    d = u - v
    return float(np.sqrt(np.mean(d*d)))

def pairwise_eoc(errors, dts):
    e = np.asarray(errors, float); h = np.asarray(dts, float)
    p = np.full(len(e)-1, np.nan)
    for i in range(len(e)-1):
        if np.isfinite(e[i]) and np.isfinite(e[i+1]) and e[i] > 0 and e[i+1] > 0 and h[i] != h[i+1]:
            p[i] = np.log(e[i+1]/e[i]) / np.log(h[i+1]/h[i])
    return p

results = {}
for name, A, b in methods:
    adv = rk_stepper(A, b)
    errs = np.full(M, np.nan); times = np.full(M, np.nan)
    for i, h in enumerate(dts):
        try:
            t1 = perf_counter()
            uT = adv(burgers.rhs, (0.0, T), u0, h)
            times[i] = perf_counter() - t1
            errs[i] = l2_grid_error(uT, u_ref_T)
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
    print(f"{name:>12s}:  R_real≈{radii[name]:7.2f},  LS order≈{p_ls:.3f}")

# ------------------------------- Plots ---------------------------------------
plt.figure(figsize=(7,5))
for name, res in results.items():
    e = res['errs']; ok = np.isfinite(e) & (e>0)
    plt.loglog(dts[ok], e[ok], 'o-', label=f'{name} (R≈{radii[name]:.1f})')
e0 = results['ESRK(15,3)']['errs']; ok0 = np.isfinite(e0) & (e0>0)
if ok0.any():
    j = np.where(ok0)[0][0]
    base = e0[j]/(dts[j]**3)
    plt.loglog(dts, base*(dts**3), '--', label='O(Δt^3) guide')
plt.gca().invert_xaxis()
plt.xlabel('Δt'); plt.ylabel('L2 error at T')
plt.title('Smooth Burgers: ESRK(15,3) vs SSPRK3(s=n²)')
plt.grid(True, which='both', ls='--'); plt.legend(); plt.tight_layout()

plt.figure(figsize=(7,4))
for name, res in results.items():
    p_pair = res['p_pair']; ok = np.isfinite(p_pair)
    plt.plot(dts[:-1][ok], p_pair[ok], '.-', label=name)
plt.axhline(3, linestyle='--', label='3')
plt.gca().set_xscale('log'); plt.gca().invert_xaxis()
plt.xlabel('Δt'); plt.ylabel('Estimated order p')
plt.title('Local pairwise orders'); plt.grid(True, which='both', ls='--')
plt.legend(); plt.tight_layout()

plt.figure(figsize=(7,4))
for name, res in results.items():
    tms = res['times']; ok = np.isfinite(tms) & (tms>0)
    plt.loglog(dts[ok], tms[ok], 'o-', label=name)
plt.gca().invert_xaxis()
plt.xlabel('Δt'); plt.ylabel('Wall time per solve [s]')
plt.title('Cost vs step size'); plt.grid(True, which='both', ls='--')
plt.legend(); plt.tight_layout()

plt.show()
