#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

np.seterr(over='raise', invalid='raise', divide='raise', under='warn')

# ---------- your raw rows (list of lists); some lines can be short/long ----------
A21_rows = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0477859117523170, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0259186933858971, -0.000342225369733892, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0259186933858971, 0.0259186933858971, -0.0379306642681654, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0713548421395141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259359352931570, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, -0.00953495091906422, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0904519523018936, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, -0.000396135089732896, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, -0.153935717033075, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0106794630936747, 0.000795951292330683, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0106794630936747, -0.115335444191199, -0.119588952205909, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0106794630936747, -0.115335444191199, 0.157354569317741, 0.164687679052309, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0106794630936747, -0.115335444191199, 0.157354569317741, 0.0996953916040489, -0.151151371693320, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0106794630936747, -0.115335444191199, 0.157354569317741, 0.0996953916040489, 0.124626570680465, -0.185777493787929, 0, 0, 0, 0, 0, 0, 0],
    [0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0106794630936747, -0.115335444191199, 0.157354569317741, 0.0996953916040489, 0.124626570680465, 0.0999254297810373, 0.181570806943121, 0, 0, 0, 0, 0, 0],
    [0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0106794630936747, -0.115335444191199, 0.157354569317741, 0.0996953916040489, 0.124626570680465, 0.0999254297810373, -0.157721301562393, 9.54651547687642e-5, 0, 0, 0, 0, 0],
    [0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0106794630936747, -0.115335444191199, 0.157354569317741, 0.0996953916040489, 0.124626570680465, 0.0999254297810373, -0.157721301562393, 0.171838581104214, 0.188961619753159, 0, 0, 0, 0],
    [0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0106794630936747, -0.115335444191199, 0.157354569317741, 0.0996953916040489, 0.124626570680465, 0.0999254297810373, -0.157721301562393, 0.171838581104214, -0.159282253882384, 0.163589906237245, 0, 0, 0],
    [0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0106794630936747, -0.115335444191199, 0.157354569317741, 0.0996953916040489, 0.124626570680465, 0.0999254297810373, -0.157721301562393, 0.171838581104214, -0.159282253882384, 0.153692305711512, -0.000244631681385317, 0, 0],
    [0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0259186933858971, 0.0106794630936747, -0.115335444191199, 0.157354569317741, 0.0996953916040489, 0.124626570680465, 0.0999254297810373, -0.157721301562393, 0.171838581104214, -0.159282253882384, 0.153692305711512, 0.135802482016176, -0.146394354124576, 0]
]

b21 = np.array([
    0.025918693385897126, 0.025918693385897126, 0.025918693385897126,
    0.025918693385897126, 0.025918693385897126, 0.025918693385897126,
    0.025918693385897126, 0.025918693385897126, 0.010679463093674657,
   -0.11533544419119937,  0.15735456931774092,  0.09969539160404886,
    0.12462657068046491,  0.09992542978103731, -0.1577213015623934,
    0.17183858110421352, -0.1592822538823839,  0.15369230571151177,
    0.13580248201617606,  0.12674481627127573,  0.14462984296865586
], dtype=float)
s = len(b21)

# ---- normalize/pad/clip into a strict lower-triangular 21x21 matrix ----------
def build_strict_lower(rows, s):
    M = np.zeros((s, s), dtype=float)
    for i, row in enumerate(rows):
        if i >= s: break
        L = len(row)
        if L != s:
            print(f"[warn] row {i+1} has {L} entries, expected {s}; "
                  f"{'padding' if L<s else 'clipping'} applied.")
        # keep only the first i entries (strict lower-triangular), pad if needed
        take = min(i, L)
        M[i, :take] = row[:take]
        if L < i:
            # implicit padding with zeros already in M
            pass
    return M

A21 = build_strict_lower(A21_rows, s)
assert np.allclose(A21, np.tril(A21, -1)), "A21 is not strictly lower-triangular."

# -------------------------- NodePy SSPRK3(s=n^2) ------------------------------
try:
    from nodepy import runge_kutta_method as rk
except ImportError as e:
    raise SystemExit("This script needs NodePy for SSPRK3(s). Install:  pip install nodepy") from e

ss_list = [9, 16, 25, 36,49,64]
SSP_methods = [rk.SSPRK3(s) for s in ss_list]

# ----------------------------- Brusselator RHS --------------------------------
def f_bruss(t, y, A=1.0, B=3.0):
    u, v = y
    du = A + u*u*v - (B+1.0)*u
    dv = B*u - u*u*v
    return np.array([du, dv], dtype=float)

# ------------------------------ RK stepper ------------------------------------
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
                Ai = A[i]
                for j in range(i):
                    aij = Ai[j]
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

# -------------------------- Stability radius (real axis) ----------------------
def phi_from_Ab(A, b):
    s = len(b)
    A = np.asarray(A, float); b = np.asarray(b, float)
    def phi(z):
        z = np.asarray(z, dtype=complex)
        y = np.zeros(z.shape + (s,), dtype=complex)
        for i in range(s):
            acc = 0.0 if i == 0 else np.tensordot(A[i,:i], y[..., :i], axes=(0, -1))
            y[..., i] = 1.0 + z*acc
        by = np.tensordot(b, y, axes=(0, -1))
        return 1.0 + z*by
    return phi

def real_axis_radius(phi, start=80.0, max_to=100000.0):
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
                scan_to *= 2; continue
            a, c = z_out, last_inside
            for _ in range(80):
                m = 0.5*(a+c)
                if np.abs(phi(m)) <= 1.0: c = m
                else: a = m
                if np.abs(c-a) <= 1e-12*max(1.0, np.abs(c)): break
            return float(-c)
        scan_to *= 2
    if last_inside is not None:
        return float(-last_inside)
    return 0.0

R_real = real_axis_radius(phi_from_Ab(A21, b21), start=120.0, max_to=200000.0)
print(f"[ESRK(21,3)] estimated real-axis radius R_R ≈ {R_real:.3f}")

# ------------------------------ Experiment setup ------------------------------
Apar, Bpar = 1.0, 3.0
y0 = np.array([1.2, 3.1], dtype=float)
T  = 20.0

methods = [('ESRK(21,3)', A21, b21)]
for m in SSP_methods:
    methods.append((m.name, m.A, m.b))

# Δt grid (shared for all)
dt_max = 0.2
M = 18
exps = np.linspace(0.0, 6.0, M)  # ~2^6 span
dts = dt_max*(2.0**(-exps))
dt_min = dts[-1]

# Reference with ESRK(21,3) (aligned refine)
advance_esrk = rk_stepper(A21, b21)
dt_ref = dt_min/12.0
t0 = perf_counter()
y_ref_T = advance_esrk(lambda t,y: f_bruss(t,y,Apar,Bpar), (0.0, T), y0, dt_ref)
print(f"[ref] dt_ref={dt_ref:.3e}, time={perf_counter()-t0:.3f}s")

# ------------------------------- Run all --------------------------------------
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
            yT = adv(lambda t,y: f_bruss(t,y,Apar,Bpar), (0.0, T), y0, h)
            times[i] = perf_counter() - t1
            errs[i] = l2_err(yT, y_ref_T)
        except FloatingPointError:
            pass
    p_pair = pairwise_eoc(errs, dts)
    # LS slope on interior
    i0, i1 = 2, M-2
    mask = (np.arange(M) >= i0) & (np.arange(M) <= i1) & np.isfinite(errs) & (errs > 0)
    p_ls = np.polyfit(np.log(dts[mask]), np.log(errs[mask]), 1)[0] if mask.sum() >= 2 else np.nan
    results[name] = dict(errs=errs, times=times, p_pair=p_pair, p_ls=p_ls)
    print(f"{name:>12s}: LS order≈{p_ls:.3f}, median err={np.nanmedian(errs):.3e}, "
          f"median time={np.nanmedian(times):.4f}s")

# ------------------------------- Plots ----------------------------------------
plt.figure(figsize=(7,5))
for name, res in results.items():
    e = res['errs']; ok = np.isfinite(e) & (e>0)
    plt.loglog(dts[ok], e[ok], 'o-', label=name)
# 3rd order guide
es = results['ESRK(21,3)']['errs']; ok = np.isfinite(es) & (es>0)
if ok.any():
    j = np.where(ok)[0][0]
    base = es[j]/(dts[j]**3)
    plt.loglog(dts, base*(dts**3), '--', label='O(Δt^3) guide')
plt.gca().invert_xaxis()
plt.xlabel('Δt'); plt.ylabel('L2 error at T')
plt.title('Brusselator ODE: ESRK(21,3) vs SSPRK3(s=n²)')
plt.grid(True, which='both', ls='--'); plt.legend(); plt.tight_layout()

plt.figure(figsize=(7,4))
for name, res in results.items():
    p = res['p_pair']; ok = np.isfinite(p)
    plt.plot(dts[:-1][ok], p[ok], '.-', label=name)
plt.axhline(3, linestyle='--', label='3')
plt.gca().set_xscale('log'); plt.gca().invert_xaxis()
plt.xlabel('Δt'); plt.ylabel('Estimated order p')
plt.title('Local pairwise orders'); plt.grid(True, which='both', ls='--'); plt.legend(); plt.tight_layout()

plt.figure(figsize=(7,4))
for name, res in results.items():
    tms = res['times']; ok = np.isfinite(tms) & (tms>0)
    plt.loglog(dts[ok], tms[ok], 'o-', label=name)
plt.gca().invert_xaxis()
plt.xlabel('Δt'); plt.ylabel('Wall time per solve [s]')
plt.title('Cost vs step size'); plt.grid(True, which='both', ls='--'); plt.legend(); plt.tight_layout()

plt.figure(figsize=(7,5))
for name, res in results.items():
    e, t = res['errs'], res['times']
    ok = np.isfinite(e) & (e>0) & np.isfinite(t) & (t>0)
    plt.loglog(t[ok], e[ok], 'o-', label=name)
plt.xlabel('Wall time per solve [s]'); plt.ylabel('L2 error at T')
plt.title('Work–precision: Brusselator ODE'); plt.grid(True, which='both', ls='--'); plt.legend(); plt.tight_layout()

plt.show()
