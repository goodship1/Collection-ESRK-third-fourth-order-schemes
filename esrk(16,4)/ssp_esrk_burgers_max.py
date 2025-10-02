#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

np.seterr(over='raise', invalid='raise', divide='raise', under='warn')

# --------------------------- helpers -----------------------------------------
def kahan_sum(values):
    s = 0.0; c = 0.0
    for v in values:
        y = v - c
        t = s + y
        c = (t - s) - y
        s = t
    return s

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

# ---- Stability function & real-axis radius (robust) --------------------------
def stability_function_value(A, b, z):
    """
    R(z) = 1 + z * b^T * (I - z A)^{-1} * 1
    Solve (I - zA) y = 1 by forward substitution (A is strictly lower-triangular).
    """
    s = len(b)
    y = np.empty(s, dtype=float)
    for i in range(s):
        acc = 0.0
        Ai = A[i]
        for j in range(i):
            aij = Ai[j]
            if aij != 0.0:
                acc += aij * y[j]
        # (I - zA) y = 1  =>  y_i - z*sum_{j<i} A_ij y_j = 1
        y[i] = 1.0 + z * acc
    return 1.0 + z * float(b @ y)

def real_axis_radius(A, b, zmin_exp=0, zmax_exp=8, refine_iters=60):
    """
    Find max |z| on negative real axis with |R(z)|<=1.
    1) coarse log sweep over z = -10^e, e in [zmin_exp..zmax_exp]
    2) bracket last-inside / first-outside
    3) bisection refine
    Returns R_R = |z*| (>=0).
    """
    zs = -10.0 ** np.linspace(zmin_exp, zmax_exp, num=200)  # negative grid, increasing magnitude
    inside_mask = np.array([abs(stability_function_value(A, b, z)) <= 1.0 for z in zs], dtype=bool)

    if not inside_mask.any():
        return 0.0

    k_inside = np.where(inside_mask)[0][-1]
    z_in = zs[k_inside]

    if k_inside == len(zs) - 1:
        # entire tested range is inside; return largest magnitude tested
        return abs(z_in)

    z_out = zs[k_inside + 1]  # first outside after last inside (more negative)
    a, c = z_out, z_in  # a: outside, c: inside

    for _ in range(refine_iters):
        m = 0.5 * (a + c)
        if not np.isfinite(m):
            break
        if abs(stability_function_value(A, b, m)) <= 1.0:
            c = m
        else:
            a = m
        if abs(c - a) <= 1e-12 * max(1.0, abs(c)):
            break

    return abs(c)

# ----------------------- Burgers 1D (periodic) --------------------------------
class Burgers1D:
    """
    u_t + u u_x = nu u_xx on [0, 2π], periodic.
    Pseudo-spectral with 2/3 de-aliasing.
    """
    def __init__(self, N=200, nu=0.02):
        self.N = int(N); self.nu = float(nu)
        self.L = 2*np.pi
        self.x  = np.linspace(0.0, self.L, self.N, endpoint=False)
        self.dx = self.L/self.N
        # Fourier wavenumbers (domain 2π so that d/dx ↔ i*k)
        self.k  = np.fft.fftfreq(self.N, d=self.dx/(2*np.pi))
        self.ik = 1j*self.k
        self.k2 = (self.k*self.k).astype(float)
        # 2/3 de-aliasing mask
        idx = np.fft.fftfreq(self.N, d=1.0/self.N)
        cutoff = int(np.floor((2/3)*(self.N/2)))
        self.dealias = (np.abs(idx) <= cutoff)

    def rhs(self, t, u):
        if not np.all(np.isfinite(u)):
            raise FloatingPointError("non-finite state")
        uhat = np.fft.fft(u)
        diff_hat   = -self.nu * self.k2 * uhat
        u2hat      = np.fft.fft(u*u) * self.dealias
        nonlin_hat = -0.5 * (self.ik * u2hat)
        return np.fft.ifft(diff_hat + nonlin_hat).real

# --------------------- explicit RK solver (A,b) --------------------------------
def rk_explicit(f, t_span, y0, h, A, b):
    t0, tf = t_span
    t = float(t0); y = y0.copy()
    s = len(b); steps = int(np.ceil((tf - t0)/h))
    c = np.array([kahan_sum(A[i][:i]) for i in range(s)], float)

    for _ in range(steps):
        if t + h > tf + 1e-14:
            h = tf - t
        if h <= 0:
            break
        k = [np.zeros_like(y) for _ in range(s)]
        for i in range(s):
            yi = y.copy()
            Ai = A[i]
            for j in range(i):
                aij = Ai[j]
                if aij != 0.0:
                    yi += h*aij*k[j]
            ti = t + c[i]*h
            k[i] = f(ti, yi)
        for i in range(s):
            bi = b[i]
            if bi != 0.0:
                y += h*bi*k[i]
        t += h
        if not np.all(np.isfinite(y)) or np.linalg.norm(y) > 1e12:
            raise FloatingPointError("solution blow-up")
    return y

# ----------------- ESRK(16,4) tableau (yours) ---------------------------------
A_ESRK16 = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0.297950632696351,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0.0892840764249344,0.522026933033341,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0.0892840764249344,0.0892840764249344,0.144349746352280,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0.0892840764249344,0.0892840764249344,0.0892840764249344,-0.000371956295732390,0,0,0,0,0,0,0,0,0,0,0,0],
    [0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,-0.124117473662160,0,0,0,0,0,0,0,0,0,0,0],
    [0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.192800131150961,0,0,0,0,0,0,0,0,0,0],
    [0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,-0.00721201688860849,0,0,0,0,0,0,0,0,0],
    [0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.385496874023061,0,0,0,0,0,0,0,0],
    [0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.248192855959921,0,0,0,0,0,0,0],
    [0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,-0.295854950063981,-4.25371891111175e-5,0,0,0,0,0,0],
    [0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,-0.295854950063981,0.163017169512979,0.138371044215410,0,0,0,0,0],
    [0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,-0.295854950063981,0.163017169512979,-0.0819824325549522,0.403108090476214,0,0,0,0],
    [0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,-0.295854950063981,0.163017169512979,-0.0819824325549522,0.546008221888163,0.125164780662438,0,0,0],
    [0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,-0.295854950063981,0.163017169512979,-0.0819824325549522,0.546008221888163,-0.0422844329611440,-0.00579862710501764,0,0],
    [0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,-0.295854950063981,0.163017169512979,-0.0819824325549522,0.546008221888163,-0.0422844329611440,0.467431197768081,0.502036131647685,0]
], dtype=float)

b_ESRK16 = np.array([
    0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,
    0.0892840764249344,0.0892840764249344,0.0892840764249344,0.0892840764249344,
    -0.29585495006398077,0.1630171695129791,-0.08198243255495223,0.5460082218881631,
    -0.04228443296114401,0.46743119776808084,-0.45495020324595,-0.01565718174267131
], dtype=float)

# ----------------- SSPRK(10,4) tableau (yours) --------------------------------
A_SSP10 = np.zeros((10, 10), dtype=float)
A_SSP10[1,0] = 1.0/6.0
A_SSP10[2,0] = 1.0/6.0; A_SSP10[2,1] = 1.0/6.0
A_SSP10[3,0] = 1.0/6.0; A_SSP10[3,1] = 1.0/6.0; A_SSP10[3,2] = 1.0/6.0
A_SSP10[4,0] = 1.0/6.0; A_SSP10[4,1] = 1.0/6.0; A_SSP10[4,2] = 1.0/6.0; A_SSP10[4,3] = 1.0/6.0
for j in range(5): A_SSP10[5, j] = 1.0/15.0
for j in range(5): A_SSP10[6, j] = 1.0/15.0
A_SSP10[6, 5] = 1.0/6.0
for j in range(5): A_SSP10[7, j] = 1.0/15.0
A_SSP10[7, 5] = 1.0/6.0; A_SSP10[7, 6] = 1.0/6.0
for j in range(5): A_SSP10[8, j] = 1.0/15.0
A_SSP10[8, 5] = 1.0/6.0; A_SSP10[8, 6] = 1.0/6.0; A_SSP10[8, 7] = 1.0/6.0
for j in range(5): A_SSP10[9, j] = 1.0/15.0
for j in range(5, 9): A_SSP10[9, j] = 1.0/6.0
b_SSP10 = np.full(10, 1.0/10.0, dtype=float)

# ------------------------------- main -----------------------------------------
if __name__ == "__main__":
    # You can reduce nu or add higher-frequency modes to push stability differences.
    N, nu, T = 200, 0.02, 0.5
    burgers = Burgers1D(N=N, nu=nu)
    x = burgers.x
    u0 = (np.sin(12*x) + 0.5*np.sin(5*x)).astype(float)
    kmax = np.max(np.abs(burgers.k))

    # Compute real-axis radii from tableaux
    R_esrk = real_axis_radius(A_ESRK16, b_ESRK16)
    R_ssp  = real_axis_radius(A_SSP10,  b_SSP10)
    print(f"Real-axis radius:  ESRK(16,4)≈{R_esrk:.6g},  SSPRK(10,4)≈{R_ssp:.6g}")

    # ==================== Experiment A: equal-Δt grid =====================
    safety = 0.7
    dt_ceiling = safety * R_esrk / (nu*(kmax**2) + 1e-30)
    total_halvings = 7
    M = 20
    exps = np.linspace(0.0, total_halvings, M)
    dts = dt_ceiling * (2.0 ** (-exps))
    dt_min = dts[-1]
    print(f"[Equal-Δt] dt_max={dts[0]:.6g}, dt_min={dt_min:.6g}, points={M}")

    def rk_to_final(h, A, b):
        return rk_explicit(lambda t,u: burgers.rhs(t,u), (0.0, T), u0, h, A, b)

    # Reference using ESRK(16,4) at dt_ref (aligned)
    dt_ref = dt_min / 8.0
    t0 = perf_counter()
    u_ref_T = rk_to_final(dt_ref, A_ESRK16, b_ESRK16)
    print(f"[ref] ESRK(16,4) dt_ref={dt_ref:.3e}, time={perf_counter()-t0:.4f}s")

    methods = [("ESRK(16,4)", A_ESRK16, b_ESRK16),
               ("SSPRK(10,4)", A_SSP10,  b_SSP10)]
    results = {}
    for name, A, b in methods:
        errs = np.full(M, np.nan); times = np.full(M, np.nan)
        for i, h in enumerate(dts):
            try:
                t1 = perf_counter()
                uT = rk_to_final(h, A, b)
                times[i] = perf_counter() - t1
                errs[i] = l2_grid_error(uT, u_ref_T)
            except FloatingPointError:
                pass
        p_pair = pairwise_eoc(errs, dts)
        i0, i1 = 2, M-2
        idx = np.arange(M)
        mask = (idx >= i0) & (idx <= i1) & np.isfinite(errs) & (errs > 0)
        p_ls = np.polyfit(np.log(dts[mask]), np.log(errs[mask]), 1)[0] if mask.sum()>=2 else np.nan
        results[name] = dict(errs=errs, times=times, p_pair=p_pair, p_ls=p_ls)
        print(f"[Equal-Δt] {name}: LS order≈{p_ls:.4f}, median err={np.nanmedian(errs):.3e}, median time={np.nanmedian(times):.4f}s")

    # Plots (equal-Δt)
    plt.figure(figsize=(7,5))
    for (label, _, _ ) in methods:
        e = results[label]['errs']; m = np.isfinite(e)
        plt.loglog(dts[m], e[m], 'o-', label=f'{label} error @ T')
    base = results['ESRK(16,4)']['errs'][2]/(dts[2]**4)
    plt.loglog(dts, base*(dts**4), '--', label='O(Δt^4) guide')
    plt.gca().invert_xaxis(); plt.xlabel('Δt'); plt.ylabel('L2 error at T')
    plt.title('Equal-Δt: ESRK(16,4) vs SSPRK(10,4)'); plt.grid(True, which='both', ls='--'); plt.legend(); plt.tight_layout()

    plt.figure(figsize=(7,4))
    for (label, _, _ ) in methods:
        p = results[label]['p_pair']; m = np.isfinite(p)
        plt.plot(dts[:-1][m], p[m], '.-', label=f'{label} pairwise EOC')
    plt.axhline(4, linestyle='--', label='4')
    plt.gca().set_xscale('log'); plt.gca().invert_xaxis()
    plt.xlabel('Δt'); plt.ylabel('Estimated order p'); plt.title('Equal-Δt: pairwise EOC')
    plt.grid(True, which='both', ls='--'); plt.legend(); plt.tight_layout()

    plt.figure(figsize=(7,4))
    for (label, _, _ ) in methods:
        t = results[label]['times']; m = np.isfinite(t)
        plt.loglog(dts[m], t[m], 'o-', label=f'{label} wall time')
    plt.gca().invert_xaxis(); plt.xlabel('Δt'); plt.ylabel('Wall time [s]')
    plt.title('Equal-Δt: cost vs step size'); plt.grid(True, which='both', ls='--'); plt.legend(); plt.tight_layout()

    # ================== Experiment B: Max-Δt showcase (per-method) ==================
    safety = 0.7
    dtmax_ESRK = safety * R_esrk / (nu*(kmax**2) + 1e-30)
    dtmax_SSP  = safety * R_ssp  / (nu*(kmax**2) + 1e-30)
    print(f"[Max-Δt] ESRK dt_max≈{dtmax_ESRK:.6g}, SSP dt_max≈{dtmax_SSP:.6g}")

    # short per-method grids near each ceiling (5 points each, halving)
    def run_grid(A, b, dt_max, K=5):
        hs = dt_max * (2.0 ** (-np.arange(K)))
        errs, times, kept_h = [], [], []
        for h in hs:
            try:
                t1 = perf_counter()
                uT = rk_to_final(h, A, b)
                times.append(perf_counter() - t1)
                errs.append(l2_grid_error(uT, u_ref_T))
                kept_h.append(h)
            except FloatingPointError:
                # skip if unstable even with safety
                continue
        return np.array(kept_h), np.array(errs), np.array(times)

    hE, eE, tE = run_grid(A_ESRK16, b_ESRK16, dtmax_ESRK)
    hS, eS, tS = run_grid(A_SSP10,  b_SSP10,  dtmax_SSP)

    plt.figure(figsize=(7,5))
    if len(tE)>0: plt.loglog(tE, eE, 'o-', label='ESRK(16,4) @ own Δt range')
    if len(tS)>0: plt.loglog(tS, eS, 'o-', label='SSPRK(10,4) @ own Δt range')
    plt.xlabel('Wall time per solve [s]'); plt.ylabel('L2 error at T')
    plt.title('Max-Δt showcase: error vs time (each at its own ceiling)')
    plt.grid(True, which='both', ls='--'); plt.legend(); plt.tight_layout()

    # Quick equal-accuracy cost estimate (fit err ≈ C h^4, time ≈ α/h)
    target_err = 5e-8
    def fit_cost(hs, errs, times):
        if len(hs) < 2: return np.nan, np.nan
        p = np.polyfit(np.log(hs), np.log(errs), 1)
        C = np.exp(p[1])  # logC
        h_star = (target_err / C)**0.25
        alpha = np.median(times * hs)  # time ≈ α/h
        t_pred = alpha / h_star
        return h_star, t_pred

    hE_star, tE_pred = fit_cost(hE, eE, tE)
    hS_star, tS_pred = fit_cost(hS, eS, tS)
    print(f"[Target err≈{target_err:.1e}] ESRK: h*≈{hE_star:.3e}, time≈{tE_pred:.3f}s | "
          f"SSP: h*≈{hS_star:.3e}, time≈{tS_pred:.3f}s")

    plt.show()
