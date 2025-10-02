#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Viscous Burgers 1D (periodic, pseudo-spectral, 2/3 de-aliasing)
Third-order study with your ESRK(21,3) + SSPRK3(s) competitors (large s).

Outputs:
- burgers21_error_dt.png
- burgers21_pairwise.png
- burgers21_cost.png
"""

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

np.seterr(over='raise', invalid='raise', divide='raise', under='warn')

# ==================== ESRK(21,3) coefficients (YOUR DATA) =====================
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

def build_strict_lower(rows, s):
    A = np.zeros((s, s), dtype=float)
    for i, row in enumerate(rows):
        take = min(i, len(row))
        A[i, :take] = row[:take]
    return A

A21 = build_strict_lower(A21_rows, len(b21))
assert np.allclose(A21, np.tril(A21, -1)), "A must be strictly lower triangular"

# ====================== RK stepper & stability (φ from A,b) ===================
def rk_stepper(A, b):
    A = np.asarray(A, float); b = np.asarray(b, float); s = len(b)
    c = np.array([A[i,:i].sum() for i in range(s)], float)
    def advance(rhs, t_span, y0, h):
        t0, tf = t_span
        t = float(t0); y = y0.copy()
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
                k[i] = rhs(t + c[i]*h, yi)
            for i in range(s):
                bi = b[i]
                if bi != 0.0:
                    y += h*bi*k[i]
            t += h
            if not np.all(np.isfinite(y)) or np.linalg.norm(y) > 1e12:
                raise FloatingPointError("solution blow-up")
        return y
    return advance

def phi_from_Ab(A, b):
    A = np.asarray(A, float); b = np.asarray(b, float); s = len(b)
    def phi(z):
        z = np.asarray(z, dtype=complex)
        y = np.zeros(z.shape + (s,), dtype=complex)
        for i in range(s):
            acc = 0.0 if i == 0 else np.tensordot(A[i,:i], y[..., :i], axes=(0,-1))
            y[..., i] = 1.0 + z*acc
        by = np.tensordot(b, y, axes=(0,-1))
        return 1.0 + z*by
    return phi

def radius_on_ray(phi, theta, start=40.0, max_to=2.0e5):
    """Largest r with |phi(r e^{iθ})|<=1, by scan + bisection."""
    eitheta = np.exp(1j*theta)
    scan_to = start
    last_inside = None
    while scan_to <= max_to:
        rs = np.linspace(0.0, scan_to, int(max(2000, scan_to*80))+1)
        vals = np.abs(phi(rs*eitheta))
        inside = vals <= 1.0
        if inside.any():
            k_last = np.where(inside)[0][-1]
            r_in = rs[k_last]; last_inside = r_in
            if k_last+1 < len(rs): r_out = rs[k_last+1]
            else: scan_to *= 2; continue
            a, c = r_out, r_in
            for _ in range(100):
                m = 0.5*(a+c)
                if np.abs(phi(m*eitheta)) <= 1.0: c = m
                else: a = m
                if abs(c-a) <= 1e-12*max(1.0, abs(c)): break
            return float(c)
        scan_to *= 2
    return float(last_inside) if last_inside is not None else 0.0

# ======================== Burgers PDE (periodic, spectral) ====================
class Burgers1D:
    """
    u_t + u u_x = nu u_xx on [0, 2π], periodic.
    Pseudo-spectral with 2/3 de-aliasing.
    """
    def __init__(self, N=256, nu=0.02):
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
        if not np.all(np.isfinite(u)): raise FloatingPointError("non-finite state")
        uhat = np.fft.fft(u)
        diff_hat   = -self.nu * self.k2 * uhat
        u2hat      = np.fft.fft(u*u) * self.dealias
        nonlin_hat = -0.5 * (self.ik * u2hat)
        out = np.fft.ifft(diff_hat + nonlin_hat).real
        if not np.all(np.isfinite(out)): raise FloatingPointError("non-finite rhs")
        return out

# ============================= Utilities/metrics ==============================
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

def plateau_ls_order(dts, errs, p_target=3.0, tol=0.3, min_pairs=4):
    """Fit LS slope on the EOC~p plateau; fallback to last few points."""
    p_pair = pairwise_eoc(errs, dts)
    valid_pairs = np.isfinite(p_pair) & (errs[:-1] > 0)
    plateau = valid_pairs & (np.abs(p_pair - p_target) <= tol)
    if plateau.sum() >= min_pairs:
        fit_mask = np.zeros_like(errs, dtype=bool)
        fit_mask[np.where(plateau)[0] + 1] = True
    else:
        ok = (errs > 0) & np.isfinite(errs)
        idx = np.where(ok)[0]
        idx = idx[-6:] if len(idx) >= 6 else idx
        fit_mask = np.zeros_like(errs, dtype=bool)
        fit_mask[idx] = True
    if fit_mask.sum() >= 2:
        p_ls, _ = np.polyfit(np.log(dts[fit_mask]), np.log(errs[fit_mask]), 1)
    else:
        p_ls = np.nan
    return p_ls, p_pair

def richardson_ref(stepper, rhs, T, y0, h, p=3):
    """O(h^{p+1}) reference via Richardson with a pth-order method."""
    u_h  = stepper(rhs, (0.0, T), y0.copy(), h)
    u_h2 = stepper(rhs, (0.0, T), y0.copy(), h/2)
    return u_h2 + (u_h2 - u_h) / (2**p - 1)

# =================================== Main =====================================
if __name__ == "__main__":
    # ESRK method + stability radius
    esrk_name = "ESRK(21,3)"
    esrk_step = rk_stepper(A21, b21)
    esrk_phi  = phi_from_Ab(A21, b21)
    Rr_esrk   = radius_on_ray(esrk_phi, np.pi)    # negative real axis
    print(f"{esrk_name}: estimated real-axis stability radius R_R ≈ {Rr_esrk:.2f}")

    # Try to import NodePy for SSPRK3(s) competitors
    methods = [(esrk_name, A21, b21, esrk_step, Rr_esrk)]
    have_nodepy = True
    try:
        from nodepy import runge_kutta_method as rk
    except Exception as e:
        have_nodepy = False
        print("\n(NodePy not found; to add SSP competitors: pip install nodepy)\n")

    if have_nodepy:
        # Pick large perfect-square stage counts so R_R gets close to ~144
        s_list = [9, 16, 25, 36, 49, 64, 81, 100]   # n^2 stages
        for s in s_list:
            m = rk.SSPRK3(s)        # NodePy's family with m=n^2 stages
            phi_m = phi_from_Ab(m.A, m.b)
            Rr_m  = radius_on_ray(phi_m, np.pi)
            methods.append((m.name, m.A, m.b, rk_stepper(m.A, m.b), Rr_m))

        print("\nApproximate real-axis stability radii:")
        for name, _, _, _, Rr in methods:
            print(f"  {name:>12s}: R_R ≈ {Rr:7.2f}")

    # Burgers setup (smooth, shock-free horizon)
    N, nu, T = 256, 0.02, 0.4
    pb = Burgers1D(N=N, nu=nu)
    x  = pb.x
    u0 = (np.sin(9*x) + 0.5*np.sin(5*x)).astype(float)

    # Diffusion-limited λmax (controls real-axis step)
    kmax = np.max(np.abs(pb.k))
    lam_real = nu * (kmax**2) + 1e-300
    safety = 0.6

    # Per-method Δt grids
    M = 18
    grids = {}
    for name, _, _, _, Rr in methods:
        dt_ceiling = safety * (Rr / lam_real)
        grids[name] = dt_ceiling * (2.0 ** (-np.linspace(0.0, 6.0, M)))

    # Reference via ESRK Richardson at a modest step from ESRK grid
    dts_esrk = grids[esrk_name]
    h_ref = dts_esrk[-3]  # small but not crazy
    t0 = perf_counter()
    u_ref_T = richardson_ref(esrk_step, pb.rhs, T, u0, h_ref, p=3)
    print(f"\n[ref] Richardson with h={h_ref:.3e} computed in {perf_counter()-t0:.2f}s\n")

    # Run all methods
    results = {}
    for name, A, b, step, _ in methods:
        dts = grids[name]
        errs = np.full_like(dts, np.nan, dtype=float)
        times = np.full_like(dts, np.nan, dtype=float)
        for i, h in enumerate(dts):
            try:
                t1 = perf_counter()
                uT = step(pb.rhs, (0.0, T), u0.copy(), h)
                times[i] = perf_counter() - t1
                errs[i]  = l2_grid_error(uT, u_ref_T)
            except FloatingPointError:
                pass
        p_ls, p_pair = plateau_ls_order(dts, errs, p_target=3.0, tol=0.3, min_pairs=4)
        results[name] = dict(dts=dts, errs=errs, times=times, p_pair=p_pair, p_ls=p_ls)
        print(f"{name:>12s}: LS order on plateau ≈ {p_ls:.4f}")

    # --------- Plots ---------
    # Error vs dt + O(dt^3) guide (anchor on ESRK plateau point)
    plt.figure(figsize=(7,5))
    # anchor
    okE = (results[esrk_name]['errs']>0) & np.isfinite(results[esrk_name]['errs'])
    if okE.any():
        # pick a middle valid point as anchor
        j = np.where(okE)[0][min(5, np.sum(okE)-1)]
        base = results[esrk_name]['errs'][j]/(results[esrk_name]['dts'][j]**3)
        plt.loglog(results[esrk_name]['dts'], base*(results[esrk_name]['dts']**3), '--', label='O(Δt^3) guide')
    # curves
    for name, res in results.items():
        dts, errs = res['dts'], res['errs']
        ok = (errs>0) & np.isfinite(errs)
        plt.loglog(dts[ok], errs[ok], 'o-', label=f'{name}')
    plt.gca().invert_xaxis()
    plt.xlabel('Δt'); plt.ylabel('L2 error at T')
    plt.title('Viscous Burgers (ν=0.02): error vs Δt')
    plt.grid(True, which='both', ls='--'); plt.legend(); plt.tight_layout()
    plt.savefig("burgers21_error_dt.png", dpi=150)

    # Pairwise EOC
    plt.figure(figsize=(7,4))
    for name, res in results.items():
        dts, p_pair = res['dts'], res['p_pair']
        okp = np.isfinite(p_pair)
        plt.plot(dts[:-1][okp], p_pair[okp], '.-', label=name)
    plt.axhline(3, linestyle='--', label='3')
    plt.gca().set_xscale('log'); plt.gca().invert_xaxis()
    plt.xlabel('Δt'); plt.ylabel('Estimated order p')
    plt.title('Local pairwise orders'); plt.grid(True, which='both', ls='--')
    plt.legend(ncol=2); plt.tight_layout()
    plt.savefig("burgers21_pairwise.png", dpi=150)

    # Cost vs step size
    plt.figure(figsize=(7,4))
    for name, res in results.items():
        dts, times = res['dts'], res['times']
        okt = np.isfinite(times) & (times>0)
        plt.loglog(dts[okt], times[okt], 'o-', label=name)
    plt.gca().invert_xaxis()
    plt.xlabel('Δt'); plt.ylabel('Wall time per solve [s]')
    plt.title('Cost vs step size'); plt.grid(True, which='both', ls='--')
    plt.legend(ncol=2); plt.tight_layout()
    plt.savefig("burgers21_cost.png", dpi=150)

    plt.show()
    print("Saved: burgers21_error_dt.png, burgers21_pairwise.png, burgers21_cost.png")
