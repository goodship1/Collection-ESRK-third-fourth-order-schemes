#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

np.seterr(over='raise', invalid='raise', divide='raise', under='warn')

# ==================== ESRK(21,3) coefficients (your data) =====================
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
    M = np.zeros((s, s), dtype=float)
    for i, row in enumerate(rows):
        take = min(i, len(row))
        M[i, :take] = row[:take]
    return M

A21 = build_strict_lower(A21_rows, len(b21))
assert np.allclose(A21, np.tril(A21, -1))

# ============================= NodePy competitors =============================
try:
    from nodepy import runge_kutta_method as rk
except ImportError as e:
    raise SystemExit("Please install NodePy:\n  pip install nodepy") from e

SSP_methods = [rk.SSPRK3(m) for m in (9, 16, 25, 36)]  # s = n^2

# ============================ RK stepper & stability ==========================
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
        return y
    return advance

def phi_from_Ab(A, b):
    s = len(b)
    A = np.asarray(A, float); b = np.asarray(b, float)
    def phi(z):
        z = np.asarray(z, dtype=complex)
        y = np.zeros(z.shape + (s,), dtype=complex)
        for i in range(s):
            acc = 0.0 if i==0 else np.tensordot(A[i,:i], y[..., :i], axes=(0,-1))
            y[..., i] = 1.0 + z*acc
        by = np.tensordot(b, y, axes=(0,-1))
        return 1.0 + z*by
    return phi

def radius_on_ray(phi, theta, start=40.0, max_to=100000.0):
    scan_to = start
    eitheta = np.exp(1j*theta)
    last_inside = None
    while scan_to <= max_to:
        rs = np.linspace(0.0, scan_to, int(max(2000, scan_to*60))+1)
        vals = np.abs(phi(rs*eitheta))
        inside = vals <= 1.0
        if inside.any():
            k_last = np.where(inside)[0][-1]
            r_in = rs[k_last]
            last_inside = r_in
            if k_last+1 < len(rs):
                r_out = rs[k_last+1]
            else:
                scan_to *= 2; continue
            a, c = r_out, r_in
            for _ in range(80):
                m = 0.5*(a+c)
                if np.abs(phi(m*eitheta)) <= 1.0: c = m
                else: a = m
                if abs(c-a) <= 1e-12*max(1.0, abs(c)): break
            return float(c)
        scan_to *= 2
    return float(last_inside) if last_inside is not None else 0.0

def method_radii(A, b):
    phi = phi_from_Ab(A, b)
    return radius_on_ray(phi, np.pi), radius_on_ray(phi, np.pi/2.0)

# ================================= Heat PDE ===================================
class Fourier1D:
    def __init__(self, N):
        self.N = N
        self.L = 2*np.pi
        self.x = np.linspace(0.0, self.L, N, endpoint=False)
        self.dx = self.L/N
        self.k = np.fft.fftfreq(N, d=self.dx/(2*np.pi))   # integer wavenumbers

def make_heat(N=512, nu=0.02):
    F = Fourier1D(N)
    k2 = (F.k**2).astype(float)
    def rhs(t, u):
        uhat = np.fft.fft(u)
        return np.fft.ifft(-nu*k2*uhat).real
    def exact(u0, T):
        u0_hat = np.fft.fft(u0)
        return np.fft.ifft(u0_hat*np.exp(-nu*k2*T)).real
    lam_real = nu*(np.max(np.abs(F.k))**2)   # |λ|max on real axis
    return F, rhs, exact, lam_real

# ============== helpers for errors and EOC (order of convergence) =============
def l2_err(u, v): 
    d = u - v
    return float(np.sqrt(np.mean(d*d)))

def pairwise_eoc(errors, dts):
    e = np.asarray(errors, float); h = np.asarray(dts, float)
    p = np.full(len(e)-1, np.nan)
    for i in range(len(e)-1):
        if e[i] > 0 and e[i+1] > 0 and h[i] != h[i+1]:
            p[i] = np.log(e[i+1]/e[i]) / np.log(h[i+1]/h[i])
    return p

# =================================== Main =====================================
if __name__ == "__main__":
    # Problem & IC
    N, nu, T = 512, 0.02, 0.5
    F, rhs, exact, lam_real = make_heat(N=N, nu=nu)
    x = F.x
    u0 = np.sin(3*x) + 0.5*np.cos(5*x)

    # Methods
    ESRK = ("ESRK(21,3)", A21, b21)
    SSPs = [(m.name, m.A, m.b) for m in SSP_methods]

    # Stability radii (we only need real-axis for heat)
    Rr_E, Ri_E = method_radii(A21, b21)
    print(f"ESRK(21,3): R_real≈{Rr_E:.2f}, R_imag≈{Ri_E:.2f}")
    radii = {ESRK[0]: (Rr_E, Ri_E)}
    for name, A, b in SSPs:
        rr, ii = method_radii(A, b)
        print(f"{name:>8s}: R_real≈{rr:.2f}, R_imag≈{ii:.2f}")
        radii[name] = (rr, ii)

    # Per-method Δt ceilings (real-axis)
    kmax = np.max(np.abs(F.k))
    dtceil = {name: 0.7 * (Rr / lam_real) for name, (Rr, _) in radii.items()}

    # Build Δt grids (16 points: halvings)
    M = 16
    grids = {name: dtceil[name] * (2.0 ** (-np.linspace(0.0, 6.0, M)))
             for name in radii}

    # Reference (analytic)
    u_ref_T = exact(u0, T)

    # Runner
    advance = {ESRK[0]: rk_stepper(ESRK[1], ESRK[2])}
    for name, A, b in SSPs:
        advance[name] = rk_stepper(A, b)

    results = {}
    print("\nLeast-squares convergence order (interior fit):")
    for name in advance:
        dts = grids[name]
        errs = np.zeros_like(dts)
        times = np.zeros_like(dts)
        for i, h in enumerate(dts):
            t0 = perf_counter()
            uT = advance[name](rhs, (0.0, T), u0.copy(), h)
            times[i] = perf_counter() - t0
            errs[i] = l2_err(uT, u_ref_T)

        # EOC computations
        p_pair = pairwise_eoc(errs, dts)
        idx = np.arange(M)
        i0, i1 = 2, M-3  # interior fit to avoid edge effects
        mask = (idx >= i0) & (idx <= i1) & np.isfinite(errs) & (errs > 0)
        if mask.sum() >= 2:
            slope, C = np.polyfit(np.log(dts[mask]), np.log(errs[mask]), 1)
            p_ls = slope  # should be ≈ 3
        else:
            p_ls = np.nan

        print(f"  {name:>10s}: LS order ≈ {p_ls:.4f}")

        # Save per-method CSV
        # Note: p_pair has length M-1; we pad with NaN to align columns.
        p_pad = np.append(p_pair, np.nan)
        data = np.stack([dts, errs, times, p_pad], axis=1)
        header = "dt,error_L2,time_sec,pairwise_EOC(next)"
        np.savetxt(f"heat_{name.replace('(','_').replace(')','')}.csv",
                   data, delimiter=",", header=header, comments="")

        results[name] = (dts, errs, times, p_pair, p_ls)

    # --------- Plots ---------
    # error vs dt
    plt.figure(figsize=(7,5))
    for name,(dts,errs,_,_,_) in results.items():
        ok = (errs>0) & np.isfinite(errs)
        plt.loglog(dts[ok], errs[ok], 'o-', label=name)
    # O(dt^3) guide
    dtsE, errsE, *_ = results["ESRK(21,3)"]
    base = errsE[5]/(dtsE[5]**3)
    plt.loglog(dtsE, base*(dtsE**3), '--', label="O(Δt^3) guide")
    plt.gca().invert_xaxis()
    plt.xlabel('Δt'); plt.ylabel('L2 error at T')
    plt.title('Heat equation (ν=0.02): error vs Δt')
    plt.grid(True, which='both', ls='--'); plt.legend(); plt.tight_layout()
    plt.savefig("heat_error_dt.png", dpi=150)

    # work–precision
    plt.figure(figsize=(7,5))
    for name,(_,errs,times,_,_) in results.items():
        ok = (errs>0) & np.isfinite(errs) & (times>0)
        plt.loglog(times[ok], errs[ok], 'o-', label=name)
    plt.xlabel('Wall time per solve [s]'); plt.ylabel('L2 error at T')
    plt.title('Heat equation (ν=0.02): work–precision')
    plt.grid(True, which='both', ls='--'); plt.legend(); plt.tight_layout()
    plt.savefig("heat_work_precision.png", dpi=150)

    # pairwise EOC
    plt.figure(figsize=(7,4.5))
    for name,(dts,_,_,p_pair,_) in results.items():
        ok = np.isfinite(p_pair)
        plt.plot(dts[:-1][ok], p_pair[ok], '.-', label=name)
    plt.axhline(3.0, linestyle='--', label='Order 3')
    plt.gca().set_xscale('log'); plt.gca().invert_xaxis()
    plt.xlabel('Δt'); plt.ylabel('Pairwise EOC')
    plt.title('Heat equation (ν=0.02): pairwise order of convergence')
    plt.grid(True, which='both', ls='--'); plt.legend(); plt.tight_layout()
    plt.savefig("heat_pairwise_eoc.png", dpi=150)

    plt.show()
    print("\nSaved files:")
    print("  heat_error_dt.png")
    print("  heat_work_precision.png")
    print("  heat_pairwise_eoc.png")
    for name in results:
        print(f"  heat_{name.replace('(','_').replace(')','')}.csv")

