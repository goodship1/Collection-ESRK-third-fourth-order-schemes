#!/usr/bin/env python3
import numpy as np
import time
import matplotlib.pyplot as plt

np.seterr(over='raise', invalid='raise', divide='raise', under='warn')

# -------------------- ESRK(16,4) coefficients --------------------
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

def kahan_sum(values):
    s=0.0; c=0.0
    for v in values:
        y = v - c
        t = s + y
        c = (t - s) - y
        s = t
    return s

# -------------------- PDE + exact solution (advection) --------------------
class Advection1D:
    def __init__(self, N=200, a=1.0, L=2*np.pi, spatial='spectral'):
        self.N = int(N); self.a = float(a); self.L = float(L)
        self.x = np.linspace(0.0, self.L, self.N, endpoint=False)
        self.dx = self.L / self.N
        self.spatial = spatial.lower()
        if self.spatial == 'spectral':
            # Fourier wavenumbers for domain [0,L], d in fftfreq is spacing in x/(2π)
            self.k = np.fft.fftfreq(self.N, d=self.dx/(2*np.pi))
            self.ik = 1j*self.k
        elif self.spatial == 'fd2':
            pass
        else:
            raise ValueError("spatial must be 'spectral' or 'fd2'")

    def u0(self, x, k0=3):
        # smooth IC with exact translation solution
        return np.sin(k0*x) + 0.2*np.sin(2*k0*x)

    def exact(self, t, k0=3):
        # u(x,t) = u0(x - a t) (periodic)
        xshift = (self.x - self.a * t) % self.L
        return self.u0(xshift, k0=k0)

    def rhs(self, t, u):
        if self.spatial == 'spectral':
            uhat = np.fft.fft(u)
            ux = np.fft.ifft(self.ik * uhat).real
        else:  # 2nd-order centered finite difference (periodic)
            ux = (np.roll(u, -1) - np.roll(u, 1)) / (2*self.dx)
        return -self.a * ux

# -------------------- explicit RK integrator --------------------
def rk_explicit(f, t_span, y0, h, A, b):
    t0, tf = t_span
    t = float(t0); y = y0.copy()
    s = len(b)
    c = np.array([kahan_sum(A[i][:i]) for i in range(s)], float)
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

# -------------------- norms & order helpers --------------------
def l2_error(u, v):
    r = u - v
    return float(np.sqrt(np.mean(np.abs(r)**2)))  # real & complex safe

def pairwise_eoc(errs, dts):
    e = np.asarray(errs); h = np.asarray(dts)
    p = np.full(len(e)-1, np.nan)
    for i in range(len(e)-1):
        if e[i]>0 and e[i+1]>0 and h[i]!=h[i+1]:
            p[i] = np.log(e[i+1]/e[i]) / np.log(h[i+1]/h[i])
    return p

# ------------------------------ main ------------------------------
if __name__ == "__main__":
    # Config
    N = 200
    a = 1.0
    L = 2*np.pi
    T = 1.0
    k0 = 3
    SPATIAL = 'spectral'   # 'spectral' or 'fd2'
    M = 20                 # number of Δt points
    beta_imag = 4.92145307073  # ESRK(16,4) imaginary-axis stability length

    adv = Advection1D(N=N, a=a, L=L, spatial=SPATIAL)
    u0 = adv.u0(adv.x, k0=k0)
    u_exact_T = adv.exact(T, k0=k0)

    # Stability-based dt range
    if SPATIAL == 'spectral':
        kmax = adv.N//2
        dt_max = 0.8 * beta_imag / (abs(a) * kmax)
    else:  # fd2
        dx = adv.dx
        dt_max = 0.8 * (beta_imag/2.0) * dx / abs(a)

    # Build 20 log-spaced Δt in a safe band
    span_halvings = 7.0
    exps = np.linspace(0.0, span_halvings, M)
    dts = dt_max * (2.0 ** (-exps))

    print(f"Testing ESRK(16,4) on 1D advection u_t + a u_x = 0, a={a}, L={L}, k0={k0}, spatial={SPATIAL}")
    errs = []
    times = []
    for h in dts:
        t0 = time.perf_counter()
        uT = rk_explicit(lambda t,u: adv.rhs(t,u), (0.0, T), u0, h, A_ESRK16, b_ESRK16)
        dtm = time.perf_counter() - t0
        err = l2_error(uT, u_exact_T)
        errs.append(err); times.append(dtm)
        print(f"Δt = {h:.6g} :  L2 error = {err:.6e},   time = {dtm:.3f}s")

    errs = np.array(errs)
    p_pair = pairwise_eoc(errs, dts)
    # LS-fit on interior (skip 2 endpoints)
    idx = np.arange(M)
    mask = (idx>=2) & (idx<=M-3) & (errs>0)
    p_ls, b_ls = np.polyfit(np.log(dts[mask]), np.log(errs[mask]), 1)
    C = np.exp(b_ls)

    print("\nPairwise observed orders:", np.array2string(p_pair, precision=4, separator=', '))
    print(f"Least-squares fitted order p ≈ {p_ls:.4f} (expect ~4)")
    print(f"Estimated constant C (e ≈ C Δt^p) ≈ {C:.3e}")

    # Plots
    plt.figure(figsize=(7,5))
    plt.loglog(dts, errs, 'o-', label='L2 error @ T (exact shift)')
    base = errs[5]/(dts[5]**4)
    plt.loglog(dts, base*(dts**4), 'k--', label='O(Δt^4) guide')
    plt.gca().invert_xaxis()
    plt.xlabel('Δt'); plt.ylabel('L2 error'); plt.title(f'Advection ({SPATIAL}, N={N}): ESRK(16,4) 4th-order')
    plt.grid(True, which='both', ls='--'); plt.legend(); plt.tight_layout()

    plt.figure(figsize=(7,4))
    plt.plot(dts[:-1], p_pair, '.-')
    plt.axhline(4, color='k', linestyle='--', label='4')
    plt.gca().set_xscale('log'); plt.gca().invert_xaxis()
    plt.xlabel('Δt'); plt.ylabel('Pairwise EOC p'); plt.title('Local pairwise orders')
    plt.grid(True, which='both', ls='--'); plt.legend(); plt.tight_layout()

    plt.show()
