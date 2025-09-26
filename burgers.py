#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.seterr(over='raise', invalid='raise', divide='raise', under='warn')

##using a fourth order 16 stage scheme 

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
        if e[i] > 0 and e[i+1] > 0 and h[i] != h[i+1]:
            p[i] = np.log(e[i+1]/e[i]) / np.log(h[i+1]/h[i])
    return p

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

# ----------------- ESRK(16,4) tableau (your coefficients) ---------------------
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

# ------------------------------- main -----------------------------------------
if __name__ == "__main__":
    # Problem: smooth & active, short horizon to stay shock-free
    N, nu, T = 200, 0.02, 0.5
    burgers = Burgers1D(N=N, nu=nu)
    x = burgers.x
    u0 = (np.sin(12*x) + 0.5*np.sin(5*x)).astype(float)

    # Stability-informed Δt range
    R_real = 13.9170474646
    kmax = np.max(np.abs(burgers.k))
    dt_ceiling = R_real / (nu*(kmax**2) + 1e-30)
    dt_max = 0.7 * dt_ceiling           # safe margin inside stability
    total_halvings = 7                  # span ~2^7 across the 20 points (fast)
    M = 20                              # <-- exactly 20 points
    exps = np.linspace(0.0, total_halvings, M)
    dts = dt_max * (2.0 ** (-exps))
    dt_min = dts[-1]
    print(f"dt_max={dt_max:.6g}, dt_min={dt_min:.6g}, points={M}")

    # Build a single tight reference (aligned) — plenty accurate & fast
    dt_ref = dt_min / 8.0               # error ~ 8^4 = 4096× smaller
    tref = np.arange(0.0, T + 1e-15, dt_ref)
    # integrate with exact landing on each tref point
    def rk_to_final(h):
        return rk_explicit(lambda t,u: burgers.rhs(t,u), (0.0, T), u0, h, A_ESRK16, b_ESRK16)

    u_ref_T = rk_to_final(dt_ref)

    # Errors at final time for all 20 Δt
    errs = np.empty(M)
    for i, h in enumerate(dts):
        uT = rk_to_final(h)
        errs[i] = l2_grid_error(uT, u_ref_T)

    # Pairwise EOC and least-squares slope on interior points
    p_pair = pairwise_eoc(errs, dts)
    i0, i1 = 2, M-2                     # fit interior to avoid edge effects
    mask = (np.arange(M) >= i0) & (np.arange(M) <= i1) & (errs > 0)
    p_ls, C = np.polyfit(np.log(dts[mask]), np.log(errs[mask]), 1)
    print(f"Least-squares order on interior ({i0}..{i1}) ≈ {p_ls:.4f} (expect ~4)")

    # ----------------------------- Plots --------------------------------------
    plt.figure(figsize=(7,5))
    plt.loglog(dts, errs, 'o-', label='ESRK(16,4) error @ T')
    base = errs[i0]/(dts[i0]**4)
    plt.loglog(dts, base*(dts**4), 'k--', label='O(Δt^4) guide')
    plt.gca().invert_xaxis()
    plt.xlabel('Δt'); plt.ylabel('L2 error at T')
    plt.title('Burgers (smooth, N=200): ESRK(16,4) — 20-point 4th-order study')
    plt.grid(True, which='both', ls='--'); plt.legend(); plt.tight_layout()

    plt.figure(figsize=(7,4))
    plt.plot(dts[:-1], p_pair, '.-', label='Pairwise EOC')
    plt.axhline(4, color='k', linestyle='--', label='4')
    plt.gca().set_xscale('log'); plt.gca().invert_xaxis()
    plt.xlabel('Δt'); plt.ylabel('Estimated order p')
    plt.title('Local pairwise orders (adjacent Δt)')
    plt.grid(True, which='both', ls='--'); plt.legend(); plt.tight_layout()

    plt.show()
