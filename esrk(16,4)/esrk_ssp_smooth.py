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

    plt.show()#!/usr/bin/env python3
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

# ----------------- SSPRK(10,4) tableau (your coefficients) --------------------
# Constructed exactly as you provided, cast to float64.
A_SSP10 = np.zeros((10, 10), dtype=float)

# Stage 2: c2 = 1/6
A_SSP10[1,0] = 1.0/6.0

# Stage 3: c3 = 1/3
A_SSP10[2,0] = 1.0/6.0
A_SSP10[2,1] = 1.0/6.0

# Stage 4: c4 = 1/2
A_SSP10[3,0] = 1.0/6.0
A_SSP10[3,1] = 1.0/6.0
A_SSP10[3,2] = 1.0/6.0

# Stage 5: c5 = 2/3
A_SSP10[4,0] = 1.0/6.0
A_SSP10[4,1] = 1.0/6.0
A_SSP10[4,2] = 1.0/6.0
A_SSP10[4,3] = 1.0/6.0

# Stage 6: c6 = 1/3  (five entries of 1/15)
for j in range(5):
    A_SSP10[5, j] = 1.0/15.0

# Stage 7: c7 = 1/2  (five 1/15, then 1/6)
for j in range(5):
    A_SSP10[6, j] = 1.0/15.0
A_SSP10[6, 5] = 1.0/6.0

# Stage 8: c8 = 2/3  (five 1/15, then 1/6, 1/6)
for j in range(5):
    A_SSP10[7, j] = 1.0/15.0
A_SSP10[7, 5] = 1.0/6.0
A_SSP10[7, 6] = 1.0/6.0

# Stage 9: c9 = 5/6  (five 1/15, then 1/6, 1/6, 1/6)
for j in range(5):
    A_SSP10[8, j] = 1.0/15.0
A_SSP10[8, 5] = 1.0/6.0
A_SSP10[8, 6] = 1.0/6.0
A_SSP10[8, 7] = 1.0/6.0

# Stage 10: c10 = 1  (five 1/15, then four 1/6)
for j in range(5):
    A_SSP10[9, j] = 1.0/15.0
for j in range(5, 9):
    A_SSP10[9, j] = 1.0/6.0

b_SSP10 = np.full(10, 1.0/10.0, dtype=float)

# ------------------------------- main -----------------------------------------
if __name__ == "__main__":
    # Problem: smooth & active, short horizon to stay shock-free
    N, nu, T = 200, 0.02, 0.5
    burgers = Burgers1D(N=N, nu=nu)
    x = burgers.x
    u0 = (np.sin(12*x) + 0.5*np.sin(5*x)).astype(float)

    # Stability-informed Δt range (based on ESRK real-axis radius)
    R_real_ESRK = 13.9170474646
    kmax = np.max(np.abs(burgers.k))
    dt_ceiling = R_real_ESRK / (nu*(kmax**2) + 1e-30)
    dt_max = 0.7 * dt_ceiling
    total_halvings = 7
    M = 20
    exps = np.linspace(0.0, total_halvings, M)
    dts = dt_max * (2.0 ** (-exps))
    dt_min = dts[-1]
    print(f"dt_max={dt_max:.6g}, dt_min={dt_min:.6g}, points={M}")

    # Reference using ESRK(16,4) at dt_ref (aligned)
    dt_ref = dt_min / 8.0
    def rk_to_final(h, A, b):
        return rk_explicit(lambda t,u: burgers.rhs(t,u), (0.0, T), u0, h, A, b)

    t0 = perf_counter()
    u_ref_T = rk_to_final(dt_ref, A_ESRK16, b_ESRK16)
    ref_time = perf_counter() - t0
    print(f"[ref] ESRK(16,4) dt_ref={dt_ref:.3e}, time={ref_time:.4f}s")

    # Evaluate both methods on identical Δt grid; robust to blow-ups
    methods = [
        ("ESRK(16,4)", A_ESRK16, b_ESRK16),
        ("SSPRK(10,4)", A_SSP10, b_SSP10),
    ]

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
                # leave NaN; this dt likely outside stability for this method
                pass
        p_pair = pairwise_eoc(errs, dts)

        # least-squares order on interior finite points
        i0, i1 = 2, M-2
        idx = np.arange(M)
        mask = (idx >= i0) & (idx <= i1) & np.isfinite(errs) & (errs > 0)
        if mask.sum() >= 2:
            p_ls, C = np.polyfit(np.log(dts[mask]), np.log(errs[mask]), 1)
        else:
            p_ls = np.nan
        results[name] = dict(errs=errs, times=times, p_pair=p_pair, p_ls=p_ls)

        # Small console report
        s = len(b)
        med_err = np.nanmedian(errs)
        med_time = np.nanmedian(times)
        print(f"{name}: stages/step={s}, LS order≈{p_ls:.4f}, "
              f"median err={med_err:.3e}, median time/solve={med_time:.4f}s")

    # ----------------------------- Plots --------------------------------------
    # Error vs dt
    plt.figure(figsize=(7,5))
    for (label, _, _ ) in methods:
        errs = results[label]['errs']
        plt.loglog(dts[np.isfinite(errs)], errs[np.isfinite(errs)], 'o-', label=f'{label} error @ T')
    # 4th-order guide (based on ESRK interior normalization, if available)
    esrk_errs = results['ESRK(16,4)']['errs']
    if np.isfinite(esrk_errs[2]):
        base = esrk_errs[2]/(dts[2]**4)
        plt.loglog(dts, base*(dts**4), '--', label='O(Δt^4) guide')
    plt.gca().invert_xaxis()
    plt.xlabel('Δt'); plt.ylabel('L2 error at T')
    plt.title('Burgers (smooth, N=200): ESRK(16,4) vs SSPRK(10,4)')
    plt.grid(True, which='both', ls='--'); plt.legend(); plt.tight_layout()

    # Pairwise EOC
    plt.figure(figsize=(7,4))
    for (label, _, _ ) in methods:
        p_pair = results[label]['p_pair']
        finite_idx = np.isfinite(p_pair)
        plt.plot(dts[:-1][finite_idx], p_pair[finite_idx], '.-', label=f'{label} pairwise EOC')
    plt.axhline(4, linestyle='--', label='4')
    plt.gca().set_xscale('log'); plt.gca().invert_xaxis()
    plt.xlabel('Δt'); plt.ylabel('Estimated order p')
    plt.title('Local pairwise orders (adjacent Δt)')
    plt.grid(True, which='both', ls='--'); plt.legend(); plt.tight_layout()

    # Time vs dt
    plt.figure(figsize=(7,4))
    for (label, _, _ ) in methods:
        times = results[label]['times']
        plt.loglog(dts[np.isfinite(times)], times[np.isfinite(times)], 'o-', label=f'{label} wall time')
    plt.gca().invert_xaxis()
    plt.xlabel('Δt'); plt.ylabel('Wall time per solve [s]')
    plt.title('Cost vs step size')
    plt.grid(True, which='both', ls='--'); plt.legend(); plt.tight_layout()

    plt.show()
