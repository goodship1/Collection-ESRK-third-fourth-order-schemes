#ssp(10,4) vs esrk(16,4)


#!/usr/bin/env python3
import numpy as np
import time
import tracemalloc
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Configure NumPy to raise exceptions on floating-point errors
np.seterr(over='raise', invalid='raise', divide='raise', under='warn')

###############################################################################
# Kahan summation for improved floating-point precision
###############################################################################
def kahan_sum(values):
    """
    Perform Kahan summation on an array of values to reduce numerical error.
    """
    sum_ = np.float128(0.0)
    compensation = np.float128(0.0)
    for value in values:
        y = np.float128(value) - compensation
        temp = sum_ + y
        compensation = (temp - sum_) - y
        sum_ = temp
    return sum_

###############################################################################
# ODE Brusselator system
###############################################################################
def brusselator_ode(t, y, a_param, b_param):
    """
    The 2D Brusselator ODE system:
       du/dt = a - (b+1)*u + u^2 * v
       dv/dt = b*u - u^2 * v
    """
    u, v = y
    du_dt = a_param - (b_param + 1.0) * u + u**2 * v
    dv_dt = b_param * u - u**2 * v
    return np.array([du_dt, dv_dt], dtype=np.float128)

###############################################################################
# General explicit Runge-Kutta integrator (Butcher A,b)
###############################################################################
def runge_kutta_general(f, t_span, y0, h, a_values, b_values, args=()):
    """
    General explicit Runge-Kutta integration with user-provided Butcher coefficients.
    """
    t0, tf = t_span
    t = np.float128(t0)
    y = y0.astype(np.float128)
    t_values = [t0]
    y_values = [y.copy()]

    s = len(b_values)  # number of stages
    num_steps = int(np.ceil((tf - t0) / h))

    # Precompute c-values from A (row sums up to i-1)
    c_values = np.array([kahan_sum(a_values[i][:i]) for i in range(s)],
                        dtype=np.float128)

    for _ in range(num_steps):
        # Adjust last step to end exactly at tf
        if t + h > tf + np.float128(1e-12):
            h = tf - t
        if h <= 0:
            break

        k = [np.zeros_like(y, dtype=np.float128) for _ in range(s)]
        for i in range(s):
            # stage state
            y_stage = y.copy()
            for j in range(i):
                if a_values[i][j] != 0:
                    y_stage += h * a_values[i][j] * k[j]
            t_stage = t + c_values[i] * h
            try:
                k[i] = f(t_stage, y_stage, *args)
            except FloatingPointError as e:
                print(f"Floating point error at time {t_stage}: {e}")
                return (np.array(t_values, dtype=np.float128),
                        np.array(y_values, dtype=np.float128))

        # Combine stages
        for i in range(s):
            if b_values[i] != 0:
                y += h * b_values[i] * k[i]

        t += h
        if t > t_values[-1] + np.float128(1e-12):
            t_values.append(t)
            y_values.append(y.copy())

    return (np.array(t_values, dtype=np.float128),
            np.array(y_values, dtype=np.float128))

###############################################################################
# L2 norm error calculation
###############################################################################
def l2_norm_error(y_numerical, y_reference):
    """
    L2 (RMS over time) error between y_numerical and y_reference with shape
    (num_times, dim).
    """
    error = y_numerical - y_reference
    l2_per_time = np.sqrt(np.sum(error**2, axis=1))
    return np.sqrt(np.mean(l2_per_time**2, dtype=np.float128))

###############################################################################
# Experimental order of convergence
###############################################################################
def calculate_order_of_convergence(errors, hs):
    orders = []
    for i in range(1, len(errors)):
        if not np.isfinite(errors[i]) or not np.isfinite(errors[i-1]) or errors[i-1] == 0 or hs[i-1] == hs[i]:
            orders.append(np.nan)
        else:
            order = np.log(errors[i] / errors[i-1]) / np.log(hs[i] / hs[i-1])
            orders.append(order)
    return orders

###############################################################################
# Interpolate solution to a reference grid
###############################################################################
def interpolate_solution(t_values, y_values, t_values_ref):
    """
    Interpolate y_values(t_values) to t_values_ref (component-wise cubic).
    """
    dim = y_values.shape[1]
    y_interpolated = []
    for i in range(dim):
        f_interp = interp1d(t_values, y_values[:, i], kind='cubic', fill_value="extrapolate")
        y_interpolated.append(f_interp(t_values_ref))
    return np.array(y_interpolated).T  # (len(t_ref), dim)

###############################################################################
# ESRK-16 (2S–8) tableau (4th order) — YOUR coefficients
###############################################################################
A_ESRK16 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.297950632696351, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0892840764249344, 0.522026933033341, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.144349746352280, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, -0.000371956295732390, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, -0.124117473662160, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.192800131150961, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, -0.00721201688860849, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.385496874023061, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.248192855959921, 0, 0, 0, 0, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, -0.295854950063981, -4.25371891111175e-5, 0, 0, 0, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, -0.295854950063981, 0.163017169512979, 0.138371044215410, 0, 0, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, -0.295854950063981, 0.163017169512979, -0.0819824325549522, 0.403108090476214, 0, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, -0.295854950063981, 0.163017169512979, -0.0819824325549522, 0.546008221888163, 0.125164780662438, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, -0.295854950063981, 0.163017169512979, -0.0819824325549522, 0.546008221888163, -0.0422844329611440, -0.00579862710501764, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, -0.295854950063981, 0.163017169512979, -0.0819824325549522, 0.546008221888163, -0.0422844329611440, 0.467431197768081, 0.502036131647685, 0]
], dtype=np.float128)

b_ESRK16 = np.array([
    0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344,
    0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344,
    -0.29585495006398077, 0.1630171695129791, -0.08198243255495223, 0.5460082218881631,
    -0.04228443296114401, 0.46743119776808084, -0.45495020324595, -0.01565718174267131
], dtype=np.float128)

###############################################################################
# SSPRK(10,4) Butcher tableau (explicit; b_i = 1/10)
# Matches the LaTeX tableau you provided (10 stages; first row is zeros).
###############################################################################
A_SSP104 = np.zeros((10, 10), dtype=np.float128)

# Stage 2: c2 = 1/6
A_SSP104[1,0] = np.float128(1)/np.float128(6)

# Stage 3: c3 = 1/3
A_SSP104[2,0] = np.float128(1)/np.float128(6)
A_SSP104[2,1] = np.float128(1)/np.float128(6)

# Stage 4: c4 = 1/2
A_SSP104[3,0] = np.float128(1)/np.float128(6)
A_SSP104[3,1] = np.float128(1)/np.float128(6)
A_SSP104[3,2] = np.float128(1)/np.float128(6)

# Stage 5: c5 = 2/3
A_SSP104[4,0] = np.float128(1)/np.float128(6)
A_SSP104[4,1] = np.float128(1)/np.float128(6)
A_SSP104[4,2] = np.float128(1)/np.float128(6)
A_SSP104[4,3] = np.float128(1)/np.float128(6)

# Stage 6: c6 = 1/3, first five entries = 1/15
for j in range(5):
    A_SSP104[5,j] = np.float128(1)/np.float128(15)

# Stage 7: c7 = 1/2, first five 1/15, then 1/6
for j in range(5):
    A_SSP104[6,j] = np.float128(1)/np.float128(15)
A_SSP104[6,5] = np.float128(1)/np.float128(6)

# Stage 8: c8 = 2/3, first five 1/15, then 1/6, 1/6
for j in range(5):
    A_SSP104[7,j] = np.float128(1)/np.float128(15)
A_SSP104[7,5] = np.float128(1)/np.float128(6)
A_SSP104[7,6] = np.float128(1)/np.float128(6)

# Stage 9: c9 = 5/6, first five 1/15, then 1/6, 1/6, 1/6
for j in range(5):
    A_SSP104[8,j] = np.float128(1)/np.float128(15)
A_SSP104[8,5] = np.float128(1)/np.float128(6)
A_SSP104[8,6] = np.float128(1)/np.float128(6)
A_SSP104[8,7] = np.float128(1)/np.float128(6)

# Stage 10: c10 = 1, first five 1/15, then four 1/6
for j in range(5):
    A_SSP104[9,j] = np.float128(1)/np.float128(15)
for j in range(5, 9):
    A_SSP104[9,j] = np.float128(1)/np.float128(6)

b_SSP104 = np.array([np.float128(1)/np.float128(10)]*10, dtype=np.float128)

###############################################################################
# Main script
###############################################################################
if __name__ == "__main__":
    # Parameters for the Brusselator ODE
    a_param = 1.0
    b_param = 3.0  # bounded limit cycle; fine for ODE experiments

    # Time span
    t_span = (0.0, 10.0)

    # Initial conditions (slightly perturbed around the steady state)
    np.random.seed(0)
    u0 = a_param + 0.1 * np.random.rand()
    v0 = (b_param / a_param) + 0.1 * np.random.rand()
    y0 = np.array([u0, v0], dtype=np.float128)

    # Step sizes to test
    hs = np.logspace(-3, -1, 100)  # 1e-3 ... 1e-1

    ###########################################################################
    # Reference solution (tiny step, SSPRK(10,4) to avoid bias)
    ###########################################################################
    print("Generating reference (SSPRK(10,4)) with h = 1e-5 ...")
    h_ref = np.float128(1e-5)
    t_values_ref, y_values_ref = runge_kutta_general(
        brusselator_ode, t_span, y0, h_ref, A_SSP104, b_SSP104,
        args=(a_param, b_param)
    )
    print("Reference ready.")

    ###########################################################################
    # Convergence: ESRK-16 (your scheme)
    ###########################################################################
    errors_esrk, steps_esrk, mem_esrk, time_esrk = [], [], [], []
    for h in hs:
        print(f"\nESRK-16 h = {h} ...")
        tracemalloc.start(); t0 = time.time()
        try:
            t_values, y_values = runge_kutta_general(
                brusselator_ode, t_span, y0, np.float128(h), A_ESRK16, b_ESRK16,
                args=(a_param, b_param)
            )
        except FloatingPointError as e:
            print(f"ESRK fail at h={h}: {e}")
            errors_esrk.append(np.nan); steps_esrk.append(np.nan)
            time_esrk.append(np.nan); mem_esrk.append(np.nan)
            tracemalloc.stop(); continue

        comp_time = time.time() - t0
        current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        y_interp = interpolate_solution(t_values, y_values, t_values_ref)
        err = l2_norm_error(y_interp, y_values_ref)

        errors_esrk.append(err)
        steps_esrk.append(len(t_values)-1)
        time_esrk.append(comp_time)
        mem_esrk.append(peak)
        print(f"ESRK h={h}: err={err:.6e}, steps={steps_esrk[-1]}, time={comp_time:.6f}s, mem={peak}B")

    orders_esrk = calculate_order_of_convergence(errors_esrk, hs)

    ###########################################################################
    # Convergence: SSPRK(10,4)
    ###########################################################################
    errors_ssp, steps_ssp, mem_ssp, time_ssp = [], [], [], []
    for h in hs:
        print(f"\nSSPRK(10,4) h = {h} ...")
        tracemalloc.start(); t0 = time.time()
        try:
            t_values, y_values = runge_kutta_general(
                brusselator_ode, t_span, y0, np.float128(h), A_SSP104, b_SSP104,
                args=(a_param, b_param)
            )
        except FloatingPointError as e:
            print(f"SSP fail at h={h}: {e}")
            errors_ssp.append(np.nan); steps_ssp.append(np.nan)
            time_ssp.append(np.nan); mem_ssp.append(np.nan)
            tracemalloc.stop(); continue

        comp_time = time.time() - t0
        current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        y_interp = interpolate_solution(t_values, y_values, t_values_ref)
        err = l2_norm_error(y_interp, y_values_ref)

        errors_ssp.append(err)
        steps_ssp.append(len(t_values)-1)
        time_ssp.append(comp_time)
        mem_ssp.append(peak)
        print(f"SSP h={h}: err={err:.6e}, steps={steps_ssp[-1]}, time={comp_time:.6f}s, mem={peak}B")

    orders_ssp = calculate_order_of_convergence(errors_ssp, hs)

    ###########################################################################
    # Summaries
    ###########################################################################
    print("\n=== ESRK-16 summary ===")
    print("Errors:", errors_esrk)
    print("Pairwise orders:", orders_esrk)
    print("Steps:", steps_esrk)
    print("Times (s):", time_esrk)
    print("Peak mem (B):", mem_esrk)

    print("\n=== SSPRK(10,4) summary ===")
    print("Errors:", errors_ssp)
    print("Pairwise orders:", orders_ssp)
    print("Steps:", steps_ssp)
    print("Times (s):", time_ssp)
    print("Peak mem (B):", mem_ssp)

    ###########################################################################
    # Plots
    ###########################################################################
    plt.figure(figsize=(8, 6))
    plt.loglog(hs, errors_esrk, 'o-', label='ESRK-16 (4th)')
    plt.loglog(hs, errors_ssp,  's-', label='SSPRK(10,4)')

    # 4th-order reference slope
    # choose the first finite error from either method to set the slope line
    base_idx = None
    for arr in (errors_esrk, errors_ssp):
        idx = np.where(np.isfinite(arr) & (np.array(arr) > 0))[0]
        if idx.size:
            base_idx = idx[0]
            base_errors = arr
            break
    if base_idx is not None:
        cst = base_errors[base_idx] / (hs[base_idx]**4)
        plt.loglog(hs, [cst * (h**4) for h in hs], 'k--', label='O(h^4)')

    plt.xlabel('Time step Δt')
    plt.ylabel('L2 error (vs reference)')
    plt.title('Brusselator: ESRK-16 vs SSPRK(10,4) — Convergence')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.show()
