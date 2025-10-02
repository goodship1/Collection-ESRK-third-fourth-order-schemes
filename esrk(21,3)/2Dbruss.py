#!/usr/bin/env python3

import numpy as np
import time
import tracemalloc
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


np.seterr(over='raise', invalid='raise', divide='raise', under='warn')

###############################################################################
# 1) Kahan Summation (helper for c_i)
###############################################################################
def kahan_sum(values):
    """
    Kahan summation to improve numerical stability in summing a list of floats.
    Returns the sum as a float128.
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
# 2) The 2D Brusselator PDE ODE Right-Hand Side with 21 stage esrk
###############################################################################
def brusselator_1d(t, y, N, dx, D_u, D_v, a_param, b_param):
    """
    2D Brusselator PDE with Neumann boundary conditions, discretized in space.
    
    PDE system:
      u_t = a - (b+1)*u + u^2*v + D_u*u_xx
      v_t = b*u - u^2*v + D_v*v_xx

    Discretization in space (finite differences):
      y = [u_0, u_1, ..., u_{N-1}, v_0, v_1, ..., v_{N-1}],
      N grid points from x=0..L.

    Neumann BC: derivative at boundaries = 0 => we approximate by
      u_xx(0)   = (u(2) - 2u(1) + u(0))   / dx^2
      u_xx(N-1) = (u(N-1) - 2u(N-2) + u(N-3)) / dx^2
    (similarly for v).

    Returns dy/dt as a length 2*N vector.
    """
    # Unpack
    u = y[:N]
    v = y[N:]

    # Prepare arrays
    du_dt = np.zeros(N, dtype=np.float128)
    dv_dt = np.zeros(N, dtype=np.float128)
    u_xx  = np.zeros(N, dtype=np.float128)
    v_xx  = np.zeros(N, dtype=np.float128)

    # Interior points (standard 2nd-order finite difference)
    u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx**2)
    v_xx[1:-1] = (v[2:] - 2*v[1:-1] + v[:-2]) / (dx**2)

    # Neumann boundary conditions for each end
    # For the left endpoint (i=0)
    u_xx[0]  = (u[2]   - 2*u[1]   + u[0])   / (dx**2)
    v_xx[0]  = (v[2]   - 2*v[1]   + v[0])   / (dx**2)
    # For the right endpoint (i=N-1)
    u_xx[-1] = (u[-1]  - 2*u[-2]  + u[-3])  / (dx**2)
    v_xx[-1] = (v[-1]  - 2*v[-2]  + v[-3])  / (dx**2)

    # Reaction terms
    reaction_u = a_param - (b_param+1)*u + (u**2)*v
    reaction_v = b_param*u - (u**2)*v

    # Combine reaction + diffusion
    du_dt = D_u*u_xx + reaction_u
    dv_dt = D_v*v_xx + reaction_v

    return np.concatenate([du_dt, dv_dt])

###############################################################################
# 3) The 21-Stage ESRK Butcher Tableau (3rd-order)
###############################################################################
a_21 = np.array([
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
], dtype=np.float128)

b_21 = np.array([
    0.025918693385897126,  0.025918693385897126,  0.025918693385897126,
    0.025918693385897126,  0.025918693385897126,  0.025918693385897126,
    0.025918693385897126,  0.025918693385897126,  0.010679463093674657,
   -0.11533544419119937,   0.15735456931774092,   0.09969539160404886,
    0.12462657068046491,   0.09992542978103731,  -0.1577213015623934,
    0.17183858110421352,  -0.1592822538823839,    0.15369230571151177,
    0.13580248201617606,   0.12674481627127573,   0.14462984296865586
], dtype=np.float128)

###############################################################################
# 4) Generic ESRK Solver With 21 Stages
###############################################################################
def runge_kutta_21stage(f, t_span, y0, h, a_values, b_values, args=()):
    """
    A 21-stage explicit Runge-Kutta (ESRK) integrator. 
    We assume a_values is 21x21, b_values is length=21.
    
    f(t, y, *args) -> array of length = len(y).
    We integrate from t_span[0] to t_span[1] with fixed step h (adjust last step).
    """
    t0, tf = t_span
    t = np.float128(t0)
    y = y0.copy()
    t_values = [t0]
    y_values = [y.copy()]

    s = len(b_values)  # should be 21
    num_steps = int(np.ceil((tf - t0) / h))
    
    # c_i = sum_j(a[i][j]) for j in [0..i-1], 
    # (We can use kahan_sum for potential numeric safety, but typically it's small.)
    c_values = np.array([kahan_sum(a_values[i, :i]) for i in range(s)], dtype=np.float128)
    
    # We'll store each stage k_i in a list.  Alternatively we can do low-storage,
    # but let's keep it simpler so it's easy to read.
    k_stages = [np.zeros_like(y, dtype=np.float128) for _ in range(s)]

    for step in range(num_steps):
        # If we overshoot, adjust h
        if t + h > tf + np.float128(1e-15):
            h = tf - t
        if h <= 0:
            break

        # Compute the k_i in sequence
        for i in range(s):
            # Build y_stage = y + h * sum_{j=0..i-1} a_{i,j} * k_stages[j]
            y_stage = y.copy()
            for j in range(i):
                alpha_ij = a_values[i,j]
                if alpha_ij != 0:
                    y_stage += (h * alpha_ij) * k_stages[j]
            
            t_stage = t + c_values[i]*h
            # Evaluate slope
            k_stages[i][:] = f(t_stage, y_stage, *args)
        
        # Once we have all k_i, do the final combination
        for i in range(s):
            beta_i = b_values[i]
            if beta_i != 0:
                y += (h * beta_i) * k_stages[i]
        
        t += h
        # Save 
        t_values.append(t)
        y_values.append(y.copy())

        if t >= tf - 1e-15:
            # we are effectively done
            break

    return np.array(t_values, dtype=np.float128), np.array(y_values, dtype=np.float128)

###############################################################################
# 5) Utility: L2 norm of solution difference
###############################################################################
def l2_norm_error(y_numerical, y_reference):
    """
    Given two arrays of shape (#timepoints, #state_variables),
    compute a single L2 measure by comparing final solutions
    (or do something more advanced if you want).
    
    As a demonstration, let's just do:
       L2 = sqrt( average( || y_num - y_ref ||^2 ) ).
    Here we assume y_num and y_ref have the same shape,
    or at least we can align them in time or something.
    """
    error = y_numerical - y_reference
    l2_norms = np.linalg.norm(error, axis=1)
    # We'll just return the average of these or something similar.
    return np.sqrt(np.mean(l2_norms**2, dtype=np.float128))

###############################################################################
# 6) Interpolate Solutions (for comparing to reference)
###############################################################################
def interpolate_solution(t_values, y_values, t_values_ref):
    """
    Interpolate y_values(t_values) onto t_values_ref using cubic interpolation
    for each component.
    """
    y_interpolated = []
    for i in range(y_values.shape[1]):
        f_interp = interp1d(t_values, y_values[:, i], kind='cubic', fill_value="extrapolate")
        y_interpolated.append(f_interp(t_values_ref))
    return np.array(y_interpolated).T

###############################################################################
# 7) Compute Orders (between consecutive steps)
###############################################################################
def calculate_order_of_convergence(errors, hs):
    """
    For i=1..len(errors)-1,
      order_i = log(errors[i]/errors[i-1]) / log(hs[i]/hs[i-1])
    """
    orders = []
    for i in range(1, len(errors)):
        if errors[i-1] == 0 or hs[i-1] == hs[i] or np.isnan(errors[i]) or np.isnan(errors[i-1]):
            orders.append(np.nan)
        else:
            order = np.log(errors[i] / errors[i-1]) / np.log(hs[i] / hs[i-1])
            orders.append(order)
    return orders

###############################################################################
# 8) Main Demo
###############################################################################
if __name__ == "__main__":
    # PDE and discretization params
    t_span = (0.0, 10.0)
    L = 10.0
    N = 100
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]

    # PDE coefficients
    D_u = 0.1
    D_v = 0.05
    a_param = 1.0
    b_param = 3.0

    # Initial condition: random perturbation around (u0=a_param, v0=b_param/a_param)
    np.random.seed(0)
    u0_vals = a_param + 0.1*np.random.rand(N)
    v0_vals = b_param/a_param + 0.1*np.random.rand(N)
    y0 = np.concatenate([u0_vals, v0_vals]).astype(np.float128)

    # Stability estimate for an explicit scheme (rough)
    D_max = max(D_u, D_v)
    dt_max = dx**2 / (2 * D_max)
    print(f"Estimated stable dt_max ~ {dt_max}")

    # List of step sizes to test
    hs = np.linspace(0.01, 0.001, 120) # smaller steps down to 0.001

    # We'll do a reference solution at a small step size (say 1e-4):
    h_ref = 1e-4
    print(f"\nGenerating reference solution with h_ref={h_ref} ...")
    t_values_ref, y_values_ref = runge_kutta_21stage(
        brusselator_1d, t_span, y0, h_ref, a_21, b_21,
        args=(N, dx, D_u, D_v, a_param, b_param)
    )
    print("Reference solution generated.\n")

    # Convergence study
    errors = []
    iteration_counts = []
    memory_usage = []
    computation_times = []

    for h in hs:
        print(f"Running 21-stage ESRK with dt={h} ...")
        tracemalloc.start()
        start_time = time.time()

        # Integrate
        try:
            t_values, y_values = runge_kutta_21stage(
                brusselator_1d, t_span, y0, h, a_21, b_21,
                args=(N, dx, D_u, D_v, a_param, b_param)
            )
        except FloatingPointError as e:
            print(f"Floating point error for dt={h}: {e}")
            errors.append(np.nan)
            iteration_counts.append(np.nan)
            computation_times.append(np.nan)
            memory_usage.append(np.nan)
            continue

        comp_time = time.time() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        computation_times.append(comp_time)
        memory_usage.append(peak_mem)

        iteration_counts.append(len(t_values)-1)

        # Interpolate to reference time grid
        y_values_interpolated = interpolate_solution(t_values, y_values, t_values_ref)
        # L2 norm of difference
        err = l2_norm_error(y_values_interpolated, y_values_ref)
        errors.append(err)

        print(f"dt={h}: error={err}, steps={iteration_counts[-1]}, time={comp_time:.4f}s, peak mem={peak_mem}")

    # Compute orders
    orders = calculate_order_of_convergence(errors, hs)

    # Print summary some nice metrics for the user to impect error order condtions 
    print("\n--- 2D Brusselator PDE with 21-Stage ESRK (3rd-order) ---")
    print("Step sizes (hs)   =", hs)
    print("Errors           =", errors)
    print("Orders of conv.  =", orders)
    print("Steps per h      =", iteration_counts)
    print("Times (seconds)  =", computation_times)
    print("Memory usage     =", memory_usage)

    # Plot
    plt.figure(figsize=(8,6))
    plt.loglog(hs, errors, 'o-', label='Numerical error')
    # Plot a slope=3 reference line
    if len(hs) > 1 and not np.isnan(errors[0]):
        c_ref = errors[0] / (hs[0]**3)
        ref_line = [c_ref*(h_**3) for h_ in hs]
        plt.loglog(hs, ref_line, 'k--', label='Slope 3 reference')

    plt.xlabel("Time step size (Δt)")
    plt.ylabel("Error norm")
    plt.title("1D Brusselator PDE: 21-Stage ESRK (3rd order) Convergence")
    plt.legend()
    plt.grid(True)
    plt.show()
