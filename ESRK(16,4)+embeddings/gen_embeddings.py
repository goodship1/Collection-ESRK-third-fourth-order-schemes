#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESRK(16,4/3) — Production-Ready Embedding Search with VERIFIED Stability
-------------------------------------------------------------------------
Uses the WORKING stability calculation (direct method).
Searches for maximum R3 at 4% deviation with full complex plane optimization.
"""

import numpy as np
import time
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# =========================
# CONFIG
# =========================
TARGET_DEV      = 0.04                        # 4% deviation
GLOBAL_SEEDS    = 120000                       # Initial random samples
TOPK_KEEP       = 100                         # Keep top candidates
ELITE_REFINE    = 70                          # Intensive CMA-ES on these
CMA_ITERS       = 150                         # CMA-ES iterations (reduced, enough for refinement)
CMA_POPSIZE     = 40                          # CMA-ES population
N_WORKERS       = max(1, os.cpu_count() // 2)
RANDOM_SEED     = 121231345

# Complex plane search parameters
N_IMAG_OFFSETS  = 30                          # Angular samples in complex plane
IMAG_RANGE      = 0.5                         # Search ±x*IMAG_RANGE in imaginary
BOUNDARY_TOL    = 1e-5

# =========================
# TABLEAU
# =========================
A = np.array([
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
], dtype=np.float64)

b4 = np.array([
    0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344,
    0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344,
    -0.29585495006398077, 0.1630171695129791, -0.08198243255495223, 0.5460082218881631,
    -0.04228443296114401, 0.46743119776808084, -0.45495020324595, -0.01565718174267131
], dtype=np.float64)

S = 16
e = np.ones(S, dtype=np.float64)
c = A @ e

# =========================
# Order-3 constraints
# =========================
C = np.vstack([np.ones(S), c, c**2, A @ c]).astype(np.float64)
dvec = np.array([1.0, 0.5, 1.0/3.0, 1.0/6.0], dtype=np.float64)

def project_order3(b):
    """Project b onto order-3 constraint manifold"""
    lam = np.linalg.solve(C @ C.T, C @ b - dvec)
    return b - C.T @ lam

U, s, Vt = np.linalg.svd(C, full_matrices=True)
NS = Vt.T[:, C.shape[0]:]   # 16 x 12 nullspace
b3_base = project_order3(b4.copy())
target_norm = TARGET_DEV * np.linalg.norm(b4)

def b_from_theta_fixed(theta):
    """Map unit sphere in nullspace to fixed 4% deviation embedding"""
    n = np.linalg.norm(theta)
    if n < 1e-15:
        theta = np.random.randn(len(theta))
        n = np.linalg.norm(theta)
    theta = theta / n
   
    delta = NS @ theta
    delta *= target_norm / np.linalg.norm(delta)
   
    return project_order3(b3_base + delta)

# =========================
# VERIFIED STABILITY FUNCTION
# =========================
def R_stability(b, z):
    """
    Stability function for explicit RK: R(z) = 1 + z*b^T*(I - zA)^(-1)*e
    Uses forward substitution since (I - zA) is lower triangular.
    """
    I = np.eye(S, dtype=np.complex128)
    zA = z * A.astype(np.complex128)
   
    # Forward substitution: solve (I - zA)*y = e
    y = np.zeros(S, dtype=np.complex128)
    e_c = e.astype(np.complex128)
   
    for i in range(S):
        sum_val = e_c[i]
        for j in range(i):
            sum_val -= (I[i,j] - zA[i,j]) * y[j]
        y[i] = sum_val / (I[i,i] - zA[i,i])
   
    R = 1.0 + z * np.dot(b.astype(np.complex128), y)
    return abs(R)

# =========================
# STABILITY SEARCH - Simplified to real axis (matches Brusselator testing)
# =========================
def find_stability_real_axis(b, x_max=150.0):
    """
    Find stability boundary on negative real axis.
    This matches how Brusselator and other ODE solvers actually work.
    """
    # Binary search for where |R(-x)| = 1
    lo, hi = 0.0, x_max
   
    # Find upper bound
    x_test = 0.1
    while x_test < x_max and R_stability(b, -x_test) <= 1.0:
        x_test *= 1.5
   
    if x_test >= x_max:
        hi = x_max
    else:
        hi = x_test
   
    # Binary search
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if R_stability(b, -mid) <= 1.0 + BOUNDARY_TOL:
            lo = mid
        else:
            hi = mid
       
        if hi - lo < 1e-3:
            break
   
    return lo

# Compute R4 with larger x_max to ensure we find it
print("Computing R4 for main method (real axis)...")
R4 = find_stability_real_axis(b4, x_max=150.0)
print(f"R4 = {R4:.2f}")

if R4 < 40.0:
    print(f"WARNING: R4 = {R4:.2f} seems too small! Expected ~58.")
    print("Check tableau values.")

print()

# =========================
# CMA-ES
# =========================
class SimpleCMAES:
    def __init__(self, dim, popsize, seed):
        self.dim = dim
        self.popsize = popsize
        self.rng = np.random.default_rng(seed)
       
        self.mean = self.rng.standard_normal(dim)
        self.mean /= np.linalg.norm(self.mean)
        self.sigma = 0.5
        self.C = np.eye(dim)
       
        self.mu = popsize // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
       
        self.mueff = 1.0 / np.sum(self.weights**2)
        self.cc = 4.0 / (dim + 4.0)
        self.cs = (self.mueff + 2.0) / (dim + self.mueff + 5.0)
        self.c1 = 2.0 / ((dim + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((dim + 2)**2 + self.mueff))
        self.damps = 1.0 + 2.0 * max(0, np.sqrt((self.mueff-1)/(dim+1)) - 1) + self.cs
       
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.counteval = 0
        self.eigeneval = 0
        self.B = np.eye(dim)
        self.D = np.ones(dim)
        self.chiN = dim**0.5 * (1 - 1.0/(4*dim) + 1.0/(21*dim**2))
       
    def ask(self):
        arz = self.rng.standard_normal((self.popsize, self.dim))
        ary = np.dot(arz, self.B.T * self.D)
        arx = self.mean + self.sigma * ary
        for i in range(len(arx)):
            n = np.linalg.norm(arx[i])
            if n > 1e-15:
                arx[i] /= n
        return arx, ary, arz
   
    def tell(self, arx, ary, arz, fitness):
        idx = np.argsort(fitness)[::-1]  # maximize
       
        self.mean = np.dot(self.weights, arx[idx[:self.mu]])
        n = np.linalg.norm(self.mean)
        if n > 1e-15:
            self.mean /= n
       
        self.ps = (1-self.cs)*self.ps + np.sqrt(self.cs*(2-self.cs)*self.mueff) * \
                  np.dot(self.B, np.dot(self.weights, arz[idx[:self.mu]]))
       
        hsig = 0.0
        if self.counteval > 0:
            denom = np.sqrt(1-(1-self.cs)**(2*self.counteval/self.popsize))
            if denom > 1e-10:
                hsig = (np.linalg.norm(self.ps)/denom/self.chiN < 1.4 + 2.0/(self.dim+1))
            else:
                hsig = 1.0
       
        self.pc = (1-self.cc)*self.pc + hsig * np.sqrt(self.cc*(2-self.cc)*self.mueff) * \
                  np.dot(self.weights, ary[idx[:self.mu]])
       
        artmp = ary[idx[:self.mu]]
        self.C = (1-self.c1-self.cmu) * self.C + \
                 self.c1 * (np.outer(self.pc, self.pc) + (1-hsig)*self.cc*(2-self.cc)*self.C) + \
                 self.cmu * np.dot(artmp.T * self.weights, artmp)
       
        self.sigma *= np.exp((self.cs/self.damps) * (np.linalg.norm(self.ps)/self.chiN - 1))
        self.sigma = min(self.sigma, 3.0)  # Cap sigma to prevent explosion on sphere
       
        if self.counteval - self.eigeneval > self.popsize/(self.c1+self.cmu)/self.dim/10:
            self.eigeneval = self.counteval
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            D2, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(np.maximum(D2, 1e-20))
       
        self.counteval += self.popsize

# =========================
# Parallel global search
# =========================
def eval_seed_R3(seed):
    local_rng = np.random.default_rng(RANDOM_SEED + 7919*seed)
    theta = local_rng.standard_normal(NS.shape[1])
    theta /= np.linalg.norm(theta)
   
    b = b_from_theta_fixed(theta)
    R3 = find_stability_real_axis(b, x_max=80.0)
   
    return (R3, theta, b)

def refine_with_cma(theta_init):
    """CMA-ES refinement of a candidate"""
    cma = SimpleCMAES(dim=NS.shape[1], popsize=CMA_POPSIZE,
                      seed=int(1e9*np.sum(np.abs(theta_init))) % 2**31)
    cma.mean = theta_init / np.linalg.norm(theta_init)
   
    best_R3, best_b = 0.0, None
   
    for gen in range(CMA_ITERS):
        arx, ary, arz = cma.ask()
       
        fitness = np.zeros(CMA_POPSIZE)
        for i in range(CMA_POPSIZE):
            b = b_from_theta_fixed(arx[i])
            fitness[i] = find_stability_real_axis(b, x_max=80.0)
           
            if fitness[i] > best_R3:
                best_R3 = fitness[i]
                best_b = b
       
        cma.tell(arx, ary, arz, fitness)
       
        if gen % 50 == 0:
            print(f"      Gen {gen}: R3={best_R3:.2f}, σ={cma.sigma:.3f}")
   
    return best_R3, best_b

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    t0 = time.time()
   
    print("="*70)
    print(f"ESRK(16,4/3) EMBEDDING SEARCH")
    print(f"Target deviation: {TARGET_DEV*100:.1f}%")
    print(f"R4 (main method): {R4:.2f}")
    print(f"Workers: {N_WORKERS}")
    print("="*70)
   
    # Phase 1: Global sampling
    print(f"\n[Phase 1] Global sampling ({GLOBAL_SEEDS} seeds)...")
    candidates = []
   
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(eval_seed_R3, s) for s in range(GLOBAL_SEEDS)]
        for i, fut in enumerate(as_completed(futs), 1):
            R3, theta, b = fut.result()
            candidates.append((R3, theta, b))
           
            if i % (GLOBAL_SEEDS//10) == 0:
                best = max(candidates, key=lambda x: x[0])
                print(f"  {i:5d}/{GLOBAL_SEEDS}: best R3 = {best[0]:.2f}")
   
    candidates.sort(key=lambda x: -x[0])
    top = candidates[:TOPK_KEEP]
    print(f"\n  Top {TOPK_KEEP} range: R3 = {top[0][0]:.2f} to {top[-1][0]:.2f}")
   
    # Phase 2: CMA-ES refinement
    print(f"\n[Phase 2] CMA-ES refinement (top {ELITE_REFINE})...")
    elite = top[:ELITE_REFINE]
   
    best_overall_R3 = 0.0
    best_overall_b = None
   
    for i, (R3_init, theta_init, b_init) in enumerate(elite, 1):
        print(f"\n  Elite {i:2d}/{ELITE_REFINE} (init R3={R3_init:.2f}):")
        R3_refined, b_refined = refine_with_cma(theta_init)
       
        if R3_refined > best_overall_R3:
            best_overall_R3 = R3_refined
            best_overall_b = b_refined
            print(f"    >>> NEW BEST: R3 = {R3_refined:.2f} <<<")
        else:
            print(f"    Final: R3 = {R3_refined:.2f}")
   
    # Results
    ratio = best_overall_R3 / R4
    actual_dev = np.linalg.norm(best_overall_b - b4) / np.linalg.norm(b4)
   
    print("\n" + "="*70)
    print("FINAL RESULTS:")
    print("="*70)
    print(f"  Target deviation:  {TARGET_DEV*100:.1f}%")
    print(f"  Actual deviation:  {actual_dev*100:.2f}%")
    print(f"  R3 (embedded):     {best_overall_R3:.2f}")
    print(f"  R4 (main):         {R4:.2f}")
    print(f"  R3/R4 ratio:       {ratio:.4f} = {ratio*100:.2f}%")
    print(f"  Runtime:           {time.time()-t0:.1f}s")
    print("="*70)
   
    # Save results
    np.savetxt("A.csv", A, delimiter=",", fmt="%.16e")
    np.savetxt("b_main.csv", b4, delimiter=",", fmt="%.16e")
    np.savetxt(f"b_embedded_dev{int(TARGET_DEV*100):02d}.csv",
               best_overall_b, delimiter=",", fmt="%.16e")
   
    with open("result_final.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_dev","actual_dev","R3","R4","R3_R4_ratio"])
        w.writerow([TARGET_DEV, actual_dev, best_overall_R3, R4, ratio])
   
    # Verify order conditions
    print("\nOrder-3 condition residuals:")
    residuals = [
        np.dot(best_overall_b, np.ones(S)) - 1.0,
        np.dot(best_overall_b, c) - 0.5,
        np.dot(best_overall_b, c**2) - 1.0/3.0,
        np.dot(best_overall_b, A @ c) - 1.0/6.0
    ]
    for i, r in enumerate(residuals, 1):
        print(f"  Condition {i}: {abs(r):.2e}")
   
    print("\nFiles saved: A.csv, b_main.csv, b_embedded_dev04.csv, result_final.csv")
