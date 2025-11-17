#!/usr/bin/env julia
# ================================================================
# FitzHugh–Nagumo PDE — TRUE ROCK4 Diagnostics (Julia ODE)
# Matching ESRK(16,4/3) Python script
# ================================================================

using DifferentialEquations
using LinearAlgebra
using Statistics
using Printf
using Plots

println("\nFitzHugh–Nagumo PDE — TRUE ROCK4 Diagnostics (Julia ODE)\n")

# ------------------------------------------------------------
# 1D Laplacian (periodic BCs)
# ------------------------------------------------------------
function lap(u, dx)
    N = length(u)
    out = similar(u)
    out[1]     = (u[2] - 2u[1] + u[N]) / dx^2
    out[N]     = (u[1] - 2u[N] + u[N-1]) / dx^2
    @inbounds for i in 2:(N-1)
        out[i] = (u[i+1]-2u[i]+u[i-1]) / dx^2
    end
    return out
end

# ------------------------------------------------------------
# FitzHugh–Nagumo RHS on concatenated y=[u; v]
# ------------------------------------------------------------
function fhn!(du, y, p, t)
    N, dx, Du, eps, γ, a = p
    u = @view y[1:N]
    v = @view y[N+1:2N]

    du_u = @view du[1:N]
    du_v = @view du[N+1:2N]

    uxx = lap(u, dx)

    du_u .= Du .* uxx .+ u .* (u .- a) .* (1 .- u) .- v
    du_v .= eps .* (u .- γ .* v)
end

# ------------------------------------------------------------
# Problem setup
# ------------------------------------------------------------
N = 200
L = 1.0
x = LinRange(0, L, N)
dx = L/N

Du   = 0.005
eps  = 0.05
γ    = 2.0
a    = 0.1
params = (N, dx, Du, eps, γ, a)

u0 = 0.5 .* (1 .+ tanh.(10 .* (0.5 .- x)))
v0 = zeros(N)
y0 = vcat(u0, v0)

t0, t1 = 0.0, 0.1

# ============================================================
# Reference solution (very fine dt)
# ============================================================
println("Computing reference solution using ROCK4 dt=5e-5 ...")
prob_ref = ODEProblem(fhn!, y0, (t0, t1), params)

sol_ref = solve(prob_ref, ROCK4();
                dt = 5e-5,
                adaptive = false,
                maxiters = 1e10)

y_ref = sol_ref(t1)
println("Reference max|u| = ", maximum(abs.(y_ref[1:N])))

# ============================================================
# FIXED-STEP ROCK4 CONVERGENCE
# ============================================================
println("\n=== ROCK4 Fixed-Step Convergence ===")

h_list = [0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625]
errs = Float64[]

for h in h_list
    prob = ODEProblem(fhn!, y0, (t0, t1), params)
    sol = solve(prob, ROCK4();
                dt=h, adaptive=false, maxiters=1e10)

    y = sol(t1)

    if any(isnan.(y))
        @printf("dt=%8.5f  [UNSTABLE]\n", h)
        continue
    end

    e = norm(y - y_ref) / sqrt(length(y))
    push!(errs, e)
    @printf("dt=%8.5f  RMS=%10.3e\n", h, e)
end

if length(errs) >= 2
    orders = [ log(errs[i]/errs[i+1]) /
               log(h_list[i]/h_list[i+1])
               for i in 1:length(errs)-1 ]
    @printf("Observed order ≈ %.6f\n", mean(orders))
end

# ============================================================
# ADAPTIVE ROCK4 (like ESRK Python)
# ============================================================
println("\n=== Adaptive ROCK4 ===")

tols = 10.0 .^ range(-2, -5, length=6)

# FLOP model (same as Python ESRK)
F_RHS = 20N     # FLOPs per RHS evaluation
# FLOPs = fevals * F_RHS

for tol in tols
    prob = ODEProblem(fhn!, y0, (t0, t1), params)

    sol = solve(prob, ROCK4();
                reltol=tol, abstol=tol,
                maxiters=1e10)

    y = sol(t1)
    err = norm(y - y_ref) / sqrt(length(y))

    # Older DifferentialEquations.jl uses these fields:
    steps   = sol.destats.naccept    # number of accepted steps
    rejects = sol.destats.nreject    # rejected steps
    fevals  = sol.destats.nf         # RHS evaluations

    hs = diff(sol.t)
    mean_h = mean(hs)

    FLOPs = fevals * F_RHS

    @printf("tol=%7.1e  steps=%6d  rej=%3d  err=%10.3e  mean(h)=%8.3e  FLOPs≈%.2e\n",
            tol, steps, rejects, err, mean_h, FLOPs)
end

println("\nREADY.\n")
