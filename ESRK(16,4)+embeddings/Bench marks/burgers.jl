#!/usr/bin/env julia
# ====================================================================
# ROCK4 — 1D Viscous Burgers' Equation (Flux Form, Periodic BCs)
# Matching ESRK(16,4/3) Python Script Exactly
# ====================================================================

using DifferentialEquations
using LinearAlgebra
using Statistics
using Printf
using Plots

println("\nRunning TRUE ROCK4 — 1D Viscous Burgers’ Equation (Flux Form)\n")

# ------------------------------------------------------------
# 1️⃣ Burgers’ PDE RHS (Flux Conservative Form)
# ------------------------------------------------------------
function burgers_rhs!(du, u, p, t)
    N, L, ν = p
    h = L / N

    up = circshift(u, -1)
    um = circshift(u, 1)

    # flux form: du/dt = -(1/2) d(u^2)/dx + ν u_xx
    flux = 0.5 .* (up.^2 .- um.^2) ./ (2h)
    uxx  = (up .- 2u .+ um) ./ h^2

    du .= -flux .+ ν .* uxx
end

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
L = 1.0
N = 128
x = range(0, L; length=N+1)[1:end-1]

t0, t1 = 0.0, 0.25
nus = [5e-3, 1e-3, 5e-4]   # mild → moderate stiffness (match ESRK script)

# FLOP MODEL (match ESRK)
FLOPS_PER_RHS = 10

# Initial condition
function init_u()
    @. sin(2π * x)
end

for ν in nus
    println("\n==============================")
    @printf("ν = %.1e\n", ν)
    println("==============================")

    u0 = init_u()
    p = (N, L, ν)

    # ------------------------------------------------------------
    # Reference solution (very fine dt, non-adaptive)
    # ------------------------------------------------------------
    println("\nComputing reference with ROCK4 dt=1e-4 …")

    prob_ref = ODEProblem(burgers_rhs!, u0, (t0, t1), p)
    sol_ref = solve(prob_ref, ROCK4(); dt=1e-4, adaptive=false, maxiters=1e9)
    u_ref = sol_ref(t1)

    @printf("Reference RMS(|u|) = %.4e\n", norm(u_ref))

    # ------------------------------------------------------------
    # FIXED STEP CONVERGENCE
    # ------------------------------------------------------------
    println("\n=== ROCK4 Fixed-Step Convergence ===")
    h_list = [0.01, 0.005, 0.0025, 0.00125]
    errs = Float64[]

    for h in h_list
        prob = ODEProblem(burgers_rhs!, u0, (t0, t1), p)
        sol = solve(prob, ROCK4(); dt=h, adaptive=false, maxiters=1e9)
        u = sol(t1)

        if any(isnan.(u))
            @printf("h=%.5f  [UNSTABLE]\n", h)
            continue
        end

        e = norm(u - u_ref) / norm(u_ref)
        push!(errs, e)
        @printf("h=%.5f  RMS Error=%.3e\n", h, e)
    end

    # order estimate
    if length(errs) ≥ 2
        p_est = mean(log.(errs[1:end-1] ./ errs[2:end]) ./ log.(h_list[1:end-1] ./ h_list[2:end]))
        @printf("Observed order ≈ %.3f\n", p_est)
    end

    # Plot convergence
    plot(h_list, errs, xaxis=:log, yaxis=:log, marker=:circle,
         xlabel="Δt", ylabel="RMS Error", lw=2,
         title="ROCK4 Burgers Convergence ν=$ν")
    savefig("rock4_burgers_convergence_ν$(Int(round(ν*1e5))).png")

    # ------------------------------------------------------------
    # ADAPTIVE ROCK4 SWEEP
    # ------------------------------------------------------------
    println("\n=== Adaptive ROCK4 (matching ESRK sweep) ===")
    tols = 10.0 .^ range(-2, -5, length=7)

    for tol in tols
        prob = ODEProblem(burgers_rhs!, u0, (t0, t1), p)

        sol = solve(prob, ROCK4();
                    reltol=tol, abstol=tol,
                    maxiters=1e9)

        u = sol(t1)
        err = norm(u - u_ref) / norm(u_ref)

        steps   = sol.destats.naccept
        rejects = sol.destats.nreject
        fevals  = sol.destats.nf      # RHS eval count

        dt_hist = diff(sol.t)
        mean_h = mean(dt_hist)

        FLOPs = fevals * FLOPS_PER_RHS

        @printf("tol=%8.1e  steps=%5d  rej=%3d  err=%10.3e  mean(h)=%.3e  FLOPs≈%.2e\n",
                tol, steps, rejects, err, mean_h, FLOPs)
    end
end

println("\nAll ROCK4 Burgers tests complete. ✓\n")

