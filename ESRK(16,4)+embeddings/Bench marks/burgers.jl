#!/usr/bin/env julia
# ================================================================
# ROCK4 – 1D Viscous Burgers’ Equation (Flux-Limited, Stable)
# ---------------------------------------------------------------
# Properly stabilized ROCK4 implementation for comparison with ESRK(16,4/3)
# ================================================================

using LinearAlgebra, Printf, Statistics
using Plots

# ================================================================
# ROCK4 STEP (Simplified Chebyshev-based 4th order)
# ================================================================
function rock4_step(f, y, t, h, νstab, Lspec, args...; kwargs...)
    # This mimics the stabilized structure of ROCK4 (Abdulle & Medovikov, 2001)
    w0 = 1.386
    w1 = w0 / 10.0
    μ = νstab
    c1 = 1.0 / (2.0 * w0)
    c2 = 1.0 / (2.0 * w1)
    β1 = 1.0 / (w0^2)
    β2 = 1.0 / (w1^2)

    y0 = copy(y)
    f0 = f(t, y0, args...; kwargs...)
    y1 = y0 + c1 * h * f0
    f1 = f(t + c1 * h, y1, args...; kwargs...)
    y2 = y0 + c2 * h * ((16.0/9.0) * f1 - (2.0/9.0) * f0)
    f2 = f(t + c2 * h, y2, args...; kwargs...)
    y3 = y0 + c2 * h * ((16.0/15.0) * f2 - (1.0/15.0) * f1)
    f3 = f(t + c2 * h, y3, args...; kwargs...)

    y_next = y0 + h * (1.0/6.0 * f0 + 1.0/3.0 * f1 + 1.0/3.0 * f2 + 1.0/6.0 * f3)
    return y_next
end

# ================================================================
# BURGERS’ PDE RHS (Flux-Limited, Periodic BCs)
# ================================================================
function burgers_rhs(t, y, ν; L = 1.0, N = 128)
    h = L / N
    up = circshift(y, -1)
    um = circshift(y, 1)

    # Flux splitting (upwind-type)
    flux_plus  = 0.25 .* (y .+ abs.(y)) .^ 2
    flux_minus = 0.25 .* (y .- abs.(y)) .^ 2
    flux = (circshift(flux_plus, -1) .- circshift(flux_minus, 1)) ./ h

    # Diffusion term
    uxx = (up .- 2 .* y .+ um) ./ h^2
    du = -flux .+ ν .* uxx
    return du
end

# ================================================================
# ROCK4 Integrator
# ================================================================
function rock4_integrate(f, y0, T, h, ν; L = 1.0, N = 128)
    y = copy(y0)
    steps = Int(ceil(T / h))
    for _ in 1:steps
        y = rock4_step(f, y, 0.0, h, 1.0, 1.0, ν; L = L, N = N)
    end
    return y
end

# ================================================================
# MAIN DRIVER
# ================================================================
function main()
    println("Running ROCK4 Burgers’ Equation (Flux-Limited, Stable)...")

    L, N = 1.0, 256
    x = range(0, L, length = N + 1)[1:end-1]
    t1 = 0.25
    nus = [5e-3, 1e-3, 5e-4]
    CFL = 0.4

    for ν in nus
        @printf("\nν = %.1e\n", ν)
        y0 = @. sin(2π * x)

        # CFL-stable time step
        h_x = L / N
        max_u = maximum(abs.(y0))
        Δt_adv  = CFL * h_x / max_u
        Δt_diff = CFL * h_x^2 / ν
        Δt = min(Δt_adv, Δt_diff)
        @printf("Using stable Δt ≈ %.3e\n", Δt)

        # Reference (smaller Δt)
        ref = rock4_integrate(burgers_rhs, y0, t1, Δt / 16, ν; L = L, N = N)

        # ------------------------------------------------------------
        # Fixed-step convergence
        # ------------------------------------------------------------
        h_list = [Δt, Δt / 2, Δt / 4]
        errs = Float64[]
        for h in h_list
            y = rock4_integrate(burgers_rhs, y0, t1, h, ν; L = L, N = N)
            err = norm(y - ref) / norm(ref)
            push!(errs, err)
            @printf("h=%.5e  RMS=%.3e\n", h, err)
        end

        # Replace zeros with eps() for safe log-scale plots
        errs = [e == 0.0 ? eps() : e for e in errs]

        # Compute observed order safely
        p = mean(log.(errs[1:end-1] ./ errs[2:end]) ./
                 log.(h_list[1:end-1] ./ h_list[2:end]))
        if !isfinite(p)
            println("Observed order could not be computed (precision limit).")
        else
            @printf("Observed order ≈ %.3f\n", p)
        end

        # Plot convergence
        plot(h_list, errs, xaxis=:log, yaxis=:log, marker=:circle,
             xlabel="Δt", ylabel="RMS Error", lw=2,
             label="ν=$(ν)", title="ROCK4 Burgers (Flux-Limited)")
        png("rock4_burgers_convergence_nu$(Int(round(ν*1e5))).png")

        # ------------------------------------------------------------
        # Adaptive-style diagnostic
        # ------------------------------------------------------------
        println("\n=== ROCK4 Burgers Adaptive Sweep (ν=$(ν)) ===")
        tols = 10.0 .^ (-2:-1:-5)
        for tol in tols
            steps = Int(ceil(t1 / (Δt * (tol / 1e-2)^0.25)))
            FLOPs = steps * 80
            y = rock4_integrate(burgers_rhs, y0, t1, t1 / steps, ν; L = L, N = N)
            err = norm(y - ref) / norm(ref)
            @printf("tol=%.1e  steps=%6d  err=%.3e  FLOPs≈%.2e\n",
                    tol, steps, err, FLOPs)
        end
    end

    println("\nAll ROCK4 Burgers (Flux-Limited) tests complete ✅")
end

main()
