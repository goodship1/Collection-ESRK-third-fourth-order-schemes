#!/usr/bin/env julia
using DifferentialEquations, LinearAlgebra, Printf, Statistics, CSV, DataFrames, Plots

# ============================================================
# 1️⃣ Schnakenberg PDE (Periodic 1D)
# ============================================================
function schnakenberg_pde!(du, u, p, t)
    a, b, Du, Dv, L, N = p
    h = L / N
    u1 = @view u[1:N]
    v1 = @view u[N+1:2N]
    du1 = @view du[1:N]
    dv1 = @view du[N+1:2N]

    ∂²u = similar(u1)
    ∂²v = similar(v1)

    # Periodic Laplacian
    ∂²u[1]   = (u1[2] - 2u1[1] + u1[end]) / h^2
    ∂²u[end] = (u1[1] - 2u1[end] + u1[end-1]) / h^2
    ∂²v[1]   = (v1[2] - 2v1[1] + v1[end]) / h^2
    ∂²v[end] = (v1[1] - 2v1[end] + v1[end-1]) / h^2
    for i in 2:N-1
        ∂²u[i] = (u1[i+1] - 2u1[i] + u1[i-1]) / h^2
        ∂²v[i] = (v1[i+1] - 2v1[i] + v1[i-1]) / h^2
    end

    du1 .= Du .* ∂²u .+ a .- u1 .+ u1.^2 .* v1
    dv1 .= Dv .* ∂²v .+ b .- u1.^2 .* v1
end

# ============================================================
# 2️⃣ Base parameters
# ============================================================
N = 64
L = 1.0
a, b = 0.2, 1.3
Du_base, Dv_base = 1e-4, 5e-5
tspan = (0.0, 5.0)
x = LinRange(0, L, N)
FLOPS_PER_RHS = 10.0

# ============================================================
# 3️⃣ Stiffness scales to test
# ============================================================
scales = [0.1, 1.0, 10.0, 100.0]
@info "Running ROCK4 Schnakenberg PDE across scales = $(scales)"

for scale in scales
    println("\n" * "="^30)
    @printf("Scale = %.1f, Du=%.2e, Dv=%.2e\n", scale, Du_base*scale, Dv_base*scale)
    println("="^30)

    Du = Du_base * scale
    Dv = Dv_base * scale
    u0 = @. a + b + 0.05 * sin(2π * x / L)
    v0 = @. b / (a + b)^2 + 0.05 * cos(2π * x / L)
    u_init = vcat(u0, v0)
    p = (a, b, Du, Dv, L, N)

    # ============================================================
    # 4️⃣ Reference solution
    # ============================================================
    ref_prob = ODEProblem(schnakenberg_pde!, u_init, tspan, p)
    @info "Computing reference (dt=1e-3)..."
    ref_sol = solve(ref_prob, ROCK4(), dt=1e-3, adaptive=false)
    y_ref = ref_sol(tspan[2])

    # ============================================================
    # 5️⃣ Fixed-step convergence
    # ============================================================
    hs = [0.05, 0.025, 0.0125, 0.00625]
    errs, flops_conv = Float64[], Float64[]
    @info "=== ROCK4 Schnakenberg Fixed-Step (scale=$(scale)) ==="
    for h in hs
        prob = ODEProblem(schnakenberg_pde!, u_init, tspan, p)
        sol = solve(prob, ROCK4(), dt=h, adaptive=false)
        sol_val = sol(tspan[2])
        err = norm(sol_val - y_ref) / sqrt(2N)
        nfe = sol.destats.nf
        total_flops = nfe * FLOPS_PER_RHS
        push!(errs, err)
        push!(flops_conv, total_flops)
        @printf "h=%7.5f, RMS=%10.3e, nfe=%6d, FLOPs≈%.2e\n" h err nfe total_flops
    end
    order = mean([log(errs[i]/errs[i+1]) / log(hs[i]/hs[i+1]) for i in 1:length(hs)-1])
    @printf "Observed temporal order ≈ %.3f\n" order

    plt1 = plot(hs, errs; xaxis=:log, yaxis=:log, lw=2, xlabel="Δt", ylabel="RMS Error",
                label="ROCK4 (scale=$(scale))", grid=:both,
                title="ROCK4 Schnakenberg PDE (scale=$(scale))")
    savefig(plt1, "rock4_schnak_conv_scale$(scale).png")

    # ============================================================
    # 6️⃣ Adaptive tolerance sweep
    # ============================================================
    tols = 10.0 .^ (-2:-1:-5)
    errs_ad, steps_ad, rej_ad, flops_ad = Float64[], Int[], Int[], Float64[]
    @info "=== ROCK4 Schnakenberg Adaptive Sweep (scale=$(scale)) ==="
    for tol in tols
        prob = ODEProblem(schnakenberg_pde!, u_init, tspan, p)
        sol = solve(prob, ROCK4(), abstol=tol, reltol=tol, dt=0.02)
        sol_val = sol(tspan[2])
        err = norm(sol_val - y_ref) / sqrt(2N)
        nfe = sol.destats.nf
        total_flops = nfe * FLOPS_PER_RHS
        push!(errs_ad, err)
        push!(steps_ad, sol.destats.naccept)
        push!(rej_ad, sol.destats.nreject)
        push!(flops_ad, total_flops)
        @printf "tol=%-8.1e  steps=%5d  rej=%3d  err=%10.3e  FLOPs≈%.2e\n" tol sol.destats.naccept sol.destats.nreject err total_flops
    end

    # ============================================================
    # 7️⃣ Save plots + CSV
    # ============================================================
    df = DataFrame(tol=tols, steps=steps_ad, rejects=rej_ad,
                   global_err=errs_ad, flops_est=flops_ad)
    CSV.write("rock4_schnak_adaptive_scale$(scale).csv", df)

    plt2 = plot(tols, errs_ad; xaxis=:log, yaxis=:log, lw=2, marker=:o,
                xlabel="Tolerance", ylabel="RMS Error", grid=:both,
                label="ROCK4 (scale=$(scale))",
                title="Error vs Tolerance — scale=$(scale)")
    savefig(plt2, "rock4_schnak_error_vs_tol_scale$(scale).png")

    plt3 = plot(flops_ad, errs_ad; xaxis=:log, yaxis=:log, lw=2, marker=:o,
                xlabel="Estimated FLOPs", ylabel="RMS Error", grid=:both,
                label="ROCK4 (scale=$(scale))",
                title="Error vs FLOPs — scale=$(scale)")
    savefig(plt3, "rock4_schnak_error_vs_flops_scale$(scale).png")

    println("\nSaved results for scale=$(scale).")
end

println("\nAll ROCK4 Schnakenberg stiffness scales complete ✅")
