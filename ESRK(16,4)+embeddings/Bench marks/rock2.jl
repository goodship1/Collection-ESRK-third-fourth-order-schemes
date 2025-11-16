#!/usr/bin/env julia
using DifferentialEquations, LinearAlgebra, Printf, Statistics, CSV, DataFrames, Plots

# ============================================================
# 1️⃣ 1D Reaction–Diffusion Brusselator PDE
# ============================================================
function brusselator_pde!(du, u, p, t)
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

    du1 .= Du .* ∂²u .+ a .- (b + 1) .* u1 .+ u1.^2 .* v1
    dv1 .= Dv .* ∂²v .+ b .* u1 .- u1.^2 .* v1
end

# ============================================================
# 2️⃣ Parameters
# ============================================================
N = 64
L = 1.0
a, b = 1.0, 3.0
Du_base, Dv_base = 5e-4, 2.5e-4
tspan = (0.0, 5.0)
x = LinRange(0, L, N)
u0 = @. 1.0 + 0.1 * sin(2π * x / L)
v0 = @. 3.0 + 0.1 * cos(2π * x / L)
u_init = vcat(u0, v0)
FLOPS_PER_RHS = 10.0

# ============================================================
# 3️⃣ Scales of stiffness to test
# ============================================================
scales = [0.1, 1.0, 10.0, 100.0]
@info "Running ROCK4 PDE diagnostics across scales = $(scales)"

for scale in scales
    Du, Dv = Du_base * scale, Dv_base * scale
    p = (a, b, Du, Dv, L, N)
    println("\n==============================")
    println("Scale = $(scale), Du=$(Du), Dv=$(Dv)")
    println("==============================")

    # Reference high-accuracy
    ref_prob = ODEProblem(brusselator_pde!, u_init, tspan, p)
    @info "Computing reference (dt=1e-3)..."
    ref_sol = solve(ref_prob, ROCK4(), dt=1e-3, adaptive=false)
    y_ref = ref_sol(tspan[2])

    # ------------------------------------------------------------
    # Fixed-step convergence
    # ------------------------------------------------------------
    hs = [0.05, 0.025, 0.0125, 0.00625]
    errs, flops_conv = Float64[], Float64[]
    @info "=== ROCK4 PDE Fixed-Step Convergence (scale=$(scale)) ==="
    for h in hs
        prob = ODEProblem(brusselator_pde!, u_init, tspan, p)
        sol = solve(prob, ROCK4(), dt=h, adaptive=false)
        sol_val = sol(tspan[2])
        err = norm(sol_val - y_ref) / sqrt(2N)
        nfe = sol.destats.nf
        total_flops = nfe * FLOPS_PER_RHS
        push!(errs, err)
        push!(flops_conv, total_flops)
        @printf "h=%7.5f, RMS=%.3e, nfe=%6d, FLOPs≈%.2e\n" h err nfe total_flops
    end
    orders = [log(errs[i]/errs[i+1]) / log(hs[i]/hs[i+1]) for i in 1:length(hs)-1]
    p_est = mean(orders)
    @printf "Observed temporal order ≈ %.3f\n" p_est

    # Plot convergence
    plt1 = plot(hs, errs; seriestype=:scatter, xaxis=:log, yaxis=:log,
        xlabel="Δt", ylabel="RMS Error", label="ROCK4 scale=$(scale)",
        grid=:both, title="ROCK4 PDE Convergence (scale=$(scale))")
    plot!(plt1, hs, errs, lw=2)
    savefig(plt1, "rock4_pde_convergence_scale_$(scale).png")

    # ------------------------------------------------------------
    # Adaptive tolerance sweep
    # ------------------------------------------------------------
    tols = 10.0 .^ (-2:-1:-5)
    global_errs, steps_list, rej_list, flops_tol =
        Float64[], Int[], Int[], Float64[]

    @info "=== ROCK4 PDE Adaptive Sweep (scale=$(scale)) ==="
    for tol in tols
        prob = ODEProblem(brusselator_pde!, u_init, tspan, p)
        sol = solve(prob, ROCK4(), abstol=tol, reltol=tol)
        sol_val = sol(tspan[2])
        err = norm(sol_val - y_ref) / sqrt(2N)
        nfe = sol.destats.nf
        total_flops = nfe * FLOPS_PER_RHS
        push!(global_errs, err)
        push!(steps_list, sol.destats.naccept)
        push!(rej_list, sol.destats.nreject)
        push!(flops_tol, total_flops)
        @printf "tol=%-8.1e  steps=%5d  rej=%3d  err=%10.3e  FLOPs≈%.2e\n" tol sol.destats.naccept sol.destats.nreject err total_flops
    end

    # ------------------------------------------------------------
    # Plots for adaptive analysis
    # ------------------------------------------------------------
    plt2 = plot(tols, global_errs; seriestype=:scatter, xaxis=:log, yaxis=:log,
        xlabel="Tolerance", ylabel="RMS Error",
        label="ROCK4 scale=$(scale)", grid=:both)
    plot!(plt2, tols, global_errs, lw=2)
    savefig(plt2, "rock4_pde_error_vs_tol_scale_$(scale).png")

    plt3 = plot(flops_tol, global_errs; seriestype=:scatter, xaxis=:log, yaxis=:log,
        xlabel="Estimated FLOPs", ylabel="RMS Error",
        label="ROCK4 scale=$(scale)", grid=:both)
    plot!(plt3, flops_tol, global_errs, lw=2)
    savefig(plt3, "rock4_pde_error_vs_flops_scale_$(scale).png")

    plt4 = plot(tols, steps_list; seriestype=:scatter, xaxis=:log, yaxis=:log,
        xlabel="Tolerance", ylabel="Accepted Steps",
        label="ROCK4 scale=$(scale)", grid=:both)
    plot!(plt4, tols, steps_list, lw=2)
    savefig(plt4, "rock4_pde_steps_vs_tol_scale_$(scale).png")

    # ------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------
    df = DataFrame(
        tol = tols,
        steps = steps_list,
        rejects = rej_list,
        global_err = global_errs,
        flops_est = flops_tol,
        scale = fill(scale, length(tols))
    )
    CSV.write("bruss_pde_adaptive_summary_scale_$(scale).csv", df)

    println("\nSaved scale=$(scale) results and plots.\n")
end

println("\nAll ROCK4 diagnostic scales complete ✅")
