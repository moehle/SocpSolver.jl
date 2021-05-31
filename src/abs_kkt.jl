@with_kw struct AbstractKktRhs{T <: AbstractFloat}
    dx::Vector{T}
    dy::Vector{T}
    dz::Vector{T}
    ds::Vector{T}
    dτ::T
    dκ::T
    @assert length(ds) == length(dz)
end


function get_aff_dir!(iter::Iterate{T}, resid::Residual{T}, λ::Vector{T}, prob::Problem{T},
                      kkt::KktCache{T}, scaling::CompositeConeScaling{T},
                      settings::Settings{T}, stats::Stats{T}) where T <: AbstractFloat
    # Form the The RHS of (44a-b), i.e., (dx, dy, dz, dτ, ds, dκ).
    ds = jordan(λ, λ, prob.C)
    dκ = iter.κ * iter.τ

    # Solve KKT.
    rhs = AbstractKktRhs{T}(resid.rx, resid.ry, resid.rz, ds, resid.rτ, dκ)
    aff_dir, ir_iters = solve_abstract_kkt(iter, λ, prob, rhs, kkt, scaling)
    register_aff_dir_stats!(stats, ir_iters)
    return aff_dir
end


function get_comb_dir!(iter::Iterate{T}, resid::Residual{T}, λ::Vector{T}, prob::Problem{T},
                       kkt::KktCache{T}, scaling::CompositeConeScaling{T}, aff_dir::StepDirection{T},
                       settings::Settings{T}, σ::T, stats::Stats{T}) where T <: AbstractFloat
    # These parameters map (38a-b) to (44a-b), by defining (dx, dy, dz, dτ, ds, dκ).
    μ_hat = (iter.s' * iter.z + iter.κ * iter.τ) / (degree(prob.C) + 1)
    dx = (1 - σ) .* resid.rx
    dy = (1 - σ) .* resid.ry
    dz = (1 - σ) .* resid.rz
    dτ = (1 - σ) .* resid.rτ

    ds = (jordan(λ, λ, prob.C) 
            + jordan(scaling \ aff_dir.Δs, scaling * aff_dir.Δz, prob.C)
            - σ * μ_hat * get_id(prob.C))
    dκ = iter.κ * iter.τ + aff_dir.Δκ * aff_dir.Δτ - σ*μ_hat

    # Solve KKT.
    rhs = AbstractKktRhs(dx, dy, dz, ds, dτ, dκ)
    comb_dir, ir_iters = solve_abstract_kkt(iter, λ, prob, rhs, kkt, scaling)
    register_comb_dir_stats!(stats, ir_iters, μ_hat)
    return comb_dir
end


function solve_abstract_kkt(iter::Iterate{T}, λ::Vector{T}, prob::Problem{T}, rhs::AbstractKktRhs{T},
                            kkt::KktCache{T}, scaling::CompositeConeScaling{T}) where T <: AbstractFloat
    W = scaling
    temp_z = W * jordan_inv(λ, rhs.ds, prob.C)
    new_rhs = [rhs.dx, -rhs.dy, temp_z - rhs.dz, zeros(2*kkt.q)]
    (x2, y2, z2), best_ref_ratio, last_ref_ratio, num_iters, corr_ratio = solve_and_refine(
                     kkt, new_rhs; max_iters=3)

    Wz1 = W * kkt.z1
    Δτ = ((rhs.dτ - rhs.dκ / iter.τ + prob.c'*x2 + prob.b'*y2 + prob.h'*z2) 
                / (iter.κ ./ iter.τ - prob.c'*kkt.x1 - prob.b'*kkt.y1 - prob.h'*kkt.z1))
    Δx = x2 + Δτ * kkt.x1
    Δy = y2 + Δτ * kkt.y1
    Δz = z2 + Δτ * kkt.z1
    Δs = -temp_z - W*(W*Δz)
    Δκ = -(rhs.dκ + iter.κ*Δτ)/iter.τ 
    return StepDirection(Δx, Δy, Δz, Δτ, Δs, Δκ), num_iters
end
