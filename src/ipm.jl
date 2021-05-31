function solve(prob::Problem, settings::Settings)
    stats = Stats(settings)
    prob, equil = equilibrate(prob, max_iters=settings.prob_equil_iters)
    term_cache = TermCache()
    scaling = CompositeConeScaling(prob.C)
    kkt = KktCache(prob, settings, scaling)
    iter = get_init_iter(prob, kkt, settings)
    best_iter = copy(iter)
    print_first_line(stats, settings)
    for stats.iters in 1:settings.max_iters
        term = iterate!(iter, best_iter, term_cache, kkt, prob, scaling, settings, stats)
        if term
            break
        end
        print_stats_iter(stats, settings)
    end
    sol = unpack_sol(best_iter, prob, stats, settings)
    unscale!(sol, equil)
    truncate_stats!(stats)
    return sol, stats
end


function iterate!(iter::Iterate, best_iter::Iterate, term_cache::TermCache,
                  kkt::KktCache, prob::Problem,
                  scaling::CompositeConeScaling, settings::Settings, stats::Stats)

    # Step 1.
    resid = get_residual(iter, prob, settings)
    if check_term_conds!(iter, best_iter, term_cache, resid, prob, settings, stats)
        return true
    end

    # Step 2.
    update!(scaling, iter.s, iter.z)
    λ = get_scaled_var(scaling)
    update!(kkt, iter, prob, scaling, settings)
    cache_first_solve!(kkt, prob, settings, stats)
    aff_dir = get_aff_dir!(iter, resid, λ, prob, kkt, scaling, settings, stats)

    # Step 3.
    α1 = get_max_step(iter, aff_dir, λ, prob, scaling)
    σ = get_center_param(α1)
    σ = max(σ, settings.min_centering)

    # Step 4.
    comb_dir = get_comb_dir!(iter, resid, λ, prob, kkt, scaling, aff_dir, settings, σ, stats)

    # Step 5.
    α2 = settings.step_param*get_max_step(iter, comb_dir, λ, prob, scaling)
    α2 = min(α2, settings.max_step)
    if α2 > settings.min_step && is_valid(iter, prob) && is_valid(comb_dir) && isfinite(α2)
        take_step!(iter, comb_dir, α2, prob)
    end

    update_stats!(stats, iter, σ, α1, α2)
    return false
end


function update_stats!(stats::Stats{T}, iter::Iterate{T}, σ::T, α1::T, α2::T) where T
    i = stats.iters
    stats.σ[i] = σ
    stats.α_aff[i] = α1
    stats.α_comb[i] = α2
    stats.κ[i] = iter.κ
    stats.τ[i] = iter.τ
end


function take_step!(iter::Iterate{T}, dir::StepDirection{T},
                    α::T, prob::Problem{T}) where T <: AbstractFloat
    if iter.s + α * dir.Δs ∈ prob.C && iter.z + α * dir.Δz ∈ prob.C
        iter.x = iter.x + α * dir.Δx
        iter.y = iter.y + α * dir.Δy
        iter.z = iter.z + α * dir.Δz
        iter.s = iter.s + α * dir.Δs
        iter.τ = iter.τ + α * dir.Δτ
        iter.κ = iter.κ + α * dir.Δκ
    end
end


get_center_param(α::AbstractFloat) = (1 - α)^3


function get_max_step(iter::Iterate{T}, dir::StepDirection{T}, λ::Vector{T},
                      prob::Problem{T}, scaling::CompositeConeScaling{T}) where T
    Δs_tilde = scaling \ dir.Δs
    Δz_tilde = scaling * dir.Δz
    α = min(get_max_step(λ, Δs_tilde, Δz_tilde, prob.C),
            get_max_step([iter.τ], [dir.Δτ], NonnegCone(1)),
            get_max_step([iter.κ], [dir.Δκ], NonnegCone(1)))
    return α
end


function get_residual(iter::Iterate{T}, prob::Problem{T}, settings::Settings{T}) where T
    rx = -prob.A' * iter.y - prob.G' * iter.z - prob.c * iter.τ
    ry = prob.A * iter.x - prob.b * iter.τ
    rz = iter.s + prob.G * iter.x - prob.h * iter.τ
    rτ = iter.κ + prob.c' * iter.x + prob.b' * iter.y + prob.h' * iter.z
    return Residual{T}(rx, ry, rz, rτ)
end
 

function get_init_iter(prob::Problem{T}, kkt::KktCache{T}, settings::Settings{T}) where T
    x_hat, s_hat = get_init_xs(prob, kkt::KktCache)
    y_hat, z_hat = get_init_yz(prob, kkt::KktCache)
    #if s_hat ∈ prob.C && z_hat ∈ prob.C
    if false
        iter = Iterate(x_hat, y_hat, z_hat, s_hat)
    else
        iter = Iterate(prob)
    end
    return iter
end


function get_init_xs(prob::Problem{T}, kkt::KktCache{T}) where T <: AbstractFloat
    n = get_num_vars(prob)
    (x_hat, _, s_tilde), _, _, _, _ = solve_and_refine(kkt, [zeros(T, n), prob.b, prob.h])
    e = convert(Vector{T}, get_id(prob.C))
    αp = dist_along_id_ray(s_tilde, prob.C)
    s_hat = s_tilde + (1 + 1.1*αp).*e
    return x_hat, s_hat
end


function get_init_yz(prob::Problem{T}, kkt::KktCache{T}) where T <: AbstractFloat
    _, m, p = get_prob_sizes(prob)
    (_, y_hat, z_tilde), _, _, _, _ = solve_and_refine(kkt, [-prob.c, zeros(T, m), zeros(T, p)])
    e = convert(Vector{T}, get_id(prob.C))
    αd = dist_along_id_ray(z_tilde, prob.C)
    z_hat = z_tilde + (1 + αd).*e
    return y_hat, z_hat
end
