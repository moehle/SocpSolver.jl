mutable struct KktCache{T}
    K::SparseMatrixCSC{T, Int64}
    Kt::SparseMatrixCSC{T, Int64}
    ldl_fact::QDLDLFactorisation{T, Int64}
    equil::Vector{T}
    reg::T
    x1::Vector{T}
    y1::Vector{T}
    z1::Vector{T}
    n::Int64
    m::Int64
    p::Int64
    q::Int64
end


function KktCache(prob::Problem{T}, settings::Settings{T}, scaling::CompositeConeScaling{T}) where T
    n, m, p = get_prob_sizes(prob)
    P = SparseMatrixCSC(Diagonal(fill(settings.prim_reg, n)))
    S = SparseMatrixCSC(Diagonal(fill(settings.dual_reg, m)))
    R = settings.init_scaling_param*speye(p)
    q = get_num_quad_cones(prob.C)
    sz(dims...) = spzeros(T, dims...)
    K = [         P   prob.A'   prob.G'  sz(n, 2q)
             prob.A        -S  sz(m, p)  sz(m, 2q)
             prob.G  sz(p, m)        -R  sz(p, 2q)
          sz(2q, n) sz(2q, m) sz(2q, p) sz(2q, 2q) ]
    K[n+m+1:end, n+m+1:end] -= get_lower_kkt_block(scaling, identity=true)

    x1 = fill(NaN, n)
    y1 = fill(NaN, m)
    z1 = fill(NaN, p)
    n, m, p = get_prob_sizes(prob)

    perm = amd(K)
    Kt, equil, _, = equilibrate(K, max_iters=settings.kkt_equil_iters)
    reg = settings.init_reg
    ldl_fact = qdldl(Kt, perm=perm, reg=reg)
    kkt = KktCache{T}(K, Kt, ldl_fact, equil, reg, x1, y1, z1, n, m, p, q)
    return kkt
end


function factorize!(kkt::KktCache) 
    kkt.ldl_fact = qdldl(kkt.Kt, perm=kkt.ldl_fact.perm, reg=kkt.reg)
end


function update!(kkt::KktCache, iter::Iterate, prob::Problem, W::CompositeConeScaling, settings::Settings)
    μ_hat = (iter.s' * iter.z + iter.κ * iter.τ) / (degree(prob.C) + 1)
    n, m, p = get_prob_sizes(prob)

    reg_x = min.(settings.prim_reg .+ settings.trust_reg_scaling * μ_hat ./ iter.x.^2, settings.max_trust_reg)
    reg_y = min.(settings.prim_reg .+ settings.trust_reg_scaling * μ_hat ./ iter.y.^2, settings.max_trust_reg)
    P = SparseMatrixCSC(Diagonal(reg_x))
    S = SparseMatrixCSC(Diagonal(reg_y))
    R = SparseMatrixCSC(Diagonal(fill(settings.dual_reg, p)))
    kkt.K[1:n, 1:n] = P
    kkt.K[n+1:n+m, n+1:n+m] = -S
    kkt.K[n+m+1:n+m+p, n+m+1:n+m+p] = -R
    kkt.K[n+m+1:end, n+m+1:end] -= get_lower_kkt_block(W)
    kkt.Kt, kkt.equil, _, = equilibrate(kkt.K, max_iters=5)
    kkt.equil = kkt.equil
    factorize!(kkt)
end


function cache_first_solve!(kkt::KktCache{T}, prob::Problem{T}, settings::Settings{T},
                            stats::Stats{T}) where T
    rhs = [-prob.c, prob.b, prob.h, zeros(2*kkt.q)]
    (kkt.x1, kkt.y1, kkt.z1, _), best_ref_ratio, last_ref_ratio, num_iters, corr_ratio = 
                solve_and_refine(kkt, rhs, max_iters=settings.max_iters_ir)
    register_cache_stats!(stats, num_iters, kkt.reg)

    kkt.reg = get_next_reg(best_ref_ratio, last_ref_ratio, corr_ratio, kkt.reg, settings)
end


function get_next_reg(best_ref_ratio::T, last_ref_ratio::T, corr_ratio::T,
                      last_reg::T, settings::Settings{T}) where T <: AbstractFloat
    if last_ref_ratio > settings.great_ref_ratio
        next_reg = max(last_reg/settings.reg_decrease_param, settings.min_reg)
    elseif (last_ref_ratio < settings.min_ref_ratio_ir) || (corr_ratio < settings.min_corr_ratio_ir) 
        next_reg = min(settings.reg_increase_param*last_reg, settings.max_reg)
    else
        next_reg = max(last_reg/settings.reg_decrease_param, settings.min_reg)
    end
    return next_reg
end


function mult_by_vector(kkt::KktCache{T}, prob::Problem{T}, scaling::CompositeConeScaling{T},
                        x::Vector{T}, y::Vector{T}, z::Vector{T}) where T <: AbstractFloat
    W = convert(Vector{T}, scaling.W)
    Ax = prob.A' * y + prob.G' * z
    Ay = prob.A * x
    Az = -prob.G * x + W'*W*z1
    return Ax, Ay, Az
end


function solve(kkt::KktCache, rhs::Vector{T}) where T <: AbstractFloat
    n_orig = length(rhs)
    n_extra = length(kkt.equil) - n_orig 
    rhs = [rhs; zeros(n_extra)]

    rhs_e = kkt.equil .* rhs
    sol_e = kkt.ldl_fact \ rhs_e
    sol = sol_e .* kkt.equil
    return sol[1:n_orig]
end

 
function split(y::Vector{T}, lens::Vector{Int64}) where T <: AbstractFloat
    cum_lens = [0; cumsum(lens)]
    y_split = [y[cum_lens[i]+1:cum_lens[i+1]] for i in 1:length(lens)]
    return y_split
end


# TODO change to "get_error"
function multiply(kkt::KktCache, x::Vector{T}) where T <: AbstractFloat
    n_orig = length(x)
    n_extra = length(kkt.equil) - n_orig 
    x = [x; zeros(n_extra)]
    E = Diagonal(kkt.ldl_fact.workspace.E[kkt.ldl_fact.iperm])

    x_e = x ./ kkt.equil
    y_e = kkt.Kt * x_e
    y_e_corr = y_e + E * x_e
    y = y_e ./ kkt.equil
    y_corr = y_e_corr ./ kkt.equil
    return y[1:n_orig], y_corr[1:n_orig]
end


function solve_and_refine(kkt::KktCache, rhs::Vector{T}; max_iters=5, tol=1e-13,
                          non_improv_tol=2.) where T <: AbstractFloat

    num_iters = 0
    x = solve(kkt, rhs)
    x_prec = x_best = convert(Vector{T}, x)
    (max_iters == 0) && return x_prec, 1., num_iters, 1.  # If we don't do iterative refinement, just return.

    # Compute error:
    Kx, Kx_plus_Ex = multiply(kkt, x_prec)
    err = rhs - Kx
    corr_err = norm(rhs - Kx_plus_Ex, Inf)
    first_err = last_err = best_err = norm(err, Inf)
    corr_err_ratio = first_err / corr_err

    while num_iters < max_iters
        # Iterative refinement:
        corr = solve(kkt, err)
        x_prec += corr

        # Compute forward error:
        err = rhs - multiply(kkt, x_prec)[1]
        norm_err = norm(err, Inf)

        # First termination condition:
        (last_err < norm_err) && break
        if norm_err < best_err
            x_best = copy(x_prec)
            best_err = norm_err
        end

        # Update:
        x_last = copy(x_prec)
        num_iters += 1

        # Second termination condition:
        (last_err / norm_err < non_improv_tol) && break
        last_err = norm_err
        (norm_err < tol) && break
    end
    return x_prec, first_err/best_err, first_err/last_err, num_iters, corr_err_ratio
end


function solve(kkt::KktCache, x::Vector{Vector{T}}) where T <: AbstractFloat
    rhs = vcat(x...)
    sol = solve(kkt, rhs)
    return split(sol, map(length, x))
end


function solve_and_refine(kkt::KktCache, x::Vector{Vector{T}}; kwargs...) where T <: AbstractFloat
    rhs = vcat(x...)
    sol, best_ref_ratio, last_ref_ratio, num_iters, corr_err_ratio = solve_and_refine(kkt, rhs; kwargs...)
    return split(sol, map(length, x)), best_ref_ratio, last_ref_ratio, num_iters, corr_err_ratio
end
