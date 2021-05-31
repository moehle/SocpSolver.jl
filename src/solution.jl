@enum Status begin
    unsolved_status
    optimal_status
    unbounded_status
    infeasible_status
    max_iters_status
    numerical_error_status
end


@with_kw mutable struct Solution{T <: AbstractFloat}
    status::Status = unsolved_status
    optval::T = NaN
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
    s::Vector{T}
end

Solution{T}(n::Int64, m::Int64, p::Int64) where T = Solution{T}(unsolved_status, NaN, fill(NaN, n),
                                                                fill(NaN, m), fill(NaN, p), fill(NaN, p))
Solution(prob::Problem{T}) where T = Solution{T}(get_prob_sizes(prob)...)


mutable struct Stats{T <: AbstractFloat}
    iters::Int64
    residual::Vector{T}
    best::Vector{Bool}
    σ::Vector{T}
    μ::Vector{T}
    κ::Vector{T}
    τ::Vector{T}
    α_aff::Vector{T}
    α_comb::Vector{T}
    ir_iters_cache::Vector{Int64}
    ir_iters_aff::Vector{Int64}
    ir_iters_comb::Vector{Int64}
    regularization::Vector{T}
end


function Stats(settings::Settings)
    max_iters = settings.max_iters
    max_iters_ir = settings.max_iters_ir

    iters = 0
    residual = fill(NaN, max_iters)
    best = fill(false, max_iters)
    σ = fill(NaN, max_iters)
    μ = fill(NaN, max_iters)
    κ = fill(NaN, max_iters)
    τ = fill(NaN, max_iters)
    α_aff = fill(NaN, max_iters)
    α_comb = fill(NaN, max_iters)
    ir_iters_cache = fill(0, max_iters)
    ir_iters_aff = fill(0, max_iters)
    ir_iters_comb = fill(0, max_iters)
    regularization = fill(NaN, max_iters)
    return Stats(iters, residual, best, σ, μ, κ, τ, α_aff, α_comb,
                 ir_iters_cache, ir_iters_aff, ir_iters_comb, regularization)
end


# TODO can this be shortened?
function truncate_stats!(stats::Stats)
    i = stats.iters
    stats.residual = stats.residual[1:i]
    stats.best = stats.best[1:i]
    stats.σ = stats.σ[1:i]
    stats.μ = stats.μ[1:i]
    stats.κ = stats.κ[1:i]
    stats.τ = stats.τ[1:i]
    stats.α_aff = stats.α_aff[1:i]
    stats.α_comb = stats.α_comb[1:i]
    stats.ir_iters_cache = stats.ir_iters_cache[1:i]
    stats.ir_iters_aff = stats.ir_iters_aff[1:i]
    stats.ir_iters_comb = stats.ir_iters_comb[1:i]
    stats.regularization = stats.regularization[1:i]
end

function register_aff_dir_stats!(stats::Stats, ir_iters::Int64)
    stats.ir_iters_aff[stats.iters] = ir_iters
end

function register_comb_dir_stats!(stats::Stats{T}, ir_iters::Int64, μ_hat::T) where T
    stats.ir_iters_comb[stats.iters] = ir_iters
    stats.μ[stats.iters] = μ_hat
end

function register_cache_stats!(stats::Stats{T}, ir_iters::Int64, reg::T) where T
    stats.ir_iters_cache[stats.iters] = ir_iters
    stats.regularization[stats.iters] = reg
end
