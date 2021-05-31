@with_kw mutable struct TermCache{T <: AbstractFloat}
    best_resid::T = Inf
    non_improv_iters::Int64 = 0
end


function unpack_sol(iter::Iterate{T}, prob::Problem{T}, stats::Stats{T}, settings::Settings{T}) where T
    sol = Solution(prob)
    prim_obj = prob.c'*iter.x
    dual_obj = prob.b'*iter.y + prob.h'*iter.z
    if iter.τ ≤ EPS
        sol.x = iter.x
        sol.y = iter.y
        sol.z = iter.z
        sol.s = iter.s
        if prim_obj < EPS
            sol.status = unbounded_status
            sol.optval = -Inf
        elseif dual_obj < EPS
            sol.status = infeasible_status
            sol.optval = Inf
        else
            sol.status = numerical_error_status
            sol.optval = NaN
        end
    else
        sol.status = optimal_status
        sol.x = iter.x / iter.τ
        sol.y = iter.y / iter.τ
        sol.z = iter.z / iter.τ
        sol.s = iter.s / iter.τ
        sol.optval = prim_obj / iter.τ
    end
    if stats.iters == settings.max_iters
        sol.status = max_iters_status
    end
    return sol
end


function check_term_conds!(iter::Iterate{T}, best_iter::Iterate{T}, term_cache::TermCache{T},
                           resid::Residual{T}, prob::Problem{T}, settings::Settings{T},
                           stats::Stats{T}) where T
    norm_resid = norm(stack(resid), Inf)
    terminate = norm_resid ≤ settings.tol

    if norm_resid ≥ settings.non_improv_ratio_thres * term_cache.best_resid 
        term_cache.non_improv_iters += 1
    else
        term_cache.non_improv_iters = 0
    end
    if term_cache.non_improv_iters > settings.max_non_improv_iters
        terminate = true
    end

    if norm_resid < term_cache.best_resid
        term_cache.best_resid = norm_resid
        copy_to!(iter, best_iter)
        stats.best[stats.iters] = true
    else
        stats.best[stats.iters] = false
    end
    stats.residual[stats.iters] = norm_resid

    return terminate
end


