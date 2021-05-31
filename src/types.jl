@with_kw mutable struct Iterate{T}
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
    s::Vector{T}
    τ::T
    κ::T
end


# τ0 = κ0 = 1 specified in CVXOPT User Guide, §7.1, pg 17:
Iterate{T}(x, y, z, s) where T = Iterate{T}(x, y, z, s, 1., 1.,)

function Iterate(prob::Problem{T}) where T
    x = zeros(T, get_num_vars(prob))
    y = zeros(T, get_num_constrs(prob))
    s = get_id(prob.C)
    z = get_id(prob.C)
    return Iterate{T}(x, y, z, s)
end


copy(iter::Iterate) = Iterate(copy(iter.x), copy(iter.y), copy(iter.z), copy(iter.s), copy(iter.τ), copy(iter.κ))


function copy_to!(iter::Iterate, best_iter::Iterate)
    best_iter.x = iter.x
    best_iter.y = iter.y
    best_iter.z = iter.z
    best_iter.s = iter.s
    best_iter.τ = iter.τ
    best_iter.κ = iter.κ
end

function is_valid(iter::Iterate{T}, prob::Problem{T}) where T
    return (all(isfinite.(iter.x)) && all(isfinite.(iter.y)) && all(isfinite.(iter.z)) && 
            all(isfinite.(iter.s)) && isfinite(iter.τ) && isfinite(iter.κ) &&
            iter.s ∈ prob.C && iter.z ∈ prob.C)
end


@with_kw struct Residual{T}
    rx::Vector{T}
    ry::Vector{T}
    rz::Vector{T}
    rτ::T

    @assert !any(isnan.(rx))
    @assert !any(isnan.(ry))
    @assert !any(isnan.(rz))
    @assert !any(isnan.(rτ))
end


stack(resid::Residual) = [resid.rx, resid.ry, resid.rz, resid.rτ] 


@with_kw struct StepDirection{T}
    Δx::Vector{T}
    Δy::Vector{T}
    Δz::Vector{T}
    Δτ::T
    Δs::Vector{T}
    Δκ::T

    #@assert !any(isnan.(Δx))
    #@assert !any(isnan.(Δy))
    #@assert !any(isnan.(Δz))
    #@assert !any(isnan.(Δτ))
    #@assert !any(isnan.(Δs))
    #@assert !any(isnan.(Δκ))
    #@assert length(Δs) == length(Δz)
end


function StepDirection(Δx::Vector{T}, Δy::Vector{T}, Δz::Vector{T}, Δτ::Vector{T},
                       Δs::Vector{T}, Δκ::Vector{T}) where {T <: AbstractFloat}
    return StepDirection(Δx, Δy, Δz, Δτ[1], Δs, Δκ[1])
end

is_valid(dir::StepDirection) = (all(isfinite.(dir.Δx)) && all(isfinite.(dir.Δy)) && all(isfinite.(dir.Δz)) && 
                                all(isfinite.(dir.Δs)) && isfinite(dir.Δτ) && isfinite(dir.Δκ))
