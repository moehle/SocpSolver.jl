function equilibrate(A::SparseMatrixCSC{T}; max_iters=10, max_val::T=2., is_symm::Bool=false,
                     blk_sizes::Vector{Int64}=ones(Int64, size(A,1))) where T <: AbstractFloat
    m, n = size(A)
    At = copy(A)
    D1 = ones(T, m)
    D2 = ones(T, n)
    Dr = ones(T, m)
    Dc = ones(T, n)
    @assert all(isfinite.(A.nzval))
    for i in 1:max_iters
        abs_At = abs.(At)
        max_dim1 = maximum(abs_At, dims=1)
        max_dim2 = maximum_by_block(abs_At, blk_sizes)
        if max(maximum(max_dim1), maximum(max_dim2)) < max_val
            break
        end
        Dc = vec(sqrt.(max_dim1))
        Dr = vec(sqrt.(max_dim2))
        bound_below!(Dr, 1e-10)
        bound_below!(Dc, 1e-10)
        D1 = D1 ./ Dr
        D2 = D2 ./ Dc
        At = Diagonal(D1) * A * Diagonal(D2)
    end
    @assert all(isfinite.(D1))
    @assert all(isfinite.(D2))
    return At, D1, D2
end


function bound_below!(x::Vector{T}, eps::T) where T <: AbstractFloat
    @inbounds for i in 1:length(x)
        x[i] = max(x[i], eps)
    end
end


function maximum_by_block(A::AbstractMatrix{T}, blk_sizes::Vector{Int}) where T <: Real
    ptrs = [1, cumsum(blk_sizes)]
    offset = 0
    blk_maxes = Vector{T}(undef, size(A, 1))
    for sz in blk_sizes
        idx = offset+1:offset+sz
        max_blk = maximum(A[idx, :])
        blk_maxes[idx] .= max_blk
        offset += sz
    end
    return blk_maxes
end


struct ProblemEquilibration{T <: AbstractFloat}
    x_scale::Vector{T}
    y_scale::Vector{T}
    z_scale::Vector{T}
    obj_scale::T
    const_scale::T
end


function equilibrate(prob::Problem{T}; max_iters=5) where T
    n, m, p = get_prob_sizes(prob)
    blk_sizes = [ones(Int64, m+1); elem_cone_dims(prob.C)]
    M = [prob.c' spzeros(1,1); prob.A prob.b; prob.G prob.h]
    Mt, D1, D2 = equilibrate(M, max_iters=max_iters, blk_sizes=blk_sizes)

    x_scale = D2[1:end-1]
    const_scale = D2[end]
    y_scale = D1[2:get_num_constrs(prob)+1]
    z_scale = D1[get_num_constrs(prob)+2:end]
    obj_scale = D1[1]

    At = copy(prob.A)
    Gt = copy(prob.G)
    ct = copy(prob.c)
    bt = copy(prob.b)
    ht = copy(prob.h)

    At = Diagonal(y_scale) * prob.A * Diagonal(x_scale)
    Gt = Diagonal(z_scale) * prob.G * Diagonal(x_scale)
    ct = Diagonal(x_scale) * prob.c * obj_scale
    bt = Diagonal(y_scale) * prob.b * const_scale
    ht = Diagonal(z_scale) * prob.h * const_scale

    new_prob = Problem(At, Gt, ct, bt, ht, prob.C)
    equil = ProblemEquilibration(x_scale, y_scale, z_scale, obj_scale, const_scale)
    return new_prob, equil
end

function unscale!(sol::Solution{T}, equil::ProblemEquilibration{T}) where T <: AbstractFloat
    sol.x = sol.x .* equil.x_scale ./ equil.const_scale
    sol.y = sol.y .* equil.y_scale ./ equil.obj_scale
    sol.z = sol.z .* equil.z_scale ./ equil.obj_scale
    sol.s = sol.s ./ equil.z_scale ./ equil.const_scale
    sol.optval = sol.optval / (equil.obj_scale * equil.const_scale)
end
