@with_kw struct Problem{T <: AbstractFloat}
    A::SparseMatrixCSC{T, Int64}
    G::SparseMatrixCSC{T, Int64}
    c::Vector{T}
    b::Vector{T}
    h::Vector{T}
    C::CompositeCone{T}

    @assert all(isfinite.(A.nzval))
    @assert all(isfinite.(G.nzval))
    @assert all(isfinite.(b))
    @assert all(isfinite.(c))
    @assert all(isfinite.(h))
    @assert size(A) == (length(b), length(c))
    @assert size(G) == (length(h), length(c))
    @assert dim(C) == length(h)
end

Problem(A, G, c, b, h, C::NonnegCone) = Problem(A, G, c, b, h, CompositeCone(C))
Problem(A, G, c, b, h, C::QuadCone) = Problem(A, G, c, b, h, CompositeCone(C))

get_num_vars(prob::Problem) = length(prob.c)
get_cone_size(prob::Problem) = length(prob.h)
get_num_constrs(prob::Problem) = length(prob.b)
get_prob_sizes(prob::Problem) = (get_num_vars(prob), get_num_constrs(prob), get_cone_size(prob))

function LinearProgram(A::AbstractMatrix{T}, G::AbstractMatrix{T}, c::Vector{T},
                       b::Vector{T}, h::Vector{T}) where T <: AbstractFloat
    return Problem(A, G, c, b, h, NonnegCone{T}(length(h)))
end

function QuadraticProgram(A::SparseMatrixCSC{T, Int64}, G::SparseMatrixCSC{T, Int64},
                          P::SparseMatrixCSC{T, Int64}, q::Vector{T}, b::Vector{T},
                          h::Vector{T}) where T <: AbstractFloat
    m, n = size(A)
    p, n = size(G)
    idx = diag(P) .> 0
    Pe = P[idx, idx]
    ne = sum(idx)
    ldl_fact = QDLDL.qdldl(Pe; reg=1e-14, is_psd=true)
    L = (ldl_fact.L + sparse(Diagonal(ones(ne)))) * sqrt.(ldl_fact.D)
    D = ldl_fact.D
    perm = ldl_fact.P
    Ue = sparse(L')[:, sortperm(perm)] / sqrt(2)
    U = spzeros(ne, n)
    U[:, idx] = Ue

    At = [spzeros(m) A]
    bt = b
    ct = [1; q]
    Gt = [spzeros(p) G ; -1 spzeros(1,n); spzeros(ne) -2*U; 1 spzeros(1,n)]
    ht = [h; 1; zeros(ne); 1]
    if p > 0
        Ct = CompositeCone([NonnegCone{T}(p), QuadCone{T}(ne+2)])
    else
        Ct = CompositeCone([QuadCone{T}(ne+2)])
    end
    return Problem(At, Gt, ct, bt, ht, Ct)
end
