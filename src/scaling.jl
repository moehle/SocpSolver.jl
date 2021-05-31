abstract type SimpleConeScaling{T <: AbstractFloat} end


@with_kw mutable struct NonnegConeScaling{T <: AbstractFloat} <: SimpleConeScaling{T}
    w::Vector{T}
    λ::Vector{T}
    @assert !any(isnan.(w))
    @assert !any(isnan.(λ))
end

NonnegConeScaling(C::NonnegCone) = NonnegConeScaling(get_id(C), get_id(C))


@with_kw mutable struct QuadConeScaling{T <: AbstractFloat} <: SimpleConeScaling{T}
    w::Vector{T}
    λ::Vector{T}
    η::T
    @assert !any(isnan.(w))
    @assert !any(isnan.(λ))
end

QuadConeScaling(C::QuadCone) = QuadConeScaling(get_id(C), get_id(C), 1.)

get_scaling(C::NonnegCone) = NonnegConeScaling(C)
get_scaling(C::QuadCone) = QuadConeScaling(C)


@with_kw mutable struct CompositeConeScaling{T <: AbstractFloat}
    scalings::Vector{SimpleConeScaling{T}}
end

CompositeConeScaling(C::CompositeCone{T}) where T = CompositeConeScaling{T}(get_scaling.(C))

iterate(W::CompositeConeScaling, args...) = iterate(W.scalings, args...)
getindex(W::CompositeConeScaling, args...) = getindex(W.scalings, args...)
length(W::CompositeConeScaling) = length(W.scalings)
size(W::CompositeConeScaling) = (length(W),)

dim(W::NonnegConeScaling) = length(W.w)
dim(W::QuadConeScaling)  = length(W.w)
dim(W::CompositeConeScaling) = sum(map(dim, W))

get_scaled_var(W::NonnegConeScaling) = W.λ
get_scaled_var(W::QuadConeScaling) = W.λ
function get_scaled_var(W::CompositeConeScaling{T}) where T
    return vcat(T[], get_scaled_var.(W)...)
end

*(W::NonnegConeScaling{T}, x::Vector{T}) where T <: AbstractFloat = W.w .* x
*(x::Vector{T}, W::NonnegConeScaling{T}) where T <: AbstractFloat = W * x
\(W::NonnegConeScaling{T}, x::Vector{T}) where T <: AbstractFloat = W.w .\ x
/(x::Vector{T}, W::NonnegConeScaling{T}) where T <: AbstractFloat = W \ x

function mult_or_div(W::QuadConeScaling{T}, x::Vector{T}; is_mult=true) where T <: AbstractFloat
    y = Vector{T}(undef, length(x))
    sign = is_mult ? 1 : -1
    w = W.w

    wTx = w[2:end]'*x[2:end]
    y[1] = w[1]*x[1] + sign*wTx
    y[2:end] = sign*w[2:end]*x[1] + x[2:end] + w[2:end]*wTx/(w[1] + 1)
    y *= W.η^sign
    return y
end

*(W::QuadConeScaling{T}, x::Vector{T}) where T <: AbstractFloat = mult_or_div(W, x, is_mult=true)
\(W::QuadConeScaling{T}, x::Vector{T}) where T <: AbstractFloat = mult_or_div(W, x, is_mult=false)

function *(W::CompositeConeScaling{T}, x::Vector{T}) where T <: AbstractFloat
    x_list = split_by_cone(x, W)
    return vcat(T[], (W .* x_list)...)
end

function \(W::CompositeConeScaling{T}, x::Vector{T}) where T <: AbstractFloat
    x_list = split_by_cone(x, W)
    y = Vector{T}(undef, length(x))
    idx = 1
    for Wi in W
        dim_Wi = dim(Wi)
        y[idx: idx+dim_Wi-1] = Wi \ x[idx: idx+dim_Wi-1]
        idx += dim_Wi
    end
    return y
end

function update!(W::NonnegConeScaling{T}, s::Vector{T}, z::Vector{T}) where T <: AbstractFloat
    W.w = sqrt.(s ./ z)
    W.λ = sqrt.(s) .* sqrt.(z)
end

function update!(W::QuadConeScaling{T}, s::Vector{T}, z::Vector{T}) where T <: AbstractFloat
    zbar = z / sqrt(pos(z'*(J*z)))
    sbar = s / sqrt(pos(s'*(J*s)))
    γ = sqrt((1 + zbar'*sbar)/2)
    w = .5*(sbar + J*zbar)/γ
    W.w = w
    W.η = pos((s'*(J*s)) / (z'*(J*z)))^(1/4)
    W.λ = W * z
end

function update!(W::CompositeConeScaling{T}, s::Vector{T}, z::Vector{T}) where T <: AbstractFloat
    s_list = split_by_cone(s, W)
    z_list = split_by_cone(z, W)
    update!.(W, s_list, z_list)
end

get_num_quad_cones(W::CompositeConeScaling) = sum([isa(Wi, QuadConeScaling) for Wi in W])

function quad_cone_idxs(W::CompositeConeScaling)
    n_qc = get_num_quad_cones(W)
    start_idxs = Vector{Int64}(undef, n_qc)
    wbars = Vector{Vector{Float64}}(undef, n_qc)
    idx = 1
    for Wi in W
        if isa(Wi, QuadConeScaling)
            wbars[i] = Wi.wbar
            start_idxs[i] = idx
            idx += length(Wi.wbar)
        end
    end
    return wbars, start_idxs
end

function get_lower_kkt_block(W::CompositeConeScaling{T}; identity=false) where T
    p = dim(W)
    q = get_num_quad_cones(W)
    Kl = spzeros(T, p + 2q, p + 2q)
    i = 1  # Cone dimension offset.
    k = 1  # Quad cone index.
    j = 1
    for Wi in W
        nW = dim(Wi)
        if isa(Wi, NonnegConeScaling)
            if identity
                Kl[i:i+nW-1, i:i+nW-1] = speye(dim(Wi))
            else
                Kl[i:i+nW-1, i:i+nW-1] = sparse(Diagonal(Wi.w.^2))
            end
        elseif isa(Wi, QuadConeScaling)
            if identity
                u = 1e-20*ones(T, dim(Wi))
                v = 1e-20*ones(T, dim(Wi))
                η2 = 1.
                D = speye(T, dim(Wi))
            else
                w = Wi.w
                η2 = Wi.η^2
                w0 = w[1]
                w1 = w[2:end]
                α = 1 + w0 + w1'*w1/(1 + w0)
                β = 1 + 2/(1 + w0) + w1'*w1/(1 + w0)^2
                d = .5*(w0^2 + w1'*w1*(1 - α^2/(1 + w1'*w1*β)))
                u0 = sqrt(w0^2 + w1'*w1 - d)
                u1 = α/u0
                v1 = sqrt(pos(u1^2 - β))
                u = [u0; u1*w1]
                v = [0; v1*w1] 
                D = sparse(Diagonal([d; ones(T, nW-1)]))
            end

            full_block = true
            if full_block && !identity
                Wm = Wi.η*[w0  w1'; w1  speye(nW-1) + w1*w1'/(w0+1)]
                W2 = Wm'*Wm
                Kl[i:i+nW-1, i:i+nW-1] = W2
                Kl[p+k, p+k] = 1.
                Kl[p+k+1, p+k+1] = 1.
            else
                Kl[i:i+nW-1, i:i+nW-1] = η2*D
                Kl[p+k, i:i+nW-1] = η2*v'
                Kl[i:i+nW-1, p+k] = η2*v
                Kl[p+k+1, i:i+nW-1] = η2*u'
                Kl[i:i+nW-1, p+k+1] = η2*u
                Kl[p+k, p+k] = η2
                Kl[p+k+1, p+k+1] = -η2
            end
            k += 2
            j += nW
        else
            error("Unrecognized cone.")
        end
        i += nW
    end
    return Kl
end

function split_by_cone(x::Vector{T}, C::Union{CompositeCone{T}, CompositeConeScaling{T}}) where T
    j = 0
    x_list = []
    for Ci in C
        push!(x_list, x[j+1:j+dim(Ci)])
        j += dim(Ci)
    end
    return x_list
end
