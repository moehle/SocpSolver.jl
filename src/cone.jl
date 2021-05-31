abstract type SimpleCone{T <: AbstractFloat} end


@with_kw struct NonnegCone{T <: AbstractFloat} <: SimpleCone{T}
    dim::Int64
    @assert dim > 0
end

NonnegCone(dim::Int64) = NonnegCone{Float64}(dim)


@with_kw struct QuadCone{T <: AbstractFloat} <: SimpleCone{T}
    dim::Int64
    @assert dim > 0
end

QuadCone(dim::Int64) = QuadCone{Float64}(dim)


struct CompositeCone{T <: AbstractFloat}
    cones::Vector{SimpleCone{T}}
end

CompositeCone(C::SimpleCone{T}) where T <: AbstractFloat = CompositeCone(SimpleCone{T}[C])

iterate(C::CompositeCone, args...) = iterate(C.cones, args...)
getindex(C::CompositeCone, args...) = getindex(C.cones, args...)
length(C::CompositeCone) = length(C.cones)
size(C::CompositeCone) = (length(C),)

dim(C::NonnegCone) = C.dim
dim(C::QuadCone)  = C.dim
dim(C::CompositeCone) = sum([dim(Ci) for Ci in C]) # TODO map

elem_cone_dims(C::NonnegCone) = ones(Int64, dim(C))
elem_cone_dims(C::QuadCone) = dim(C)
elem_cone_dims(C::CompositeCone{T}) where T = vcat(Int64[], map(elem_cone_dims, C)...)

∈(x::Vector{T}, C::NonnegCone) where T <: AbstractFloat = all(x .> 0)
∈(x::Vector{T}, C::QuadCone) where T <: AbstractFloat = x[2:end]'*x[2:end] < x[1]^2 
function ∈(x::Vector{T}, C::CompositeCone) where T <: AbstractFloat
    x_list = split_by_cone(x, C)
    return all((x_list .∈ C))
end

degree(C::NonnegCone) = dim(C)
degree(C::QuadCone) = 1
degree(C::CompositeCone) = sum([degree(Ci) for Ci in C]) # TODO map

jordan(s::Vector{T}, z::Vector{T}, C::NonnegCone) where T <: AbstractFloat = s .* z
jordan(s::Vector{T}, z::Vector{T}, C::QuadCone) where T <: AbstractFloat = [s'*z; s[1]*z[2:end] + z[1]*s[2:end]]
function jordan(s::Vector{T}, z::Vector{T}, C::CompositeCone) where T <: AbstractFloat
    s_list = split_by_cone(s, C)
    z_list = split_by_cone(z, C)
    return vcat(T[], jordan.(s_list, z_list, C)...)
end

# Defined in §5.4, pg 14.
function jordan_inv(s::Vector{T}, z::Vector{T}, C::NonnegCone) where {T <: AbstractFloat}
    return s .\ z
end

function jordan_inv(u::Vector{T}, w::Vector{T}, C::QuadCone) where {T <: AbstractFloat}
    y = Vector{T}(undef, length(u))
    ρ = u[1]^2 - u[2:end]'*u[2:end]
    ν = u[2:end]'*w[2:end]
    y[1] = u[1]*w[1] - ν
    y[2:end] = (ν/u[1] - w[1])*u[2:end] + (ρ/u[1])*w[2:end]
    return y / ρ
end

function jordan_inv(s::Vector{T}, z::Vector{T}, C::CompositeCone) where {T <: AbstractFloat}
    s_list = split_by_cone(s, C)
    z_list = split_by_cone(z, C)
    return vcat(T[], [jordan_inv(si, zi, Ci) for (si, zi, Ci) in zip(s_list, z_list, C)]...)
end

get_id(C::NonnegCone) = ones(dim(C))
get_id(C::QuadCone) = [1; zeros(dim(C) - 1)]
get_id(C::CompositeCone{T}) where T = vcat(T[], get_id.(C)...)

function get_max_step(x::Vector{T}, δx::Vector{T}, C::NonnegCone) where T <: AbstractFloat
    if x in C
        return 1/pos(-minimum(δx ./ x))
    else
        return NaN
    end
end

function get_max_step(λ::Vector{T}, Δs::Vector{T}, Δz::Vector{T}, C::NonnegCone) where T <: AbstractFloat
    ρ = Δs ./ λ
    σ = Δz ./ λ
    @assert dim(C) > 0
    @assert length(ρ) > 0
    @assert length(σ) > 0
    α = 1/max(0, -minimum(ρ), -minimum(σ))
    return α
end

function get_max_step(λ::Vector{T}, Δs::Vector{T}, Δz::Vector{T}, C::QuadCone) where T <: AbstractFloat
    sqrtλJλ = sqrt(pos(λ'*(J*λ)))
    λbar = λ / sqrtλJλ
    ρ0 = 1/sqrtλJλ * λbar'*(J*Δs)
    ρ1 = 1/sqrtλJλ * (Δs[2:end] - (λbar'*(J*Δs) + Δs[1])/(λbar[1] + 1)*λbar[2:end])
    σ0 = 1/sqrtλJλ * λbar'*(J*Δz)
    σ1 = 1/sqrtλJλ * (Δz[2:end] - (λbar'*(J*Δz) + Δz[1])/(λbar[1] + 1)*λbar[2:end])
    α = 1/max(0, norm(ρ1) - ρ0, norm(σ1) - σ0)
    return α
end

function get_max_step(λ::Vector{T}, Δs::Vector{T}, Δz::Vector{T}, C::CompositeCone) where T <: AbstractFloat
    λ_list = split_by_cone(λ, C)
    Δs_list = split_by_cone(Δs, C)
    Δz_list = split_by_cone(Δz, C)
    return min(Inf, get_max_step.(λ_list, Δs_list, Δz_list, C)...)
end

function dist_along_id_ray(x::Vector{T}, C::NonnegCone{T}) where T <: AbstractFloat
    return maximum(neg.(x))
end

function dist_along_id_ray(x::Vector{T}, C::QuadCone{T}) where T <: AbstractFloat
    return pos(norm(x[2:end]) - x[1])
end

function dist_along_id_ray(x::Vector{T}, C::CompositeCone{T}) where T <: AbstractFloat
    return max(0, [dist_along_id_ray(xi, Ci) for (xi, Ci) in zip(split_by_cone(x, C), C)]...)
end

get_num_quad_cones(C::CompositeCone) = sum([isa(Ci, QuadCone) for Ci in C])
