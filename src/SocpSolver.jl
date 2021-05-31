module SocpSolver

export Problem, LinearProgram, QuadraticProgram, Settings, solve

import Base: size, copy, iterate, length, getindex, ∈, *, \, /
using Printf
using Parameters
using SparseArrays
using LinearAlgebra
using AMD
using MathOptInterface


include("utils.jl")
include("qdldl.jl")
include("cone.jl")
include("problem.jl")
include("types.jl")
include("scaling.jl")
include("settings.jl")
include("solution.jl")
include("equil.jl")
include("kkt.jl")
include("abs_kkt.jl")
include("term.jl")
include("ipm.jl")
include("printing.jl")
include("moi_wrapper.jl")

end
