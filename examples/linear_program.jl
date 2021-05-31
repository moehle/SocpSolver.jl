using LinearAlgebra
using SparseArrays
using Random
Random.seed!(2)
include("../src/SocpSolver.jl")
using .SocpSolver

A = sparse(randn(2, 5))
c = rand(5)
b = A*randn(5)
G = -sparse(Diagonal(ones(5)))
h = zeros(5)
prob = LinearProgram(A, G, c, b, h)
sol, stats = solve(prob, Settings());
