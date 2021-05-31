include("../src/SocpSolver.jl")
using .SocpSolver
import Convex
using Test

@time @testset "Convex Problem Depot tests" begin
    Convex.ProblemDepot.run_tests(;  exclude=[r"mip", r"exp", r"sdp"]) do problem
        Convex.solve!(problem, () -> SocpSolver.Optimizer(verbose=false))
    end
end;
