const MOI = MathOptInterface
const CI = MOI.ConstraintIndex
const VI = MOI.VariableIndex

const MOIU = MOI.Utilities

struct WrapperSolution
    ret_val::Status
    primal::Vector{Float64}
    dual_eq::Vector{Float64}
    dual_ineq::Vector{Float64}
    slack::Vector{Float64}
    objective_value::Float64
    dual_objective_value::Float64
    objective_constant::Float64
    solve_time::Float64
end

WrapperSolution() = WrapperSolution(unsolved_status, Float64[], Float64[], Float64[],
                      Float64[], NaN, NaN, NaN, NaN)

# Used to build the data with allocate-load during `copy_to`.
mutable struct ModelData
    m::Int # Number of rows/constraints
    n::Int # Number of cols/variables
    IA::Vector{Int} # List of conic rows
    JA::Vector{Int} # List of conic cols
    VA::Vector{Float64} # List of conic coefficients
    b::Vector{Float64} # List of conic coefficients
    IG::Vector{Int} # List of equality rows
    JG::Vector{Int} # List of equality cols
    VG::Vector{Float64} # List of equality coefficients
    h::Vector{Float64} # List of equality coefficients
    objective_constant::Float64 # The objective is min c'x + objective_constant
    c::Vector{Float64}
end

# This is tied to our internal representation
mutable struct ConeData
    f::Int # number of linear equality constraints
    l::Int # length of LP cone
    q::Int # length of SOC cone
    qa::Vector{Int} # array of second-order cone constraints
    # The following four field store model information needed to compute `ConstraintPrimal` and `ConstraintDual`
    eqnrows::Dict{Int, Int}   # The number of rows of Zeros
    ineqnrows::Dict{Int, Int} # The number of rows of each vector sets except Zeros
    function ConeData()
        new(0, 0, 0, Int[], Dict{Int, UnitRange{Int}}(), Dict{Int, UnitRange{Int}}())
    end
end

mutable struct Optimizer <: MOI.AbstractOptimizer
    cone::ConeData
    maxsense::Bool
    data::Union{Nothing, ModelData} # only non-Nothing between MOI.copy_to and MOI.optimize!
    sol::WrapperSolution
    silent::Bool
    options::Dict{Symbol, Any}
    function Optimizer(; kwargs...)
        optimizer = new(ConeData(), false, nothing, WrapperSolution(), false, Dict{Symbol, Any}())
        for (key, value) in kwargs
            MOI.set(optimizer, MOI.RawParameter(String(key)), value)
        end
        return optimizer
    end
end

MOI.get(::Optimizer, ::MOI.SolverName) = "CoSo"

function MOI.set(optimizer::Optimizer, param::MOI.RawParameter, value)
    if !(param.name isa String)
        Base.depwarn(
            "passing `$(param.name)` to `MOI.RawParameter` as type " *
            "`$(typeof(param.name))` is deprecated. Use a string instead.",
            Symbol("MOI.set")
        )
    end
    optimizer.options[Symbol(param.name)] = value
end
function MOI.get(optimizer::Optimizer, param::MOI.RawParameter)
    if !(param.name isa String)
        Base.depwarn(
            "passing $(param.name) to `MOI.RawParameter` as type " *
            "$(typeof(param.name)) is deprecated. Use a string instead.",
            Symbol("MOI.get")
        )
    end
    # TODO: This gives a poor error message if the name of the parameter is invalid.
    return optimizer.options[Symbol(param.name)]
end

MOI.supports(::Optimizer, ::MOI.Silent) = true
function MOI.set(optimizer::Optimizer, ::MOI.Silent, value::Bool)
    optimizer.silent = value
end
MOI.get(optimizer::Optimizer, ::MOI.Silent) = optimizer.silent

function MOI.is_empty(optimizer::Optimizer)
    !optimizer.maxsense && optimizer.data === nothing
end

function MOI.empty!(optimizer::Optimizer)
    optimizer.maxsense = false
    optimizer.data = nothing # It should already be nothing except if an error is thrown inside copy_to
    optimizer.sol = WrapperSolution()
end

MOIU.supports_allocate_load(::Optimizer, copy_names::Bool) = !copy_names

function MOI.supports(::Optimizer,
                      ::Union{MOI.ObjectiveSense,
                              MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}})
    return true
end

function MOI.supports_constraint(::Optimizer,
                                 ::Type{MOI.VectorAffineFunction{Float64}},
                                 ::Type{<:Union{MOI.Zeros, MOI.Nonnegatives,
                                                MOI.SecondOrderCone}})
    return true
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike; kws...)
    return MOIU.automatic_copy_to(dest, src; kws...)
end

# Computes cone dimensions
constroffset(cone::ConeData, ci::CI{<:MOI.AbstractFunction, MOI.Zeros}) = ci.value
function _allocate_constraint(cone::ConeData, f, s::MOI.Zeros)
    ci = cone.f
    cone.f += MOI.dimension(s)
    ci
end
constroffset(cone::ConeData, ci::CI{<:MOI.AbstractFunction, MOI.Nonnegatives}) = ci.value
function _allocate_constraint(cone::ConeData, f, s::MOI.Nonnegatives)
    ci = cone.l
    cone.l += MOI.dimension(s)
    ci
end
constroffset(cone::ConeData, ci::CI{<:MOI.AbstractFunction, <:MOI.SecondOrderCone}) = cone.l + ci.value
function _allocate_constraint(cone::ConeData, f, s::MOI.SecondOrderCone)
    push!(cone.qa, s.dimension)
    ci = cone.q
    cone.q += MOI.dimension(s)
    ci
end
constroffset(optimizer::Optimizer, ci::CI) = constroffset(optimizer.cone, ci::CI)
function MOIU.allocate_constraint(optimizer::Optimizer, f::F, s::S) where {F <: MOI.AbstractFunction, S <: MOI.AbstractSet}
    CI{F, S}(_allocate_constraint(optimizer.cone, f, s))
end

# Build constraint matrix
output_index(t::MOI.VectorAffineTerm) = t.output_index
variable_index_value(t::MOI.ScalarAffineTerm) = t.variable_index.value
variable_index_value(t::MOI.VectorAffineTerm) = variable_index_value(t.scalar_term)
coefficient(t::MOI.ScalarAffineTerm) = t.coefficient
coefficient(t::MOI.VectorAffineTerm) = coefficient(t.scalar_term)
constrrows(s::MOI.AbstractVectorSet) = 1:MOI.dimension(s)
constrrows(optimizer::Optimizer, ci::CI{<:MOI.AbstractVectorFunction, MOI.Zeros}) = 1:optimizer.cone.eqnrows[constroffset(optimizer, ci)]
constrrows(optimizer::Optimizer, ci::CI{<:MOI.AbstractVectorFunction, <:MOI.AbstractVectorSet}) = 1:optimizer.cone.ineqnrows[constroffset(optimizer, ci)]
matrix(data::ModelData, s::MOI.Zeros) = data.b, data.IA, data.JA, data.VA
matrix(data::ModelData, s::Union{MOI.Nonnegatives, MOI.SecondOrderCone}) = data.h, data.IG, data.JG, data.VG
matrix(optimizer::Optimizer, s) = matrix(optimizer.data, s)
function MOIU.load_constraint(optimizer::Optimizer, ci::MOI.ConstraintIndex, f::MOI.VectorAffineFunction, s::MOI.AbstractVectorSet)
    func = MOIU.canonical(f)
    I = Int[output_index(term) for term in func.terms]
    J = Int[variable_index_value(term) for term in func.terms]
    V = Float64[-coefficient(term) for term in func.terms]
    offset = constroffset(optimizer, ci)
    rows = constrrows(s)
    if s isa MOI.Zeros
        optimizer.cone.eqnrows[offset] = length(rows)
    else
        optimizer.cone.ineqnrows[offset] = length(rows)
    end
    i = offset .+ rows
    # Our solver format is b - Ax ∈ cone
    # so minus=false for b and minus=true for A
    b, Is, Js, Vs = matrix(optimizer, s)
    b[i] .= f.constants
    append!(Is, offset .+ I)
    append!(Js, J)
    append!(Vs, V)
end

function MOIU.allocate_variables(optimizer::Optimizer, nvars::Integer)
    optimizer.cone = ConeData()
    VI.(1:nvars)
end

function MOIU.load_variables(optimizer::Optimizer, nvars::Integer)
    cone = optimizer.cone
    m = cone.l + cone.q
    IA = Int[]
    JA = Int[]
    VA = Float64[]
    b = zeros(cone.f)
    IG = Int[]
    JG = Int[]
    VG = Float64[]
    h = zeros(m)
    c = zeros(nvars)
    optimizer.data = ModelData(m, nvars, IA, JA, VA, b, IG, JG, VG, h, 0., c)
end

function MOIU.allocate(optimizer::Optimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    optimizer.maxsense = sense == MOI.MAX_SENSE
end
function MOIU.allocate(::Optimizer, ::MOI.ObjectiveFunction,
                       ::MOI.ScalarAffineFunction{Float64})
end

function MOIU.load(::Optimizer, ::MOI.ObjectiveSense, ::MOI.OptimizationSense)
end
function MOIU.load(optimizer::Optimizer, ::MOI.ObjectiveFunction,
                   f::MOI.ScalarAffineFunction)
    c0 = Vector(sparsevec(variable_index_value.(f.terms), coefficient.(f.terms),
                          optimizer.data.n))
    optimizer.data.objective_constant = f.constant
    optimizer.data.c = optimizer.maxsense ? -c0 : c0
    return nothing
end

function MOI.optimize!(optimizer::Optimizer)
    if optimizer.data === nothing
        # optimize! has already been called and no new model has been copied
        return
    end
    cone = optimizer.cone
    m = optimizer.data.m
    n = optimizer.data.n
    A = sparse(optimizer.data.IA, optimizer.data.JA, optimizer.data.VA, cone.f, n)
    b = optimizer.data.b
    G = sparse(optimizer.data.IG, optimizer.data.JG, optimizer.data.VG, m, n)
    h = optimizer.data.h
    objective_constant = optimizer.data.objective_constant
    c = optimizer.data.c

    C = get_cone_from_moi_data(cone)
    problem = Problem(A, G, c, b, h, C)

    settings = Settings{Float64}(; optimizer.options...)
    sol, stats = solve(problem, settings)
    ret_val = sol.status
    solve_time = 0.  # TODO
    primal    = sol.x
    dual_eq   = sol.y
    dual_ineq = sol.z
    slack     = sol.s
    objective_value = (optimizer.maxsense ? -1 : 1) * sol.optval
    dual_objective_value = (optimizer.maxsense ? -1 : 1) * sol.optval  # TODO
    optimizer.sol = WrapperSolution(ret_val, primal, dual_eq, dual_ineq, slack, objective_value,
                            dual_objective_value, objective_constant, solve_time)
end

MOI.get(optimizer::Optimizer, ::MOI.SolveTime) = optimizer.sol.solve_time
function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    flag = optimizer.sol.ret_val
    if flag == unsolved_status
        return "Optimize not called"
    elseif flag == optimal_status
        return "Problem solved to optimality"
    elseif flag == infeasible_status
        return "Found certificate of primal infeasibility"
    elseif flag == unbounded_status
        return "Found certificate of dual infeasibility"
    elseif flag == max_iters_status
        return "Maximum number of iterations reached"
    elseif flag == numerical_error_status
        return "Numerical error"
    else
        return "Unknown problem in solver"
    end
end

# Implements getter for result value and statuses
function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    flag = optimizer.sol.ret_val
    if flag == unsolved_status
        return MOI.OPTIMIZE_NOT_CALLED
    elseif flag == optimal_status
        return MOI.OPTIMAL
    elseif flag == infeasible_status
        return MOI.INFEASIBLE
    elseif flag == unbounded_status
        return MOI.DUAL_INFEASIBLE
    elseif flag == max_iters_status
        return MOI.ITERATION_LIMIT
    elseif flag == numerical_error_status
        return MOI.NUMERICAL_ERROR
    else
        error("Unrecognized solver status flag: $flag.")
    end
end

function MOI.get(optimizer::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(optimizer, attr)
    value = optimizer.sol.objective_value
    if !MOIU.is_ray(MOI.get(optimizer, MOI.PrimalStatus()))
        value += optimizer.sol.objective_constant
    end
    return value
end
function MOI.get(optimizer::Optimizer, attr::MOI.DualObjectiveValue)
    MOI.check_result_index_bounds(optimizer, attr)
    value = optimizer.sol.dual_objective_value
    if !MOIU.is_ray(MOI.get(optimizer, MOI.DualStatus()))
        value += optimizer.sol.objective_constant
    end
    return value
end

function MOI.get(optimizer::Optimizer, attr::MOI.PrimalStatus)
    if attr.N > MOI.get(optimizer, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    flag = optimizer.sol.ret_val
    if flag == optimal_status
        return MOI.FEASIBLE_POINT
    elseif flag == infeasible_status
        return MOI.INFEASIBLE_POINT
    elseif flag == unbounded_status
        return MOI.INFEASIBILITY_CERTIFICATE
    elseif (flag == max_iters_status) || (flag == numerical_error_status)
        return MOI.UNKNOWN_RESULT_STATUS
    else
        return MOI.OTHER_RESULT_STATUS
    end
end
function MOI.get(optimizer::Optimizer, attr::MOI.VariablePrimal, vi::VI)
    MOI.check_result_index_bounds(optimizer, attr)
    optimizer.sol.primal[vi.value]
end
MOI.get(optimizer::Optimizer, a::MOI.VariablePrimal, vi::Vector{VI}) = MOI.get.(optimizer, Ref(a), vi)
function MOI.get(optimizer::Optimizer, attr::MOI.ConstraintPrimal, ci::CI{<:MOI.AbstractFunction, MOI.Zeros})
    MOI.check_result_index_bounds(optimizer, attr)
    rows = constrrows(optimizer, ci)
    return zeros(length(rows))
end
function MOI.get(optimizer::Optimizer, attr::MOI.ConstraintPrimal, ci::CI{<:MOI.AbstractFunction, S}) where S <: MOI.AbstractSet
    MOI.check_result_index_bounds(optimizer, attr)
    offset = constroffset(optimizer, ci)
    rows = constrrows(optimizer, ci)
    return optimizer.sol.slack[offset .+ rows]
end

function MOI.get(optimizer::Optimizer, attr::MOI.DualStatus)
    if attr.N > MOI.get(optimizer, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    flag = optimizer.sol.ret_val
    if flag == optimal_status
        return MOI.FEASIBLE_POINT
    elseif flag == infeasible_status
        return MOI.INFEASIBILITY_CERTIFICATE
    elseif flag == unbounded_status
        return MOI.INFEASIBLE_POINT
    elseif (flag == max_iters_status) || (flag == numerical_error_status)
        return MOI.UNKNOWN_RESULT_STATUS
    else
        return MOI.OTHER_RESULT_STATUS
    end
end
_dual(optimizer, ci::CI{<:MOI.AbstractFunction, <:MOI.Zeros}) = optimizer.sol.dual_eq
_dual(optimizer, ci::CI) = optimizer.sol.dual_ineq
function MOI.get(optimizer::Optimizer, attr::MOI.ConstraintDual, ci::CI{<:MOI.AbstractFunction, S}) where S <: MOI.AbstractSet
    MOI.check_result_index_bounds(optimizer, attr)
    offset = constroffset(optimizer, ci)
    rows = constrrows(optimizer, ci)
    return _dual(optimizer, ci)[offset .+ rows]
end

MOI.get(optimizer::Optimizer, ::MOI.ResultCount) = 1


function get_cone_from_moi_data(cone_data::ConeData)
    cones = SimpleCone{Float64}[]
    cone_data.l > 0 && push!(cones, NonnegCone{Float64}(cone_data.l))
    append!(cones, [QuadCone{Float64}(qi) for qi in cone_data.qa])
    return CompositeCone(cones)
end
