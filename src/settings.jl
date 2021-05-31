@with_kw struct Settings{T <: AbstractFloat}
    max_iters::Int64 = 100
    max_step::T = 1.0
    step_param::T = 0.99
    min_centering::T = 1e-5
    init_scaling_param::T = 1.
    verbose::Bool = true
    prob_equil_iters::Int64 = 3

    # Termination criteria
    tol::T = 1e-8
    max_non_improv_iters::Int64 = 5
    non_improv_ratio_thres::T = 1.

    # Linear algebra
    prim_reg::T = 1e-14
    dual_reg::T = 1e-14
    init_reg::T = 1e-9
    trust_reg_scaling::T = 1.
    kkt_equil_iters::Int64 = 1
    max_iters_ir::Int64 = 3
    min_ref_ratio_ir::T = 1e-1
    min_corr_ratio_ir::T = 1.
    reg_decrease_param::T = 10.
    reg_increase_param::T = 10.
    great_ref_ratio::T = 1e4
    min_reg = 1e-11
    max_reg = 1e-6
    max_trust_reg::T = 1e-6
    min_step::T = 1e-3

    @assert max_iters > 0
    @assert tol > 0
    @assert step_param > 0
    @assert min_centering > 0
    @assert init_scaling_param > 0
    @assert prob_equil_iters ≥ 0
    @assert tol ≥ 0
    @assert max_non_improv_iters ≥ 0
    @assert non_improv_ratio_thres ≥ 0
end
