# ============================================================================
# test_energy.jl — Energy functions and the cost-module execution path
# ============================================================================

using Test
using FunctorFlow

@testset "Energy functions (numeric)" begin
    # --- L2 / cosine / smooth-L1 builtins ---
    @test FunctorFlow.energy_l2([1.0, 2.0], [1.0, 2.0]) ≈ 0.0
    @test FunctorFlow.energy_l2([1.0, 0.0], [0.0, 0.0]) ≈ 1.0
    @test FunctorFlow.energy_cosine([1.0, 0.0], [1.0, 0.0]) ≈ 0.0 atol = 1e-6
    @test FunctorFlow.energy_cosine([1.0, 0.0], [0.0, 1.0]) ≈ 1.0 atol = 1e-6
    @test FunctorFlow.energy_smooth_l1([0.0], [0.0]) ≈ 0.0

    X = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    Y = X .+ 0.01

    # --- VICReg: invariance ~0 when X≈Y but variance/cov terms keep it >0 ---
    @test energy_vicreg(X, X) ≥ 0.0
    @test energy_vicreg(X, Y) > 0.0
    # Increasing the invariance gap increases the energy.
    @test energy_vicreg(X, X .+ 1.0; var_coeff=0.0, cov_coeff=0.0) >
          energy_vicreg(X, X .+ 0.1; var_coeff=0.0, cov_coeff=0.0)

    # --- Barlow Twins: identical batch → near-identity cross-corr → small ---
    @test energy_barlow_twins(X, X) ≥ 0.0
    @test isfinite(energy_barlow_twins(X, Y))
    # Single-sample inputs fall back to L2.
    @test energy_barlow_twins([1.0, 2.0], [1.0, 2.0]) ≈ 0.0

    # --- Contrastive (InfoNCE): matched columns give lower loss than shuffled ---
    @test energy_contrastive(X, X) ≈ 0.0 atol = 1e-3
    @test energy_contrastive(X, X) < energy_contrastive(X, reverse(X; dims=2))
    @test energy_contrastive(X, X) ≥ 0.0

    # --- registries now include the self-supervised energies ---
    for k in (:l2, :cosine, :smooth_l1, :vicreg, :barlow_twins, :contrastive)
        @test haskey(BUILTIN_ENERGY_FUNCTIONS, k)
    end
    @test haskey(BUILTIN_REGULARIZERS, :variance)
    @test haskey(BUILTIN_REGULARIZERS, :covariance)
end

@testset "Energy / cost execution path" begin
    X = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    Y = X .+ 0.05

    D = energy_block(; config=EnergyBlockConfig(
        prediction_object=:Prediction, target_object=:Target,
        energy_type=:l2, prediction_weight=1.0,
        variance_weight=1.0, covariance_weight=0.5))

    res = FunctorFlow.run(D, Dict(:Prediction => X, :Target => Y))

    # compute_energies evaluates the declared EnergyFunction against the run.
    energies = compute_energies(D, res)
    @test haskey(energies, :energy)
    @test energies[:energy] ≈ FunctorFlow.energy_l2(X, Y)

    # evaluate_energy directly on the EnergyFunction value.
    ef = first(values(get_energy_functions(D)))
    @test evaluate_energy(ef, res.values) ≈ energies[:energy]

    # compute_costs evaluates the IC+TC decomposition.
    costs = compute_costs(D, res)
    @test haskey(costs, :cost)
    comp = costs[:cost]["components"]
    @test haskey(comp, "prediction_cost")
    @test haskey(comp, "variance_cost")
    @test haskey(comp, "covariance_cost")
    @test costs[:cost]["total"] ≈ comp["prediction_cost"] + comp["variance_cost"] + comp["covariance_cost"]
    # prediction component equals weight * l2 energy
    @test comp["prediction_cost"] ≈ FunctorFlow.energy_l2(X, Y)

    # run_with_costs convenience returns consistent values.
    r2, e2, c2 = run_with_costs(D, Dict(:Prediction => X, :Target => Y))
    @test e2[:energy] ≈ energies[:energy]
    @test c2[:cost]["total"] ≈ costs[:cost]["total"]
end

@testset "Trainable cost critic wiring" begin
    X = [1.0 0.0; 0.0 1.0]
    D = Diagram(:CriticTest)
    add_object!(D, :Pred; kind=:representation)
    add_object!(D, :Tgt; kind=:representation)
    add_cost_module!(D, :cost;
        intrinsic_costs=[IntrinsicCost(:p; cost_type=:prediction, source_refs=[:Pred, :Tgt])],
        trainable_costs=[TrainableCost(:critic; weight=2.0)])

    res = ExecutionResult(Dict{Symbol,Any}(:Pred => X, :Tgt => X), Dict{Symbol,Float64}())
    # No critic supplied → trainable cost contributes 0.
    c0 = compute_costs(D, res)
    @test c0[:cost]["components"]["critic"] == 0.0
    # Critic supplied as env -> Real, weighted by 2.0.
    c1 = compute_costs(D, res; trainable_costs=Dict(:critic => env -> 3.0))
    @test c1[:cost]["components"]["critic"] ≈ 6.0

    # Unknown energy type raises a clear error.
    bad = EnergyFunction(:bad, [:Pred, :Tgt]; energy_type=:does_not_exist)
    @test_throws ArgumentError evaluate_energy(bad, res.values)
end
