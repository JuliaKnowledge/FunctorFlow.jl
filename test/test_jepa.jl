# ============================================================================
# test_jepa.jl — Tests for coalgebra, JEPA, and energy modules
# ============================================================================

@testset "Coalgebra Module" begin
    @testset "Coalgebra type" begin
        c = Coalgebra(:WorldModel, :LatentState, :dynamics)
        @test c.name == :WorldModel
        @test c.state == :LatentState
        @test c.transition == :dynamics
        @test c.functor_type == :identity

        c2 = Coalgebra(:Stoch, :S, :trans; functor_type=:distribution,
                        description="Stochastic model")
        @test c2.functor_type == :distribution
        @test c2.description == "Stochastic model"
    end

    @testset "CoalgebraMorphism type" begin
        cm = CoalgebraMorphism(:enc, :obs_coalg, :repr_coalg, :encoder)
        @test cm.name == :enc
        @test cm.source == :obs_coalg
        @test cm.target == :repr_coalg
        @test cm.hom == :encoder
    end

    @testset "FinalCoalgebraWitness type" begin
        w = FinalCoalgebraWitness(:optimal, :final_coalg, :anamorphism)
        @test w.name == :optimal
        @test w.carrier == :final_coalg
        @test w.desc_map == :anamorphism
        @test w.is_isomorphism == true  # Lambek default
    end

    @testset "Bisimulation type" begin
        b = Bisimulation(:equiv, :coalg_a, :coalg_b, :R)
        @test b.name == :equiv
        @test b.coalgebra_a == :coalg_a
        @test b.coalgebra_b == :coalg_b
        @test b.relation == :R
    end

    @testset "StochasticCoalgebra type" begin
        sc = StochasticCoalgebra(:noisy, :State, :stoch_trans;
                                  distribution_type=:gaussian)
        @test sc.name == :noisy
        @test sc.distribution_type == :gaussian
    end

    @testset "Diagram integration" begin
        D = Diagram(:CoalgTest)
        add_object!(D, :Z; kind=:latent_state)
        add_morphism!(D, :dynamics, :Z, :Z)

        c = add_coalgebra!(D, :world_model;
                           state=:Z, transition=:dynamics,
                           description="Test coalgebra")
        @test c.name == :world_model

        coalgebras = get_coalgebras(D)
        @test haskey(coalgebras, :world_model)
        @test coalgebras[:world_model].state == :Z
    end

    @testset "Bisimulation in diagram" begin
        D = Diagram(:BisimTest)
        add_object!(D, :Z1; kind=:latent_state)
        add_object!(D, :Z2; kind=:latent_state)
        add_morphism!(D, :d1, :Z1, :Z1)
        add_morphism!(D, :d2, :Z2, :Z2)
        add_morphism!(D, :R, :Z1, :Z2)

        add_coalgebra!(D, :c1; state=:Z1, transition=:d1)
        add_coalgebra!(D, :c2; state=:Z2, transition=:d2)
        b = add_bisimulation!(D, :equiv; coalgebra_a=:c1, coalgebra_b=:c2, relation=:R)

        bisims = get_bisimulations(D)
        @test haskey(bisims, :equiv)
        @test bisims[:equiv].coalgebra_a == :c1
    end

    @testset "World model block" begin
        D = world_model_block()
        @test D.name == :WorldModel
        @test haskey(D.objects, :Observation)
        @test haskey(D.objects, :LatentState)
        @test haskey(D.operations, :encode)
        @test haskey(D.operations, :dynamics)
        @test haskey(D.operations, :decode)
        @test haskey(D.operations, :encode_then_predict)
        @test haskey(D.operations, :autoencoder)

        coalgebras = get_coalgebras(D)
        @test haskey(coalgebras, :coalgebra)
        @test coalgebras[:coalgebra].state == :LatentState

        # Custom config
        D2 = world_model_block(; name=:MyWorld,
                                 state_object=:S, observation_object=:O,
                                 latent_object=:Z)
        @test D2.name == :MyWorld
        @test haskey(D2.objects, :O)
        @test haskey(D2.objects, :Z)
    end

    @testset "Pretty printing" begin
        c = Coalgebra(:WM, :Z, :d)
        s = sprint(show, c)
        @test contains(s, "Coalgebra")
        @test contains(s, "WM")

        cm = CoalgebraMorphism(:e, :a, :b, :h)
        s = sprint(show, cm)
        @test contains(s, "CoalgebraMorphism")

        w = FinalCoalgebraWitness(:opt, :fc, :ana)
        s = sprint(show, w)
        @test contains(s, "Lambek")

        b = Bisimulation(:eq, :a, :b, :R)
        s = sprint(show, b)
        @test contains(s, "Bisimulation")

        sc = StochasticCoalgebra(:n, :S, :t; distribution_type=:categorical)
        s = sprint(show, sc)
        @test contains(s, "categorical")
    end
end

@testset "JEPA Module" begin
    @testset "JEPA block" begin
        D = jepa_block()
        @test D.name == :JEPA

        # Check objects
        @test haskey(D.objects, :Observation)
        @test haskey(D.objects, :Target)
        @test haskey(D.objects, :ContextRepr)
        @test haskey(D.objects, :TargetRepr)

        # Check morphisms
        @test haskey(D.operations, :context_encoder)
        @test haskey(D.operations, :target_encoder)
        @test haskey(D.operations, :predictor)
        @test haskey(D.operations, :prediction_path)

        # Check loss
        @test haskey(D.losses, :prediction_loss)
        loss = D.losses[:prediction_loss]
        @test loss.comparator == :l2
        @test :JEPA in values(loss.metadata)

        # Check coalgebra
        coalgebras = get_coalgebras(D)
        @test haskey(coalgebras, :jepa_dynamics)

        # Check ports
        @test haskey(D.ports, :context_input)
        @test haskey(D.ports, :target_input)
        @test haskey(D.ports, :prediction)
        @test haskey(D.ports, :loss)
        @test D.ports[:context_input].direction == INPUT
        @test D.ports[:loss].direction == OUTPUT
    end

    @testset "JEPA custom config" begin
        D = jepa_block(; name=:ImageJEPA,
                         observation_object=:Patches,
                         target_object=:MaskedPatches,
                         comparator=:cosine)
        @test D.name == :ImageJEPA
        @test haskey(D.objects, :Patches)
        @test haskey(D.objects, :MaskedPatches)
        @test D.losses[:prediction_loss].comparator == :cosine
    end

    @testset "H-JEPA block" begin
        D = hjepa_block()
        @test D.name == :HJEPA

        # Should have included levels
        @test haskey(D.objects, :fine__Obs_fine)
        @test haskey(D.objects, :coarse__Obs_coarse)

        # Abstraction morphism between levels
        @test haskey(D.operations, :abstract_fine_to_coarse)

        # Ports
        @test haskey(D.ports, :input)
        @test haskey(D.ports, :fine_repr)
        @test haskey(D.ports, :coarse_repr)
    end

    @testset "H-JEPA three levels" begin
        D = hjepa_block(; levels=[:fine, :medium, :coarse])
        @test D.name == :HJEPA

        # Should have 3 levels
        @test haskey(D.objects, :fine__Obs_fine)
        @test haskey(D.objects, :medium__Obs_medium)
        @test haskey(D.objects, :coarse__Obs_coarse)

        # Two abstraction morphisms
        @test haskey(D.operations, :abstract_fine_to_medium)
        @test haskey(D.operations, :abstract_medium_to_coarse)
    end

    @testset "KAN-JEPA block" begin
        D = kan_jepa_block()
        @test D.name == :KanJEPA

        # Check objects
        @test haskey(D.objects, :Observation)
        @test haskey(D.objects, :Target)
        @test haskey(D.objects, :ContextRepr)
        @test haskey(D.objects, :TargetRepr)
        @test haskey(D.objects, :Neighborhood)

        # Predictor should be a Kan extension
        agg = D.operations[:predict_aggregate]
        @test agg isa KanExtension
        @test agg.direction == LEFT

        # Check loss references the aggregation directly
        @test haskey(D.losses, :prediction_loss)

        # Check coalgebra
        coalgebras = get_coalgebras(D)
        @test haskey(coalgebras, :kan_jepa_dynamics)
    end

    @testset "EMA update" begin
        target = [Float32[1.0, 2.0, 3.0]]
        online = [Float32[4.0, 5.0, 6.0]]
        ema_update!(target, online; decay=0.9)
        @test target[1] ≈ Float32[0.9 * 1.0 + 0.1 * 4.0,
                                   0.9 * 2.0 + 0.1 * 5.0,
                                   0.9 * 3.0 + 0.1 * 6.0]
    end

    @testset "Macro library registration" begin
        @test haskey(FunctorFlow.MACRO_LIBRARY, :jepa)
        @test haskey(FunctorFlow.MACRO_LIBRARY, :hjepa)
        @test haskey(FunctorFlow.MACRO_LIBRARY, :kan_jepa)
        @test haskey(FunctorFlow.MACRO_LIBRARY, :world_model)

        # Build via macro library
        D = build_macro(:jepa)
        @test D.name == :JEPA
    end

    @testset "JEPA compilation and execution" begin
        D = jepa_block()

        # Bind simple implementations
        bind_morphism!(D, :context_encoder, x -> x .* 2)
        bind_morphism!(D, :target_encoder, x -> x .* 2)
        bind_morphism!(D, :predictor, identity)

        compiled = compile_to_callable(D)
        result = FunctorFlow.run(compiled, Dict(
            :Observation => [1.0, 2.0, 3.0],
            :Target => [1.0, 2.0, 3.0],
        ))

        # Prediction path = predictor(encoder(x)) = identity(2x) = 2x
        @test result.values[:prediction_path] == [2.0, 4.0, 6.0]
        # Target encoder = 2x
        @test result.values[:target_encoder] == [2.0, 4.0, 6.0]
        # Loss should be ~0 since prediction matches target encoding
        @test result.losses[:prediction_loss] ≈ 0.0 atol=1e-10
    end

    @testset "JEPA with non-trivial obstruction" begin
        D = jepa_block()

        # Encoder and predictor that don't commute perfectly
        bind_morphism!(D, :context_encoder, x -> x .* 2)
        bind_morphism!(D, :target_encoder, x -> x .* 3)  # different!
        bind_morphism!(D, :predictor, identity)

        compiled = compile_to_callable(D)
        result = FunctorFlow.run(compiled, Dict(
            :Observation => [1.0, 2.0],
            :Target => [1.0, 2.0],
        ))

        # Prediction = identity(2x) = [2,4], Target enc = 3x = [3,6]
        # Loss = ||[2,4]-[3,6]|| = sqrt(1 + 4) = sqrt(5) (L2 norm)
        @test result.losses[:prediction_loss] > 0.0
        @test result.losses[:prediction_loss] ≈ sqrt(5.0) atol=1e-10
    end
end

@testset "Energy Module" begin
    @testset "EnergyFunction type" begin
        ef = EnergyFunction(:pred_energy, [:ContextRepr, :TargetRepr];
                            energy_type=:cosine, temperature=0.1)
        @test ef.name == :pred_energy
        @test ef.domain == [:ContextRepr, :TargetRepr]
        @test ef.energy_type == :cosine
        @test ef.temperature == 0.1
    end

    @testset "IntrinsicCost type" begin
        ic = IntrinsicCost(:pred; cost_type=:prediction, weight=2.0,
                           source_refs=[:X, :Y])
        @test ic.name == :pred
        @test ic.cost_type == :prediction
        @test ic.weight == 2.0
        @test ic.source_refs == [:X, :Y]
    end

    @testset "TrainableCost type" begin
        tc = TrainableCost(:critic; critic_morphism=:value_fn,
                           weight=0.5, lookahead=3, discount=0.95)
        @test tc.name == :critic
        @test tc.critic_morphism == :value_fn
        @test tc.lookahead == 3
        @test tc.discount == 0.95
    end

    @testset "CostModule type" begin
        ic = IntrinsicCost(:pred; cost_type=:prediction)
        tc = TrainableCost(:critic)
        cm = CostModule(:total; intrinsic_costs=[ic], trainable_costs=[tc])
        @test cm.name == :total
        @test length(cm.intrinsic_costs) == 1
        @test length(cm.trainable_costs) == 1
    end

    @testset "Configurator type" begin
        cfg = Configurator(:ctrl;
                           cost_weights=Dict(:pred => 1.0, :var => 0.1),
                           module_configs=Dict(:encoder => "large"))
        @test cfg.name == :ctrl
        @test cfg.cost_weights[:pred] == 1.0
        @test cfg.module_configs[:encoder] == "large"
    end

    @testset "CollapsePreventionStrategy enum" begin
        @test EMA_TARGET isa CollapsePreventionStrategy
        @test CONTRASTIVE isa CollapsePreventionStrategy
        @test VICREG isa CollapsePreventionStrategy
        @test BARLOW_TWINS isa CollapsePreventionStrategy
        @test WHITENING isa CollapsePreventionStrategy
    end

    @testset "Built-in energy functions" begin
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]

        # L2 energy
        e_l2 = energy_l2(x, y)
        @test e_l2 ≈ 27.0  # (3² + 3² + 3²)
        @test energy_l2(x, x) ≈ 0.0 atol=1e-10

        # Cosine energy
        e_cos = energy_cosine(x, x)
        @test e_cos ≈ 0.0 atol=1e-6

        # Cosine of orthogonal vectors
        e_orth = energy_cosine([1.0, 0.0], [0.0, 1.0])
        @test e_orth ≈ 1.0 atol=1e-6

        # Smooth L1
        e_sl1 = energy_smooth_l1(x, y)
        @test e_sl1 > 0.0
        @test energy_smooth_l1(x, x) ≈ 0.0 atol=1e-10
    end

    @testset "Variance regularization" begin
        # Matrix where each column is a sample
        # High variance: should have low penalty
        Z = Float64[1 -1; 2 -2; 3 -3]  # 3 dims, 2 samples
        var_loss = variance_regularization(Z; gamma=1.0)
        # Variances are large, so penalty should be small or zero
        @test var_loss isa Float64

        # Collapsed representations: all same
        Z_collapsed = ones(3, 4)
        var_loss_collapsed = variance_regularization(Z_collapsed; gamma=1.0)
        @test var_loss_collapsed > 0.0  # should penalize
    end

    @testset "Covariance regularization" begin
        Z = Float64[1 -1; 0 0; -1 1]  # 3 dims, 2 samples
        cov_loss = covariance_regularization(Z)
        @test cov_loss isa Float64
        @test cov_loss >= 0.0

        # Identity-like covariance should have minimal off-diagonal
        Z_decorr = Float64[1 0 0; 0 1 0; 0 0 1]'  # diagonal
        cov_decorr = covariance_regularization(Z_decorr)
        @test cov_decorr >= 0.0
    end

    @testset "Diagram integration" begin
        D = Diagram(:EnergyTest)
        add_object!(D, :X; kind=:representation)
        add_object!(D, :Y; kind=:representation)

        ef = add_energy_function!(D, :compat;
                                  domain=[:X, :Y], energy_type=:cosine)
        @test ef.name == :compat

        energy_fns = get_energy_functions(D)
        @test haskey(energy_fns, :compat)
        @test energy_fns[:compat].energy_type == :cosine

        ic = IntrinsicCost(:pred; cost_type=:prediction)
        cm = add_cost_module!(D, :cost; intrinsic_costs=[ic])
        @test cm.name == :cost

        cost_mods = get_cost_modules(D)
        @test haskey(cost_mods, :cost)
    end

    @testset "Energy block builder" begin
        D = energy_block()
        @test D.name == :EnergyCost
        @test haskey(D.objects, :Prediction)
        @test haskey(D.objects, :Target)

        energy_fns = get_energy_functions(D)
        @test haskey(energy_fns, :energy)

        cost_mods = get_cost_modules(D)
        @test haskey(cost_mods, :cost)
    end

    @testset "Energy block with VICReg" begin
        D = energy_block(; name=:VICRegCost,
                           variance_weight=0.5,
                           covariance_weight=0.1)
        @test D.name == :VICRegCost

        cost_mods = get_cost_modules(D)
        cm = cost_mods[:cost]
        @test length(cm.intrinsic_costs) == 3  # prediction + variance + covariance
        @test cm.intrinsic_costs[2].cost_type == :variance
        @test cm.intrinsic_costs[3].cost_type == :covariance
    end

    @testset "Macro library registration" begin
        @test haskey(FunctorFlow.MACRO_LIBRARY, :energy)
        D = build_macro(:energy)
        @test D.name == :EnergyCost
    end

    @testset "Pretty printing" begin
        ef = EnergyFunction(:e, [:X, :Y]; energy_type=:l2)
        s = sprint(show, ef)
        @test contains(s, "EnergyFunction")
        @test contains(s, "l2")

        ic = IntrinsicCost(:p; cost_type=:prediction)
        s = sprint(show, ic)
        @test contains(s, "IntrinsicCost")

        tc = TrainableCost(:c)
        s = sprint(show, tc)
        @test contains(s, "TrainableCost")

        cm = CostModule(:total; intrinsic_costs=[ic], trainable_costs=[tc])
        s = sprint(show, cm)
        @test contains(s, "CostModule")
        @test contains(s, "1 IC")
        @test contains(s, "1 TC")

        cfg = Configurator(:ctrl)
        s = sprint(show, cfg)
        @test contains(s, "Configurator")
    end
end

@testset "JEPA Proof Interface" begin
    @testset "JEPA certificate generation" begin
        D = jepa_block()
        cert = render_jepa_certificate(D)

        @test contains(cert, "FunctorFlowProofs.Generated")
        @test contains(cert, "exportedDiagram")
        @test contains(cert, "coalgebra_")
        @test contains(cert, "jepa_prediction_loss_is_obstruction")
        @test contains(cert, "jepa_exact_implies_coalgebra_morphism")
    end

    @testset "KAN-JEPA certificate" begin
        D = kan_jepa_block()
        cert = render_jepa_certificate(D)

        @test contains(cert, "CoalgebraDecl")
        @test contains(cert, "kan_jepa_dynamics")
    end

    @testset "World model certificate" begin
        D = world_model_block()
        cert = render_jepa_certificate(D)

        @test contains(cert, "coalgebra_coalgebra")
        @test contains(cert, "LatentState")
    end

    @testset "Certificate with bisimulation" begin
        D = jepa_block()
        add_coalgebra!(D, :c2; state=:TargetRepr, transition=:target_encoder)
        add_bisimulation!(D, :enc_equiv;
                          coalgebra_a=:jepa_dynamics, coalgebra_b=:c2,
                          relation=:predictor)
        cert = render_jepa_certificate(D)

        @test contains(cert, "BisimulationDecl")
        @test contains(cert, "bisim_enc_equiv")
        @test contains(cert, "bisimilar_iff_final_coalgebra_equal")
    end

    @testset "Certificate with energy functions" begin
        D = jepa_block()
        add_energy_function!(D, :compat;
                             domain=[:ContextRepr, :TargetRepr],
                             energy_type=:l2)
        cert = render_jepa_certificate(D)

        @test contains(cert, "EnergyDecl")
        @test contains(cert, "energy_nonneg")
    end
end
