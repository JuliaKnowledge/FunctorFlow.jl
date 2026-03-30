using FunctorFlow
using Test

@testset "V1 Features" begin
    # ==================================================================
    # A. Semantic Kernel (catlab_interop.jl)
    # ==================================================================
    @testset "Semantic Kernel" begin
        @testset "CategoricalModelObject construction" begin
            obj = CategoricalModelObject(:TestModel;
                                          ambient_category=:FinSet,
                                          semantic_laws=[(_obj -> true)])
            @test obj.name == :TestModel
            @test obj.ambient_category == :FinSet
            @test length(obj.semantic_laws) == 1
            @test obj.diagram === nothing

            # From Diagram
            D = ket_block()
            cat_obj = CategoricalModelObject(D)
            @test cat_obj.name == D.name
            @test cat_obj.diagram === D
            @test length(cat_obj.interface_ports) == length(D.ports)
        end

        @testset "to_diagram" begin
            D = ket_block()
            cat_obj = CategoricalModelObject(D)
            D2 = to_diagram(cat_obj)
            @test D2 isa FunctorFlow.Diagram
            @test D2.name == D.name
        end

        @testset "ModelMorphism" begin
            f = ModelMorphism(:f, :A, :B; functor_data=x -> x * 2)
            @test f.name == :f
            @test f.source == :A
            @test f.target == :B
            @test f.functor_data !== nothing
        end

        @testset "compose ModelMorphism" begin
            f = ModelMorphism(:f, :A, :B; functor_data=x -> x * 2)
            g = ModelMorphism(:g, :B, :C; functor_data=x -> x + 1)
            fg = FunctorFlow.compose(f, g)
            @test fg.source == :A
            @test fg.target == :C
            @test occursin("f", string(fg.name))

            # Composed functor_data preserves both as NamedTuple
            @test fg.functor_data isa NamedTuple
            @test fg.functor_data.second(fg.functor_data.first(3)) == 3 * 2 + 1

            # Incompatible morphisms
            h = ModelMorphism(:h, :X, :Y)
            @test_throws ArgumentError FunctorFlow.compose(f, h)
        end

        @testset "apply ModelMorphism" begin
            obj_A = CategoricalModelObject(:A; ambient_category=:FinSet)
            f = ModelMorphism(:f, :A, :B; functor_data=identity)
            obj_B = FunctorFlow.apply(f, obj_A)
            @test obj_B.name == :B
            @test obj_B.ambient_category == :FinSet

            # Wrong source
            obj_C = CategoricalModelObject(:C)
            @test_throws ArgumentError FunctorFlow.apply(f, obj_C)
        end

        @testset "NaturalTransformation" begin
            α = NaturalTransformation(:alpha, :F, :G;
                                       components=Dict(:X => identity, :Y => x -> x + 1))
            @test α.name == :alpha
            @test α.source_functor == :F
            @test α.target_functor == :G
            @test length(α.components) == 2
        end

        @testset "is_natural" begin
            F = ModelMorphism(:F, :A, :B; functor_data=x -> x * 2)
            G = ModelMorphism(:G, :A, :B; functor_data=x -> x + 10)
            obj_A = CategoricalModelObject(:A)

            α = NaturalTransformation(:alpha, :F, :G;
                                       components=Dict(:A => x -> x + 10 - x * 2))
            @test FunctorFlow.is_natural(α, F, G, [obj_A])

            # Missing component
            α2 = NaturalTransformation(:alpha2, :F, :G;
                                        components=Dict(:Z => identity))
            @test !FunctorFlow.is_natural(α2, F, G, [obj_A])
        end

        @testset "check_laws" begin
            # Law that accepts obj argument
            obj = CategoricalModelObject(:Test;
                                          semantic_laws=[(_obj -> true)])
            results = check_laws(obj)
            @test length(results) == 1
            @test results[1][2] == true

            obj2 = CategoricalModelObject(:Empty)
            @test isempty(check_laws(obj2))
        end

        @testset "MODEL_REGISTRY" begin
            obj = CategoricalModelObject(:RegTest)
            register_model!(obj)
            @test get_model(:RegTest) === obj
            @test haskey(MODEL_REGISTRY, :RegTest)
            delete!(MODEL_REGISTRY, :RegTest)
        end
    end

    # ==================================================================
    # B. Universal Constructions (universal.jl)
    # ==================================================================
    @testset "Universal Constructions" begin
        @testset "pullback" begin
            ket1 = ket_block(; name=:KET1)
            ket2 = ket_block(; name=:KET2)
            pb = pullback(ket1, ket2; over=:SharedContext)
            @test pb isa PullbackResult
            @test pb.cone isa FunctorFlow.Diagram
            @test pb.shared_object == :SharedContext
            @test pb.projection1 == :left
            @test pb.projection2 == :right
            @test length(pb.interface_morphisms) > 0
        end

        @testset "pushout" begin
            ket1 = ket_block(; name=:KET1)
            ket2 = ket_block(; name=:KET2)
            po = pushout(ket1, ket2; along=:SharedBase)
            @test po isa PushoutResult
            @test po.cocone isa FunctorFlow.Diagram
            @test po.shared_object == :SharedBase
        end

        @testset "product" begin
            D1 = ket_block(; name=:A)
            D2 = ket_block(; name=:B)
            D3 = ket_block(; name=:C)

            prod2 = product(D1, D2)
            @test prod2 isa ProductResult
            @test length(prod2.projections) == 2

            prod3 = product(D1, D2, D3)
            @test length(prod3.projections) == 3
        end

        @testset "coproduct" begin
            D1 = ket_block(; name=:A)
            D2 = ket_block(; name=:B)
            coprod = coproduct(D1, D2)
            @test coprod isa CoproductResult
            @test length(coprod.injections) == 2
        end

        @testset "equalizer" begin
            D = db_square(; first_impl=x -> x * 2, second_impl=x -> x + 1)
            eq = equalizer(D, :f, :g)
            @test eq isa EqualizerResult
            @test eq.equalizer_map isa Symbol
        end

        @testset "coequalizer" begin
            D = Diagram(:ParallelPair)
            add_object!(D, :A; kind=:state)
            add_object!(D, :B; kind=:state)
            add_morphism!(D, :f, :A, :B)
            add_morphism!(D, :g, :A, :B)

            coeq = coequalizer(D, :f, :g)
            @test coeq isa CoequalizerResult
            @test coeq.coequalizer_map isa Symbol
            @test coeq.quotient_object isa Symbol
            @test haskey(coeq.coequalizer_diagram.objects, coeq.quotient_object)
            @test haskey(coeq.coequalizer_diagram.operations, coeq.coequalizer_map)
            @test !isempty(coeq.coequalizer_diagram.losses)
        end

        @testset "verify pullback" begin
            pb = pullback(ket_block(; name=:A), ket_block(; name=:B); over=:Shared)
            v = verify(pb)
            @test v.construction == :pullback
            @test v.passed isa Bool
            @test haskey(v.checks, :has_left_factor)
            @test haskey(v.checks, :has_right_factor)
            @test haskey(v.checks, :has_shared_object)
        end

        @testset "verify pushout" begin
            po = pushout(ket_block(; name=:A), ket_block(; name=:B); along=:Shared)
            v = verify(po)
            @test v.construction == :pushout
            @test v.passed isa Bool
        end

        @testset "verify product" begin
            prod = product(ket_block(; name=:A), ket_block(; name=:B))
            v = verify(prod)
            @test v.construction == :product
            @test v.passed isa Bool
        end

        @testset "verify coproduct" begin
            coprod = coproduct(ket_block(; name=:A), ket_block(; name=:B))
            v = verify(coprod)
            @test v.construction == :coproduct
            @test v.passed isa Bool
        end

        @testset "verify equalizer" begin
            D = db_square(; first_impl=x -> x * 2, second_impl=x -> x + 1)
            eq = equalizer(D, :f, :g)
            v = verify(eq)
            @test v.construction == :equalizer
            @test haskey(v.checks, :has_equalizer_map)
        end

        @testset "verify coequalizer" begin
            D = Diagram(:PP)
            add_object!(D, :A; kind=:state)
            add_object!(D, :B; kind=:state)
            add_morphism!(D, :f, :A, :B)
            add_morphism!(D, :g, :A, :B)
            coeq = coequalizer(D, :f, :g)
            v = verify(coeq)
            @test v.construction == :coequalizer
            @test v.passed
            @test haskey(v.checks, :has_quotient_object)
            @test haskey(v.checks, :has_coequalizer_map)
            @test haskey(v.checks, :map_targets_quotient)
            @test haskey(v.checks, :has_coeq_loss)
        end

        @testset "compile_construction" begin
            pb = pullback(ket_block(; name=:A), ket_block(; name=:B); over=:Shared)
            compiled = compile_construction(pb)
            @test compiled isa FunctorFlow.CompiledDiagram
        end

        @testset "universal_morphism" begin
            pb = pullback(ket_block(; name=:A), ket_block(; name=:B); over=:Shared)
            D_cone = ket_block(; name=:Cone)
            um = universal_morphism(pb, D_cone)
            @test um isa FunctorFlow.Diagram
        end
    end

    # ==================================================================
    # C. Causal Semantics (causal.jl)
    # ==================================================================
    @testset "Causal Semantics" begin
        @testset "CausalContext" begin
            ctx = CausalContext(:test;
                                observational_regime=:obs,
                                interventional_regime=:do)
            @test ctx.name == :test
            @test ctx.observational_regime == :obs
            @test ctx.interventional_regime == :do
        end

        @testset "build_causal_diagram" begin
            ctx = CausalContext(:test; observational_regime=:obs, interventional_regime=:do)
            cd = build_causal_diagram(:CausalTest; context=ctx)
            @test cd isa CausalDiagram
            @test cd.base_diagram isa FunctorFlow.Diagram
            @test haskey(cd.base_diagram.operations, :intervene)
            @test haskey(cd.base_diagram.operations, :condition)
            @test cd.base_diagram.operations[:intervene].direction == FunctorFlow.LEFT
            @test cd.base_diagram.operations[:condition].direction == FunctorFlow.RIGHT
        end

        @testset "causal_transport" begin
            ctx = CausalContext(:test; observational_regime=:obs, interventional_regime=:do)
            cd_src = build_causal_diagram(:Src; context=ctx)
            cd_tgt = build_causal_diagram(:Tgt; context=ctx)

            # Without density ratio
            D = causal_transport(cd_src, cd_tgt)
            @test D isa FunctorFlow.Diagram

            # With density ratio function
            density = x -> x * 1.5
            D2 = causal_transport(cd_src, cd_tgt; density_ratio=density)
            @test D2 isa FunctorFlow.Diagram
            @test haskey(D2.objects, :DensityRatio)
            @test haskey(D2.operations, :rn_reweight)
        end

        @testset "interventional_expectation" begin
            ctx = CausalContext(:test; observational_regime=:obs, interventional_regime=:do)
            cd = build_causal_diagram(:IntExp; context=ctx)
            # Provide data matching diagram objects
            obs_data = Dict(
                :Observations => Dict(:a => 1.0, :b => 2.0),
                :CausalStructure => Dict((:a, :b) => true)
            )
            result = interventional_expectation(cd, obs_data)
            @test result isa Dict
            @test haskey(result, :all_values)
        end

        @testset "is_identifiable" begin
            ctx = CausalContext(:test; observational_regime=:obs, interventional_regime=:do)
            cd = build_causal_diagram(:Ident; context=ctx)
            result = is_identifiable(cd, :Y)
            @test result isa NamedTuple
            @test haskey(result, :identifiable)
            @test haskey(result, :rule)
            @test haskey(result, :reasoning)
        end
    end

    # ==================================================================
    # D. Topos Foundations (topos.jl)
    # ==================================================================
    @testset "Topos Foundations" begin
        @testset "SubobjectClassifier" begin
            omega = SubobjectClassifier(:Omega)
            @test omega.truth_object == :Omega
            @test omega.true_map == :true_map
            @test Symbol("true") in omega.truth_values
            @test Symbol("false") in omega.truth_values
        end

        @testset "SheafSection" begin
            sec = SheafSection(:s1; base_space=:X, section_data=[1, 2, 3])
            @test sec.name == :s1
            @test sec.base_space == :X
            @test sec.section_data == [1, 2, 3]
        end

        @testset "InternalPredicate" begin
            omega = SubobjectClassifier(:Omega)
            pred = InternalPredicate(:positive, omega;
                                      characteristic_map=x -> x > 0)
            @test pred.name == :positive
            @test pred.classifier === omega
            @test pred.characteristic_map !== nothing
        end

        @testset "build_sheaf_diagram" begin
            sec1 = SheafSection(:s1; base_space=:X, section_data=[1, 2, 3])
            sec2 = SheafSection(:s2; base_space=:X, section_data=[2, 3, 4])
            D = build_sheaf_diagram([sec1, sec2]; name=:GlueTest)
            @test haskey(D.operations, :glue)
            @test D.operations[:glue].reducer == :set_union
        end

        @testset "evaluate_predicate" begin
            omega = SubobjectClassifier(:Omega)
            pred = InternalPredicate(:positive, omega;
                                      characteristic_map=x -> x > 0)

            @test evaluate_predicate(pred, 5) == Symbol("true")
            @test evaluate_predicate(pred, -1) == Symbol("false")
            @test evaluate_predicate(pred, 0) == Symbol("false")

            # No characteristic map
            pred2 = InternalPredicate(:empty, omega)
            @test_throws ArgumentError evaluate_predicate(pred2, 1)
        end

        @testset "classify_subobject" begin
            omega = SubobjectClassifier(:Omega)

            # Dict data
            data = Dict(:a => 5, :b => -3, :c => 10)
            cl = classify_subobject(omega, x -> x > 0, data)
            @test cl[:a] == Symbol("true")
            @test cl[:b] == Symbol("false")
            @test cl[:c] == Symbol("true")

            # Vector data
            vec_data = [1, -2, 3, 0, -5]
            cl2 = classify_subobject(omega, x -> x > 0, vec_data)
            @test cl2[1] == Symbol("true")
            @test cl2[2] == Symbol("false")
            @test cl2[3] == Symbol("true")
            @test cl2[4] == Symbol("false")
            @test cl2[5] == Symbol("false")
        end

        @testset "check_coherence" begin
            sec1 = SheafSection(:s1; base_space=:X, section_data=[1, 2, 3],
                                domain=Set([:a, :b, :c]))
            sec2 = SheafSection(:s2; base_space=:X, section_data=[2, 3, 4],
                                domain=Set([:b, :c, :d]))
            check = SheafCoherenceCheck([sec1, sec2])
            result = check_coherence(check)
            @test result isa NamedTuple
            @test haskey(result, :passed)
            @test haskey(result, :locality)
            @test haskey(result, :gluing)
            @test haskey(result, :stability)
        end

        @testset "internal logic operators" begin
            omega = SubobjectClassifier(:Omega)
            p = InternalPredicate(:pos, omega; characteristic_map=x -> x > 0)
            q = InternalPredicate(:even, omega; characteristic_map=x -> x % 2 == 0)

            # AND
            pq = internal_and(p, q)
            @test pq.characteristic_map(4) == true
            @test pq.characteristic_map(-2) == false
            @test pq.characteristic_map(3) == false

            # OR
            p_or_q = internal_or(p, q)
            @test p_or_q.characteristic_map(4) == true
            @test p_or_q.characteristic_map(-2) == true
            @test p_or_q.characteristic_map(3) == true
            @test p_or_q.characteristic_map(-3) == false

            # NOT
            not_p = internal_not(p)
            @test not_p.characteristic_map(5) == false
            @test not_p.characteristic_map(-1) == true
        end
    end

    # ==================================================================
    # E. Proof Interface (proof_interface.jl)
    # ==================================================================
    @testset "Proof Interface — Construction Certificates" begin
        @testset "pullback certificate" begin
            pb = pullback(ket_block(; name=:A), ket_block(; name=:B); over=:Shared)
            cert = render_construction_certificate(pb)
            @test cert isa String
            @test occursin("theorem", cert)
            @test occursin("pullback", lowercase(cert))
        end

        @testset "pushout certificate" begin
            po = pushout(ket_block(; name=:A), ket_block(; name=:B); along=:Shared)
            cert = render_construction_certificate(po)
            @test cert isa String
            @test occursin("pushout", lowercase(cert))
        end

        @testset "product certificate" begin
            prod = product(ket_block(; name=:A), ket_block(; name=:B))
            cert = render_construction_certificate(prod)
            @test cert isa String
            @test occursin("product", lowercase(cert))
        end

        @testset "coproduct certificate" begin
            coprod = coproduct(ket_block(; name=:A), ket_block(; name=:B))
            cert = render_construction_certificate(coprod)
            @test cert isa String
            @test occursin("coproduct", lowercase(cert))
        end

        @testset "equalizer certificate" begin
            D = db_square(; first_impl=x -> x * 2, second_impl=x -> x + 1)
            eq = equalizer(D, :f, :g)
            cert = render_construction_certificate(eq)
            @test cert isa String
            @test occursin("equalizer", lowercase(cert))
        end

        @testset "coequalizer certificate" begin
            D = Diagram(:PP)
            add_object!(D, :A; kind=:state)
            add_object!(D, :B; kind=:state)
            add_morphism!(D, :f, :A, :B)
            add_morphism!(D, :g, :A, :B)
            coeq = coequalizer(D, :f, :g)
            cert = render_construction_certificate(coeq)
            @test cert isa String
            @test occursin("coequalizer", lowercase(cert))
        end

        @testset "lean certificate roundtrip" begin
            D = ket_block()
            cert = render_lean_certificate(D)
            @test cert isa String
            @test occursin("structure", cert) || occursin("def", cert)
        end
    end
end
