using Test
using FunctorFlow
import Catlab

@testset "FunctorFlow.jl" begin

    @testset "Core Types" begin
        obj = FFObject(:Tokens; kind=:messages, description="Token collection")
        @test obj.name == :Tokens
        @test obj.kind == :messages
        @test !FunctorFlow._is_placeholder(obj)

        placeholder = FFObject(:X)
        @test FunctorFlow._is_placeholder(placeholder)

        m = Morphism(:encode, :Tokens, :Latent)
        @test m.source == :Tokens
        @test m.target == :Latent

        comp = Composition(:pipeline, [:encode, :decode], :Tokens, :Output)
        @test comp.chain == [:encode, :decode]

        kan = KanExtension(:agg, LEFT, :Values, :Incidence; reducer=:sum)
        @test kan.direction == LEFT
        @test kan.reducer == :sum

        loss = ObstructionLoss(:obs, [(:p1, :p2)]; comparator=:l2, weight=2.0)
        @test loss.weight == 2.0
        @test loss.paths == [(:p1, :p2)]

        port = Port(:input, :Tokens; kind=:object, port_type=:messages, direction=INPUT)
        @test port.direction == INPUT

        adapter = Adapter(:ctx_to_plan, :contextualized_messages, :plan_candidates)
        @test adapter.source_type == :contextualized_messages
    end

    @testset "Diagram Construction" begin
        D = Diagram(:TestDiagram)
        @test D.name == :TestDiagram
        @test isempty(D.objects)

        add_object!(D, :Tokens; kind=:messages)
        @test haskey(D.objects, :Tokens)
        @test D.objects[:Tokens].kind == :messages

        add_object!(D, :Latent; kind=:hidden_state)
        add_morphism!(D, :encode, :Tokens, :Latent)
        @test haskey(D.operations, :encode)
        @test D.operations[:encode] isa Morphism

        add_morphism!(D, :decode, :Latent, :Tokens)
        comp = compose!(D, :encode, :decode; name=:roundtrip)
        @test comp.source == :Tokens
        @test comp.target == :Tokens

        # Auto-create placeholder objects
        D2 = Diagram(:AutoCreate)
        add_morphism!(D2, :f, :A, :B)
        @test haskey(D2.objects, :A)
        @test haskey(D2.objects, :B)
    end

    @testset "Kan Extensions" begin
        D = Diagram(:KanTest)
        add_object!(D, :Values; kind=:messages)
        add_object!(D, :Incidence; kind=:relation)

        kan = add_left_kan!(D, :aggregate; source=:Values, along=:Incidence, reducer=:sum)
        @test kan.direction == LEFT
        @test kan.reducer == :sum

        rkan = add_right_kan!(D, :complete; source=:Values, along=:Incidence, reducer=:first_non_null)
        @test rkan.direction == RIGHT
    end

    @testset "Obstruction Loss" begin
        D = Diagram(:DBTest)
        add_object!(D, :State; kind=:state)
        add_morphism!(D, :f, :State, :State; implementation=x -> x * 2)
        add_morphism!(D, :g, :State, :State; implementation=x -> x + 1)
        compose!(D, :f, :g; name=:fg)
        compose!(D, :g, :f; name=:gf)
        add_obstruction_loss!(D, :obs; paths=[(:fg, :gf)], comparator=:l2)
        @test haskey(D.losses, :obs)
    end

    @testset "Ports" begin
        D = Diagram(:PortTest)
        add_object!(D, :Tokens; kind=:messages)
        p = expose_port!(D, :input, :Tokens; direction=INPUT)
        @test p.port_type == :messages
        @test p.direction == INPUT

        p2 = get_port(D, :input)
        @test p2.ref == :Tokens
    end

    @testset "IR and Serialization" begin
        D = ket_block()
        ir = to_ir(D)
        @test ir.name == :KETBlock
        @test length(ir.objects) >= 3
        @test length(ir.operations) >= 1

        d = as_dict(ir)
        @test d["name"] == "KETBlock"
        @test d["objects"] isa Vector

        json_str = to_json(D)
        @test occursin("KETBlock", json_str)
    end

    @testset "Reducers" begin
        source = Dict(1 => 10.0, 2 => 20.0, 3 => 30.0)
        relation = Dict("a" => [1, 2], "b" => [2, 3])

        result = FunctorFlow._reduce_sum(source, relation, Dict())
        @test result["a"] ≈ 30.0
        @test result["b"] ≈ 50.0

        result = FunctorFlow._reduce_mean(source, relation, Dict())
        @test result["a"] ≈ 15.0
        @test result["b"] ≈ 25.0
    end

    @testset "Comparators" begin
        @test FunctorFlow._l2_distance([1.0, 2.0], [1.0, 2.0]) ≈ 0.0
        @test FunctorFlow._l2_distance([1.0, 0.0], [0.0, 0.0]) ≈ 1.0
        @test FunctorFlow._l1_distance([1.0, 2.0], [0.0, 0.0]) ≈ 3.0
        @test FunctorFlow._jaccard_distance("draft plan repair", "repair draft") ≈ 1 / 3
    end

    @testset "Compiler & Execution" begin
        D = Diagram(:CompilerTest)
        add_object!(D, :X; kind=:input)
        add_object!(D, :Y; kind=:output)
        add_morphism!(D, :double, :X, :Y; implementation=x -> x * 2)

        compiled = compile_to_callable(D)
        result = FunctorFlow.run(compiled, Dict(:X => 5))
        @test result.values[:double] == 10
        @test isempty(result.losses)

        # KET execution
        D2 = Diagram(:KETExec)
        add_object!(D2, :Values; kind=:messages)
        add_object!(D2, :Incidence; kind=:relation)
        add_left_kan!(D2, :agg; source=:Values, along=:Incidence, reducer=:sum)

        compiled2 = compile_to_callable(D2)
        result2 = FunctorFlow.run(compiled2, Dict(
            :Values => Dict(1 => 1.0, 2 => 2.0, 3 => 4.0),
            :Incidence => Dict("left" => [1, 2], "right" => [2, 3])
        ))
        @test result2.values[:agg]["left"] ≈ 3.0
        @test result2.values[:agg]["right"] ≈ 6.0

        # Obstruction loss execution
        D3 = Diagram(:LossExec)
        add_object!(D3, :S; kind=:state)
        add_morphism!(D3, :f, :S, :S; implementation=x -> x * 2)
        add_morphism!(D3, :g, :S, :S; implementation=x -> x + 1)
        compose!(D3, :f, :g; name=:fg)
        compose!(D3, :g, :f; name=:gf)
        add_obstruction_loss!(D3, :obs; paths=[(:fg, :gf)])

        result3 = FunctorFlow.run(D3, Dict(:S => 3.0))
        @test result3.values[:fg] == 7.0   # (3*2)+1 = 7
        @test result3.values[:gf] == 8.0   # (3+1)*2 = 8
        @test result3.losses[:obs] > 0.0   # non-commutative
    end

    include("test_lux_ext.jl")

    @testset "Composition" begin
        child = ket_block()
        parent = Diagram(:Parent)
        inc = include!(parent, child; namespace=:enc)

        @test haskey(parent.objects, :enc__Values)
        @test haskey(parent.operations, :enc__aggregate)
        @test object_ref(inc, :Values) == :enc__Values
        @test operation_ref(inc, :aggregate) == :enc__aggregate

        p = port_spec(inc, :output)
        @test p.ref == :enc__aggregate
    end

    @testset "Adapters" begin
        D = Diagram(:AdapterTest)
        register_adapter!(D, :ctx_to_plan;
                          source_type=:contextualized_messages,
                          target_type=:plan_candidates,
                          implementation=identity)
        @test haskey(D.adapters, (:contextualized_messages, :plan_candidates))

        # Adapter library
        D2 = Diagram(:LibTest)
        use_adapter_library!(D2, :standard)
        @test haskey(D2.adapters, (:contextualized_messages, :plan_candidates))
        @test haskey(D2.adapters, (:plan_candidates, :plan))
    end

    @testset "Block Builders" begin
        ket = ket_block()
        @test ket.name == :KETBlock
        @test haskey(ket.operations, :aggregate)
        @test haskey(ket.ports, :input)
        @test haskey(ket.ports, :output)

        db = db_square(; first_impl=x -> x * 2, second_impl=x -> x + 1)
        @test haskey(db.losses, :obstruction)

        gt = gt_neighborhood_block()
        @test haskey(gt.operations, :lift)
        @test haskey(gt.operations, :aggregate)

        comp = completion_block()
        @test comp.operations[:complete] isa KanExtension
        @test comp.operations[:complete].direction == RIGHT

        basket = basket_workflow_block()
        @test basket.operations[:compose_fragments] isa KanExtension

        rocket = rocket_repair_block()
        @test rocket.operations[:repair] isa KanExtension
        @test rocket.operations[:repair].direction == RIGHT

        glue = democritus_gluing_block()
        @test glue.operations[:glue].reducer == :set_union

        demo_assembly = democritus_assembly_pipeline()
        @test haskey(demo_assembly.losses, :section_coherence)

        topo = topocoend_block()
        @test haskey(topo.operations, :infer_neighborhood)
        @test haskey(topo.operations, :coend_aggregate)

        horn = horn_fill_block()
        @test haskey(horn.losses, :horn_obstruction)

        higher_horn = higher_horn_block()
        @test haskey(higher_horn.losses, :higher_horn_obstruction)
        @test haskey(higher_horn.ports, :d03)

        quotient = bisimulation_quotient_block()
        @test haskey(quotient.ports, :output)

        # build_macro
        ket2 = build_macro(:ket; name=:CustomKET)
        @test ket2.name == :CustomKET
        @test build_macro(:higher_horn).name == :HigherHorn
    end

    @testset "Structured LM Duality" begin
        dual = structured_lm_duality()
        @test dual.name == :StructuredLMDuality
        @test haskey(dual.ports, :input)
        @test haskey(dual.ports, :predict_output)
        @test haskey(dual.ports, :repair_output)
    end

    @testset "BASKET-ROCKET Pipeline" begin
        pipeline = basket_rocket_pipeline()
        @test pipeline.name == :BasketRocketPipeline
        @test haskey(pipeline.ports, :output)
        @test haskey(pipeline.losses, :draft_repair_consistency)
    end

    @testset "CATAGI-Inspired Execution" begin
        topo = topocoend_block(;
            infer_neighborhood_impl=x -> Dict(:all => collect(keys(x))),
            lift_impl=identity
        )
        topo_result = FunctorFlow.run(topo, Dict(:Tokens => Dict(:a => 1.0, :b => 3.0)))
        @test topo_result.values[:coend_aggregate][:all] ≈ 2.0

        topo_example = build_topocoend_triage_example()
        topo_example_result = execute_topocoend_triage_example(topo_example)
        topo_summary = summarize_topocoend_triage_example(topo_example)
        learned_cover = topo_example_result.values[topo_example[:diagram].ports[:learned_relation].ref]
        @test :respiratory_focus in keys(learned_cover)
        @test :chief_complaint in learned_cover[:respiratory_focus]
        @test :lactate in learned_cover[:metabolic_focus]
        @test topo_summary["highest_priority_context"] == "respiratory_focus"
        @test topo_summary["counts"]["contexts"] == 3
        @test topo_summary["context_risks"]["respiratory_focus"] ≈ (0.95 + 0.90 + 0.60) / 3
        @test topo_summary["context_risks"]["metabolic_focus"] ≈ (0.70 + 0.85) / 2

        bisim_example = build_bisimulation_quotient_example()
        bisim_result = execute_bisimulation_quotient_example(bisim_example)
        bisim_summary = summarize_bisimulation_quotient_example(bisim_example)
        quotient_codes = bisim_result.values[bisim_example[:diagram].ports[:output].ref]
        @test quotient_codes[:acute_reroute] == 21.0
        @test quotient_codes[:watchful_recovery] == 10.0
        @test quotient_codes[:steady_progress] == 0.0
        @test bisim_result.losses[:behavioral_class_coeq_loss] ≈ 0.0
        @test "controller_alignment" in bisim_summary["declared_bisimulations"]
        @test bisim_summary["counts"]["quotient_classes"] == 3
        @test bisim_summary["quotient_labels"]["acute_reroute"] == "acute_reroute"

        horn = horn_fill_block(;
            first_face_impl=x -> x + 1,
            second_face_impl=x -> x * 2,
            filler_impl=x -> (x + 1) * 2
        )
        horn_result = FunctorFlow.run(horn, Dict(:Vertex0 => 3))
        @test horn_result.values[:horn_boundary] == 8
        @test horn_result.losses[:horn_obstruction] ≈ 0.0

        higher_horn = higher_horn_block(;
            config=HigherHornConfig(
                filler_faces=[:d03_exact, :d03_relaxed]
            ),
            boundary_face_impls=[x -> x + 1, x -> x * 2, x -> x - 3],
            filler_impls=[x -> ((x + 1) * 2) - 3, x -> ((x + 1) * 2) - 2]
        )
        higher_horn_result = FunctorFlow.run(higher_horn, Dict(:Vertex0 => 3))
        @test higher_horn_result.values[:higher_horn_boundary] == 5
        @test higher_horn_result.values[:d03_exact] == 5
        @test higher_horn_result.values[:d03_relaxed] == 6
        @test higher_horn_result.losses[:higher_horn_obstruction] ≈ 1.0

        demo = democritus_assembly_pipeline(; restrict_impl=identity)
        demo_input = Dict(
            demo.ports[:input].ref => Dict(:r1 => Set(["A->B"]), :r2 => Set(["B->C"])),
            demo.ports[:relation].ref => Dict(:global => [:r1, :r2])
        )
        demo_result = FunctorFlow.run(demo, demo_input)
        @test haskey(demo_result.losses, :section_coherence)

        democritus_example = build_democritus_assembly_example()
        democritus_result = execute_democritus_assembly_example(democritus_example)
        democritus_summary = summarize_democritus_assembly_example(democritus_example)
        global_claims = democritus_result.values[democritus_example[:diagram].ports[:global_output].ref][:labor_market]
        @test "minimum wage -> employment" in global_claims
        @test "minimum wage -> employment" in democritus_summary["global_inferred_claims"]
        @test "minimum wage -> employment" in democritus_summary["fragment_repairs"]["policy"]
        @test democritus_summary["counts"]["repaired_claims"] >= 1
        @test democritus_summary["coherence_loss"] > 0.0

        pipeline = basket_rocket_pipeline()
        br_input = Dict(
            pipeline.ports[:input].ref => Dict(:step1 => "fetch data", :step2 => "clean data"),
            pipeline.ports[:draft_relation].ref => Dict(:draft => [:step1, :step2]),
            pipeline.ports[:repair_relation].ref => Dict(:draft => [:draft])
        )
        br_result = FunctorFlow.run(pipeline, br_input)
        @test haskey(br_result.losses, :draft_repair_consistency)
    end

    @testset "Tutorial Libraries" begin
        lib = get_tutorial_library(:foundations)
        @test :ket in lib.macro_names
        @test :db_square in lib.macro_names

        builders = macro_builders(lib)
        @test haskey(builders, :ket)

        D = Diagram(:TutTest)
        install_tutorial_library!(D, :planning)
        @test haskey(D.adapters, (:contextualized_messages, :plan_candidates))

        planning = get_tutorial_library(:planning)
        @test :democritus_assembly in planning.macro_names
        @test :higher_horn in planning.macro_names
    end

    @testset "DSL Macros" begin
        D = @diagram TestMacro begin
            @object Tokens kind=:messages
            @object Nbrs kind=:relation
            @left_kan aggregate source=Tokens along=Nbrs reducer=:sum
        end
        @test D.name == :TestMacro
        @test haskey(D.objects, :Tokens)
        @test D.objects[:Tokens].kind == :messages
        @test haskey(D.operations, :aggregate)
        @test D.operations[:aggregate] isa KanExtension
    end

    @testset "Proof Interface" begin
        D = ket_block()
        payload = diagram_certificate_payload(D)
        @test payload["diagram_name"] == "KETBlock"
        @test "Values" in payload["objects"]

        lean = render_lean_certificate(D)
        @test occursin("FunctorFlowProofs.Generated", lean)
        @test occursin("exportedDiagram", lean)
        @test occursin("exportedArtifact", lean)
    end

    @testset "Universal Constructions" begin
        ket1 = ket_block(; name=:KET1)
        ket2 = ket_block(; name=:KET2)

        pb = pullback(ket1, ket2; over=:SharedContext)
        @test pb isa PullbackResult
        @test pb.cone isa Diagram

        po = pushout(ket1, ket2; along=:SharedBase)
        @test po isa PushoutResult

        prod = product(ket1, ket2)
        @test prod isa ProductResult
        @test length(prod.projections) == 2

        coprod = coproduct(ket1, ket2)
        @test coprod isa CoproductResult
        @test length(coprod.injections) == 2

        D = db_square(; first_impl=x -> x * 2, second_impl=x -> x + 1)
        eq = equalizer(D, :f, :g)
        @test eq isa EqualizerResult
    end

    @testset "Causal Semantics" begin
        ctx = CausalContext(:test; observational_regime=:obs, interventional_regime=:do)
        @test ctx.name == :test

        cd = build_causal_diagram(:CausalTest; context=ctx)
        @test cd.base_diagram isa Diagram
        @test haskey(cd.base_diagram.operations, :intervene)
        @test haskey(cd.base_diagram.operations, :condition)
        @test cd.base_diagram.operations[:intervene].direction == LEFT
        @test cd.base_diagram.operations[:condition].direction == RIGHT
    end

    @testset "Topos Foundations" begin
        omega = SubobjectClassifier(:Omega)
        @test omega.truth_object == :Omega

        sec = SheafSection(:s1; base_space=:X, section_data=[1, 2, 3])
        @test sec.base_space == :X

        D = build_sheaf_diagram([sec]; name=:GlueTest)
        @test haskey(D.operations, :glue)
        @test D.operations[:glue].reducer == :set_union
    end

    @testset "Display" begin
        D = ket_block()
        s = sprint(show, D)
        @test occursin("Diagram", s)
        @test occursin("KETBlock", s)

        s2 = sprint(show, D.operations[:aggregate])
        @test occursin("Σ", s2)  # Unicode display

        s3 = sprint(show, FFObject(:X; kind=:messages))
        @test occursin("messages", s3)
    end

    # Include comprehensive V1 feature tests
    include("test_v1.jl")

    @testset "Unicode Operators" begin
        D = Diagram(:UnicodeTest)
        add_object!(D, :Tokens; kind=:messages)
        add_object!(D, :Nbrs; kind=:relation)

        # Σ (left Kan)
        kan = Σ(D, :Tokens; along=:Nbrs, reducer=:sum, name=:aggregate)
        @test kan isa KanExtension
        @test kan.direction == LEFT
        @test kan.reducer == :sum

        # Δ (right Kan)
        kan2 = Δ(D, :Tokens; along=:Nbrs, name=:complete)
        @test kan2 isa KanExtension
        @test kan2.direction == RIGHT
        @test kan2.reducer == :first_non_null

        # Auto-generated name
        D2 = Diagram(:AutoName)
        add_object!(D2, :A; kind=:value)
        add_object!(D2, :R; kind=:relation)
        kan3 = Σ(D2, :A; along=:R)
        @test kan3.name == :Σ_A_along_R

        # ⊗ (product)
        k1 = ket_block(; name=:K1)
        k2 = ket_block(; name=:K2)
        prod = k1 ⊗ k2
        @test prod isa ProductResult
        @test length(prod.projections) == 2

        # ⊕ (coproduct)
        coprod = k1 ⊕ k2
        @test coprod isa CoproductResult
        @test length(coprod.injections) == 2

        # ⋅ (ModelMorphism composition)
        f = ModelMorphism(:f, :A, :B; functor_data=x -> x * 2)
        g = ModelMorphism(:g, :B, :C; functor_data=x -> x + 1)
        h = f ⋅ g
        @test h.source == :A
        @test h.target == :C

        # ASCII aliases
        @test left_kan === Σ
        @test right_kan === Δ
    end

    @testset "@functorflow Macro" begin
        # Object declarations with ::
        D = @functorflow MacroTest begin
            Tokens::messages
            Nbrs::relation
            Ctx::contextualized_messages
        end
        @test D.name == :MacroTest
        @test D.objects[:Tokens].kind == :messages
        @test D.objects[:Nbrs].kind == :relation

        # Morphism with →
        D2 = @functorflow ArrowTest begin
            A::input
            B::output
            f = A → B
        end
        @test D2.operations[:f] isa Morphism
        @test D2.operations[:f].source == :A
        @test D2.operations[:f].target == :B

        # Σ/Δ in macro
        D3 = @functorflow KanTest begin
            Tokens::messages
            Nbrs::relation
            aggregate = Σ(:Tokens; along=:Nbrs, reducer=:sum)
            complete = Δ(:Tokens; along=:Nbrs)
        end
        @test D3.operations[:aggregate] isa KanExtension
        @test D3.operations[:aggregate].direction == LEFT
        @test D3.operations[:complete] isa KanExtension
        @test D3.operations[:complete].direction == RIGHT
    end

    @testset "ACSet Schema" begin
        # to_acset
        D = ket_block()
        acs = to_acset(D)
        @test nparts(acs, :Node) == length(D.objects)
        @test nparts(acs, :Kan) >= 1
        node_names = subpart(acs, :node_name)
        @test :Values in node_names
        @test :Incidence in node_names

        # DB square has edges
        D2 = db_square(; first_impl=x -> x * 2, second_impl=x -> x + 1)
        acs2 = to_acset(D2)
        @test nparts(acs2, :Edge) >= 2

        # from_acset roundtrip
        D3 = from_acset(acs; name=:Roundtrip)
        @test Set(keys(D3.objects)) == Set(keys(D.objects))

        # to_symbolic
        sym = to_symbolic(D2)
        @test :State in keys(sym.objects)
        @test :f in keys(sym.morphisms)
        @test :g in keys(sym.morphisms)

        # to_presentation
        pres = to_presentation(D2)
        obs = Catlab.Theories.generators(pres, :Ob)
        @test length(obs) >= 1

        # diagram_to_acset (alias)
        acs3 = diagram_to_acset(D)
        @test nparts(acs3, :Node) == nparts(acs, :Node)

        # acset_to_diagram (alias)
        D4 = acset_to_diagram(acs3; name=:Alias)
        @test D4.name == :Alias
    end

    # JEPA, Coalgebra, and Energy tests
    include("test_jepa.jl")
end
