# ============================================================================
# test_lux_ext.jl — Tests for the FunctorFlowLuxExt extension
# ============================================================================

using Test
using FunctorFlow
using Lux
using LuxCore
import LinearAlgebra
using Random

# Package extensions need explicit import
import FunctorFlow: compile_to_lux
using Base: @invokelatest
const LuxExt = let ext = Base.get_extension(FunctorFlow, :FunctorFlowLuxExt)
    ext === nothing ? FunctorFlow : ext
end

@testset "FunctorFlowLuxExt" begin

    rng = Random.MersenneTwister(42)

    # -----------------------------------------------------------------
    @testset "DiagramDenseLayer basics" begin
        layer = LuxExt.DiagramDenseLayer(4, 3; name=:test_dense)
        ps = LuxCore.initialparameters(rng, layer)
        st = LuxCore.initialstates(rng, layer)

        @test size(ps.weight) == (3, 4)
        @test size(ps.bias) == (3,)
        @test st == NamedTuple()

        x = randn(rng, Float32, 4, 2)  # (in_dims, batch)
        y, st2 = layer(x, ps, st)
        @test size(y) == (3, 2)
        @test st2 == st
    end

    # -----------------------------------------------------------------
    @testset "DiagramDenseLayer with activation" begin
        layer = LuxExt.DiagramDenseLayer(4, 3; activation=relu, name=:relu_dense)
        ps = LuxCore.initialparameters(rng, layer)
        st = LuxCore.initialstates(rng, layer)

        x = randn(rng, Float32, 4, 2)
        y, _ = layer(x, ps, st)
        @test all(y .>= 0)  # ReLU output
    end

    # -----------------------------------------------------------------
    @testset "DiagramChainLayer" begin
        l1 = LuxExt.DiagramDenseLayer(4, 8; name=:d1)
        l2 = LuxExt.DiagramDenseLayer(8, 3; name=:d2)
        chain = LuxExt.DiagramChainLayer(
            LuxCore.AbstractLuxLayer[l1, l2];
            name=:test_chain, layer_names=[:d1, :d2])

        ps = LuxCore.initialparameters(rng, chain)
        st = LuxCore.initialstates(rng, chain)

        @test haskey(ps, :d1)
        @test haskey(ps, :d2)
        @test size(ps.d1.weight) == (8, 4)
        @test size(ps.d2.weight) == (3, 8)

        x = randn(rng, Float32, 4, 2)
        y, st2 = chain(x, ps, st)
        @test size(y) == (3, 2)
    end

    # -----------------------------------------------------------------
    @testset "KETAttentionLayer single-head" begin
        d_model = 8
        layer = LuxExt.KETAttentionLayer(d_model; n_heads=1, name=:attn)
        ps = LuxCore.initialparameters(rng, layer)
        st = LuxCore.initialstates(rng, layer)

        @test size(ps.W_q) == (d_model, d_model)
        @test size(ps.W_k) == (d_model, d_model)
        @test size(ps.W_v) == (d_model, d_model)
        @test size(ps.W_o) == (d_model, d_model)

        # Without mask
        seq_len = 5
        x = randn(rng, Float32, d_model, seq_len)
        y, st2 = layer(x, ps, st)
        @test size(y) == (d_model, seq_len)

        # With mask (causal)
        mask = Float32.(ones(seq_len, seq_len))
        for i in 1:seq_len, j in (i+1):seq_len
            mask[i, j] = 0.0f0  # causal: only attend to past
        end
        y_masked, _ = layer((x, mask), ps, st)
        @test size(y_masked) == (d_model, seq_len)
        # Masked and unmasked should differ (unless trivially equal)
        @test y_masked != y || seq_len <= 1
    end

    # -----------------------------------------------------------------
    @testset "KETAttentionLayer causal mask blocks future leakage" begin
        d_model = 4
        seq_len = 4
        layer = LuxExt.KETAttentionLayer(d_model; n_heads=1, name=:causal_attn)
        st = LuxCore.initialstates(rng, layer)

        eye4 = Matrix{Float32}(LinearAlgebra.I, d_model, d_model)
        zeros4 = zeros(Float32, d_model)
        ps = (
            W_q = eye4,
            W_k = eye4,
            W_v = eye4,
            W_o = eye4,
            b_q = zeros4,
            b_k = zeros4,
            b_v = zeros4,
            b_o = zeros4,
        )

        mask = Float32.([i <= j ? 1.0f0 : 0.0f0 for i in 1:seq_len, j in 1:seq_len])

        x1 = zeros(Float32, d_model, seq_len)
        x1[:, 1] .= Float32[1, 0, 0, 0]
        x1[:, 2] .= Float32[0, 1, 0, 0]
        x1[:, 3] .= Float32[0, 0, 1, 0]
        x1[:, 4] .= Float32[0, 0, 0, 1]

        x2 = copy(x1)
        x2[:, 4] .= Float32[10, 20, 30, 40]

        y1, _ = layer((x1, mask), ps, st)
        y2, _ = layer((x2, mask), ps, st)

        @test isapprox(y1[:, 1], y2[:, 1]; atol=1f-5)
        @test isapprox(y1[:, 2], y2[:, 2]; atol=1f-5)
        @test isapprox(y1[:, 3], y2[:, 3]; atol=1f-5)
        @test !isapprox(y1[:, 4], y2[:, 4]; atol=1f-5)
    end

    # -----------------------------------------------------------------
    @testset "KETAttentionLayer multi-head" begin
        d_model = 16
        n_heads = 4
        layer = LuxExt.KETAttentionLayer(d_model; n_heads=n_heads, name=:mh_attn)
        ps = LuxCore.initialparameters(rng, layer)
        st = LuxCore.initialstates(rng, layer)

        seq_len = 6
        x = randn(rng, Float32, d_model, seq_len)
        mask = Float32.(ones(seq_len, seq_len))
        y, _ = layer((x, mask), ps, st)
        @test size(y) == (d_model, seq_len)
    end

    # -----------------------------------------------------------------
    @testset "KETAttentionLayer causal mask blocks future leakage for batched inputs" begin
        d_model = 4
        seq_len = 4
        batch = 2
        layer = LuxExt.KETAttentionLayer(d_model; n_heads=1, name=:batched_causal_attn)
        st = LuxCore.initialstates(rng, layer)

        eye4 = Matrix{Float32}(LinearAlgebra.I, d_model, d_model)
        zeros4 = zeros(Float32, d_model)
        ps = (
            W_q = eye4,
            W_k = eye4,
            W_v = eye4,
            W_o = eye4,
            b_q = zeros4,
            b_k = zeros4,
            b_v = zeros4,
            b_o = zeros4,
        )

        mask = Float32.([i <= j ? 1.0f0 : 0.0f0 for i in 1:seq_len, j in 1:seq_len])

        x1 = zeros(Float32, d_model, seq_len, batch)
        x1[:, 1, 1] .= Float32[1, 0, 0, 0]
        x1[:, 2, 1] .= Float32[0, 1, 0, 0]
        x1[:, 3, 1] .= Float32[0, 0, 1, 0]
        x1[:, 4, 1] .= Float32[0, 0, 0, 1]
        x1[:, :, 2] .= x1[:, :, 1]

        x2 = copy(x1)
        x2[:, 4, 2] .= Float32[10, 20, 30, 40]

        y1, _ = layer((x1, mask), ps, st)
        y2, _ = layer((x2, mask), ps, st)

        @test isapprox(y1[:, 1:3, 1], y2[:, 1:3, 1]; atol=1f-5)
        @test isapprox(y1[:, 1:3, 2], y2[:, 1:3, 2]; atol=1f-5)
        @test !isapprox(y1[:, 4, 2], y2[:, 4, 2]; atol=1f-5)
    end

    # -----------------------------------------------------------------
    @testset "KETAttentionLayer multi-head causal mask blocks future leakage" begin
        d_model = 4
        seq_len = 4
        layer = LuxExt.KETAttentionLayer(d_model; n_heads=2, name=:mh_causal_attn)
        st = LuxCore.initialstates(rng, layer)

        eye4 = Matrix{Float32}(LinearAlgebra.I, d_model, d_model)
        zeros4 = zeros(Float32, d_model)
        ps = (
            W_q = eye4,
            W_k = eye4,
            W_v = eye4,
            W_o = eye4,
            b_q = zeros4,
            b_k = zeros4,
            b_v = zeros4,
            b_o = zeros4,
        )

        mask = Float32.([i <= j ? 1.0f0 : 0.0f0 for i in 1:seq_len, j in 1:seq_len])

        x1 = zeros(Float32, d_model, seq_len)
        x1[:, 1] .= Float32[1, 0, 0, 0]
        x1[:, 2] .= Float32[0, 1, 0, 0]
        x1[:, 3] .= Float32[0, 0, 1, 0]
        x1[:, 4] .= Float32[0, 0, 0, 1]

        x2 = copy(x1)
        x2[:, 4] .= Float32[10, 20, 30, 40]

        y1, _ = layer((x1, mask), ps, st)
        y2, _ = layer((x2, mask), ps, st)

        @test isapprox(y1[:, 1], y2[:, 1]; atol=1f-5)
        @test isapprox(y1[:, 2], y2[:, 2]; atol=1f-5)
        @test isapprox(y1[:, 3], y2[:, 3]; atol=1f-5)
        @test !isapprox(y1[:, 4], y2[:, 4]; atol=1f-5)
    end

    # -----------------------------------------------------------------
    @testset "Neural comparators" begin
        a = Float32[1.0, 2.0, 3.0]
        b = Float32[1.0, 2.0, 4.0]

        l2 = LuxExt.neural_l2_comparator(a, b)
        @test l2 ≈ 1.0f0

        l1 = LuxExt.neural_l1_comparator(a, b)
        @test l1 ≈ 1.0f0

        cos_dist = LuxExt.neural_cosine_comparator(a, a)
        @test isapprox(cos_dist, 0.0f0; atol=1e-5)
    end

    # -----------------------------------------------------------------
    @testset "compile_to_lux basic KET block" begin
        D = ket_block(; name=:TestKET, reducer=:ket_attention)
        d_model = 8
        attn = LuxExt.KETAttentionLayer(d_model; n_heads=1)
        model = compile_to_lux(D; reducer_layers=Dict(:ket_attention => attn))

        @test model isa LuxExt.LuxDiagramModel

        ps, st = Lux.setup(rng, model)
        @test haskey(ps, Symbol("red_ket_attention"))

        seq_len = 4
        inputs = Dict(
            :Values => randn(rng, Float32, d_model, seq_len),
            :Incidence => Float32.(ones(seq_len, seq_len))
        )
        result, st2 = model(inputs, ps, st)
        @test haskey(result, :values)
        @test haskey(result, :losses)
        @test haskey(result[:values], :aggregate)
        @test size(result[:values][:aggregate]) == (d_model, seq_len)
    end

    # -----------------------------------------------------------------
    @testset "compile_to_lux DB square with neural morphisms" begin
        D = db_square(; name=:TestDB)
        d_model = 4

        f_layer = LuxExt.DiagramDenseLayer(d_model, d_model; name=:f)
        g_layer = LuxExt.DiagramDenseLayer(d_model, d_model; name=:g)

        model = compile_to_lux(D;
            morphism_layers=Dict(:f => f_layer, :g => g_layer))

        ps, st = Lux.setup(rng, model)
        @test haskey(ps, Symbol("morph_f"))
        @test haskey(ps, Symbol("morph_g"))

        inputs = Dict(:State => randn(rng, Float32, d_model, 2))
        result, _ = model(inputs, ps, st)

        @test haskey(result[:values], :p1)
        @test haskey(result[:values], :p2)
        @test size(result[:values][:p1]) == (d_model, 2)

        # Obstruction loss should be computed
        @test haskey(result[:losses], :obstruction)
        @test result[:losses][:obstruction] >= 0
    end

    # -----------------------------------------------------------------
    @testset "compile_to_lux GT neighborhood block" begin
        d_model = 8
        D = gt_neighborhood_block(; reducer=:ket_attention)
        lift = LuxExt.DiagramDenseLayer(d_model, d_model; name=:lift)
        attn = LuxExt.KETAttentionLayer(d_model; n_heads=1)
        model = compile_to_lux(D;
            morphism_layers=Dict(:lift => lift),
            reducer_layers=Dict(:ket_attention => attn))

        ps, st = Lux.setup(rng, model)
        seq_len = 5
        inputs = Dict(
            :Tokens => randn(rng, Float32, d_model, seq_len),
            :Incidence => Float32.(ones(seq_len, seq_len))
        )
        result, _ = model(inputs, ps, st)
        @test haskey(result[:values], :aggregate)
        @test size(result[:values][:aggregate]) == (d_model, seq_len)
    end

    # -----------------------------------------------------------------
    @testset "build_ket_lux_model convenience" begin
        d_model = 16
        model, D = LuxExt.build_ket_lux_model(d_model; n_heads=2)

        @test model isa LuxExt.LuxDiagramModel
        @test D isa Diagram

        ps, st = Lux.setup(rng, model)
        seq_len = 6
        inputs = Dict(
            :Values => randn(rng, Float32, d_model, seq_len),
            :Incidence => Float32.(ones(seq_len, seq_len))
        )
        result, _ = model(inputs, ps, st)
        @test size(result[:values][:aggregate]) == (d_model, seq_len)
    end

    # -----------------------------------------------------------------
    @testset "build_db_lux_model convenience" begin
        d_model = 4
        model, D = LuxExt.build_db_lux_model(d_model)

        ps, st = Lux.setup(rng, model)
        inputs = Dict(:State => randn(rng, Float32, d_model, 2))
        result, _ = model(inputs, ps, st)
        @test result[:losses][:obstruction] >= 0
    end

    # -----------------------------------------------------------------
    @testset "build_gt_lux_model convenience" begin
        d_model = 8
        model, D = LuxExt.build_gt_lux_model(d_model; n_heads=2)

        ps, st = Lux.setup(rng, model)
        seq_len = 4
        inputs = Dict(
            :Tokens => randn(rng, Float32, d_model, seq_len),
            :Incidence => Float32.(ones(seq_len, seq_len))
        )
        result, _ = model(inputs, ps, st)
        @test size(result[:values][:aggregate]) == (d_model, seq_len)
    end

    # -----------------------------------------------------------------
    @testset "build_basket_rocket_lux_model convenience" begin
        d_model = 8
        model, D = LuxExt.build_basket_rocket_lux_model(d_model; n_heads=2)

        @test model isa LuxExt.LuxDiagramModel
        @test D isa Diagram
        @test haskey(D.losses, :draft_repair_consistency)

        ps, st = Lux.setup(rng, model)
        seq_len = 4
        mask = Float32.(ones(seq_len, seq_len))
        inputs = Dict(
            D.ports[:input].ref => randn(rng, Float32, d_model, seq_len),
            D.ports[:draft_relation].ref => mask,
            D.ports[:repair_relation].ref => mask
        )
        result, _ = model(inputs, ps, st)
        @test size(result[:values][D.ports[:draft].ref]) == (d_model, seq_len)
        @test size(result[:values][D.ports[:output].ref]) == (d_model, seq_len)
        @test result[:losses][:draft_repair_consistency] >= 0
    end

    # -----------------------------------------------------------------
    @testset "RelationInferenceLayer learns a relation matrix" begin
        d_model = 8
        seq_len = 5
        layer = RelationInferenceLayer(d_model; name=:infer_relation)
        ps, st = Lux.setup(rng, layer)
        source = randn(rng, Float32, d_model, seq_len)
        relation, _ = layer(source, ps, st)
        @test size(relation) == (seq_len, seq_len)
        @test all(relation .>= 0.0f0)
        @test all(relation .<= 1.0f0)
        @test all(LinearAlgebra.diag(relation) .≈ 1.0f0)
    end

    # -----------------------------------------------------------------
    @testset "build_topocoend_lux_model convenience" begin
        d_model = 8
        model, D = build_topocoend_lux_model(d_model; n_heads=2)

        ps, st = Lux.setup(rng, model)
        seq_len = 4
        inputs = Dict(
            D.ports[:input].ref => randn(rng, Float32, d_model, seq_len),
        )
        result, _ = model(inputs, ps, st)
        @test size(result[:values][D.ports[:learned_relation].ref]) == (seq_len, seq_len)
        @test size(result[:values][D.ports[:output].ref]) == (d_model, seq_len)
    end

    # -----------------------------------------------------------------
    @testset "build_horn_lux_model convenience" begin
        d_model = 6
        model, D = build_horn_lux_model(d_model)

        ps, st = Lux.setup(rng, model)
        inputs = Dict(:Vertex0 => randn(rng, Float32, d_model, 3))
        result, _ = model(inputs, ps, st)
        @test size(result[:values][:horn_boundary]) == (d_model, 3)
        @test size(result[:values][:d02]) == (d_model, 3)
        @test result[:losses][:horn_obstruction] >= 0
    end

    # -----------------------------------------------------------------
    @testset "build_higher_horn_lux_model convenience" begin
        d_model = 6
        model, D = build_higher_horn_lux_model(d_model;
            config=HigherHornConfig(filler_faces=[:d03_exact, :d03_relaxed]))

        ps, st = Lux.setup(rng, model)
        inputs = Dict(:Vertex0 => randn(rng, Float32, d_model, 2))
        result, _ = model(inputs, ps, st)
        @test size(result[:values][:higher_horn_boundary]) == (d_model, 2)
        @test size(result[:values][:d03_exact]) == (d_model, 2)
        @test size(result[:values][:d03_relaxed]) == (d_model, 2)
        @test result[:losses][:higher_horn_obstruction] >= 0
    end

    # -----------------------------------------------------------------
    @testset "build_bisimulation_quotient_lux_model convenience" begin
        d_model = 5
        model, D = build_bisimulation_quotient_lux_model(d_model)

        ps, st = Lux.setup(rng, model)
        seq_len = 3
        inputs = Dict(
            D.ports[:relation].ref => randn(rng, Float32, d_model, seq_len),
        )
        result, _ = model(inputs, ps, st)
        @test size(result[:values][D.ports[:left_behavior].ref]) == (d_model, seq_len)
        @test size(result[:values][D.ports[:right_behavior].ref]) == (d_model, seq_len)
        @test size(result[:values][D.ports[:output].ref]) == (d_model, seq_len)
        @test result[:losses][:behavior_quotient_coeq_loss] >= 0
    end

    # -----------------------------------------------------------------
    @testset "Mixed neural/symbolic execution" begin
        # Build a diagram with some neural ops and some symbolic ops
        D = Diagram(:MixedModel)
        add_object!(D, :Input; kind=:state)
        add_object!(D, :Hidden; kind=:state)
        add_object!(D, :Relation; kind=:relation)
        add_object!(D, :Output; kind=:state)

        # Neural morphism: Input -> Hidden
        add_morphism!(D, :encode, :Input, :Hidden)
        # Symbolic Kan (left-Kan with sum reducer)
        add_left_kan!(D, :aggregate; source=:Hidden, along=:Relation,
                      target=:Output, reducer=:sum)

        d_model = 4
        enc = LuxExt.DiagramDenseLayer(d_model, d_model; name=:encode)
        model = compile_to_lux(D; morphism_layers=Dict(:encode => enc))

        ps, st = Lux.setup(rng, model)

        # Use dict-based input for the symbolic sum reducer
        inputs = Dict(
            :Input => randn(rng, Float32, d_model, 3),
            :Relation => Dict("ctx" => [1, 2, 3])
        )
        result, _ = model(inputs, ps, st)
        @test haskey(result[:values], :encode)
        @test haskey(result[:values], :aggregate)
    end

    # -----------------------------------------------------------------
    @testset "predict_detach_source uses stop-gradient semantics" begin
        using Zygote: gradient

        hidden = reshape(Float32[1, 2, 3, 4], 2, 2, 1)
        pos = zeros(Float32, 2, 2, 1)
        embed = Float32[1 0; 0 1]
        head = Float32[2 1; -1 3]
        readout = reshape(Float32[1, -2, 3, -4], 2, 2, 1)

        function detached_loss(head_w)
            logits = reshape(head_w * reshape(hidden, 2, :), 2, 2, 1)
            source = FunctorFlow.predict_detach_source(logits, embed; position_bias=pos)
            sum(source .* readout)
        end

        function leaky_loss(head_w)
            logits = reshape(head_w * reshape(hidden, 2, :), 2, 2, 1)
            probs = LuxExt._softmax_cols(logits)
            probs_2d = reshape(probs, size(probs, 1), :)
            predicted = reshape(embed * probs_2d, size(embed, 1), 2, 1)
            source = predicted .+ pos
            sum(source .* readout)
        end

        grad_detached = something(gradient(detached_loss, head)[1], zeros(Float32, size(head)...))
        grad_leaky = gradient(leaky_loss, head)[1]

        @test isapprox(grad_detached, zeros(Float32, size(head)...); atol=1f-6)
        @test !isapprox(grad_leaky, zeros(Float32, size(head)...); atol=1f-6)
    end

end
