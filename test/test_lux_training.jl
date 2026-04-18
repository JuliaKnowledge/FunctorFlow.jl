# ============================================================================
# test_lux_training.jl — End-to-end training test for the Lux extension
#
# Asserts that gradients flow through compile_to_lux and that 100 Adam
# steps actually decrease loss. Closes audit P0-FF-2.
# ============================================================================

using Test
using FunctorFlow
using Lux
using LuxCore
using Optimisers
using Zygote: gradient
using Random

const _LuxExt = let ext = Base.get_extension(FunctorFlow, :FunctorFlowLuxExt)
    ext === nothing && error("FunctorFlowLuxExt not loaded — Lux/LuxCore must be in test target")
    ext
end

@testset "Lux training loop — loss decreases" begin
    rng = Random.MersenneTwister(0)

    in_dim, hidden_dim, out_dim = 32, 16, 4

    D = FunctorFlow.Diagram(:train_test)
    FunctorFlow.add_object!(D, :input;  kind=:state)
    FunctorFlow.add_object!(D, :hidden; kind=:state)
    FunctorFlow.add_object!(D, :output; kind=:state)
    FunctorFlow.add_morphism!(D, :l1, :input,  :hidden)
    FunctorFlow.add_morphism!(D, :l2, :hidden, :output)

    morphism_layers = Dict{Symbol, LuxCore.AbstractLuxLayer}(
        :l1 => _LuxExt.DiagramDenseLayer(in_dim, hidden_dim; activation=Lux.relu, name=:l1),
        :l2 => _LuxExt.DiagramDenseLayer(hidden_dim, out_dim; name=:l2),
    )

    model = compile_to_lux(D; morphism_layers=morphism_layers)
    @test model isa _LuxExt.LuxDiagramModel

    ps, st = Lux.setup(rng, model)

    batch = 16
    # Synthetic linearly-separable target: Y = W*X + noise, learnable.
    true_W = randn(rng, Float32, out_dim, in_dim)
    X = randn(rng, Float32, in_dim, batch)
    Y = true_W * X .+ 0.05f0 .* randn(rng, Float32, out_dim, batch)

    function loss_fn(p)
        result, _ = model(Dict(:input => X), p, st)
        ŷ = result[:values][:output]
        sum((ŷ .- Y) .^ 2) / length(Y)
    end

    initial_loss = loss_fn(ps)
    @test isfinite(initial_loss)
    @test initial_loss > 0

    opt_state = Optimisers.setup(Optimisers.Adam(1f-2), ps)
    for _ in 1:100
        gs = gradient(loss_fn, ps)[1]
        @test gs !== nothing
        opt_state, ps = Optimisers.update(opt_state, ps, gs)
    end

    final_loss = loss_fn(ps)
    @info "Lux training" initial_loss final_loss ratio = final_loss / initial_loss
    @test isfinite(final_loss)
    @test final_loss < 0.5 * initial_loss
end
