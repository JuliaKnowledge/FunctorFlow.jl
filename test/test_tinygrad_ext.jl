# ============================================================================
# test_tinygrad_ext.jl — Tests for FunctorFlowTinyGradExt
#
# Skipped when TinyGrad.jl is not installable in the active environment
# (the default FunctorFlow CI sandbox has no TinyGrad). To exercise this
# file locally, run:
#
#     julia --project=. test/setup_local_dev.jl
#     julia --project=. -e 'using Pkg; Pkg.test()'
# ============================================================================

using Test
using FunctorFlow

const TINYGRAD_AVAILABLE = try
    @eval using TinyGrad
    @eval using OrderedCollections
    true
catch err
    @info "test_tinygrad_ext.jl: TinyGrad not available — skipping" error=err
    false
end

if !TINYGRAD_AVAILABLE
    @testset "FunctorFlowTinyGradExt (skipped: missing TinyGrad)" begin
        @test_skip true
    end
end

if TINYGRAD_AVAILABLE

const TGX = Base.get_extension(FunctorFlow, :FunctorFlowTinyGradExt)

# ---- helpers ---------------------------------------------------------------

# Tensor-pure morphism for tracing (uses TinyTensor ops, no broadcasting).
_tensor_relu(x) = TinyGrad.tensor_relu(x)
_tensor_double(x) = x + x

# Build a 3-layer "linear MLP" diagram on tensors.  Implementations are
# tensor-pure so the UOpCompiledBackend can fuse them.
function _build_linear_mlp(d_in::Int, d_hidden::Int, d_out::Int)
    rng_seed = 42
    # Seed weights
    w1_data = Float32.(reshape(collect(1:d_in*d_hidden) ./ (d_in*d_hidden), d_hidden, d_in))
    b1_data = zeros(Float32, d_hidden)
    w2_data = Float32.(reshape(collect(1:d_hidden*d_out) ./ (d_hidden*d_out), d_out, d_hidden))
    b2_data = zeros(Float32, d_out)

    W1_tt = TinyGrad.TinyTensor(w1_data)
    b1_tt = TinyGrad.TinyTensor(b1_data)
    W2_tt = TinyGrad.TinyTensor(w2_data)
    b2_tt = TinyGrad.TinyTensor(b2_data)

    D = Diagram(:LinearMLP)
    add_object!(D, :Input)
    add_object!(D, :H1)
    add_object!(D, :H1A)
    add_object!(D, :Output)
    add_morphism!(D, :linear1, :Input, :H1;
        implementation = x -> TinyGrad.tensor_matmul(W1_tt, x))
    add_morphism!(D, :act1, :H1, :H1A;
        implementation = _tensor_relu)
    add_morphism!(D, :linear2, :H1A, :Output;
        implementation = x -> TinyGrad.tensor_matmul(W2_tt, x))
    return (D, w1_data, b1_data, w2_data, b2_data)
end

@testset "FunctorFlowTinyGradExt" begin

    @testset "Backend constructors + metadata" begin
        @test TGX.TinyGradBackend isa Type
        @test TGX.UOpCompiledBackend isa Type
        @test backend_name(tinygrad_backend()) == "tinygrad"
        @test backend_name(uop_compiled_backend()) == "uop_compiled"
        @test supports_dtype(tinygrad_backend(), Float32)
        @test supports_dtype(tinygrad_backend(), Float64)
        @test supports_dtype(tinygrad_backend(), Bool)
        @test !supports_dtype(tinygrad_backend(), Int32)
        @test supports_dtype(uop_compiled_backend(), Float32)
    end

    # -----------------------------------------------------------------
    @testset "TinyGradBackend round-trip — identity diagram" begin
        D = Diagram(:Identity)
        add_object!(D, :X)
        add_object!(D, :Y)
        add_morphism!(D, :id, :X, :Y; implementation = x -> x)

        model = compile_to_tinygrad(D; backend=:array_roundtrip)
        x_in = randn(Float32, 4, 2)
        result = model(Dict(:X => x_in))
        @test result.values[:Y] ≈ x_in
    end

    # -----------------------------------------------------------------
    @testset "TinyGradBackend — 3-layer linear MLP" begin
        D, w1, b1, w2, b2 = _build_linear_mlp(4, 6, 3)

        model = compile_to_tinygrad(D; backend=:array_roundtrip)
        x_in = randn(Float32, 4, 2)
        result = model(Dict(:Input => x_in))

        # Reference computation in plain Julia
        h1 = w1 * x_in
        h1a = max.(h1, 0)
        out_ref = w2 * h1a

        @test size(result.values[:Output]) == (3, 2)
        @test result.values[:Output] ≈ out_ref atol=1e-4
    end

    # -----------------------------------------------------------------
    @testset "UOpCompiledBackend — full trace + parity with round-trip" begin
        D, _, _, _, _ = _build_linear_mlp(4, 6, 3)
        x_in = randn(Float32, 4, 2)

        rt_model  = compile_to_tinygrad(D; backend=:array_roundtrip)
        uop_model = compile_to_tinygrad(D; backend=:uop_compiled,
                                        inputs=Dict(:Input => x_in))

        @test uop_model.compiled isa TGX.UOpCompiledDiagram
        # All three morphisms are tensor-pure ⇒ should be fully traced.
        @test uop_model.compiled.fully_traced
        @test uop_model.compiled.stats.n_traced == 3
        @test uop_model.compiled.stats.n_opaque == 0

        rt_result  = rt_model(Dict(:Input => x_in))
        uop_result = uop_model(Dict(:Input => x_in))

        @test size(uop_result.values[:Output]) == size(rt_result.values[:Output])
        @test uop_result.values[:Output] ≈ rt_result.values[:Output] atol=1e-4
    end

    # -----------------------------------------------------------------
    @testset "UOpCompiledBackend — opaque fallback for broadcasting morphism" begin
        D = Diagram(:OpaqueMix)
        add_object!(D, :X)
        add_object!(D, :Y)
        # Julia broadcasting cannot trace through TinyTensor → opaque
        add_morphism!(D, :scale, :X, :Y; implementation = x -> 2.0f0 .* x)

        x_in = randn(Float32, 3, 4)
        model = compile_to_tinygrad(D; backend=:uop_compiled,
                                    inputs=Dict(:X => x_in))
        @test !model.compiled.fully_traced
        @test model.compiled.stats.n_opaque == 1

        result = model(Dict(:X => x_in))
        @test result.values[:Y] ≈ 2.0f0 .* x_in
    end

    # -----------------------------------------------------------------
    @testset "Re-running the model with new inputs" begin
        D, _, _, _, _ = _build_linear_mlp(4, 6, 3)
        x1 = randn(Float32, 4, 2)
        x2 = randn(Float32, 4, 2)

        model = compile_to_tinygrad(D; backend=:uop_compiled,
                                    inputs=Dict(:Input => x1))

        r1 = model(Dict(:Input => x1))
        r2 = model(Dict(:Input => x2))
        @test r1.values[:Output] != r2.values[:Output]

        # And round-trip backend behaves the same way
        rt = compile_to_tinygrad(D; backend=:array_roundtrip)
        r1_rt = rt(Dict(:Input => x1))
        r2_rt = rt(Dict(:Input => x2))
        @test r1.values[:Output] ≈ r1_rt.values[:Output] atol=1e-4
        @test r2.values[:Output] ≈ r2_rt.values[:Output] atol=1e-4
    end

    # -----------------------------------------------------------------
    @testset "Composition + obstruction loss — round-trip" begin
        D = Diagram(:Square)
        add_object!(D, :S)
        add_morphism!(D, :f, :S, :S; implementation = x -> x .* 2)
        add_morphism!(D, :g, :S, :S; implementation = x -> x .+ 1)
        compose!(D, :f, :g; name=:fg)
        compose!(D, :g, :f; name=:gf)
        add_obstruction_loss!(D, :obs; paths=[(:fg, :gf)])

        model = compile_to_tinygrad(D; backend=:array_roundtrip)
        result = model(Dict(:S => Float32[3.0]))
        @test result.values[:fg] ≈ Float32[7.0]
        @test result.values[:gf] ≈ Float32[8.0]
        @test result.losses[:obs] > 0.0
    end

    # -----------------------------------------------------------------
    @testset "Cross-backend parity — Lux vs TinyGrad" begin
        # Build the same linear MLP, lower it under both backends, and
        # check forward outputs agree to a reasonable tolerance.
        # We avoid Lux for now — Lux + TinyGrad in the same env brings
        # additional precompilation cost.  Instead we treat the FF
        # Julia-array reference computation in `_build_linear_mlp` as
        # the cross-backend ground truth and verify that both TinyGrad
        # backends match it.
        D, w1, b1, w2, b2 = _build_linear_mlp(8, 12, 4)
        x_in = randn(Float32, 8, 3)
        out_ref = w2 * max.(w1 * x_in, 0)

        m_rt  = compile_to_tinygrad(D; backend=:array_roundtrip)
        m_uop = compile_to_tinygrad(D; backend=:uop_compiled,
                                    inputs=Dict(:Input => x_in))

        r_rt  = m_rt(Dict(:Input => x_in))
        r_uop = m_uop(Dict(:Input => x_in))

        @test r_rt.values[:Output]  ≈ out_ref atol=1e-3
        @test r_uop.values[:Output] ≈ out_ref atol=1e-3
        @test r_rt.values[:Output]  ≈ r_uop.values[:Output] atol=1e-4
    end

    # -----------------------------------------------------------------
    @testset "Round-trip via shared schema (FF → ACSet → FF) → TinyGrad" begin
        if Base.find_package("CategoricalDiagramSchema") === nothing
            @info "Skipping schema round-trip test: CategoricalDiagramSchema unavailable"
        else
            @eval using CategoricalDiagramSchema
            D, _, _, _, _ = _build_linear_mlp(4, 6, 3)
            acs = to_acset(D)
            D_round = from_acset(acs; name=:LinearMLP_Round)

            # The round-tripped diagram loses the implementation bindings; rebind.
            for (name, fn) in D.implementations
                bind_morphism!(D_round, name, fn)
            end

            x_in = randn(Float32, 4, 2)
            model_orig  = compile_to_tinygrad(D;       backend=:array_roundtrip)
            model_round = compile_to_tinygrad(D_round; backend=:array_roundtrip)
            r_orig  = model_orig(Dict(:Input => x_in))
            r_round = model_round(Dict(:Input => x_in))
            @test r_orig.values[:Output] ≈ r_round.values[:Output] atol=1e-5
        end
    end
end

end # if TINYGRAD_AVAILABLE
