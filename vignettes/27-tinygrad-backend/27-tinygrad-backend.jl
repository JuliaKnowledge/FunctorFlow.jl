#!/usr/bin/env julia
# 27 — TinyGrad backend for FunctorFlow
#
# This vignette shows how to compile a FunctorFlow `Diagram` to two
# TinyGrad-backed execution engines:
#
#   1. `TinyGradBackend` — round-trips Julia arrays through `TinyTensor`
#      for every morphism. Always works, even when reducers/morphisms are
#      opaque Julia callables. Good baseline; correctness-equivalent to
#      the default Julia backend.
#
#   2. `UOpCompiledBackend` — attempts to *trace* each morphism into the
#      shared TinyGrad UOp DAG. When all ops trace cleanly the entire
#      diagram becomes a single fused UOp graph that can be re-realised
#      with new inputs without re-walking Julia code (see the
#      `compiled.fully_traced` flag). Falls back to opaque execution per
#      op when tracing fails (e.g. for `:ket` reducers operating on
#      Dicts).
#
# Pattern parity with `CatNet.jl/ext/CatNetTinyGradExt`. Together these
# three packages (FunctorFlow, CatNet, TinyGrad) form the
# CDS ⇄ FF ⇄ CN ⇄ TinyGrad shared-schema pipeline.
#
# ## Environment note
#
# TinyGrad's `Symbolics.jl 7` dep transitively requires
# `MultivariatePolynomials ≥ 0.5.12`, which conflicts with FF's
# `Catlab → GATlab → DataStructures = "0.18"` (whose latest
# `MultivariatePolynomials` is 0.5.9). The two packages cannot resolve
# in a single project today.
#
# **Workaround**: run this vignette in a sandbox env that has FunctorFlow
# *as a dev source* with Catlab disabled, or wait for upstream
# DataStructures = "0.19" support across Compose / GATlab / ACSets.
# When Catlab is dropped from the env, `Pkg.develop("TinyGrad")` resolves
# cleanly and `using TinyGrad` triggers `FunctorFlowTinyGradExt` to load.

using FunctorFlow
using TinyGrad   # triggers ext loading; will fail in standard FF env

# ## 1. Build a tiny diagram: 2-layer MLP -------------------------------

W1 = randn(Float32, 4, 3)   # 3→4
W2 = randn(Float32, 2, 4)   # 4→2

D = @diagram :mlp begin
    @object x  shape=(3,)  dtype=Float32
    @object h  shape=(4,)  dtype=Float32
    @object y  shape=(2,)  dtype=Float32

    @morphism layer1 x => h impl = (xi -> max.(W1 * xi, 0f0))
    @morphism layer2 h => y impl = (hi -> W2 * hi)
end

x_in = randn(Float32, 3)

# ## 2. Compile to TinyGradBackend (round-trip) -------------------------

m_rt = compile_to_tinygrad(D; mode = :round_trip)
y_rt = m_rt(Dict("x" => x_in))["y"]
@info "TinyGradBackend output" y_rt

# ## 3. Compile to UOpCompiledBackend (lazy trace) ----------------------

m_uop = compile_to_tinygrad(D; mode = :uop)
y_uop = m_uop(Dict("x" => x_in))["y"]
@info "UOpCompiledBackend output" y_uop fully_traced = m_uop.compiled.fully_traced

# Both should match the reference Julia execution to within Float32
# round-off:
y_ref = compile_to_callable(D)(Dict("x" => x_in))["y"]
@assert isapprox(y_rt,  y_ref; atol = 1e-4)
@assert isapprox(y_uop, y_ref; atol = 1e-3)

# ## 4. Re-running the compiled UOp graph -------------------------------
#
# When `fully_traced == true`, the second invocation reuses the cached
# UOp DAG — it only swaps the input tensor data and re-runs the
# interpreter. This is the path you want for production inference.

x_in_2 = randn(Float32, 3)
y_uop_2 = m_uop(Dict("x" => x_in_2))["y"]
@info "UOp re-run" y_uop_2

# ## 5. Schema round-trip (CDS interop) ---------------------------------
#
# The same diagram serialises to / deserialises from a CategoricalDiagram
# ACSet via `to_acset`/`from_acset`. After round-trip you must rebind the
# Julia closures (FF can't serialise opaque code) before re-compiling:

if Base.find_package("CategoricalDiagramSchema") !== nothing
    A = to_acset(D)
    D2 = from_acset(A)
    for (name, m) in D.morphisms
        D2.morphisms[name].implementation = m.implementation
    end
    m_rt2 = compile_to_tinygrad(D2; mode = :round_trip)
    y_rt2 = m_rt2(Dict("x" => x_in))["y"]
    @assert isapprox(y_rt2, y_ref; atol = 1e-4)
    @info "Schema round-trip OK"
end

# ## 6. Performance comparison (informal) -------------------------------

using BenchmarkTools
inputs = Dict("x" => x_in)
println("Reference Julia   : ", @benchmark compile_to_callable($D)($inputs))
println("TinyGradBackend   : ", @benchmark $m_rt($inputs))
println("UOpCompiledBackend: ", @benchmark $m_uop($inputs))

# Round-trip backend pays Julia↔TinyTensor conversion every call so it
# is typically the slowest. The UOp backend is faster on the second and
# subsequent calls thanks to the cached DAG.
