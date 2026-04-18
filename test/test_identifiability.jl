# ============================================================================
# test_identifiability.jl — Shpitser-Pearl ID algorithm tests
# ============================================================================

using FunctorFlow
using Test

@testset "CausalDAG primitives" begin
    G = CausalDAG(nodes=[:Z, :X, :Y], directed=[(:Z,:X),(:X,:Y),(:Z,:Y)])
    @test G.nodes == [:Z, :X, :Y]
    @test length(G.directed) == 3
    @test isempty(G.bidirected)

    @test ancestors_inclusive(G, [:Y]) == [:Z, :X, :Y]
    @test ancestors_inclusive(G, [:X]) == [:Z, :X]
    @test ancestors_inclusive(G, [:Z]) == [:Z]

    Gsub = subgraph(G, [:X, :Y])
    @test Gsub.nodes == [:X, :Y]
    @test (:X, :Y) in Gsub.directed
    @test !((:Z, :X) in Gsub.directed)

    Gxbar = remove_incoming(G, [:X])
    @test (:X, :Y) in Gxbar.directed
    @test !((:Z, :X) in Gxbar.directed)
    @test (:Z, :Y) in Gxbar.directed

    @test topological_order(G) == [:Z, :X, :Y]

    G2 = CausalDAG(nodes=[:A,:B,:C,:D], bidirected=[(:A,:B),(:B,:C)])
    cs = c_components(G2)
    sets = Set(Set(c) for c in cs)
    @test Set([:A,:B,:C]) in sets
    @test Set([:D]) in sets
end

@testset "Shpitser-Pearl ID — canonical examples" begin

    # ---- 1. Backdoor admissible: Z → X → Y, Z → Y ----------------------
    @testset "backdoor admissible" begin
        G = CausalDAG(nodes=[:Z,:X,:Y],
                      directed=[(:Z,:X),(:X,:Y),(:Z,:Y)])
        r = identify_effect(G, [:Y], [:X])
        @test r.identifiable
        @test r.expression !== nothing
        @test r.failure_reason === nothing
        @test r.algorithm == :id
        s = pretty_print(r.expression)
        # Expected: Σ_Z P(Z) · P(Y | Z, X) — both summands and conditional
        # should be visible.
        @test occursin("Σ", s)
        @test occursin("Z", s)
        @test occursin("Y", s)
        # Backdoor criterion: {Z} is a valid adjustment set.
        @test is_backdoor_admissible(G, [:X], [:Y], [:Z])
        @test !is_backdoor_admissible(G, [:X], [:Y], Symbol[])
    end

    # ---- 2. Front-door: X → M → Y, X ↔ Y --------------------------------
    @testset "front-door" begin
        G = CausalDAG(nodes=[:X,:M,:Y],
                      directed=[(:X,:M),(:M,:Y)],
                      bidirected=[(:X,:Y)])
        r = identify_effect(G, [:Y], [:X])
        @test r.identifiable
        @test r.expression !== nothing
        @test r.algorithm == :id
        s = pretty_print(r.expression)
        # Expected: Σ_M P(M|X) · Σ_{X'} P(X') P(Y | X', M)
        @test occursin("M", s)
        @test occursin("Σ", s)
        # No back-door admissible set (X ↔ Y blocks all of them).
        @test !is_backdoor_admissible(G, [:X], [:Y], Symbol[])
        @test !is_backdoor_admissible(G, [:X], [:Y], [:M])  # M is descendant
    end

    # ---- 3. Bow arc: X → Y, X ↔ Y (NON-identifiable) -------------------
    @testset "bow arc" begin
        G = CausalDAG(nodes=[:X,:Y],
                      directed=[(:X,:Y)],
                      bidirected=[(:X,:Y)])
        r = identify_effect(G, [:Y], [:X])
        @test !r.identifiable
        @test r.failure_reason == :hedge
        @test r.witness !== nothing
        @test Set(r.witness.F) == Set([:X, :Y])
    end

    # ---- 4. W-graph: X → Y, W → X, W → Y, X ↔ Y (NON-identifiable) -----
    @testset "W-graph (hedge)" begin
        G = CausalDAG(nodes=[:W,:X,:Y],
                      directed=[(:W,:X),(:W,:Y),(:X,:Y)],
                      bidirected=[(:X,:Y)])
        r = identify_effect(G, [:Y], [:X])
        @test !r.identifiable
        @test r.failure_reason == :hedge
        @test r.witness !== nothing
    end

    # ---- 5. Tian (2002) example: X1 → X2 → Y with X1 ↔ Y ---------------
    @testset "Tian 2002 — three observed, one hidden" begin
        G = CausalDAG(nodes=[:X1,:X2,:Y],
                      directed=[(:X1,:X2),(:X2,:Y)],
                      bidirected=[(:X1,:Y)])
        r = identify_effect(G, [:Y], [:X1])
        @test r.identifiable
        @test r.expression !== nothing
        s = pretty_print(r.expression)
        @test occursin("X2", s)
    end

    # ---- 6. Pearl's napkin graph ---------------------------------------
    # Variant: Z → W → X → Y, with Z ↔ X and Z ↔ Y as latent confounders.
    # P(Y | do(X)) is identifiable in this graph (Pearl & Mackenzie 2018).
    @testset "Pearl's napkin" begin
        G = CausalDAG(nodes=[:Z,:W,:X,:Y],
                      directed=[(:Z,:W),(:W,:X),(:X,:Y)],
                      bidirected=[(:Z,:X),(:Z,:Y)])
        r = identify_effect(G, [:Y], [:X])
        @test r.identifiable
        @test r.expression !== nothing
    end

    # ---- 7. Sequential do on a chain -----------------------------------
    # X1 → X2 → X3 → Y, do(X1, X2). g-formula:
    # Σ_{X3} P(X3 | X1, X2) · P(Y | X1, X2, X3)
    @testset "sequential do (g-formula)" begin
        G = CausalDAG(nodes=[:X1,:X2,:X3,:Y],
                      directed=[(:X1,:X2),(:X2,:X3),(:X3,:Y)])
        r = identify_effect(G, [:Y], [:X1, :X2])
        @test r.identifiable
        @test r.expression !== nothing
        s = pretty_print(r.expression)
        @test occursin("X3", s)
    end
end

@testset "ID algorithm — edge cases" begin
    # do(∅) reduces to the marginal P(Y).
    G = CausalDAG(nodes=[:X,:Y], directed=[(:X,:Y)])
    r = identify_effect(G, [:Y], Symbol[])
    @test r.identifiable
    @test r.expression !== nothing

    # No causal path from X to Y: P(Y|do(X)) = P(Y).
    G2 = CausalDAG(nodes=[:X,:Y], directed=Tuple{Symbol,Symbol}[])
    r2 = identify_effect(G2, [:Y], [:X])
    @test r2.identifiable

    # Disjointness check.
    G3 = CausalDAG(nodes=[:A,:B], directed=[(:A,:B)])
    @test_throws ArgumentError identify_effect(G3, [:A], [:A])

    # Cycle detection.
    @test_throws ArgumentError topological_order(
        CausalDAG(nodes=[:A,:B], directed=[(:A,:B),(:B,:A)]))
end

@testset "IdentifiabilityResult shape and show" begin
    G = CausalDAG(nodes=[:Z,:X,:Y], directed=[(:Z,:X),(:X,:Y),(:Z,:Y)])
    r = identify_effect(G, [:Y], [:X])
    @test r isa IdentifiabilityResult
    io = IOBuffer()
    show(io, MIME"text/plain"(), r)
    s = String(take!(io))
    @test occursin("identifiable", s)
    @test occursin("expression", s)

    # Hedge gets pretty-printed too.
    Gbow = CausalDAG(nodes=[:X,:Y], directed=[(:X,:Y)], bidirected=[(:X,:Y)])
    rb = identify_effect(Gbow, [:Y], [:X])
    io = IOBuffer()
    show(io, MIME"text/plain"(), rb)
    s = String(take!(io))
    @test occursin("hedge", s)
end
