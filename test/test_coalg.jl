# ============================================================================
# test_coalg.jl — coalgebras / automata (state machines & RNNs)
# ============================================================================

using Test
using FunctorFlow
const Cat = FunctorFlow.Cat

# a Moore machine with a bisimilar pair (s1 ~ s2)
M = Cat.MooreMachine([:s0, :s1, :s2], [:a], [:x, :y],
    Dict((:s0, :a) => :s1, (:s1, :a) => :s2, (:s2, :a) => :s1),
    Dict(:s0 => :x, :s1 => :y, :s2 => :y))

@testset "Behaviour (run) and bisimulation" begin
    @test Cat.moore_run(M, :s0, [:a, :a, :a]) == [:x, :y, :y, :y]
    @test Cat.is_bisimulation(M, [(:s1, :s2), (:s2, :s1), (:s1, :s1), (:s2, :s2), (:s0, :s0)])
    @test !Cat.is_bisimulation(M, [(:s0, :s1)])     # different outputs
end

@testset "Bisimilarity (minimization classes)" begin
    cls = Cat.bisimilar(M)
    @test cls[:s1] == cls[:s2]                       # behaviourally equivalent
    @test cls[:s0] != cls[:s1]
end

@testset "Minimization and the quotient coalgebra morphism" begin
    Mmin = Cat.minimize(M)
    @test length(Mmin.states) == 2                   # {s1,s2} collapsed
    cls = Cat.bisimilar(M)
    qmap = Dict(s => Symbol("q", cls[s]) for s in M.states)
    @test Cat.coalgebra_morphism(M, Mmin, qmap)      # the quotient is a homomorphism
    # behaviour is preserved by minimization
    start = qmap[:s0]
    @test Cat.moore_run(Mmin, start, [:a, :a, :a]) == Cat.moore_run(M, :s0, [:a, :a, :a])
end

@testset "Bisimulation Lean certificate renders" begin
    cert = render_bisimulation_certificate(M, [(:s1, :s2), (:s2, :s1), (:s1, :s1), (:s2, :s2), (:s0, :s0)])
    @test occursin("MooreDecl", cert) && occursin("isBisimulation", cert) && occursin("native_decide", cert)
end
