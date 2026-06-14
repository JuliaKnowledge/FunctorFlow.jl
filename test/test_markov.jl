# ============================================================================
# test_markov.jl — Markov categories (probability & causality)
# ============================================================================

using Test
using FunctorFlow
const Cat = FunctorFlow.Cat

D(pairs) = Cat.Dist(Dict{Any, Rational{Int}}(pairs))

@testset "Distributions and Chapman–Kolmogorov composition" begin
    @test_throws ArgumentError D([:a => 1 // 2, :b => 1 // 4])     # doesn't sum to 1
    @test Cat.dirac(:a) == D([:a => 1 // 1])

    f = Cat.StochMap([:s], [:0, :1], Dict{Any, Cat.Dist}(:s => D([:0 => 1 // 3, :1 => 2 // 3])))
    flip = Cat.StochMap([:0, :1], [:0, :1], Dict{Any, Cat.Dist}(
        :0 => D([:0 => 9 // 10, :1 => 1 // 10]),
        :1 => D([:0 => 1 // 10, :1 => 9 // 10])))
    h = Cat.markov_compose(f, flip)
    # P(out=1) = 1/3·1/10 + 2/3·9/10 = 19/30
    @test Cat.prob(h.kernel[:s], :1) == 19 // 30
end

@testset "Markov-category laws & structure maps" begin
    flip = Cat.StochMap([:0, :1], [:0, :1], Dict{Any, Cat.Dist}(
        :0 => D([:0 => 9 // 10, :1 => 1 // 10]),
        :1 => D([:0 => 1 // 10, :1 => 9 // 10])))
    @test Cat.markov_laws([flip, Cat.markov_id([:0, :1])])
    @test Cat.is_deterministic(Cat.markov_id([:0, :1]))
    @test !Cat.is_deterministic(flip)
    # copy is the comonoid; discard is the counit
    @test Cat.markov_copy([:0, :1]).kernel[:0] == Cat.dirac((:0, :0))
    @test Cat.markov_discard([:0, :1]).kernel[:1] == Cat.dirac(())
    # tensor of independent kernels multiplies probabilities
    t = Cat.markov_tensor(flip, flip)
    @test Cat.prob(t.kernel[(:0, :0)], (:1, :1)) == (1 // 10) * (1 // 10)
end

@testset "Causal DAG as a Markov-category morphism" begin
    G = CausalDAG(; nodes=[:Z, :X, :Y], directed=[(:Z, :X), (:Z, :Y)])
    mech = Dict{Symbol, Any}(
        :Z => (pv -> D([0 => 1 // 2, 1 => 1 // 2])),
        :X => (pv -> D([pv[1] => 3 // 4, 1 - pv[1] => 1 // 4])),
        :Y => (pv -> D([pv[1] => 1 // 1])))           # Y copies Z
    res = causal_markov_kernel(G, mech)
    @test res.order == [:X, :Y, :Z]
    @test sum(values(res.dist.support)) == 1 // 1
    # since Y copies Z, the joint never has Y ≠ Z
    @test all(t[findfirst(==(:Y), res.order)] == t[findfirst(==(:Z), res.order)]
              for t in keys(res.dist.support))
end

@testset "Bayesian update is disintegration" begin
    prior = D([0 => 1 // 2, 1 => 1 // 2])
    like = Cat.StochMap([0, 1], [0, 1], Dict{Any, Cat.Dist}(
        0 => D([0 => 4 // 5, 1 => 1 // 5]),
        1 => D([0 => 1 // 5, 1 => 4 // 5])))
    post = Cat.bayes_update(prior, like, 1)
    # P(Z=1 | Y=1) = (1/2·4/5)/((1/2·1/5)+(1/2·4/5)) = 4/5
    @test Cat.prob(post, 1) == 4 // 5
    @test Cat.prob(post, 0) == 1 // 5
    @test_throws ArgumentError Cat.bayes_update(prior,
        Cat.StochMap([0, 1], [0], Dict{Any, Cat.Dist}(0 => Cat.dirac(0), 1 => Cat.dirac(0))), 1)
end
