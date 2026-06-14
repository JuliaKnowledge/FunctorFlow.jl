# ============================================================================
# test_dynamics_recursion.jl — Rel/powerset, Poly (dynamics), F-algebras (folds)
# ============================================================================

using Test
using FunctorFlow
const Cat = FunctorFlow.Cat

@testset "Rel and the powerset (nondeterminism) monad" begin
    R = Cat.RelMap([:a, :b], [1, 2], [(:a, 1), (:a, 2), (:b, 2)])
    S = Cat.RelMap([1, 2], [:x], [(1, :x), (2, :x)])
    @test Cat.rel_compose(R, S).pairs == Set([(:a, :x), (:b, :x)])
    @test Cat.rel_compose(Cat.rel_id([:a, :b]), R) == R
    @test Cat.rel_compose(R, Cat.rel_id([1, 2])) == R
    @test Cat.rel_dagger(Cat.rel_dagger(R)) == R
    @test Cat.rel_laws([Cat.rel_id([:a, :b]), Cat.rel_id([:a, :b])])
    # Rel ≅ Kleisli(powerset)
    @test Cat.kleisli_to_rel([:a, :b], Dict(:a => Set([1, 2]), :b => Set([2]))).pairs == R.pairs
    @test Cat.powerset_unit(:a) == Set([:a])
    @test Cat.powerset_mult([Set([1, 2]), Set([2, 3])]) == Set([1, 2, 3])
end

@testset "Polynomial functors: a Moore machine IS a dependent lens" begin
    M = Cat.MooreMachine([:s0, :s1], [:i], [:o0, :o1],
        Dict((:s0, :i) => :s1, (:s1, :i) => :s0), Dict(:s0 => :o0, :s1 => :o1))
    φ = Cat.moore_to_poly(M)                # S·y^S → O·y^I
    @test Cat.is_poly_morphism(φ)
    @test φ.on_pos[:s0] == :o0              # readout
    @test φ.on_dir[:s0][:i] == :s1          # dynamics (s,i) ↦ next
    @test Cat.is_poly_morphism(Cat.poly_id(Cat.monomial([:s0, :s1])))
    # composition of poly morphisms is a poly morphism
    p = Cat.monomial([:s0, :s1])
    @test Cat.is_poly_morphism(Cat.poly_compose(Cat.poly_id(p), Cat.poly_id(p)))
end

@testset "F-algebras & catamorphisms (folds / recursion schemes)" begin
    sig = Cat.arithmetic_signature()
    terms = Cat.terms_upto(sig, 3)
    @test !isempty(terms)
    eval_alg = Cat.FAlgebra(collect(0:8), Dict{Symbol, Function}(
        :zero => (a -> 0), :one => (a -> 1), :add => (a -> a[1] + a[2]), :mul => (a -> a[1] * a[2])))
    t = Cat.Term(:mul, [Cat.Term(:add, [Cat.Term(:one), Cat.Term(:one)]), Cat.Term(:one)])
    @test Cat.cata(eval_alg, t) == 2        # (1+1)*1
    # a different algebra = a different fold (node count)
    size_alg = Cat.FAlgebra(collect(0:30), Dict{Symbol, Function}(
        :zero => (a -> 1), :one => (a -> 1), :add => (a -> 1 + a[1] + a[2]), :mul => (a -> 1 + a[1] + a[2])))
    @test Cat.cata(size_alg, t) == 5        # mul, add, one, one, one
    # the catamorphism is an F-algebra homomorphism on all terms up to depth 3
    @test Cat.cata_is_homomorphism(eval_alg, terms)
    @test Cat.cata_is_homomorphism(size_alg, terms)
end
