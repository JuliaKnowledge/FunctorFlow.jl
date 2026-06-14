# ============================================================================
# test_presented.jl — finitely-presented categories (free + relations)
# ============================================================================

using Test
using FunctorFlow
const Cat = FunctorFlow.Cat

@testset "Commutative square vs free diamond" begin
    sq = Cat.commutative_square()
    # the relation f·h = g·k collapses the two a→d paths into one
    @test Cat.hom_cardinality(sq, :a, :d) == 1
    @test Cat.hom_cardinality(sq, :a, :b) == 1
    @test Cat.check_category_laws(sq)
    # contrast: the *free* diamond has two a→d paths
    free_diamond = FreeCat([:a, :b, :c, :d],
                           [(:f, :a, :b), (:g, :a, :c), (:h, :b, :d), (:k, :c, :d)])
    @test Cat.hom_cardinality(free_diamond, :a, :d) == 2

    # in the commutative square a is genuinely initial
    @test all(Cat.hom_cardinality(sq, :a, x) == 1 for x in (:a, :b, :c, :d))
    @test Cat.is_adjunction(Cat.initial_object_adjunction(sq, :a))
end

@testset "Presentation validation & normalisation" begin
    # a relation must be between parallel paths
    @test_throws ArgumentError Cat.FinPresentedCat([:a, :b, :c],
        [(:f, :a, :b), (:g, :b, :c)],
        [(PathMor(:a, :b, [:f]), PathMor(:b, :c, [:g]))])   # not parallel

    sq = Cat.commutative_square()
    # both a→d generator-paths normalise to the same representative
    @test Cat.normalize(sq, PathMor(:a, :d, [:f, :h])) == Cat.normalize(sq, PathMor(:a, :d, [:g, :k]))
    # identity & composition land in classes
    @test Cat.id(sq, :a) == PathMor(:a, :a, Symbol[])
    @test Cat.compose(sq, PathMor(:a, :b, [:f]), PathMor(:b, :d, [:h])) ==
          Cat.normalize(sq, PathMor(:a, :d, [:f, :h]))
end

@testset "Yoneda over a presented category" begin
    sq = Cat.commutative_square()
    yc = Cat.representable_functor(sq, :a)
    @test Cat.is_functorial(yc)
    @test Cat.yoneda_lemma_holds(sq, :a, yc)
    @test Cat.yoneda_lemma_holds(sq, :d, yc)
    @test Cat.is_representable(yc).representable
end

@testset "SetFunctors must respect relations" begin
    sq = Cat.commutative_square()
    # respects f·h = g·k  ⇒ functorial
    ok = Cat.SetFunctor(sq;
        ob_map=Dict(:a => Cat.FinSet([1]), :b => Cat.FinSet([:b1]),
                    :c => Cat.FinSet([:c1]), :d => Cat.FinSet([:d1])),
        edge_map=Dict(:f => Cat.FinFunction(Cat.FinSet([1]), Cat.FinSet([:b1]), [1 => :b1]),
                      :g => Cat.FinFunction(Cat.FinSet([1]), Cat.FinSet([:c1]), [1 => :c1]),
                      :h => Cat.FinFunction(Cat.FinSet([:b1]), Cat.FinSet([:d1]), [:b1 => :d1]),
                      :k => Cat.FinFunction(Cat.FinSet([:c1]), Cat.FinSet([:d1]), [:c1 => :d1])))
    @test Cat.is_functorial(ok)

    # violates the relation (the two a→d composites disagree) ⇒ NOT functorial
    bad = Cat.SetFunctor(sq;
        ob_map=Dict(:a => Cat.FinSet([1]), :b => Cat.FinSet([:b1]),
                    :c => Cat.FinSet([:c1]), :d => Cat.FinSet([:d1, :d2])),
        edge_map=Dict(:f => Cat.FinFunction(Cat.FinSet([1]), Cat.FinSet([:b1]), [1 => :b1]),
                      :g => Cat.FinFunction(Cat.FinSet([1]), Cat.FinSet([:c1]), [1 => :c1]),
                      :h => Cat.FinFunction(Cat.FinSet([:b1]), Cat.FinSet([:d1, :d2]), [:b1 => :d1]),
                      :k => Cat.FinFunction(Cat.FinSet([:c1]), Cat.FinSet([:d1, :d2]), [:c1 => :d2])))
    @test !Cat.is_functorial(bad)
end
