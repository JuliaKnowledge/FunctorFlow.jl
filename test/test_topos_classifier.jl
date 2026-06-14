# ============================================================================
# test_topos_classifier.jl — subobject classifier of the presheaf topos
# ============================================================================

using Test
using FunctorFlow
const Cat = FunctorFlow.Cat

@testset "Ω over the arrow category" begin
    arrow = FreeCat([:a, :b], [(:f, :a, :b)])
    Ω = Cat.subobject_classifier(arrow)
    @test Cat.is_functorial(Ω)
    @test length(Cat.ob(Ω, :a)) == 3      # cosieves on a: ∅, {f}, {id_a, f}
    @test length(Cat.ob(Ω, :b)) == 2      # cosieves on b: ∅, {id_b}
    # the truth values are the maximal cosieves
    tru = Cat.omega_true(arrow)
    @test tru[:a] in Cat.ob(Ω, :a).elements
    @test length(tru[:a]) == 2            # {id_a, f}
end

@testset "Classification theorem χ⁻¹(true) = sub" begin
    arrow = FreeCat([:a, :b], [(:f, :a, :b)])
    X = Cat.SetFunctor(arrow;
        ob_map=Dict(:a => Cat.FinSet([1, 2]), :b => Cat.FinSet([:p, :q])),
        edge_map=Dict(:f => Cat.FinFunction(Cat.FinSet([1, 2]), Cat.FinSet([:p, :q]), [1 => :p, 2 => :q])))

    sub = Dict(:a => Set([1]), :b => Set([:p]))     # closed: X(f)(1)=:p ∈ sub[b]
    @test Cat.is_subfunctor(X, sub)
    χ = Cat.classify(X, sub)
    @test Cat.is_natural(χ)
    @test Cat.verify_classifies(X, sub, χ)          # the subobject-classifier theorem

    # not closed under X(f) ⇒ not a subfunctor
    @test !Cat.is_subfunctor(X, Dict(:a => Set([1]), :b => Set([:q])))

    # the empty and full subfunctors classify to "false" and "true" everywhere
    full = Dict(:a => Set([1, 2]), :b => Set([:p, :q]))
    χfull = Cat.classify(X, full)
    @test Cat.verify_classifies(X, full, χfull)
    @test all(χfull.components[c](x) == Cat.omega_true(arrow)[c]
              for c in (:a, :b) for x in Cat.ob(X, c).elements)
end

@testset "Ω over a presented category (commutative square)" begin
    sq = Cat.commutative_square()
    Ω = Cat.subobject_classifier(sq)
    @test Cat.is_functorial(Ω)
    # a (trivially closed) subfunctor still classifies correctly
    X = Cat.representable_functor(sq, :a)            # Hom(a, -)
    full = Dict(o => Set(Cat.ob(X, o).elements) for o in Cat.objects(sq))
    @test Cat.verify_classifies(X, full, Cat.classify(X, full))
end
