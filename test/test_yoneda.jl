# ============================================================================
# test_yoneda.jl — representables, presheaves, and the Yoneda lemma
# ============================================================================

using Test
using FunctorFlow
const Cat = FunctorFlow.Cat

@testset "Representable functors" begin
    chain = FreeCat([:a, :b, :c], [(:f, :a, :b), (:g, :b, :c)])
    yc = representable_functor(chain, :a)          # Hom(a, -)
    @test Cat.is_functorial(yc)
    @test length(Cat.ob(yc, :a)) == 1              # {id_a}
    @test length(Cat.ob(yc, :b)) == 1              # {f}
    @test length(Cat.ob(yc, :c)) == 1              # {g∘f}

    yb = representable_functor(chain, :b)          # Hom(b, -)
    @test length(Cat.ob(yb, :a)) == 0              # no b→a
    @test length(Cat.ob(yb, :c)) == 1

    # contravariant representable Hom(-, c) via the opposite category
    pc = representable_presheaf(chain, :c)
    @test Cat.is_functorial(pc)
    @test length(Cat.ob(pc, :a)) == 1              # one a→c
    @test length(Cat.ob(pc, :c)) == 1
end

@testset "Yoneda bijection: round-trip and naturality" begin
    arrow = FreeCat([:a, :b], [(:f, :a, :b)])
    F = Cat.SetFunctor(arrow;
        ob_map=Dict(:a => Cat.FinSet([1, 2]), :b => Cat.FinSet([:p, :q, :r])),
        edge_map=Dict(:f => Cat.FinFunction(Cat.FinSet([1, 2]), Cat.FinSet([:p, :q, :r]),
                                            [1 => :p, 2 => :q])))
    # yoneda_inverse ∘ yoneda_map == id on F(a), and each image is natural
    for e in Cat.ob(F, :a).elements
        α = yoneda_map(arrow, :a, F, e)
        @test Cat.is_natural(α)
        @test yoneda_inverse(arrow, :a, α) == e
    end
    # element ∉ F(a) is rejected
    @test_throws ArgumentError yoneda_map(arrow, :a, F, 99)
end

@testset "Yoneda lemma holds (Nat(Hom(c,-), F) ≅ F(c))" begin
    arrow = FreeCat([:a, :b], [(:f, :a, :b)])
    chain = FreeCat([:a, :b, :c], [(:f, :a, :b), (:g, :b, :c)])

    F = Cat.SetFunctor(arrow;
        ob_map=Dict(:a => Cat.FinSet([1, 2]), :b => Cat.FinSet([:p, :q, :r])),
        edge_map=Dict(:f => Cat.FinFunction(Cat.FinSet([1, 2]), Cat.FinSet([:p, :q, :r]),
                                            [1 => :p, 2 => :q])))
    @test yoneda_lemma_holds(arrow, :a, F)
    @test yoneda_lemma_holds(arrow, :b, F)

    # at a representable, and on a longer category
    @test yoneda_lemma_holds(chain, :a, representable_functor(chain, :a))
    @test yoneda_lemma_holds(chain, :b, representable_functor(chain, :a))

    # explicit cardinality witness: |Nat(Hom(a,-), F)| == |F(a)|
    n = Cat.count_nat_transformations(representable_functor(arrow, :a), F)
    @test n == length(Cat.ob(F, :a))
end

@testset "Representability detection" begin
    arrow = FreeCat([:a, :b], [(:f, :a, :b)])

    # a representable is detected, with the representing object as witness
    rep = representable_functor(arrow, :a)
    r = is_representable(rep)
    @test r.representable && r.witness == :a

    # a non-representable functor: sizes (1, 2) match no Hom(c, -) on the arrow
    F = Cat.SetFunctor(arrow;
        ob_map=Dict(:a => Cat.FinSet([1]), :b => Cat.FinSet([:p, :q])),
        edge_map=Dict(:f => Cat.FinFunction(Cat.FinSet([1]), Cat.FinSet([:p, :q]), [1 => :p])))
    @test !is_representable(F).representable
end
