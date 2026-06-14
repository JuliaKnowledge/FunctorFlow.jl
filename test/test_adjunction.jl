# ============================================================================
# test_adjunction.jl — functor composition, natural transformations,
# adjunctions (triangle identities), and the restriction functor F*
# ============================================================================

using Test
using FunctorFlow
const Cat = FunctorFlow.Cat

@testset "Identity functor & composition" begin
    chain = FreeCat([:a, :b, :c], [(:f, :a, :b), (:g, :b, :c)])
    Id = Cat.identity_functor(chain)
    @test Cat.is_functorial(Id)
    @test Id(Cat.homset(chain, :a, :c)[1]) == Cat.homset(chain, :a, :c)[1]

    arrow = FreeCat([:a, :b], [(:f, :a, :b)])
    F = FinFunctor(arrow, chain; ob_map=Dict(:a => :a, :b => :b),
                   edge_map=Dict(:f => PathMor(:a, :b, [:f])))
    GF = Cat.compose(F, Cat.identity_functor(chain))
    @test Cat.is_functorial(GF)
    @test GF.ob_map[:b] == :b
end

@testset "Natural transformations between functors" begin
    # codomain with a parallel pair so non-naturality is expressible
    D = FreeCat([:x, :y], [(:p, :x, :y), (:q, :x, :y)])
    arrow = FreeCat([:a, :b], [(:f, :a, :b)])
    Fp = FinFunctor(arrow, D; ob_map=Dict(:a => :x, :b => :y), edge_map=Dict(:f => PathMor(:x, :y, [:p])))
    Gq = FinFunctor(arrow, D; ob_map=Dict(:a => :x, :b => :y), edge_map=Dict(:f => PathMor(:x, :y, [:q])))

    # identity components are natural F⇒F
    natF = Cat.FunctorNatTrans(Fp, Fp; components=Dict(:a => Cat.id(D, :x), :b => Cat.id(D, :y)))
    @test Cat.is_natural(natF)

    # but F⇒G with identity components is NOT natural (p ≠ q)
    bad = Cat.FunctorNatTrans(Fp, Gq; components=Dict(:a => Cat.id(D, :x), :b => Cat.id(D, :y)))
    @test !Cat.is_natural(bad)
end

@testset "Adjunction: initial object as a left adjoint" begin
    chain = FreeCat([:a, :b, :c], [(:f, :a, :b), (:g, :b, :c)])
    adj = Cat.initial_object_adjunction(chain, :a)
    @test Cat.is_adjunction(adj)                 # triangle identities hold
    @test adj.left isa FinFunctor && adj.right isa FinFunctor

    # :b is not initial (no morphism b → a) ⇒ construction must fail
    @test_throws ArgumentError Cat.initial_object_adjunction(chain, :b)

    # a fan s → {a, b} has s as a genuine initial object
    fan = FreeCat([:s, :a2, :b2], [(:e1, :s, :a2), (:e2, :s, :b2)])
    @test Cat.is_adjunction(Cat.initial_object_adjunction(fan, :s))
    @test_throws ArgumentError Cat.initial_object_adjunction(fan, :a2)   # leaf is not initial

    # in the *free* category on a diamond the source is NOT initial: there are
    # two parallel paths s→w (free categories impose no commuting relation),
    # so the construction correctly refuses it.
    diamond = FreeCat([:s, :u, :v, :w],
                      [(:su, :s, :u), (:sv, :s, :v), (:uw, :u, :w), (:vw, :v, :w)])
    @test Cat.hom_cardinality(diamond, :s, :w) == 2
    @test_throws ArgumentError Cat.initial_object_adjunction(diamond, :s)
end

@testset "Restriction functor F* (Σ_F ⊣ F* ⊣ Π_F)" begin
    chain = FreeCat([:a, :b, :c], [(:f, :a, :b), (:g, :b, :c)])
    one = Cat.terminal_category()
    F = FinFunctor(one, chain; ob_map=Dict(:★ => :a), edge_map=Dict{Symbol,PathMor}())
    X = Cat.SetFunctor(chain;
        ob_map=Dict(:a => Cat.FinSet([1, 2]), :b => Cat.FinSet([:p]), :c => Cat.FinSet([:u])),
        edge_map=Dict(:f => Cat.FinFunction(Cat.FinSet([1, 2]), Cat.FinSet([:p]), [1 => :p, 2 => :p]),
                      :g => Cat.FinFunction(Cat.FinSet([:p]), Cat.FinSet([:u]), [:p => :u])))
    FX = Cat.restrict(X, F)
    @test Cat.is_functorial(FX)
    @test Cat.ob(FX, :★) == Cat.FinSet([1, 2])    # (F*X)(★) = X(a)
end
