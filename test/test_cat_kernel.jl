# ============================================================================
# test_cat_kernel.jl — the verified categorical kernel (FunctorFlow.Cat)
# ============================================================================

using Test
using FunctorFlow
const Cat = FunctorFlow.Cat

@testset "FinSet & FinFunction" begin
    A = FinSet([1, 2, 3]); B = FinSet([:x, :y])
    @test length(A) == 3 && 2 in A && !(9 in A)
    @test FinSet([1, 2]) == FinSet([2, 1])              # equality by set

    f = FinFunction(A, B, [1 => :x, 2 => :y, 3 => :x])
    @test f(1) == :x && f(3) == :x
    @test Cat.dom(f) == A && Cat.cod(f) == B
    # totality + codomain checks
    @test_throws ArgumentError FinFunction(A, B, [1 => :x, 2 => :y])      # not total
    @test_throws ArgumentError FinFunction(A, B, [1 => :z, 2 => :y, 3 => :x])  # image ∉ cod

    # category-of-FinSet laws
    g = FinFunction(B, A, [:x => 1, :y => 2])
    @test Cat.compose(f, g)(2) == 2
    @test Cat.compose(Cat.id(A), f) == f
    @test Cat.compose(f, Cat.id(B)) == f
    @test_throws ArgumentError Cat.compose(g, g)         # not composable
end

@testset "FreeCat — small shape categories" begin
    arrow = FreeCat([:a, :b], [(:f, :a, :b)])
    @test Set(Cat.objects(arrow)) == Set([:a, :b])
    @test length(Cat.homset(arrow, :a, :b)) == 1
    @test length(Cat.homset(arrow, :a, :a)) == 1         # identity only
    @test length(Cat.homset(arrow, :b, :a)) == 0
    @test Cat.check_category_laws(arrow)

    chain = FreeCat([:a, :b, :c], [(:f, :a, :b), (:g, :b, :c)])
    @test Cat.hom_cardinality(chain, :a, :c) == 1
    @test Cat.check_category_laws(chain)

    # free category on a diamond has TWO distinct a→d paths (square need not commute)
    diamond = FreeCat([:a, :b, :c, :d],
                      [(:f, :a, :b), (:g, :a, :c), (:h, :b, :d), (:k, :c, :d)])
    @test Cat.hom_cardinality(diamond, :a, :d) == 2
    @test Cat.check_category_laws(diamond)

    # composition concatenates paths; identity is the empty path
    f = Cat.homset(arrow, :a, :b)[1]
    @test Cat.compose(arrow, Cat.id(arrow, :a), f) == f
    @test Cat.dom(arrow, f) == :a && Cat.cod(arrow, f) == :b

    # cyclic generators are rejected (would give infinite hom-sets)
    @test_throws ArgumentError FreeCat([:a, :b], [(:f, :a, :b), (:g, :b, :a)])
    # unknown endpoint rejected
    @test_throws ArgumentError FreeCat([:a], [(:f, :a, :z)])
end

@testset "FinFunctor — functors between categories" begin
    arrow = FreeCat([:a, :b], [(:f, :a, :b)])
    chain = FreeCat([:a, :b, :c], [(:f, :a, :b), (:g, :b, :c)])
    # send the arrow onto the composite a→c
    F = FinFunctor(arrow, chain; ob_map=Dict(:a => :a, :b => :c),
                   edge_map=Dict(:f => Cat.homset(chain, :a, :c)[1]))
    @test Cat.is_functorial(F)
    @test F(Cat.homset(arrow, :a, :b)[1]) == Cat.homset(chain, :a, :c)[1]
    @test F(Cat.id(arrow, :a)) == Cat.id(chain, :a)       # preserves identities

    # mismatched edge endpoints ⇒ not functorial
    bad = FinFunctor(arrow, chain; ob_map=Dict(:a => :a, :b => :b),
                     edge_map=Dict(:f => Cat.homset(chain, :a, :c)[1]))  # f should land in a→b
    @test !Cat.is_functorial(bad)
end

@testset "SetFunctor — a graph as a C-Set" begin
    # schema  E ⇉ V
    sch = FreeCat([:E, :V], [(:src, :E, :V), (:tgt, :E, :V)])
    V = Cat.FinSet([:v1, :v2, :v3]); E = Cat.FinSet([:e1, :e2])
    G = Cat.SetFunctor(sch;
        ob_map=Dict(:E => E, :V => V),
        edge_map=Dict(:src => Cat.FinFunction(E, V, [:e1 => :v1, :e2 => :v2]),
                      :tgt => Cat.FinFunction(E, V, [:e1 => :v2, :e2 => :v3])))
    @test Cat.is_functorial(G)
    @test Cat.ob(G, :V) == V
    @test length(Cat.ob(G, :E)) == 2

    # endpoint set mismatch is rejected at construction
    @test_throws ArgumentError Cat.SetFunctor(sch;
        ob_map=Dict(:E => E, :V => V),
        edge_map=Dict(:src => Cat.FinFunction(E, Cat.FinSet([:v1]), [:e1 => :v1, :e2 => :v1]),
                      :tgt => Cat.FinFunction(E, V, [:e1 => :v2, :e2 => :v3])))
end

@testset "CatNatTrans — naturality squares" begin
    arrow = FreeCat([:a, :b], [(:f, :a, :b)])
    S = Cat.FinSet([1, 2])
    F = Cat.SetFunctor(arrow; ob_map=Dict(:a => S, :b => S),
                       edge_map=Dict(:f => Cat.id(S)))
    Gg = Cat.SetFunctor(arrow; ob_map=Dict(:a => S, :b => S),
                        edge_map=Dict(:f => Cat.id(S)))
    # identity components ⇒ natural
    nat = Cat.CatNatTrans(F, Gg; components=Dict(:a => Cat.id(S), :b => Cat.id(S)))
    @test Cat.is_natural(nat)
    # swap at b but id at a ⇒ square fails to commute
    swap = Cat.FinFunction(S, S, [1 => 2, 2 => 1])
    bad = Cat.CatNatTrans(F, Gg; components=Dict(:a => Cat.id(S), :b => swap))
    @test !Cat.is_natural(bad)
end

@testset "Diagram ↔ category bridge" begin
    # a ket_block's schema is a genuine (law-abiding) category
    sc = diagram_freecat(ket_block())
    @test sc isa FreeCat
    @test Cat.check_category_laws(sc)

    # a concrete instance of a tiny diagram is a functor to Set
    D = Diagram(:Bridge)
    add_object!(D, :X); add_object!(D, :Y)
    add_morphism!(D, :f, :X, :Y)
    sf = diagram_setfunctor(D; sets=Dict(:X => [1, 2], :Y => [:a, :b]),
                            functions=Dict(:f => [1 => :a, 2 => :b]))
    @test sf isa SetFunctor
    @test Cat.is_functorial(sf)
    @test Cat.ob(sf, :Y) == Cat.FinSet([:a, :b])
end
