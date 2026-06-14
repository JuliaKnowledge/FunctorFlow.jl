# ============================================================================
# test_kan_general.jl — Kan extensions Lan_F / Ran_F along an arbitrary functor
# ============================================================================

using Test
using FunctorFlow
const Cat = FunctorFlow.Cat

arrow = FreeCat([:a, :b], [(:f, :a, :b)])
X = Cat.SetFunctor(arrow;
    ob_map=Dict(:a => Cat.FinSet([1, 2]), :b => Cat.FinSet([:p, :q, :r])),
    edge_map=Dict(:f => Cat.FinFunction(Cat.FinSet([1, 2]), Cat.FinSet([:p, :q, :r]), [1 => :p, 2 => :q])))

one = Cat.terminal_category()
bang = Cat.FinFunctor(arrow, one; ob_map=Dict(:a => :★, :b => :★),
                      edge_map=Dict(:f => PathMor(:★, :★, Symbol[])))

@testset "Kan along the terminal functor = colimit / limit" begin
    Lan = Cat.left_kan(bang, X)
    Ran = Cat.right_kan(bang, X)
    @test Cat.is_functorial(Lan) && Cat.is_functorial(Ran)
    @test length(Cat.ob(Lan, :★)) == length(Cat.colimit(X).apex)   # Lan_! = colim
    @test length(Cat.ob(Ran, :★)) == length(Cat.limit(X).apex)     # Ran_! = lim
end

@testset "Kan along the identity functor ≅ X" begin
    Lan = Cat.left_kan(Cat.identity_functor(arrow), X)
    Ran = Cat.right_kan(Cat.identity_functor(arrow), X)
    @test Cat.is_functorial(Lan) && Cat.is_functorial(Ran)
    @test length(Cat.ob(Lan, :a)) == 2 && length(Cat.ob(Lan, :b)) == 3
    @test length(Cat.ob(Ran, :a)) == 2 && length(Cat.ob(Ran, :b)) == 3
end

@testset "Adjunction triple Lan_F ⊣ F* ⊣ Ran_F (hom-set cardinalities)" begin
    # along the terminal functor
    Y1 = Cat.SetFunctor(one; ob_map=Dict(:★ => Cat.FinSet([10, 20])),
                        edge_map=Dict{Symbol, Cat.FinFunction}())
    F1Y = Cat.restrict(Y1, bang)
    @test Cat.count_nat_transformations(Cat.left_kan(bang, X), Y1) ==
          Cat.count_nat_transformations(X, F1Y)                       # Lan ⊣ F*
    @test Cat.count_nat_transformations(F1Y, X) ==
          Cat.count_nat_transformations(Y1, Cat.right_kan(bang, X))   # F* ⊣ Ran

    # along an inclusion  ι : 1 → arrow  picking the object a
    incl = Cat.FinFunctor(one, arrow; ob_map=Dict(:★ => :a), edge_map=Dict{Symbol, PathMor}())
    W = Cat.SetFunctor(one; ob_map=Dict(:★ => Cat.FinSet([:s, :t])),
                       edge_map=Dict{Symbol, Cat.FinFunction}())   # W : 1 → Set
    LanW = Cat.left_kan(incl, W)
    RanW = Cat.right_kan(incl, W)
    @test Cat.is_functorial(LanW) && Cat.is_functorial(RanW)
    # Lan_ι W ⊣ ι* : Nat(Lan_ι W, X) ≅ Nat(W, ι*X)
    @test Cat.count_nat_transformations(LanW, X) ==
          Cat.count_nat_transformations(W, Cat.restrict(X, incl))
    # ι* ⊣ Ran_ι W : Nat(ι*X, W) ≅ Nat(X, Ran_ι W)
    @test Cat.count_nat_transformations(Cat.restrict(X, incl), W) ==
          Cat.count_nat_transformations(X, RanW)
end
