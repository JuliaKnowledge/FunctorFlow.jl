# ============================================================================
# test_twocat.jl — strict 2-categories / bicategories
#
# Exercises the strict-2-category kernel (twocat.jl): the law checks
# (vertical/horizontal associativity & unit, and the interchange law), three
# concrete worked examples (delooping of a commutative monoid; a hand-built
# 2-category drawn from Cat with FinFunctor 1-cells and FunctorNatTrans
# 2-cells), and a NEGATIVE CONTROL whose composition tables violate the
# interchange law and are duly rejected. Also checks Para's reparametrisation
# 2-cells and the documented "Para is a bicategory" realization.
# ============================================================================

using Test
using FunctorFlow
const Cat = FunctorFlow.Cat

@testset "Strict 2-category: delooping a commutative monoid (Eckmann–Hilton)" begin
    # B²(Z/3, +): one 0-cell ★, one 1-cell id, 2-cells = {0,1,2}, both
    # compositions = +, both identities = 0.
    M = Cat.deloop_monoid(0:2, (x, y) -> mod(x + y, 3), 0)

    @test Cat.check_vertical_category_laws(M)
    @test Cat.check_horizontal_category_laws(M)
    @test Cat.check_interchange_law(M)          # ⇔ Eckmann–Hilton for a comm. monoid
    @test Cat.check_two_category_laws(M)

    # concrete arithmetic: both products are addition mod 3
    @test Cat.vcomp(M, 1, 2) == 0
    @test Cat.hcomp(M, 1, 2) == 0
    @test Cat.vcomp(M, 2, 2) == 1
    @test Cat.id2(M, Cat.id1(M, :★)) == 0       # identity 2-cell is the monoid unit

    # a *non-commutative* monoid does NOT deloop to a strict 2-category:
    # take the 2-element left-zero semigroup-with-unit is associative+unital but
    # interchange would force commutativity, which fails. We build it by hand in
    # the negative-control testset below; here we just confirm a commutative one
    # of size 4 (Z/2 × Z/2) also works.
    add4(x, y) = (mod(x[1] + y[1], 2), mod(x[2] + y[2], 2))
    elts = [(i, j) for i in 0:1 for j in 0:1]
    M4 = Cat.deloop_monoid(elts, add4, (0, 0))
    @test Cat.check_two_category_laws(M4)
end

@testset "Strict 2-category from Cat (functors as 1-cells, nat-transs as 2-cells)" begin
    # 0-cell: the small category D = (x --v--> y).
    # 1-cells: idD and the constant functor K collapsing D onto x.
    #          (K∘K = K, K∘idD = idD∘K = K, idD∘idD = idD ⇒ {idD,K} closed.)
    # 2-cells: id_idD, id_K, and e : K ⇒ idD (id at x, the edge v at y).
    D   = Cat.FreeCat([:x, :y], [(:v, :x, :y)])
    idD = Cat.identity_functor(D)
    K   = Cat.FinFunctor(D, D;
            ob_map = Dict(:x => :x, :y => :x),
            edge_map = Dict(:v => Cat.PathMor(:x, :x, Symbol[])))
    @test Cat.is_functorial(K)

    e = Cat.FunctorNatTrans(K, idD;
            components = Dict(:x => Cat.id(D, :x), :y => Cat.PathMor(:x, :y, [:v])))
    @test Cat.is_natural(e)

    id_idD = Cat.identity_nat(idD)
    id_K   = Cat.identity_nat(K)
    @test Cat.is_natural(id_idD)
    @test Cat.is_natural(id_K)

    # vertical / horizontal composites of nat-transs are computed honestly:
    @test Cat._nat_key(Cat.vcompose(e, id_K))   == Cat._nat_key(e)    # e ∘ id_K = e
    @test Cat._nat_key(Cat.vcompose(id_idD, e)) == Cat._nat_key(e)    # id_idD ∘ e = e
    @test Cat._nat_key(Cat.hcompose(e, e))      == Cat._nat_key(e)    # e ∗ e = e

    K2 = Cat.cat_two_category(;
        cats     = Dict(:D => D),
        functors = Dict(:idD => idD, :K => K),
        nats     = Dict(:i_idD => id_idD, :i_K => id_K, :e => e),
        id1      = Dict(:D => :idD),
        id2      = Dict(:idD => :i_idD, :K => :i_K))

    # all axioms, derived from real functor/nat-trans computation:
    @test Cat.check_vertical_category_laws(K2)
    @test Cat.check_horizontal_category_laws(K2)
    @test Cat.check_interchange_law(K2)
    @test Cat.check_two_category_laws(K2)

    # spot-check the tabulated composites
    @test Cat.vcomp(K2, :e, :i_K)   == :e
    @test Cat.vcomp(K2, :i_idD, :e) == :e
    @test Cat.hcomp(K2, :e, :e)     == :e
    @test Cat.id1(K2, :D)  == :idD
    @test Cat.id2(K2, :K)  == :i_K
    @test length(Cat.onecells(K2, :D, :D)) == 2
    @test length(Cat.twocells(K2, :K, :idD)) == 1   # exactly e : K ⇒ idD

    # closure is enforced: a `nats` set that is NOT closed under the induced
    # composites is rejected at build time. Here e ∗ e = e and id_idD ∘ e = e,
    # so leaving `e` out of `nats` while keeping it as a participant is not how
    # it fails; instead omit `i_K`, which the horizontal composite e ∗ id_idD
    # (and others) require — building then throws on the missing result.
    @test_throws ArgumentError Cat.cat_two_category(;
        cats     = Dict(:D => D),
        functors = Dict(:idD => idD, :K => K),
        nats     = Dict(:i_idD => id_idD, :e => e),   # i_K omitted ⇒ set not closed
        id1      = Dict(:D => :idD),
        id2      = Dict(:idD => :i_idD, :K => :i_K))
end

@testset "Constructor validates typing (ill-typed tables are rejected)" begin
    star = :★
    one1 = Cat.OneCell(:id, star, star)
    twos = [Cat.TwoCell(:e, :id, :id)]
    good_vc = Dict{Any,Any}((:e, :e) => :e)
    good_hc = Dict{Any,Any}((:e, :e) => :e)

    # missing identity 2-cell
    @test_throws ArgumentError Cat.TwoCategory(;
        zerocells = [star], onecells = [one1], twocells = twos,
        id1 = Dict(star => :id), id2 = Dict{Any,Any}(),   # missing id2[:id]
        vcomp = good_vc, hcomp = good_hc)

    # a 2-cell between non-parallel 1-cells is rejected
    a = :a; b = :b
    f = Cat.OneCell(:f, a, b)
    g = Cat.OneCell(:g, b, a)   # not parallel to f
    @test_throws ArgumentError Cat.TwoCategory(;
        zerocells = [a, b], onecells = [f, g],
        twocells = [Cat.TwoCell(:bad, :f, :g)],
        id1 = Dict(a => :f, b => :g),     # (irrelevant; throws earlier)
        id2 = Dict{Any,Any}(), vcomp = Dict{Any,Any}(), hcomp = Dict{Any,Any}())
end

@testset "NEGATIVE CONTROL: interchange-violating tables are rejected" begin
    # One 0-cell ★, one 1-cell id, 2-cells {e, a} : id ⇒ id, identity 2-cell e.
    # vcomp makes {e,a} the group Z/2 (a∘a = e).
    # hcomp makes {e,a} the idempotent monoid (a∗a = a).
    # Both are valid monoids with unit e, so the vertical and horizontal CATEGORY
    # laws hold — but the two products disagree, so INTERCHANGE must fail.
    star = :★
    one1 = Cat.OneCell(:id, star, star)
    twos = [Cat.TwoCell(:e, :id, :id), Cat.TwoCell(:a, :id, :id)]
    vc = Dict{Any,Any}(); hc = Dict{Any,Any}()
    for x in (:e, :a), y in (:e, :a)
        vc[(x, y)] = x == :e ? y : (y == :e ? x : :e)   # a∘a = e (Z/2)
        hc[(x, y)] = x == :e ? y : (y == :e ? x : :a)   # a∗a = a (idempotent)
    end

    bad = Cat.TwoCategory(;
        zerocells = [star], onecells = [one1], twocells = twos,
        id1 = Dict(star => :id), id2 = Dict(:id => :e),
        vcomp = vc, hcomp = hc)

    # the table is well-typed (constructor accepts it) ...
    @test bad isa Cat.TwoCategory
    # ... and each composition is independently a valid (unital, associative) category ...
    @test Cat.check_vertical_category_laws(bad)
    @test Cat.check_horizontal_category_laws(bad)
    # ... but interchange — the defining strict-2-category law — is VIOLATED:
    @test !Cat.check_interchange_law(bad)
    @test !Cat.check_two_category_laws(bad)

    # exhibit a concrete witness: (a∘a) ∗ (a∘a) = e∗e = e,
    # whereas (a∗a) ∘ (a∗a) = a∘a = e ... pick a witness that actually differs:
    # LHS (a∘e)∗(a∘e) = a∗a = a ; RHS (a∗a)∘(e∗e) = a∘e = a — equal here.
    # The genuine failure: take left col α=a,β=a and right col α′=a,β′=a:
    #   LHS = (β′∘α′) ∗ (β∘α) = (a∘a) ∗ (a∘a) = e ∗ e = e
    #   RHS = (β′∗β) ∘ (α′∗α) = (a∗a) ∘ (a∗a) = a ∘ a = e        (this pair agrees)
    # use the asymmetric witness instead: α=a,β=e (left), α′=e,β′=a (right):
    #   LHS = (a∘e) ∗ (e∘... )  — compute directly via the tables:
    lhs = Cat.hcomp(bad, Cat.vcomp(bad, :a, :e), Cat.vcomp(bad, :e, :a))  # (β′∘α′)∗(β∘α)
    rhs = Cat.vcomp(bad, Cat.hcomp(bad, :a, :e), Cat.hcomp(bad, :e, :a))  # (β′∗β)∘(α′∗α)
    @test lhs == :a            # (a∘e)∗(e∘a) = a ∗ a = a
    @test rhs == :e            # (a∗e)∘(e∗a) = a ∘ a = e
    @test lhs != rhs           # ⇒ interchange genuinely broken
end

@testset "Para: reparametrisations as 2-cells (a bicategory)" begin
    # f, g : A → A with A = Z/3, impl(p, a) = a + p (mod 3).
    A = [0, 1, 2]
    mk(P) = Cat.ParaMap(collect(P), A, A,
        Cat.FinFunction(Cat.FinSet(Any[(p, a) for p in P for a in A]),
                        Cat.FinSet(A),
                        Dict{Any,Any}((p, a) => mod(a + p, 3) for p in P for a in A)))

    f = mk([0, 1]); g = mk([0, 1])
    rid = Cat.FinFunction(Cat.FinSet([0, 1]), Cat.FinSet([0, 1]),
                          Dict{Any,Any}(0 => 0, 1 => 1))
    @test Cat.para_reparam_two_cell(f, g, rid)   # identity reparam is a 2-cell

    # f2, g2 with params {0,2}; the reparam matching them is a 2-cell.
    f2 = mk([0, 2]); g2 = mk([0, 2])
    rmatch = Cat.FinFunction(Cat.FinSet([0, 2]), Cat.FinSet([0, 2]),
                             Dict{Any,Any}(0 => 0, 2 => 2))
    @test Cat.para_reparam_two_cell(f2, g2, rmatch)

    # a reparam that does NOT make the implementations agree is rejected.
    rbad = Cat.FinFunction(Cat.FinSet([0, 2]), Cat.FinSet([0, 1]),
                           Dict{Any,Any}(0 => 0, 2 => 1))
    @test !Cat.para_reparam_two_cell(f, g2, rbad)

    note = Cat.para_is_bicategory_note()
    @test occursin("BICATEGORY", note)
    @test occursin("up to", note)               # weak associativity is the point
end
