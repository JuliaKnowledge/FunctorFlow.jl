# ============================================================================
# test_operad.jl — finite (symmetric) operads / multicategories
# ============================================================================

using Test
using FunctorFlow
const Cat = FunctorFlow.Cat

@testset "Operad — construction & basic accessors" begin
    O = Cat.commutative_operad(; max_arity=3)
    @test Cat.operad_id(O) == Symbol("•", 1)
    @test length(Cat.operad_ops(O, 0)) == 1
    @test length(Cat.operad_ops(O, 2)) == 1
    @test Cat.operad_arity(O, Symbol("•", 2)) == 2
    @test_throws ArgumentError Cat.operad_arity(O, :nope)

    # identity must live in O(1)
    badops = Dict(0 => Any[], 1 => Any[:e])
    @test_throws ArgumentError Cat.Operad(badops, :notthere, (θ,φs)->θ)
end

@testset "Operad — γ composition arity bookkeeping" begin
    O = Cat.commutative_operad(; max_arity=4)
    θ = Symbol("•", 2)                       # binary
    φ1 = Symbol("•", 2); φ2 = Symbol("•", 1) # arities 2 and 1
    r = Cat.operad_compose(O, θ, [φ1, φ2])
    @test Cat.operad_arity(O, r) == 3        # 2 + 1
    # wrong number of inner ops for a binary operation
    @test_throws ArgumentError Cat.operad_compose(O, θ, [φ1])
    # varargs form agrees
    @test Cat.operad_compose(O, θ, φ1, φ2) == r
end

@testset "Operad laws — commutative (terminal) operad" begin
    O = Cat.commutative_operad(; max_arity=3)
    @test Cat.operad_laws(O)
    @test Cat.operad_symmetry_laws(O)
end

@testset "Operad laws — associative operad (symmetric, nontrivial)" begin
    O = Cat.associative_operad(; max_arity=3)
    @test Cat.operad_id(O) == [1]
    @test length(Cat.operad_ops(O, 3)) == 6   # 3! orders
    @test Cat.operad_laws(O)
    @test Cat.operad_symmetry_laws(O)

    # γ genuinely reorders: plug a swapped binary order into a swapped binary.
    swap = [2, 1]
    r = Cat.operad_compose(O, swap, [swap, [1]])   # θ=[2,1], φ1=[2,1], φ2=[1]
    @test Cat.operad_arity(O, r) == 3
    @test r isa Vector{Int} && sort(r) == [1, 2, 3]
end

@testset "Operad laws — wiring & little-intervals examples" begin
    W = Cat.wiring_operad(; max_arity=3)
    @test Cat.operad_laws(W)
    L = Cat.little_intervals_operad(; max_arity=3)
    @test Cat.operad_laws(L)
    @test Cat.operad_symmetry_laws(L)        # vacuously true (non-symmetric)
    @test_throws ArgumentError Cat.operad_act(L, [1], [1])  # no symmetry
end

@testset "Operad → underlying monoid of unary operations" begin
    O = Cat.associative_operad(; max_arity=2)
    ops, mul, unit = Cat.unary_monoid(O)
    @test unit == [1]
    @test ops == [[1]]                       # O(1) is a single point here
    # monoid laws on this (trivial) unary part
    for a in ops, b in ops
        @test mul(unit, a) == a
        @test mul(a, unit) == a
    end
end

@testset "Operad laws — NEGATIVE CONTROL: non-associative / non-unital γ rejected" begin
    # Build a candidate with the same operations as the associative operad but a
    # BROKEN γ that ignores the inner orders and always returns the identity
    # order. This violates associativity (and unit) and must be REJECTED.
    base = Cat.associative_operad(; max_arity=3)
    bad_gamma(θ::Vector{Int}, φs) =
        collect(1:(isempty(φs) ? 0 : sum(length(φ) for φ in φs)))
    Obad = Cat.Operad(base.ops, [1], bad_gamma; symmetry=base.symmetry, max_arity=3)
    @test Cat.operad_laws(Obad) == false      # rejected

    # A second negative control built on the unary part (a one-object would-be
    # category): an operad restricted to O(1) is a monoid, so a NON-ASSOCIATIVE
    # binary table on O(1) must be rejected by operad_laws. O(1) = {e, a},
    # composition γ(θ; φ) = mul(θ, φ); we use a magma where (a·a)·a ≠ a·(a·a).
    #   mul(e, x) = x, mul(x, e) = x   (e is a unit, so unit laws hold)
    #   mul(a, a) = a                  (… but we sabotage triple products below)
    # To make it provably non-associative we override the table so that the two
    # bracketings of a·a·a disagree. We encode "memory" via a sentinel element.
    ops = Dict(0 => Any[], 1 => Any[:e, :a, :b])
    # multiplication table chosen so that (a·a)·a = b·a = e but a·(a·a) = a·b = a.
    tbl = Dict(
        (:e, :e) => :e, (:e, :a) => :a, (:e, :b) => :b,
        (:a, :e) => :a, (:b, :e) => :b,
        (:a, :a) => :b, (:a, :b) => :a,
        (:b, :a) => :e, (:b, :b) => :b,
    )
    function ng(θ, φs)
        isempty(φs) && return :e             # arity-0 (unused here)
        return tbl[(θ, φs[1])]               # unary composition = magma mul
    end
    # max_arity=1: only the unary monoid axioms are exercised — exactly the
    # associativity of `tbl`, which fails since (a·a)·a = e ≠ a = a·(a·a).
    Ong = Cat.Operad(ops, :e, ng; max_arity=1)
    @test Cat.operad_laws(Ong) == false       # non-associative ⇒ rejected
end
