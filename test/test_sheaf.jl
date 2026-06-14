# ============================================================================
# test_sheaf.jl — Grothendieck (co)topologies, sheaf condition, sheafification
#
# Variance: the kernel is covariant (copresheaves `C → Set`) and uses cosieves,
# so the sheaf theory here is the exact dual of the textbook presheaf/sieve
# story (see sheaf.jl's module docstring). The worked site is the span
# `a ← s → b` whose only nontrivial cover is {p, q} on s; a copresheaf is a
# sheaf there iff F(s) ≅ F(a) × F(b) via the two legs.
# ============================================================================

using Test
using FunctorFlow
const Cat = FunctorFlow.Cat

@testset "Coverage is a Grothendieck (co)topology" begin
    C, J = Cat.span_site()
    # covers are genuine cosieves and the maximal cosieve is always present
    for c in (:s, :a, :b)
        Rs = Cat.covering_sieves(J, c)
        @test !isempty(Rs)
        mx = Cat._maximal_cosieve(C, c)
        @test any(Set(R) == Set(mx) for R in Rs)
    end
    # the {p,q} cover on s is present and is not the maximal cosieve (id_s ∉ it)
    p = Cat.PathMor(:s, :a, [:p]); q = Cat.PathMor(:s, :b, [:q])
    @test any(Set(R) == Set([p, q]) for R in Cat.covering_sieves(J, :s))
    # axioms hold
    @test Cat.is_grothendieck_topology(J)
end

@testset "Matching families and amalgamations" begin
    C, J = Cat.span_site()
    F = Cat.span_sheaf()
    @test Cat.is_functorial(F)
    p = Cat.PathMor(:s, :a, [:p]); q = Cat.PathMor(:s, :b, [:q])
    R = sort([p, q]; by=Cat._mor_key)
    fams = Cat.matching_families(F, R)
    # legs are independent (a, b are sinks) ⇒ |matching families| = |F(a)|·|F(b)| = 4
    @test length(fams) == 4
    # for the product sheaf every matching family has a unique amalgamation
    for fam in fams
        @test length(Cat.amalgamations(F, R, fam)) == 1
    end
end

@testset "The product copresheaf IS a sheaf" begin
    _, J = Cat.span_site()
    F = Cat.span_sheaf()
    @test Cat.is_separated(F, J)
    @test Cat.is_sheaf(F, J)
end

@testset "The diagonal copresheaf is NOT a sheaf" begin
    C, J = Cat.span_site()
    G = Cat.span_non_sheaf()
    @test Cat.is_functorial(G)
    # separated (gluing map injective) but not a sheaf (not surjective)
    @test Cat.is_separated(G, J)
    @test !Cat.is_sheaf(G, J)
    # concretely: the matching family (x_p, x_q) = (0, 1) has NO amalgamation
    p = Cat.PathMor(:s, :a, [:p]); q = Cat.PathMor(:s, :b, [:q])
    R = sort([p, q]; by=Cat._mor_key)
    fam = Dict{Cat.PathMor,Any}(p => 0, q => 1)
    @test Cat._is_matching(G, R, fam)            # it is a legitimate matching family
    @test isempty(Cat.amalgamations(G, R, fam))  # ...with no amalgamation
end

@testset "Non-separated copresheaf fails is_separated" begin
    # F(s) collapses the diagonal: two elements 'u','v' with identical legs ⇒
    # the gluing map F(s) → F(a)×F(b) is non-injective ⇒ not separated.
    C, J = Cat.span_site()
    Sset = Cat.FinSet([:u, :v]); Aset = Cat.FinSet([0]); Bset = Cat.FinSet([0])
    p = Cat.FinFunction(Sset, Aset, Dict{Any,Any}(:u => 0, :v => 0))
    q = Cat.FinFunction(Sset, Bset, Dict{Any,Any}(:u => 0, :v => 0))
    H = Cat.SetFunctor(C; ob_map=Dict(:s=>Sset, :a=>Aset, :b=>Bset),
                          edge_map=Dict(:p=>p, :q=>q))
    @test !Cat.is_separated(H, J)
    @test !Cat.is_sheaf(H, J)
end

@testset "Separated reflection makes a presheaf separated" begin
    C, J = Cat.span_site()
    # same non-separated H as above
    Sset = Cat.FinSet([:u, :v]); Aset = Cat.FinSet([0]); Bset = Cat.FinSet([0])
    p = Cat.FinFunction(Sset, Aset, Dict{Any,Any}(:u => 0, :v => 0))
    q = Cat.FinFunction(Sset, Bset, Dict{Any,Any}(:u => 0, :v => 0))
    H = Cat.SetFunctor(C; ob_map=Dict(:s=>Sset, :a=>Aset, :b=>Bset),
                          edge_map=Dict(:p=>p, :q=>q))
    Hs, η = Cat.separated_reflection(H, J)
    @test Cat.is_functorial(Hs)
    @test Cat.is_natural(η)
    @test Cat.is_separated(Hs, J)
    # u and v got identified ⇒ |Hs(s)| == 1
    @test length(Cat.ob(Hs, :s)) == 1
    @test length(Cat.ob(H, :s)) == 2          # original was 2
    # reflecting an already-separated sheaf changes nothing essential
    F = Cat.span_sheaf()
    Fs, ηF = Cat.separated_reflection(F, J)
    @test Cat.is_separated(Fs, J)
    @test length(Cat.ob(Fs, :s)) == length(Cat.ob(F, :s))
end
