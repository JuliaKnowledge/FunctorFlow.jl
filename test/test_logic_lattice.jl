# ============================================================================
# test_logic_lattice.jl — Heyting (internal logic), Galois/FCA, Grothendieck
# ============================================================================

using Test
using FunctorFlow
const Cat = FunctorFlow.Cat

@testset "Heyting algebra / intuitionistic internal logic" begin
    arrow = FreeCat([:a, :b], [(:f, :a, :b)])
    H = Cat.cosieve_heyting(arrow, :a)
    @test Cat.is_heyting_algebra(H)
    @test length(H.elements) == 3
    b = Cat.hbot(H); t = Cat.htop(H)
    @test Cat.hle(H, b, t) && !Cat.hle(H, t, b)
    @test Cat.himply(H, b, b) == t              # ⊥ ⇒ ⊥ = ⊤
    @test Cat.hmeet(H, b, t) == b && Cat.hjoin(H, b, t) == t
    # ¬¬ is not the identity in a (non-Boolean) Heyting algebra — pick the middle cosieve
    mid = first(x for x in H.elements if x != b && x != t)
    @test Cat.hle(H, mid, Cat.hneg(H, Cat.hneg(H, mid)))   # x ≤ ¬¬x always
    # the Lean certificate renders
    cert = render_heyting_certificate(H)
    @test occursin("HeytingDecl", cert) && occursin("isHeyting", cert)
end

@testset "Galois connections & formal concept analysis" begin
    P = Cat.Poset([0, 1, 2], Dict((x, y) => (x <= y) for x in 0:2 for y in 0:2))
    @test Cat.is_poset(P)
    @test Cat.is_galois_connection(P, P, Dict(x => x for x in 0:2), Dict(x => x for x in 0:2))
    # a non-monotone "f" breaks the connection
    @test !Cat.is_galois_connection(P, P, Dict(0 => 2, 1 => 1, 2 => 0), Dict(x => x for x in 0:2))

    # FCA: the concept lattice of a context
    objs = [:o1, :o2, :o3]; attrs = [:m1, :m2]
    inc = Set([(:o1, :m1), (:o2, :m1), (:o2, :m2), (:o3, :m2)])
    concepts = Cat.formal_concepts(objs, attrs, inc)
    @test all(Cat.is_formal_concept(c.extent, c.intent, objs, attrs, inc) for c in concepts)
    @test !Cat.is_formal_concept(Set([:o1]), Set([:m2]), objs, attrs, inc)   # not closed
    @test render_galois_certificate(P, P, Dict(x => x for x in 0:2), Dict(x => x for x in 0:2)) |>
          c -> occursin("isGalois", c)
end

@testset "Grothendieck construction (category of elements)" begin
    arrow = FreeCat([:a, :b], [(:f, :a, :b)])
    X = Cat.SetFunctor(arrow; ob_map=Dict(:a => Cat.FinSet([1, 2]), :b => Cat.FinSet([:p, :q])),
        edge_map=Dict(:f => Cat.FinFunction(Cat.FinSet([1, 2]), Cat.FinSet([:p, :q]), [1 => :p, 2 => :q])))
    E = Cat.category_of_elements(X)
    @test E isa FreeCat
    @test Cat.check_category_laws(E)
    @test length(Cat.objects(E)) == 4                 # (a,1),(a,2),(b,p),(b,q)
    @test Cat.is_functorial(Cat.elements_projection(X))   # ∫F → C is a functor (fibration)
    @test occursin("isCategory", render_cat_certificate(E))   # ∫F is Lean-certifiable as a category
end
