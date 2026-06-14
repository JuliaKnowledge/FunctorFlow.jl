# ============================================================================
# test_coend.jl — profunctors & coends (coend = coequalizer of dinaturality),
# culminating in "attention as a coend".
# ============================================================================

using Test
using FunctorFlow
const Cat = FunctorFlow.Cat

FS(xs)      = Cat.FinSet(collect(xs))
FF(A, B, p) = Cat.FinFunction(A, B, p)

# ----------------------------------------------------------------------------
# 1. A hand-computable coend over the single-arrow category a --f--> b.
#
#    P(a,a) = {a1,a2}, P(b,b) = {b1,b2}, het P(b,a) = {m1,m2}.
#    λ_f = P(f,id): m1↦a1, m2↦a2     (act on contravariant slot)
#    ρ_f = P(id,f): m1↦b1, m2↦b1     (act on covariant slot)
#    Coend identifies a1∼b1, a2∼b1  ⇒  classes {a1,a2,b1}, {b2}  (2 elements).
# ----------------------------------------------------------------------------
@testset "Coend = coequalizer of the dinaturality maps (by hand)" begin
    C = FreeCat([:a, :b], [(:f, :a, :b)])
    Paa = FS([:a1, :a2]); Pbb = FS([:b1, :b2]); Pba = FS([:m1, :m2])
    λ = FF(Pba, Paa, [:m1 => :a1, :m2 => :a2])
    ρ = FF(Pba, Pbb, [:m1 => :b1, :m2 => :b1])

    P = Cat.Profunctor(C; diag = Dict(:a => Paa, :b => Pbb),
                          het  = Dict(:f => Pba),
                          lact = Dict(:f => λ), ract = Dict(:f => ρ))

    co = Cat.coend(P)
    @test length(co.apex) == 2                          # {a1,a2,b1}, {b2}
    @test Cat.verify_coend(co)                          # dinaturality + univ. prop.

    # The identifications actually happened in the coend:
    @test Cat.coend_class(co, :a, :a1) == Cat.coend_class(co, :b, :b1)   # a1 ∼ b1
    @test Cat.coend_class(co, :a, :a2) == Cat.coend_class(co, :b, :b1)   # a2 ∼ b1
    @test Cat.coend_class(co, :a, :a1) == Cat.coend_class(co, :a, :a2)   # ⇒ a1 ∼ a2
    @test Cat.coend_class(co, :b, :b2) != Cat.coend_class(co, :a, :a1)   # b2 apart
end

# ----------------------------------------------------------------------------
# 2. The coend reuses Cat.coproduct / Cat.coequalizer: the apex is *exactly*
#    the coequalizer apex, and the two dinaturality legs are parallel.
# ----------------------------------------------------------------------------
@testset "Coend is literally the coequalizer of ⊔het ⇉ ⊔diag" begin
    C = FreeCat([:a, :b], [(:f, :a, :b)])
    Paa = FS([1, 2]); Pbb = FS([3, 4]); Pba = FS([:m])
    λ = FF(Pba, Paa, [:m => 1]); ρ = FF(Pba, Pbb, [:m => 3])
    P = Cat.Profunctor(C; diag=Dict(:a=>Paa,:b=>Pbb), het=Dict(:f=>Pba),
                          lact=Dict(:f=>λ), ract=Dict(:f=>ρ))
    co = Cat.coend(P)
    @test co.apex == co.coeq.apex
    @test co.source.dom == co.target.dom            # ⊔het
    @test co.source.cod == co.target.cod            # ⊔diag
    # only one identification (1∼3) ⇒ 4 diagonal elts collapse to 3 classes
    @test length(co.apex) == 3
    @test Cat.coend_class(co, :a, 1) == Cat.coend_class(co, :b, 3)
end

# ----------------------------------------------------------------------------
# 3. Discrete category (no morphisms) ⇒ coend = disjoint union of diagonals
#    (no relations to quotient), and the dual end = product of diagonals.
# ----------------------------------------------------------------------------
@testset "Discrete C: coend = ⊔ P(c,c), end = ∏ P(c,c)" begin
    C = FreeCat([:x, :y], Tuple{Symbol,Symbol,Symbol}[])
    Pxx = FS([:p, :q]); Pyy = FS([:r])
    P = Cat.Profunctor(C; diag=Dict(:x=>Pxx, :y=>Pyy),
                          het=Dict{Symbol,Cat.FinSet}(),
                          lact=Dict{Symbol,Cat.FinFunction}(),
                          ract=Dict{Symbol,Cat.FinFunction}())
    co = Cat.coend(P)
    @test length(co.apex) == 2 + 1                    # ⊔ : |Pxx| + |Pyy|
    en = Cat.end_(P)
    @test length(en.apex) == 2 * 1                    # ∏ : |Pxx| * |Pyy|
end

# ----------------------------------------------------------------------------
# 4. The dual end on the single-arrow category: the wedge condition.
#    Same P as test 1. A diagonal family (x_a ∈ P(a,a), x_b ∈ P(b,b)) is in the
#    end iff some m ∈ P(b,a) has λ(m)=x_a AND ρ(m)=x_b.
#    Witnesses: (a1,b1) via m1, (a2,b1) via m2. Nothing hits b2. ⇒ end = 2 elts.
# ----------------------------------------------------------------------------
@testset "End = equalizer (the dinatural wedge)" begin
    C = FreeCat([:a, :b], [(:f, :a, :b)])
    Paa = FS([:a1, :a2]); Pbb = FS([:b1, :b2]); Pba = FS([:m1, :m2])
    λ = FF(Pba, Paa, [:m1 => :a1, :m2 => :a2])
    ρ = FF(Pba, Pbb, [:m1 => :b1, :m2 => :b1])
    P = Cat.Profunctor(C; diag=Dict(:a=>Paa,:b=>Pbb), het=Dict(:f=>Pba),
                          lact=Dict(:f=>λ), ract=Dict(:f=>ρ))
    en = Cat.end_(P)
    @test length(en.apex) == 2                        # (a1,b1) and (a2,b1)
    fams = Set(en.apex.elements)
    @test (:a1, :b1) in fams
    @test (:a2, :b1) in fams
    # nothing pairs with b2 (no het witness maps to b2)
    @test all(fam[2] != :b2 for fam in en.apex.elements)
end

# ----------------------------------------------------------------------------
# 5. ATTENTION AS A COEND.
#
# Self-attention reads, for a query q, a context aggregated over key/value
# positions: out(q) = ⊕_k  compat(q,k) · value(k).  Categorically this is a
# coend ∫^k  Hom(q,k) ⊗ V(k): the universal dinatural object that glues the
# "weighted values" along the key-indexing morphisms, so that transporting a
# key along a morphism and *then* reading its value equals reading the value and
# *then* transporting it — exactly the dinaturality the coequalizer enforces.
#
# Concrete tiny instance. Key category K: positions k1 --r--> k2 (a morphism
# routing/merging key k1 into k2, e.g. a shared head). The diagonal profunctor
# P(k,k) = { (q, v) : query q attends to key k with value v } is the bag of
# (query, value) readings at each key. The routing edge r:k1→k2 says the value
# read at k1 should be identified with its image at k2. The coend therefore
# aggregates the per-key readings into per-query context vectors: the surviving
# classes are exactly the merged (query ⊕ value) contexts.
# ----------------------------------------------------------------------------
@testset "Attention as a coend (context aggregation)" begin
    # Two key positions, one routing morphism k1 → k2.
    K = FreeCat([:k1, :k2], [(:r, :k1, :k2)])

    # Diagonal readings P(k,k) = {(query, value-token) attended at key k}.
    # Query q1 reads value v_a at k1 and (via routing) the SAME content at k2.
    P11 = FS([(:q1, :va), (:q2, :vb)])      # readings at k1
    P22 = FS([(:q1, :va2), (:q2, :vc)])     # readings at k2
    # Heteromorphism slot P(k2,k1) — routing witnesses linking a k1-reading to k2.
    Phet = FS([(:q1, :va)])                 # only q1's reading is routed/merged

    # λ_r = P(r,id): the witness, as a k1-reading (contravariant slot)
    λ = FF(Phet, P11, [(:q1,:va) => (:q1,:va)])
    # ρ_r = P(id,r): the SAME witness, as the k2-reading it merges into
    ρ = FF(Phet, P22, [(:q1,:va) => (:q1,:va2)])

    P = Cat.Profunctor(K; diag=Dict(:k1=>P11, :k2=>P22), het=Dict(:r=>Phet),
                          lact=Dict(:r=>λ), ract=Dict(:r=>ρ))
    @test Cat.profunctor_diag(P, :k1) == P11

    ctx = Cat.coend(P)
    @test Cat.verify_coend(ctx)

    # Aggregation happened: q1's k1-reading and its routed k2-reading are now the
    # SAME context element (the merged query-1 context).
    @test Cat.coend_class(ctx, :k1, (:q1,:va)) == Cat.coend_class(ctx, :k2, (:q1,:va2))

    # Unrouted readings stay distinct (q2 at k1 vs q2 at k2 are different contexts).
    @test Cat.coend_class(ctx, :k1, (:q2,:vb)) != Cat.coend_class(ctx, :k2, (:q2,:vc))

    # |diag| = 4 readings; one identification ⇒ 3 aggregated context classes.
    @test length(ctx.apex) == 3
end

# ----------------------------------------------------------------------------
# 6. The textbook identity: Cat.left_kan already computes a (pointwise) coend.
#    Lan_F X (d) = ∫^c Hom_D(F c, d) × X c.  We check, for F = id_C, that the
#    left Kan extension's value at each object has the same size as the coend of
#    the corresponding diagonal profunctor — i.e. the colimit/coend agreement.
#    (Here, simply: Lan_id X ≅ X, and X(c) is a "trivial" coend over the
#    discrete diagonal, confirming left_kan is a coend machine.)
# ----------------------------------------------------------------------------
@testset "left_kan is a coend machine (Lan_id X ≅ X)" begin
    C = FreeCat([:a, :b], [(:f, :a, :b)])
    X = Cat.SetFunctor(C;
        ob_map = Dict(:a => FS([1, 2]), :b => FS([:p, :q, :r])),
        edge_map = Dict(:f => FF(FS([1,2]), FS([:p,:q,:r]), [1 => :p, 2 => :q])))
    Lan = Cat.left_kan(Cat.identity_functor(C), X)
    @test Cat.is_functorial(Lan)
    # Lan_id X (c) is a coend that recovers X(c).
    @test length(Cat.ob(Lan, :a)) == length(Cat.ob(X, :a))
    @test length(Cat.ob(Lan, :b)) == length(Cat.ob(X, :b))

    # And the *diagonal* coend of X (viewed as a profunctor constant in the
    # contravariant slot, with the trivial het = ∅ over the discrete shadow) is
    # the colimit of X — the canonical "coend = colimit of a diagonal" instance.
    col = Cat.colimit(X)
    Cdisc = FreeCat([:a, :b], Tuple{Symbol,Symbol,Symbol}[])
    Pdiag = Cat.Profunctor(Cdisc; diag=Dict(:a=>Cat.ob(X,:a), :b=>Cat.ob(X,:b)),
                            het=Dict{Symbol,Cat.FinSet}(),
                            lact=Dict{Symbol,Cat.FinFunction}(),
                            ract=Dict{Symbol,Cat.FinFunction}())
    co = Cat.coend(Pdiag)
    # discrete coend = ⊔ diagonals = 5; colimit quotients further by f.
    @test length(co.apex) == length(Cat.ob(X,:a)) + length(Cat.ob(X,:b))
    @test length(col.apex) <= length(co.apex)
end
