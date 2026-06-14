# ============================================================================
# optics.jl — lenses & Para: the categorical foundation of gradient learning
# (included into module Cat)
#
# A **lens** `(get : S→A, put : S×A→S)` is the bidirectional/backward structure
# behind backpropagation; lenses form a category. A **Para** morphism
# `P×A → B` is a parametric map — a learnable layer; Para morphisms compose by
# pairing parameters. Together (Para ∘ Lens) they are the modern categorical
# account of gradient-based learning (Cruttwell–Gavranović–Ghani–Wilson–Zanasi).
# Everything is finite (over `FinSet`), so the lens laws are decidable and
# Lean-certifiable (`Optics.lean`).
# ============================================================================

# ---- Lenses (very-well-behaved / cartesian form: get:S→A, put:S×A→S) ----

"""
    Lens(S, A, get, put)

A lens with focus `A` on state `S`: `get : S → A` reads the focus and
`put : S×A → S` updates it. (`put`'s domain is the product `S × A`.)
"""
struct Lens
    S::Vector{Any}
    A::Vector{Any}
    get::FinFunction          # S → A
    put::FinFunction          # S×A → S
end

"""Cartesian product `X×Y` as a `FinSet` of `(x, y)` pairs (the carrier for lens domains)."""
_prod(X::Vector, Y::Vector) = FinSet(Any[(x, y) for x in X for y in Y])

"""`lens_id(S)` — the identity lens (`get = id`, `put` keeps the new value)."""
function lens_id(S)
    Sv = collect(S)
    SS = _prod(Sv, Sv)
    Lens(Sv, Sv,
         FinFunction(FinSet(Sv), FinSet(Sv), Dict{Any,Any}(s => s for s in Sv)),
         FinFunction(SS, FinSet(Sv), Dict{Any,Any}((s, s2) => s2 for (s, s2) in SS.elements)))
end

"""`lens_compose(l, m)` — sequential composition of `l : S⇄A` and `m : A⇄B` into `S⇄B`."""
function lens_compose(l::Lens, m::Lens)
    Sv, Bv = l.S, m.A
    SB = _prod(Sv, Bv)
    get = FinFunction(FinSet(Sv), FinSet(Bv), Dict{Any,Any}(s => m.get(l.get(s)) for s in Sv))
    put = FinFunction(SB, FinSet(Sv),
        Dict{Any,Any}((s, b) => l.put((s, m.put((l.get(s), b)))) for (s, b) in SB.elements))
    Lens(Sv, Bv, get, put)
end

"""`lens_get_put(l)` — the GetPut law: `put(s, get(s)) = s`."""
lens_get_put(l::Lens) = all(l.put((s, l.get(s))) == s for s in l.S)

"""`lens_put_get(l)` — the PutGet law: `get(put(s, a)) = a`."""
lens_put_get(l::Lens) = all(l.get(l.put((s, a))) == a for s in l.S, a in l.A)

"""`lens_put_put(l)` — the PutPut law: `put(put(s, a), a') = put(s, a')`."""
lens_put_put(l::Lens) = all(l.put((l.put((s, a)), a2)) == l.put((s, a2)) for s in l.S, a in l.A, a2 in l.A)

"""`is_very_well_behaved(l)` — all three lens laws hold."""
is_very_well_behaved(l::Lens) = lens_get_put(l) && lens_put_get(l) && lens_put_put(l)

"""
    record_lens(firsts, seconds) -> Lens

The canonical (very-well-behaved) lens focusing on the first component of a
pair: `S = firsts × seconds`, `get((a,b)) = a`, `put((a,b), a') = (a',b)`.
"""
function record_lens(firsts, seconds)
    A = collect(firsts); Bs = collect(seconds)
    S = Any[(a, b) for a in A for b in Bs]
    SA = _prod(S, A)
    get = FinFunction(FinSet(S), FinSet(A), Dict{Any,Any}((a, b) => a for (a, b) in S))
    put = FinFunction(SA, FinSet(S), Dict{Any,Any}(((a, b), a2) => (a2, b) for ((a, b), a2) in SA.elements))
    Lens(S, A, get, put)
end

# ---- Para: the category of parametric (learnable) morphisms ----

"""
    ParaMap(P, A, B, impl)

A parametric morphism `A → B` with parameter space `P`: `impl : P×A → B`. A
learnable layer (the parameters are what training adjusts).
"""
struct ParaMap
    P::Vector{Any}
    A::Vector{Any}
    B::Vector{Any}
    impl::FinFunction          # P×A → B
end

"""`para_id(A)` — the identity parametric map (trivial one-point parameter)."""
function para_id(A)
    Av = collect(A)
    PA = _prod(Any[:unit], Av)
    ParaMap(Any[:unit], Av, Av, FinFunction(PA, FinSet(Av), Dict{Any,Any}((p, a) => a for (p, a) in PA.elements)))
end

"""
    para_compose(f, g) -> ParaMap

Compose parametric maps `f : A→B` (params `P`) and `g : B→C` (params `Q`) into
`A→C` with parameter space `Q×P`: `((q,p), a) ↦ g(q, f(p, a))`.
"""
function para_compose(f::ParaMap, g::ParaMap)
    QP = Any[(q, p) for q in g.P for p in f.P]
    dom = _prod(QP, f.A)
    impl = FinFunction(dom, FinSet(g.B),
        Dict{Any,Any}(((q, p), a) => g.impl((q, f.impl((p, a)))) for ((q, p), a) in dom.elements))
    ParaMap(QP, f.A, g.B, impl)
end

"""`para_apply(f, p, a)` — run the layer `f` at parameters `p` on input `a`."""
para_apply(f::ParaMap, p, a) = f.impl((p, a))
