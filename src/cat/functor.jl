# ============================================================================
# functor.jl — functors, Set-valued functors (C-Sets), natural transformations
# (included into module Cat)
# ============================================================================

# ----------------------------------------------------------------------------
# FinFunctor — a functor between free categories
# ----------------------------------------------------------------------------

"""
    FinFunctor(dom::FreeCat, cod::FreeCat; ob_map, edge_map)

A functor `dom → cod`. `ob_map` sends each object to an object; `edge_map`
sends each *generating edge* (by name) to a morphism (`PathMor`) of `cod`. The
action on a general path is the composite of its edges' images, so
functoriality (`F(id)=id`, `F(g∘f)=F(g)∘F(f)`) holds by construction once the
edge images are well-typed — which [`is_functorial`](@ref) checks.
"""
struct FinFunctor
    dom::AbstractCategory
    cod::AbstractCategory
    ob_map::Dict{Symbol, Symbol}
    edge_map::Dict{Symbol, PathMor}
end

function FinFunctor(dom::AbstractCategory, cod::AbstractCategory; ob_map::AbstractDict, edge_map::AbstractDict)
    om = Dict{Symbol, Symbol}(Symbol(k) => Symbol(v) for (k, v) in ob_map)
    em = Dict{Symbol, PathMor}(Symbol(k) => v for (k, v) in edge_map)
    FinFunctor(dom, cod, om, em)
end

"""Apply a `FinFunctor` to a morphism (path)."""
function (F::FinFunctor)(p::PathMor)
    result = id(F.cod, F.ob_map[p.dom])
    for e in p.edges
        result = compose(F.cod, result, F.edge_map[e])
    end
    result
end

"""
    is_functorial(F::FinFunctor) -> Bool

Check that each generating edge `n : s → t` maps to a morphism
`ob_map[s] → ob_map[t]` in the codomain (which, for free domains, is exactly
functoriality).
"""
function is_functorial(F::FinFunctor)
    for (n, s, t) in F.dom.edges
        haskey(F.ob_map, s) && haskey(F.ob_map, t) || return false
        haskey(F.edge_map, n) || return false
        img = F.edge_map[n]
        (img.dom == F.ob_map[s] && img.cod == F.ob_map[t]) || return false
        img.edges ⊆ [e[1] for e in F.cod.edges] || return false
    end
    # over a presented domain, a functor must send congruent paths together
    if F.dom isa FinPresentedCat
        for (p, q) in F.dom.relations
            F(p) == F(q) || return false
        end
    end
    true
end

# ----------------------------------------------------------------------------
# SetFunctor — a copresheaf C → FinSet (i.e. a C-Set / ACSet)
# ----------------------------------------------------------------------------

"""
    SetFunctor(cat::FreeCat; ob_map, edge_map)

A functor `cat → FinSet`: `ob_map` assigns a [`FinSet`](@ref) to each object,
`edge_map` a [`FinFunction`](@ref) to each generating edge. This is precisely a
**C-Set** (copresheaf) on the schema `cat` — generalising graphs and relational
data. Construction validates that each edge function's (co)domain matches the
sets assigned to the edge's endpoints.
"""
struct SetFunctor
    cat::AbstractCategory
    ob_map::Dict{Symbol, FinSet}
    edge_map::Dict{Symbol, FinFunction}
end

function SetFunctor(cat::AbstractCategory; ob_map::AbstractDict, edge_map::AbstractDict)
    om = Dict{Symbol, FinSet}(Symbol(k) => v for (k, v) in ob_map)
    em = Dict{Symbol, FinFunction}(Symbol(k) => v for (k, v) in edge_map)
    for o in cat.objects
        haskey(om, o) || throw(ArgumentError("SetFunctor missing a FinSet for object $o"))
    end
    for (n, s, t) in cat.edges
        haskey(em, n) || throw(ArgumentError("SetFunctor missing a FinFunction for edge $n"))
        em[n].dom == om[s] || throw(ArgumentError("edge $n: dom mismatch with object $s"))
        em[n].cod == om[t] || throw(ArgumentError("edge $n: cod mismatch with object $t"))
    end
    SetFunctor(cat, om, em)
end

"""`ob(F, x)` — the FinSet assigned to object `x`."""
ob(F::SetFunctor, x) = F.ob_map[Symbol(x)]

"""`hommap(F, p::PathMor)` — the FinFunction `F(p) : F(dom p) → F(cod p)`."""
function hommap(F::SetFunctor, p::PathMor)
    result = id(F.ob_map[p.dom])
    for e in p.edges
        result = compose(result, F.edge_map[e])
    end
    result
end

"""
    is_functorial(F::SetFunctor) -> Bool

Verify `F` respects identities and composition over *all* morphisms of the
schema (not just generators): `F(id)=id` and `F(p·q)=F(p)·F(q)`.
"""
function is_functorial(F::SetFunctor)
    objs = objects(F.cat)
    allmors = PathMor[]
    for a in objs, b in objs
        append!(allmors, homset(F.cat, a, b))
    end
    for x in objs
        hommap(F, id(F.cat, x)) == id(F.ob_map[x]) || return false
    end
    for p in allmors, q in allmors
        p.cod == q.dom || continue
        hommap(F, compose(F.cat, p, q)) == compose(hommap(F, p), hommap(F, q)) || return false
    end
    # over a presented category, F must also respect the generating relations
    if F.cat isa FinPresentedCat
        for (p, q) in F.cat.relations
            hommap(F, p) == hommap(F, q) || return false
        end
    end
    true
end

# ----------------------------------------------------------------------------
# CatNatTrans — a natural transformation of SetFunctors
# ----------------------------------------------------------------------------

"""
    CatNatTrans(dom::SetFunctor, cod::SetFunctor; components)

A natural transformation `α : F ⇒ G` between copresheaves on the same schema.
`components` maps each object `x` to a [`FinFunction`](@ref) `α_x : F(x) → G(x)`.
"""
struct CatNatTrans
    dom::SetFunctor
    cod::SetFunctor
    components::Dict{Symbol, FinFunction}
end

function CatNatTrans(F::SetFunctor, G::SetFunctor; components::AbstractDict)
    F.cat === G.cat || F.cat == G.cat ||
        throw(ArgumentError("natural transformation requires F and G on the same schema"))
    comp = Dict{Symbol, FinFunction}(Symbol(k) => v for (k, v) in components)
    CatNatTrans(F, G, comp)
end

"""
    is_natural(α::CatNatTrans) -> Bool

Check every naturality square commutes: for each generating edge `f : x → y`,
`α_y ∘ F(f) = G(f) ∘ α_x` (diagrammatically, `F(f)·α_y = α_x·G(f)`).
"""
function is_natural(α::CatNatTrans)
    F, G = α.dom, α.cod
    for x in F.cat.objects
        haskey(α.components, x) || return false
        c = α.components[x]
        (c.dom == ob(F, x) && c.cod == ob(G, x)) || return false
    end
    for (n, x, y) in F.cat.edges
        left = compose(F.edge_map[n], α.components[y])   # F(f) then α_y
        right = compose(α.components[x], G.edge_map[n])   # α_x then G(f)
        left == right || return false
    end
    true
end
