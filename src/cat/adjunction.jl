# ============================================================================
# adjunction.jl — functor composition, general natural transformations,
# adjunctions (unit/counit + triangle identities), and the restriction
# functor F* whose adjoints are the Kan extensions. (included into module Cat)
# ============================================================================

# ----------------------------------------------------------------------------
# Identity functor and functor composition
# ----------------------------------------------------------------------------

"""`identity_functor(C)` — the identity endofunctor on `C`."""
identity_functor(C::AbstractCategory) = FinFunctor(C, C;
    ob_map=Dict(o => o for o in C.objects),
    edge_map=Dict(n => PathMor(s, t, Symbol[n]) for (n, s, t) in C.edges))

"""`compose(F, G)` — functor composition `G ∘ F` (diagrammatic: `F` then `G`)."""
function compose(F::FinFunctor, G::FinFunctor)
    F.cod == G.dom || throw(ArgumentError("functors not composable: cod(F) ≠ dom(G)"))
    FinFunctor(F.dom, G.cod;
        ob_map=Dict(o => G.ob_map[F.ob_map[o]] for o in F.dom.objects),
        edge_map=Dict(n => G(F.edge_map[n]) for (n, _, _) in F.dom.edges))
end

# ----------------------------------------------------------------------------
# Natural transformations between functors (components are morphisms)
# ----------------------------------------------------------------------------

"""
    FunctorNatTrans(F::FinFunctor, G::FinFunctor; components)

A natural transformation `α : F ⇒ G` between parallel functors `C → D`.
`components` maps each object `c` to a morphism `α_c : F(c) → G(c)` of `D`.
"""
struct FunctorNatTrans
    dom::FinFunctor
    cod::FinFunctor
    components::Dict{Symbol, PathMor}
end

function FunctorNatTrans(F::FinFunctor, G::FinFunctor; components::AbstractDict)
    (F.dom == G.dom && F.cod == G.cod) ||
        throw(ArgumentError("natural transformation requires parallel functors"))
    FunctorNatTrans(F, G, Dict{Symbol, PathMor}(Symbol(k) => v for (k, v) in components))
end

"""
    is_natural(α::FunctorNatTrans) -> Bool

Check the naturality square in `D` for every generating edge `f : c → c'`:
`F(f) · α_{c'} = α_c · G(f)`, and that each `α_c : F(c) → G(c)` is well-typed.
"""
function is_natural(α::FunctorNatTrans)
    F, G = α.dom, α.cod
    C, D = F.dom, F.cod
    for c in C.objects
        haskey(α.components, c) || return false
        comp = α.components[c]
        (comp.dom == F.ob_map[c] && comp.cod == G.ob_map[c]) || return false
    end
    for (n, s, t) in C.edges
        f = PathMor(s, t, Symbol[n])
        compose(D, F(f), α.components[t]) == compose(D, α.components[s], G(f)) || return false
    end
    true
end

# ----------------------------------------------------------------------------
# Adjunctions
# ----------------------------------------------------------------------------

"""
    Adjunction(left::FinFunctor, right::FinFunctor; unit, counit)

An adjunction `F ⊣ G` with `F = left : C → D`, `G = right : D → C`, unit
`η : Id_C ⇒ G∘F` and counit `ε : F∘G ⇒ Id_D`. Validity (naturality of η, ε and
the two triangle identities) is checked by [`is_adjunction`](@ref).
"""
struct Adjunction
    left::FinFunctor
    right::FinFunctor
    unit::FunctorNatTrans
    counit::FunctorNatTrans
end

Adjunction(left::FinFunctor, right::FinFunctor; unit::FunctorNatTrans, counit::FunctorNatTrans) =
    Adjunction(left, right, unit, counit)

"""
    is_adjunction(adj::Adjunction) -> Bool

Verify `F ⊣ G`: η and ε are natural, and both triangle identities hold —
`ε_{Fc} · F(η_c) = id_{Fc}` (for all `c ∈ C`) and
`G(ε_d) · η_{Gd} = id_{Gd}` (for all `d ∈ D`).
"""
function is_adjunction(adj::Adjunction)
    F, G = adj.left, adj.right
    C, D = F.dom, F.cod
    is_natural(adj.unit) && is_natural(adj.counit) || return false
    for c in C.objects
        Fc = F.ob_map[c]
        compose(D, F(adj.unit.components[c]), adj.counit.components[Fc]) == id(D, Fc) || return false
    end
    for d in D.objects
        Gd = G.ob_map[d]
        compose(C, adj.unit.components[Gd], G(adj.counit.components[d])) == id(C, Gd) || return false
    end
    true
end

# ----------------------------------------------------------------------------
# A worked adjunction: the initial object as a left adjoint to `! : C → 1`
# ----------------------------------------------------------------------------

"""The terminal category `1` (one object, only its identity)."""
terminal_category() = FreeCat([:★], Tuple{Symbol,Symbol,Symbol}[])

"""
    initial_object_adjunction(C::FreeCat, initial) -> Adjunction

Build the adjunction `(initial : 1 → C) ⊣ (! : C → 1)` that characterises
`initial` as the initial object of `C`. Errors if `initial` is not initial
(some `Hom(initial, x)` is not a singleton), so a *successful* construction
whose `is_adjunction` returns true is a genuine certificate of initiality.
"""
function initial_object_adjunction(C::AbstractCategory, initial)
    initial = Symbol(initial)
    one = terminal_category()
    F = FinFunctor(one, C; ob_map=Dict(:★ => initial), edge_map=Dict{Symbol,PathMor}())
    G = FinFunctor(C, one;
        ob_map=Dict(o => :★ for o in C.objects),
        edge_map=Dict(n => PathMor(:★, :★, Symbol[]) for (n, _, _) in C.edges))
    unit = FunctorNatTrans(identity_functor(one), compose(F, G);
        components=Dict(:★ => PathMor(:★, :★, Symbol[])))
    counit_components = Dict{Symbol, PathMor}()
    for x in C.objects
        hs = homset(C, initial, x)
        length(hs) == 1 ||
            throw(ArgumentError("$initial is not initial: |Hom($initial, $x)| = $(length(hs))"))
        counit_components[x] = hs[1]
    end
    counit = FunctorNatTrans(compose(G, F), identity_functor(C); components=counit_components)
    Adjunction(F, G, unit, counit)
end

# ----------------------------------------------------------------------------
# Restriction functor F* (the middle of Σ_F ⊣ F* ⊣ Π_F)
# ----------------------------------------------------------------------------

"""
    restrict(X::SetFunctor, F::FinFunctor) -> SetFunctor

Restriction (precomposition) of a copresheaf `X : D → Set` along a functor
`F : C → D`, giving `F*X = X ∘ F : C → Set`. `F*` is the reindexing functor
whose left and right adjoints are the left/right Kan extensions along `F`
(`Σ_F ⊣ F* ⊣ Π_F`) — the genuine categorical Kan extensions, of which the
DSL's `Σ`/`Δ` aggregation primitives are the operational shadow.
"""
function restrict(X::SetFunctor, F::FinFunctor)
    X.cat == F.cod || throw(ArgumentError("restrict: X must be a functor on cod(F)"))
    C = F.dom
    ob_map = Dict{Symbol, FinSet}(c => ob(X, F.ob_map[c]) for c in C.objects)
    edge_map = Dict{Symbol, FinFunction}(
        n => hommap(X, F(PathMor(s, t, Symbol[n]))) for (n, s, t) in C.edges)
    SetFunctor(C; ob_map=ob_map, edge_map=edge_map)
end
