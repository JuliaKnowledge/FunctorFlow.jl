# ============================================================================
# comonad.jl — comonads: context-dependent computation (dual of monads.jl)
# (included into module Cat)
#
# A comonad `(T, ε, δ)` on a small category is an endofunctor with counit
# `ε : T ⇒ Id` (extract) and comultiplication `δ : T ⇒ T²` (duplicate),
# satisfying the dual of the monad laws. Comonads model *context-dependent*
# computation — the categorical account of convolution, attention windows, and
# streaming/stencil computations (each output depends on a local context). This
# completes the (co)monad duality alongside the (co)algebra and (co)limit duals.
# ============================================================================

"""
    Comonad(functor::FinFunctor; counit, comult)

A comonad `(T, ε, δ)`: `functor = T : C → C`, `counit = ε : T ⇒ Id`,
`comult = δ : T ⇒ T∘T`. Laws checked by [`is_comonad`](@ref).
"""
struct Comonad
    functor::FinFunctor
    counit::FunctorNatTrans
    comult::FunctorNatTrans
end

function Comonad(functor::FinFunctor; counit::FunctorNatTrans, comult::FunctorNatTrans)
    functor.dom == functor.cod || throw(ArgumentError("a comonad's functor must be an endofunctor"))
    Comonad(functor, counit, comult)
end

"""
    is_comonad(w::Comonad) -> Bool

Verify the comonad laws by enumeration: naturality of `ε`, `δ`, the counit laws
(`ε_{Tc} · δ_c = id`, `T(ε_c) · δ_c = id`) and coassociativity
(`δ_{Tc} · δ_c = T(δ_c) · δ_c`).
"""
function is_comonad(w::Comonad)
    T = w.functor; C = T.dom
    ε, δ = w.counit, w.comult
    is_natural(ε) && is_natural(δ) || return false
    for c in objects(C)
        Tc = T.ob_map[c]
        # counit laws (dual of the monad unit laws)
        compose(C, δ.components[c], ε.components[Tc]) == id(C, Tc) || return false   # (εT)·δ
        compose(C, δ.components[c], T(ε.components[c])) == id(C, Tc) || return false # (Tε)·δ
        # coassociativity (dual of associativity)
        lhs = compose(C, δ.components[c], δ.components[Tc])
        rhs = compose(C, δ.components[c], T(δ.components[c]))
        lhs == rhs || return false
    end
    true
end

"""`identity_comonad(C)` — the trivial comonad `(Id, id, id)` on `C`."""
function identity_comonad(C::AbstractCategory)
    Id = identity_functor(C)
    comps = Dict{Symbol, PathMor}(c => id(C, c) for c in objects(C))
    ε = FunctorNatTrans(Id, Id; components=comps)
    δ = FunctorNatTrans(Id, compose(Id, Id); components=comps)
    Comonad(Id, ε, δ)
end

"""
    comonad_from_adjunction(adj::Adjunction) -> Comonad

The comonad induced on `D` (the codomain of the left adjoint `F`) by `F ⊣ G`:
`T = F∘G`, counit `ε` the adjunction counit, and `δ_d = F(η_{G d})` — dual to
[`monad_from_adjunction`](@ref).
"""
function comonad_from_adjunction(adj::Adjunction)
    F, G = adj.left, adj.right
    D = F.cod
    T = compose(G, F)                       # F∘G : D → D
    T2 = compose(T, T)
    δ_components = Dict{Symbol, PathMor}()
    for d in objects(D)
        δ_components[d] = F(adj.unit.components[G.ob_map[d]])   # F(η_{G d})
    end
    δ = FunctorNatTrans(T, T2; components=δ_components)
    Comonad(T, adj.counit, δ)
end
