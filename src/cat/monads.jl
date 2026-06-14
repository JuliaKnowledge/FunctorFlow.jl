# ============================================================================
# monads.jl — monads, the monad laws, Kleisli categories, and monads from
# adjunctions. (included into module Cat)
#
# A monad on a small category `C` is an endofunctor `T : C → C` with unit
# `η : Id ⇒ T` and multiplication `μ : T∘T ⇒ T` satisfying the unit and
# associativity laws. As with everything in the kernel these are *checked* by
# enumeration over the (finite) category. A closure operator on a poset is the
# canonical worked example (a poset is a thin category; a monotone, inflationary,
# idempotent endofunctor is exactly a monad).
# ============================================================================

"""
    Monad(functor::FinFunctor; unit, mult)

A monad `(T, η, μ)` on a small category: `functor = T : C → C`,
`unit = η : Id_C ⇒ T`, `mult = μ : T∘T ⇒ T`. Laws are checked by
[`is_monad`](@ref).
"""
struct Monad
    functor::FinFunctor
    unit::FunctorNatTrans
    mult::FunctorNatTrans
end

function Monad(functor::FinFunctor; unit::FunctorNatTrans, mult::FunctorNatTrans)
    functor.dom == functor.cod || throw(ArgumentError("a monad's functor must be an endofunctor"))
    Monad(functor, unit, mult)
end

"""
    is_monad(m::Monad) -> Bool

Verify the monad laws by enumeration over the base category:
naturality of `η` and `μ`, the left/right unit laws
(`μ_c · η_{Tc} = id`, `μ_c · T(η_c) = id`) and associativity
(`μ_c · μ_{Tc} = μ_c · T(μ_c)`).
"""
function is_monad(m::Monad)
    T = m.functor
    C = T.dom
    η, μ = m.unit, m.mult
    is_natural(η) && is_natural(μ) || return false
    for c in objects(C)
        Tc = T.ob_map[c]
        # unit laws
        compose(C, η.components[Tc], μ.components[c]) == id(C, Tc) || return false   # μ·ηT
        compose(C, T(η.components[c]), μ.components[c]) == id(C, Tc) || return false # μ·Tη
        # associativity: μ·μT = μ·Tμ
        lhs = compose(C, μ.components[Tc], μ.components[c])
        rhs = compose(C, T(μ.components[c]), μ.components[c])
        lhs == rhs || return false
    end
    true
end

# ----------------------------------------------------------------------------
# Kleisli category
# ----------------------------------------------------------------------------

"""`kleisli_hom(m, a, b)` — Kleisli morphisms `a → b`, i.e. `Hom_C(a, T(b))`."""
kleisli_hom(m::Monad, a, b) = homset(m.functor.dom, Symbol(a), m.functor.ob_map[Symbol(b)])

"""`kleisli_id(m, a)` — the Kleisli identity at `a`, i.e. `η_a : a → T(a)`."""
kleisli_id(m::Monad, a) = m.unit.components[Symbol(a)]

"""
    kleisli_compose(m, f, g)

Kleisli composition of `f : a → T(b)` and `g : b → T(c)`:
`a →f→ T(b) →T(g)→ T²(c) →μ_c→ T(c)`.
"""
function kleisli_compose(m::Monad, f::PathMor, g::PathMor)
    C = m.functor.dom
    # g : b → T(c); recover c as the object whose T-image is g.cod
    objs = objects(C)
    k = findfirst(o -> m.functor.ob_map[o] == g.cod, objs)
    k === nothing && throw(ArgumentError("kleisli_compose: cod $(g.cod) of g is not in the image of T (g is not a Kleisli morphism for this monad)"))
    cobj = objs[k]
    compose(C, f, compose(C, m.functor(g), m.mult.components[cobj]))
end

"""
    check_kleisli_laws(m::Monad) -> Bool

Confirm the Kleisli construction is a category: `η` is a two-sided identity for
Kleisli composition, and Kleisli composition is associative.
"""
function check_kleisli_laws(m::Monad)
    C = m.functor.dom
    objs = objects(C)
    # identity laws
    for a in objs, b in objs
        for f in kleisli_hom(m, a, b)
            kleisli_compose(m, kleisli_id(m, a), f) == f || return false
            kleisli_compose(m, f, kleisli_id(m, b)) == f || return false
        end
    end
    # associativity
    for a in objs, b in objs, c in objs, d in objs
        for f in kleisli_hom(m, a, b), g in kleisli_hom(m, b, c), h in kleisli_hom(m, c, d)
            lhs = kleisli_compose(m, kleisli_compose(m, f, g), h)
            rhs = kleisli_compose(m, f, kleisli_compose(m, g, h))
            lhs == rhs || return false
        end
    end
    true
end

# ----------------------------------------------------------------------------
# Monad from an adjunction; identity monad
# ----------------------------------------------------------------------------

"""
    monad_from_adjunction(adj::Adjunction) -> Monad

The monad induced on `C` (the domain of the left adjoint `F`) by `F ⊣ G`:
`T = G∘F`, unit `η` the adjunction unit, and `μ_c = G(ε_{F c})`.
"""
function monad_from_adjunction(adj::Adjunction)
    F, G = adj.left, adj.right
    C = F.dom
    T = compose(F, G)                       # G∘F : C → C
    T2 = compose(T, T)
    μ_components = Dict{Symbol, PathMor}()
    for c in objects(C)
        μ_components[c] = G(adj.counit.components[F.ob_map[c]])   # G(ε_{F c})
    end
    μ = FunctorNatTrans(T2, T; components=μ_components)
    Monad(T, adj.unit, μ)
end

"""`identity_monad(C)` — the trivial monad `(Id, id, id)` on `C`."""
function identity_monad(C::AbstractCategory)
    Id = identity_functor(C)
    comps = Dict{Symbol, PathMor}(c => id(C, c) for c in objects(C))
    η = FunctorNatTrans(Id, Id; components=comps)
    μ = FunctorNatTrans(compose(Id, Id), Id; components=comps)
    Monad(Id, η, μ)
end

"""
    closure_monad(C::FinPresentedCat or FreeCat, T_obj) -> Monad

Build the monad on a (thin) poset category given by a closure operator: `T_obj`
maps each object to its closure. Requires the closure to be monotone,
inflationary (`x ≤ T x`) and idempotent (`T(T x) = T x`); in a thin category
these data force the unit/multiplication, and the monad laws then hold.
"""
function closure_monad(C::AbstractCategory, T_obj::AbstractDict)
    objs = objects(C)
    ob_map = Dict{Symbol, Symbol}(c => Symbol(T_obj[c]) for c in objs)
    # functor on generators: edge n:s→t ↦ the unique morphism T(s)→T(t)
    edge_map = Dict{Symbol, PathMor}()
    for (n, s, t) in C.edges
        hs = homset(C, ob_map[s], ob_map[t])
        length(hs) == 1 || throw(ArgumentError("closure_monad needs a thin category; |Hom($(ob_map[s]),$(ob_map[t]))|=$(length(hs))"))
        edge_map[n] = hs[1]
    end
    T = FinFunctor(C, C; ob_map=ob_map, edge_map=edge_map)
    is_functorial(T) || throw(ArgumentError("the supplied closure is not a functor (not monotone?)"))
    # unit η_c : c → T(c)  (unique morphism; inflationary)
    η_c = Dict{Symbol, PathMor}()
    for c in objs
        hs = homset(C, c, ob_map[c])
        length(hs) == 1 || throw(ArgumentError("not inflationary/thin: |Hom($c,$(ob_map[c]))|=$(length(hs))"))
        η_c[c] = hs[1]
    end
    η = FunctorNatTrans(identity_functor(C), T; components=η_c)
    # μ_c : T(T(c)) → T(c)  (unique morphism; idempotent ⇒ T(T c)=T c so this is id)
    T2 = compose(T, T)
    μ_c = Dict{Symbol, PathMor}()
    for c in objs
        hs = homset(C, T2.ob_map[c], ob_map[c])
        length(hs) == 1 || throw(ArgumentError("not idempotent/thin at $c"))
        μ_c[c] = hs[1]
    end
    μ = FunctorNatTrans(T2, T; components=μ_c)
    Monad(T, η, μ)
end
