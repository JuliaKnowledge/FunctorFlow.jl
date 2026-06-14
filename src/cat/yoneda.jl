# ============================================================================
# yoneda.jl — representable functors, presheaves, and the Yoneda lemma
# (included into module Cat)
#
# Over a FreeCat (finite hom-sets) every representable is a concrete SetFunctor
# and the Yoneda bijection  Nat(Hom(c,-), F) ≅ F(c)  is *computable*. The
# functions here construct both directions and verify, by enumeration, that
# they are mutually inverse and that the constructed transformations are
# natural — a genuine computational witness of the Yoneda lemma.
# ============================================================================

"""
    op(C::FreeCat) -> FreeCat

The opposite category: same objects, every generating edge reversed. (Stays
acyclic, so hom-sets remain finite.)
"""
op(C::FreeCat) = FreeCat(C.objects, [(n, t, s) for (n, s, t) in C.edges])

"""
    representable_functor(C::FreeCat, c) -> SetFunctor

The covariant representable copresheaf `Hom(c, -) : C → FinSet` (the Yoneda
embedding of `c`). On an object `x` it is the set of morphisms `c → x`; on an
edge `f : x → y` it is post-composition `(c→x) ↦ (c→y)`.
"""
function representable_functor(C::AbstractCategory, c)
    c = Symbol(c)
    ob_map = Dict{Symbol, FinSet}(x => FinSet(homset(C, c, x)) for x in C.objects)
    edge_map = Dict{Symbol, FinFunction}()
    for (n, x, y) in C.edges
        e = PathMor(x, y, Symbol[n])
        edge_map[n] = FinFunction(ob_map[x], ob_map[y],
            Dict{Any,Any}(p => compose(C, p, e) for p in ob_map[x].elements))
    end
    SetFunctor(C; ob_map=ob_map, edge_map=edge_map)
end

"""
    representable_presheaf(C::FreeCat, c) -> SetFunctor

The contravariant representable presheaf `Hom(-, c) : Cᵒᵖ → FinSet`, realised as
the covariant representable of `c` in the opposite category `Cᵒᵖ`.
"""
representable_presheaf(C::FreeCat, c) = representable_functor(op(C), c)

"""
    yoneda_map(C, c, F::SetFunctor, element) -> CatNatTrans

The forward Yoneda map `F(c) → Nat(Hom(c,-), F)`. Given `element ∈ F(c)`, build
the natural transformation `α` with `α_x(p : c→x) = F(p)(element)`.
"""
function yoneda_map(C::AbstractCategory, c, F::SetFunctor, element)
    c = Symbol(c)
    element in ob(F, c) || throw(ArgumentError("element $(repr(element)) ∉ F($c)"))
    yc = representable_functor(C, c)
    components = Dict{Symbol, FinFunction}()
    for x in C.objects
        dom_set = ob(yc, x)        # Hom(c, x)
        cod_set = ob(F, x)         # F(x)
        components[x] = FinFunction(dom_set, cod_set,
            Dict{Any,Any}(p => hommap(F, p)(element) for p in dom_set.elements))
    end
    CatNatTrans(yc, F; components=components)
end

"""
    yoneda_inverse(C, c, α::CatNatTrans) -> element of F(c)

The inverse Yoneda map `Nat(Hom(c,-), F) → F(c)`, evaluating `α` at the
identity: `α_c(id_c)`.
"""
function yoneda_inverse(C::AbstractCategory, c, α::CatNatTrans)
    c = Symbol(c)
    α.components[c](id(C, c))
end

# ----------------------------------------------------------------------------
# Verification of the Yoneda lemma by enumeration
# ----------------------------------------------------------------------------

"""All FinFunctions `A → B` (used to enumerate candidate components)."""
function _all_functions(A::FinSet, B::FinSet)
    A.elements |> isempty && return FinFunction[FinFunction(A, B, Dict{Any,Any}())]
    out = FinFunction[]
    for choice in Iterators.product((B.elements for _ in A.elements)...)
        push!(out, FinFunction(A, B, Dict{Any,Any}(zip(A.elements, choice))))
    end
    out
end

"""
    count_nat_transformations(F, G; bound=100_000) -> Int

Number of natural transformations `F ⇒ G`, by brute-force enumeration of
component tuples. Returns `-1` if the search space exceeds `bound`.
"""
function count_nat_transformations(F::SetFunctor, G::SetFunctor; bound::Int=100_000)
    objs = F.cat.objects
    cand = Dict{Symbol, Vector{FinFunction}}(x => _all_functions(ob(F, x), ob(G, x)) for x in objs)
    total = prod(length(cand[x]) for x in objs; init=1)
    total > bound && return -1
    n = 0
    for combo in Iterators.product((cand[x] for x in objs)...)
        components = Dict{Symbol, FinFunction}(objs[i] => combo[i] for i in eachindex(objs))
        is_natural(CatNatTrans(F, G; components=components)) && (n += 1)
    end
    n
end

"""
    yoneda_lemma_holds(C, c, F::SetFunctor) -> Bool

Computationally verify the Yoneda lemma at `(c, F)`:
1. `yoneda_inverse ∘ yoneda_map = id` on `F(c)`;
2. every transformation produced by `yoneda_map` is natural;
3. `yoneda_map` is injective; and, when the search space is small enough,
4. the count of *all* natural transformations equals `|F(c)|` (bijectivity).
"""
function yoneda_lemma_holds(C::AbstractCategory, c, F::SetFunctor)
    c = Symbol(c)
    elements = ob(F, c).elements
    images = []
    for e in elements
        α = yoneda_map(C, c, F, e)
        is_natural(α) || return false                       # (2)
        yoneda_inverse(C, c, α) == e || return false        # (1)
        push!(images, α.components)
    end
    # (3) injectivity: distinct elements ⇒ distinct transformations
    for i in eachindex(images), j in (i+1):length(images)
        images[i] == images[j] && return false
    end
    # (4) bijectivity when feasible
    total = count_nat_transformations(representable_functor(C, c), F)
    total == -1 || total == length(elements) || return false
    true
end

# ----------------------------------------------------------------------------
# Representability
# ----------------------------------------------------------------------------

"""Is a FinFunction a bijection?"""
function _is_iso(f::FinFunction)
    length(f.dom) == length(f.cod) || return false
    length(Set(f.map[x] for x in f.dom.elements)) == length(f.dom)
end

"""
    is_representable(F::SetFunctor) -> NamedTuple

Search for an object `c` and `element ∈ F(c)` whose Yoneda transformation
`Hom(c,-) ⇒ F` is an isomorphism (every component a bijection), witnessing
`F ≅ Hom(c,-)`. Returns `(representable, witness, element)`.
"""
function is_representable(F::SetFunctor)
    C = F.cat
    for c in C.objects
        for e in ob(F, c).elements
            α = yoneda_map(C, c, F, e)
            if all(_is_iso(α.components[x]) for x in C.objects)
                return (representable=true, witness=c, element=e)
            end
        end
    end
    (representable=false, witness=nothing, element=nothing)
end
