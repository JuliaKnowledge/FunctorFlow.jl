# ============================================================================
# rel.jl — the category Rel and the powerset (nondeterminism) monad
# (included into module Cat)
#
# Rel (sets and relations) is the Kleisli category of the powerset monad — the
# categorical home of nondeterministic computation. Composition is relational
# composition; the converse gives the dagger structure.
# ============================================================================

"""
    RelMap(dom, cod, pairs)

A morphism of Rel: a relation `R ⊆ dom × cod`, carried as its set of pairs.
"""
struct RelMap
    dom::Vector{Any}
    cod::Vector{Any}
    pairs::Set{Tuple{Any,Any}}
    function RelMap(dom, cod, pairs)
        new(Any[x for x in dom], Any[y for y in cod],
            Set{Tuple{Any,Any}}((a, b) for (a, b) in pairs))
    end
end

Base.:(==)(R::RelMap, S::RelMap) = Set(R.dom) == Set(S.dom) && Set(R.cod) == Set(S.cod) && R.pairs == S.pairs

"""`rel_id(A)` — the identity relation (diagonal)."""
rel_id(A) = RelMap(A, A, Set{Tuple{Any,Any}}((a, a) for a in A))

"""`rel_compose(R, S)` — relational composition `{(a,c) : ∃b. aRb ∧ bSc}`."""
function rel_compose(R::RelMap, S::RelMap)
    pairs = Set{Tuple{Any,Any}}()
    for (a, b) in R.pairs, (b2, c) in S.pairs
        b == b2 && push!(pairs, (a, c))
    end
    RelMap(R.dom, S.cod, pairs)
end

"""`rel_dagger(R)` — the converse relation `Rᵒᵖ` (Rel is a dagger category)."""
rel_dagger(R::RelMap) = RelMap(R.cod, R.dom, Set{Tuple{Any,Any}}((b, a) for (a, b) in R.pairs))

"""`rel_laws(maps)` — identity and associativity of relational composition."""
function rel_laws(maps::AbstractVector{RelMap})
    for R in maps
        rel_compose(rel_id(R.dom), R) == R || return false
        rel_compose(R, rel_id(R.cod)) == R || return false
    end
    for R in maps, S in maps
        Set(R.cod) == Set(S.dom) || continue
        for T in maps
            Set(S.cod) == Set(T.dom) || continue
            rel_compose(rel_compose(R, S), T) == rel_compose(R, rel_compose(S, T)) || return false
        end
    end
    true
end

# ----------------------------------------------------------------------------
# Powerset monad (Rel = its Kleisli category)
# ----------------------------------------------------------------------------

"""`powerset_unit(a)` — the monad unit `η : A → P(A)`, `a ↦ {a}`."""
powerset_unit(a) = Set([a])

"""`powerset_mult(ss)` — the monad multiplication `μ : P(P(A)) → P(A)`, union of a set of sets."""
powerset_mult(ss) = reduce(union, ss; init=Set())

"""
    kleisli_to_rel(dom, k) -> RelMap

The isomorphism `Kleisli(P) ≅ Rel`: a Kleisli morphism `k : A → P(B)` is the
relation `{(a,b) : b ∈ k(a)}`.
"""
function kleisli_to_rel(dom, k::AbstractDict)
    pairs = Set{Tuple{Any,Any}}()
    cod = Set()
    for a in dom, b in k[a]
        push!(pairs, (a, b)); push!(cod, b)
    end
    RelMap(collect(dom), collect(cod), pairs)
end
