# ============================================================================
# topos.jl — the subobject classifier of the presheaf topos [C, Set]
# (included into module Cat)
#
# For a small category `C`, the copresheaf topos `[C, Set]` has a subobject
# classifier `Ω`: a copresheaf with `Ω(c) = {cosieves on c}` (a cosieve = a set
# of morphisms out of `c` closed under post-composition). Every subobject
# `A ↪ X` is classified by a unique characteristic map `χ : X → Ω` with
# `A = χ⁻¹(true)`. This is all *computable* over finite `C`, so the topos layer
# is re-founded rigorously: `subobject_classifier`, `omega_true`, `classify`,
# and a checkable classification theorem.
# ============================================================================

# canonical key/ordering for a morphism (so cosieves have a canonical form)
_mor_key(m::PathMor) = (string(m.dom), string(m.cod), join(string.(m.edges), "."))

# all morphisms with domain `c`
function _morphisms_out(C::AbstractCategory, c)
    out = PathMor[]
    for d in objects(C), m in homset(C, c, d)
        push!(out, m)
    end
    out
end

# every cosieve on `c`: subset of (morphisms out of c) closed under post-composition
function _cosieves(C::AbstractCategory, c; max_out::Int=12)
    mors = _morphisms_out(C, c)
    n = length(mors)
    n <= max_out || throw(ArgumentError(
        "subobject_classifier: object $c has $n outgoing morphisms (> max_out=$max_out); " *
        "cosieve enumeration is exponential — raise max_out if you really mean it"))
    out = Vector{PathMor}[]
    for mask in 0:(2^n - 1)
        S = Set(mors[i] for i in 1:n if (mask >> (i - 1)) & 1 == 1)
        closed = true
        for h in S
            for d in objects(C), k in homset(C, h.cod, d)
                if !(compose(C, h, k) in S)
                    closed = false; break
                end
            end
            closed || break
        end
        closed && push!(out, sort(collect(S); by=_mor_key))
    end
    out
end

# the maximal cosieve on c (all morphisms out of c) — the local truth value
_maximal_cosieve(C::AbstractCategory, c) = sort(_morphisms_out(C, c); by=_mor_key)

"""
    subobject_classifier(C) -> SetFunctor

The subobject classifier `Ω : C → Set` of the copresheaf topos `[C, Set]`:
`Ω(c)` is the set of cosieves on `c`, and for `f : c → d`, `Ω(f)` sends a
cosieve `R` to `{g : d → · | f·g ∈ R}`.
"""
function subobject_classifier(C::AbstractCategory)
    objs = objects(C)
    cos = Dict(c => _cosieves(C, c) for c in objs)
    ob_map = Dict{Symbol, FinSet}(c => FinSet(cos[c]) for c in objs)
    edge_map = Dict{Symbol, FinFunction}()
    for (n, c, d) in C.edges
        fe = PathMor(c, d, Symbol[n])
        m = Dict{Any,Any}()
        for R in cos[c]
            Rset = Set(R)
            newR = PathMor[g for g in _morphisms_out(C, d) if compose(C, fe, g) in Rset]
            m[R] = sort(newR; by=_mor_key)
        end
        edge_map[n] = FinFunction(ob_map[c], ob_map[d], m)
    end
    SetFunctor(C; ob_map=ob_map, edge_map=edge_map)
end

"""
    omega_true(C) -> Dict{Symbol, Vector{PathMor}}

The truth values `true_c ∈ Ω(c)` (the maximal cosieve at each object) — the
components of the truth arrow `1 → Ω`.
"""
omega_true(C::AbstractCategory) = Dict(c => _maximal_cosieve(C, c) for c in objects(C))

# is `sub` (object ↦ set of "in" elements) a subfunctor of X? (closed under X(f))
"""
    is_subfunctor(X::SetFunctor, sub::AbstractDict) -> Bool

Check that `sub` (each object ↦ the subset of `X(object)` that is "in") is a
sub-copresheaf of `X`: closed under the action of every generating morphism.
"""
function is_subfunctor(X::SetFunctor, sub::AbstractDict)
    for (n, c, d) in X.cat.edges
        f = hommap(X, PathMor(c, d, Symbol[n]))
        for x in sub[c]
            f(x) in sub[d] || return false
        end
    end
    true
end

"""
    classify(X::SetFunctor, sub::AbstractDict) -> CatNatTrans

The characteristic map `χ : X → Ω` of a subfunctor `sub`. For `x ∈ X(c)`,
`χ_c(x) = {f : c → d | X(f)(x) ∈ sub(d)}`, a cosieve on `c`.
"""
function classify(X::SetFunctor, sub::AbstractDict)
    C = X.cat
    Ω = subobject_classifier(C)
    components = Dict{Symbol, FinFunction}()
    for c in objects(C)
        m = Dict{Any,Any}()
        for x in ob(X, c).elements
            cosieve = PathMor[f for f in _morphisms_out(C, c) if hommap(X, f)(x) in sub[f.cod]]
            m[x] = sort(cosieve; by=_mor_key)
        end
        components[c] = FinFunction(ob(X, c), ob(Ω, c), m)
    end
    CatNatTrans(X, Ω; components=components)
end

"""
    verify_classifies(X, sub, χ) -> Bool

Verify the subobject-classifier theorem for `sub`: `χ` is natural and
`χ⁻¹(true) = sub` (an element is sent to the maximal cosieve iff it is in the
subfunctor).
"""
function verify_classifies(X::SetFunctor, sub::AbstractDict, χ::CatNatTrans)
    is_natural(χ) || return false
    C = X.cat
    tru = omega_true(C)
    for c in objects(C), x in ob(X, c).elements
        in_sub = x in sub[c]
        hits_true = χ.components[c](x) == tru[c]
        in_sub == hits_true || return false
    end
    true
end
