# ============================================================================
# kan.jl — finite colimits / limits of Set-valued functors, as Kan extensions
# along the terminal functor C → 1. (included into module Cat)
#
# For X : C → FinSet,
#   colimit(X) = Σ_! X = Lan_!(X)   (the colimit of X)
#   limit(X)   = Π_! X = Ran_!(X)   (the limit of X)
# where ! : C → 1. These are the genuine Kan extensions along ! — the adjoints
# of the constant-functor restriction Δ = !* (colim ⊣ Δ ⊣ lim). Each carries
# its universal (co)cone and a `verify_*` that checks the universal property by
# enumeration, exactly as in `limits.jl`. (Kan extension `Lan_F`/`Ran_F` along
# an arbitrary functor F is the natural next increment.)
# ============================================================================

# ----------------------------------------------------------------------------
# Colimit  (Σ_! = Lan_!)
# ----------------------------------------------------------------------------

struct ColimitCocone
    apex::FinSet
    legs::Dict{Symbol, FinFunction}    # object ↦ leg X(object) → colim
    functor::SetFunctor
end

"""
    colimit(X::SetFunctor) -> ColimitCocone

The colimit of `X : C → FinSet`: the disjoint union `⊔_c X(c)` quotiented by
`x ∼ X(f)(x)` for every generating morphism `f`. This is the left Kan extension
`Σ_! X` along the terminal functor.
"""
function colimit(X::SetFunctor)
    cat = X.cat
    objs = objects(cat)
    tagged = Any[(o, x) for o in objs for x in ob(X, o).elements]
    idx = Dict{Any, Int}(t => i for (i, t) in enumerate(tagged))
    parent = collect(1:length(tagged))
    find(i) = (while parent[i] != i; parent[i] = parent[parent[i]]; i = parent[i]; end; i)
    for (n, s, t) in cat.edges
        fn = X.edge_map[n]
        for x in ob(X, s).elements
            i, j = find(idx[(s, x)]), find(idx[(t, fn(x))])
            i == j || (parent[i] = j)
        end
    end
    repof = Dict{Any, Any}()
    for t in tagged
        repof[t] = tagged[find(idx[t])]
    end
    apex = FinSet(unique(values(repof)))
    legs = Dict{Symbol, FinFunction}(
        o => FinFunction(ob(X, o), apex, Dict{Any,Any}(x => repof[(o, x)] for x in ob(X, o).elements))
        for o in objs)
    ColimitCocone(apex, legs, X)
end

"""Is a family of maps `q_c : X(c) → Y` a cocone (compatible with all edges)?"""
function _is_cocone(X::SetFunctor, q::AbstractDict)
    for (n, s, t) in X.cat.edges
        compose(X.edge_map[n], q[t]) == q[s] || return false
    end
    true
end

"""Mediating `colim X → Y` for a cocone `q`."""
function comediate(col::ColimitCocone, q::AbstractDict)
    _is_cocone(col.functor, q) || throw(ArgumentError("q is not a cocone under X"))
    X = col.functor
    Y = first(values(q)).cod
    m = Dict{Any,Any}()
    for o in objects(X.cat), x in ob(X, o).elements
        r = col.legs[o](x)
        haskey(m, r) || (m[r] = q[o](x))
    end
    FinFunction(col.apex, Y, m)
end

"""Verify the colimit universal property against probe objects."""
function verify_colimit(col::ColimitCocone; probes=_DEFAULT_PROBES)
    X = col.functor; objs = objects(X.cat)
    for Y in probes
        cand = [(_all_functions(ob(X, o), Y)) for o in objs]
        for combo in Iterators.product(cand...)
            q = Dict{Symbol, FinFunction}(objs[i] => combo[i] for i in eachindex(objs))
            _is_cocone(X, q) || continue
            u = comediate(col, q)
            all(compose(col.legs[o], u) == q[o] for o in objs) || return false
            count(v -> all(compose(col.legs[o], v) == q[o] for o in objs),
                  _all_functions(col.apex, Y)) == 1 || return false
        end
    end
    true
end

# ----------------------------------------------------------------------------
# Limit  (Π_! = Ran_!)
# ----------------------------------------------------------------------------

struct LimitCone
    apex::FinSet
    legs::Dict{Symbol, FinFunction}    # object ↦ leg lim → X(object)
    functor::SetFunctor
    order::Vector{Symbol}
end

"""
    limit(X::SetFunctor) -> LimitCone

The limit of `X : C → FinSet`: the set of compatible families
`(x_c)_c ∈ ∏_c X(c)` with `X(f)(x_s) = x_t` for every generating `f : s → t`.
This is the right Kan extension `Π_! X` along the terminal functor.
"""
function limit(X::SetFunctor)
    cat = X.cat
    objs = objects(cat)
    families = Any[]
    for combo in Iterators.product((ob(X, o).elements for o in objs)...)
        fam = Dict{Symbol, Any}(objs[i] => combo[i] for i in eachindex(objs))
        ok = all(X.edge_map[n](fam[s]) == fam[t] for (n, s, t) in cat.edges)
        ok && push!(families, Tuple(fam[o] for o in objs))
    end
    apex = FinSet(families)
    legs = Dict{Symbol, FinFunction}(
        objs[i] => FinFunction(apex, ob(X, objs[i]), Dict{Any,Any}(e => e[i] for e in families))
        for i in eachindex(objs))
    LimitCone(apex, legs, X, objs)
end

"""Is a family of maps `q_c : Y → X(c)` a cone (compatible with all edges)?"""
function _is_cone(X::SetFunctor, q::AbstractDict)
    for (n, s, t) in X.cat.edges
        compose(q[s], X.edge_map[n]) == q[t] || return false
    end
    true
end

"""Mediating `Y → lim X` for a cone `q`."""
function mediate(lim::LimitCone, q::AbstractDict)
    _is_cone(lim.functor, q) || throw(ArgumentError("q is not a cone over X"))
    Y = first(values(q)).dom
    FinFunction(Y, lim.apex,
        Dict{Any,Any}(y => Tuple(q[o](y) for o in lim.order) for y in Y.elements))
end

"""Verify the limit universal property against probe objects."""
function verify_limit(lim::LimitCone; probes=_DEFAULT_PROBES)
    X = lim.functor; objs = objects(X.cat)
    for Y in probes
        cand = [(_all_functions(Y, ob(X, o))) for o in objs]
        for combo in Iterators.product(cand...)
            q = Dict{Symbol, FinFunction}(objs[i] => combo[i] for i in eachindex(objs))
            _is_cone(X, q) || continue
            u = mediate(lim, q)
            all(compose(u, lim.legs[o]) == q[o] for o in objs) || return false
            count(v -> all(compose(v, lim.legs[o]) == q[o] for o in objs),
                  _all_functions(Y, lim.apex)) == 1 || return false
        end
    end
    true
end

# Kan-extension-along-the-terminal aliases (Σ_! and Π_!).
const left_kan_along_terminal = colimit
const right_kan_along_terminal = limit
