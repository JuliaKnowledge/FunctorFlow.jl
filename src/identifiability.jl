# ============================================================================
# identifiability.jl — Shpitser-Pearl ID algorithm for causal identifiability
# ============================================================================
#
# Implements Algorithm 1 from
#   Shpitser & Pearl (2008). "Complete Identification Methods for the Causal
#   Hierarchy." Journal of Machine Learning Research 9: 1941-1979.
#
# Given a causal DAG `G` over observed variables `V` (directed edges encode
# functional dependencies, bidirected edges encode latent common causes) and
# a query "what is P(y | do(x))?", the algorithm returns either a symbolic
# expression for the post-intervention distribution (when identifiable) or a
# hedge witness (when not identifiable). The algorithm is sound and complete:
# if it returns FAIL, the effect is provably non-identifiable.
# ============================================================================

# ------------------------------------------------------------------
# Causal DAG with bidirected (latent) edges
# ------------------------------------------------------------------

"""
    CausalDAG(nodes, directed, bidirected)

A causal DAG over observed variables. Directed edges `(u, v)` encode
functional dependence; bidirected edges `(u, v)` encode an unobserved
common cause shared by `u` and `v` (i.e. an Acyclic Directed Mixed Graph,
ADMG).
"""
struct CausalDAG
    nodes::Vector{Symbol}
    directed::Vector{Tuple{Symbol, Symbol}}
    bidirected::Vector{Tuple{Symbol, Symbol}}
end

function CausalDAG(; nodes::Vector{Symbol},
                    directed::Vector{<:Tuple}=Tuple{Symbol,Symbol}[],
                    bidirected::Vector{<:Tuple}=Tuple{Symbol,Symbol}[])
    d = Tuple{Symbol,Symbol}[(Symbol(a), Symbol(b)) for (a, b) in directed]
    b = Tuple{Symbol,Symbol}[(Symbol(a), Symbol(b)) for (a, b) in bidirected]
    for (u, v) in d
        u in nodes || throw(ArgumentError("Edge endpoint $u not in nodes"))
        v in nodes || throw(ArgumentError("Edge endpoint $v not in nodes"))
    end
    for (u, v) in b
        u in nodes || throw(ArgumentError("Bidirected endpoint $u not in nodes"))
        v in nodes || throw(ArgumentError("Bidirected endpoint $v not in nodes"))
    end
    CausalDAG(copy(nodes), d, b)
end

Base.copy(G::CausalDAG) =
    CausalDAG(copy(G.nodes), copy(G.directed), copy(G.bidirected))

"""
    parents(G::CausalDAG, v)

Direct parents of `v` along directed edges.
"""
function parents(G::CausalDAG, v::Symbol)
    Symbol[u for (u, w) in G.directed if w == v]
end

"""
    ancestors_inclusive(G::CausalDAG, ys)

All ancestors of nodes in `ys` along directed edges, including `ys` themselves.
"""
function ancestors_inclusive(G::CausalDAG, ys::AbstractVector{Symbol})
    visited = Set{Symbol}(ys)
    frontier = collect(ys)
    while !isempty(frontier)
        v = pop!(frontier)
        for u in parents(G, v)
            if !(u in visited)
                push!(visited, u)
                push!(frontier, u)
            end
        end
    end
    Symbol[n for n in G.nodes if n in visited]
end

"""
    subgraph(G::CausalDAG, sub)

Induced subgraph of `G` on the node set `sub` (preserves both directed and
bidirected edges whose endpoints both lie in `sub`).
"""
function subgraph(G::CausalDAG, sub::AbstractVector{Symbol})
    keep = Set{Symbol}(sub)
    nodes = Symbol[n for n in G.nodes if n in keep]
    d = [(u, v) for (u, v) in G.directed if (u in keep) && (v in keep)]
    b = [(u, v) for (u, v) in G.bidirected if (u in keep) && (v in keep)]
    CausalDAG(nodes, d, b)
end

"""
    remove_incoming(G::CausalDAG, x)

Remove all directed edges into nodes in `x` (the mutilation `G_x̄`).
Bidirected edges are preserved.
"""
function remove_incoming(G::CausalDAG, x::AbstractVector{Symbol})
    xs = Set{Symbol}(x)
    d = [(u, v) for (u, v) in G.directed if !(v in xs)]
    CausalDAG(copy(G.nodes), d, copy(G.bidirected))
end

"""
    topological_order(G::CausalDAG) -> Vector{Symbol}

Topological ordering of the DAG along directed edges only. Throws if a
directed cycle is detected.
"""
function topological_order(G::CausalDAG)
    indeg = Dict{Symbol, Int}(n => 0 for n in G.nodes)
    for (u, v) in G.directed
        indeg[v] += 1
    end
    queue = Symbol[n for n in G.nodes if indeg[n] == 0]
    order = Symbol[]
    while !isempty(queue)
        n = popfirst!(queue)
        push!(order, n)
        for (u, v) in G.directed
            if u == n
                indeg[v] -= 1
                indeg[v] == 0 && push!(queue, v)
            end
        end
    end
    length(order) == length(G.nodes) ||
        throw(ArgumentError("CausalDAG has a directed cycle"))
    order
end

"""
    c_components(G::CausalDAG) -> Vector{Vector{Symbol}}

Partition the node set into c-components (confounded components): two nodes
are in the same component iff they are connected by a path of bidirected
edges. Each isolated node is its own component.
"""
function c_components(G::CausalDAG)
    parent = Dict{Symbol, Symbol}(n => n for n in G.nodes)
    function find(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        x
    end
    function union!(a, b)
        ra, rb = find(a), find(b)
        ra == rb || (parent[ra] = rb)
    end
    for (u, v) in G.bidirected
        union!(u, v)
    end
    groups = Dict{Symbol, Vector{Symbol}}()
    for n in G.nodes
        r = find(n)
        push!(get!(groups, r, Symbol[]), n)
    end
    # Stable order: by appearance in G.nodes
    seen = Set{Symbol}()
    out = Vector{Symbol}[]
    for n in G.nodes
        r = find(n)
        if !(r in seen)
            push!(seen, r)
            push!(out, groups[r])
        end
    end
    out
end

# ------------------------------------------------------------------
# IDExpression AST
# ------------------------------------------------------------------

"""
    IDExpression

Symbolic algebra for post-intervention distributions produced by the ID
algorithm. Concrete subtypes:

- `Joint(vars)`               — original joint P over `vars`
- `CondP(vars, conds)`        — conditional P(vars | conds), optionally
                                relative to a recursively derived current P
- `Marginal(margin, expr)`    — Σ_{margin} expr
- `Product(factors)`          — ∏ factors
- `QFactor(subset, order)`    — Q[subset] in topological `order`:
                                ∏_{v ∈ subset} P(v | π_<v), optionally
                                derived from a recursively threaded current P
"""
abstract type IDExpression end

struct Joint <: IDExpression
    vars::Vector{Symbol}
end

struct CondP <: IDExpression
    vars::Vector{Symbol}
    conds::Vector{Symbol}
    base::Union{Nothing, IDExpression}
end

function CondP(vars, conds; base::Union{Nothing, IDExpression}=nothing)
    CondP(Symbol.(collect(vars)), Symbol.(collect(conds)), base)
end

struct Marginal <: IDExpression
    margin::Vector{Symbol}
    expr::IDExpression
end

struct Product <: IDExpression
    factors::Vector{IDExpression}
end

struct QFactor <: IDExpression
    subset::Vector{Symbol}
    order::Vector{Symbol}
    base::Union{Nothing, IDExpression}
end

function QFactor(subset, order; base::Union{Nothing, IDExpression}=nothing)
    QFactor(Symbol.(collect(subset)), Symbol.(collect(order)), base)
end

# ---- Helpers ----

_set(xs) = Set(xs)
_setdiff(a, b) = [x for x in a if !(x in _set(b))]
_intersect(a, b) = [x for x in a if x in _set(b)]
_union_sorted(order, a, b) = [x for x in order if x in _set(a) || x in _set(b)]
_restrict_order(order, subset) = [x for x in order if x in _set(subset)]

_conditional_from(P::Joint, vars::Vector{Symbol}, conds::Vector{Symbol}) = CondP(vars, conds)
_conditional_from(P::IDExpression, vars::Vector{Symbol}, conds::Vector{Symbol}) = CondP(vars, conds; base=P)

_qfactor_from(P::Joint, subset::Vector{Symbol}, order::Vector{Symbol}) = QFactor(subset, order)
_qfactor_from(P::IDExpression, subset::Vector{Symbol}, order::Vector{Symbol}) = QFactor(subset, order; base=P)

function _make_marginal(margin::Vector{Symbol}, expr::IDExpression)
    isempty(margin) && return expr
    Marginal(margin, expr)
end

function _make_product(factors::Vector{<:IDExpression})
    flat = IDExpression[]
    for f in factors
        if f isa Product
            append!(flat, f.factors)
        else
            push!(flat, f)
        end
    end
    length(flat) == 1 ? flat[1] : Product(flat)
end

# ---- Pretty printing ----

function pretty_print(io::IO, e::Joint)
    print(io, "P(", join(e.vars, ", "), ")")
end
function pretty_print(io::IO, e::CondP)
    if e.base === nothing
        print(io, "P(", join(e.vars, ", "))
    else
        print(io, "P[")
        pretty_print(io, e.base)
        print(io, "](", join(e.vars, ", "))
    end
    isempty(e.conds) || print(io, " | ", join(e.conds, ", "))
    print(io, ")")
end
function pretty_print(io::IO, e::Marginal)
    print(io, "[Σ_{", join(e.margin, ","), "} ")
    pretty_print(io, e.expr)
    print(io, "]")
end
function pretty_print(io::IO, e::Product)
    if isempty(e.factors)
        print(io, "1")
    else
        for (i, f) in enumerate(e.factors)
            i > 1 && print(io, " · ")
            pretty_print(io, f)
        end
    end
end
function pretty_print(io::IO, e::QFactor)
    print(io, "Q[", join(e.subset, ","))
    if e.base !== nothing
        print(io, " | ")
        pretty_print(io, e.base)
    end
    print(io, "]")
end

pretty_print(e::IDExpression) = sprint(pretty_print, e)
Base.show(io::IO, ::MIME"text/plain", e::IDExpression) = pretty_print(io, e)

# ------------------------------------------------------------------
# Hedge witness and IdentifiabilityResult
# ------------------------------------------------------------------

"""
    Hedge(F, F_prime, R)

Hedge witness of non-identifiability (Shpitser & Pearl 2008, §3): the pair
of c-forests `(F, F')` with `F' ⊆ F` such that both have a single root set
contained in the do-set `x` (here returned as `R`).
"""
struct Hedge
    F::Vector{Symbol}
    F_prime::Vector{Symbol}
    R::Vector{Symbol}
end

"""
    IdentifiabilityResult

Outcome of running the Shpitser-Pearl ID algorithm on a causal query
`P(y | do(x))`.

Fields:
- `identifiable::Bool`    — true iff the effect is provably identifiable
- `expression`            — symbolic `IDExpression` for `P(y|do(x))` when
                            identifiable, else `nothing`
- `failure_reason`        — `nothing` when identifiable, else a `Symbol`
                            (e.g. `:hedge`)
- `witness`               — `Hedge` structure when `failure_reason == :hedge`
- `algorithm::Symbol`     — which algorithm produced the result
                            (`:id`, `:backdoor`, `:frontdoor`, or `:trivial`)
- `notes::String`         — human-readable annotation
"""
struct IdentifiabilityResult
    identifiable::Bool
    expression::Union{IDExpression, Nothing}
    failure_reason::Union{Symbol, Nothing}
    witness::Union{Hedge, Nothing}
    algorithm::Symbol
    notes::String
end

function Base.show(io::IO, ::MIME"text/plain", r::IdentifiabilityResult)
    println(io, "IdentifiabilityResult(")
    println(io, "  identifiable = ", r.identifiable)
    println(io, "  algorithm    = ", r.algorithm)
    if r.expression !== nothing
        println(io, "  expression   = ", pretty_print(r.expression))
    end
    if r.witness !== nothing
        println(io, "  hedge        = (F=", r.witness.F,
                ", F'=", r.witness.F_prime, ", R=", r.witness.R, ")")
    end
    if r.failure_reason !== nothing
        println(io, "  failure      = ", r.failure_reason)
    end
    print(io, "  notes        = ", r.notes, "\n)")
end

# ------------------------------------------------------------------
# The ID algorithm (Shpitser & Pearl 2008, Algorithm 1)
# ------------------------------------------------------------------

"""
    identify_effect(G::CausalDAG, y, x) -> IdentifiabilityResult

Decide whether `P(y | do(x))` is identifiable from observational data given
the causal DAG `G` (an ADMG). Returns an `IdentifiabilityResult` carrying
the symbolic form when identifiable, or a hedge witness when not.

Implements Algorithm 1 of Shpitser & Pearl (JMLR 2008). The algorithm is
sound and complete.
"""
function identify_effect(G::CausalDAG,
                         y::AbstractVector{Symbol},
                         x::AbstractVector{Symbol})
    y_v = Symbol[Symbol(s) for s in y]
    x_v = Symbol[Symbol(s) for s in x]
    for v in y_v
        v in G.nodes || throw(ArgumentError("y node $v not in DAG"))
    end
    for v in x_v
        v in G.nodes || throw(ArgumentError("x node $v not in DAG"))
    end
    overlap = _intersect(y_v, x_v)
    isempty(overlap) || throw(ArgumentError("y and x must be disjoint; overlap=$overlap"))

    π = topological_order(G)
    P0 = Joint(copy(π))
    try
        expr = _id(y_v, x_v, P0, G, π)
        return IdentifiabilityResult(true, expr, nothing, nothing, :id,
            "Identified by Shpitser-Pearl ID algorithm.")
    catch e
        if e isa _IDFail
            return IdentifiabilityResult(false, nothing, :hedge, e.hedge, :id,
                "Non-identifiable: hedge witness detected (Shpitser-Pearl).")
        else
            rethrow()
        end
    end
end

identify_effect(G::CausalDAG, y::Symbol, x::Symbol) =
    identify_effect(G, [y], [x])
identify_effect(G::CausalDAG, y::Symbol, x::AbstractVector) =
    identify_effect(G, [y], collect(x))
identify_effect(G::CausalDAG, y::AbstractVector, x::Symbol) =
    identify_effect(G, collect(y), [x])

# Internal exception used to short-circuit recursion on FAIL.
struct _IDFail <: Exception
    hedge::Hedge
end

# Recursive workhorse. P is the current expression for the joint over the
# current node set V = G.nodes. π is the original topological ordering of V0
# (used only for q-factor symbolic expansion in line 5).
function _id(y::Vector{Symbol}, x::Vector{Symbol},
             P::IDExpression, G::CausalDAG, π::Vector{Symbol})
    V = G.nodes

    # ---- Line 1: x = ∅ ----
    if isempty(x)
        margin = _setdiff(V, y)
        return _make_marginal(margin, P)
    end

    # ---- Line 2: V ≠ An(Y)_G ----
    An_y = ancestors_inclusive(G, y)
    if Set(An_y) != Set(V)
        G_sub = subgraph(G, An_y)
        # New P is the marginal of the current P onto An_y
        P_sub = _make_marginal(_setdiff(V, An_y), P)
        return _id(y, _intersect(x, An_y), P_sub, G_sub, π)
    end

    # ---- Line 3: forced intervention W = (V\X) \ An_{G_x̄}(Y) ----
    G_xbar = remove_incoming(G, x)
    An_y_xbar = ancestors_inclusive(G_xbar, y)
    W = _setdiff(_setdiff(V, x), An_y_xbar)
    if !isempty(W)
        new_x = _union_sorted(V, x, W)
        return _id(y, new_x, P, G, π)
    end

    # ---- Line 4: c-component decomposition of G[V\X] ----
    V_minus_x = _setdiff(V, x)
    G_minus_x = subgraph(G, V_minus_x)
    Cs = c_components(G_minus_x)
    if length(Cs) > 1
        factors = IDExpression[]
        for S in Cs
            push!(factors, _id(S, _setdiff(V, S), P, G, π))
        end
        margin = _setdiff(V, _union_sorted(V, y, x))
        return _make_marginal(margin, _make_product(factors))
    end

    # ---- Line 5: single c-component S ----
    S = Cs[1]
    Cg = c_components(G)

    # 5a: If C(G) = {V}, FAIL with hedge (V, S).
    if length(Cg) == 1 && Set(Cg[1]) == Set(V)
        throw(_IDFail(Hedge(copy(V), copy(S), copy(x))))
    end

    # Find the c-component T of G that contains S.
    T_idx = findfirst(c -> issubset(Set(S), Set(c)), Cg)
    T_idx === nothing && error("ID invariant: S has no enclosing c-component in G")
    T = Cg[T_idx]

    # 5b: S itself is a c-component of G  →  identifiable directly.
    if Set(S) == Set(T)
        # Σ_{S \ Y} ∏_{Vi ∈ S} P(vi | π_<vi)
        factors = IDExpression[]
        current_order = _restrict_order(π, V)
        for v in S
            π_lt = _prefix(current_order, v)
            push!(factors, _conditional_from(P, [v], π_lt))
        end
        return _make_marginal(_setdiff(S, y), _make_product(factors))
    end

    # 5c: S ⊊ T → recurse with Q[T] as the new P, restricted to G[T].
    GT = subgraph(G, T)
    QT = _qfactor_from(P, copy(T), _restrict_order(π, V))
    return _id(y, _intersect(x, T), QT, GT, π)
end

_prefix(order::Vector{Symbol}, v::Symbol) =
    Symbol[order[i] for i in 1:(findfirst(==(v), order) - 1)]

# ------------------------------------------------------------------
# Quick fast-path checks (back-door / front-door)
# ------------------------------------------------------------------

"""
    is_backdoor_admissible(G::CausalDAG, x, y, Z) -> Bool

Check whether `Z` satisfies the back-door criterion for the effect of `x`
on `y` (Pearl 1995):
  1. No node of `Z` is a descendant of any node in `x`.
  2. `Z` blocks every back-door path from `x` to `y`.
"""
function is_backdoor_admissible(G::CausalDAG, x::AbstractVector{Symbol},
                                 y::AbstractVector{Symbol},
                                 Z::AbstractVector{Symbol})
    descendants_of_x = _descendants_inclusive(G, x)
    for z in Z
        z in descendants_of_x && return false
    end
    # Build proper back-door graph: remove outgoing edges from x.
    G_under = _remove_outgoing(G, x)
    return _d_separated(G_under, x, y, Set(Z))
end

function _descendants_inclusive(G::CausalDAG, xs::AbstractVector{Symbol})
    visited = Set{Symbol}(xs)
    frontier = collect(xs)
    while !isempty(frontier)
        v = pop!(frontier)
        for (u, w) in G.directed
            if u == v && !(w in visited)
                push!(visited, w)
                push!(frontier, w)
            end
        end
    end
    visited
end

function _remove_outgoing(G::CausalDAG, x::AbstractVector{Symbol})
    xs = Set{Symbol}(x)
    d = [(u, v) for (u, v) in G.directed if !(u in xs)]
    CausalDAG(copy(G.nodes), d, copy(G.bidirected))
end

# d-separation via the moralization-of-ancestral-graph approach.
function _d_separated(G::CausalDAG, x::AbstractVector{Symbol},
                       y::AbstractVector{Symbol}, Z::Set{Symbol})
    # Ancestral graph of {x ∪ y ∪ Z}.
    targets = unique(vcat(collect(x), collect(y), collect(Z)))
    An = Set(ancestors_inclusive(G, targets))
    # Build undirected "moral" graph: connect parents that share a child.
    children = Dict{Symbol, Vector{Symbol}}(n => Symbol[] for n in G.nodes)
    for (u, v) in G.directed
        push!(children[u], v)
    end
    parents_of = Dict{Symbol, Vector{Symbol}}(n => Symbol[] for n in G.nodes)
    for (u, v) in G.directed
        push!(parents_of[v], u)
    end
    adj = Dict{Symbol, Set{Symbol}}(n => Set{Symbol}() for n in An)
    function add_edge!(a, b)
        if a in An && b in An && a != b
            push!(adj[a], b)
            push!(adj[b], a)
        end
    end
    for (u, v) in G.directed
        add_edge!(u, v)
    end
    for (u, v) in G.bidirected
        add_edge!(u, v)
    end
    # Moralize: connect any two parents (or bidirected partners) of a common
    # node that lies in An.
    for n in An
        ps = copy(parents_of[n])
        # Bidirected partners are also "parents" via latent confounder.
        for (a, b) in G.bidirected
            a == n && push!(ps, b)
            b == n && push!(ps, a)
        end
        unique!(ps)
        for i in 1:length(ps), j in (i+1):length(ps)
            add_edge!(ps[i], ps[j])
        end
    end
    # Remove conditioned nodes Z.
    blocked = Set{Symbol}(Z)
    # BFS from any x to any y avoiding blocked nodes.
    xs = Set(x); ys = Set(y)
    for start in x
        start in An || continue
        visited = Set{Symbol}([start])
        queue = [start]
        while !isempty(queue)
            n = popfirst!(queue)
            (n in ys) && return false
            for nbr in adj[n]
                nbr in blocked && continue
                if !(nbr in visited)
                    push!(visited, nbr)
                    push!(queue, nbr)
                end
            end
        end
    end
    true
end

# ------------------------------------------------------------------
# Backward-compatible shim for is_identifiable on CausalDiagram
# ------------------------------------------------------------------

"""
    is_identifiable(G::CausalDAG, y, x) -> IdentifiabilityResult

Convenience wrapper around `identify_effect` for `CausalDAG` queries.
"""
is_identifiable(G::CausalDAG, y, x) = identify_effect(G, y, x)
