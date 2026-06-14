# ============================================================================
# counterfactuals.jl — Causal counterfactuals grounded in identify_effect
#
# Ports and *improves on* CLIFF_CatAgi's `democritus_counterfactuals.py`. The
# Python version emits templated claim-flips with no identifiability content.
# Here every counterfactual is routed through the complete Shpitser–Pearl ID
# algorithm (`identify_effect`, src/identifiability.jl): a proposed
# intervention `do(X)` on outcome `Y` carries a genuine verdict
# (identifiable + symbolic estimand, or a hedge witness of non-identifiability)
# alongside the predicted direction of effect.
#
# This reflects the central point of Mahadevan's *Cognitive Categorical
# Transformer* (arXiv:2605.28864): observational corpora alone cannot
# distinguish causal from correlational structure — the interventional
# `do`-operator and an identifiability check are what give a counterfactual
# its force.
# ============================================================================

# ---------------------------------------------------------------------------
# Relation → polarity lexicon (mirrors causal_homotopy.relation_polarity)
# ---------------------------------------------------------------------------

const _POSITIVE_RELATIONS = Set([
    "causes", "cause", "increase", "increases", "raise", "raises", "promote",
    "promotes", "leads to", "leads_to", "improve", "improves", "boost",
    "boosts", "enhance", "enhances", "elevate", "elevates", "drives", "drive",
    "amplify", "amplifies", "support", "supports", "positive",
])

const _NEGATIVE_RELATIONS = Set([
    "decrease", "decreases", "reduce", "reduces", "inhibit", "inhibits",
    "lower", "lowers", "suppress", "suppresses", "harm", "harms", "worsen",
    "worsens", "prevent", "prevents", "block", "blocks", "diminish",
    "diminishes", "weaken", "weakens", "negative",
])

"""
    relation_polarity(rel) -> Int

Classify a causal relation phrase as promoting (`+1`), inhibiting (`-1`), or
unknown (`0`). Matching is case-insensitive on the normalised phrase.
"""
function relation_polarity(rel)
    r = lowercase(strip(String(rel)))
    r = replace(r, "_" => " ")
    r in _POSITIVE_RELATIONS && return 1
    r in _NEGATIVE_RELATIONS && return -1
    # substring fallbacks for compound phrases
    for w in _NEGATIVE_RELATIONS
        occursin(w, r) && return -1
    end
    for w in _POSITIVE_RELATIONS
        occursin(w, r) && return 1
    end
    0
end

# ---------------------------------------------------------------------------
# Causal triple
# ---------------------------------------------------------------------------

"""
    CausalTriple(subj, rel, obj; polarity, domain, statement, metadata)

An extracted causal claim `subj --rel--> obj`. `polarity` is derived from
`rel` via [`relation_polarity`](@ref) unless given explicitly.
"""
struct CausalTriple
    subj::Symbol
    rel::Symbol
    obj::Symbol
    polarity::Int
    domain::String
    statement::String
    metadata::Dict{Symbol, Any}
end

function CausalTriple(subj, rel, obj;
                      polarity::Union{Nothing, Integer}=nothing,
                      domain::AbstractString="",
                      statement::AbstractString="",
                      metadata::Dict=Dict{Symbol, Any}())
    pol = polarity === nothing ? relation_polarity(rel) : Int(polarity)
    CausalTriple(Symbol(subj), Symbol(rel), Symbol(obj), pol,
                 String(domain), String(statement),
                 Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    causal_triple(claim::AbstractString; rel="causes", kwargs...) -> CausalTriple

Parse a `"source -> target"` claim string (the Democritus claim format) into a
`CausalTriple`.
"""
function causal_triple(claim::AbstractString; rel="causes", kwargs...)
    src, dst = _claim_endpoints(claim)
    CausalTriple(Symbol(replace(strip(src), " " => "_")), rel,
                 Symbol(replace(strip(dst), " " => "_")); statement=claim, kwargs...)
end

# ---------------------------------------------------------------------------
# Build an (acyclic) CausalDAG from triples
# ---------------------------------------------------------------------------

"""
    build_causal_dag_from_triples(triples; latent_pairs=[]) -> (G, dropped_edges)

Assemble a `CausalDAG` from causal triples: each distinct `subj`/`obj` is a
node and each triple a directed edge. Edges that would introduce a directed
cycle are dropped (recorded in `dropped_edges`) so the result is a valid ADMG
on which `identify_effect` can run. `latent_pairs` adds bidirected
(unobserved-confounder) edges.
"""
function build_causal_dag_from_triples(triples::AbstractVector{CausalTriple};
                                       latent_pairs::AbstractVector=Tuple{Symbol,Symbol}[])
    nodes = Symbol[]
    seen = Set{Symbol}()
    for t in triples
        for n in (t.subj, t.obj)
            n in seen || (push!(nodes, n); push!(seen, n))
        end
    end

    directed = Tuple{Symbol, Symbol}[]
    dropped = Tuple{Symbol, Symbol}[]
    # Greedily add edges, skipping any that close a directed cycle.
    succ = Dict{Symbol, Vector{Symbol}}(n => Symbol[] for n in nodes)
    reachable(from, to) = begin
        # is `to` already reachable from `from` along current edges?
        stack = Symbol[from]; vis = Set{Symbol}()
        while !isempty(stack)
            v = pop!(stack)
            v == to && return true
            v in vis && continue
            push!(vis, v)
            append!(stack, succ[v])
        end
        false
    end
    for t in triples
        t.subj == t.obj && (push!(dropped, (t.subj, t.obj)); continue)
        (t.subj, t.obj) in directed && continue
        if reachable(t.obj, t.subj)          # adding subj→obj would create a cycle
            push!(dropped, (t.subj, t.obj))
            continue
        end
        push!(directed, (t.subj, t.obj))
        push!(succ[t.subj], t.obj)
    end

    bidirected = Tuple{Symbol, Symbol}[(Symbol(a), Symbol(b)) for (a, b) in latent_pairs]
    G = CausalDAG(; nodes=nodes, directed=directed, bidirected=bidirected)
    (G, dropped)
end

# ---------------------------------------------------------------------------
# Path polarity (predicted direction of effect)
# ---------------------------------------------------------------------------

"""Shortest directed path `x → … → y` (node list), or `nothing`."""
function _directed_path(G::CausalDAG, x::Symbol, y::Symbol)
    x == y && return Symbol[x]
    prev = Dict{Symbol, Symbol}()
    queue = Symbol[x]; vis = Set{Symbol}([x])
    while !isempty(queue)
        v = popfirst!(queue)
        for (u, w) in G.directed
            u == v || continue
            w in vis && continue
            prev[w] = v
            w == y && begin
                path = Symbol[y]
                while path[1] != x
                    pushfirst!(path, prev[path[1]])
                end
                return path
            end
            push!(vis, w); push!(queue, w)
        end
    end
    nothing
end

"""Product of edge polarities along a node path, using a `(subj,obj)=>polarity` map."""
function _path_polarity(path::Vector{Symbol}, edge_pol::Dict{Tuple{Symbol,Symbol}, Int})
    isempty(path) && return 0
    sign = 1
    for i in 1:(length(path) - 1)
        p = get(edge_pol, (path[i], path[i+1]), 0)
        p == 0 && return 0          # unknown polarity anywhere ⇒ unknown overall
        sign *= p
    end
    sign
end

# ---------------------------------------------------------------------------
# Counterfactual record
# ---------------------------------------------------------------------------

"""
    Counterfactual

A single interventional counterfactual `do(intervention)` on `outcome`, with
the identifiability verdict from `identify_effect` and the predicted direction
of effect.
"""
struct Counterfactual
    intervention::Symbol
    intervention_level::Symbol          # :increase | :decrease
    outcome::Symbol
    identifiable::Bool
    estimand::Union{IDExpression, Nothing}
    failure_reason::Union{Symbol, Nothing}
    witness::Union{Hedge, Nothing}
    expected_direction::Int             # +1 up, -1 down, 0 unknown / no path
    expected_shift::String
    text::String
    path::Vector{Symbol}
    support::Int
    domain::String
    metadata::Dict{Symbol, Any}
end

_direction_word(d) = d > 0 ? "increase" : d < 0 ? "decrease" : "change ambiguously"
_direction_word_past(d) = d > 0 ? "increased" : d < 0 ? "decreased" : "changed ambiguously"

"""
    counterfactual_effect(G, triples, x, y; intervention_level=:increase, domain="") -> Counterfactual

Build the counterfactual for intervening on `x` (the cause) and observing the
outcome `y`. The identifiability verdict and symbolic estimand come from
`identify_effect(G, [y], [x])`; the predicted direction comes from the product
of edge polarities along a causal path `x → … → y`. `intervention_level`
(`:increase`/`:decrease`) flips the predicted sign.
"""
function counterfactual_effect(G::CausalDAG, triples::AbstractVector{CausalTriple},
                               x::Symbol, y::Symbol;
                               intervention_level::Symbol=:increase,
                               domain::AbstractString="")
    edge_pol = Dict{Tuple{Symbol,Symbol}, Int}()
    for t in triples
        haskey(edge_pol, (t.subj, t.obj)) || (edge_pol[(t.subj, t.obj)] = t.polarity)
    end

    path = something(_directed_path(G, x, y), Symbol[])
    base_dir = _path_polarity(path, edge_pol)
    flip = intervention_level === :decrease ? -1 : 1
    dir = base_dir * flip

    res = identify_effect(G, [y], [x])

    iv = "$(_direction_word(flip)) $(x)"
    shift = isempty(path) ? "no causal path from $(x) to $(y); effect is $(y) ⟂ do($(x))" :
            "$(y) would $(_direction_word_past(dir))"
    verdict = res.identifiable ?
        "identifiable; estimand $(pretty_print(res.expression))" :
        "not identifiable (hedge witness)"
    text = "Had we intervened to $(iv), $(shift) [$(verdict)]."

    Counterfactual(x, intervention_level, y,
                   res.identifiable, res.expression, res.failure_reason, res.witness,
                   dir, shift, text, path, max(0, length(path) - 1), String(domain),
                   Dict{Symbol, Any}(:algorithm => res.algorithm, :notes => res.notes))
end

# friendly alias
counterfactual(G::CausalDAG, triples, x, y; kwargs...) =
    counterfactual_effect(G, triples, Symbol(x), Symbol(y); kwargs...)

# ---------------------------------------------------------------------------
# Batch claim-based API (parity with democritus_counterfactuals.py)
# ---------------------------------------------------------------------------

"""
    build_counterfactuals_from_triples(triples; domain="", limit=24,
                                       intervention_level=:increase, latent_pairs=[]) -> Dict

Convert a set of extracted causal triples into interventional counterfactuals.
Each direct claim `subj → obj` becomes the counterfactual `do(subj)` on `obj`,
checked for identifiability against the assembled causal DAG. Returns a payload
with the counterfactuals, identifiability tallies, non-identifiable hedges, and
a DAG summary.
"""
function build_counterfactuals_from_triples(triples::AbstractVector{CausalTriple};
                                            domain::AbstractString="",
                                            limit::Integer=24,
                                            intervention_level::Symbol=:increase,
                                            latent_pairs::AbstractVector=Tuple{Symbol,Symbol}[])
    G, dropped = build_causal_dag_from_triples(triples; latent_pairs=latent_pairs)
    cfs = Counterfactual[]
    seen = Set{Tuple{Symbol,Symbol}}()
    for t in triples
        length(cfs) >= limit && break
        (t.subj, t.obj) in seen && continue
        push!(seen, (t.subj, t.obj))
        t.subj in G.nodes && t.obj in G.nodes || continue
        push!(cfs, counterfactual_effect(G, triples, t.subj, t.obj;
                                         intervention_level=intervention_level,
                                         domain=isempty(domain) ? t.domain : domain))
    end

    identifiable = count(c -> c.identifiable, cfs)
    Dict{String, Any}(
        "domain" => String(domain),
        "counterfactuals" => [as_dict(c) for c in cfs],
        "counts" => Dict(
            "triples" => length(triples),
            "counterfactuals" => length(cfs),
            "identifiable" => identifiable,
            "non_identifiable" => length(cfs) - identifiable,
            "dropped_cyclic_edges" => length(dropped),
        ),
        "non_identifiable" => [as_dict(c) for c in cfs if !c.identifiable],
        "dag" => Dict(
            "nodes" => String.(G.nodes),
            "edges" => [[String(u), String(v)] for (u, v) in G.directed],
            "latent_pairs" => [[String(u), String(v)] for (u, v) in G.bidirected],
        ),
    )
end

# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

function as_dict(c::Counterfactual)
    Dict(
        "intervention" => String(c.intervention),
        "intervention_level" => String(c.intervention_level),
        "outcome" => String(c.outcome),
        "identifiable" => c.identifiable,
        "estimand" => c.estimand === nothing ? nothing : pretty_print(c.estimand),
        "failure_reason" => c.failure_reason === nothing ? nothing : String(c.failure_reason),
        "hedge" => c.witness === nothing ? nothing : Dict(
            "F" => String.(c.witness.F), "F_prime" => String.(c.witness.F_prime),
            "R" => String.(c.witness.R)),
        "expected_direction" => c.expected_direction,
        "expected_shift" => c.expected_shift,
        "text" => c.text,
        "path" => String.(c.path),
        "support" => c.support,
        "domain" => c.domain,
    )
end

to_json(c::Counterfactual) = JSON3.write(as_dict(c))

function Base.show(io::IO, ::MIME"text/plain", c::Counterfactual)
    print(io, "Counterfactual(do(", c.intervention, ") → ", c.outcome,
          "; ", c.identifiable ? "identifiable" : "non-identifiable",
          ", dir=", c.expected_direction, ")\n  ", c.text)
end
