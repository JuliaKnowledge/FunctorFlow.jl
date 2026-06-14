# ============================================================================
# corpus_synthesis.jl — Multi-document causal-claim synthesis
#
# A real port of CLIFF_CatAgi's `democritus_corpus_synthesis.py`. Local claims
# extracted from many documents are synthesised into a coherent corpus-level
# causal graph through four stages:
#
#   1. normalize  — canonicalise subject / relation / object tokens
#   2. glue       — merge wording variants of the same claim across documents
#                   (sheaf gluing of local sections), tallying document support
#   3. truth      — assign a support tier (entailed / strong / provisional / weak)
#   4. synthesize — score coherence (simplicial horn-fill), align to the query,
#                   and rank; surface polarity disagreements
#
# Coherence is measured by the **horn-fill ratio** of the claim complex — the
# proportion of open 2-horns (a→b, b→c with a→c missing) that are filled by a
# transitive edge. This is the same simplicial structural signal that
# Mahadevan's *Cognitive Categorical Transformer* (arXiv:2605.28864) finds to
# be the dominant contributor to model quality, and it is the categorical
# gluing axiom for the claim presheaf made quantitative.
# ============================================================================

# ---------------------------------------------------------------------------
# Local claim
# ---------------------------------------------------------------------------

"""
    CorpusClaim(subj, rel, obj; polarity, document, domain, statement)

A single causal claim extracted from one source `document`. `polarity` is
derived from `rel` via [`relation_polarity`](@ref) unless given.
"""
struct CorpusClaim
    subj::Symbol
    rel::Symbol
    obj::Symbol
    polarity::Int
    document::String
    domain::String
    statement::String
end

function CorpusClaim(subj, rel, obj;
                     polarity::Union{Nothing, Integer}=nothing,
                     document::AbstractString="doc",
                     domain::AbstractString="",
                     statement::AbstractString="")
    pol = polarity === nothing ? relation_polarity(rel) : Int(polarity)
    CorpusClaim(Symbol(subj), Symbol(rel), Symbol(obj), pol,
                String(document), String(domain), String(statement))
end

_entity_tokens(s::Symbol) = Set(split(lowercase(replace(String(s), "_" => " "))))
_jaccard(a::Set, b::Set) = (isempty(a) && isempty(b)) ? 1.0 :
    length(intersect(a, b)) / length(union(a, b))

# ---------------------------------------------------------------------------
# Synthesized (glued) claim
# ---------------------------------------------------------------------------

"""
    SynthesizedClaim

A corpus-level claim formed by gluing wording variants of the same
subject→object relation across documents. Carries the supporting documents,
the support count, the truth tier, and a polarity-conflict flag.
"""
struct SynthesizedClaim
    canonical_subj::Symbol
    canonical_rel::Symbol
    canonical_obj::Symbol
    polarity::Int
    documents::Vector{String}
    support::Int
    variants::Vector{CorpusClaim}
    truth_value::Symbol
    conflicted::Bool
    relevance::Float64
    relevance_label::Symbol
    metadata::Dict{Symbol, Any}
end

"""
    corpus_truth_value(support, total) -> Symbol

Support tier from the number of distinct documents backing a claim:
`:entailed` (all), `:strong_support` (≥50%), `:provisional_support`
(≥40% or ≥2 docs), else `:weak_support`.
"""
function corpus_truth_value(support::Integer, total::Integer)
    total <= 0 && return :weak_support
    frac = support / total
    support == total && return :entailed
    frac >= 0.5 && return :strong_support
    (frac >= 0.4 || support >= 2) && return :provisional_support
    :weak_support
end

"""
    glue_corpus_claims(claims; jaccard_threshold=0.65) -> Vector{SynthesizedClaim}

Merge wording variants of the same claim across documents. Two claims glue when
their subject token sets and object token sets each have Jaccard similarity ≥
`jaccard_threshold` (polarity is *not* used to split — mixed polarities within a
group are flagged as `conflicted`). Document support is counted over distinct
`document` ids.
"""
function glue_corpus_claims(claims::AbstractVector{CorpusClaim};
                            jaccard_threshold::Real=0.65)
    n = length(claims)
    parent = collect(1:n)
    function find(i)
        while parent[i] != i
            parent[i] = parent[parent[i]]
            i = parent[i]
        end
        i
    end
    function unite(i, j)
        ri, rj = find(i), find(j)
        ri == rj || (parent[ri] = rj)
    end

    subj_tok = [_entity_tokens(c.subj) for c in claims]
    obj_tok  = [_entity_tokens(c.obj)  for c in claims]
    for i in 1:n, j in (i+1):n
        if _jaccard(subj_tok[i], subj_tok[j]) >= jaccard_threshold &&
           _jaccard(obj_tok[i],  obj_tok[j])  >= jaccard_threshold
            unite(i, j)
        end
    end

    groups = Dict{Int, Vector{Int}}()
    for i in 1:n
        push!(get!(groups, find(i), Int[]), i)
    end

    out = SynthesizedClaim[]
    total_docs = length(unique(c.document for c in claims))
    for idxs in values(groups)
        variants = CorpusClaim[claims[i] for i in idxs]
        docs = sort(unique(v.document for v in variants))
        support = length(docs)
        # representative = the variant appearing in the most documents (modal subj/obj)
        rep = variants[argmax([count(v -> v.subj == w.subj && v.obj == w.obj, variants) for w in variants])]
        pols = unique(v.polarity for v in variants if v.polarity != 0)
        conflicted = length(pols) > 1
        polarity = isempty(pols) ? 0 : (conflicted ? 0 : pols[1])
        push!(out, SynthesizedClaim(
            rep.subj, rep.rel, rep.obj, polarity, docs, support, variants,
            corpus_truth_value(support, total_docs), conflicted, 0.0, :low,
            Dict{Symbol, Any}(:total_documents => total_docs)))
    end
    out
end

# ---------------------------------------------------------------------------
# Simplicial coherence (horn-fill)
# ---------------------------------------------------------------------------

"""
    CoherenceMetrics

Simplicial structure of the corpus claim complex: vertices (entities), edges
(claims), filled triangles (a→b, b→c, a→c all present), open 2-horns (a→b,
b→c with a→c missing), the horn-fill ratio, and connected components. `state`
is `:coherent`, `:partially_glued`, or `:disconnected`.
"""
struct CoherenceMetrics
    vertices::Int
    edges::Int
    triangles::Int
    open_horns::Int
    horn_fill_ratio::Float64
    components::Int
    state::Symbol
end

"""
    homotopy_coherence(claims) -> CoherenceMetrics

Compute the simplicial coherence of a set of (synthesized or local) claims.
"""
function homotopy_coherence(claims)
    edges = Set{Tuple{Symbol,Symbol}}()
    verts = Set{Symbol}()
    for c in claims
        push!(edges, (c.canonical_subj, c.canonical_obj))
        push!(verts, c.canonical_subj); push!(verts, c.canonical_obj)
    end
    has(a, b) = (a, b) in edges
    succ = Dict{Symbol, Vector{Symbol}}()
    for (a, b) in edges
        push!(get!(succ, a, Symbol[]), b)
    end

    triangles = 0; open_horns = 0
    for (a, b) in edges
        for c in get(succ, b, Symbol[])
            c == a && continue
            if has(a, c)
                triangles += 1
            else
                open_horns += 1
            end
        end
    end

    # connected components over the undirected edge graph
    vlist = collect(verts)
    idx = Dict(v => i for (i, v) in enumerate(vlist))
    parent = collect(1:length(vlist))
    function find(i)
        while parent[i] != i
            parent[i] = parent[parent[i]]
            i = parent[i]
        end
        i
    end
    for (a, b) in edges
        ra, rb = find(idx[a]), find(idx[b])
        ra == rb || (parent[ra] = rb)
    end
    components = isempty(vlist) ? 0 : length(unique(find(i) for i in 1:length(vlist)))

    denom = triangles + open_horns
    ratio = denom == 0 ? 1.0 : triangles / denom
    state = length(edges) == 0 ? :disconnected :
            ratio >= 0.66 ? :coherent :
            ratio > 0.0 ? :partially_glued :
            components <= 1 ? :coherent : :disconnected
    CoherenceMetrics(length(verts), length(edges), triangles, open_horns,
                     ratio, components, state)
end

# ---------------------------------------------------------------------------
# Query alignment
# ---------------------------------------------------------------------------

"""
    query_alignment(claim, query) -> (score::Float64, label::Symbol)

Relevance of a synthesized claim to a free-text query, by token overlap of the
query with the claim's subject/object, plus a small corroboration bonus for
multi-document support. `label` is `:high`/`:moderate`/`:weak`/`:low`.
"""
function query_alignment(claim::SynthesizedClaim, query::AbstractString)
    qtok = Set(t for t in split(lowercase(query)) if length(t) >= 3)
    isempty(qtok) && return (0.0, :low)
    ctok = union(_entity_tokens(claim.canonical_subj), _entity_tokens(claim.canonical_obj))
    base = length(intersect(qtok, ctok)) / length(qtok)
    corrob = min(0.15, 0.05 * max(0, claim.support - 1))
    score = min(1.0, base + corrob)
    label = score >= 0.6 ? :high : score >= 0.35 ? :moderate : score > 0.0 ? :weak : :low
    (score, label)
end

# ---------------------------------------------------------------------------
# Synthesis result + driver
# ---------------------------------------------------------------------------

"""
    CorpusSynthesisResult

Output of [`synthesize_corpus`](@ref): ranked claims, polarity disagreements,
the coherence metrics of the glued complex, and document counts.
"""
struct CorpusSynthesisResult
    query::String
    claims::Vector{SynthesizedClaim}          # ranked, most relevant/supported first
    disagreements::Vector{SynthesizedClaim}   # polarity-conflicted
    coherence::CoherenceMetrics
    n_documents::Int
    metadata::Dict{Symbol, Any}
end

"""
    synthesize_corpus(claims; query="", jaccard_threshold=0.65, limit=0) -> CorpusSynthesisResult

Run the full corpus-synthesis pipeline over per-document `CorpusClaim`s. Glues
variants, assigns truth tiers, scores coherence and query relevance, and ranks
the result (by relevance, then support, then document count). `limit>0` caps the
number of returned claims. Polarity-conflicted claims are surfaced separately.
"""
function synthesize_corpus(claims::AbstractVector{CorpusClaim};
                           query::AbstractString="",
                           jaccard_threshold::Real=0.65,
                           limit::Integer=0)
    glued = glue_corpus_claims(claims; jaccard_threshold=jaccard_threshold)

    # attach query relevance
    scored = SynthesizedClaim[]
    for c in glued
        s, lbl = query_alignment(c, query)
        push!(scored, SynthesizedClaim(c.canonical_subj, c.canonical_rel, c.canonical_obj,
            c.polarity, c.documents, c.support, c.variants, c.truth_value, c.conflicted,
            s, lbl, c.metadata))
    end

    # rank: relevance desc, support desc, then alphabetical for stability
    sort!(scored; by = c -> (-c.relevance, -c.support, String(c.canonical_subj), String(c.canonical_obj)))
    ranked = limit > 0 ? scored[1:min(limit, length(scored))] : scored

    disagreements = [c for c in scored if c.conflicted]
    coherence = homotopy_coherence(scored)
    n_docs = length(unique(c.document for c in claims))

    CorpusSynthesisResult(String(query), ranked, disagreements, coherence, n_docs,
        Dict{Symbol, Any}(:total_claims => length(claims),
                          :glued_claims => length(glued),
                          :jaccard_threshold => Float64(jaccard_threshold)))
end

# ---------------------------------------------------------------------------
# Serialization / summary
# ---------------------------------------------------------------------------

function as_dict(c::SynthesizedClaim)
    Dict(
        "subj" => String(c.canonical_subj),
        "rel" => String(c.canonical_rel),
        "obj" => String(c.canonical_obj),
        "polarity" => c.polarity,
        "support" => c.support,
        "documents" => copy(c.documents),
        "truth_value" => String(c.truth_value),
        "conflicted" => c.conflicted,
        "relevance" => c.relevance,
        "relevance_label" => String(c.relevance_label),
        "n_variants" => length(c.variants),
    )
end

as_dict(m::CoherenceMetrics) = Dict(
    "vertices" => m.vertices, "edges" => m.edges, "triangles" => m.triangles,
    "open_horns" => m.open_horns, "horn_fill_ratio" => m.horn_fill_ratio,
    "components" => m.components, "state" => String(m.state),
)

"""
    summarize_corpus_synthesis(result) -> Dict

JSON-friendly summary of a [`CorpusSynthesisResult`](@ref).
"""
function summarize_corpus_synthesis(result::CorpusSynthesisResult)
    by_tier = Dict{String, Int}()
    for c in result.claims
        by_tier[String(c.truth_value)] = get(by_tier, String(c.truth_value), 0) + 1
    end
    Dict(
        "query" => result.query,
        "n_documents" => result.n_documents,
        "counts" => Dict(
            "claims" => length(result.claims),
            "disagreements" => length(result.disagreements),
            "by_truth_value" => by_tier,
        ),
        "coherence" => as_dict(result.coherence),
        "claims" => [as_dict(c) for c in result.claims],
        "disagreements" => [as_dict(c) for c in result.disagreements],
    )
end

as_dict(result::CorpusSynthesisResult) = summarize_corpus_synthesis(result)
to_json(result::CorpusSynthesisResult) = JSON3.write(summarize_corpus_synthesis(result))

function Base.show(io::IO, ::MIME"text/plain", r::CorpusSynthesisResult)
    print(io, "CorpusSynthesisResult(", length(r.claims), " claims from ", r.n_documents,
          " docs; coherence=", r.coherence.state,
          " fill=", round(r.coherence.horn_fill_ratio; digits=2),
          ", ", length(r.disagreements), " disagreement(s))")
end

# ---------------------------------------------------------------------------
# Worked example (parity reference / tests)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Categorical re-founding: claim gluing is a colimit in FinSet
#
# The "merge variant claims across documents" step is, categorically, the
# colimit of the diagram  R ⇉ A  (the variant-pair relation, with its two
# projections, over the set of all claims). Its apex is `A` quotiented by the
# generated congruence — exactly the equivalence classes `glue_corpus_claims`
# computes. This is the Democritus sheaf-gluing of local claim-sections made
# literal via `FunctorFlow.Cat`, not an analogy.
# ---------------------------------------------------------------------------

"""
    corpus_gluing_diagram(claims; jaccard_threshold=0.65) -> Cat.SetFunctor

Build the gluing diagram `R ⇉ A` as a `SetFunctor` over the parallel-pair
category: `A` is the set of all claims (by index) and `R` the set of
variant-pairs that should be identified, with the two projection edges. The
colimit of this diagram is the glued corpus.
"""
function corpus_gluing_diagram(claims::AbstractVector{CorpusClaim}; jaccard_threshold::Real=0.65)
    n = length(claims)
    subj = [_entity_tokens(c.subj) for c in claims]
    obj = [_entity_tokens(c.obj) for c in claims]
    pairs = Tuple{Int,Int}[]
    for i in 1:n, j in (i+1):n
        if _jaccard(subj[i], subj[j]) >= jaccard_threshold && _jaccard(obj[i], obj[j]) >= jaccard_threshold
            push!(pairs, (i, j))
        end
    end
    shape = Cat.FreeCat([:R, :A], [(:p, :R, :A), (:q, :R, :A)])
    A = Cat.FinSet(collect(1:n))
    R = Cat.FinSet(collect(1:length(pairs)))
    p = Cat.FinFunction(R, A, Dict{Any,Any}(k => pairs[k][1] for k in 1:length(pairs)))
    q = Cat.FinFunction(R, A, Dict{Any,Any}(k => pairs[k][2] for k in 1:length(pairs)))
    Cat.SetFunctor(shape; ob_map=Dict(:R => R, :A => A), edge_map=Dict(:p => p, :q => q))
end

"""
    corpus_colimit(claims; jaccard_threshold=0.65) -> Cat.ColimitCocone

The colimit of [`corpus_gluing_diagram`](@ref): its apex is the set of glued
corpus claims (equivalence classes of variants). The number of classes equals
`length(glue_corpus_claims(claims))` — the engine and the categorical colimit
agree.
"""
corpus_colimit(claims::AbstractVector{CorpusClaim}; jaccard_threshold::Real=0.65) =
    Cat.colimit(corpus_gluing_diagram(claims; jaccard_threshold=jaccard_threshold))

"""
    build_corpus_synthesis_example() -> NamedTuple

A small three-document corpus on the minimum-wage / employment literature with
wording variants, multi-document corroboration, a transitive chain (for
horn-fill), and one polarity disagreement. Returns `(claims=, query=)`.
"""
function build_corpus_synthesis_example()
    claims = CorpusClaim[
        # corroborated across all three docs (entailed); differing surface
        # statements glue onto one backbone claim (same extracted entities)
        CorpusClaim("minimum_wage", "raises", "earnings"; document="doc1", domain="labor",
                    statement="raising the minimum wage lifts worker earnings"),
        CorpusClaim("minimum wage", "increases", "earnings"; document="doc2", domain="labor",
                    statement="a higher minimum wage increases earnings"),
        CorpusClaim("minimum_wage", "boosts", "earnings"; document="doc3", domain="labor",
                    statement="minimum wage hikes boost take-home earnings"),
        # transitive chain earnings → demand → employment (fills a 2-horn)
        CorpusClaim("earnings", "increases", "demand"; document="doc1", domain="labor"),
        CorpusClaim("demand", "increases", "employment"; document="doc2", domain="labor"),
        CorpusClaim("earnings", "increases", "employment"; document="doc3", domain="labor"),
        # a polarity disagreement on minimum_wage → employment
        CorpusClaim("minimum_wage", "reduces", "employment"; document="doc1", domain="labor"),
        CorpusClaim("minimum_wage", "raises", "employment"; document="doc2", domain="labor"),
    ]
    (claims = claims, query = "does the minimum wage affect employment and earnings")
end
