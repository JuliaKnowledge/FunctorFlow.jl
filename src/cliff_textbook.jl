# ============================================================================
# cliff_textbook.jl — Textbook-grounded routing for the CLIFF layer
#
# CLIFF ("Conscious Layer Interface to FunctorFlow", Mahadevan's
# CLIFF_CatAgi) is a *textbook-centric* AGI assistant: every route links back
# to a chapter of *Categories for AGI*, each chapter connects to runnable
# FunctorFlow demos (block macros), and queries can be answered with a
# textbook backstop even when no external engine is available.
#
# This module ports that capability to FunctorFlow.jl. It provides:
#   - `TextbookChapter` + the `CATAGI_TEXTBOOK` registry (real chapter titles)
#   - `recommend_chapters(query)` — rank chapters by thematic overlap
#   - `chapters_for_route` / `chapters_for_primitive` — the route/demo linkage
#   - `route_with_textbook(router, query)` — route a query *and* surface the
#     chapters + runnable demos that back the chosen route
#
# Chapter titles are taken from the companion Lean formalisation `catagi`.
# ============================================================================

"""
    TextbookChapter

A chapter of Sridhar Mahadevan's *Categories for AGI*, annotated with the
thematic keywords used for retrieval, the FunctorFlow block macros
("runnable demos") it grounds, and the CLIFF routes it backs.
"""
struct TextbookChapter
    number::Int
    title::String
    summary::String
    themes::Vector{String}
    primitives::Vector{Symbol}   # FunctorFlow block macros (keys of MACRO_LIBRARY)
    routes::Vector{Symbol}       # CLIFF route names this chapter backs
end

function TextbookChapter(number, title;
                         summary="",
                         themes::AbstractVector=String[],
                         primitives::AbstractVector=Symbol[],
                         routes::AbstractVector=Symbol[])
    TextbookChapter(
        Int(number),
        String(title),
        String(summary),
        String[lowercase(String(t)) for t in themes],
        Symbol[Symbol(p) for p in primitives],
        Symbol[Symbol(r) for r in routes],
    )
end

"""
    CATAGI_TEXTBOOK :: OrderedDict{Int, TextbookChapter}

The *Categories for AGI* chapter registry that CLIFF routes link back to.
Chapter titles match the companion `catagi` Lean formalisation.
"""
const CATAGI_TEXTBOOK = OrderedDict{Int, TextbookChapter}()

function _register_chapter!(chapter::TextbookChapter)
    CATAGI_TEXTBOOK[chapter.number] = chapter
    chapter
end

for _ch in (
    TextbookChapter(1, "Category Theory for AGI";
        summary="Objects, morphisms, composition, and identity as the substrate for AGI systems.",
        themes=["category", "object", "morphism", "composition", "identity", "foundations", "what is category theory"],
        routes=[:course_demo]),
    TextbookChapter(2, "Functors for AGI";
        summary="Functors as structure-preserving maps and the basis of functorial data migration.",
        themes=["functor", "structure preserving", "data migration", "presheaf", "mapping"],
        routes=[:course_demo, :democritus]),
    TextbookChapter(3, "Representable Functors and the Yoneda Lemma";
        summary="Representability and the Yoneda embedding: objects are known by their relationships.",
        themes=["yoneda", "representable", "embedding", "natural transformation", "probe"],
        routes=[:course_demo]),
    TextbookChapter(4, "Diagrams and Universal Constructions";
        summary="Limits and colimits — pullbacks, pushouts, products, coproducts, equalizers — as universal glue.",
        themes=["diagram", "universal", "pullback", "pushout", "product", "coproduct",
                "limit", "colimit", "equalizer", "compare", "similarity", "merge", "join"],
        primitives=Symbol[],
        routes=[:course_demo, :company_similarity]),
    TextbookChapter(5, "Categorical Deep Learning";
        summary="Neural architectures as diagrams: attention and message passing as Kan-extension aggregation.",
        themes=["neural", "deep learning", "attention", "transformer", "kan extension",
                "pooling", "message passing", "embedding", "representation", "jepa", "energy"],
        primitives=[:ket, :gt_neighborhood, :jepa, :kan_jepa, :energy, :db_square],
        routes=[:course_demo, :product_feedback]),
    TextbookChapter(7, "Geometric Transformers";
        summary="Message passing over simplicial/graph structure and horn-filling coherence.",
        themes=["graph", "geometric", "simplicial", "neighborhood", "message passing",
                "horn", "transformer", "diffusion", "gluing"],
        primitives=[:gt_neighborhood, :horn_fill, :higher_horn, :topocoend],
        routes=[:course_demo, :company_similarity]),
    TextbookChapter(8, "Dynamic Compositionality";
        summary="Composing plan fragments (BASKET) and repairing them (ROCKET) as agentic workflows.",
        themes=["composition", "workflow", "plan", "planning", "fragment", "basket",
                "rocket", "repair", "itinerary", "tour", "filing", "sec", "pipeline"],
        primitives=[:basket_workflow, :rocket_repair, :basket_rocket_pipeline],
        routes=[:basket_rocket_sec, :culinary_tour]),
    TextbookChapter(11, "Adjoint Functors";
        summary="Left/right adjoints and Kan extensions: free/forgetful, aggregation/completion duality.",
        themes=["adjoint", "adjunction", "kan extension", "left kan", "right kan",
                "free", "forgetful", "completion", "duality"],
        primitives=[:ket, :completion, :kan_jepa],
        routes=[:course_demo, :democritus]),
    TextbookChapter(13, "Topos Causal Models";
        summary="Sheaves, the subobject classifier, and topos logic for local-to-global causal gluing.",
        themes=["topos", "sheaf", "subobject classifier", "internal logic", "gluing",
                "local to global", "democritus", "causal", "intervention", "do-calculus"],
        primitives=[:democritus_gluing, :democritus_assembly, :topocoend],
        routes=[:democritus, :company_similarity]),
    TextbookChapter(14, "Judo Calculus";
        summary="A categorical do-calculus: interventions, identifiability, and back/front-door reasoning.",
        themes=["do-calculus", "intervention", "identifiability", "backdoor", "frontdoor",
                "causal effect", "judo", "hedge", "confounder"],
        primitives=Symbol[],
        routes=[:democritus]),
    TextbookChapter(15, "Causal Density Functions";
        summary="Structural causal models and causal density/distribution functors.",
        themes=["causal density", "distribution", "scm", "structural causal model",
                "counterfactual", "probability", "study", "evidence"],
        primitives=Symbol[],
        routes=[:democritus, :product_feedback]),
    TextbookChapter(16, "Consciousness";
        summary="A global-workspace functor: salience-ranked broadcast and conscious access for orchestration.",
        themes=["consciousness", "conscious", "workspace", "global workspace", "broadcast",
                "attention", "salience", "routing", "orchestration", "field of view"],
        primitives=Symbol[],
        routes=[:company_similarity, :democritus, :basket_rocket_sec,
                :culinary_tour, :product_feedback, :course_demo]),
)
    _register_chapter!(_ch)
end

"""
    textbook_chapter(n::Integer) -> TextbookChapter

Look up a *Categories for AGI* chapter by number.
"""
function textbook_chapter(n::Integer)
    haskey(CATAGI_TEXTBOOK, Int(n)) || throw(ArgumentError("No registered textbook chapter $(n)"))
    CATAGI_TEXTBOOK[Int(n)]
end

textbook_chapters() = collect(values(CATAGI_TEXTBOOK))

"""
    chapters_for_route(route) -> Vector{TextbookChapter}

All textbook chapters that back a CLIFF route (accepts a `Symbol`, a string,
or a `CLIFFRouteDecision`). Returned in chapter-number order.
"""
function chapters_for_route(route::Symbol)
    [c for c in values(CATAGI_TEXTBOOK) if route in c.routes]
end
chapters_for_route(route::AbstractString) = chapters_for_route(Symbol(route))
chapters_for_route(decision::CLIFFRouteDecision) = chapters_for_route(decision.route_name)

"""
    chapters_for_primitive(macro_name) -> Vector{TextbookChapter}

All textbook chapters whose runnable demos include the given FunctorFlow
block macro (e.g. `:ket`, `:democritus_gluing`).
"""
function chapters_for_primitive(macro_name::Symbol)
    [c for c in values(CATAGI_TEXTBOOK) if macro_name in c.primitives]
end
chapters_for_primitive(macro_name::AbstractString) = chapters_for_primitive(Symbol(macro_name))

"""
    runnable_demos(chapter::TextbookChapter) -> Vector{Symbol}

The FunctorFlow block macros that ground a chapter and are actually present
in `MACRO_LIBRARY` (i.e. directly runnable via `build_macro`).
"""
runnable_demos(chapter::TextbookChapter) =
    Symbol[p for p in chapter.primitives if haskey(MACRO_LIBRARY, p)]

# ----------------------------------------------------------------------------
# Retrieval: rank chapters by thematic overlap with a query
# ----------------------------------------------------------------------------

function _chapter_score(chapter::TextbookChapter, normalized_query::AbstractString)
    score = 0
    for theme in chapter.themes
        occursin(theme, normalized_query) && (score += length(split(theme)) >= 2 ? 3 : 2)
    end
    # Title words are a weaker signal than curated themes.
    for word in split(lowercase(chapter.title))
        length(word) >= 4 && occursin(word, normalized_query) && (score += 1)
    end
    score
end

"""
    recommend_chapters(query; limit=3, include_zero=false) -> Vector{TextbookChapter}

Rank *Categories for AGI* chapters by thematic overlap with `query`. Returns
up to `limit` chapters, highest score first (ties broken by chapter number).
Chapters with no overlap are dropped unless `include_zero=true`.
"""
function recommend_chapters(query; limit::Integer=3, include_zero::Bool=false)
    normalized = _normalize_cliff_query(query)
    scored = [(c, _chapter_score(c, normalized)) for c in values(CATAGI_TEXTBOOK)]
    include_zero || (scored = [(c, s) for (c, s) in scored if s > 0])
    sort!(scored; by=((cs),) -> (-cs[2], cs[1].number))
    [c for (c, _) in scored[1:min(limit, length(scored))]]
end

"""
    route_with_textbook(router, query; limit=3, kwargs...) -> NamedTuple

Route a CLIFF query and return, alongside the routing `decision`, the
textbook chapters that back the chosen route and the chapters most relevant
to the query text, plus the runnable demos for the union. This reproduces
CLIFF_CatAgi's "every route links back to the textbook" behaviour.

Fields: `decision`, `route_chapters`, `query_chapters`, `demos`.
"""
function route_with_textbook(router::CLIFFQueryRouter, query;
                             limit::Integer=3, route_override=:auto, execution_mode=:quick)
    decision = route_cliff_query(router, query; route_override=route_override, execution_mode=execution_mode)
    route_chapters = chapters_for_route(decision)
    query_chapters = recommend_chapters(query; limit=limit)
    demos = unique(reduce(vcat, [runnable_demos(c) for c in vcat(route_chapters, query_chapters)]; init=Symbol[]))
    (decision=decision, route_chapters=route_chapters, query_chapters=query_chapters, demos=demos)
end

route_with_textbook(query; kwargs...) = route_with_textbook(build_cliff_query_router(), query; kwargs...)

# ----------------------------------------------------------------------------
# Serialization
# ----------------------------------------------------------------------------

function as_dict(chapter::TextbookChapter)
    Dict(
        "number" => chapter.number,
        "title" => chapter.title,
        "summary" => chapter.summary,
        "themes" => copy(chapter.themes),
        "primitives" => String.(chapter.primitives),
        "routes" => String.(chapter.routes),
        "runnable_demos" => String.(runnable_demos(chapter)),
    )
end

to_json(chapter::TextbookChapter) = JSON3.write(as_dict(chapter))

function Base.show(io::IO, chapter::TextbookChapter)
    print(io, "TextbookChapter($(chapter.number), \"$(chapter.title)\"; ",
          "$(length(chapter.primitives)) demos, $(length(chapter.routes)) routes)")
end
