# ============================================================================
# cat_integration.jl — the horizontal, end-to-end layer
#
# Ties the remaining applied surfaces to the categorical kernel and exposes a
# single pipeline that flows a query through every layer:
#
#   query  ──CLIFF route──▶  textbook chapters  ──grounds──▶  runnable demos
#     │            │                  │                            │
#     │     (route↦chapter↦demo is a genuine category: cliff_knowledge_category)
#     ▼
#   if causal/evidence:  causal model ─as a category→ intervention ─functor→
#                        twin network ─pushout→ identifiability + counterfactual
#
# Plus `jepa_square_category`: JEPA exactness IS the commutativity of a square
# (a presented category), so "obstruction loss = 0" is a categorical statement.
# ============================================================================

# ----------------------------------------------------------------------------
# CLIFF route↦chapter↦demo linkage as a category
# ----------------------------------------------------------------------------

_route_obj(r) = Symbol("route_$(r)")
_chapter_obj(n) = Symbol("ch$(n)")
_demo_obj(d) = Symbol("demo_$(d)")

"""
    cliff_knowledge_category(; router=build_cliff_query_router(), routes=…) -> Cat.FreeCat

The CLIFF knowledge structure as a genuine category: objects are routes,
*Categories for AGI* chapters, and runnable demos; generating morphisms are
`route —backs→ chapter` and `chapter —grounds→ demo`. Composition `route → demo`
recovers exactly the runnable demos backing a route — the textbook-grounding
linkage *is* a category (a 3-layer DAG, so the kernel accepts it and it is
Lean-certifiable via `render_cat_certificate`).
"""
function cliff_knowledge_category(; router=build_cliff_query_router(),
                                  routes=collect(keys(router.routes)))
    routeset = Set(Symbol.(routes))
    chapters = [ch for ch in textbook_chapters() if any(r in routeset for r in ch.routes)]
    objs = Symbol[]
    for r in routes; push!(objs, _route_obj(r)); end
    for ch in chapters; push!(objs, _chapter_obj(ch.number)); end
    demos = Symbol[]
    seen = Set{Symbol}()
    for ch in chapters, d in runnable_demos(ch)
        d in seen || (push!(demos, d); push!(seen, d))
    end
    for d in demos; push!(objs, _demo_obj(d)); end

    edges = Tuple{Symbol,Symbol,Symbol}[]
    for ch in chapters, r in ch.routes
        r in routeset || continue
        push!(edges, (Symbol("backs_$(r)_$(ch.number)"), _route_obj(r), _chapter_obj(ch.number)))
    end
    for ch in chapters, d in runnable_demos(ch)
        push!(edges, (Symbol("grounds_$(ch.number)_$(d)"), _chapter_obj(ch.number), _demo_obj(d)))
    end
    Cat.FreeCat(objs, edges)
end

"""
    demos_reachable_from(K, route) -> Vector{Symbol}

The demos reachable from a route object in the knowledge category `K` — i.e.
those `d` with a non-empty `Hom(route, demo_d)` (a `route→chapter→demo` path).
This is the categorical recovery of "which demos back this route".
"""
function demos_reachable_from(K::Cat.FreeCat, route)
    robj = route isa Symbol && startswith(string(route), "route_") ? route : _route_obj(route)
    out = Symbol[]
    for o in Cat.objects(K)
        startswith(string(o), "demo_") || continue
        isempty(Cat.homset(K, robj, o)) || push!(out, Symbol(replace(string(o), "demo_" => "")))
    end
    sort(out)
end

# ----------------------------------------------------------------------------
# JEPA exactness as a commuting square
# ----------------------------------------------------------------------------

"""
    jepa_square_category() -> Cat.FinPresentedCat

The JEPA square as a presented category: `X —enc_x→ Z —pred→ Zt` and
`X —γ→ Xp —enc_y→ Zt`, with the relation `pred∘enc_x = enc_y∘γ`. JEPA
*exactness* (the prediction obstruction loss being zero) is exactly this
square commuting — so `Hom(X, Zt)` collapses to a single morphism.
"""
function jepa_square_category()
    objs = [:X, :Z, :Xp, :Zt]
    edges = [(:enc_x, :X, :Z), (:pred, :Z, :Zt), (:gamma, :X, :Xp), (:enc_y, :Xp, :Zt)]
    p1 = Cat.PathMor(:X, :Zt, [:enc_x, :pred])
    p2 = Cat.PathMor(:X, :Zt, [:gamma, :enc_y])
    Cat.FinPresentedCat(objs, edges, [(p1, p2)])
end

# ----------------------------------------------------------------------------
# The end-to-end pipeline
# ----------------------------------------------------------------------------

# routes that build causal/evidence state (so the causal capstone is relevant)
const _CAUSAL_ROUTES = Set([:democritus, :company_similarity, :product_feedback, :basket_rocket_sec])

"""
    integrated_pipeline(query; router=build_cliff_query_router()) -> Dict

Flow `query` through the whole stack and return a unified report:
CLIFF routing + textbook backing, the knowledge category's categorical recovery
of the route's demos, and — for evidence/causal routes — the causal capstone
(model-as-category → intervention functor → twin-network pushout →
identifiability → counterfactual). One query, every layer, one report.
"""
function integrated_pipeline(query; router=build_cliff_query_router())
    rt = route_with_textbook(router, query)
    K = cliff_knowledge_category(; router=router)
    via_cat = demos_reachable_from(K, rt.decision.route_name)

    causal = rt.decision.route_name in _CAUSAL_ROUTES ? causal_capstone() : nothing

    Dict(
        "query" => String(query),
        "route" => String(rt.decision.route_name),
        "textbook_chapters" => [c.number for c in rt.route_chapters],
        "demos_from_textbook" => sort(String.(rt.demos)),
        "demos_via_category" => String.(via_cat),
        "knowledge_category" => Dict(
            "objects" => length(Cat.objects(K)),
            "morphism_kinds" => ["route→chapter (backs)", "chapter→demo (grounds)"]),
        "causal_capstone" => causal,
        "layers_exercised" => causal === nothing ?
            ["cliff_routing", "textbook", "knowledge_category"] :
            ["cliff_routing", "textbook", "knowledge_category",
             "causal_category", "intervention_functor", "twin_network_pushout",
             "identifiability", "counterfactual"],
    )
end

"""
    end_to_end_capstone() -> Dict

Run the integrated pipeline on a canonical causal query and assemble a single
report that touches every layer of FunctorFlow — from CLIFF routing down to the
Lean-certified categorical kernel.
"""
function end_to_end_capstone()
    pipeline = integrated_pipeline("Analyze recent studies on minimum wage and employment")
    # corpus-synthesis-as-colimit on a small evidence corpus, for completeness
    ex = build_corpus_synthesis_example()
    col = corpus_colimit(ex.claims)
    pipeline["corpus_synthesis"] = Dict(
        "glued_claims" => length(col.apex),
        "is_colimit" => Cat.verify_colimit(col),
        "agrees_with_engine" => length(col.apex) == length(glue_corpus_claims(ex.claims)))
    pipeline["jepa_exactness_is_commutativity"] =
        Cat.hom_cardinality(jepa_square_category(), :X, :Zt) == 1
    pipeline
end
