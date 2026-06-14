# ============================================================================
# cat_causal.jl — the causal / counterfactual layer, re-founded on the kernel
#
# This is the capstone tying the whole stack together:
#   • a causal DAG *is* a category               (causal_category → Cat.FreeCat)
#   • an intervention *is* a functor             (intervention_functor: the mutilation)
#   • a counterfactual twin network *is* a pushout (twin_network, computed and
#     verified by the kernel's FinSet pushout — the parallel-worlds graph is the
#     amalgam of the factual and interventional worlds over the shared background)
#   • identifiability / counterfactual direction come from `identify_effect` /
#     `counterfactual_effect`
#   • and the resulting causal category is Lean-certifiable via
#     `render_cat_certificate`.
# ============================================================================

_causal_edge_name(u, v) = Symbol(string(u, "__", v))

"""
    causal_category(G::CausalDAG) -> Cat.FreeCat

The free category of a causal DAG: variables are objects and directed edges are
generators (a directed path `u → ⋯ → w` is a morphism). Acyclic by definition,
so the kernel accepts it — and `Cat.check_category_laws` / `render_cat_certificate`
apply directly.
"""
function causal_category(G::CausalDAG)
    edges = Tuple{Symbol,Symbol,Symbol}[(_causal_edge_name(u, v), u, v) for (u, v) in G.directed]
    Cat.FreeCat(G.nodes, edges)
end

"""
    intervention_functor(G::CausalDAG, x) -> NamedTuple

The do-intervention `do(x)` as a functor. The mutilated DAG `G_x̄` (incoming
edges to `x` removed) is a subcategory, and there is an inclusion functor
`G_x̄ → G`. Returns `(functor, mutilated, full)`.
"""
function intervention_functor(G::CausalDAG, x)
    xs = Symbol[Symbol(v) for v in (x isa Union{Symbol,AbstractString} ? [x] : x)]
    Gx = remove_incoming(G, xs)
    Cmut = causal_category(Gx)
    Cfull = causal_category(G)
    F = Cat.FinFunctor(Cmut, Cfull;
        ob_map=Dict(o => o for o in Gx.nodes),
        edge_map=Dict(n => Cat.PathMor(s, t, Symbol[n]) for (n, s, t) in Cmut.edges))
    (functor=F, mutilated=Cmut, full=Cfull)
end

"""
    twin_network(G::CausalDAG, x) -> NamedTuple

The counterfactual **twin network** as a pushout. The factual world and the
interventional world both contain the descendants of `x`; they *share* the
background (the non-descendants of `x`). The node set of the twin network is
therefore the pushout `World_factual ⊔_{background} World_cf`, computed and
verified by the kernel. Returns `(pushout, shared, descendants, factual, counterfactual)`.
"""
function twin_network(G::CausalDAG, x)
    xs = Symbol[Symbol(v) for v in (x isa Union{Symbol,AbstractString} ? [x] : x)]
    desc = _descendants_inclusive(G, xs)
    shared = Symbol[v for v in G.nodes if !(v in desc)]
    descv = Symbol[v for v in G.nodes if v in desc]
    Wf = Cat.FinSet(vcat(Any[s for s in shared], Any[(:f, d) for d in descv]))
    Wc = Cat.FinSet(vcat(Any[s for s in shared], Any[(:c, d) for d in descv]))
    S = Cat.FinSet(Any[s for s in shared])
    legf = Cat.FinFunction(S, Wf, Dict{Any,Any}(s => s for s in shared))
    legc = Cat.FinFunction(S, Wc, Dict{Any,Any}(s => s for s in shared))
    po = Cat.pushout(legf, legc)
    (pushout=po, shared=shared, descendants=descv, factual=Wf, counterfactual=Wc)
end

"""
    twin_causal_diagram(G::CausalDAG, x; name=:TwinNetwork) -> Diagram

A FunctorFlow `Diagram` of the twin network (factual world = full `G`,
counterfactual world = `G` mutilated at `x`, sharing the background), suitable
for `plot_diagram`. Background nodes are shared; descendants are duplicated with
`f_`/`c_` prefixes.
"""
function twin_causal_diagram(G::CausalDAG, x; name::Union{Symbol,AbstractString}=:TwinNetwork)
    xs = Symbol[Symbol(v) for v in (x isa Union{Symbol,AbstractString} ? [x] : x)]
    desc = _descendants_inclusive(G, xs)
    twin_name(v, world) = v in desc ? Symbol(string(world, "_", v)) : v
    D = Diagram(Symbol(name))
    for v in G.nodes
        v in desc || add_object!(D, v; kind=:background)
    end
    for v in G.nodes
        if v in desc
            add_object!(D, twin_name(v, :f); kind=:factual)
            add_object!(D, twin_name(v, :c); kind=:counterfactual)
        end
    end
    # factual world = full G
    for (u, v) in G.directed
        add_morphism!(D, Symbol(string("f_", u, "_", v)), twin_name(u, :f), twin_name(v, :f))
    end
    # counterfactual world = mutilated G (do(x))
    for (u, v) in remove_incoming(G, xs).directed
        su, tv = twin_name(u, :c), twin_name(v, :c)
        # avoid duplicating shared→shared edges already added in the factual pass
        (u in desc || v in desc) && add_morphism!(D, Symbol(string("c_", u, "_", v)), su, tv)
    end
    D
end

# ----------------------------------------------------------------------------
# Causal DAG as a morphism in the Markov category
# ----------------------------------------------------------------------------

"""
    causal_markov_kernel(G::CausalDAG, mechanisms) -> NamedTuple

Factorise a causal DAG's joint distribution as a composite in the Markov
category: `P(V) = ∏_v P(v | pa(v))`. `mechanisms[node]` maps the tuple of parent
values (in `parents(G, node)` order) to a `Cat.Dist` over that node's values.
Returns `(dist, order)` — the joint as a `Cat.Dist` over value tuples in sorted
node order. This realises the SCM as a genuine Markov-category morphism.
"""
function causal_markov_kernel(G::CausalDAG, mechanisms)
    order = topological_order(G)
    states = Tuple{Dict{Symbol, Any}, Rational{Int}}[(Dict{Symbol, Any}(), 1 // 1)]
    for node in order
        pars = parents(G, node)
        newstates = Tuple{Dict{Symbol, Any}, Rational{Int}}[]
        for (assign, p) in states
            parvals = Tuple(assign[pn] for pn in pars)
            d = mechanisms[node](parvals)
            for (v, pv) in d.support
                na = copy(assign); na[node] = v
                push!(newstates, (na, p * pv))
            end
        end
        states = newstates
    end
    keyorder = sort(order)
    acc = Dict{Any, Rational{Int}}()
    for (assign, p) in states
        key = Tuple(assign[n] for n in keyorder)
        acc[key] = get(acc, key, 0 // 1) + p
    end
    (dist=Cat.Dist(acc), order=keyorder)
end

# ----------------------------------------------------------------------------
# The grand finale: every layer, one example
# ----------------------------------------------------------------------------

"""
    build_causal_capstone_example() -> NamedTuple

A confounded mediation model `Z→X, Z→Y, X→M→Y` (the back-door-adjustable case),
with the matching causal triples. `(dag, triples, treatment, outcome)`.
"""
function build_causal_capstone_example()
    dag = CausalDAG(; nodes=[:Z, :X, :M, :Y],
                      directed=[(:Z, :X), (:Z, :Y), (:X, :M), (:M, :Y)])
    triples = [CausalTriple(:Z, "increases", :X), CausalTriple(:Z, "increases", :Y),
               CausalTriple(:X, "increases", :M), CausalTriple(:M, "increases", :Y)]
    (dag=dag, triples=triples, treatment=:X, outcome=:Y)
end

"""
    causal_capstone(; example=build_causal_capstone_example()) -> Dict

Run the full categorical causal pipeline on one model and return a summary
showing each layer agreeing: the DAG-as-category (law-checked), the
intervention functor (functoriality), the twin network (a verified pushout),
the Shpitser–Pearl identifiability verdict + symbolic estimand, the
counterfactual direction, and a Lean certificate of the causal category.
"""
function causal_capstone(; example=build_causal_capstone_example())
    G, triples, x, y = example.dag, example.triples, example.treatment, example.outcome

    Ccat = causal_category(G)
    iv = intervention_functor(G, x)
    twin = twin_network(G, x)
    ident = identify_effect(G, [y], [x])
    cf = counterfactual_effect(G, triples, x, y)
    lean = render_cat_certificate(Ccat; module_name="CausalCapstone")

    Dict(
        "model" => Dict("nodes" => String.(G.nodes),
                        "edges" => [[String(u), String(v)] for (u, v) in G.directed],
                        "treatment" => String(x), "outcome" => String(y)),
        "causal_category" => Dict(
            "objects" => length(Cat.objects(Ccat)),
            "is_category" => Cat.check_category_laws(Ccat),
            "hom_X_to_Y" => Cat.hom_cardinality(Ccat, x, y)),
        "intervention_functor" => Dict(
            "is_functor" => Cat.is_functorial(iv.functor),
            "removed_edges" => length(G.directed) - length(iv.mutilated.edges)),
        "twin_network_pushout" => Dict(
            "shared_background" => String.(twin.shared),
            "duplicated" => String.(twin.descendants),
            "twin_nodes" => length(twin.pushout.apex),
            "is_pushout" => Cat.verify_pushout(twin.pushout)),
        "identifiability" => Dict(
            "identifiable" => ident.identifiable,
            "algorithm" => String(ident.algorithm),
            "estimand" => ident.expression === nothing ? nothing : pretty_print(ident.expression)),
        "counterfactual" => Dict(
            "text" => cf.text, "identifiable" => cf.identifiable,
            "expected_direction" => cf.expected_direction, "path" => String.(cf.path)),
        "lean_certificate_lines" => length(split(lean, "\n")),
    )
end
