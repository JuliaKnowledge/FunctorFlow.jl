# FunctorFlow.jl

[![Build Status](https://github.com/JuliaKnowledge/FunctorFlow.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaKnowledge/FunctorFlow.jl/actions)
[![Documentation](https://github.com/JuliaKnowledge/FunctorFlow.jl/actions/workflows/Documentation.yml/badge.svg)](https://JuliaKnowledge.github.io/FunctorFlow.jl/dev/)

A categorical DSL and executable IR for building diagrammatic AI systems in
Julia, grounded in [*Categories for AGI*](https://people.cs.umass.edu/~mahadeva/papers/catagi.pdf).

```
Diagram / Spec → Categorical IR → Neural Architecture
```

## Acknowledgement

FunctorFlow.jl is a Julia port of the Python
[FunctorFlow](https://github.com/sridharmahadevan/catagi) package by
**Sridhar Mahadevan**, which is the first executable software implementation
of the categorical systems developed in:

- Sridhar Mahadevan, [*Categories for AGI*](https://people.cs.umass.edu/~mahadeva/papers/catagi.pdf)
- Sridhar Mahadevan, [*Large Causal Models from Large Language Models*](https://arxiv.org/abs/2512.07796) (arXiv:2512.07796)
- The Lean 4 formalization: [catagi](https://github.com/sridharmahadevan/catagi)

This Julia implementation extends the original with Julia macro-based DSL
support, integration with the [AlgebraicJulia](https://www.algebraicjulia.org/)
ecosystem, [Lux.jl](https://github.com/LuxDL/Lux.jl) as the neural backend,
and v1 features including universal constructions and causal semantics.

## Installation

Until registration, install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaKnowledge/FunctorFlow.jl")
```

Once the package is registered, installation will simplify to:

```julia
using Pkg
Pkg.add("FunctorFlow")
```

### Installing CategoricalDiagramSchema (for ACSet interop)

As of v0.2, FunctorFlow's ACSet schema lives in the companion package
[`CategoricalDiagramSchema.jl`](https://github.com/JuliaKnowledge/CategoricalDiagramSchema.jl)
(shared with [`CatNet.jl`](https://github.com/JuliaKnowledge/CatNet.jl)). It is
declared as a **weak dependency**: FunctorFlow itself doesn't pull it in, but
the `to_acset` / `from_acset` / `diagram_to_acset` / `acset_to_diagram`
functions only have methods once it is loaded.

To enable ACSet conversion:

```julia
using Pkg
Pkg.add(url = "https://github.com/JuliaKnowledge/CategoricalDiagramSchema.jl")

using FunctorFlow, CategoricalDiagramSchema
acs = to_acset(D)            # CategoricalDiagramACSet
D2  = from_acset(acs)        # FunctorFlow.Diagram
```

If you only `using FunctorFlow`, calling `to_acset` raises a `MethodError`
that points you here.

### Installing Catlab (for symbolic Catlab projection)

As of **v0.5.0**, Catlab is also a **weak dependency** (was a hard dep in
v0.4.0 and earlier). The functions `to_presentation`, `to_symbolic`, and
`define_theory` only have methods once `Catlab` is loaded; the types
`CategoricalModelObject`, `ModelMorphism`, and `NaturalTransformation`
remain available without Catlab. The previously-re-exported names
`nparts`, `subpart`, `add_part!`, and `incident` are no longer brought
into scope by `using FunctorFlow` — load them from
`Catlab.CategoricalAlgebra` directly.

To enable Catlab-backed methods:

```julia
using Pkg
Pkg.add("Catlab")
using FunctorFlow, Catlab
pres = to_presentation(D)    # Catlab Presentation (FreeCategory)
sym  = to_symbolic(D)        # NamedTuple of FreeCategory Ob/Hom
```

This change resolves the transitive dependency conflict (Catlab→Compose→
DataStructures vs `TinyGrad.jl`→Symbolics→MultivariatePolynomials) that
prevented `using FunctorFlow, TinyGrad` in the same session in v0.4.0.

## Quick start

### Programmatic API

```julia
using FunctorFlow

# Build a KET (Kan Extension Template) block
D = Diagram(:MyKET)
add_object!(D, :Values; kind=:messages)
add_object!(D, :Incidence; kind=:relation)
add_left_kan!(D, :aggregate; source=:Values, along=:Incidence, reducer=:sum)

# Compile and run
compiled = compile_to_callable(D)
result = run(compiled, Dict(
    :Values => Dict(1 => 1.0, 2 => 2.0, 3 => 4.0),
    :Incidence => Dict("left" => [1, 2], "right" => [2, 3])
))
result.values[:aggregate]
# Dict("left" => 3.0, "right" => 6.0)
```

### Macro DSL

```julia
using FunctorFlow

D = @diagram KET begin
    @object Tokens kind=:messages
    @object Nbrs kind=:relation
    @object Ctx kind=:contextualized_messages
    @left_kan aggregate source=Tokens along=Nbrs target=Ctx reducer=:sum
    @port input Tokens direction=:input type=:messages
    @port output aggregate direction=:output type=:contextualized_messages
end
```

### Named blocks

```julia
using FunctorFlow

# Pre-built block patterns
ket = ket_block(; name=:EdgeAggregator, reducer=:mean)
db  = db_square(; first_impl=x -> x*2, second_impl=x -> x+1)
gt  = gt_neighborhood_block()

# From the registry
diagram = build_macro(:ket; name=:TutorialKET)
```

## Core concepts

FunctorFlow operationalizes the categorical design language from *Categories for AGI*:

| Concept | FunctorFlow primitive | Covers |
|---------|----------------------|--------|
| **KET** (Kan Extension Transformer) | `left_kan` | Attention, pooling, message passing, context fusion |
| **DB** (Diagrammatic Backpropagation) | `obstruction_loss` | Commutativity control, consistency-aware learning |
| **GT** (Graph Transformer) | `gt_neighborhood_block` | Geometric message passing over simplicial structure |
| **BASKET** | `basket_workflow_block` | Plan fragment composition via left-Kan aggregation |
| **ROCKET** | `rocket_repair_block` | Plan repair via right-Kan completion |
| **Democritus** | `democritus_gluing_block` | Sheaf-theoretic local-to-global gluing |
| **Higher horns** | `higher_horn_block` | Multi-step simplicial coherence and horn-family regularization |

## Compilation pipeline

```
Surface DSL (Diagram)
    ↓  to_ir()
Normalized IR (DiagramIR)
    ↓  compile_to_callable()
Backend-neutral executor (CompiledDiagram)
    ↓  [with Lux.jl extension]
Neural architecture (Lux layer)
```

## Categorical kernel (`FunctorFlow.Cat`)

A pure-Julia, **law-checkable** core of *concrete, finite* category theory — no
Catlab dependency, so it is always available. Because everything is finite,
the categorical laws are *verified by enumeration*, not asserted:

```julia
using FunctorFlow
const Cat = FunctorFlow.Cat

# the free category on a finite DAG  a →f→ b →g→ c
C = FreeCat([:a, :b, :c], [(:f, :a, :b), (:g, :b, :c)])
Cat.check_category_laws(C)          # true (associativity + identities)
Cat.homset(C, :a, :c)               # [a→c via f·g]

# a Diagram instance is literally a functor to Set (a C-Set)
D = Diagram(:T); add_object!(D, :X); add_object!(D, :Y); add_morphism!(D, :f, :X, :Y)
sf = diagram_setfunctor(D; sets=Dict(:X=>[1,2], :Y=>[:a,:b]),
                            functions=Dict(:f=>[1=>:a, 2=>:b]))
Cat.is_functorial(sf)               # true
```

### Yoneda lemma — computed, not quoted

```julia
F  = representable_functor(C, :a)   # Hom(a, -) : C → FinSet
yoneda_lemma_holds(C, :a, F)        # true: Nat(Hom(a,-), F) ≅ F(a), verified by enumeration
is_representable(F)                  # (representable=true, witness=:a, element=…)
```

`yoneda_map` / `yoneda_inverse` realise the bijection both ways.

### Universal properties — verified, not asserted

Limits and colimits are built concretely in FinSet, and their universal
property is *checked* (a unique mediating map exists for every probe cone):

```julia
A = Cat.FinSet([1,2]); B = Cat.FinSet([:x,:y,:z])
pb = Cat.product(A, B)
Cat.verify_product(pb, A, B)        # true — existence + uniqueness of ⟨q1,q2⟩
```

`Cat.product / coproduct / equalizer / coequalizer / pullback / pushout` each
come with `mediate`/`comediate` and a `verify_*`.

### Adjunctions

Functors, natural transformations, and adjunctions with the triangle
identities verified by enumeration:

```julia
C   = FreeCat([:a,:b,:c], [(:f,:a,:b), (:g,:b,:c)])
adj = Cat.initial_object_adjunction(C, :a)   # (a : 1→C) ⊣ (! : C→1)
Cat.is_adjunction(adj)              # true — both triangle identities hold
```

`Cat.restrict(X, F)` is the reindexing functor `F*` whose adjoints
`Σ_F ⊣ F* ⊣ Π_F` are the genuine Kan extensions.

### Categories with relations, and Kan extensions

Commuting diagrams are real categories via `FinPresentedCat` (free category
modulo declared path equalities, with congruence closure):

```julia
sq = commutative_square()           # f·h = g·k
Cat.hom_cardinality(sq, :a, :d)     # 1 — the square commutes (vs 2 in the free diamond)
Cat.check_category_laws(sq)         # true
yoneda_lemma_holds(sq, :a, representable_functor(sq, :a))   # Yoneda works over it
```

Colimits and limits of `Set`-valued functors are the Kan extensions along the
terminal functor (`Σ_! = colimit`, `Π_! = limit`), with verified universal
properties:

```julia
col = Cat.colimit(X)   # a span's colimit is the pushout
Cat.verify_colimit(col)
lim = Cat.limit(X)     # a cospan's limit is the pullback
Cat.verify_limit(lim)
```

### Monads and Kan extensions along any functor

```julia
chain = FreeCat([:a,:b,:c], [(:f,:a,:b), (:g,:b,:c)])
m = Cat.closure_monad(chain, Dict(:a=>:b, :b=>:b, :c=>:c))   # a closure operator
Cat.is_monad(m)             # true — unit + associativity laws
Cat.check_kleisli_laws(m)   # the Kleisli category is a category

# Lan_F ⊣ F* ⊣ Ran_F along an arbitrary functor F : C → D
LanX = Cat.left_kan(F, X)   # left Kan extension (pointwise colimit over F↓d)
RanX = Cat.right_kan(F, X)  # right Kan extension (pointwise limit over d↓F)
# verified: Nat(Lan_F X, Y) ≅ Nat(X, F*Y)  and  Nat(F*Y, X) ≅ Nat(Y, Ran_F X)
```

The classical categorical foundation is now largely complete — categories
(free and with relations), functors, natural transformations, C-Sets, Yoneda,
limits/colimits, adjunctions, monads/Kleisli, and Kan extensions along any
functor — all law-checked.

### The kernel applied, and certified

The applied layers are being re-expressed *on* the kernel. Corpus synthesis is
now a genuine colimit:

```julia
col = corpus_colimit(claims)          # claim gluing = a coequalizer (colimit) in FinSet
length(col.apex) == length(glue_corpus_claims(claims))   # the colimit IS the glued corpus
```

And the kernel's laws are machine-checked in Lean (not just at runtime):

```julia
render_cat_certificate(commutative_square())   # → Lean `isCategory = true` by native_decide
render_functor_certificate(F)                  # → Lean `isFunctor  = true` by native_decide
```

### Capstone: causal models *are* categories

The causal/counterfactual layer is built on the kernel — a causal DAG is a
category, an intervention is a functor, and a counterfactual twin network is a
pushout:

```julia
ex   = build_causal_capstone_example()       # Z→X, Z→Y, X→M→Y
summ = causal_capstone(; example=ex)
# DAG-as-category (laws) → intervention functor → twin-network PUSHOUT →
# Shpitser–Pearl identifiability + estimand → counterfactual → Lean certificate
summ["twin_network_pushout"]["is_pushout"]   # true — twin network IS a pushout
summ["identifiability"]["estimand"]          # the back-door-adjusted estimand
```

And the kernel's laws are **machine-checked in Lean** — categories, functors,
adjunctions, monads, **and colimits/limits** all carry `native_decide`
certificates (`render_cat_certificate`, `render_functor_certificate`,
`render_adjunction_certificate`, `render_monad_certificate`,
`render_colimit_certificate`, `render_limit_certificate`).

The topos layer is rigorous too: `subobject_classifier(C)` is the genuine
classifier `Ω` of the presheaf topos (`Ω(c)` = cosieves on `c`), and
`classify(X, sub)` gives the characteristic map with the classification
theorem `χ⁻¹(true) = sub` verified.

### Backpropagation is a functor

The deepest category-theory ↔ AI bridge — gradient backprop is the
reverse-derivative (transpose) functor, and the **chain rule is functoriality**.
In `FinVect_n` (linear maps over ℤ_n), a layer is a matrix, a network a
composite, the forward pass applies the matrix and the backward pass applies the
transpose:

```julia
W1 = Cat.LinMap(7, 3, 2, [1 0 2; 0 1 1]);  W2 = Cat.LinMap(7, 2, 1, reshape([1,1],1,2))
Cat.transpose_is_functorial(W1, W2)   # (g∘f)ᵀ = fᵀ∘gᵀ  — backprop reverses the network
Cat.backprop_demo()                   # forward + chain-rule backprop, two ways, agree

render_backprop_certificate(W1, W2)   # → Lean `chainRuleHolds … = true` by native_decide
```

### State machines as coalgebras; probability as a Markov category

```julia
M = Cat.MooreMachine([:s0,:s1,:s2], [:a], [:x,:y],
    Dict((:s0,:a)=>:s1,(:s1,:a)=>:s2,(:s2,:a)=>:s1), Dict(:s0=>:x,:s1=>:y,:s2=>:y))
Cat.minimize(M)                       # quotient by bisimilarity = image in the final coalgebra
render_bisimulation_certificate(M, R) # → Lean native_decide proof of behavioural equivalence

f = Cat.StochMap(...); Cat.markov_compose(f, g)   # Chapman–Kolmogorov in the Markov category
causal_markov_kernel(dag, mechanisms)             # a causal DAG as a Markov-category morphism
Cat.bayes_update(prior, likelihood, obs)          # Bayesian inference = disintegration
```

### Embeddings as enriched categories; learning as lenses

```julia
M = Cat.embedding_metric(Dict(:a=>[0,0], :b=>[3,0], :c=>[3,4]); metric=:l1)
Cat.is_lawvere_metric(M)        # an embedding IS a category enriched over costs (triangle = composition)
render_metric_certificate(M)    # → Lean native_decide proof of the enriched-category axioms

l = Cat.record_lens([:a1,:a2], [:b1,:b2])
Cat.is_very_well_behaved(l)     # GetPut/PutGet/PutPut — the backward-pass laws
render_lens_certificate(l)      # → Lean native_decide proof of the lens laws
```

### Internal logic, concepts, data, dynamics, recursion

```julia
Cat.is_heyting_algebra(Cat.cosieve_heyting(C, c))   # intuitionistic logic of the topos (Ω)
Cat.formal_concepts(objects, attributes, incidence) # concept learning via a Galois connection
Cat.category_of_elements(X)                         # ∫F — the C-Set's "database of rows"
Cat.moore_to_poly(M)                                # a state machine IS a dependent lens S·y^S → O·y^I
Cat.cata(eval_algebra, term)                        # a fold/tree-RNN IS a catamorphism (initial algebra)
```

### Coends, operads, 2-categories, sheaves

The higher-categorical layer — each built on (and reusing the verified universal
properties of) the kernel:

```julia
Cat.coend(P)                       # ∫^c P(c,c) as a coequalizer of dinaturality — "attention as a coend"
Cat.wiring_operad(...)             # diagrams/wiring as an operad: γ = "substitute a sub-architecture for a box"
Cat.operad_laws(O)                 # operad associativity + unit, checked by enumeration
Cat.cat_two_category(...)          # a strict 2-category of categories/functors/natural transformations
Cat.check_interchange_law(K)       # the 2-cell interchange law (pasting squares agree)
Cat.is_sheaf(C, J, F)              # the sheaf condition: every matching family glues to a unique amalgamation
```

- **Coends** (`src/cat/coend.jl`): a `Profunctor` `Cᵒᵖ × C → Set` and its coend
  `∫^c P(c,c)`, realised as the coequalizer of the two dinaturality maps — so it
  inherits the *verified* universal property of `Cat.coequalizer`. The worked
  example is **attention as a coend** (the same colimit-of-a-bimodule that
  `left_kan` computes pointwise).
- **Operads / multicategories** (`src/cat/operad.jl`): one-colored (symmetric)
  operads with substitution `γ`, the associativity/unit (and equivariance) laws
  checked by enumeration. A FunctorFlow wiring diagram *is* an operation in the
  operad of wiring diagrams; `γ` is "plug a sub-architecture into a box".
- **Strict 2-categories / bicategories** (`src/cat/twocat.jl`): 0/1/2-cells with
  vertical and horizontal composition tied by the **interchange law**;
  `cat_two_category` builds the 2-category of small categories, functors and
  natural transformations (Para reparametrisations form a *bi*category).
- **Sheaves** (`src/cat/sheaf.jl`): Grothendieck (co)topologies, matching
  families, amalgamations, and `is_separated` / `is_sheaf` — local-to-global
  gluing, the structural backbone of corpus-synthesis-as-colimit.

So the connection between category theory and AI is concrete and certified:
**backprop-as-functor**, JEPA-exactness-as-commutativity, attention/KET-as-Kan-
extension (and **attention-as-a-coend**), RNNs/automata-as-coalgebras,
folds/tree-RNNs-as-catamorphisms,
probability/causality-as-a-Markov-category, embeddings-as-enriched-categories,
learning-as-lenses/Para, dynamical-systems-as-polynomial-functors,
concept-learning-as-Galois-connections, neuro-symbolic-logic-as-a-Heyting-algebra,
wiring-diagrams-as-operads, pasting-squares-as-2-cells, local-to-global-gluing-as-sheaves,
causal-models-as-categories, counterfactuals-as-pushouts, and
corpus-synthesis-as-colimit — every piece on one kernel, runtime-checked and
(for the kernel laws) machine-checked in Lean (including a Mathlib-free,
induction-proved **general chain rule** `(A·B)ᵀ = Bᵀ·Aᵀ`).

The categorical foundation is now complete, applied, and certified: every
classical construction is law-checked at runtime *and* Lean-certified, and the
corpus-synthesis and causal layers are genuinely built on the kernel.

### One end-to-end pipeline, every layer

The whole system runs on the single kernel — CLIFF routing, textbook grounding,
causal/counterfactual reasoning, corpus synthesis and JEPA all share it:

```julia
report = integrated_pipeline("How similar is Adobe to Nike across filings?")
report["route"]                              # "company_similarity"  (CLIFF)
report["textbook_chapters"]                  # backing Categories-for-AGI chapters
report["demos_via_category"]                 # demos recovered by category composition
report["causal_capstone"]["twin_network_pushout"]["is_pushout"]   # the causal layer
report["layers_exercised"]                   # 8 layers, one query

end_to_end_capstone()   # adds corpus-as-colimit + JEPA-as-commuting-square cross-checks
```

`cliff_knowledge_category()` makes the `route → chapter → demo` linkage a genuine
category (composition recovers a route's demos); `jepa_square_category()` makes
JEPA exactness the commutativity of a square. Every applied surface now sits on
the same law-checked, Lean-certified categorical kernel.

## Diagram composition

Diagrams compose via namespaced inclusion:

```julia
parent = Diagram(:Parent)
child = ket_block()
inc = include!(parent, child; namespace=:encoder)

# Access namespaced elements
operation_ref(inc, :aggregate)  # :encoder__aggregate
port_spec(inc, :output)         # Port with namespaced ref
```

## v1 features

### Universal constructions

Build models through categorical universal properties:

```julia
ket1 = ket_block(; name=:KET1)
ket2 = ket_block(; name=:KET2)

# Pullback: joint constraint-compatible model
pb = pullback(ket1, ket2; over=:SharedContext)

# Product: independent combination
prod = product(ket1, ket2)

# Pushout: merge along shared interface
po = pushout(ket1, ket2; along=:SharedBase)
```

### Causal semantics (RN-Kan-Do-Calculus)

Explicit causal interpretation of Kan primitives:

```julia
ctx = CausalContext(:experiment;
    observational_regime=:obs,
    interventional_regime=:do)

cd = build_causal_diagram(:CausalModel; context=ctx)
# cd.base_diagram has:
#   :intervene (left-Kan → intervention/do-calculus)
#   :condition (right-Kan → conditioning/observational)
```

### Lean proof certificates

```julia
D = ket_block()
lean_code = render_lean_certificate(D)
write_lean_certificate(D; output_dir="proofs/generated")
```

### CLIFF-style orchestration layer

FunctorFlow.jl now also includes a Julia-native orchestration surface inspired by
the `CLIFF_CatAgi` router and conscious-workspace layer:

```julia
router = build_cliff_query_router()
decision = route_cliff_query(router, "How similar is Adobe to Nike?"; execution_mode=:deep)

example = build_cliff_orchestration_example()
summary = summarize_cliff_orchestration_example(example)
summary["route_decision"]["route_name"]   # "company_similarity"
summary["convergence"]["stop_trigger"]    # "stability"

runtime_example = build_cliff_runtime_example()
trace = execute_cliff_runtime_example(runtime_example)
trace.result.status                         # :completed
summarize_cliff_route_trace(trace)["counts"]["updates"]  # 2
```

The optional `AgentFramework.jl` extension can then turn `CLIFFAgentSpec`,
`InteractiveCheckpointRequest`, and `RouteRunResult` values into concrete Julia
agents, workflow checkpoint requests, and checkpoint payloads for LLM-backed
execution.

### Textbook-grounded routing

Following `CLIFF_CatAgi`'s principle that *every route links back to the
textbook*, routes and block macros are linked to chapters of *Categories for
AGI*:

```julia
rt = route_with_textbook("How similar is Adobe to Nike across recent filings?")
rt.decision.route_name        # :company_similarity
[c.number for c in rt.route_chapters]   # chapters backing the route, e.g. [4, 7, 13, 16]
rt.demos                      # runnable FunctorFlow block macros for those chapters

recommend_chapters("use do-calculus to check identifiability")[1].title
# "Judo Calculus"   (Chapter 14)

chapters_for_primitive(:ket)  # which chapters ground the KET demo (Ch. 5)
```

## Energy-based cost modules

Energy functions and cost modules are **executable**, not just declarative.
Build an energy block, run it, and evaluate the `C = Σ uᵢ·ICᵢ + Σ vⱼ·TCⱼ`
decomposition against the result:

```julia
D = energy_block(; config=EnergyBlockConfig(energy_type=:vicreg,
                                            variance_weight=1.0,
                                            covariance_weight=0.5))

result, energies, costs = run_with_costs(D, Dict(:Prediction => X, :Target => Y))
energies[:energy]            # the declared energy, actually computed
costs[:cost]["total"]        # weighted IC + TC total
costs[:cost]["components"]   # per-component breakdown
```

Built-in energies: `:l2`, `:cosine`, `:smooth_l1`, `:vicreg`, `:barlow_twins`,
and `:contrastive` (InfoNCE); regularisers `:variance` and `:covariance`.

## Causal counterfactuals

Counterfactuals are routed through the complete Shpitser–Pearl ID algorithm
(`identify_effect`), so each "what if we intervened" carries a genuine
identifiability verdict — not just a templated claim flip:

```julia
triples = [CausalTriple(:smoking, "increases", :tar),
           CausalTriple(:tar,     "increases", :cancer)]

# latent confounder smoking ↔ cancer ⇒ front-door identifiable
G, _ = build_causal_dag_from_triples(triples; latent_pairs=[(:smoking, :cancer)])

cf = counterfactual_effect(G, triples, :smoking, :cancer)
cf.identifiable          # true (front-door criterion)
cf.estimand              # symbolic post-intervention distribution
cf.expected_direction    # +1  (product of edge polarities along the path)
cf.text                  # "Had we intervened to increase smoking, cancer would have increased […]"

# a bow arc X→Y with X↔Y is NOT identifiable — you get a hedge witness:
build_counterfactuals_from_triples(triples; domain="health")["counts"]
# Dict("identifiable" => …, "non_identifiable" => …, …)
```

This mirrors the point of the *Cognitive Categorical Transformer*
(arXiv:2605.28864): observational data alone can't separate causal from
correlational structure — the `do`-operator plus an identifiability check are
what give a counterfactual its force.

## Corpus synthesis

Synthesise causal claims extracted from many documents into a coherent
corpus-level graph: glue wording variants, tier by document support, score
simplicial **horn-fill coherence**, align to a query, and surface
disagreements:

```julia
ex  = build_corpus_synthesis_example()           # 3-document minimum-wage corpus
res = synthesize_corpus(ex.claims; query=ex.query)

res.coherence.horn_fill_ratio   # filled 2-horns / (filled + open)
res.disagreements               # claims with conflicting polarity across docs
summarize_corpus_synthesis(res) # ranked claims, truth tiers, coherence
```

Truth tiers (`corpus_truth_value`) are `:entailed` (all docs), `:strong_support`
(≥50%), `:provisional_support` (≥40% or ≥2 docs), `:weak_support`. The horn-fill
ratio is the claim presheaf's gluing axiom made quantitative — the same
simplicial signal the CCT paper finds dominant.

## Visualization

With a Makie backend loaded, render any diagram as a layered graph:

```julia
using FunctorFlow, CairoMakie

fig = plot_diagram(ket_block())          # objects, morphisms, Σ/Δ Kan, losses
save("ket.png", fig)
```

## Lean certificates

FunctorFlow.jl emits Lean 4 certificate files that are checked against a
self-contained Lake project under `proofs/` (no Mathlib dependency).

```julia
using FunctorFlow

D = ket_block()
cert = render_lean_certificate(D; module_name="MyKET")  # structural only
write("proofs/FunctorFlowProofs/Generated/MyKET.lean",
      "import FunctorFlowProofs\n\n" * cert)

# Verified exactness requires a real execution result with zero obstruction loss.
DB = db_square(; first_impl=identity, second_impl=identity)
result = FunctorFlow.run(DB, Dict(:State => 3.0))
exact_cert = render_lean_certificate(DB, result; module_name="MyExactDB")
```

Then verify with [elan / lake](https://leanprover-community.github.io/install/):

```bash
cd proofs && lake build
```

### What the certificates actually prove

The certificates carry **decidable, falsifiable** content (verified by
`native_decide`), not schema-level `True` placeholders:

- **Structural well-formedness** — every operation/port reference (and, for
  universal constructions, every projection/injection morphism, shared
  object, equalizer map, and factor namespace) is actually declared in the
  diagram. A malformed emission fails to type-check.
- **Zero obstruction** — `CommutingSquare`, `UniversalCone`/`Cocone`,
  `ParallelAgreement`, `QuotientAgreement`, and JEPA `CoalgebraExact` are
  proven *only* for **verified** certificates emitted from a real
  `ExecutionResult.losses` (or explicitly supplied `loss_values`) whose
  recorded obstruction losses are all zero.
- **Structural-only fallback** — `render_lean_certificate(D)` and
  `render_construction_certificate(uc)` still emit well-formed structural
  certificates, but they intentionally omit exactness theorems unless real
  zero-loss evidence is supplied.

Because the categorical properties are derived from recorded real loss
observations, `render_lean_certificate(D, result)` throws if any obstruction
loss is nonzero or missing; callers must then emit a structural certificate
instead.

The full Julia → Lean roundtrip — including a negative-control certificate
that must be *rejected* — lives in `test/test_lean_certificates.jl` and is
opt-in:

```bash
FF_LEAN_CI=true julia --project=. -e 'using Pkg; Pkg.test()'
```

The same flow runs in CI via `.github/workflows/lean.yml`.

## Dependencies

- **Required**: OrderedCollections.jl, JSON3.jl
- **Optional**: Lux.jl (neural backend), Catlab.jl (categorical algebra), Makie.jl (visualization)

## Related packages

- [Catlab.jl](https://github.com/AlgebraicJulia/Catlab.jl) — Applied category theory
- [ACSets.jl](https://github.com/AlgebraicJulia/ACSets.jl) — Attributed C-Sets
- [GATlab.jl](https://github.com/AlgebraicJulia/GATlab.jl) — Generalized Algebraic Theories
- [CSQL.jl](https://github.com/JuliaKnowledge/CSQL.jl) — Causal SQL databases
- [CQL.jl](https://github.com/JuliaKnowledge/CQL.jl) — Categorical Query Language

## License

MIT
