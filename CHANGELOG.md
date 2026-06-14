# Changelog

All notable changes to FunctorFlow.jl are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] — 2026-06-13

### Higher-categorical layer: coends, operads, 2-categories, sheaves + a general chain rule
Four more kernel modules — the higher-categorical structures behind attention,
wiring, pasting, and gluing — plus the backprop chain rule upgraded from
per-instance certificates to a real, Mathlib-free Lean theorem. (Developed as
five isolated experiments, then merged onto the kernel.)

- **Coends & profunctors** (`src/cat/coend.jl`): `Profunctor` (a functor
  `Cᵒᵖ × C → FinSet`) and its **coend** `∫^c P(c,c)`, realised as the
  coequalizer of the two dinaturality maps — so it inherits the *verified*
  universal property already proved for `Cat.coequalizer` in `limits.jl`. The
  worked example is **attention as a coend**: the same colimit-of-a-bimodule
  that `Cat.left_kan` computes pointwise. Also ships `end_` (the dual end).
  Exports: `Profunctor`, `profunctor_diag`, `CoendCocone`, `coend`,
  `coend_class`, `verify_coend`, `EndCone`, `end_`.
- **Operads / multicategories** (`src/cat/operad.jl`): finite one-colored
  (symmetric) operads with substitution `γ : O(n) × O(k₁) × … → O(Σkᵢ)`, the
  associativity & unit laws (and optional `Sₙ`-equivariance) checked by
  enumeration. A FunctorFlow wiring diagram *is* an operation in the operad of
  wiring diagrams — `γ` is "substitute a sub-architecture for a box". Ships
  `commutative_operad`, `associative_operad`, `wiring_operad`,
  `little_intervals_operad`, `unary_monoid`, plus `operad_laws` /
  `operad_symmetry_laws`.
- **Strict 2-categories / bicategories** (`src/cat/twocat.jl`): 0-, 1-, and
  2-cells with **vertical** (`vcomp`) and **horizontal** (`hcomp`) composition
  tied by the **interchange law**, all tabulated and checked by enumeration
  (`check_two_category_laws`). `cat_two_category` builds the 2-category of small
  categories / `FinFunctor`s / `FunctorNatTrans`es by actually computing the
  natural-transformation composites; `deloop_monoid` is the smallest example.
  Para reparametrisations form a *bi*category (`para_is_bicategory_note`).
- **Sheaves** (`src/cat/sheaf.jl`): Grothendieck (co)topologies in the dual
  ("co") setting consistent with the kernel's copresheaves + cosieves —
  `Coverage`, `covering_sieves`, `is_grothendieck_topology`, `matching_families`,
  `amalgamations`, `is_separated`, `is_sheaf`, `separated_reflection`. Sheaf
  gluing (every matching family has a unique amalgamation) is the structural
  backbone of corpus-synthesis-as-colimit; ships `span_site` / `span_sheaf` /
  `span_non_sheaf` worked examples.
- **General chain rule, proved in Lean 4** (`proofs/FunctorFlowProofs/ChainRule.lean`):
  where `Learn.lean` certifies *instances* of `(g∘f)ᵀ = fᵀ∘gᵀ` over `ℤ_n`
  matrices by `native_decide`, this proves the **general** theorem
  `(A·B)ᵀ = Bᵀ·Aᵀ` for all conforming matrices, **by induction, with no
  Mathlib and no `native_decide`**. An index-based `Mat` representation gives
  `transp_mul` (unconditional given conformance), and `matT_matMul` bridges it
  to `Learn.lean`'s `List (List Nat)` encoding. Verified `sorry`/`admit`-free;
  `#print axioms` shows only `propext` / `Classical.choice` / `Quot.sound` (no
  `sorryAx`). Backprop functoriality — the chain rule — is now a genuine
  theorem, not only a per-network certificate.
- Tests: `test_coend.jl`, `test_operad.jl`, `test_twocat.jl`, `test_sheaf.jl`
  (coend = `left_kan` agreement and a non-universal cocone rejected; operad laws
  with a broken-associativity negative control; the interchange law with a
  violation caught; the sheaf condition with a non-sheaf presheaf rejected). Full
  suite: **1493 pass, 0 fail, 2 broken** (the expected TinyGrad/Makie skips); the
  Lean `proofs/` project (16 modules incl. `ChainRule`) builds clean.

### Breadth wave: internal logic, Galois/FCA, Grothendieck, Rel, Poly, F-algebras
Six more categorical structures, each with an AI reading:

- **Heyting algebras / intuitionistic internal logic** (`src/cat/heyting.jl`):
  `HeytingAlgebra` with `∧, ∨, ⇒, ¬, ⊤, ⊥` and the Heyting adjunction;
  `cosieve_heyting(C, c)` is the topos's truth-value algebra `Ω(c)` — the home
  of neuro-symbolic / intuitionistic reasoning. Lean `Heyting.lean` +
  `render_heyting_certificate`.
- **Galois connections & FCA** (`src/cat/galois.jl`): `Poset`,
  `is_galois_connection` (the simplest adjunction), and `formal_concepts` —
  Formal Concept Analysis (concept learning) as the Galois connection of a
  context. Lean `Galois.lean` + `render_galois_certificate`.
- **Grothendieck construction** (`src/cat/grothendieck.jl`):
  `category_of_elements(F)` (`∫F`, the "database of rows" of a C-Set) and its
  projection `∫F → C`; Lean-certifiable as a category via the existing cat cert.
- **Rel & the powerset monad** (`src/cat/rel.jl`): the category of relations =
  Kleisli of the powerset (nondeterminism) monad, with the dagger (converse).
- **Polynomial functors** (`src/cat/poly.jl`): Spivak's `Poly`; poly morphisms
  are **dependent lenses**, and `moore_to_poly` realises a Moore machine as a
  lens `S·y^S → O·y^I`, unifying the lens (`optics.jl`) and coalgebra
  (`coalg.jl`) stories — interfaces & dynamical systems.
- **F-algebras & catamorphisms** (`src/cat/falgebra.jl`): the dual of the
  coalgebra story — `Signature`, the term algebra, `cata` (a fold), and
  `cata_is_homomorphism`. Structural recursion / tree-RNNs / evaluators *are*
  catamorphisms (induction; vs. the coalgebraic coinduction of `coalg.jl`).
- **Comonads** (`src/cat/comonad.jl`): the dual of `monads.jl` — `Comonad`
  `(T, ε, δ)` with `is_comonad` (counit + coassociativity), `identity_comonad`,
  and `comonad_from_adjunction`. Context-dependent computation — the categorical
  model of convolution / attention windows. Completes the (co)monad duality
  alongside the (co)algebra and (co)limit duals.
- Tests: `test_logic_lattice.jl` + `test_dynamics_recursion.jl` + comonad
  testset (+39), with negative controls (non-Heyting / non-monotone Galois /
  non-concept) and Lean certs for Heyting and Galois.

### Yet more breadth: enriched/metric categories + lenses & Para
- **Enriched categories / metric spaces** (`src/cat/enriched.jl`): Lawvere's
  "a metric space *is* a category enriched over the cost quantale". `MetricCat`,
  `is_lawvere_metric` (identity `d(x,x)=0` + triangle = enriched composition),
  `is_enriched_functor` (non-expansive / 1-Lipschitz map), and `embedding_metric`
  (a representation/embedding → an enriched category). This is the categorical
  home of metric/embedding representation learning. Lean
  (`proofs/FunctorFlowProofs/Enriched.lean`): decidable `isLawvereMetric` /
  `isNonExpansive`; `render_metric_certificate`.
- **Lenses & Para** (`src/cat/optics.jl`): the modern categorical foundation of
  gradient-based learning. `Lens` (get/put) with `lens_compose`, the
  very-well-behaved laws (`lens_get_put`/`put_get`/`put_put`, `is_very_well_behaved`),
  and `record_lens`; `ParaMap` (a parametric/learnable layer `P×A→B`) with
  `para_compose`/`para_id` (the category of learnable maps). Together with
  backprop-as-functor this is the Para∘Lens account of learning. Lean
  (`proofs/FunctorFlowProofs/Optics.lean`): decidable lens laws;
  `render_lens_certificate`.
- Tests: `test_enriched.jl` + `test_optics.jl` (+25) — the metric/enriched
  axioms (with a triangle-violation rejected), non-expansive functors, the lens
  laws (with a bad lens caught), lens composition, and Para composition.

### More breadth: coalgebras (automata) + Markov categories (probability)
- **Coalgebras / automata** (`src/cat/coalg.jl`): `MooreMachine` as a coalgebra
  for `F(X) = O × X^I` — the categorical home of state machines and recurrent
  models. `moore_run` (behaviour), `is_bisimulation` (behavioural equivalence),
  `bisimilar` (coarsest stable partition), `minimize` (the quotient = image in
  the final coalgebra), and `coalgebra_morphism` (homomorphism check). Lean
  (`proofs/FunctorFlowProofs/Automata.lean`): decidable `isBisimulation` /
  `isCoalgMorphism`; `render_bisimulation_certificate` emits a `native_decide`
  proof of behavioural equivalence.
- **Markov categories** (`src/cat/markov.jl`): the Kleisli category of the
  finite-distribution monad — probability and causality, categorically.
  `Dist` (exact `Rational` distributions), `StochMap` (Markov kernels),
  `markov_compose` (Chapman–Kolmogorov), `markov_id` (Dirac), `markov_copy` /
  `markov_discard` (the comonoid structure), `markov_tensor`, and the laws via
  `markov_laws`. `causal_markov_kernel` realises a `CausalDAG` as a genuine
  Markov-category morphism (joint = product of mechanisms), and `bayes_update`
  is Bayesian inference as disintegration. This unifies the probabilistic and
  causal layers with the kernel.
- Tests: `test_coalg.jl` + `test_markov.jl` (+24) — bisimulation/minimization,
  Chapman–Kolmogorov, the Markov laws, causal factorization, and Bayesian update.

### Backpropagation is a functor — categorical deep learning, Lean-certified
The deepest category-theory ↔ AI bridge, made concrete and machine-checked.

- **`src/cat/learn.jl`** — `FinVect_n`, the category of linear maps over ℤ_n:
  `LinMap` (a neural layer), `forward` (forward pass), `lin_compose` (network
  composition), and `lin_transpose` / `backward` (the reverse-derivative, i.e.
  the vector–Jacobian product / backward pass). The headline:
  `transpose_is_functorial` proves `(g∘f)ᵀ = fᵀ∘gᵀ` — **backpropagation is the
  reverse-derivative functor and the chain rule is its functoriality**. Includes
  `finvect_category_laws` and a 2-layer `backprop_demo` showing whole-network and
  layer-by-layer backprop agree.
- **`proofs/FunctorFlowProofs/Learn.lean`** — `matMul`/`matT` over ℤ_n with a
  decidable `chainRuleHolds` ((g∘f)ᵀ = fᵀ∘gᵀ) and `matAssocHolds`.
  `render_backprop_certificate(f, g)` emits a `native_decide` proof of the chain
  rule for a network's actual layer matrices; the opt-in test certifies a 2-layer
  ℤ₇ network. (Over ℤ_n matrix arithmetic is exact, so the chain rule — a genuine
  theorem — is verified by computation for the concrete network.)
- Tests: `test_learn.jl` (+15) — forward functoriality, the chain rule as
  transpose functoriality, `FinVect_n` laws, non-commutativity of composition
  (why backprop must reverse), the demo, and certificate rendering.

This completes the arc: from the abstract kernel (categories … Kan extensions,
all Lean-certified) to a *rich, machine-checked connection between category
theory and AI* — backprop-as-functor, JEPA-exactness-as-commutativity,
attention/KET-as-Kan-extension, causal-models-as-categories,
counterfactuals-as-pushouts, and corpus-synthesis-as-colimit — every piece on
one kernel, runtime-checked and (for the kernel laws) Lean-certified.

### More applied surfaces + Lean-certified colimit/limit laws
- **Subobject classifier of the presheaf topos** (`src/cat/topos.jl`):
  `subobject_classifier(C)` builds `Ω : C → Set` (`Ω(c)` = cosieves on `c`, with
  the pullback restriction action); `omega_true` is the truth arrow; and
  `classify(X, sub)` produces the characteristic map `χ : X → Ω` of a
  subfunctor, with `verify_classifies` checking the classification theorem
  `χ⁻¹(true) = sub`. The topos layer is now rigorous and computable (e.g. over
  the arrow category `Ω(a)` has 3 cosieves, `Ω(b)` has 2). Works over presented
  categories too.
- **Lean certification of the colimit/limit laws** (`proofs/FunctorFlowProofs/Limits.lean`):
  a finite Set-valued diagram + cocone/cone is emitted as a `ColimitCert` /
  `LimitCert`, with decidable `isColimit` (cocone commutes ∧ jointly surjective
  ∧ identifications = connected components) and `isLimit` (cone commutes ∧
  projections biject onto the compatible families). `render_colimit_certificate`
  / `render_limit_certificate` emit `native_decide` proofs — certifying the Kan
  extensions `Σ_!` / `Π_!` along the terminal functor. The opt-in test now
  machine-checks a pushout-colimit and a pullback-limit, and rejects a corrupted
  colimit apex.

### Horizontal: every layer on the kernel + one end-to-end pipeline
The integration pass — the remaining applied surfaces re-expressed on the
kernel, and a single pipeline that flows a query through all of them.

- **`src/cat_integration.jl`**:
  - `cliff_knowledge_category()` — the CLIFF `route → chapter → demo` linkage as
    a genuine `Cat.FreeCat` (a 3-layer DAG): `demos_reachable_from(K, route)`
    recovers exactly the textbook-backed runnable demos *via category
    composition*. Law-checked and Lean-certifiable.
  - `jepa_square_category()` — JEPA exactness expressed as a commuting square
    (`FinPresentedCat` with `pred∘enc_x = enc_y∘γ`): "obstruction loss = 0" is
    literally `Hom(X, Zt)` collapsing to one morphism (vs. two in the free
    square).
  - `integrated_pipeline(query)` — flows a query through CLIFF routing →
    textbook chapters/demos → knowledge-category recovery → (for evidence/causal
    routes) the causal capstone (model-as-category → intervention functor →
    twin-network pushout → identifiability → counterfactual). One query, up to
    **eight layers**, one report.
  - `end_to_end_capstone()` — the canonical run, additionally cross-checking
    corpus-synthesis-as-colimit and JEPA-exactness-as-commutativity.
- Tests: `test_cat_integration.jl` (+22) — categorical recovery of the route
  linkage, JEPA commuting-square vs. free square, the full pipeline on causal
  and non-causal routes, and the end-to-end capstone.

FunctorFlow is now an integrated end-to-end system on one categorical kernel:
routing, textbook grounding, causal/counterfactual reasoning, corpus synthesis,
JEPA, energy, limits/colimits, adjunctions, monads and Kan extensions all share
the same law-checked, Lean-certified foundation.

### Capstone: causal categories + full Lean-certified kernel
The grand finale — the causal/counterfactual layer expressed entirely on the
kernel, and the Lean certification completed.

- **`src/cat_causal.jl`** re-founds the causal layer:
  - `causal_category(dag)` — a causal DAG *is* a `Cat.FreeCat` (variables =
    objects, directed edges = generators), law-checked and Lean-certifiable.
  - `intervention_functor(G, x)` — the `do(x)` mutilation *is* a functor
    `G_x̄ → G` (verified via `is_functorial`).
  - `twin_network(G, x)` — the counterfactual **twin / parallel-worlds network
    *is* a pushout**: the factual and interventional worlds amalgamated over the
    shared background (non-descendants of `x`), computed and `verify_pushout`-ed
    by the kernel's FinSet pushout.
  - `causal_capstone()` — one model (`Z→X, Z→Y, X→M→Y`) flowing through every
    layer: DAG-as-category (laws) → intervention functor → twin-network pushout
    → Shpitser–Pearl identifiability + symbolic estimand → counterfactual
    direction → emitted Lean certificate. All layers agree.
- **Lean certification completed** (`proofs/FunctorFlowProofs/Cat.lean`):
  `AdjunctionDecl` / `MonadDecl` with decidable `isAdjunction` (both triangle
  identities) / `isMonad` (unit + associativity) checks, plus emitters
  `render_adjunction_certificate` / `render_monad_certificate`. The opt-in
  `test_lean_certificates.jl` now machine-checks (via `native_decide`) the
  category, functor, **adjunction** (initial object) and **monad** (closure
  operator) laws — and rejects a corrupted category table. The kernel's laws are
  now certified in Lean, not only at runtime.
- Tests: `test_cat_causal.jl` (+23): DAG-as-category, intervention functor, the
  twin-network pushout, and the end-to-end capstone agreement.

The categorical foundation is now complete *and* applied *and* machine-certified:
every classical construction (categories with relations, functors, naturals,
C-Sets, Yoneda, limits/colimits, adjunctions, monads/Kleisli, Kan extensions)
is law-checked at runtime and Lean-certified, and two applied layers (corpus
synthesis, causal/counterfactuals) are genuinely built on the kernel.

### Foundation, applied: corpus-as-colimit + Lean-certified kernel
Connecting the categorical kernel to the applied layers, and to machine-checked proof:

- **Corpus synthesis re-founded as a colimit.** `corpus_gluing_diagram(claims)`
  builds the variant-pair relation `R ⇉ A` as a `Cat.SetFunctor`, and
  `corpus_colimit(claims) = Cat.colimit(…)` computes the glued corpus as a
  genuine colimit (coequalizer) in FinSet. A test confirms the colimit's apex
  has exactly `length(glue_corpus_claims(claims))` elements and satisfies the
  universal property — the Democritus sheaf-gluing of local claim-sections is
  now literally a colimit, not an analogy.
- **Lean certification of the kernel laws** (`proofs/FunctorFlowProofs/Cat.lean`):
  `CatTable` / `FunctorDecl` with decidable `isCategory` / `isFunctor` checks.
  `render_cat_certificate(C)` / `render_functor_certificate(F)` emit a finite
  category / functor as tables with `native_decide` proofs of the category and
  functor laws — a kernel-checked counterpart to the runtime enumeration checks.
  `test_lean_certificates.jl` now also builds a presented-category and a functor
  certificate, and a **corrupted category table (empty composition) is correctly
  rejected** by Lean — the Cat certification has teeth too.

### Monads/Kleisli + Kan extensions along an arbitrary functor (`FunctorFlow.Cat`)
- **`src/cat/monads.jl`** — `Monad` (endofunctor `T` + unit `η` + multiplication
  `μ`) with `is_monad` verifying the unit and associativity laws by enumeration;
  the **Kleisli category** (`kleisli_hom`/`kleisli_id`/`kleisli_compose` +
  `check_kleisli_laws`); `monad_from_adjunction` (`T = G∘F`, `μ = G(εF)`);
  `identity_monad`; and `closure_monad`, the canonical worked example (a closure
  operator on a poset is a monad).
- **`src/cat/kan_general.jl`** — `left_kan(F, X) = Lan_F X` and
  `right_kan(F, X) = Ran_F X` along an *arbitrary* functor `F : C → D`, computed
  pointwise as the colimit/limit over the comma categories `F↓d` / `d↓F`, each a
  full `SetFunctor` on `D` (with its action on `D`-morphisms). These are the
  genuine adjoints of the restriction functor: the **whole adjoint triple
  `Lan_F ⊣ F* ⊣ Ran_F`** is verified in the tests via the hom-set bijections
  `Nat(Lan_F X, Y) ≅ Nat(X, F*Y)` and `Nat(F*Y, X) ≅ Nat(Y, Ran_F X)`. The
  `colimit`/`limit` of `kan.jl` are recovered as the special case `F = ! : C → 1`.
- Tests: `test_monads.jl` + `test_kan_general.jl` (+23): monad/Kleisli laws,
  closure-operator and adjunction-induced monads, Kan reduction to colimit/limit
  along the terminal, the identity-functor case, and the full adjoint-triple
  cardinalities (terminal functor *and* an inclusion).

With this the classical categorical foundation is largely complete (categories
with relations, functors, natural transformations, C-Sets, Yoneda, limits/
colimits, adjunctions, monads/Kleisli, and Kan extensions along any functor).
Remaining: re-founding the applied layers (counterfactuals, corpus synthesis,
CLIFF) on the kernel, and Lean certification of the kernel's laws.

### Presented categories (relations) + Kan extensions (`FunctorFlow.Cat`)
- **`src/cat/presented.jl`** — `FinPresentedCat`: the free category on a DAG
  *quotiented by relations* (declared equalities of parallel paths), with
  congruence closure computed by enumeration so hom-sets are finite classes of
  canonical representatives. Commuting diagrams are now genuine categories:
  `commutative_square()` has `|Hom(a,d)| = 1` (vs. 2 in the free diamond), and
  `a` is genuinely initial. The whole kernel (`SetFunctor`, `representable_functor`,
  Yoneda, `FinFunctor`, adjunctions, `check_category_laws`) was generalised from
  `FreeCat` to `AbstractCategory`, so it all runs over presented categories;
  Set-valued functors and functors out of a presented category are checked to
  **respect the generating relations**.
- **`src/cat/kan.jl`** — general finite `colimit` and `limit` of a `SetFunctor`
  `C → FinSet`, which are exactly the Kan extensions along the terminal functor
  `C → 1` (`Σ_! = colimit`, `Π_! = limit`, the adjoints of the constant-functor
  restriction `Δ`). Each carries its universal (co)cone with `mediate`/`comediate`
  and a `verify_*` that checks the universal property by enumeration. As limits
  in a diagram category these recover (co)products, (co)equalizers, push/pullbacks
  (e.g. a span's colimit is the pushout; a cospan's limit is the pullback).
- Tests: `test_presented.jl` + `test_kan.jl` (+30): commutative-vs-free
  distinction, congruence normalisation, Yoneda over a presented category,
  relation-respecting (and relation-violating) functoriality, and colimit/limit
  universal properties recovering pushout/pullback/coequalizer.

Remaining for a complete foundation: monads/Kleisli, Kan extension `Lan_F`/`Ran_F`
along an *arbitrary* functor (the adjoints of `restrict`), re-founding the
applied layers on the kernel, and Lean certification of the categorical laws.

### Verified universal properties + adjunctions (`FunctorFlow.Cat`)
Extending the kernel toward a complete foundation:

- **`src/cat/limits.jl`** — `product`, `coproduct`, `equalizer`, `coequalizer`,
  `pullback`, `pushout` constructed concretely in FinSet, each with a
  mediating-morphism builder (`mediate` / `comediate`) and a `verify_*` that
  **checks the universal property by enumeration**: for every probe test
  (co)cone there is a *unique* mediating map making the triangles commute.
  This is the genuine universal-property verification the diagram-level
  constructions in `universal.jl` only assert structurally (and a non-universal
  cone is correctly rejected).
- **`src/cat/adjunction.jl`** — `identity_functor`, functor `compose`,
  `FunctorNatTrans` (general natural transformations between functors, with a
  naturality-square `is_natural` check), and `Adjunction` with
  `is_adjunction` verifying **both triangle identities** (`ε_{Fc}·F(η_c)=id`,
  `G(ε_d)·η_{Gd}=id`). Ships `initial_object_adjunction`, a worked
  `(initial : 1→C) ⊣ (! : C→1)` whose successful construction is a certificate
  of initiality, and `restrict` (the reindexing functor `F*`, whose adjoints
  `Σ_F ⊣ F* ⊣ Π_F` are the genuine Kan extensions — of which the DSL's `Σ`/`Δ`
  are the operational shadow).
- Tests: `test_limits.jl` + `test_adjunction.jl` (+35 assertions): all six
  universal properties (incl. a falsified non-universal cone), naturality
  (positive + non-natural on a parallel pair), the triangle identities, the
  free-vs-commutative distinction (a free diamond's source is *not* initial),
  and restriction.

Remaining for a complete foundation: monads/Kleisli, genuine Kan extensions
`Σ_F`/`Π_F` along a functor (the adjoints of `restrict`), finitely-presented
categories *with relations*, and re-expressing the applied layers
(counterfactuals, corpus synthesis, CLIFF) on the kernel.

### Verified categorical kernel + Yoneda (`FunctorFlow.Cat`)
A first, genuinely *founded* categorical core — pure Julia, no Catlab
dependency, so it is always available and law-checkable in the base
environment. "Concrete and finite" by design, so every law is verified by
enumeration rather than asserted:

- **`src/cat/Cat.jl`** — the kernel:
  - `FinSet` / `FinFunction` (the category **FinSet**): totality-checked
    functions, `compose`, `id`.
  - `FreeCat` — the free category on a finite DAG: objects + generating edges,
    path hom-sets (`homset`), composition by concatenation, identities as empty
    paths; a directed-cycle guard (finite hom-sets); `check_category_laws`
    verifies associativity + identity by enumeration.
  - `FinFunctor` (functors between categories) and `SetFunctor` (copresheaves
    `C → FinSet`, i.e. **C-Sets** / ACSets) with `is_functorial` checks;
    `CatNatTrans` (natural transformations) with an `is_natural` square check.
- **`src/cat/yoneda.jl`** — representables and the Yoneda lemma, *computably*:
  - `representable_functor(C, c)` = `Hom(c, -)`; `representable_presheaf(C, c)`
    = `Hom(-, c)` (via the opposite category).
  - `yoneda_map` / `yoneda_inverse` realise the bijection
    `Nat(Hom(c,-), F) ≅ F(c)`; `yoneda_lemma_holds(C, c, F)` verifies it
    (round-trip, naturality, injectivity, and — when feasible — the
    `|Nat| = |F(c)|` cardinality via brute-force enumeration).
  - `is_representable(F)` searches for a representing object.
- **`src/cat_bridge.jl`** — `diagram_freecat(D)` exposes a `Diagram`'s shape as
  a genuine `FreeCat`, and `diagram_setfunctor(D; sets, functions)` realises a
  diagram instance as a functor to Set — making "a Diagram instance is a
  functor to Set" literal and law-checked rather than a slogan.
- Interface verbs (`compose`/`id`/`dom`/`cod`/`homset`/`is_natural`) stay
  namespaced inside `Cat` to avoid clashing with FunctorFlow's own
  `compose`/`is_natural`; the type names and Yoneda functions are re-exported.
- Tests: `test_cat_kernel.jl` + `test_yoneda.jl` (+59 assertions) covering the
  category laws, functoriality, naturality, the Yoneda bijection on several
  categories, representability detection, and the Diagram bridge.

This is step 1 toward a complete categorical foundation; still ahead are
adjunctions (unit/counit/triangle identities), monads/Kleisli, genuine
universal-property verification for the limit/colimit constructions, and
finitely-presented categories *with relations* (the kernel currently covers
free categories on finite DAGs).

### Causal counterfactuals + corpus synthesis (CLIFF_CatAgi ports)
Two categorically-meaningful capabilities from `CLIFF_CatAgi`, ported and
upgraded rather than copied:

- **Counterfactuals on `identify_effect`** (`src/counterfactuals.jl`). Where
  `CLIFF_CatAgi`'s `democritus_counterfactuals.py` emits *templated*
  claim-flips with no identifiability content, every counterfactual here is
  routed through the complete Shpitser–Pearl ID algorithm:
  - `CausalTriple` (+ `relation_polarity` lexicon, `causal_triple` claim
    parser); `build_causal_dag_from_triples` assembles an acyclic `CausalDAG`
    (cycle-breaking recorded; latent confounders → bidirected edges).
  - `counterfactual_effect(G, triples, x, y)` returns a `Counterfactual`
    carrying the identifiability verdict + symbolic estimand (or a hedge
    witness of non-identifiability) from `identify_effect`, plus the predicted
    direction of effect from the product of edge polarities along the causal
    path. `intervention_level=:decrease` flips the sign.
  - `build_counterfactuals_from_triples` is the batch claim API (parity with
    the Python), with identifiability tallies and a DAG summary.
  - This follows the central point of Mahadevan's *Cognitive Categorical
    Transformer* (arXiv:2605.28864): observational data alone cannot
    distinguish causal from correlational structure — the `do`-operator and
    an identifiability check are what give a counterfactual its force.

- **Real corpus-synthesis engine** (`src/corpus_synthesis.jl`), porting
  `democritus_corpus_synthesis.py`: `CorpusClaim`s extracted per-document are
  normalised, **glued** across documents (`glue_corpus_claims`, Jaccard
  variant merge with polarity-conflict detection), assigned a support tier
  (`corpus_truth_value`: entailed / strong / provisional / weak), scored for
  **simplicial coherence** (`homotopy_coherence`: vertices, edges, filled
  triangles, open 2-horns, horn-fill ratio, components) and query relevance
  (`query_alignment`), then ranked by `synthesize_corpus` with disagreements
  surfaced. The horn-fill ratio is the same simplicial structural signal the
  CCT paper finds dominant, and is the claim presheaf's gluing axiom made
  quantitative. Ships `build_corpus_synthesis_example` + `summarize_corpus_synthesis`.
- Tests: `test_counterfactuals.jl` (front-door identifiable, bow-arc hedge,
  polarity composition, cycle-breaking) and `test_corpus_synthesis.jl`
  (gluing, truth tiers, horn-fill, query alignment, end-to-end) — +67 assertions.

### Lean certificates — no longer vacuous
- The Lean 4 certificate schema (`proofs/FunctorFlowProofs/`) previously
  defined every *categorical* property as `Prop := True` (proved by
  `trivial`) or via `rfl` on a constant, so the universal-construction /
  JEPA / energy "theorems" held for any input whatsoever. They now carry
  **genuine, falsifiable content**, verified by `native_decide` against the
  data the emitter records:
  - `Core.lean`: `LoweringArtifact.AllLossesZero` (every recorded obstruction
    loss has `value = 0`), `lossIsObstruction` (the named loss is actually
    tracked), and `CoalgebraExact = AllLossesZero` — each with a decidable
    check and soundness lemma. The emitter now records the diagram's
    obstruction losses on the artifact (default `value = 0`, overridable via
    `render_lean_certificate(D; loss_values=…)`), and emits
    `exportedArtifact_lossesZero`.
  - `Construction.lean`: `CommutingSquare` / `UniversalCone` /
    `UniversalCocone` / `ParallelAgreement` / `QuotientAgreement` are now
    `StructurallyValid ∧ AllLossesZero`, where `StructurallyValid` is a
    decidable check that the construction's projection/injection morphisms,
    shared objects, equalizer/coequalizer maps, and factor namespaces are
    actually declared in the carried diagram.
  - `Energy.lean`: `EnergyDecl.evaluate` returns the carried `value` (not a
    hard-coded `0`); `Compatible` (zero energy) is a falsifiable predicate.
  - `Coalgebra.lean`: removed the `rfl`-on-a-constant
    `bisim_implies_final_eq`; replaced with a falsifiable
    `BisimulationDecl.WellFormed` + soundness lemma.
  - **Teeth, demonstrated:** a certificate emitted with a *nonzero* obstruction
    loss now fails `lake build` (its exactness theorem is unprovable). The
    opt-in `test/test_lean_certificates.jl` asserts both that genuine
    certificates type-check *and* that a nonzero-loss certificate is rejected.
- **Bug fix (found while wiring this up):** `DiagramDecl.declaredRefs` omitted
  obstruction-loss names, so `check`/`WellFormed` was `false` for any diagram
  with a loss-kind port (`db_square`, `jepa_block`, …) — the old test only
  ever exercised `ket_block`, which has no loss port. `DiagramDecl` now
  carries `lossNames`, and the emitter populates it.

### Added
- **Executable energy / cost modules.** The energy-based cost layer
  (`add_energy_function!` / `add_cost_module!`) is no longer
  declaration-only — it now has an execution path:
  - `compute_energies(D, result)` evaluates every declared `EnergyFunction`
    against a run, `evaluate_energy(ef, env)` evaluates one.
  - `evaluate_cost_module` / `compute_costs(D, result)` evaluate the
    `C = Σ uᵢ·ICᵢ + Σ vⱼ·TCⱼ` decomposition, mapping `IntrinsicCost`
    types (`:prediction`/`:reconstruction` → L2 energy; `:variance`/
    `:covariance` → the matching regulariser) to concrete numbers.
    Trainable costs are evaluated when a critic callable is supplied.
  - `run_with_costs(D, inputs)` runs a diagram and returns
    `(result, energies, costs)` in one call.
- **Self-supervised energies implemented.** `energy_vicreg`,
  `energy_barlow_twins`, and `energy_contrastive` (InfoNCE) are now real
  functions and registered in `BUILTIN_ENERGY_FUNCTIONS` (previously these
  energy types were documented but had no implementation).
- **Textbook-grounded CLIFF routing** (`src/cliff_textbook.jl`), porting the
  signature capability of Mahadevan's `CLIFF_CatAgi` — "every route links
  back to the textbook." New `TextbookChapter` type + `CATAGI_TEXTBOOK`
  registry (real *Categories for AGI* chapter titles), with:
  - `recommend_chapters(query)` — rank chapters by thematic overlap.
  - `chapters_for_route` / `chapters_for_primitive` — route/demo linkage
    (each chapter lists the FunctorFlow block macros that ground it).
  - `route_with_textbook(router, query)` — route a query *and* surface the
    backing chapters + runnable demos.
- **`FunctorFlowMakieExt`** (`ext/FunctorFlowMakieExt/`): the Makie
  extension that was declared in `Project.toml` but whose source file did
  **not exist** (a load-time bug when Makie was present) is now
  implemented. `plot_diagram(D)` / `plot_diagram!(ax, D)` render a diagram
  as a layered graph (objects, morphisms, compositions, Σ/Δ Kan
  extensions, obstruction losses) with a legend, using a self-contained
  pure-Julia layout (no Graphs.jl / GraphMakie dependency).

### Fixed
- **`ConsciousFieldOfView` capacity validation was dead code.** The
  `capacity ≥ 1` check lived in an outer constructor that was shadowed by
  the struct's auto-generated `::Int` inner constructor, so
  `ConsciousFieldOfView(0)` silently succeeded. Moved the check into an
  inner constructor. (Found by the new `test_consciousness.jl`.)

### Testing
- `test/runtests.jl` now gates the `test_lux_ext.jl` / `test_lux_training.jl`
  includes behind `Base.find_package("Lux")`, matching how the TinyGrad /
  AgentFramework / Schema / Lean / Makie suites are already gated. The core
  suite now runs to completion in environments without the optional neural
  deps (previously it aborted at the unguarded `using Lux`).
- New test files: `test_energy.jl` (energy math + cost execution, 36
  assertions), `test_consciousness.jl` (conscious-workspace layer, prev.
  untested), `test_cliff_textbook.jl` (textbook routing), and
  `test_makie_ext.jl` (Makie smoke test, gated on a backend). Core suite:
  **790 pass, 0 fail** (plus the gated-skip testsets).

## [0.5.1] — 2026-05-09

### Added
- **JSON-portable ACSet emission** via a new `json_portable=true` keyword
  on `to_acset`. The portable form uses `ShapeType=Vector{Int}` and
  `DTypeType=Symbol`, matching the contract documented on
  `cds_from_json`. This unlocks a direct `Diagram → ACSet → JSON →
  ACSet → Diagram` round-trip via
  `CategoricalDiagramSchema.cds_to_json` / `cds_from_json` without
  needing to hop through `TinyGrad.to_cds_acset`. Default mode
  (`json_portable=false`) is unchanged and backwards-compatible.
  - Symbol-valued user metadata (e.g. `:kind => :skip_connection`) is
    normalised to `String` at write time so that JSON3 round-trip is a
    fixed point (JSON has no Symbol type). The same normalisation is
    applied to `Vector{Symbol}` and `Tuple` values, recursively into
    nested `AbstractDict`s.
- `from_acset` now reads both the default form (`Tuple` shape, `Type`
  dtype) *and* the JSON-portable form (`Vector{Int}` shape, `Symbol`
  dtype) transparently, so consumers don't need to know which form
  produced the ACSet. JP-form dtypes surface through
  `obj.metadata[:dtype]` as a `Symbol` (e.g. `:Float32`).
- New `@testset "json_portable=true round-trip via cds_to_json"` in
  `test/test_schema_roundtrip.jl` exercises both halves of the contract.

## [0.5.0] — 2026-05-09

### Changed (BREAKING)
- **`Catlab` moved from `[deps]` to `[weakdeps]`** to resolve the
  transitive dependency conflict (Catlab → Compose 0.9 → DataStructures
  0.18 vs `TinyGrad.jl` → Symbolics → MultivariatePolynomials → older
  DataStructures) that prevented `using FunctorFlow, TinyGrad` in the
  same Julia session in v0.4.0.
- **Three Catlab-using functions** (`to_presentation`, `to_symbolic`,
  `define_theory`) moved into a new **`FunctorFlowCatlabExt`** extension
  (`ext/FunctorFlowCatlabExt/`). They remain exported as stub generic
  functions; the extension provides their methods. Calling them without
  `using Catlab` now raises a `MethodError`. After `using Catlab` they
  behave identically to v0.4.0.
- The Catlab-using file `src/symbolic_catlab.jl` moved into the new
  extension. The file `src/catlab_interop.jl` was renamed to
  `src/categorical_model.jl` (keeps `CategoricalModelObject`,
  `ModelMorphism`, `NaturalTransformation`, `verify_naturality`,
  `is_natural`, `check_laws`, `register_model!`, `get_model`,
  `MODEL_REGISTRY`, `to_diagram`, `diagram_to_acset`, `acset_to_diagram`,
  and the `compose`/`apply` methods on `ModelMorphism` — all pure Julia,
  no Catlab dependency).

### Removed (BREAKING)
- Re-exports of `nparts`, `subpart`, `add_part!`, and `incident` from
  Catlab.CategoricalAlgebra. Users who relied on these names being
  available via `using FunctorFlow` should now `using Catlab` or
  `using Catlab.CategoricalAlgebra` directly.

### Preserved
- `CategoricalModelObject`, `ModelMorphism`, `NaturalTransformation`
  remain exported from `FunctorFlow` itself (they have no Catlab
  dependency in their definition). Code that pattern-matches on these
  types continues to compile and run with no changes.
- `to_acset`/`from_acset` (provided by `FunctorFlowSchemaExt` when
  `CategoricalDiagramSchema` is loaded) — unchanged.
- The full v0.4.0 test suite passes: 877 pass, 1 broken, 0 failed,
  identical to the v0.4.0 baseline. Tests gated on Catlab availability
  are wrapped in `HAS_CATLAB`/`HAS_CDS` skip-guards so the suite runs
  cleanly in environments without Catlab.

### Verified
- `using FunctorFlow, TinyGrad` now succeeds in the same Julia session
  (was previously blocked by the Compose↔Symbolics conflict via Catlab).
  A small `compile_to_tinygrad` round-trip pipeline executes correctly.
- The deeper conflict introduced by `CategoricalDiagramSchema → Catlab`
  vs `TinyGrad → Symbolics` is **not** resolved by this release: any
  environment that pulls in both CDS (or Catlab directly) and TinyGrad
  still fails to resolve. Resolving that requires CDS to also weak-dep
  Catlab — out of scope here.

### Migration
Add `import Catlab` (or `using Catlab`) to any file that calls
`to_presentation`, `to_symbolic`, `define_theory`, or that uses
the formerly re-exported `nparts`, `subpart`, `add_part!`, `incident`.
The two affected vignettes (`01-getting-started`, `02-dsl-macros`) have
been updated accordingly.

## [0.4.0] — 2026-04-18

### Added
- **`FunctorFlowTinyGradExt` extension** (`ext/FunctorFlowTinyGradExt/`):
  compiles a `Diagram` to one of two TinyGrad-backed engines:
  - `TinyGradBackend` (`mode = :round_trip`) — round-trips Julia arrays
    through `TinyGrad.TinyTensor` for every morphism. Always works,
    regardless of whether reducers/morphisms are opaque Julia callables.
  - `UOpCompiledBackend` (`mode = :uop`) — attempts to trace each
    morphism into the shared TinyGrad UOp DAG. When all ops trace
    cleanly the entire diagram becomes a single fused UOp graph that can
    be re-realised with new inputs without re-walking Julia code (see
    the `compiled.fully_traced` flag). Falls back to opaque per-op
    execution when tracing fails (e.g. for `:ket` reducers operating on
    `Dict`s).
  - Public entry point: `compile_to_tinygrad(D; mode = :round_trip)`
    returns a callable `FFTinyGradModel`. Lower-level constructors
    `tinygrad_backend()` / `uop_compiled_backend()` are also exported.
  - Architectural pattern parity with
    `CatNet.jl/ext/CatNetTinyGradExt`. Together with CDS this completes
    the `CDS ⇄ FF ⇄ CN ⇄ TinyGrad` shared-schema pipeline.
- **`AbstractFunctorFlowBackend`** abstract type plus generic methods
  `lower(backend, D)`, `realize(backend, compiled, inputs)`,
  `backend_name(backend)`, `supports_dtype(backend, T)`. Tagged with
  `ChainRulesCore.@non_differentiable` on the ext-lookup helper to keep
  Zygote from chasing the extension boundary.
- **Vignette 27** (`vignettes/27-tinygrad-backend/`): end-to-end demo of
  both backends, schema round-trip, and an informal performance
  comparison.
- **`test/test_tinygrad_ext.jl`** — 8 testsets covering backend
  metadata, identity diagrams, a 3-layer MLP, UOp full-trace parity,
  opaque fallback, re-run, composition + obstruction loss, and schema
  round-trip. Gated on `using TinyGrad` succeeding; skipped (with a
  banner) in environments where TinyGrad cannot resolve.
- **`test/setup_local_dev.jl`** — convenience script to `Pkg.develop`
  sibling repos (TinyGrad.jl, CategoricalDiagramSchema.jl) into the FF
  test env, mirroring CN's pattern.

### Notes
- TinyGrad is **weakdeps-only**. The standard FF env (no TinyGrad)
  continues to pass cleanly: 877 pass + 1 broken/skipped (the
  TinyGrad ext testset).
- **Known dependency conflict**: TinyGrad's transitive
  `Symbolics.jl 7 → MultivariatePolynomials ≥ 0.5.12` is incompatible
  with FF's `Catlab → Compose / GATlab → DataStructures = "0.18"`
  (whose latest `MultivariatePolynomials` is 0.5.9). Therefore the
  TinyGrad ext cannot be exercised by FF's CI in the standard
  `Pkg.test()` sandbox — users wanting to use the TinyGrad backend must
  build a custom env (FF dev source + TinyGrad without Catlab) or wait
  for upstream `DataStructures = "0.19"` adoption across Compose /
  GATlab / ACSets. Vignette 27 documents the workaround.

## [0.3.3] — 2026-04-17

### Added
- **Self-contained `proofs/` Lake project** (`FunctorFlowProofs`): Lean 4
  schema for the certificates emitted by `render_lean_certificate`,
  `render_construction_certificate`, and `render_jepa_certificate`. No
  Mathlib dependency — defines `OperationKind`, `OperationDecl`,
  `PortDecl`, `DiagramDecl`, `LossDecl`, `LoweringArtifact`
  (with `check`/`Sound`/`sound_of_check`/`lossIsObstruction`/
  `loss_obstruction_of_check`/`CoalgebraExact`/
  `coalgebra_exact_of_zero_loss`), `ConstructionDecl` (six kinds with
  trivial universal-property Props + matching constructors),
  `CoalgebraDecl` / `BisimulationDecl` (with
  `bisim_implies_final_eq`), and `EnergyDecl`
  (with `nonneg_of_standard`).
- **Lean certificate roundtrip test** (`test/test_lean_certificates.jl`):
  opt-in via `FF_LEAN_CI=true`; emits a small diagram + pullback
  certificate and runs `lake build` to verify it type-checks.
- **`Lean` GitHub Actions workflow** (`.github/workflows/lean.yml`):
  builds the bare `proofs/` Lake project on every PR/push touching
  `proofs/`, `src/proof_interface.jl`, or the test/workflow itself, then
  re-runs `Pkg.test()` with `FF_LEAN_CI=true` to verify emitted
  certificates round-trip through `lake build`.

Closes audit P1-FF-3 ("wire emitted Lean certificates into Julia CI").

## [0.3.2] — 2026-04-17

### Added
- **Construction certificates for `Product` / `Coproduct` / `Equalizer` /
  `Coequalizer`**: `render_construction_certificate` now emits full
  `ConstructionDecl`-style stanzas (kind, theorems, witness terms) instead
  of stub strings. Closes audit P1-FF-4.
- **Strict topos internal logic**: `internal_and`, `internal_or`, and
  `internal_not` now raise `ArgumentError` when an operand is missing its
  `characteristic_map`, instead of silently returning `nothing`. Closes
  audit P1-FF-5.
- **Derived summarizer cardinalities**: `summarize_predictive_state_example`
  and `summarize_temporal_repair_example` now derive their per-company /
  per-trajectory counts (`companies`, `years`, `n_local_states`,
  `n_trajectories`, `n_global_sections`) from the example data instead of
  hard-coding constants. Output is bit-identical to the Python parity
  reference. Closes audit P1-FF-6.
- **Training loops in three vignettes**: vignette 07 (DB-square obstruction
  loss, 100 Adam steps), vignette 14 (toy JEPA MSE surrogate, 100 steps),
  and vignette 17 (tiny C-JEPA predictor, 150 steps) now contain explicit
  Adam training cells with initial/final-loss prints and
  `@assert final_loss < initial_loss`. Closes audit P1-FF-7.

- **SCM monomorphism (`build_scm_monomorphism`)**: rewritten from a
  placeholder into the canonical sub-SCM inclusion `M' ↪ M` à la
  Pearl/Bareinboim, with optional variable renaming and a `strict` flag
  (`strict=false` admits soft-intervention sub-SCMs and tags them via
  `metadata[:soft_intervention]`). Closes audit P1-FF-1.

- `src/identifiability.jl`: complete Shpitser-Pearl ID algorithm
  (Algorithm 1 of Shpitser & Pearl, JMLR 2008) for deciding whether a
  causal effect `P(y | do(x))` is identifiable from observational data.
  - New `CausalDAG` type encoding an Acyclic Directed Mixed Graph (ADMG)
    with directed edges and bidirected (latent-confounder) edges.
  - New `IdentifiabilityResult` struct carrying `identifiable::Bool`,
    a symbolic `expression`, optional `Hedge` witness, `failure_reason`,
    and `algorithm` tag.
  - `IDExpression` AST: `Joint`, `CondP`, `Marginal`, `Product`,
    `QFactor` with a `pretty_print` walker.
  - `identify_effect(G, y, x)` runs the algorithm; returns the symbolic
    post-intervention distribution when identifiable, or a hedge
    `(F, F', R)` when not. The algorithm is sound and complete.
  - `is_identifiable(G::CausalDAG, y, x)` is a thin wrapper.
  - Helpers: `ancestors_inclusive`, `c_components`, `subgraph`,
    `remove_incoming`, `topological_order`.
  - `is_backdoor_admissible(G, x, y, Z)` for fast back-door checks.
- `test/test_identifiability.jl`: 109 assertions covering
  back-door admissible, front-door (`X → M → Y` with `X ↔ Y`),
  bow arc (non-identifiable hedge), W-graph (non-identifiable hedge),
  Tian's three-observed/one-hidden example, Pearl's napkin graph,
  sequential do (g-formula), edge cases, and result printing.
- Closes audit P1-FF-2.

### Notes
- Existing `is_identifiable(::CausalDiagram, ::Symbol)` API on top of the
  high-level `CausalDiagram` type is unchanged (still returns the
  `(identifiable=, rule=, reasoning=)` NamedTuple) for backward compat.
  The new `identify_effect` / `is_identifiable(::CausalDAG, ...)` methods
  expose the complete algorithm to users who supply an explicit DAG.

## [0.3.1] — 2026-04-17

### Added
- New end-to-end training test (`test/test_lux_training.jl`, 105 assertions)
  that builds a 2-layer linear `Diagram` (32 → 16 → 4), binds
  `DiagramDenseLayer` morphisms via `compile_to_lux`, runs 100 Adam(1e-2)
  steps with `Optimisers` + `Zygote.gradient`, and asserts that the
  final mean-squared-error loss is less than half of the initial loss.
  This proves that gradients actually flow through the
  `compile_to_lux → LuxDiagramModel` pipeline. Closes audit P0-FF-2.
- `Optimisers` added to `[extras]` and `[targets].test`.

## [0.3.0] — 2026-04-17

### Breaking
- `Lux` and `LuxCore` are now **weak dependencies** (moved from `[deps]` to
  `[weakdeps]`). FunctorFlow no longer pulls a full Lux install at
  precompile time. Users who want the neural backend must add `Lux` and
  `LuxCore` to their own project and `using Lux` (which automatically
  triggers loading of `FunctorFlowLuxExt`).
- All Lux-touching functions (`compile_to_lux`,
  `build_ket_lux_model`, `build_db_lux_model`, `build_gt_lux_model`,
  `build_basket_rocket_lux_model`, `build_topocoend_lux_model`,
  `build_horn_lux_model`, `build_higher_horn_lux_model`,
  `build_bisimulation_quotient_lux_model`,
  `RelationInferenceLayer`, `predict_detach_source`) are now resolved
  through `Base.get_extension` shims in `FunctorFlow`. Calling any of
  them without first `using Lux` raises a clear error.
- The Lux **layer types** (`KETAttentionLayer`, `DiagramDenseLayer`,
  `DiagramChainLayer`, `LuxDiagramModel`) live exclusively inside
  `FunctorFlowLuxExt` and are no longer re-exported from `FunctorFlow`.
  Access them via either of:
  ```julia
  using Lux
  ext = Base.get_extension(FunctorFlow, :FunctorFlowLuxExt)
  layer = ext.KETAttentionLayer(64)
  ```
  or, equivalently, with an `import` from the loaded extension module.
- `FunctorFlowMetalExt` now requires both `Metal` and `Lux` (and
  `LuxCore`) to be loaded simultaneously to activate, since the Metal
  shim itself depends on Lux compatibility.

### Internal
- Deleted `src/lux_layers.jl` (916 LoC). Its contents were a duplicate
  of `ext/FunctorFlowLuxExt/FunctorFlowLuxExt.jl` (687 LoC), with a few
  layers / builders (`RelationInferenceLayer`, the GPU+AD-compatible
  `_ket_attention_forward`, `predict_detach_source`,
  `build_basket_rocket_lux_model`, `build_topocoend_lux_model`,
  `build_horn_lux_model`, `build_higher_horn_lux_model`,
  `build_bisimulation_quotient_lux_model`) only present in `src/`.
  The extension is now the single source of truth and includes all of
  those previously-`src/`-only definitions.

## [0.2.0] — 2024 (unreleased)

### Breaking
- The ACSet schema has moved out of FunctorFlow into its own package,
  [`CategoricalDiagramSchema.jl`](https://github.com/JuliaKnowledge/CategoricalDiagramSchema.jl)
  (UUID `06663149-d6bb-42b5-8a63-d2553351277c`).
- Removed exports: `SchFunctorFlow`, `FunctorFlowGraph`,
  `AbstractFunctorFlowGraph`. There is no in-package replacement; use
  `CategoricalDiagramSchema.SchCategoricalDiagram` /
  `CategoricalDiagramSchema.CategoricalDiagramACSet` instead.
- `to_acset` and `from_acset` now require `using CategoricalDiagramSchema`
  to activate (their methods are provided by the new
  `FunctorFlowSchemaExt` package extension). Calling them without loading
  `CategoricalDiagramSchema` raises a `MethodError`.
- `to_acset` returns a `CategoricalDiagramACSet` (not the old
  `FunctorFlowGraph{Symbol}`).

### Added
- Obstruction losses are now ACSet-native via the `ObsLoss` and `ObsPath`
  parts of `SchCategoricalDiagram`; previously they lived only in a Julia
  side-table.
- Explicit `kan_tgt::Hom(Kan, Node)`. When `add_left_kan!` /
  `add_right_kan!` is called with `target=nothing`, an auto-generated
  `Symbol(name, :_target)` Node is synthesised at ACSet-construction time
  and tagged with `metadata[:auto_kan_target] = true`. The round-trip
  drops this synthetic node and restores `target=nothing`.
- Node `shape`, `dtype`, and arbitrary `metadata` are captured in the
  ACSet representation (`node_shape`, `node_dtype`, `node_metadata`).
- `Composition.chain` is preserved in `edge_metadata[:chain]` for lossless
  round-trip of compositions.
- Edge / Kan / ObsLoss `metadata` slots in the schema, populated from the
  corresponding Julia-level `metadata` dictionaries.

### Migration
1. Add `CategoricalDiagramSchema` to your project:
   ```julia
   using Pkg
   Pkg.add(url = "https://github.com/JuliaKnowledge/CategoricalDiagramSchema.jl")
   ```
2. Replace `using FunctorFlow` with `using FunctorFlow, CategoricalDiagramSchema`
   anywhere you call `to_acset` / `from_acset` / `diagram_to_acset` /
   `acset_to_diagram`.
3. Replace `FunctorFlowGraph{Symbol}()` with
   `CategoricalDiagramSchema.make_diagram()` if you build ACSets by hand.
4. Update tests that introspect ACSet structure: edges that were the
   "loss" edge are now `ObsLoss` parts (with associated `ObsPath`s).

## [0.1.0]
Initial release.
