# Changelog

All notable changes to FunctorFlow.jl are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
