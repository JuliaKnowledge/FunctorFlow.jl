# Changelog

All notable changes to FunctorFlow.jl are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
