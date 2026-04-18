"""
    FunctorFlow

A categorical DSL and executable IR for building diagrammatic AI systems,
inspired by the AlgebraicJulia ecosystem (Catlab.jl, ACSets.jl). As of
v0.5.0, Catlab is a *weak* dependency: load it alongside FunctorFlow to
enable `to_presentation`, `to_symbolic`, and `define_theory`.

FunctorFlow lets you build AI systems by categorical construction:
- **Objects** are typed interfaces (token spaces, neighborhoods, plan states)
- **Morphisms** are typed transformations (neural layers, lifts, projections)
- **Diagrams** are architectures assembled from objects, morphisms, Kan extensions, and losses
- **Σ (Left Kan)** is the universal aggregation primitive (attention, pooling, message passing)
- **Δ (Right Kan)** is the universal completion primitive (denoising, repair, reconciliation)
- **Obstruction loss** measures non-commutativity of diagram paths (DB)

The compilation pipeline: `Diagram → Categorical IR → Backend Execution`

## Quick start

```julia
using FunctorFlow

# Build a KET block with the @functorflow macro
D = @functorflow MyKET begin
    Tokens::messages
    Nbrs::relation
    Ctx::contextualized_messages
    embed = Tokens → Ctx
    aggregate = Σ(:Tokens; along=:Nbrs, reducer=:sum)
end

# Or use unicode operators directly
D = Diagram(:MyKET)
add_object!(D, :Tokens; kind=:messages)
add_object!(D, :Nbrs; kind=:relation)
Σ(D, :Tokens; along=:Nbrs, reducer=:sum, name=:aggregate)

# Compile and run
compiled = compile_to_callable(D)
result = run(compiled, Dict(:Tokens => Dict(1=>1.0, 2=>2.0),
                            :Nbrs => Dict("ctx" => [1, 2])))
```
"""
module FunctorFlow

using OrderedCollections: OrderedDict
using JSON3
using ChainRulesCore: ignore_derivatives
import ChainRulesCore
using Random: AbstractRNG

# Core types
include("types.jl")
include("diagram.jl")
include("ports.jl")
include("ir.jl")
include("reducers.jl")
include("compiler.jl")
include("composition.jl")
include("adapters.jl")
include("show.jl")

# ACSet schema stubs (CategoricalDiagramSchema integration via FunctorFlowSchemaExt)
include("schema.jl")

# Catlab-backed stub functions (methods provided by FunctorFlowCatlabExt
# when `using Catlab` is active alongside FunctorFlow). Declared early so
# that downstream files (e.g. `categorical_model.jl`) can reference them.
"""
    to_presentation(D::Diagram) -> Catlab.Presentation

Convert a Diagram into a Catlab Presentation (free category). Method
provided by `FunctorFlowCatlabExt`; requires `using Catlab` alongside
FunctorFlow. Without Catlab loaded, calling this raises a `MethodError`.
"""
function to_presentation end

"""
    to_symbolic(D::Diagram) -> NamedTuple

Convert a Diagram into symbolic Catlab category elements. Method provided
by `FunctorFlowCatlabExt`; requires `using Catlab` alongside FunctorFlow.
"""
function to_symbolic end

"""
    define_theory(objects::AbstractVector; name=:FunctorFlowTheory) -> Catlab.Presentation

Build a Catlab Presentation from CategoricalModelObject instances. Method
provided by `FunctorFlowCatlabExt`; requires `using Catlab`.
"""
function define_theory end

# Unicode operators (after diagram.jl provides add_left_kan! etc.)
# Note: unicode.jl uses compose/product/coproduct which are defined later,
# so we include it after categorical_model.jl and universal.jl

# DSL and block library
include("dsl.jl")
include("block_configs.jl")
include("blocks.jl")
include("tutorials.jl")
include("democritus_examples.jl")
include("topocoend_examples.jl")
include("bisimulation_examples.jl")

# v1: Categorical foundations (CategoricalModelObject + friends; pure Julia,
# no Catlab dep). Catlab-backed methods on `to_presentation`/`to_symbolic`/
# `define_theory` come from `FunctorFlowCatlabExt`.
include("categorical_model.jl")
include("universal.jl")
include("causal.jl")
include("identifiability.jl")
include("topos.jl")
include("scm.jl")
include("psr.jl")
include("persistent_world.jl")
include("workflows.jl")
include("consciousness.jl")
include("evidence_convergence.jl")
include("cliff_router.jl")
include("cliff_artifacts.jl")
include("cliff_runtime.jl")
include("data_bridges.jl")
include("proof_shapes.jl")
include("semantic_compiler.jl")
include("scm_examples.jl")

# v1: Coalgebra & JEPA foundations
include("coalgebra.jl")
include("jepa.jl")
include("energy.jl")

# Proof interface (after universal.jl and coalgebra/jepa so it can reference all types)
include("proof_interface.jl")

# Unicode operators (after everything they depend on is defined)
include("unicode.jl")

# ===== Public API =====

# Enums
export KanDirection, LEFT, RIGHT
export PortDirection, INPUT, OUTPUT, INTERNAL

# Core types
export AbstractFFElement, AbstractFFObject, AbstractFFOperation
export FFObject, Morphism, Composition, KanExtension, ObstructionLoss
export Port, Adapter, IncludedDiagram
export Diagram, DiagramIR, ExecutionResult, CompiledDiagram

# Diagram construction
export add_object!, add_morphism!, compose!, add_left_kan!, add_right_kan!
export add_obstruction_loss!, add!
export bind_morphism!, bind_reducer!, bind_comparator!

# Unicode operators
export Σ, Δ, ⋅, ⊗, ⊕, →
export left_kan, right_kan

# Ports
export expose_port!, get_port

# IR and serialization
export to_ir, as_dict, to_json

# Compiler
export compile_to_callable

# Composition
export include!, object_ref, operation_ref, port_spec

# Adapters
export AdapterSpec, AdapterLibrary, STANDARD_ADAPTER_LIBRARY
export register_adapter!, use_adapter_library!, coerce!
export get_adapter_library

# DSL macros
export @functorflow, @diagram

# Block configs
export KETBlockConfig, DBSquareConfig, GTNeighborhoodConfig
export CompletionBlockConfig, BASKETWorkflowConfig, ROCKETRepairConfig
export StructuredLMDualityConfig, DemocritusGluingConfig, BasketRocketPipelineConfig
export DemocritusAssemblyConfig, TopoCoendConfig, HornObstructionConfig, HigherHornConfig
export BisimulationQuotientConfig

# Block builders
export ket_block, db_square, gt_neighborhood_block, completion_block
export basket_workflow_block, rocket_repair_block
export structured_lm_duality, democritus_gluing_block, basket_rocket_pipeline
export democritus_assembly_pipeline, topocoend_block, horn_fill_block, higher_horn_block
export bisimulation_quotient_block
export democritus_repair_reducer, democritus_claim_distance, build_democritus_restrictor
export build_democritus_assembly_example, execute_democritus_assembly_example
export summarize_democritus_assembly_example
export infer_topocoend_cover, lift_topocoend_scores
export build_topocoend_triage_example, execute_topocoend_triage_example
export summarize_topocoend_triage_example
export build_bisimulation_quotient_example, execute_bisimulation_quotient_example
export summarize_bisimulation_quotient_example
export MACRO_LIBRARY, build_macro

# Tutorials
export TutorialLibrary, get_tutorial_library, install_tutorial_library!
export build_tutorial_macro, macro_builders
export FOUNDATIONS_TUTORIAL_LIBRARY, PLANNING_TUTORIAL_LIBRARY, UNIFIED_TUTORIAL_LIBRARY

# Proof interface
export diagram_certificate_payload, render_lean_certificate, write_lean_certificate
export render_construction_certificate, render_jepa_certificate
export ProofShape, PullbackProofShape, PushoutProofShape, LeftKanProofShape, RightKanProofShape, ProofBundle
export SCMMonomorphismProofShape, SCMCharacteristicMapProofShape
export prove_pullback_shape, prove_pushout_shape, prove_left_kan_shape, prove_right_kan_shape
export prove_scm_monomorphism_shape, prove_scm_characteristic_map_shape, bundle_proof_shapes
export CompilationNode, CompiledArtifact, CompilationPlan
export IRInstruction, IRType, IRTypeComponent, TypedIRValue, ExecutableIR, PlaceholderExecutionResult
export compile_v1, compile_plan, lower_artifact_to_ir, lower_plan_to_executable_ir
export compile_to_executable_ir, execute_placeholder_ir

# ACSet schema and Catlab integration
export to_acset, from_acset, to_presentation, to_symbolic
export diagram_to_acset, acset_to_diagram, define_theory
export verify_naturality
# Note: `nparts`, `subpart`, `add_part!`, `incident` are no longer re-exported
# (v0.5.0 BREAKING). Users wanting them should `using Catlab.CategoricalAlgebra`
# or `using ACSets` directly.

# v1: Catlab interop
export CategoricalModelObject, ModelMorphism, NaturalTransformation
export to_diagram, is_natural, check_laws
export register_model!, get_model
export MODEL_REGISTRY

# v1: Universal constructions
export UniversalConstruction, PullbackResult, PushoutResult
export ProductResult, CoproductResult, EqualizerResult, CoequalizerResult
export pullback, pushout, product, coproduct, equalizer, coequalizer
export verify, compile_construction, universal_morphism

# v1: Causal semantics
export CausalContext, CausalDiagram, build_causal_diagram, causal_transport
export interventional_expectation, is_identifiable
# Shpitser-Pearl identifiability
export CausalDAG, identify_effect, IdentifiabilityResult, Hedge
export IDExpression, Joint, CondP, Marginal, Product, QFactor, pretty_print
export is_backdoor_admissible
export ancestors_inclusive, c_components, subgraph, remove_incoming, topological_order

# v1: Topos foundations
export SubobjectClassifier, SheafSection, SheafCoherenceCheck
export InternalPredicate, build_sheaf_diagram
export check_coherence, evaluate_predicate, classify_subobject
export internal_and, internal_or, internal_not

# v1: SCM-specialized semantics
export SCMLocalFunctionSpec, SCMObjectSpec, SCMModelObject, SCMMorphism
export SCMPredicateClause, SCMMonomorphism, SCMSubobject, SCMPredicate
export SCMTruthValue, OmegaSCM, SCMCharacteristicMap
export validate_scm_spec, local_function_named, local_function_for_target
export build_scm_model_object, build_scm_morphism, scm_to_shared_context, compose_scm_pullback
export default_omega_truth_values, truth_value_named
export build_scm_monomorphism, build_scm_subobject, build_scm_predicate
export build_omega_scm, build_scm_characteristic_map, conjoin_scm_predicates
export scm_subobject_classifier, as_internal_predicate
export build_transport_scm_pullback_example, build_transport_scm_pullback_proof_bundle
export build_transport_scm_pullback_compilation_plan, build_transport_scm_pullback_executable_ir
export execute_transport_scm_pullback_example
export build_transport_scm_predicate_example, build_transport_scm_predicate_compilation_plan
export build_transport_scm_predicate_executable_ir, execute_transport_scm_predicate_example
export build_transport_scm_omega_example, build_transport_scm_omega_compilation_plan
export build_transport_scm_omega_executable_ir, execute_transport_scm_omega_example

# v1: Predictive-state / PSR semantics
export PredictiveContextSpec, PredictiveContext, PredictiveStateSpec, PredictiveStateModelObject
export PredictiveStateTrajectory, PredictiveGlobalSectionSpec, PredictiveGlobalSection
export build_predictive_context, build_predictive_state_model_object, build_predictive_state_trajectory
export build_predictive_global_section, build_predictive_restriction_map, build_predictive_gluing_witness
export build_predictive_state_example, build_predictive_state_compilation_plan
export build_predictive_state_executable_ir, execute_predictive_state_example
export summarize_predictive_state_example

# v1: Persistent-world / temporal semantics
export PersistentStateSpec, PersistentStateModelObject, TemporalBlockSpec, TemporalBlockModel
export PersistentTrajectory, TemporalRepairSpec, TemporalRepair
export ROCKETRefinementSpec, ROCKETRefinement
export EndpointConstraint, SchrodingerBridgeSpec, TemporalSchrodingerBridge
export build_persistent_state_model_object, build_temporal_block_model, build_persistent_trajectory
export build_temporal_repair, build_rocket_refinement, build_temporal_schrodinger_bridge
export build_temporal_repair_example, build_temporal_repair_compilation_plan
export build_temporal_repair_executable_ir, execute_temporal_repair_example
export summarize_temporal_repair_example

# v1: Workflow semantics
export AgenticWorkflowSpec, AgenticWorkflow, ROCKETWorkflowRefinementSpec, ROCKETWorkflowRefinement
export build_agentic_workflow, build_rocket_workflow_refinement
export build_agentic_workflow_example, build_agentic_workflow_compilation_plan
export build_agentic_workflow_executable_ir, execute_agentic_workflow_example
export summarize_agentic_workflow_example

# v1: CLIFF-style orchestration semantics
export UnconsciousProcess, AttentionScoreWeights, ConsciousFieldOfView, BroadcastSelection
export ConsciousWorkspaceState, ConsciousBroadcast, ConsciousBroadcastBoard, ConsciousnessFunctor
export publish!, broadcasts, messages_for_agent, clear_broadcasts!, publish_workspace!, score, competition_for_access
export used_capacity, remaining_capacity
export AbstractEvidenceConvergenceAdapter, FunctionEvidenceConvergenceAdapter
export EvidenceConvergencePolicy, EvidenceConvergenceAssessment, EvidenceConvergenceTracker
export convergence_similarity, convergence_description, assess!, last_assessment
export CLIFFRouteSpec, CLIFFRouteDecision, CLIFFQueryRouter, build_cliff_query_router, route_cliff_query
export looks_like_company_similarity_query, looks_like_culinary_tour_query, looks_like_course_demo_query
export looks_like_product_feedback_query, looks_like_sec_query
export CLIFFAgentSpec, InteractiveCheckpointRequest, SynthesisArtifact, RouteRunResult
export primary_artifact, needs_human_input, selected_agent_names
export route_matches_agent, capability_gap, supports_capabilities, select_agents_for_route
export CLIFFRouteUpdate, CLIFFRuntimeConfig, CLIFFExecutionContext, CLIFFRouteExecutor, CLIFFRuntime, CLIFFRouteTrace
export get_route_state, set_route_state!, route_capability_gap, supports_route_capabilities
export register_route_executor!, get_route_executor, list_route_executors
export execute_cliff_query, execute_cliff_route, latest_workspace, latest_assessment, broadcast_titles
export build_cliff_runtime_example, execute_cliff_runtime_example, execute_cliff_interactive_example
export summarize_cliff_route_trace
export build_cliff_orchestration_example, build_cliff_orchestration_compilation_plan
export build_cliff_orchestration_executable_ir, execute_cliff_orchestration_example
export summarize_cliff_orchestration_example
export agentframework_capabilities, build_agentframework_agent
export build_agentframework_request_event, build_agentframework_checkpoint

# v1: Categorical data bridges
export AtlasFileSet, AtlasSummary, SQLScriptSet, CSQLAtlasStudy
export CSQLTableRef, CSQLObject, CSQLMorphism, CSQLPullbackConstruction, CSQLPushoutConstruction
export CategoricalDBBridge, CSQLTruthWitness, CSQLMaterialization, IntuitionisticDBBridge
export TCCAtlasSpec, TCCEdgeWitness, TCCAtlasProfile
export TCCMethodPullbackWitness, TCCMethodConflictWitness, TCCMethodPullbackSummary
export practical_csql_truth_values
export atlas_pair_study_specs, tcc_atlas_specs, parse_atlas_summary
export locate_named_csql_study, locate_red_wine_csql_study, locate_tylenol_csql_study
export describe_named_csql_study, describe_red_wine_csql_study, describe_tylenol_csql_study
export build_named_csql_categorical_bridge, build_red_wine_csql_categorical_bridge, build_tylenol_csql_categorical_bridge
export materialize_named_csql_results, materialize_red_wine_csql_results, materialize_tylenol_csql_results
export describe_named_csql_materialization, describe_red_wine_csql_materialization, describe_tylenol_csql_materialization
export locate_tcc_atlas, materialize_tcc_atlas_profile, describe_tcc_atlas_profile
export materialize_tcc_method_pullback, describe_tcc_method_pullback
export build_categorical_db_bridge_example, build_intuitionistic_db_bridge_example, build_tcc_examples
export build_data_bridge_compilation_plan, build_data_bridge_executable_ir, execute_data_bridge_example
export summarize_data_bridge_example

# v1: Coalgebra (world models)
export Coalgebra, CoalgebraMorphism, FinalCoalgebraWitness
export Bisimulation, StochasticCoalgebra
export add_coalgebra!, get_coalgebras, add_bisimulation!, get_bisimulations
export coalgebra_residual
export WorldModelConfig, world_model_block

# v1: JEPA (Joint Embedding Predictive Architecture)
export JEPAConfig, HJEPAConfig, KanJEPAConfig
export jepa_block, hjepa_block, kan_jepa_block
export ema_update!

# v1: Energy-based cost module
export EnergyFunction, IntrinsicCost, TrainableCost, CostModule, Configurator
export CollapsePreventionStrategy, EMA_TARGET, CONTRASTIVE, VICREG, BARLOW_TWINS, WHITENING
export add_energy_function!, get_energy_functions
export add_cost_module!, get_cost_modules
export energy_l2, energy_cosine, energy_smooth_l1
export variance_regularization, covariance_regularization
export BUILTIN_ENERGY_FUNCTIONS, BUILTIN_REGULARIZERS
export EnergyBlockConfig, energy_block

# ---------------------------------------------------------------------------
# Lux neural backend — shims that resolve to FunctorFlowLuxExt
#
# Lux and LuxCore are weak dependencies (see Project.toml `[weakdeps]`).
# All neural-layer code lives in `ext/FunctorFlowLuxExt/`. The shims below
# resolve to the extension via `Base.get_extension` and emit a clear error
# when Lux is not loaded.
#
# To use the layer **types** themselves (e.g. `KETAttentionLayer`,
# `DiagramDenseLayer`, `LuxDiagramModel`, `RelationInferenceLayer`,
# `DiagramChainLayer`), do `using Lux` (which triggers ext loading) and
# then access them via `Base.get_extension(FunctorFlow, :FunctorFlowLuxExt)`
# or `using .FunctorFlowLuxExt: KETAttentionLayer`.
# ---------------------------------------------------------------------------

@inline function _lux_ext()
    ext = Base.get_extension(@__MODULE__, :FunctorFlowLuxExt)
    ext === nothing && error(
        "FunctorFlow's Lux backend requires `using Lux` (and `using LuxCore`) " *
        "before any of compile_to_lux, build_*_lux_model, predict_detach_source, " *
        "or RelationInferenceLayer. Lux/LuxCore are weak dependencies as of " *
        "FunctorFlow v0.3.0; add them to your project and `import Lux` first."
    )
    return ext
end

# The extension lookup is metadata-only and must not participate in any AD
# pullback (Zygote cannot differentiate through `Base.get_extension`).
ChainRulesCore.@non_differentiable _lux_ext()

"""
    compile_to_lux(D::Diagram; morphism_layers, reducer_layers, comparator_layers,
                   morphisms, reducers, comparators)

Compile a FunctorFlow `Diagram` to a Lux model. Requires `using Lux` first.
The returned model is a `FunctorFlowLuxExt.LuxDiagramModel` (a
`LuxCore.AbstractLuxLayer`).
"""
compile_to_lux(args...; kwargs...) = _lux_ext().compile_to_lux(args...; kwargs...)

"""
    RelationInferenceLayer(d_model::Int; symmetric=true, name=:infer_relation)

Construct a learnable relation-inference layer. Requires `using Lux` first;
the returned object is a `FunctorFlowLuxExt.RelationInferenceLayer`.
"""
RelationInferenceLayer(args...; kwargs...) = _lux_ext().RelationInferenceLayer(args...; kwargs...)

"""
    predict_detach_source(logits, embedding_weights; position_bias=nothing)

Project logits back into embedding space with stop-gradient semantics. Requires
`using Lux` first.
"""
predict_detach_source(args...; kwargs...) = _lux_ext().predict_detach_source(args...; kwargs...)

# Convenience model builders — each requires `using Lux` first.
build_ket_lux_model(args...; kwargs...) = _lux_ext().build_ket_lux_model(args...; kwargs...)
build_db_lux_model(args...; kwargs...) = _lux_ext().build_db_lux_model(args...; kwargs...)
build_gt_lux_model(args...; kwargs...) = _lux_ext().build_gt_lux_model(args...; kwargs...)
build_basket_rocket_lux_model(args...; kwargs...) = _lux_ext().build_basket_rocket_lux_model(args...; kwargs...)
build_topocoend_lux_model(args...; kwargs...) = _lux_ext().build_topocoend_lux_model(args...; kwargs...)
build_horn_lux_model(args...; kwargs...) = _lux_ext().build_horn_lux_model(args...; kwargs...)
build_higher_horn_lux_model(args...; kwargs...) = _lux_ext().build_higher_horn_lux_model(args...; kwargs...)
build_bisimulation_quotient_lux_model(args...; kwargs...) = _lux_ext().build_bisimulation_quotient_lux_model(args...; kwargs...)

"""
    train_diagram!(model, ps, st, data_loader; optimizer, n_epochs, loss_fn,
                   obstruction_weight, on_step, output_keys)
        -> (ps, st, history)

High-level training loop for a `LuxDiagramModel` that mirrors CatNet.jl's
`train_diagram!` API. Lives in `FunctorFlowLuxTrainExt` and requires
`using Lux, LuxCore, Optimisers, Zygote` to be loaded first.
"""
function train_diagram! end

# Public shim names. Layer *types* (KETAttentionLayer, DiagramDenseLayer,
# DiagramChainLayer, LuxDiagramModel) are NOT re-exported here; access them as
# `FunctorFlowLuxExt.<TypeName>` after `using Lux`.
export compile_to_lux, RelationInferenceLayer, predict_detach_source
export build_ket_lux_model, build_db_lux_model, build_gt_lux_model, build_basket_rocket_lux_model
export build_topocoend_lux_model, build_horn_lux_model, build_higher_horn_lux_model
export build_bisimulation_quotient_lux_model
export train_diagram!

# ---------------------------------------------------------------------------
# Backend abstraction — used by execution backends like TinyGrad.
#
# Introduced in v0.4.0 alongside FunctorFlowTinyGradExt.  Each backend
# implements `lower(backend, diagram, params, inputs)` and
# `realize(backend, lowered, inputs)` plus the metadata helpers
# `backend_name` and `supports_dtype`.  See the extension modules in
# `ext/` for concrete implementations.
# ---------------------------------------------------------------------------

"""Supertype for FunctorFlow execution backends (e.g. TinyGrad)."""
abstract type AbstractFunctorFlowBackend end

"""Lower a `Diagram` to a backend-specific executable representation."""
function lower(backend::AbstractFunctorFlowBackend, diagram, args...; kwargs...)
    error("lower not implemented for $(typeof(backend))")
end

"""Execute a lowered representation on inputs."""
function realize(backend::AbstractFunctorFlowBackend, lowered, inputs)
    error("realize not implemented for $(typeof(backend))")
end

"""Return the string name of the backend."""
backend_name(::AbstractFunctorFlowBackend) = "unknown"

"""Check if a dtype is supported by this backend."""
supports_dtype(::AbstractFunctorFlowBackend, ::Type) = false

export AbstractFunctorFlowBackend, lower, realize, backend_name, supports_dtype

# ---------------------------------------------------------------------------
# TinyGrad neural backend — shims that resolve to FunctorFlowTinyGradExt.
#
# TinyGrad is a weak dependency.  Concrete backend code lives in
# `ext/FunctorFlowTinyGradExt/`.  The shims below resolve via
# `Base.get_extension` once `using TinyGrad` has been called.
# ---------------------------------------------------------------------------

@inline function _tinygrad_ext()
    ext = Base.get_extension(@__MODULE__, :FunctorFlowTinyGradExt)
    ext === nothing && error(
        "FunctorFlow's TinyGrad backend requires `using TinyGrad` before any of " *
        "`compile_to_tinygrad`, `tinygrad_backend`, or `uop_compiled_backend`. " *
        "TinyGrad is a weak dependency as of FunctorFlow v0.4.0; add it to your " *
        "project and `import TinyGrad` first."
    )
    return ext
end

ChainRulesCore.@non_differentiable _tinygrad_ext()

"""
    compile_to_tinygrad(D::Diagram; backend=:array_roundtrip, kwargs...)

Compile a FunctorFlow `Diagram` to a TinyGrad-backed callable model.
Requires `using TinyGrad` first.  See
`FunctorFlowTinyGradExt.compile_to_tinygrad` for full keyword docs.
"""
compile_to_tinygrad(args...; kwargs...) = _tinygrad_ext().compile_to_tinygrad(args...; kwargs...)

"""
    tinygrad_backend() -> TinyGradBackend

Construct the array-round-trip TinyGrad backend.  Requires `using TinyGrad`.
"""
tinygrad_backend(args...; kwargs...) = _tinygrad_ext().TinyGradBackend(args...; kwargs...)

"""
    uop_compiled_backend() -> UOpCompiledBackend

Construct the UOp-compiled TinyGrad backend.  Requires `using TinyGrad`.
This backend traces FunctorFlow morphisms through TinyGrad's lazy tensor
system and falls back to array round-trip for non-traceable morphisms
(dict-based reducers, broadcasted Julia ops, etc.).
"""
uop_compiled_backend(args...; kwargs...) = _tinygrad_ext().UOpCompiledBackend(args...; kwargs...)

export compile_to_tinygrad, tinygrad_backend, uop_compiled_backend

end # module FunctorFlow
