"""
    FunctorFlow

A categorical DSL and executable IR for building diagrammatic AI systems,
built on top of the AlgebraicJulia ecosystem (Catlab.jl, ACSets.jl).

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
using Catlab
using Catlab.Theories: FreeSchema, FreeCategory, Ob, Hom, dom, codom
using Catlab.CategoricalAlgebra
import Catlab.CategoricalAlgebra: nparts, subpart, add_part!, incident
using ChainRulesCore: ignore_derivatives
using Lux
using LuxCore
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

# ACSet schema and Catlab integration
include("schema.jl")

# Unicode operators (after diagram.jl provides add_left_kan! etc.)
# Note: unicode.jl uses compose/product/coproduct which are defined later,
# so we include it after catlab_interop.jl and universal.jl

# DSL and block library
include("dsl.jl")
include("block_configs.jl")
include("blocks.jl")
include("tutorials.jl")
include("democritus_examples.jl")
include("topocoend_examples.jl")
include("bisimulation_examples.jl")

# v1: Categorical foundations
include("catlab_interop.jl")
include("universal.jl")
include("causal.jl")
include("topos.jl")
include("scm.jl")
include("psr.jl")
include("persistent_world.jl")
include("workflows.jl")
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
export SchFunctorFlow, FunctorFlowGraph, AbstractFunctorFlowGraph
export to_acset, from_acset, to_presentation, to_symbolic
export diagram_to_acset, acset_to_diagram, define_theory
export verify_naturality
# Re-export key Catlab ACSet functions for convenience
export nparts, subpart, add_part!, incident

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

# Lux neural backend (layers, compile_to_lux, KETAttentionLayer)
include("lux_layers.jl")

# Lux layer exports
export KETAttentionLayer, DiagramDenseLayer, DiagramChainLayer, RelationInferenceLayer, LuxDiagramModel
export compile_to_lux
export build_ket_lux_model, build_db_lux_model, build_gt_lux_model, build_basket_rocket_lux_model
export build_topocoend_lux_model, build_horn_lux_model, build_higher_horn_lux_model
export build_bisimulation_quotient_lux_model
export predict_detach_source

end # module FunctorFlow
