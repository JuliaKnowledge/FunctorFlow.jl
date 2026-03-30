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

# v1: Categorical foundations
include("catlab_interop.jl")
include("universal.jl")
include("causal.jl")
include("topos.jl")

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
export DemocritusAssemblyConfig, TopoCoendConfig, HornObstructionConfig, BisimulationQuotientConfig

# Block builders
export ket_block, db_square, gt_neighborhood_block, completion_block
export basket_workflow_block, rocket_repair_block
export structured_lm_duality, democritus_gluing_block, basket_rocket_pipeline
export democritus_assembly_pipeline, topocoend_block, horn_fill_block, bisimulation_quotient_block
export MACRO_LIBRARY, build_macro

# Tutorials
export TutorialLibrary, get_tutorial_library, install_tutorial_library!
export build_tutorial_macro, macro_builders
export FOUNDATIONS_TUTORIAL_LIBRARY, PLANNING_TUTORIAL_LIBRARY, UNIFIED_TUTORIAL_LIBRARY

# Proof interface
export diagram_certificate_payload, render_lean_certificate, write_lean_certificate
export render_construction_certificate, render_jepa_certificate

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
export KETAttentionLayer, DiagramDenseLayer, DiagramChainLayer, LuxDiagramModel
export compile_to_lux
export build_ket_lux_model, build_db_lux_model, build_gt_lux_model, build_basket_rocket_lux_model
export predict_detach_source

end # module FunctorFlow
