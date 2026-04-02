# ============================================================================
# persistent_world.jl — Persistent-world, temporal repair, and bridge semantics
# ============================================================================

_persistent_state_interfaces() = [
    (:actions, :action_ontology),
    (:relations, :typed_relations),
    (:motifs, :strategic_motifs),
    (:latent, :latent_summary),
    (:evidence, :document_evidence),
]

_temporal_block_interfaces() = [
    (:state_window, :temporal_block),
    (:corruption, :corruption_process),
    (:repair_objective, :denoising_objective),
]

_schrodinger_bridge_interfaces() = [
    (:reference_process, :stochastic_reference_process),
    (:endpoint_constraints, :endpoint_marginals),
    (:bridge_samples, :stochastic_bridge_samples),
    (:evaluation_metrics, :bridge_evaluation_metrics),
]

struct PersistentStateSpec
    name::Symbol
    company::String
    year::Int
    action_ontology::Vector{Symbol}
    relation_types::Vector{Symbol}
    motif_features::Vector{Symbol}
    evidence_channels::Vector{Symbol}
    metadata::Dict{Symbol, Any}
end

function PersistentStateSpec(name, company, year;
                             action_ontology::Vector{Symbol}=Symbol[],
                             relation_types::Vector{Symbol}=Symbol[],
                             motif_features::Vector{Symbol}=[:motif_counts, :latent_embedding],
                             evidence_channels::Vector{Symbol}=[:document_text, :section_metadata],
                             metadata::Dict=Dict{Symbol, Any}())
    PersistentStateSpec(Symbol(name), String(company), Int(year), copy(action_ontology), copy(relation_types),
                        copy(motif_features), copy(evidence_channels),
                        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

struct PersistentStateModelObject
    object::CategoricalModelObject
    company::String
    year::Int
    action_ontology::Vector{Symbol}
    relation_types::Vector{Symbol}
    motif_features::Vector{Symbol}
    evidence_channels::Vector{Symbol}
end

function Base.getproperty(state::PersistentStateModelObject, sym::Symbol)
    if sym === :name
        return getfield(getfield(state, :object), :name)
    elseif sym === :category
        return getfield(getfield(state, :object), :ambient_category)
    elseif sym === :interfaces
        return getfield(getfield(state, :object), :interface_ports)
    elseif sym === :metadata
        return getfield(getfield(state, :object), :metadata)
    else
        return getfield(state, sym)
    end
end

struct TemporalBlockSpec
    name::Symbol
    company::String
    years::Vector{Int}
    block_length::Int
    corruption_modes::Vector{Symbol}
    consistency_channels::Vector{Symbol}
    metadata::Dict{Symbol, Any}
end

function TemporalBlockSpec(name, company, years::Vector{Int};
                           block_length::Union{Nothing, Int}=nothing,
                           corruption_modes::Vector{Symbol}=[:mask_state, :drop_relation, :add_spurious_relation, :perturb_weights],
                           consistency_channels::Vector{Symbol}=[:adjacent_year_compatibility, :same_company_alignment, :cross_year_support],
                           metadata::Dict=Dict{Symbol, Any}())
    _sorted_years(years)
    TemporalBlockSpec(Symbol(name), String(company), copy(years), something(block_length, length(years)),
                      copy(corruption_modes), copy(consistency_channels),
                      Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

struct TemporalBlockModel
    object::CategoricalModelObject
    company::String
    years::Vector{Int}
    block_length::Int
    corruption_modes::Vector{Symbol}
    consistency_channels::Vector{Symbol}
end

function Base.getproperty(block::TemporalBlockModel, sym::Symbol)
    if sym === :name
        return getfield(getfield(block, :object), :name)
    elseif sym === :category
        return getfield(getfield(block, :object), :ambient_category)
    elseif sym === :interfaces
        return getfield(getfield(block, :object), :interface_ports)
    elseif sym === :metadata
        return getfield(getfield(block, :object), :metadata)
    else
        return getfield(block, sym)
    end
end

struct PersistentTrajectory
    functor::ModelMorphism
    company::String
    years::Vector{Int}
    states::Vector{CategoricalModelObject}
    transition_maps::Vector{ModelMorphism}
end

function Base.getproperty(traj::PersistentTrajectory, sym::Symbol)
    if sym === :name
        return getfield(getfield(traj, :functor), :name)
    elseif sym === :source_category
        return getfield(getfield(traj, :functor), :source)
    elseif sym === :target_category
        return getfield(getfield(traj, :functor), :target)
    elseif sym === :metadata
        return getfield(getfield(traj, :functor), :metadata)
    else
        return getfield(traj, sym)
    end
end

struct TemporalRepairSpec
    name::Symbol
    block_length::Int
    corruption_modes::Vector{Symbol}
    consistency_penalties::Vector{Symbol}
    repair_objective::Symbol
    metadata::Dict{Symbol, Any}
end

function TemporalRepairSpec(name;
                            block_length::Int=3,
                            corruption_modes::Vector{Symbol}=[:mask_state, :drop_relation, :add_spurious_relation, :perturb_weights],
                            consistency_penalties::Vector{Symbol}=[:trajectory_drift_penalty, :adjacent_year_support_penalty],
                            repair_objective::Union{Symbol, AbstractString}=:temporal_block_denoising,
                            metadata::Dict=Dict{Symbol, Any}())
    TemporalRepairSpec(Symbol(name), block_length, copy(corruption_modes), copy(consistency_penalties),
                       Symbol(repair_objective),
                       Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

struct TemporalRepair
    name::Symbol
    raw_trajectory::PersistentTrajectory
    repaired_trajectory::PersistentTrajectory
    temporal_block::TemporalBlockModel
    repair_map::NaturalTransformation
    block_length::Int
    corruption_modes::Vector{Symbol}
    consistency_penalties::Vector{Symbol}
    repair_objective::Symbol
    metadata::Dict{Symbol, Any}
end

struct ROCKETRefinementSpec
    name::Symbol
    neighborhood_ops::Vector{Symbol}
    reward_targets::Vector{Symbol}
    retrieval_sources::Vector{Symbol}
    edit_budget::Int
    metadata::Dict{Symbol, Any}
end

function ROCKETRefinementSpec(name;
                              neighborhood_ops::Vector{Symbol}=[:insert_action, :delete_action, :merge_macro_skill, :rewire_relations],
                              reward_targets::Vector{Symbol}=[:next_year_financial_alignment, :trajectory_coherence, :local_evidence_support],
                              retrieval_sources::Vector{Symbol}=[:local_context, :same_company_neighbors, :sector_analogues],
                              edit_budget::Int=12,
                              metadata::Dict=Dict{Symbol, Any}())
    ROCKETRefinementSpec(Symbol(name), copy(neighborhood_ops), copy(reward_targets), copy(retrieval_sources),
                         edit_budget, Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

struct ROCKETRefinement
    refinement::ModelMorphism
    base_state::CategoricalModelObject
    refined_state::CategoricalModelObject
    neighborhood_ops::Vector{Symbol}
    reward_targets::Vector{Symbol}
    retrieval_sources::Vector{Symbol}
    edit_budget::Int
end

function Base.getproperty(refinement::ROCKETRefinement, sym::Symbol)
    if sym === :name
        return getfield(getfield(refinement, :refinement), :name)
    elseif sym === :metadata
        return getfield(getfield(refinement, :refinement), :metadata)
    else
        return getfield(refinement, sym)
    end
end

struct EndpointConstraint
    company::String
    year_from::Int
    year_to::Int
    split::String
    metadata::Dict{Symbol, Any}
end

function EndpointConstraint(company, year_from, year_to, split; metadata::Dict=Dict{Symbol, Any}())
    EndpointConstraint(String(company), Int(year_from), Int(year_to), String(split),
                       Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

struct SchrodingerBridgeSpec
    name::Symbol
    dataset_label::String
    reference_process::String
    solver_family::String
    bridge_method::String
    state_family::String
    conditioning_scope::String
    metadata::Dict{Symbol, Any}
end

function SchrodingerBridgeSpec(name, dataset_label;
                               reference_process="brownian_reference",
                               solver_family="conditional_sde_flow_matching",
                               bridge_method="csb_sde_mean",
                               state_family="company_year_edge_vectors",
                               conditioning_scope="adjacent_year_endpoint_matching",
                               metadata::Dict=Dict{Symbol, Any}())
    SchrodingerBridgeSpec(Symbol(name), String(dataset_label), String(reference_process), String(solver_family),
                          String(bridge_method), String(state_family), String(conditioning_scope),
                          Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

struct TemporalSchrodingerBridge
    object::CategoricalModelObject
    dataset_label::String
    reference_process::String
    solver_family::String
    bridge_method::String
    state_family::String
    conditioning_scope::String
    endpoint_constraints::Vector{EndpointConstraint}
    linked_trajectories::Vector{PersistentTrajectory}
    summary_metrics::Dict{Symbol, Dict{Symbol, Float64}}
    metadata::Dict{Symbol, Any}
end

function Base.getproperty(bridge::TemporalSchrodingerBridge, sym::Symbol)
    if sym === :name
        return getfield(getfield(bridge, :object), :name)
    elseif sym === :category
        return getfield(getfield(bridge, :object), :ambient_category)
    elseif sym === :interfaces
        return getfield(getfield(bridge, :object), :interface_ports)
    else
        return getfield(bridge, sym)
    end
end

function as_model_object(value::PersistentStateModelObject)
    value.object
end

function as_model_object(value::TemporalBlockModel)
    value.object
end

function build_persistent_state_model_object(spec::PersistentStateSpec; category)
    obj = _semantic_object(spec.name;
        category=category,
        interfaces=_persistent_state_interfaces(),
        metadata=merge(Dict{Symbol, Any}(
            :family => :BASKET,
            :semantic_role => :persistent_state,
            :company => spec.company,
            :year => spec.year,
            :action_ontology => copy(spec.action_ontology),
            :relation_types => copy(spec.relation_types),
            :motif_features => copy(spec.motif_features),
            :evidence_channels => copy(spec.evidence_channels),
        ), spec.metadata))
    PersistentStateModelObject(obj, spec.company, spec.year, copy(spec.action_ontology), copy(spec.relation_types),
                               copy(spec.motif_features), copy(spec.evidence_channels))
end

function build_temporal_block_model(spec::TemporalBlockSpec; category)
    obj = _semantic_object(spec.name;
        category=category,
        interfaces=_temporal_block_interfaces(),
        metadata=merge(Dict{Symbol, Any}(
            :family => :BASKET,
            :semantic_role => :temporal_block,
            :company => spec.company,
            :years => copy(spec.years),
            :block_length => spec.block_length,
            :corruption_modes => copy(spec.corruption_modes),
            :consistency_channels => copy(spec.consistency_channels),
        ), spec.metadata))
    TemporalBlockModel(obj, spec.company, copy(spec.years), spec.block_length, copy(spec.corruption_modes), copy(spec.consistency_channels))
end

function build_persistent_trajectory(; name,
                                     company,
                                     years::Vector{Int},
                                     states::Vector,
                                     year_category,
                                     state_category,
                                     transition_maps::Vector{ModelMorphism}=ModelMorphism[],
                                     metadata::Dict=Dict{Symbol, Any}())
    length(years) == length(states) || throw(ArgumentError("Trajectory years and states must have the same length"))
    _sorted_years(years)
    normalized_states = [as_model_object(state) for state in states]
    isempty(transition_maps) || length(transition_maps) == length(normalized_states) - 1 ||
        throw(ArgumentError("Transition maps must connect each adjacent pair of states"))
    for (idx, morphism) in enumerate(transition_maps)
        morphism.source == normalized_states[idx].name || throw(ArgumentError("Transition maps must connect adjacent trajectory states"))
        morphism.target == normalized_states[idx + 1].name || throw(ArgumentError("Transition maps must connect adjacent trajectory states"))
    end
    functor = ModelMorphism(name, year_category, state_category;
        metadata=merge(Dict{Symbol, Any}(
            :family => :BASKET,
            :semantic_role => :persistent_trajectory,
            :company => String(company),
            :years => copy(years),
            :state_names => [state.name for state in normalized_states],
            :transition_maps => [morphism.name for morphism in transition_maps],
        ), Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata)))
    PersistentTrajectory(functor, String(company), copy(years), normalized_states, copy(transition_maps))
end

function build_temporal_repair(spec::TemporalRepairSpec;
                               raw_trajectory::PersistentTrajectory,
                               repaired_trajectory::PersistentTrajectory,
                               temporal_block::TemporalBlockModel)
    raw_trajectory.company == repaired_trajectory.company || throw(ArgumentError("Temporal repair must preserve company identity"))
    raw_trajectory.years == repaired_trajectory.years || throw(ArgumentError("Temporal repair expects aligned year indices"))
    components = Dict{Symbol, Any}()
    for (year, raw_state, repaired_state) in zip(raw_trajectory.years, raw_trajectory.states, repaired_trajectory.states)
        components[Symbol(year)] = ModelMorphism(Symbol(spec.name, :__repair_, year), raw_state.name, repaired_state.name;
            metadata=Dict(:semantic_role => :temporal_repair_component, :company => raw_trajectory.company, :year => year))
    end
    repair_map = NaturalTransformation(Symbol(spec.name, :__repair_map), raw_trajectory.name, repaired_trajectory.name;
        components=components,
        metadata=merge(Dict{Symbol, Any}(
            :family => :BASKET,
            :semantic_role => :temporal_repair_map,
            :block_length => spec.block_length,
            :corruption_modes => copy(spec.corruption_modes),
            :repair_objective => spec.repair_objective,
        ), spec.metadata))
    TemporalRepair(spec.name, raw_trajectory, repaired_trajectory, temporal_block, repair_map,
                   spec.block_length, copy(spec.corruption_modes), copy(spec.consistency_penalties),
                   spec.repair_objective, copy(spec.metadata))
end

function build_rocket_refinement(spec::ROCKETRefinementSpec; base_state, refined_state)
    source = as_model_object(base_state)
    target = as_model_object(refined_state)
    refinement = ModelMorphism(spec.name, source.name, target.name;
        metadata=merge(Dict{Symbol, Any}(
            :family => :ROCKET,
            :semantic_role => :structured_refinement,
            :neighborhood_ops => copy(spec.neighborhood_ops),
            :reward_targets => copy(spec.reward_targets),
            :retrieval_sources => copy(spec.retrieval_sources),
            :edit_budget => spec.edit_budget,
        ), spec.metadata))
    ROCKETRefinement(refinement, source, target, copy(spec.neighborhood_ops), copy(spec.reward_targets),
                     copy(spec.retrieval_sources), spec.edit_budget)
end

function build_temporal_schrodinger_bridge(spec::SchrodingerBridgeSpec;
                                           category,
                                           endpoint_constraints::Vector{EndpointConstraint}=EndpointConstraint[],
                                           linked_trajectories::Vector{PersistentTrajectory}=PersistentTrajectory[],
                                           summary_metrics::Dict{Symbol, Dict{Symbol, Float64}}=Dict{Symbol, Dict{Symbol, Float64}}())
    obj = _semantic_object(spec.name;
        category=category,
        interfaces=_schrodinger_bridge_interfaces(),
        metadata=merge(Dict{Symbol, Any}(
            :family => :SchrodingerBridge,
            :semantic_role => :temporal_trajectory_bridge,
            :dataset_label => spec.dataset_label,
            :reference_process => spec.reference_process,
            :solver_family => spec.solver_family,
            :bridge_method => spec.bridge_method,
            :state_family => spec.state_family,
            :conditioning_scope => spec.conditioning_scope,
            :endpoint_constraint_count => length(endpoint_constraints),
            :linked_trajectory_names => [trajectory.name for trajectory in linked_trajectories],
        ), spec.metadata))
    TemporalSchrodingerBridge(obj, spec.dataset_label, spec.reference_process, spec.solver_family, spec.bridge_method,
                              spec.state_family, spec.conditioning_scope, copy(endpoint_constraints),
                              copy(linked_trajectories), copy(summary_metrics), copy(spec.metadata))
end

function build_temporal_repair_example()
    state_category = :PersistentWorldStates
    year_category = :Years
    block_category = :TemporalBlocks
    bridge_category = :TemporalSchrodingerBridges

    raw_states = Dict(
        2023 => build_persistent_state_model_object(PersistentStateSpec(:AcmeRaw2023, "acme", 2023;
            action_ontology=[:detect, :pilot], relation_types=[:supports, :depends_on]); category=state_category),
        2024 => build_persistent_state_model_object(PersistentStateSpec(:AcmeRaw2024, "acme", 2024;
            action_ontology=[:pilot, :launch], relation_types=[:supports, :conflicts_with]); category=state_category),
        2025 => build_persistent_state_model_object(PersistentStateSpec(:AcmeRaw2025, "acme", 2025;
            action_ontology=[:launch, :expand], relation_types=[:supports, :amplifies]); category=state_category),
    )
    repaired_states = Dict(
        2023 => build_persistent_state_model_object(PersistentStateSpec(:AcmeRepaired2023, "acme", 2023;
            action_ontology=[:detect, :pilot], relation_types=[:supports, :depends_on]); category=state_category),
        2024 => build_persistent_state_model_object(PersistentStateSpec(:AcmeRepaired2024, "acme", 2024;
            action_ontology=[:pilot, :stabilize, :launch], relation_types=[:supports, :depends_on]); category=state_category),
        2025 => build_persistent_state_model_object(PersistentStateSpec(:AcmeRepaired2025, "acme", 2025;
            action_ontology=[:launch, :expand], relation_types=[:supports, :amplifies]); category=state_category),
    )

    raw_transitions = [
        ModelMorphism(:AcmeRaw2023to2024, :AcmeRaw2023, :AcmeRaw2024; metadata=Dict(:year_from => 2023, :year_to => 2024)),
        ModelMorphism(:AcmeRaw2024to2025, :AcmeRaw2024, :AcmeRaw2025; metadata=Dict(:year_from => 2024, :year_to => 2025)),
    ]
    repaired_transitions = [
        ModelMorphism(:AcmeRepaired2023to2024, :AcmeRepaired2023, :AcmeRepaired2024; metadata=Dict(:year_from => 2023, :year_to => 2024)),
        ModelMorphism(:AcmeRepaired2024to2025, :AcmeRepaired2024, :AcmeRepaired2025; metadata=Dict(:year_from => 2024, :year_to => 2025)),
    ]

    raw_traj = build_persistent_trajectory(name=:AcmeRawTrajectory, company="acme", years=[2023, 2024, 2025],
        states=[raw_states[2023], raw_states[2024], raw_states[2025]], year_category=year_category, state_category=state_category,
        transition_maps=raw_transitions)
    repaired_traj = build_persistent_trajectory(name=:AcmeRepairedTrajectory, company="acme", years=[2023, 2024, 2025],
        states=[repaired_states[2023], repaired_states[2024], repaired_states[2025]], year_category=year_category, state_category=state_category,
        transition_maps=repaired_transitions)

    block = build_temporal_block_model(TemporalBlockSpec(:AcmeTemporalBlock, "acme", [2023, 2024, 2025]); category=block_category)
    repair = build_temporal_repair(TemporalRepairSpec(:AcmeTemporalRepair); raw_trajectory=raw_traj,
                                   repaired_trajectory=repaired_traj, temporal_block=block)
    bridge = build_temporal_schrodinger_bridge(
        SchrodingerBridgeSpec(:AcmeTemporalBridge, "synthetic_temporal_panel");
        category=bridge_category,
        endpoint_constraints=[
            EndpointConstraint("acme", 2023, 2024, "train"),
            EndpointConstraint("acme", 2024, 2025, "eval"),
        ],
        linked_trajectories=[raw_traj, repaired_traj],
        summary_metrics=Dict(
            :csb_sde_mean => Dict(:mse => 0.12, :mae => 0.08),
            :conditional_flow => Dict(:mse => 0.09, :mae => 0.06),
        ),
    )

    Dict{Symbol, Any}(
        :raw_states => raw_states,
        :repaired_states => repaired_states,
        :raw_trajectory => raw_traj,
        :repaired_trajectory => repaired_traj,
        :temporal_block => block,
        :temporal_repair => repair,
        :bridge => bridge,
    )
end

function build_temporal_repair_compilation_plan(example::Union{Nothing, Dict{Symbol, Any}}=nothing)
    example = example === nothing ? build_temporal_repair_example() : example
    compile_plan(:TemporalRepairExamplePlan,
        values(example[:raw_states])...,
        values(example[:repaired_states])...,
        example[:raw_trajectory],
        example[:repaired_trajectory],
        example[:temporal_block],
        example[:temporal_repair],
        example[:bridge];
        metadata=Dict(:example => "temporal_repair"))
end

build_temporal_repair_executable_ir(example::Union{Nothing, Dict{Symbol, Any}}=nothing) =
    lower_plan_to_executable_ir(build_temporal_repair_compilation_plan(example))

execute_temporal_repair_example(example::Union{Nothing, Dict{Symbol, Any}}=nothing) =
    execute_placeholder_ir(build_temporal_repair_executable_ir(example))

function summarize_temporal_repair_example(example::Union{Nothing, Dict{Symbol, Any}}=nothing)
    example = example === nothing ? build_temporal_repair_example() : example
    bridge = example[:bridge]
    Dict(
        "counts" => Dict(
            "raw_states" => length(example[:raw_states]),
            "repaired_states" => length(example[:repaired_states]),
            "raw_trajectories" => 1,
            "repaired_trajectories" => 1,
            "temporal_blocks" => 1,
            "temporal_repairs" => 1,
        ),
        "bridge" => Dict(
            "name" => String(bridge.name),
            "dataset_label" => bridge.dataset_label,
            "reference_process" => bridge.reference_process,
            "solver_family" => bridge.solver_family,
            "bridge_method" => bridge.bridge_method,
            "conditioning_scope" => bridge.conditioning_scope,
            "endpoint_constraints" => length(bridge.endpoint_constraints),
            "linked_trajectories" => [String(traj.name) for traj in bridge.linked_trajectories],
            "summary_metrics" => Dict(String(method) => Dict(String(metric) => value for (metric, value) in metrics)
                                      for (method, metrics) in bridge.summary_metrics),
        ),
        "company_summaries" => [
            Dict(
                "company" => "acme",
                "years" => copy(example[:raw_trajectory].years),
                "raw_trajectory" => String(example[:raw_trajectory].name),
                "repaired_trajectory" => String(example[:repaired_trajectory].name),
                "temporal_block" => String(example[:temporal_block].name),
                "temporal_repair" => Dict(
                    "name" => String(example[:temporal_repair].name),
                    "repair_map" => String(example[:temporal_repair].repair_map.name),
                    "component_names" => sort([String(getfield(component, :name)) for component in values(example[:temporal_repair].repair_map.components)]),
                    "repair_objective" => String(example[:temporal_repair].repair_objective),
                ),
            )
        ],
    )
end
