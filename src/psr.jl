# ============================================================================
# psr.jl — Predictive-state / PSR semantics
# ============================================================================

_semantic_ports(specs::Vector{Tuple{Symbol, Symbol}}) =
    [Port(name, name; kind=kind, port_type=kind, direction=OUTPUT) for (name, kind) in specs]

function _semantic_object(name; category, interfaces::Vector{Tuple{Symbol, Symbol}}, metadata::Dict=Dict{Symbol, Any}())
    CategoricalModelObject(name;
        ambient_category=category,
        interface_ports=_semantic_ports(interfaces),
        metadata=Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

_sorted_years(years::Vector{Int}) = issorted(years) || throw(ArgumentError("Years must be sorted in increasing order"))

_psr_context_interfaces() = [
    (:actions, :context_action_alphabet),
    (:observations, :context_observation_alphabet),
    (:tests, :predictive_test_catalog),
]

_psr_state_interfaces() = [
    (:context_slice, :psr_context_slice),
    (:tests, :supported_predictive_tests),
    (:observations, :observation_heads),
    (:evidence, :workflow_outcome_evidence),
]

_psr_global_section_interfaces() = [
    (:contexts, :glued_context_family),
    (:section, :global_predictive_section),
    (:certificate, :gluing_certificate),
]

struct PredictiveContextSpec
    name::Symbol
    context_id::String
    label::String
    action_alphabet::Vector{Symbol}
    observation_alphabet::Vector{Symbol}
    metadata::Dict{Symbol, Any}
end

function PredictiveContextSpec(name, context_id, label;
                               action_alphabet::Vector{Symbol}=Symbol[],
                               observation_alphabet::Vector{Symbol}=Symbol[],
                               metadata::Dict=Dict{Symbol, Any}())
    PredictiveContextSpec(Symbol(name), String(context_id), String(label),
                          copy(action_alphabet), copy(observation_alphabet),
                          Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

struct PredictiveContext
    object::CategoricalModelObject
    context_id::String
    label::String
    action_alphabet::Vector{Symbol}
    observation_alphabet::Vector{Symbol}
end

function Base.getproperty(ctx::PredictiveContext, sym::Symbol)
    if sym === :name
        return getfield(getfield(ctx, :object), :name)
    elseif sym === :category
        return getfield(getfield(ctx, :object), :ambient_category)
    elseif sym === :metadata
        return getfield(getfield(ctx, :object), :metadata)
    else
        return getfield(ctx, sym)
    end
end

struct PredictiveStateSpec
    name::Symbol
    company::String
    year::Int
    context_id::String
    context_label::String
    sector::String
    predictive_tests::Vector{Symbol}
    observation_heads::Vector{Symbol}
    evidence_channels::Vector{Symbol}
    metadata::Dict{Symbol, Any}
end

function PredictiveStateSpec(name, company, year, context_id, context_label, sector;
                             predictive_tests::Vector{Symbol}=Symbol[],
                             observation_heads::Vector{Symbol}=Symbol[],
                             evidence_channels::Vector{Symbol}=[:workflow_extractions, :rocket_selected_plans, :financial_panel_outcomes],
                             metadata::Dict=Dict{Symbol, Any}())
    PredictiveStateSpec(Symbol(name), String(company), Int(year), String(context_id), String(context_label), String(sector),
                        copy(predictive_tests), copy(observation_heads), copy(evidence_channels),
                        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

struct PredictiveStateModelObject
    object::CategoricalModelObject
    company::String
    year::Int
    context_id::String
    context_label::String
    sector::String
    predictive_tests::Vector{Symbol}
    observation_heads::Vector{Symbol}
    evidence_channels::Vector{Symbol}
end

function Base.getproperty(state::PredictiveStateModelObject, sym::Symbol)
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

struct PredictiveStateTrajectory
    functor::ModelMorphism
    company::String
    context_id::String
    years::Vector{Int}
    states::Vector{CategoricalModelObject}
    transition_maps::Vector{ModelMorphism}
end

function Base.getproperty(traj::PredictiveStateTrajectory, sym::Symbol)
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

struct PredictiveGlobalSectionSpec
    name::Symbol
    company::String
    year::Int
    context_ids::Vector{String}
    predictive_tests::Vector{Symbol}
    metadata::Dict{Symbol, Any}
end

function PredictiveGlobalSectionSpec(name, company, year;
                                     context_ids::Vector{String}=String[],
                                     predictive_tests::Vector{Symbol}=Symbol[],
                                     metadata::Dict=Dict{Symbol, Any}())
    PredictiveGlobalSectionSpec(Symbol(name), String(company), Int(year), copy(context_ids), copy(predictive_tests),
                                Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

struct PredictiveGlobalSection
    object::CategoricalModelObject
    company::String
    year::Int
    context_ids::Vector{String}
    predictive_tests::Vector{Symbol}
end

function Base.getproperty(section::PredictiveGlobalSection, sym::Symbol)
    if sym === :name
        return getfield(getfield(section, :object), :name)
    elseif sym === :category
        return getfield(getfield(section, :object), :ambient_category)
    elseif sym === :interfaces
        return getfield(getfield(section, :object), :interface_ports)
    elseif sym === :metadata
        return getfield(getfield(section, :object), :metadata)
    else
        return getfield(section, sym)
    end
end

function build_predictive_context(spec::PredictiveContextSpec; category)
    obj = _semantic_object(spec.name;
        category=category,
        interfaces=_psr_context_interfaces(),
        metadata=merge(Dict{Symbol, Any}(
            :family => :PSR,
            :semantic_role => :predictive_context,
            :context_id => spec.context_id,
            :context_label => spec.label,
            :action_alphabet => copy(spec.action_alphabet),
            :observation_alphabet => copy(spec.observation_alphabet),
        ), spec.metadata))
    PredictiveContext(obj, spec.context_id, spec.label, copy(spec.action_alphabet), copy(spec.observation_alphabet))
end

function build_predictive_state_model_object(spec::PredictiveStateSpec; category)
    obj = _semantic_object(spec.name;
        category=category,
        interfaces=_psr_state_interfaces(),
        metadata=merge(Dict{Symbol, Any}(
            :family => :PSR,
            :semantic_role => :predictive_state,
            :company => spec.company,
            :year => spec.year,
            :context_id => spec.context_id,
            :context_label => spec.context_label,
            :sector => spec.sector,
            :predictive_tests => copy(spec.predictive_tests),
            :observation_heads => copy(spec.observation_heads),
            :evidence_channels => copy(spec.evidence_channels),
        ), spec.metadata))
    PredictiveStateModelObject(obj, spec.company, spec.year, spec.context_id, spec.context_label, spec.sector,
                               copy(spec.predictive_tests), copy(spec.observation_heads), copy(spec.evidence_channels))
end

function build_predictive_state_trajectory(name;
                                           company,
                                           context_id,
                                           years::Vector{Int},
                                           states::Vector{PredictiveStateModelObject},
                                           year_category,
                                           state_category,
                                           transition_maps::Vector{ModelMorphism}=ModelMorphism[],
                                           metadata::Dict=Dict{Symbol, Any}())
    length(years) == length(states) || throw(ArgumentError("Predictive-state trajectory years and states must have the same length"))
    _sorted_years(years)
    isempty(transition_maps) || length(transition_maps) == length(states) - 1 ||
        throw(ArgumentError("Predictive-state transition maps must connect each adjacent pair of states"))
    for (idx, morphism) in enumerate(transition_maps)
        morphism.source == states[idx].name || throw(ArgumentError("Predictive-state transition maps must connect adjacent trajectory states"))
        morphism.target == states[idx + 1].name || throw(ArgumentError("Predictive-state transition maps must connect adjacent trajectory states"))
    end
    functor = ModelMorphism(name, year_category, state_category;
        metadata=merge(Dict{Symbol, Any}(
            :family => :PSR,
            :semantic_role => :predictive_state_trajectory,
            :company => String(company),
            :context_id => String(context_id),
            :years => copy(years),
            :state_names => [state.name for state in states],
            :transition_maps => [morphism.name for morphism in transition_maps],
        ), Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata)))
    PredictiveStateTrajectory(functor, String(company), String(context_id), copy(years),
                              [state.object for state in states], copy(transition_maps))
end

function build_predictive_global_section(spec::PredictiveGlobalSectionSpec; category)
    obj = _semantic_object(spec.name;
        category=category,
        interfaces=_psr_global_section_interfaces(),
        metadata=merge(Dict{Symbol, Any}(
            :family => :PSR,
            :semantic_role => :predictive_global_section,
            :company => spec.company,
            :year => spec.year,
            :context_ids => copy(spec.context_ids),
            :predictive_tests => copy(spec.predictive_tests),
        ), spec.metadata))
    PredictiveGlobalSection(obj, spec.company, spec.year, copy(spec.context_ids), copy(spec.predictive_tests))
end

function build_predictive_restriction_map(name,
                                          source_context::PredictiveContext,
                                          target_context::PredictiveContext;
                                          shared_tests::Vector{Symbol}=Symbol[],
                                          metadata::Dict=Dict{Symbol, Any}())
    ModelMorphism(name, source_context.name, target_context.name;
        metadata=merge(Dict{Symbol, Any}(
            :family => :PSR,
            :semantic_role => :predictive_restriction_map,
            :source_context_id => source_context.context_id,
            :target_context_id => target_context.context_id,
            :shared_tests => copy(shared_tests),
        ), Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata)))
end

function build_predictive_gluing_witness(name, company, year,
                                         left_context::PredictiveContext,
                                         right_context::PredictiveContext;
                                         predictive_tests::Vector{Symbol}=Symbol[],
                                         category=:PredictiveStateGluingWitnesses,
                                         metadata::Dict=Dict{Symbol, Any}())
    _semantic_object(name;
        category=category,
        interfaces=[(:contexts, :glued_context_family), (:witness, :gluing_witness)],
        metadata=merge(Dict{Symbol, Any}(
            :family => :PSR,
            :semantic_role => :predictive_state_gluing_witness,
            :company => String(company),
            :year => Int(year),
            :left_context_id => left_context.context_id,
            :right_context_id => right_context.context_id,
            :predictive_tests => copy(predictive_tests),
        ), Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata)))
end

function build_predictive_state_example()
    context_category = :PredictiveContexts
    state_category = :CompanyLocalPredictiveStates
    year_category = :Years
    section_category = :CompanyPredictiveGlobalSections

    market = build_predictive_context(
        PredictiveContextSpec(:MarketContext, "market", "Market conditions";
            action_alphabet=[:launch, :price_cut],
            observation_alphabet=[:growth, :retention],
            metadata=Dict(:test_count => 2));
        category=context_category)
    product = build_predictive_context(
        PredictiveContextSpec(:ProductContext, "product", "Product operations";
            action_alphabet=[:pilot, :expand],
            observation_alphabet=[:quality, :satisfaction],
            metadata=Dict(:test_count => 2));
        category=context_category)

    state_specs = [
        PredictiveStateSpec(:Acme2023MarketPSR, "acme", 2023, "market", "Market conditions", "software";
            predictive_tests=[:launch_growth, :pricing_retention],
            observation_heads=[:growth, :retention]),
        PredictiveStateSpec(:Acme2024MarketPSR, "acme", 2024, "market", "Market conditions", "software";
            predictive_tests=[:launch_growth, :pricing_retention],
            observation_heads=[:growth, :retention]),
        PredictiveStateSpec(:Acme2023ProductPSR, "acme", 2023, "product", "Product operations", "software";
            predictive_tests=[:pilot_quality, :expand_satisfaction],
            observation_heads=[:quality, :satisfaction]),
        PredictiveStateSpec(:Acme2024ProductPSR, "acme", 2024, "product", "Product operations", "software";
            predictive_tests=[:pilot_quality, :expand_satisfaction],
            observation_heads=[:quality, :satisfaction]),
    ]
    states = Dict(spec.name => build_predictive_state_model_object(spec; category=state_category) for spec in state_specs)

    market_transitions = [
        ModelMorphism(:AcmeMarketTransition, :Acme2023MarketPSR, :Acme2024MarketPSR;
                      metadata=Dict(:semantic_role => :predictive_state_transition, :company => "acme", :context_id => "market"))
    ]
    product_transitions = [
        ModelMorphism(:AcmeProductTransition, :Acme2023ProductPSR, :Acme2024ProductPSR;
                      metadata=Dict(:semantic_role => :predictive_state_transition, :company => "acme", :context_id => "product"))
    ]
    market_traj = build_predictive_state_trajectory(:AcmeMarketTrajectory;
        company="acme", context_id="market",
        years=[2023, 2024],
        states=[states[:Acme2023MarketPSR], states[:Acme2024MarketPSR]],
        year_category=year_category, state_category=state_category,
        transition_maps=market_transitions)
    product_traj = build_predictive_state_trajectory(:AcmeProductTrajectory;
        company="acme", context_id="product",
        years=[2023, 2024],
        states=[states[:Acme2023ProductPSR], states[:Acme2024ProductPSR]],
        year_category=year_category, state_category=state_category,
        transition_maps=product_transitions)

    restriction = build_predictive_restriction_map(:MarketToProductRestriction, market, product;
        shared_tests=[:shared_demand_signal])
    global_2023 = build_predictive_global_section(
        PredictiveGlobalSectionSpec(:Acme2023GlobalPSR, "acme", 2023;
            context_ids=["market", "product"],
            predictive_tests=[:launch_growth, :pilot_quality],
            metadata=Dict(:pairwise_glueable => true));
        category=section_category)
    global_2024 = build_predictive_global_section(
        PredictiveGlobalSectionSpec(:Acme2024GlobalPSR, "acme", 2024;
            context_ids=["market", "product"],
            predictive_tests=[:pricing_retention, :expand_satisfaction],
            metadata=Dict(:pairwise_glueable => true));
        category=section_category)
    witness = build_predictive_gluing_witness(:Acme2024PSRWitness, "acme", 2024, market, product;
        predictive_tests=[:pricing_retention, :expand_satisfaction])

    Dict{Symbol, Any}(
        :contexts => Dict("market" => market, "product" => product),
        :local_predictive_states => Dict(
            ("acme", 2023, "market") => states[:Acme2023MarketPSR],
            ("acme", 2024, "market") => states[:Acme2024MarketPSR],
            ("acme", 2023, "product") => states[:Acme2023ProductPSR],
            ("acme", 2024, "product") => states[:Acme2024ProductPSR],
        ),
        :predictive_state_trajectories => Dict(
            ("acme", "market") => market_traj,
            ("acme", "product") => product_traj,
        ),
        :restriction_maps => Dict(("market", "product") => restriction),
        :global_sections => Dict(("acme", 2023) => global_2023, ("acme", 2024) => global_2024),
        :gluing_witnesses => Dict(("acme", 2024, "market", "product") => witness),
        :metadata => Dict(
            :companies => ["acme"],
            :context_ids => ["market", "product"],
        ),
    )
end

function build_predictive_state_compilation_plan(example::Union{Nothing, Dict{Symbol, Any}}=nothing; include_gluing_witnesses::Bool=false)
    example = example === nothing ? build_predictive_state_example() : example
    subjects = Any[]
    append!(subjects, values(example[:contexts]))
    append!(subjects, values(example[:local_predictive_states]))
    append!(subjects, values(example[:predictive_state_trajectories]))
    append!(subjects, values(example[:restriction_maps]))
    append!(subjects, values(example[:global_sections]))
    include_gluing_witnesses && append!(subjects, values(example[:gluing_witnesses]))
    compile_plan(:PredictiveStateExamplePlan, subjects...;
                 metadata=Dict(:example => "predictive_state", :include_gluing_witnesses => include_gluing_witnesses))
end

build_predictive_state_executable_ir(example::Union{Nothing, Dict{Symbol, Any}}=nothing; include_gluing_witnesses::Bool=false) =
    lower_plan_to_executable_ir(build_predictive_state_compilation_plan(example; include_gluing_witnesses=include_gluing_witnesses))

execute_predictive_state_example(example::Union{Nothing, Dict{Symbol, Any}}=nothing; include_gluing_witnesses::Bool=false) =
    execute_placeholder_ir(build_predictive_state_executable_ir(example; include_gluing_witnesses=include_gluing_witnesses))

function summarize_predictive_state_example(example::Union{Nothing, Dict{Symbol, Any}}=nothing)
    example = example === nothing ? build_predictive_state_example() : example
    Dict(
        "companies" => copy(example[:metadata][:companies]),
        "context_ids" => copy(example[:metadata][:context_ids]),
        "counts" => Dict(
            "contexts" => length(example[:contexts]),
            "local_predictive_states" => length(example[:local_predictive_states]),
            "predictive_state_trajectories" => length(example[:predictive_state_trajectories]),
            "restriction_maps" => length(example[:restriction_maps]),
            "global_sections" => length(example[:global_sections]),
            "gluing_witnesses" => length(example[:gluing_witnesses]),
        ),
        "context_summaries" => [
            Dict(
                "context_id" => ctx.context_id,
                "label" => ctx.label,
                "action_alphabet" => String.(ctx.action_alphabet),
                "observation_alphabet" => String.(ctx.observation_alphabet),
                "test_count" => Int(get(ctx.metadata, :test_count, 0)),
            )
            for ctx in values(example[:contexts])
        ],
        "company_summaries" => [
            Dict(
                "company" => "acme",
                "years" => [2023, 2024],
                "n_local_states" => 4,
                "n_trajectories" => 2,
                "n_global_sections" => 2,
                "all_global_sections_glueable" => all(get(section.metadata, :pairwise_glueable, false) for section in values(example[:global_sections])),
            )
        ],
    )
end
