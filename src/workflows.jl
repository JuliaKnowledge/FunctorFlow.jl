# ============================================================================
# workflows.jl — Agentic workflow semantics
# ============================================================================

_workflow_interfaces() = [
    (:plan_path, :agentic_workflow),
    (:control, :workflow_control),
    (:reward, :workflow_reward),
    (:evidence, :workflow_evidence),
]

_workflow_step_interfaces() = [
    (:prefix, :plan_prefix),
    (:frontier, :next_action_frontier),
    (:evidence, :statement_evidence),
]

struct AgenticWorkflowSpec
    name::Symbol
    company::String
    year::Int
    statement_id::String
    actions::Vector{String}
    edges::Vector{Tuple{String, String}}
    action_types::Vector{Tuple{String, String}}
    evidence_channels::Vector{String}
    stage::String
    metadata::Dict{Symbol, Any}
end

function AgenticWorkflowSpec(name, company, year, statement_id, actions::Vector{String};
                             edges::Vector{Tuple{String, String}}=Tuple{String, String}[],
                             action_types::Vector{Tuple{String, String}}=Tuple{String, String}[],
                             evidence_channels::Vector{String}=["statement_text", "workflow_extraction", "reranking_scores"],
                             stage="basket_extracted",
                             metadata::Dict=Dict{Symbol, Any}())
    AgenticWorkflowSpec(Symbol(name), String(company), Int(year), String(statement_id), copy(actions), copy(edges),
                        copy(action_types), copy(evidence_channels), String(stage),
                        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

struct AgenticWorkflow
    object::CategoricalModelObject
    functor::ModelMorphism
    company::String
    year::Int
    statement_id::String
    actions::Vector{String}
    edges::Vector{Tuple{String, String}}
    action_types::Vector{Tuple{String, String}}
    step_states::Vector{CategoricalModelObject}
    action_transitions::Vector{ModelMorphism}
    evidence_channels::Vector{String}
    stage::String
end

function Base.getproperty(workflow::AgenticWorkflow, sym::Symbol)
    if sym === :name
        return getfield(getfield(workflow, :object), :name)
    elseif sym === :category
        return getfield(getfield(workflow, :object), :ambient_category)
    elseif sym === :source_category
        return getfield(getfield(workflow, :functor), :source)
    elseif sym === :target_category
        return getfield(getfield(workflow, :functor), :target)
    elseif sym === :metadata
        return getfield(getfield(workflow, :object), :metadata)
    else
        return getfield(workflow, sym)
    end
end

struct ROCKETWorkflowRefinementSpec
    name::Symbol
    reward_mode::String
    reward_targets::Vector{String}
    neighborhood_sources::Vector{String}
    candidate_budget::Int
    metadata::Dict{Symbol, Any}
end

function ROCKETWorkflowRefinementSpec(name;
                                      reward_mode="financial",
                                      reward_targets::Vector{String}=["financial_alignment", "workflow_coherence", "local_evidence_support"],
                                      neighborhood_sources::Vector{String}=["basket", "local_insert", "local_merge", "macro_merge", "sector_merge"],
                                      candidate_budget::Int=12,
                                      metadata::Dict=Dict{Symbol, Any}())
    ROCKETWorkflowRefinementSpec(Symbol(name), String(reward_mode), copy(reward_targets),
                                 copy(neighborhood_sources), candidate_budget,
                                 Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

struct ROCKETWorkflowRefinement
    refinement::ModelMorphism
    base_workflow::AgenticWorkflow
    refined_workflow::AgenticWorkflow
    reward_mode::String
    reward_targets::Vector{String}
    neighborhood_sources::Vector{String}
    candidate_budget::Int
end

function Base.getproperty(refinement::ROCKETWorkflowRefinement, sym::Symbol)
    if sym === :name
        return getfield(getfield(refinement, :refinement), :name)
    elseif sym === :metadata
        return getfield(getfield(refinement, :refinement), :metadata)
    else
        return getfield(refinement, sym)
    end
end

_step_state_name(workflow_name::Symbol, step_index::Int) = Symbol(workflow_name, :__step_, lpad(step_index, 2, '0'))
_transition_name(workflow_name::Symbol, step_index::Int, action::String) = Symbol(workflow_name, :__action_, lpad(step_index, 2, '0'), :_, replace(action, ' ' => '_'))

function build_agentic_workflow(spec::AgenticWorkflowSpec;
                                workflow_category,
                                step_index_category,
                                plan_state_category)
    step_states = [
        _semantic_object(_step_state_name(spec.name, step_index);
            category=plan_state_category,
            interfaces=_workflow_step_interfaces(),
            metadata=merge(Dict{Symbol, Any}(
                :family => :BASKET,
                :semantic_role => :workflow_plan_state,
                :workflow_name => spec.name,
                :company => spec.company,
                :year => spec.year,
                :statement_id => spec.statement_id,
                :step_index => step_index,
                :action_prefix => spec.actions[1:step_index],
                :stage => spec.stage,
            ), spec.metadata))
        for step_index in 0:length(spec.actions)
    ]
    transitions = [
        ModelMorphism(_transition_name(spec.name, idx - 1, action), step_states[idx].name, step_states[idx + 1].name;
            metadata=Dict(
                :family => :BASKET,
                :semantic_role => :workflow_action_transition,
                :workflow_name => spec.name,
                :company => spec.company,
                :year => spec.year,
                :statement_id => spec.statement_id,
                :step_index => idx - 1,
                :action => action,
                :stage => spec.stage,
            ))
        for (idx, action) in enumerate(spec.actions)
    ]
    object = _semantic_object(spec.name;
        category=workflow_category,
        interfaces=_workflow_interfaces(),
        metadata=merge(Dict{Symbol, Any}(
            :family => :BASKET,
            :semantic_role => :agentic_workflow,
            :company => spec.company,
            :year => spec.year,
            :statement_id => spec.statement_id,
            :actions => copy(spec.actions),
            :edges => copy(spec.edges),
            :action_types => copy(spec.action_types),
            :evidence_channels => copy(spec.evidence_channels),
            :stage => spec.stage,
            :n_steps => length(spec.actions),
        ), spec.metadata))
    functor = ModelMorphism(Symbol(spec.name, :__functor), step_index_category, plan_state_category;
        metadata=merge(Dict{Symbol, Any}(
            :family => :BASKET,
            :semantic_role => :workflow_functor,
            :company => spec.company,
            :year => spec.year,
            :statement_id => spec.statement_id,
            :stage => spec.stage,
            :object_mapping => [(step_index, state.name) for (step_index, state) in enumerate(step_states)],
            :transition_maps => [transition.name for transition in transitions],
            :actions => copy(spec.actions),
            :edges => copy(spec.edges),
        ), spec.metadata))
    AgenticWorkflow(object, functor, spec.company, spec.year, spec.statement_id, copy(spec.actions), copy(spec.edges),
                    copy(spec.action_types), step_states, transitions, copy(spec.evidence_channels), spec.stage)
end

function build_rocket_workflow_refinement(spec::ROCKETWorkflowRefinementSpec;
                                          base_workflow::AgenticWorkflow,
                                          refined_workflow::AgenticWorkflow)
    refinement = ModelMorphism(spec.name, base_workflow.name, refined_workflow.name;
        metadata=merge(Dict{Symbol, Any}(
            :family => :ROCKET,
            :semantic_role => :workflow_refinement,
            :reward_mode => spec.reward_mode,
            :reward_targets => copy(spec.reward_targets),
            :neighborhood_sources => copy(spec.neighborhood_sources),
            :candidate_budget => spec.candidate_budget,
            :company => base_workflow.company,
            :year => base_workflow.year,
            :statement_id => base_workflow.statement_id,
        ), spec.metadata))
    ROCKETWorkflowRefinement(refinement, base_workflow, refined_workflow, spec.reward_mode,
                             copy(spec.reward_targets), copy(spec.neighborhood_sources), spec.candidate_budget)
end

function build_agentic_workflow_example()
    workflow_category = :AgenticWorkflows
    step_index_category = :WorkflowStepIndex
    plan_state_category = :WorkflowPlanStates

    raw = build_agentic_workflow(
        AgenticWorkflowSpec(:AcmeWorkflow, "acme", 2024, "stmt-001", ["detect", "prioritize", "launch"];
            edges=[("detect", "prioritize"), ("prioritize", "launch")],
            action_types=[("detect", "analysis"), ("prioritize", "ranking"), ("launch", "execution")]);
        workflow_category=workflow_category,
        step_index_category=step_index_category,
        plan_state_category=plan_state_category)
    refined = build_agentic_workflow(
        AgenticWorkflowSpec(:AcmeWorkflowRefined, "acme", 2024, "stmt-001", ["detect", "prioritize", "pilot", "launch"];
            edges=[("detect", "prioritize"), ("prioritize", "pilot"), ("pilot", "launch")],
            action_types=[("detect", "analysis"), ("prioritize", "ranking"), ("pilot", "validation"), ("launch", "execution")],
            stage="rocket_refined",
            metadata=Dict(
                :changed => true,
                :selected_source => "local_merge",
                :selected_label => "pilot insertion",
                :score_gain => 0.27,
                :score_components => [("financial_alignment", 0.81), ("workflow_coherence", 0.89)],
            ));
        workflow_category=workflow_category,
        step_index_category=step_index_category,
        plan_state_category=plan_state_category)
    refinement = build_rocket_workflow_refinement(
        ROCKETWorkflowRefinementSpec(:AcmeWorkflowRefinement; metadata=Dict(:changed => true, :selected_source => "local_merge", :selected_label => "pilot insertion", :score_gain => 0.27));
        base_workflow=raw,
        refined_workflow=refined)
    Dict{Symbol, Any}(
        :raw_workflow => raw,
        :refined_workflow => refined,
        :refinement => refinement,
    )
end

function build_agentic_workflow_compilation_plan(example::Union{Nothing, Dict{Symbol, Any}}=nothing)
    example = example === nothing ? build_agentic_workflow_example() : example
    compile_plan(:AgenticWorkflowExamplePlan,
        example[:raw_workflow],
        example[:refined_workflow],
        example[:refinement];
        metadata=Dict(:example => "agentic_workflow"))
end

build_agentic_workflow_executable_ir(example::Union{Nothing, Dict{Symbol, Any}}=nothing) =
    lower_plan_to_executable_ir(build_agentic_workflow_compilation_plan(example))

execute_agentic_workflow_example(example::Union{Nothing, Dict{Symbol, Any}}=nothing) =
    execute_placeholder_ir(build_agentic_workflow_executable_ir(example))

function summarize_agentic_workflow_example(example::Union{Nothing, Dict{Symbol, Any}}=nothing)
    example = example === nothing ? build_agentic_workflow_example() : example
    refinement = example[:refinement]
    Dict(
        "counts" => Dict(
            "raw_workflows" => 1,
            "refined_workflows" => 1,
            "rocket_workflow_refinements" => 1,
        ),
        "workflow_examples" => [
            Dict(
                "statement_id" => refinement.base_workflow.statement_id,
                "company" => refinement.base_workflow.company,
                "year" => refinement.base_workflow.year,
                "changed" => Bool(get(refinement.metadata, :changed, false)),
                "selected_source" => String(get(refinement.metadata, :selected_source, "")),
                "selected_label" => String(get(refinement.metadata, :selected_label, "")),
                "score_gain" => Float64(get(refinement.metadata, :score_gain, 0.0)),
                "base_actions" => copy(refinement.base_workflow.actions),
                "selected_actions" => copy(refinement.refined_workflow.actions),
                "selected_score_components" => Dict(String(k) => Float64(v) for (k, v) in get(refinement.refined_workflow.metadata, :score_components, [])),
            )
        ],
    )
end
