# ============================================================================
# cliff_artifacts.jl — CLIFF-style artifacts, agents, and routed results
# ============================================================================

struct CLIFFAgentSpec
    name::Symbol
    description::String
    instructions::String
    required_capabilities::Vector{Symbol}
    preferred_capabilities::Vector{Symbol}
    route_bindings::Vector{Symbol}
    metadata::Dict{Symbol, Any}
end

function CLIFFAgentSpec(name;
                        description="",
                        instructions="",
                        required_capabilities::Vector{Symbol}=Symbol[],
                        preferred_capabilities::Vector{Symbol}=Symbol[],
                        route_bindings::Vector{Symbol}=Symbol[],
                        metadata::Dict=Dict{Symbol, Any}())
    CLIFFAgentSpec(
        Symbol(name),
        String(description),
        String(instructions),
        copy(required_capabilities),
        copy(preferred_capabilities),
        copy(route_bindings),
        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata),
    )
end

struct InteractiveCheckpointRequest
    request_id::String
    route_name::Symbol
    prompt::String
    payload::Dict{Symbol, Any}
    response_type::Symbol
    metadata::Dict{Symbol, Any}
end

function InteractiveCheckpointRequest(route_name, prompt;
                                      request_id="checkpoint-request",
                                      payload::Dict=Dict{Symbol, Any}(),
                                      response_type=:freeform,
                                      metadata::Dict=Dict{Symbol, Any}())
    InteractiveCheckpointRequest(
        String(request_id),
        Symbol(route_name),
        String(prompt),
        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in payload),
        Symbol(response_type),
        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata),
    )
end

struct SynthesisArtifact
    name::Symbol
    artifact_kind::Symbol
    title::String
    summary::String
    content_ref::Union{Nothing, String}
    tags::Vector{String}
    metadata::Dict{Symbol, Any}
end

function SynthesisArtifact(name;
                           artifact_kind=:report,
                           title=String(name),
                           summary="",
                           content_ref=nothing,
                           tags::Vector{String}=String[],
                           metadata::Dict=Dict{Symbol, Any}())
    SynthesisArtifact(
        Symbol(name),
        Symbol(artifact_kind),
        String(title),
        String(summary),
        content_ref === nothing ? nothing : String(content_ref),
        copy(tags),
        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata),
    )
end

struct RouteRunResult
    route_decision::CLIFFRouteDecision
    status::Symbol
    route_outdir::Union{Nothing, String}
    artifacts::Vector{SynthesisArtifact}
    selected_agents::Vector{CLIFFAgentSpec}
    workspace::Union{Nothing, ConsciousWorkspaceState}
    convergence::Union{Nothing, EvidenceConvergenceAssessment}
    pending_checkpoint::Union{Nothing, InteractiveCheckpointRequest}
    metadata::Dict{Symbol, Any}
end

function RouteRunResult(route_decision::CLIFFRouteDecision;
                        status=:completed,
                        route_outdir=nothing,
                        artifacts::Vector{SynthesisArtifact}=SynthesisArtifact[],
                        selected_agents::Vector{CLIFFAgentSpec}=CLIFFAgentSpec[],
                        workspace=nothing,
                        convergence=nothing,
                        pending_checkpoint=nothing,
                        metadata::Dict=Dict{Symbol, Any}())
    normalized_status = status isa Symbol ? status : Symbol(lowercase(strip(String(status))))
    RouteRunResult(
        route_decision,
        normalized_status,
        route_outdir === nothing ? nothing : String(route_outdir),
        copy(artifacts),
        copy(selected_agents),
        workspace,
        convergence,
        pending_checkpoint,
        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata),
    )
end

primary_artifact(result::RouteRunResult) = isempty(result.artifacts) ? nothing : result.artifacts[1]
needs_human_input(result::RouteRunResult) = result.pending_checkpoint !== nothing || result.status == :needs_input
selected_agent_names(result::RouteRunResult) = [String(agent.name) for agent in result.selected_agents]

route_matches_agent(decision::CLIFFRouteDecision, agent::CLIFFAgentSpec) =
    isempty(agent.route_bindings) || decision.route_name in agent.route_bindings

function capability_gap(agent::CLIFFAgentSpec, available_capabilities::AbstractVector{Symbol})
    available = Set(available_capabilities)
    [capability for capability in agent.required_capabilities if capability ∉ available]
end

supports_capabilities(agent::CLIFFAgentSpec, available_capabilities::AbstractVector{Symbol}) =
    isempty(capability_gap(agent, available_capabilities))

function select_agents_for_route(decision::CLIFFRouteDecision, agents::AbstractVector{<:CLIFFAgentSpec};
                                 available_capabilities::Union{Nothing, AbstractVector{Symbol}}=nothing)
    [agent for agent in agents
     if route_matches_agent(decision, agent) &&
        (available_capabilities === nothing || supports_capabilities(agent, available_capabilities))]
end

function agentframework_capabilities end
function build_agentframework_agent end
function build_agentframework_request_event end
function build_agentframework_checkpoint end

function _snapshot_claims(snapshot)
    if snapshot isa AbstractDict
        if haskey(snapshot, :top_claims)
            return Set(String.(snapshot[:top_claims]))
        elseif haskey(snapshot, "top_claims")
            return Set(String.(snapshot["top_claims"]))
        end
    end
    Set{String}()
end

function _snapshot_summary(snapshot)
    if snapshot isa AbstractDict
        if haskey(snapshot, :summary)
            return String(snapshot[:summary])
        elseif haskey(snapshot, "summary")
            return String(snapshot["summary"])
        end
    end
    string(snapshot)
end

function _claim_convergence_adapter()
    FunctionEvidenceConvergenceAdapter(
        (previous, current; policy) -> begin
            previous_claims = _snapshot_claims(previous)
            current_claims = _snapshot_claims(current)
            union_size = length(union(previous_claims, current_claims))
            union_size == 0 ? 1.0 : length(intersect(previous_claims, current_claims)) / union_size
        end;
        describe_fn=snapshot -> _snapshot_summary(snapshot),
    )
end

function build_cliff_orchestration_example()
    router = build_cliff_query_router()
    decision = route_cliff_query(router, "How similar is Adobe to Nike across recent filings?"; execution_mode=:deep)

    scout = CLIFFAgentSpec(:retrieval_scout;
        description="Collect routed evidence and expose intermediate artifacts.",
        instructions="Retrieve filings, notes, and ranked similarities for the active route.",
        required_capabilities=[:llm_inference, :tool_calling],
        preferred_capabilities=[:web_search],
        route_bindings=[:company_similarity, :democritus, :basket_rocket_sec])
    synthesist = CLIFFAgentSpec(:synthesis_editor;
        description="Write the final routed synthesis memo.",
        instructions="Synthesize the routed artifacts into one coherent explanation.",
        required_capabilities=[:llm_inference],
        preferred_capabilities=[:structured_output],
        route_bindings=[:company_similarity, :democritus, :product_feedback, :culinary_tour])
    judge = CLIFFAgentSpec(:evidence_judge;
        description="Score evidence convergence and surface stop conditions.",
        instructions="Judge when the route has enough evidence to stop collecting more.",
        required_capabilities=[:llm_inference, :structured_output],
        preferred_capabilities=[:tool_calling],
        route_bindings=[:company_similarity, :democritus])
    available_capabilities = [:llm_inference, :tool_calling, :structured_output, :web_search]
    selected_agents = select_agents_for_route(decision, [scout, synthesist, judge]; available_capabilities=available_capabilities)

    processes = [
        UnconsciousProcess(:retrieval_cluster, "retrieval_scout";
            summary="Recent filings emphasize direct-to-consumer expansion.",
            artifact_refs=["filing-batch-01", "filing-batch-02"],
            salience=0.92, relevance=0.88, novelty=0.45, urgency=0.51, attention_cost=2,
            metadata=Dict(:route => decision.route_name)),
        UnconsciousProcess(:temporal_alignment, "evidence_judge";
            summary="Temporal diffusion signals are now consistent across both companies.",
            artifact_refs=["alignment-heatmap"],
            salience=0.79, relevance=0.95, novelty=0.38, urgency=0.64, attention_cost=2,
            metadata=Dict(:route => decision.route_name)),
        UnconsciousProcess(:long_tail_notes, "synthesis_editor";
            summary="Two weaker supporting claims remain unresolved.",
            artifact_refs=["note-17"],
            salience=0.33, relevance=0.41, novelty=0.72, urgency=0.24, attention_cost=2,
            metadata=Dict(:route => decision.route_name)),
    ]

    workspace = competition_for_access(
        ConsciousnessFunctor(field_of_view=ConsciousFieldOfView(5), weights=AttentionScoreWeights()),
        processes,
    )

    board = ConsciousBroadcastBoard()
    publish!(board;
        source_agent="retrieval_scout",
        title="SEC drift",
        summary="Adobe and Nike both show direct-channel emphasis in the current filing window.",
        tags=["company_similarity", "evidence"])
    publish!(board;
        source_agent="evidence_judge",
        title="Convergence almost met",
        summary="One more stable pass will satisfy the convergence policy.",
        tags=["company_similarity", "convergence"],
        audience="synthesis_editor")

    policy = EvidenceConvergencePolicy(3;
        stability_threshold=0.75,
        required_stable_passes=2,
        max_evidence=6,
        metadata=Dict(:route => decision.route_name))
    tracker = EvidenceConvergenceTracker(policy=policy, adapter=_claim_convergence_adapter())
    assessment_1 = assess!(tracker,
        Dict(:top_claims => ["direct-to-consumer", "subscription revenue"], :summary => "Initial evidence favors channel strategy overlap.");
        evidence_count=2)
    assessment_2 = assess!(tracker,
        Dict(:top_claims => ["direct-to-consumer", "subscription revenue"], :summary => "Third filing reinforces the same overlap.");
        evidence_count=3)
    assessment_3 = assess!(tracker,
        Dict(:top_claims => ["direct-to-consumer", "subscription revenue"], :summary => "Fourth filing keeps the top claims stable.");
        evidence_count=4)

    checkpoint_request = InteractiveCheckpointRequest(:company_similarity,
        "Approve the final comparison framing before the synthesis memo is emitted?";
        request_id="company-similarity-approval",
        payload=Dict(:choices => ["approve", "revise"]),
        response_type=:approval,
        metadata=Dict(:stage => "final_synthesis"))

    artifact = SynthesisArtifact(:company_similarity_dashboard;
        artifact_kind=:dashboard,
        title="Company Similarity Dashboard",
        summary="Temporal diffusion alignment between Adobe and Nike based on recent filings.",
        content_ref="outputs/company_similarity/company_similarity_dashboard.html",
        tags=["dashboard", "company_similarity"],
        metadata=Dict(:route => decision.route_name, :stop_trigger => assessment_3.stop_trigger))

    result = RouteRunResult(decision;
        status=:completed,
        route_outdir="outputs/company_similarity",
        artifacts=[artifact],
        selected_agents=selected_agents,
        workspace=workspace,
        convergence=assessment_3,
        pending_checkpoint=nothing,
        metadata=Dict(
            :available_capabilities => available_capabilities,
            :broadcast_count => length(broadcasts(board)),
        ))

    Dict{Symbol, Any}(
        :router => router,
        :decision => decision,
        :available_capabilities => available_capabilities,
        :agents => [scout, synthesist, judge],
        :selected_agents => selected_agents,
        :processes => processes,
        :workspace => workspace,
        :board => board,
        :policy => policy,
        :tracker => tracker,
        :assessments => [assessment_1, assessment_2, assessment_3],
        :checkpoint_request => checkpoint_request,
        :artifact => artifact,
        :result => result,
    )
end

function build_cliff_orchestration_compilation_plan(example::Union{Nothing, Dict{Symbol, Any}}=nothing)
    example = example === nothing ? build_cliff_orchestration_example() : example
    compile_plan(:CLIFFOrchestrationExamplePlan,
        example[:decision],
        example[:selected_agents]...,
        example[:workspace],
        last(example[:assessments]),
        example[:checkpoint_request],
        example[:result];
        metadata=Dict(:example => "cliff_orchestration"))
end

build_cliff_orchestration_executable_ir(example::Union{Nothing, Dict{Symbol, Any}}=nothing) =
    lower_plan_to_executable_ir(build_cliff_orchestration_compilation_plan(example))

execute_cliff_orchestration_example(example::Union{Nothing, Dict{Symbol, Any}}=nothing) =
    execute_placeholder_ir(build_cliff_orchestration_executable_ir(example))

function summarize_cliff_orchestration_example(example::Union{Nothing, Dict{Symbol, Any}}=nothing)
    example = example === nothing ? build_cliff_orchestration_example() : example
    workspace = example[:workspace]
    result = example[:result]
    Dict(
        "route_decision" => as_dict(example[:decision]),
        "counts" => Dict(
            "available_agents" => length(example[:agents]),
            "selected_agents" => length(example[:selected_agents]),
            "selected_processes" => length(workspace.selected),
            "deferred_processes" => length(workspace.deferred),
            "broadcasts" => length(broadcasts(example[:board])),
            "artifacts" => length(result.artifacts),
        ),
        "selected_agents" => selected_agent_names(result),
        "workspace" => Dict(
            "used_capacity" => used_capacity(workspace),
            "remaining_capacity" => remaining_capacity(workspace),
            "selected_processes" => [String(selection.process.name) for selection in workspace.selected],
            "deferred_processes" => [String(process.name) for process in workspace.deferred],
        ),
        "convergence" => as_dict(last(example[:assessments])),
        "broadcast_titles" => [broadcast.title for broadcast in broadcasts(example[:board])],
        "checkpoint_request" => as_dict(example[:checkpoint_request]),
        "primary_artifact" => as_dict(example[:artifact]),
    )
end

function as_dict(agent::CLIFFAgentSpec)
    Dict(
        "name" => String(agent.name),
        "description" => agent.description,
        "instructions" => agent.instructions,
        "required_capabilities" => String.(agent.required_capabilities),
        "preferred_capabilities" => String.(agent.preferred_capabilities),
        "route_bindings" => String.(agent.route_bindings),
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in agent.metadata),
    )
end

function as_dict(request::InteractiveCheckpointRequest)
    Dict(
        "request_id" => request.request_id,
        "route_name" => String(request.route_name),
        "prompt" => request.prompt,
        "payload" => Dict(String(k) => _serialize_value(v) for (k, v) in request.payload),
        "response_type" => String(request.response_type),
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in request.metadata),
    )
end

function as_dict(artifact::SynthesisArtifact)
    Dict(
        "name" => String(artifact.name),
        "artifact_kind" => String(artifact.artifact_kind),
        "title" => artifact.title,
        "summary" => artifact.summary,
        "content_ref" => artifact.content_ref,
        "tags" => copy(artifact.tags),
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in artifact.metadata),
    )
end

function as_dict(result::RouteRunResult)
    Dict(
        "route_decision" => as_dict(result.route_decision),
        "status" => String(result.status),
        "route_outdir" => result.route_outdir,
        "artifacts" => [as_dict(artifact) for artifact in result.artifacts],
        "selected_agents" => [as_dict(agent) for agent in result.selected_agents],
        "workspace" => result.workspace === nothing ? nothing : as_dict(result.workspace),
        "convergence" => result.convergence === nothing ? nothing : as_dict(result.convergence),
        "pending_checkpoint" => result.pending_checkpoint === nothing ? nothing : as_dict(result.pending_checkpoint),
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in result.metadata),
    )
end

to_json(agent::CLIFFAgentSpec) = JSON3.write(as_dict(agent))
to_json(request::InteractiveCheckpointRequest) = JSON3.write(as_dict(request))
to_json(artifact::SynthesisArtifact) = JSON3.write(as_dict(artifact))
to_json(result::RouteRunResult) = JSON3.write(as_dict(result))
