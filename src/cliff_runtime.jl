# ============================================================================
# cliff_runtime.jl — Executable CLIFF route runtimes
# ============================================================================

struct CLIFFRouteUpdate
    snapshot::Any
    evidence_delta::Int
    processes::Vector{UnconsciousProcess}
    artifacts::Vector{SynthesisArtifact}
    pending_checkpoint::Union{Nothing, InteractiveCheckpointRequest}
    status::Union{Nothing, Symbol}
    metadata::Dict{Symbol, Any}
end

function CLIFFRouteUpdate(;
                          snapshot=nothing,
                          evidence_delta::Integer=0,
                          processes::AbstractVector{<:UnconsciousProcess}=UnconsciousProcess[],
                          artifacts::AbstractVector{<:SynthesisArtifact}=SynthesisArtifact[],
                          pending_checkpoint=nothing,
                          status=nothing,
                          metadata::Dict=Dict{Symbol, Any}())
    evidence_delta >= 0 || throw(ArgumentError("evidence_delta must be non-negative"))
    normalized_status = if status === nothing
        nothing
    elseif status isa Symbol
        status
    else
        Symbol(lowercase(strip(String(status))))
    end
    CLIFFRouteUpdate(
        snapshot,
        Int(evidence_delta),
        UnconsciousProcess[process for process in processes],
        SynthesisArtifact[artifact for artifact in artifacts],
        pending_checkpoint,
        normalized_status,
        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata),
    )
end

struct CLIFFRuntimeConfig
    agents::Vector{CLIFFAgentSpec}
    available_capabilities::Vector{Symbol}
    consciousness::ConsciousnessFunctor
    convergence_policy::Union{Nothing, EvidenceConvergencePolicy}
    convergence_adapter::Union{Nothing, AbstractEvidenceConvergenceAdapter}
    route_outdir::Union{Nothing, String}
    metadata::Dict{Symbol, Any}
end

function CLIFFRuntimeConfig(;
                            agents::AbstractVector{<:CLIFFAgentSpec}=CLIFFAgentSpec[],
                            available_capabilities::AbstractVector=Symbol[],
                            consciousness::ConsciousnessFunctor=ConsciousnessFunctor(),
                            convergence_policy=nothing,
                            convergence_adapter=nothing,
                            route_outdir=nothing,
                            metadata::Dict=Dict{Symbol, Any}())
    convergence_policy === nothing || convergence_policy isa EvidenceConvergencePolicy ||
        throw(ArgumentError("convergence_policy must be an EvidenceConvergencePolicy or nothing"))
    convergence_adapter === nothing || convergence_adapter isa AbstractEvidenceConvergenceAdapter ||
        throw(ArgumentError("convergence_adapter must be an AbstractEvidenceConvergenceAdapter or nothing"))
    CLIFFRuntimeConfig(
        CLIFFAgentSpec[agent for agent in agents],
        Symbol.(available_capabilities),
        consciousness,
        convergence_policy,
        convergence_adapter,
        route_outdir === nothing ? nothing : String(route_outdir),
        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata),
    )
end

Base.@kwdef mutable struct CLIFFExecutionContext
    query::String
    route_decision::CLIFFRouteDecision
    selected_agents::Vector{CLIFFAgentSpec}
    available_capabilities::Vector{Symbol}
    board::ConsciousBroadcastBoard = ConsciousBroadcastBoard()
    route_outdir::Union{Nothing, String} = nothing
    metadata::Dict{Symbol, Any} = Dict{Symbol, Any}()
    state::Dict{Symbol, Any} = Dict{Symbol, Any}()
end

get_route_state(ctx::CLIFFExecutionContext, key, default=nothing) = get(ctx.state, Symbol(key), default)

function set_route_state!(ctx::CLIFFExecutionContext, key, value)
    ctx.state[Symbol(key)] = value
    ctx
end

struct CLIFFRouteExecutor
    route_name::Symbol
    description::String
    run::Function
    metadata::Dict{Symbol, Any}
end

function CLIFFRouteExecutor(route_name, run::Function;
                            description="",
                            metadata::Dict=Dict{Symbol, Any}())
    CLIFFRouteExecutor(
        Symbol(route_name),
        String(description),
        run,
        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata),
    )
end

mutable struct CLIFFRuntime
    router::CLIFFQueryRouter
    executors::OrderedDict{Symbol, CLIFFRouteExecutor}
    metadata::Dict{Symbol, Any}
end

function CLIFFRuntime(; router=build_cliff_query_router(), metadata::Dict=Dict{Symbol, Any}())
    CLIFFRuntime(
        router,
        OrderedDict{Symbol, CLIFFRouteExecutor}(),
        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata),
    )
end

struct CLIFFRouteTrace
    query::String
    route_decision::CLIFFRouteDecision
    selected_agents::Vector{CLIFFAgentSpec}
    updates::Vector{CLIFFRouteUpdate}
    workspaces::Vector{ConsciousWorkspaceState}
    assessments::Vector{EvidenceConvergenceAssessment}
    board::ConsciousBroadcastBoard
    result::RouteRunResult
    metadata::Dict{Symbol, Any}
end

latest_workspace(trace::CLIFFRouteTrace) = isempty(trace.workspaces) ? nothing : trace.workspaces[end]
latest_assessment(trace::CLIFFRouteTrace) = isempty(trace.assessments) ? nothing : trace.assessments[end]
broadcast_titles(trace::CLIFFRouteTrace) = [broadcast.title for broadcast in trace.board.broadcasts]

function route_capability_gap(decision::CLIFFRouteDecision, available_capabilities::AbstractVector)
    available = Set(Symbol.(available_capabilities))
    [capability for capability in decision.required_capabilities if capability ∉ available]
end

supports_route_capabilities(decision::CLIFFRouteDecision, available_capabilities::AbstractVector) =
    isempty(route_capability_gap(decision, available_capabilities))

function register_route_executor!(runtime::CLIFFRuntime, executor::CLIFFRouteExecutor)
    haskey(runtime.router.routes, executor.route_name) ||
        throw(ArgumentError("Cannot register executor for unknown route: $(executor.route_name)"))
    runtime.executors[executor.route_name] = executor
    runtime
end

function register_route_executor!(f::Function, runtime::CLIFFRuntime, route_name;
                                  description="",
                                  metadata::Dict=Dict{Symbol, Any}())
    register_route_executor!(runtime, CLIFFRouteExecutor(route_name, f; description=description, metadata=metadata))
end

function get_route_executor(runtime::CLIFFRuntime, route_name)
    key = Symbol(route_name)
    haskey(runtime.executors, key) || throw(ArgumentError("No executor registered for route: $(key)"))
    runtime.executors[key]
end

list_route_executors(runtime::CLIFFRuntime) = collect(keys(runtime.executors))

function _materialize_route_updates(raw_updates)
    if raw_updates === nothing
        return CLIFFRouteUpdate[]
    elseif raw_updates isa CLIFFRouteUpdate
        return CLIFFRouteUpdate[raw_updates]
    end
    collected = collect(raw_updates)
    all(update -> update isa CLIFFRouteUpdate, collected) ||
        throw(ArgumentError("Route executors must return CLIFFRouteUpdate values"))
    CLIFFRouteUpdate[update for update in collected]
end

function _freeze_board(board::ConsciousBroadcastBoard)
    ConsciousBroadcastBoard(copy(board.broadcasts), board.counter)
end

function _default_route_outdir(decision::CLIFFRouteDecision, config::CLIFFRuntimeConfig)
    config.route_outdir === nothing ? joinpath("outputs", String(decision.route_name)) : config.route_outdir
end

function execute_cliff_query(runtime::CLIFFRuntime, query;
                             route_override=:auto,
                             execution_mode=:quick,
                             config::CLIFFRuntimeConfig=CLIFFRuntimeConfig())
    decision = route_cliff_query(runtime.router, query; route_override=route_override, execution_mode=execution_mode)
    execute_cliff_route(runtime, decision; query=query, config=config)
end

function execute_cliff_route(runtime::CLIFFRuntime, decision::CLIFFRouteDecision;
                             query="",
                             config::CLIFFRuntimeConfig=CLIFFRuntimeConfig())
    missing_route_capabilities = route_capability_gap(decision, config.available_capabilities)
    isempty(missing_route_capabilities) || throw(ArgumentError(
        "Route $(decision.route_name) requires capabilities $(missing_route_capabilities), but only $(config.available_capabilities) were provided",
    ))

    selected_agents = select_agents_for_route(
        decision,
        config.agents;
        available_capabilities=config.available_capabilities,
    )
    ctx = CLIFFExecutionContext(
        query=String(query),
        route_decision=decision,
        selected_agents=selected_agents,
        available_capabilities=copy(config.available_capabilities),
        route_outdir=_default_route_outdir(decision, config),
        metadata=merge(copy(runtime.metadata), copy(config.metadata)),
    )

    tracker = config.convergence_policy === nothing ? nothing : EvidenceConvergenceTracker(
        policy=config.convergence_policy,
        adapter=something(config.convergence_adapter, _claim_convergence_adapter()),
    )
    executor = get_route_executor(runtime, decision.route_name)
    updates = _materialize_route_updates(executor.run(String(query), ctx))

    workspaces = ConsciousWorkspaceState[]
    assessments = EvidenceConvergenceAssessment[]
    accumulated_artifacts = SynthesisArtifact[]
    evidence_count = 0
    final_status = nothing
    pending_checkpoint = nothing
    final_workspace = nothing
    final_assessment = nothing

    for update in updates
        evidence_count += update.evidence_delta
        append!(accumulated_artifacts, update.artifacts)

        if !isempty(update.processes)
            final_workspace = competition_for_access(config.consciousness, update.processes)
            push!(workspaces, final_workspace)
        end

        if tracker !== nothing && update.snapshot !== nothing
            final_assessment = assess!(tracker, update.snapshot; evidence_count=evidence_count)
            push!(assessments, final_assessment)
        end

        if update.pending_checkpoint !== nothing
            pending_checkpoint = update.pending_checkpoint
            final_status = something(update.status, :needs_input)
            break
        end

        if update.status !== nothing
            final_status = update.status
            break
        end

        if final_assessment !== nothing && final_assessment.stop
            final_status = :completed
            break
        end
    end

    final_status = something(final_status, pending_checkpoint === nothing ? :completed : :needs_input)
    result = RouteRunResult(decision;
        status=final_status,
        route_outdir=ctx.route_outdir,
        artifacts=accumulated_artifacts,
        selected_agents=selected_agents,
        workspace=final_workspace,
        convergence=final_assessment,
        pending_checkpoint=pending_checkpoint,
        metadata=merge(copy(config.metadata), Dict{Symbol, Any}(
            :executor_description => executor.description,
            :evidence_count => evidence_count,
            :iterations => length(updates),
            :broadcast_count => length(ctx.board.broadcasts),
            :selected_agents => [agent.name for agent in selected_agents],
            :route_state => copy(ctx.state),
        )),
    )

    CLIFFRouteTrace(
        String(query),
        decision,
        selected_agents,
        updates,
        workspaces,
        assessments,
        _freeze_board(ctx.board),
        result,
        merge(copy(runtime.metadata), copy(config.metadata), Dict{Symbol, Any}(
            :evidence_count => evidence_count,
            :executor_description => executor.description,
            :route_state => copy(ctx.state),
        )),
    )
end

function _demo_cliff_agents()
    [
        CLIFFAgentSpec(:retrieval_scout;
            description="Collect routed evidence and surface intermediate findings.",
            instructions="Retrieve and summarize evidence for the active CLIFF route.",
            required_capabilities=[:llm_inference, :retrieval],
            preferred_capabilities=[:tool_calling, :web_search],
            route_bindings=[:company_similarity, :democritus]),
        CLIFFAgentSpec(:synthesis_editor;
            description="Write the final routed synthesis artifact.",
            instructions="Synthesize the active route into one coherent answer.",
            required_capabilities=[:llm_inference],
            preferred_capabilities=[:structured_output],
            route_bindings=[:company_similarity, :democritus]),
        CLIFFAgentSpec(:evidence_judge;
            description="Judge whether routed evidence has converged.",
            instructions="Decide when the route can stop gathering new evidence.",
            required_capabilities=[:llm_inference],
            preferred_capabilities=[:structured_output],
            route_bindings=[:company_similarity, :democritus]),
    ]
end

function build_cliff_runtime_example()
    runtime = CLIFFRuntime(metadata=Dict(:example => "cliff_runtime"))

    register_route_executor!(runtime, :company_similarity; description="Demo company-similarity executor") do query, ctx
        set_route_state!(ctx, :normalized_pair, ("Adobe", "Nike"))
        publish!(ctx.board;
            source_agent="retrieval_scout",
            title="Pair recognized",
            summary="The routed query compares Adobe and Nike across a recent filing window.",
            tags=["company_similarity", "routing"])

        update_1 = CLIFFRouteUpdate(
            snapshot=Dict(
                :top_claims => ["direct-to-consumer", "membership flywheel"],
                :summary => "The first comparative batch isolates shared channel-expansion themes.",
            ),
            evidence_delta=2,
            processes=[
                UnconsciousProcess(:filing_overlap, "retrieval_scout";
                    summary="Two filings already agree on direct-channel emphasis.",
                    artifact_refs=["filing-batch-01", "filing-batch-02"],
                    salience=0.88, relevance=0.94, novelty=0.41, urgency=0.38, attention_cost=2),
                UnconsciousProcess(:draft_memo, "synthesis_editor";
                    summary="A draft memo frame is ready once one more stable batch arrives.",
                    artifact_refs=["draft-note-01"],
                    salience=0.54, relevance=0.68, novelty=0.26, urgency=0.29, attention_cost=2),
            ],
        )

        publish!(ctx.board;
            source_agent="evidence_judge",
            title="Similarity signal strengthening",
            summary="The first evidence floor has been met; the next stable batch can stop the route.",
            tags=["company_similarity", "convergence"],
            audience="synthesis_editor")

        update_2 = CLIFFRouteUpdate(
            snapshot=Dict(
                :top_claims => ["direct-to-consumer", "membership flywheel"],
                :summary => "A second comparative batch keeps the dominant claims unchanged.",
            ),
            evidence_delta=1,
            processes=[
                UnconsciousProcess(:stable_claims, "evidence_judge";
                    summary="The dominant claims are now stable enough to stop gathering.",
                    artifact_refs=["stability-report"],
                    salience=0.83, relevance=0.92, novelty=0.22, urgency=0.57, attention_cost=2),
                UnconsciousProcess(:final_dashboard, "synthesis_editor";
                    summary="The dashboard is ready to be emitted.",
                    artifact_refs=["company-similarity-dashboard"],
                    salience=0.77, relevance=0.87, novelty=0.35, urgency=0.66, attention_cost=2),
            ],
            artifacts=[
                SynthesisArtifact(:company_similarity_dashboard;
                    artifact_kind=:dashboard,
                    title="Company Similarity Dashboard",
                    summary="A routed comparison between Adobe and Nike over recent filings.",
                    content_ref="outputs/company_similarity/company_similarity_dashboard.html",
                    tags=["dashboard", "company_similarity"]),
            ],
        )

        [update_1, update_2]
    end

    register_route_executor!(runtime, :democritus; description="Demo Democritus executor with checkpointing") do query, ctx
        set_route_state!(ctx, :study_focus, "minimum wage and employment")
        publish!(ctx.board;
            source_agent="retrieval_scout",
            title="Corpus focus proposed",
            summary="The route has isolated a red-wine style study corpus, but it still needs approval on the retrieval focus.",
            tags=["democritus", "routing"])

        update_1 = CLIFFRouteUpdate(
            snapshot=Dict(
                :top_claims => ["minimum wage -> employment", "heterogeneous treatment effects"],
                :summary => "The first retrieval batch supports a labor-economics focus.",
            ),
            evidence_delta=2,
            processes=[
                UnconsciousProcess(:retrieval_frontier, "retrieval_scout";
                    summary="A candidate focus set is ready for user review.",
                    artifact_refs=["corpus-frontier"],
                    salience=0.81, relevance=0.91, novelty=0.49, urgency=0.63, attention_cost=2),
            ],
        )

        checkpoint = InteractiveCheckpointRequest(
            :democritus,
            "Approve the retrieval focus before the final synthesis step?";
            request_id="democritus-focus-approval",
            payload=Dict(:choices => ["approve", "refine focus"]),
            response_type=:approval,
            metadata=Dict(:stage => "retrieval_focus_review"),
        )

        publish!(ctx.board;
            source_agent="evidence_judge",
            title="Checkpoint required",
            summary="The route pauses here so the retrieval focus can be confirmed before synthesis.",
            tags=["democritus", "checkpoint"],
            audience="synthesis_editor")

        update_2 = CLIFFRouteUpdate(
            snapshot=Dict(
                :top_claims => ["minimum wage -> employment", "heterogeneous treatment effects"],
                :summary => "The route is pausing for confirmation before more evidence is gathered.",
            ),
            evidence_delta=0,
            processes=[
                UnconsciousProcess(:pending_review, "evidence_judge";
                    summary="The route is blocked on a retrieval-focus approval.",
                    artifact_refs=["focus-approval"],
                    salience=0.74, relevance=0.95, novelty=0.12, urgency=0.88, attention_cost=2),
            ],
            pending_checkpoint=checkpoint,
        )

        [update_1, update_2]
    end

    config = CLIFFRuntimeConfig(
        agents=_demo_cliff_agents(),
        available_capabilities=[:llm_inference, :retrieval, :tool_calling, :structured_output, :web_search],
        consciousness=ConsciousnessFunctor(field_of_view=ConsciousFieldOfView(4), weights=AttentionScoreWeights()),
        convergence_policy=EvidenceConvergencePolicy(3; stability_threshold=0.8, required_stable_passes=1, max_evidence=6),
        convergence_adapter=_claim_convergence_adapter(),
        metadata=Dict(:example => "cliff_runtime"),
    )

    Dict{Symbol, Any}(
        :runtime => runtime,
        :config => config,
        :company_similarity_query => "How similar is Adobe to Nike across recent filings?",
        :democritus_query => "Analyze recent studies on minimum wage and employment",
    )
end

function execute_cliff_runtime_example(example::Union{Nothing, Dict{Symbol, Any}}=nothing)
    example = example === nothing ? build_cliff_runtime_example() : example
    execute_cliff_query(example[:runtime], example[:company_similarity_query];
        execution_mode=:deep,
        config=example[:config])
end

function execute_cliff_interactive_example(example::Union{Nothing, Dict{Symbol, Any}}=nothing)
    example = example === nothing ? build_cliff_runtime_example() : example
    execute_cliff_query(example[:runtime], example[:democritus_query];
        execution_mode=:interactive,
        config=example[:config])
end

function summarize_cliff_route_trace(trace::CLIFFRouteTrace)
    Dict(
        "query" => trace.query,
        "route_decision" => as_dict(trace.route_decision),
        "status" => String(trace.result.status),
        "selected_agents" => selected_agent_names(trace.result),
        "counts" => Dict(
            "updates" => length(trace.updates),
            "workspaces" => length(trace.workspaces),
            "assessments" => length(trace.assessments),
            "artifacts" => length(trace.result.artifacts),
            "broadcasts" => length(trace.board.broadcasts),
        ),
        "broadcast_titles" => broadcast_titles(trace),
        "latest_workspace" => latest_workspace(trace) === nothing ? nothing : as_dict(latest_workspace(trace)),
        "latest_assessment" => latest_assessment(trace) === nothing ? nothing : as_dict(latest_assessment(trace)),
        "pending_checkpoint" => trace.result.pending_checkpoint === nothing ? nothing : as_dict(trace.result.pending_checkpoint),
        "primary_artifact" => primary_artifact(trace.result) === nothing ? nothing : as_dict(primary_artifact(trace.result)),
    )
end

function as_dict(config::CLIFFRuntimeConfig)
    Dict(
        "agents" => [as_dict(agent) for agent in config.agents],
        "available_capabilities" => String.(config.available_capabilities),
        "consciousness" => Dict(
            "field_of_view" => as_dict(config.consciousness.field_of_view),
            "weights" => as_dict(config.consciousness.weights),
        ),
        "convergence_policy" => config.convergence_policy === nothing ? nothing : as_dict(config.convergence_policy),
        "route_outdir" => config.route_outdir,
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in config.metadata),
    )
end

function as_dict(update::CLIFFRouteUpdate)
    Dict(
        "snapshot" => _serialize_value(update.snapshot),
        "evidence_delta" => update.evidence_delta,
        "processes" => [as_dict(process) for process in update.processes],
        "artifacts" => [as_dict(artifact) for artifact in update.artifacts],
        "pending_checkpoint" => update.pending_checkpoint === nothing ? nothing : as_dict(update.pending_checkpoint),
        "status" => update.status === nothing ? nothing : String(update.status),
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in update.metadata),
    )
end

function as_dict(executor::CLIFFRouteExecutor)
    Dict(
        "route_name" => String(executor.route_name),
        "description" => executor.description,
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in executor.metadata),
    )
end

function as_dict(runtime::CLIFFRuntime)
    Dict(
        "router" => as_dict(runtime.router),
        "executors" => [as_dict(executor) for executor in values(runtime.executors)],
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in runtime.metadata),
    )
end

function as_dict(trace::CLIFFRouteTrace)
    Dict(
        "query" => trace.query,
        "route_decision" => as_dict(trace.route_decision),
        "selected_agents" => [as_dict(agent) for agent in trace.selected_agents],
        "updates" => [as_dict(update) for update in trace.updates],
        "workspaces" => [as_dict(workspace) for workspace in trace.workspaces],
        "assessments" => [as_dict(assessment) for assessment in trace.assessments],
        "board" => as_dict(trace.board),
        "result" => as_dict(trace.result),
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in trace.metadata),
    )
end

to_json(config::CLIFFRuntimeConfig) = JSON3.write(as_dict(config))
to_json(update::CLIFFRouteUpdate) = JSON3.write(as_dict(update))
to_json(executor::CLIFFRouteExecutor) = JSON3.write(as_dict(executor))
to_json(runtime::CLIFFRuntime) = JSON3.write(as_dict(runtime))
to_json(trace::CLIFFRouteTrace) = JSON3.write(as_dict(trace))
