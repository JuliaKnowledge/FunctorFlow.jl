# ============================================================================
# FunctorFlowAgentFrameworkExt — Optional AgentFramework.jl bridge
# ============================================================================

module FunctorFlowAgentFrameworkExt

using FunctorFlow
using AgentFramework

import FunctorFlow: agentframework_capabilities, build_agentframework_agent
import FunctorFlow: build_agentframework_request_event, build_agentframework_checkpoint
import FunctorFlow: capability_gap, supports_capabilities, route_capability_gap, supports_route_capabilities

function agentframework_capabilities(client::AgentFramework.AbstractChatClient)
    caps = Set{Symbol}([:llm_inference])
    union!(caps, AgentFramework.list_capabilities(client))
    if :web_search in caps || :file_search in caps || :tool_calling in caps
        push!(caps, :retrieval)
    end
    sort!(collect(caps); by=string)
end

capability_gap(agent::FunctorFlow.CLIFFAgentSpec, client::AgentFramework.AbstractChatClient) =
    capability_gap(agent, agentframework_capabilities(client))

supports_capabilities(agent::FunctorFlow.CLIFFAgentSpec, client::AgentFramework.AbstractChatClient) =
    isempty(capability_gap(agent, client))

route_capability_gap(decision::FunctorFlow.CLIFFRouteDecision, client::AgentFramework.AbstractChatClient) =
    route_capability_gap(decision, agentframework_capabilities(client))

supports_route_capabilities(decision::FunctorFlow.CLIFFRouteDecision, client::AgentFramework.AbstractChatClient) =
    isempty(route_capability_gap(decision, client))

function build_agentframework_agent(spec::FunctorFlow.CLIFFAgentSpec, client::AgentFramework.AbstractChatClient;
                                    tools=Any[],
                                    options=AgentFramework.ChatOptions())
    AgentFramework.Agent(
        name=String(spec.name),
        description=spec.description,
        instructions=spec.instructions,
        client=client,
        tools=collect(tools),
        options=options,
    )
end

function build_agentframework_request_event(request::FunctorFlow.InteractiveCheckpointRequest;
                                            executor_id=String(request.route_name))
    AgentFramework.event_request_info(
        request.request_id,
        String(executor_id),
        Dict(
            "route_name" => String(request.route_name),
            "prompt" => request.prompt,
            "payload" => Dict(String(k) => FunctorFlow._serialize_value(v) for (k, v) in request.payload),
            "response_type" => String(request.response_type),
            "metadata" => Dict(String(k) => FunctorFlow._serialize_value(v) for (k, v) in request.metadata),
        ),
    )
end

function build_agentframework_checkpoint(result::FunctorFlow.RouteRunResult;
                                         iteration::Int=0,
                                         previous_id=nothing,
                                         checkpoint_id=nothing,
                                         graph_signature_hash="")
    pending_requests = result.pending_checkpoint === nothing ?
        AgentFramework.WorkflowEvent[] :
        [build_agentframework_request_event(result.pending_checkpoint; executor_id=String(result.route_decision.route_name))]
    state = Dict{String, Any}("route_result" => FunctorFlow.as_dict(result))
    metadata = Dict{String, Any}(
        "route_name" => String(result.route_decision.route_name),
        "execution_mode" => String(result.route_decision.execution_mode),
        "status" => String(result.status),
        "selected_agents" => FunctorFlow.selected_agent_names(result),
        "route_outdir" => result.route_outdir,
    )
    kwargs = (
        workflow_name=String(result.route_decision.route_name),
        iteration=iteration,
        state=state,
        pending_requests=pending_requests,
        graph_signature_hash=String(graph_signature_hash),
        previous_id=previous_id === nothing ? nothing : String(previous_id),
        metadata=metadata,
    )
    checkpoint_id === nothing ?
        AgentFramework.WorkflowCheckpoint(; kwargs...) :
        AgentFramework.WorkflowCheckpoint(; id=String(checkpoint_id), kwargs...)
end

end # module FunctorFlowAgentFrameworkExt
