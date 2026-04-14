using Test
using AgentFramework
using FunctorFlow

struct MockCLIFFClient <: AgentFramework.AbstractChatClient end

AgentFramework.tool_calling_capability(::Type{MockCLIFFClient}) = AgentFramework.HasToolCalling()
AgentFramework.structured_output_capability(::Type{MockCLIFFClient}) = AgentFramework.HasStructuredOutput()
AgentFramework.web_search_capability(::Type{MockCLIFFClient}) = AgentFramework.HasWebSearch()

@testset "AgentFramework Extension" begin
    client = MockCLIFFClient()
    caps = agentframework_capabilities(client)
    @test :llm_inference in caps
    @test :tool_calling in caps
    @test :structured_output in caps
    @test :web_search in caps
    @test :retrieval in caps

    spec = CLIFFAgentSpec(:retrieval_scout;
        instructions="Retrieve filings and support tools.",
        required_capabilities=[:llm_inference, :tool_calling],
        preferred_capabilities=[:web_search],
        route_bindings=[:democritus])
    strict_spec = CLIFFAgentSpec(:strict_analyst;
        instructions="Needs file search too.",
        required_capabilities=[:llm_inference, :file_search],
        route_bindings=[:democritus])

    @test supports_capabilities(spec, client)
    @test capability_gap(strict_spec, client) == [:file_search]
    decision = route_cliff_query("Analyze recent studies on red wine")
    @test supports_route_capabilities(decision, client)
    @test isempty(route_capability_gap(decision, client))

    agent = build_agentframework_agent(spec, client)
    @test agent.name == "retrieval_scout"
    @test agent.instructions == spec.instructions

    request = InteractiveCheckpointRequest(:democritus,
        "Approve the evidence plan?";
        request_id="req-001",
        payload=Dict(:choices => ["approve", "revise"]),
        response_type=:approval)
    request_event = build_agentframework_request_event(request; executor_id="router")
    @test request_event.type == AgentFramework.EVT_REQUEST_INFO
    @test request_event.request_id == "req-001"
    @test request_event.executor_id == "router"

    decision = route_cliff_query("Analyze recent studies on red wine"; execution_mode=:interactive)
    result = RouteRunResult(decision;
        status=:needs_input,
        selected_agents=[spec],
        pending_checkpoint=request,
        metadata=Dict(:note => "awaiting approval"))
    checkpoint = build_agentframework_checkpoint(result;
        iteration=2,
        previous_id="cp-001",
        graph_signature_hash="abc123")

    @test checkpoint.workflow_name == "democritus"
    @test checkpoint.iteration == 2
    @test checkpoint.previous_id == "cp-001"
    @test checkpoint.graph_signature_hash == "abc123"
    @test checkpoint.metadata["route_name"] == "democritus"
    @test checkpoint.metadata["execution_mode"] == "interactive"
    @test length(checkpoint.pending_requests) == 1
    @test checkpoint.state["route_result"]["status"] == "needs_input"
end
