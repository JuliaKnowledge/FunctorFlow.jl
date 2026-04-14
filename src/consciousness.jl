# ============================================================================
# consciousness.jl — CLIFF-style conscious workspace semantics
# ============================================================================

struct UnconsciousProcess
    name::Symbol
    source_agent::String
    summary::String
    artifact_refs::Vector{String}
    salience::Float64
    relevance::Float64
    novelty::Float64
    urgency::Float64
    attention_cost::Int
    metadata::Dict{Symbol, Any}
end

function UnconsciousProcess(name, source_agent;
                            summary="",
                            artifact_refs::Vector{String}=String[],
                            salience=0.0,
                            relevance=0.0,
                            novelty=0.0,
                            urgency=0.0,
                            attention_cost::Integer=1,
                            metadata::Dict=Dict{Symbol, Any}())
    attention_cost >= 1 || throw(ArgumentError("attention_cost must be at least 1"))
    UnconsciousProcess(
        Symbol(name),
        String(source_agent),
        String(summary),
        copy(artifact_refs),
        Float64(salience),
        Float64(relevance),
        Float64(novelty),
        Float64(urgency),
        Int(attention_cost),
        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata),
    )
end

struct AttentionScoreWeights
    salience::Float64
    relevance::Float64
    novelty::Float64
    urgency::Float64
end

AttentionScoreWeights(; salience=0.35, relevance=0.30, novelty=0.20, urgency=0.15) =
    AttentionScoreWeights(Float64(salience), Float64(relevance), Float64(novelty), Float64(urgency))

struct ConsciousFieldOfView
    capacity::Int
end

function ConsciousFieldOfView(capacity::Integer=7)
    capacity >= 1 || throw(ArgumentError("capacity must be at least 1"))
    ConsciousFieldOfView(Int(capacity))
end

struct BroadcastSelection
    process::UnconsciousProcess
    score::Float64
end

BroadcastSelection(process::UnconsciousProcess, score) = BroadcastSelection(process, Float64(score))

struct ConsciousWorkspaceState
    field_of_view::ConsciousFieldOfView
    selected::Vector{BroadcastSelection}
    deferred::Vector{UnconsciousProcess}
end

function ConsciousWorkspaceState(field_of_view::ConsciousFieldOfView;
                                 selected::Vector{BroadcastSelection}=BroadcastSelection[],
                                 deferred::Vector{UnconsciousProcess}=UnconsciousProcess[])
    ConsciousWorkspaceState(field_of_view, copy(selected), copy(deferred))
end

used_capacity(state::ConsciousWorkspaceState) = sum(item.process.attention_cost for item in state.selected)
remaining_capacity(state::ConsciousWorkspaceState) = state.field_of_view.capacity - used_capacity(state)

struct ConsciousBroadcast
    broadcast_id::String
    source_agent::String
    title::String
    summary::String
    payload::Dict{Symbol, Any}
    tags::Vector{String}
    audience::String
    read_broadcast_ids::Vector{String}
end

function ConsciousBroadcast(broadcast_id, source_agent, title, summary;
                            payload::Dict=Dict{Symbol, Any}(),
                            tags::Vector{String}=String[],
                            audience="global",
                            read_broadcast_ids::Vector{String}=String[])
    ConsciousBroadcast(
        String(broadcast_id),
        String(source_agent),
        String(title),
        String(summary),
        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in payload),
        copy(tags),
        String(audience),
        copy(read_broadcast_ids),
    )
end

mutable struct ConsciousBroadcastBoard
    broadcasts::Vector{ConsciousBroadcast}
    counter::Int
end

ConsciousBroadcastBoard() = ConsciousBroadcastBoard(ConsciousBroadcast[], 0)

function publish!(board::ConsciousBroadcastBoard;
                  source_agent,
                  title,
                  summary,
                  payload::Dict=Dict{Symbol, Any}(),
                  tags::Vector{String}=String[],
                  audience="global",
                  read_broadcast_ids::Vector{String}=String[])
    board.counter += 1
    broadcast = ConsciousBroadcast(
        string("broadcast-", lpad(board.counter, 4, '0')),
        source_agent,
        title,
        summary;
        payload=payload,
        tags=tags,
        audience=audience,
        read_broadcast_ids=read_broadcast_ids,
    )
    push!(board.broadcasts, broadcast)
    broadcast
end

broadcasts(board::ConsciousBroadcastBoard) = copy(board.broadcasts)

function messages_for_agent(board::ConsciousBroadcastBoard, agent_name; tag=nothing)
    requested_agent = String(agent_name)
    filtered = [
        broadcast for broadcast in board.broadcasts
        if broadcast.audience in ("global", requested_agent)
    ]
    if tag !== nothing
        requested_tag = String(tag)
        filtered = [broadcast for broadcast in filtered if requested_tag in broadcast.tags]
    end
    filtered
end

function clear_broadcasts!(board::ConsciousBroadcastBoard)
    empty!(board.broadcasts)
    board.counter = 0
    board
end

function publish_workspace!(board::ConsciousBroadcastBoard, workspace::ConsciousWorkspaceState;
                            source_agent="consciousness",
                            audience="global",
                            tags::AbstractVector=String["workspace"],
                            title_prefix="Selected process")
    published = ConsciousBroadcast[]
    normalized_tags = String.(tags)
    for selection in workspace.selected
        push!(published, publish!(board;
            source_agent=source_agent,
            title="$(title_prefix): $(selection.process.name)",
            summary=selection.process.summary,
            payload=Dict(
                :process_name => selection.process.name,
                :score => selection.score,
                :artifact_refs => copy(selection.process.artifact_refs),
                :attention_cost => selection.process.attention_cost,
            ),
            tags=unique(vcat(copy(normalized_tags), ["workspace"])),
            audience=audience))
    end
    published
end

struct ConsciousnessFunctor
    field_of_view::ConsciousFieldOfView
    weights::AttentionScoreWeights
end

ConsciousnessFunctor(; field_of_view=ConsciousFieldOfView(), weights=AttentionScoreWeights()) =
    ConsciousnessFunctor(field_of_view, weights)

function score(functor::ConsciousnessFunctor, process::UnconsciousProcess)
    functor.weights.salience * process.salience +
    functor.weights.relevance * process.relevance +
    functor.weights.novelty * process.novelty +
    functor.weights.urgency * process.urgency
end

function competition_for_access(functor::ConsciousnessFunctor, processes::AbstractVector{<:UnconsciousProcess})
    ranked = sort(collect(processes); by=process -> (-score(functor, process), process.attention_cost, string(process.name)))
    selected = BroadcastSelection[]
    deferred = UnconsciousProcess[]
    remaining = functor.field_of_view.capacity
    for process in ranked
        process.attention_cost <= remaining || begin
            push!(deferred, process)
            continue
        end
        process_score = score(functor, process)
        push!(selected, BroadcastSelection(process, process_score))
        remaining -= process.attention_cost
    end
    ConsciousWorkspaceState(functor.field_of_view; selected=selected, deferred=deferred)
end

function as_dict(process::UnconsciousProcess)
    Dict(
        "name" => String(process.name),
        "source_agent" => process.source_agent,
        "summary" => process.summary,
        "artifact_refs" => copy(process.artifact_refs),
        "salience" => process.salience,
        "relevance" => process.relevance,
        "novelty" => process.novelty,
        "urgency" => process.urgency,
        "attention_cost" => process.attention_cost,
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in process.metadata),
    )
end

as_dict(weights::AttentionScoreWeights) = Dict(
    "salience" => weights.salience,
    "relevance" => weights.relevance,
    "novelty" => weights.novelty,
    "urgency" => weights.urgency,
)

as_dict(field::ConsciousFieldOfView) = Dict("capacity" => field.capacity)

as_dict(selection::BroadcastSelection) = Dict(
    "process" => as_dict(selection.process),
    "score" => selection.score,
)

function as_dict(state::ConsciousWorkspaceState)
    Dict(
        "field_of_view" => as_dict(state.field_of_view),
        "selected" => [as_dict(selection) for selection in state.selected],
        "deferred" => [as_dict(process) for process in state.deferred],
        "used_capacity" => used_capacity(state),
        "remaining_capacity" => remaining_capacity(state),
    )
end

function as_dict(broadcast::ConsciousBroadcast)
    Dict(
        "broadcast_id" => broadcast.broadcast_id,
        "source_agent" => broadcast.source_agent,
        "title" => broadcast.title,
        "summary" => broadcast.summary,
        "payload" => Dict(String(k) => _serialize_value(v) for (k, v) in broadcast.payload),
        "tags" => copy(broadcast.tags),
        "audience" => broadcast.audience,
        "read_broadcast_ids" => copy(broadcast.read_broadcast_ids),
    )
end

function as_dict(board::ConsciousBroadcastBoard)
    Dict(
        "counter" => board.counter,
        "broadcasts" => [as_dict(broadcast) for broadcast in board.broadcasts],
    )
end

to_json(process::UnconsciousProcess) = JSON3.write(as_dict(process))
to_json(weights::AttentionScoreWeights) = JSON3.write(as_dict(weights))
to_json(field::ConsciousFieldOfView) = JSON3.write(as_dict(field))
to_json(selection::BroadcastSelection) = JSON3.write(as_dict(selection))
to_json(state::ConsciousWorkspaceState) = JSON3.write(as_dict(state))
to_json(broadcast::ConsciousBroadcast) = JSON3.write(as_dict(broadcast))
to_json(board::ConsciousBroadcastBoard) = JSON3.write(as_dict(board))
