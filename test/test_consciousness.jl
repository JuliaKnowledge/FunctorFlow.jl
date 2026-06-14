# ============================================================================
# test_consciousness.jl — CLIFF conscious-workspace layer (Ch. 16)
# ============================================================================

using Test
using FunctorFlow

@testset "Attention scoring" begin
    w = AttentionScoreWeights()
    @test w.salience + w.relevance + w.novelty + w.urgency ≈ 1.0

    p = UnconsciousProcess(:p, "agent";
        salience=1.0, relevance=0.0, novelty=0.0, urgency=0.0)
    f = ConsciousnessFunctor(; weights=w)
    @test score(f, p) ≈ w.salience

    # attention_cost must be ≥ 1.
    @test_throws ArgumentError UnconsciousProcess(:bad, "agent"; attention_cost=0)
    @test_throws ArgumentError ConsciousFieldOfView(0)
end

@testset "Competition for access (capacity-bounded selection)" begin
    procs = [
        UnconsciousProcess(:high, "a"; salience=0.9, relevance=0.9, novelty=0.9, urgency=0.9, attention_cost=2),
        UnconsciousProcess(:mid,  "b"; salience=0.5, relevance=0.5, novelty=0.5, urgency=0.5, attention_cost=2),
        UnconsciousProcess(:low,  "c"; salience=0.1, relevance=0.1, novelty=0.1, urgency=0.1, attention_cost=2),
    ]
    f = ConsciousnessFunctor(; field_of_view=ConsciousFieldOfView(4))
    ws = competition_for_access(f, procs)

    # Capacity 4 with cost-2 processes → exactly the two highest-scoring selected.
    selected_names = [s.process.name for s in ws.selected]
    @test selected_names == [:high, :mid]
    @test [p.name for p in ws.deferred] == [:low]
    @test used_capacity(ws) == 4
    @test remaining_capacity(ws) == 0

    # Selected scores are sorted descending.
    @test ws.selected[1].score ≥ ws.selected[2].score

    # Tighter field of view defers more.
    ws2 = competition_for_access(ConsciousnessFunctor(; field_of_view=ConsciousFieldOfView(2)), procs)
    @test length(ws2.selected) == 1
    @test ws2.selected[1].process.name == :high
    @test length(ws2.deferred) == 2
end

@testset "Broadcast board" begin
    board = ConsciousBroadcastBoard()
    b1 = publish!(board; source_agent="scout", title="Found", summary="x",
                  tags=["routing"], audience="global")
    b2 = publish!(board; source_agent="judge", title="Stop", summary="y",
                  tags=["convergence"], audience="editor")
    @test b1.broadcast_id == "broadcast-0001"
    @test b2.broadcast_id == "broadcast-0002"
    @test length(broadcasts(board)) == 2

    # Audience filtering: editor sees global + its own; tag filtering works.
    @test length(messages_for_agent(board, "editor")) == 2
    @test length(messages_for_agent(board, "someone_else")) == 1   # only global
    @test length(messages_for_agent(board, "editor"; tag="convergence")) == 1

    clear_broadcasts!(board)
    @test isempty(broadcasts(board))
    @test board.counter == 0
end

@testset "publish_workspace! + serialization" begin
    procs = [UnconsciousProcess(:p1, "a"; salience=0.8, relevance=0.8, attention_cost=1,
                                artifact_refs=["art-1"]),
             UnconsciousProcess(:p2, "b"; salience=0.6, relevance=0.6, attention_cost=1)]
    ws = competition_for_access(ConsciousnessFunctor(), procs)
    board = ConsciousBroadcastBoard()
    published = publish_workspace!(board, ws; source_agent="consciousness")
    @test length(published) == length(ws.selected)
    @test all(b -> "workspace" in b.tags, published)

    # JSON serialization of workspace + board does not throw and carries data.
    js = to_json(ws)
    @test occursin("selected", js)
    @test occursin("used_capacity", js)
    @test occursin("broadcast-0001", to_json(board))
end
