# ============================================================================
# test_cliff_textbook.jl — Textbook-grounded CLIFF routing (Categories for AGI)
# ============================================================================

using Test
using FunctorFlow

@testset "Textbook chapter registry" begin
    @test length(CATAGI_TEXTBOOK) == 12
    @test textbook_chapter(1).title == "Category Theory for AGI"
    @test textbook_chapter(16).title == "Consciousness"
    @test_throws ArgumentError textbook_chapter(99)

    # Every chapter primitive that is a "runnable demo" is a real MACRO_LIBRARY key.
    for c in textbook_chapters()
        for d in runnable_demos(c)
            @test haskey(MACRO_LIBRARY, d)
        end
    end
end

@testset "Route / primitive linkage" begin
    # Universal constructions back the company-similarity route (Ch. 4).
    @test any(c -> c.number == 4, chapters_for_route(:company_similarity))
    # Topos / sheaf gluing backs Democritus (Ch. 13).
    @test any(c -> c.number == 13, chapters_for_route(:democritus))
    # Consciousness (Ch. 16) backs every route (it is the orchestration layer).
    for r in (:company_similarity, :democritus, :basket_rocket_sec,
              :culinary_tour, :product_feedback, :course_demo)
        @test any(c -> c.number == 16, chapters_for_route(r))
    end

    # KET demo is grounded in Categorical Deep Learning (Ch. 5).
    @test any(c -> c.number == 5, chapters_for_primitive(:ket))
    # BASKET/ROCKET demos grounded in Dynamic Compositionality (Ch. 8).
    @test any(c -> c.number == 8, chapters_for_primitive(:basket_workflow))

    # Accepts a CLIFFRouteDecision too.
    decision = route_cliff_query("How similar is Adobe to Nike?")
    @test !isempty(chapters_for_route(decision))
end

@testset "Query-driven chapter recommendation" begin
    # do-calculus / identifiability → Judo Calculus (Ch. 14) ranks first.
    recs = recommend_chapters("use do-calculus to check identifiability of a causal effect")
    @test recs[1].number == 14
    @test length(recs) <= 3

    # neural attention / message passing → Categorical Deep Learning (Ch. 5).
    recs2 = recommend_chapters("explain attention and message passing in neural networks")
    @test 5 in [c.number for c in recs2]

    # A query with no thematic overlap returns nothing unless include_zero.
    @test isempty(recommend_chapters("zzzz qqqq"; limit=3))
    @test !isempty(recommend_chapters("zzzz qqqq"; limit=3, include_zero=true))
end

@testset "route_with_textbook end-to-end" begin
    rt = route_with_textbook("How similar is Adobe to Nike across recent filings?")
    @test rt.decision.route_name == :company_similarity
    @test any(c -> c.number == 4, rt.route_chapters)
    @test rt.query_chapters isa Vector{TextbookChapter}
    @test all(d -> haskey(MACRO_LIBRARY, d), rt.demos)

    # Serialization round-trips the chapter to JSON.
    js = to_json(textbook_chapter(13))
    @test occursin("Topos Causal Models", js)
    d = as_dict(textbook_chapter(5))
    @test d["number"] == 5
    @test "ket" in d["runnable_demos"]
end
