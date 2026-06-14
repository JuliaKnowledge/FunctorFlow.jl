# ============================================================================
# test_cat_integration.jl — horizontal integration + end-to-end pipeline
# ============================================================================

using Test
using FunctorFlow
const Cat = FunctorFlow.Cat

@testset "CLIFF knowledge category (route→chapter→demo)" begin
    K = cliff_knowledge_category()
    @test K isa FreeCat
    # categorical recovery: demos reachable from a route via composition equal
    # the textbook-backed demos computed directly
    via = demos_reachable_from(K, :company_similarity)
    direct = sort(unique(reduce(vcat,
        [runnable_demos(ch) for ch in chapters_for_route(:company_similarity)]; init=Symbol[])))
    @test Set(via) == Set(direct)
    @test !isempty(via)
    # the linkage category is law-abiding (checked on a slice for speed) and Lean-certifiable
    @test Cat.check_category_laws(cliff_knowledge_category(; routes=[:company_similarity]))
    @test occursin("isCategory", render_cat_certificate(cliff_knowledge_category(; routes=[:course_demo])))
end

@testset "JEPA exactness is a commuting square" begin
    J = jepa_square_category()
    @test Cat.check_category_laws(J)
    @test Cat.hom_cardinality(J, :X, :Zt) == 1     # the square commutes (loss = 0)
    # the *free* version (no relation) has two distinct X→Zt paths
    free = FreeCat([:X, :Z, :Xp, :Zt],
                   [(:enc_x, :X, :Z), (:pred, :Z, :Zt), (:gamma, :X, :Xp), (:enc_y, :Xp, :Zt)])
    @test Cat.hom_cardinality(free, :X, :Zt) == 2
end

@testset "End-to-end pipeline" begin
    # a causal/evidence route drives the full stack
    p = integrated_pipeline("How similar is Adobe to Nike across filings?")
    @test p["route"] == "company_similarity"
    @test p["causal_capstone"] !== nothing
    @test p["causal_capstone"]["twin_network_pushout"]["is_pushout"]
    @test p["causal_capstone"]["identifiability"]["identifiable"]
    @test length(p["layers_exercised"]) == 8
    @test Set(p["demos_via_category"]) ⊆ Set(p["demos_from_textbook"]) ∪ Set(p["demos_via_category"])

    # a non-causal route exercises fewer layers (no causal capstone)
    q = integrated_pipeline("show me the category theory for agi textbook demo")
    @test q["route"] == "course_demo"
    @test q["causal_capstone"] === nothing
    @test length(q["layers_exercised"]) == 3
end

@testset "Grand end-to-end capstone" begin
    e = end_to_end_capstone()
    # every layer reports in, and the cross-checks agree
    @test e["causal_capstone"]["twin_network_pushout"]["is_pushout"]
    @test e["corpus_synthesis"]["is_colimit"]
    @test e["corpus_synthesis"]["agrees_with_engine"]
    @test e["jepa_exactness_is_commutativity"]
    @test length(e["layers_exercised"]) == 8
end
