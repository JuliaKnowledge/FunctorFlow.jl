# ============================================================================
# test_cat_causal.jl — the causal/counterfactual layer on the Cat kernel
# ============================================================================

using Test
using FunctorFlow
const Cat = FunctorFlow.Cat

ex = build_causal_capstone_example()   # Z→X, Z→Y, X→M→Y ; treat X, outcome Y

@testset "Causal DAG is a category" begin
    Ccat = causal_category(ex.dag)
    @test Ccat isa FreeCat
    @test Cat.check_category_laws(Ccat)
    @test Cat.hom_cardinality(Ccat, :X, :Y) == 1     # X→M→Y
    @test Cat.hom_cardinality(Ccat, :Y, :X) == 0
    # it is Lean-certifiable as a category
    @test occursin("isCategory", render_cat_certificate(Ccat))
end

@testset "Intervention is a functor (the mutilation)" begin
    iv = intervention_functor(ex.dag, :X)
    @test Cat.is_functorial(iv.functor)
    # do(X) removes the single incoming edge Z→X
    @test length(iv.full.edges) - length(iv.mutilated.edges) == 1
    @test :X in iv.mutilated.objects
end

@testset "Twin network is a pushout" begin
    twin = twin_network(ex.dag, :X)
    @test Cat.verify_pushout(twin.pushout)
    @test twin.shared == [:Z]                         # background shared
    @test Set(twin.descendants) == Set([:X, :M, :Y])  # duplicated across worlds
    @test length(twin.pushout.apex) == 1 + 2 * 3      # Z + 3 factual + 3 counterfactual
end

@testset "Grand finale: every layer agrees" begin
    s = causal_capstone(; example=ex)
    @test s["causal_category"]["is_category"]
    @test s["intervention_functor"]["is_functor"]
    @test s["twin_network_pushout"]["is_pushout"]
    @test s["identifiability"]["identifiable"]          # back-door adjustable
    @test s["identifiability"]["estimand"] !== nothing
    @test s["counterfactual"]["identifiable"]
    @test s["counterfactual"]["expected_direction"] == 1
end

@testset "Twin network diagram (for visualization)" begin
    td = twin_causal_diagram(ex.dag, :X)
    @test td isa Diagram
    @test haskey(td.objects, :Z)        # shared background, once
    @test haskey(td.objects, :f_X) && haskey(td.objects, :c_X)   # duplicated worlds
    @test haskey(td.objects, :f_Y) && haskey(td.objects, :c_Y)
end
