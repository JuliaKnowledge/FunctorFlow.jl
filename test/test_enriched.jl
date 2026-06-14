# ============================================================================
# test_enriched.jl — enriched categories / metric spaces (Lawvere)
# ============================================================================

using Test
using FunctorFlow
const Cat = FunctorFlow.Cat

@testset "An embedding is a Lawvere metric space (enriched category)" begin
    M = Cat.embedding_metric(Dict(:a => [0, 0], :b => [3, 0], :c => [3, 4]); metric=:l1)
    @test Cat.is_lawvere_metric(M)               # identity + triangle inequality
    @test Cat.metric_dist(M, :a, :a) == 0
    @test Cat.metric_dist(M, :a, :c) == 7        # |3| + |4|
    @test Cat.metric_dist(M, :a, :c) <= Cat.metric_dist(M, :a, :b) + Cat.metric_dist(M, :b, :c)
    @test Cat.is_lawvere_metric(Cat.embedding_metric(Dict(:p => [1, 2, 3], :q => [0, 0, 0]); metric=:linf))
end

@testset "Non-expansive map = enriched functor" begin
    M = Cat.embedding_metric(Dict(:a => [0, 0], :b => [3, 0], :c => [3, 4]); metric=:l1)
    N = Cat.embedding_metric(Dict(:p => [0, 0], :q => [1, 0]); metric=:l1)
    # collapsing b,c to q is 1-Lipschitz here
    @test Cat.is_enriched_functor(M, N, Dict(:a => :p, :b => :q, :c => :q))
    # an expanding map is NOT an enriched functor
    M2 = Cat.embedding_metric(Dict(:a => [0], :b => [1]); metric=:l1)
    N2 = Cat.embedding_metric(Dict(:p => [0], :q => [5]); metric=:l1)
    @test !Cat.is_enriched_functor(M2, N2, Dict(:a => :p, :b => :q))   # d=1 ↦ d=5
end

@testset "Triangle violations are rejected; cert renders" begin
    bad = Cat.MetricCat([:x, :y, :z],
        Dict((:x, :x) => 0, (:y, :y) => 0, (:z, :z) => 0,
             (:x, :y) => 1, (:y, :z) => 1, (:x, :z) => 5,
             (:y, :x) => 1, (:z, :y) => 1, (:z, :x) => 5))
    @test !Cat.is_lawvere_metric(bad)            # 5 > 1 + 1
    cert = render_metric_certificate(Cat.embedding_metric(Dict(:a => [0, 0], :b => [3, 4]); metric=:l1))
    @test occursin("MetricDecl", cert) && occursin("isLawvereMetric", cert)
end
