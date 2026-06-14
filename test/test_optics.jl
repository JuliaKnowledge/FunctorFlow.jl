# ============================================================================
# test_optics.jl — lenses & Para (gradient-learning foundations)
# ============================================================================

using Test
using FunctorFlow
const Cat = FunctorFlow.Cat

@testset "Record lens is very-well-behaved" begin
    l = Cat.record_lens([:a1, :a2], [:b1, :b2])
    @test Cat.lens_get_put(l)        # put(s, get s) = s
    @test Cat.lens_put_get(l)        # get(put s a) = a
    @test Cat.lens_put_put(l)        # put(put s a) a' = put s a'
    @test Cat.is_very_well_behaved(l)
    # concrete behaviour: focus on the first component
    @test l.get((:a1, :b2)) == :a1
    @test l.put(((:a1, :b2), :a2)) == (:a2, :b2)
end

@testset "Lens composition preserves the laws; identity" begin
    l = Cat.record_lens([:a1, :a2], [:b1, :b2])
    @test Cat.is_very_well_behaved(Cat.lens_compose(l, Cat.lens_id([:a1, :a2])))
    @test Cat.is_very_well_behaved(Cat.lens_id([:s1, :s2]))
end

@testset "A bad lens violates a law" begin
    S = Any[(:a1, :b1), (:a2, :b1)]
    SA = Cat.FinSet(Any[(s, a) for s in S for a in [:a1, :a2]])
    bad = Cat.Lens(S, [:a1, :a2],
        Cat.FinFunction(Cat.FinSet(S), Cat.FinSet([:a1, :a2]), Dict{Any,Any}((a, b) => a for (a, b) in S)),
        Cat.FinFunction(SA, Cat.FinSet(S), Dict{Any,Any}((s, a) => s for (s, a) in SA.elements)))  # put ignores a
    @test !Cat.lens_put_get(bad)
    @test !Cat.is_very_well_behaved(bad)
end

@testset "Para: learnable layers compose" begin
    f = Cat.ParaMap([:w0, :w1], [:x], [:y0, :y1],
        Cat.FinFunction(Cat.FinSet(Any[(p, a) for p in [:w0, :w1] for a in [:x]]),
                        Cat.FinSet([:y0, :y1]), Dict{Any,Any}((:w0, :x) => :y0, (:w1, :x) => :y1)))
    @test Cat.para_apply(f, :w0, :x) == :y0
    @test Cat.para_apply(f, :w1, :x) == :y1
    g = Cat.para_compose(f, Cat.para_id([:y0, :y1]))   # params become Q×P
    @test Cat.para_apply(g, (:unit, :w1), :x) == :y1
end

@testset "Lens-law Lean certificate renders" begin
    cert = render_lens_certificate(Cat.record_lens([:a1, :a2], [:b1, :b2]))
    @test occursin("LensDecl", cert)
    @test occursin("veryWellBehaved", cert)
    @test occursin("native_decide", cert)
end
