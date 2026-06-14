# ============================================================================
# test_limits.jl — limits & colimits in FinSet with verified universal props
# ============================================================================

using Test
using FunctorFlow
const Cat = FunctorFlow.Cat

A = Cat.FinSet([1, 2]); B = Cat.FinSet([:x, :y, :z])

@testset "Product" begin
    pc = Cat.product(A, B)
    @test length(pc.apex) == 6
    @test pc.proj1((1, :y)) == 1 && pc.proj2((1, :y)) == :y
    # mediating ⟨q1,q2⟩ commutes
    X = Cat.FinSet([:s, :t])
    q1 = Cat.FinFunction(X, A, [:s => 1, :t => 2])
    q2 = Cat.FinFunction(X, B, [:s => :x, :t => :z])
    u = Cat.mediate(pc, q1, q2)
    @test Cat.compose(u, pc.proj1) == q1 && Cat.compose(u, pc.proj2) == q2
    # full universal property (existence + uniqueness against probes)
    @test Cat.verify_product(pc, A, B)
    # a non-universal cone (constant 2nd projection) is rejected
    bad = Cat.ProductCone(pc.apex, pc.proj1,
        Cat.FinFunction(pc.apex, B, Dict{Any,Any}(p => :x for p in pc.apex.elements)))
    @test !Cat.verify_product(bad, A, B)
end

@testset "Coproduct" begin
    cc = Cat.coproduct(A, B)
    @test length(cc.apex) == 5
    @test cc.inj1(1) == (:inl, 1) && cc.inj2(:z) == (:inr, :z)
    @test Cat.verify_coproduct(cc, A, B)
end

@testset "Equalizer" begin
    f = Cat.FinFunction(A, B, [1 => :x, 2 => :y])
    g = Cat.FinFunction(A, B, [1 => :x, 2 => :z])
    eq = Cat.equalizer(f, g)
    @test collect(eq.apex) == [1]                # only element 1 has f=g
    @test Cat.verify_equalizer(eq)
    # mediate rejects a map that doesn't equalize
    X = Cat.FinSet([:s])
    nonEq = Cat.FinFunction(X, A, [:s => 2])
    @test_throws ArgumentError Cat.mediate(eq, nonEq)
end

@testset "Coequalizer" begin
    f = Cat.FinFunction(A, B, [1 => :x, 2 => :y])
    g = Cat.FinFunction(A, B, [1 => :x, 2 => :z])
    ceq = Cat.coequalizer(f, g)
    @test length(ceq.apex) == 2                  # :y and :z merged
    @test ceq.proj(:y) == ceq.proj(:z)
    @test ceq.proj(:x) != ceq.proj(:y)
    @test Cat.verify_coequalizer(ceq)
end

@testset "Pullback" begin
    Cc = Cat.FinSet([:p, :q])
    h1 = Cat.FinFunction(A, Cc, [1 => :p, 2 => :q])
    h2 = Cat.FinFunction(B, Cc, [:x => :p, :y => :p, :z => :q])
    pb = Cat.pullback(h1, h2)
    # fibre product: (1,:x),(1,:y),(2,:z)
    @test length(pb.apex) == 3
    @test all(pb.f(pb.p1(e)) == pb.g(pb.p2(e)) for e in pb.apex.elements)
    @test Cat.verify_pullback(pb)
end

@testset "Pushout" begin
    Cc = Cat.FinSet([:p, :q])
    s1 = Cat.FinFunction(Cc, A, [:p => 1, :q => 2])
    s2 = Cat.FinFunction(Cc, B, [:p => :x, :q => :y])
    po = Cat.pushout(s1, s2)
    @test Cat.verify_pushout(po)
    # the span legs are identified in the apex
    @test po.i1(1) == po.i2(:x) && po.i1(2) == po.i2(:y)
end
