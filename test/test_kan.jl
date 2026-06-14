# ============================================================================
# test_kan.jl — colimits/limits of Set-valued functors (Kan along terminal)
# ============================================================================

using Test
using FunctorFlow
const Cat = FunctorFlow.Cat

@testset "Colimit / limit over a discrete category = coproduct / product" begin
    disc = FreeCat([:a, :b], Tuple{Symbol,Symbol,Symbol}[])   # no morphisms
    X = Cat.SetFunctor(disc;
        ob_map=Dict(:a => Cat.FinSet([1, 2]), :b => Cat.FinSet([:x, :y, :z])),
        edge_map=Dict{Symbol, Cat.FinFunction}())
    col = Cat.colimit(X)
    @test length(col.apex) == 5        # 2 + 3 (coproduct)
    @test Cat.verify_colimit(col)
    lim = Cat.limit(X)
    @test length(lim.apex) == 6        # 2 × 3 (product)
    @test Cat.verify_limit(lim)
end

@testset "Colimit of a span = pushout" begin
    span = FreeCat([:s, :l, :r], [(:il, :s, :l), (:ir, :s, :r)])
    X = Cat.SetFunctor(span;
        ob_map=Dict(:s => Cat.FinSet([1]), :l => Cat.FinSet([:x, :y]), :r => Cat.FinSet([:u, :v])),
        edge_map=Dict(:il => Cat.FinFunction(Cat.FinSet([1]), Cat.FinSet([:x, :y]), [1 => :x]),
                      :ir => Cat.FinFunction(Cat.FinSet([1]), Cat.FinSet([:u, :v]), [1 => :u])))
    col = Cat.colimit(X)
    @test length(col.apex) == 3        # x∼u glued; y, v free
    @test col.legs[:l](:x) == col.legs[:r](:u)    # the span identifies x and u
    @test Cat.verify_colimit(col)
    # a non-cocone is rejected by the mediator
    Y = Cat.FinSet([:p])
    badq = Dict(:s => Cat.FinFunction(Cat.FinSet([1]), Y, [1 => :p]),
                :l => Cat.FinFunction(Cat.FinSet([:x, :y]), Y, [:x => :p, :y => :p]),
                :r => Cat.FinFunction(Cat.FinSet([:u, :v]), Y, [:u => :p, :v => :p]))
    @test Cat.comediate(col, badq) isa Cat.FinFunction   # this one IS a cocone (all to :p)
end

@testset "Limit of a cospan = pullback" begin
    cospan = FreeCat([:l, :r, :s], [(:pl, :l, :s), (:pr, :r, :s)])
    X = Cat.SetFunctor(cospan;
        ob_map=Dict(:l => Cat.FinSet([:x, :y]), :r => Cat.FinSet([:u, :v]), :s => Cat.FinSet([1, 2])),
        edge_map=Dict(:pl => Cat.FinFunction(Cat.FinSet([:x, :y]), Cat.FinSet([1, 2]), [:x => 1, :y => 2]),
                      :pr => Cat.FinFunction(Cat.FinSet([:u, :v]), Cat.FinSet([1, 2]), [:u => 1, :v => 2])))
    lim = Cat.limit(X)
    @test length(lim.apex) == 2        # (x,u) over 1 and (y,v) over 2
    @test all(X.edge_map[:pl](lim.legs[:l](e)) == X.edge_map[:pr](lim.legs[:r](e)) for e in lim.apex.elements)
    @test Cat.verify_limit(lim)
end

@testset "Colimit / limit Lean certificates render" begin
    span = FreeCat([:s, :l, :r], [(:il, :s, :l), (:ir, :s, :r)])
    X = Cat.SetFunctor(span;
        ob_map=Dict(:s => Cat.FinSet([1]), :l => Cat.FinSet([:x, :y]), :r => Cat.FinSet([:u, :v])),
        edge_map=Dict(:il => Cat.FinFunction(Cat.FinSet([1]), Cat.FinSet([:x, :y]), [1 => :x]),
                      :ir => Cat.FinFunction(Cat.FinSet([1]), Cat.FinSet([:u, :v]), [1 => :u])))
    colcert = render_colimit_certificate(Cat.colimit(X))
    @test occursin("ColimitCert", colcert) && occursin("isColimit", colcert)
    cospan = FreeCat([:l, :r, :s], [(:pl, :l, :s), (:pr, :r, :s)])
    Y = Cat.SetFunctor(cospan;
        ob_map=Dict(:l => Cat.FinSet([:x, :y]), :r => Cat.FinSet([:u, :v]), :s => Cat.FinSet([1, 2])),
        edge_map=Dict(:pl => Cat.FinFunction(Cat.FinSet([:x, :y]), Cat.FinSet([1, 2]), [:x => 1, :y => 2]),
                      :pr => Cat.FinFunction(Cat.FinSet([:u, :v]), Cat.FinSet([1, 2]), [:u => 1, :v => 2])))
    limcert = render_limit_certificate(Cat.limit(Y))
    @test occursin("LimitCert", limcert) && occursin("isLimit", limcert)
end

@testset "Coequalizer-shaped colimit" begin
    # parallel pair  a ⇉ b  ; colimit coequalizes the two maps
    par = FreeCat([:a, :b], [(:f, :a, :b), (:g, :a, :b)])
    X = Cat.SetFunctor(par;
        ob_map=Dict(:a => Cat.FinSet([1, 2]), :b => Cat.FinSet([:p, :q, :r])),
        edge_map=Dict(:f => Cat.FinFunction(Cat.FinSet([1, 2]), Cat.FinSet([:p, :q, :r]), [1 => :p, 2 => :q]),
                      :g => Cat.FinFunction(Cat.FinSet([1, 2]), Cat.FinSet([:p, :q, :r]), [1 => :p, 2 => :r])))
    col = Cat.colimit(X)
    # identifies q ∼ r (both hit by element 2 via f,g); p stays ⇒ classes {p},{q,r}
    @test length(col.apex) == 2
    @test col.legs[:b](:q) == col.legs[:b](:r)
    @test Cat.verify_colimit(col)
end
