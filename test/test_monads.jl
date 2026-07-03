# ============================================================================
# test_monads.jl — monads, the monad laws, Kleisli categories
# ============================================================================

using Test
using FunctorFlow
const Cat = FunctorFlow.Cat

chain = FreeCat([:a, :b, :c], [(:f, :a, :b), (:g, :b, :c)])

@testset "Identity monad" begin
    m = Cat.identity_monad(chain)
    @test Cat.is_monad(m)
    @test Cat.check_kleisli_laws(m)
    # Kleisli hom a→b is Hom_C(a, T(b)) = Hom_C(a, b) since T = Id
    @test length(Cat.kleisli_hom(m, :a, :c)) == 1
    @test Cat.kleisli_id(m, :a) == Cat.id(chain, :a)
end

@testset "Closure-operator monad on a poset" begin
    # close everything ≤ b up to b: T(a)=b, T(b)=b, T(c)=c
    m = Cat.closure_monad(chain, Dict(:a => :b, :b => :b, :c => :c))
    @test Cat.is_monad(m)
    @test Cat.check_kleisli_laws(m)
    @test m.functor.ob_map[:a] == :b               # T(a) = b
    # η_a : a → T(a) = b is the generator f
    @test Cat.kleisli_id(m, :a) == PathMor(:a, :b, [:f])

    # T is not injective on objects (T(a) = T(b) = b), so Kleisli codomains
    # must be tracked explicitly instead of reconstructed from g.cod alone.
    kaa = only(Cat.kleisli_hom(m, :a, :a))
    kab = only(Cat.kleisli_hom(m, :a, :b))
    @test kaa != kab
    @test kaa == PathMor(:a, :b, [:f])
    @test kab == PathMor(:a, :b, [:f])

    # a non-monotone "closure" is rejected (T can't be a functor)
    @test_throws ArgumentError Cat.closure_monad(chain, Dict(:a => :c, :b => :a, :c => :c))
end

@testset "Monad from an adjunction" begin
    Id = Cat.identity_functor(chain)
    comps = Dict(o => Cat.id(chain, o) for o in Cat.objects(chain))
    η = Cat.FunctorNatTrans(Id, Id; components=comps)
    adj = Cat.Adjunction(Id, Id, η, η)         # Id ⊣ Id
    @test Cat.is_adjunction(adj)
    m = Cat.monad_from_adjunction(adj)
    @test Cat.is_monad(m)                       # the induced monad is the identity monad
    @test all(m.functor.ob_map[o] == o for o in Cat.objects(chain))
end

@testset "Comonads (dual of monads)" begin
    @test Cat.is_comonad(Cat.identity_comonad(chain))
    @test Cat.check_kleisli_laws(Cat.identity_monad(chain))   # sanity: monad side still holds
    Id = Cat.identity_functor(chain)
    comps = Dict(o => Cat.id(chain, o) for o in Cat.objects(chain))
    η = Cat.FunctorNatTrans(Id, Id; components=comps)
    adj = Cat.Adjunction(Id, Id, η, η)
    @test Cat.is_comonad(Cat.comonad_from_adjunction(adj))    # the cofree comonad of Id⊣Id
end
