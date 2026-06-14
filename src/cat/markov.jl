# ============================================================================
# markov.jl — Markov categories: probability and causality, categorically
# (included into module Cat)
#
# The Kleisli category of the finite-distribution monad is a *Markov category*:
# objects are finite sets, morphisms `A → B` are stochastic maps (Markov
# kernels), composition is Chapman–Kolmogorov, and the cartesian product gives
# `copy`/`discard` (the comonoid structure). This is the categorical home of
# probability and causal inference: a causal DAG factorises a joint as a
# composite of mechanisms, and Bayesian updating is the disintegration. Weights
# are exact `Rational`s, so everything is computable (and the stochasticity law
# is Lean-certifiable; see `Markov.lean`).
# ============================================================================

"""
    Dist(pairs)

A finite probability distribution: outcome ↦ probability (exact `Rational`),
required to sum to 1.
"""
struct Dist
    support::Dict{Any, Rational{Int}}
    function Dist(pairs)
        d = Dict{Any, Rational{Int}}()
        for (k, v) in pairs
            d[k] = get(d, k, 0 // 1) + Rational{Int}(v)
        end
        s = sum(values(d); init=0 // 1)
        s == 1 // 1 || throw(ArgumentError("distribution must sum to 1, got $s"))
        new(d)
    end
end

"""`dirac(x)` — the point mass at `x`."""
dirac(x) = Dist(Dict{Any, Rational{Int}}(x => 1 // 1))

prob(d::Dist, x) = get(d.support, x, 0 // 1)
Base.:(==)(a::Dist, b::Dist) = a.support == b.support

"""
    StochMap(dom, cod, kernel)

A morphism of the Markov category: a stochastic map `dom → Dist(cod)`, i.e. a
Markov kernel `kernel[a]` for each `a ∈ dom`.
"""
struct StochMap
    dom::Vector{Any}
    cod::Vector{Any}
    kernel::Dict{Any, Dist}
    function StochMap(dom, cod, kernel)
        d = collect(dom); c = collect(cod); cset = Set(c)
        k = Dict{Any, Dist}(a => kernel[a] for a in d)
        for a in d
            for o in keys(k[a].support)
                o in cset || throw(ArgumentError("kernel image $o ∉ codomain"))
            end
        end
        new(d, c, k)
    end
end

Base.:(==)(f::StochMap, g::StochMap) =
    Set(f.dom) == Set(g.dom) && Set(f.cod) == Set(g.cod) &&
    all(f.kernel[a] == g.kernel[a] for a in f.dom)

"""`markov_id(A)` — the identity stochastic map (Dirac kernel)."""
markov_id(A) = StochMap(A, A, Dict{Any, Dist}(a => dirac(a) for a in A))

"""`markov_compose(f, g)` — Kleisli/Chapman–Kolmogorov composition `g ∘ f`."""
function markov_compose(f::StochMap, g::StochMap)
    # `Dist` accumulates repeated outcomes, so we hand it the (c, pb·pc) pairs directly.
    kernel = Dict{Any, Dist}(
        a => Dist((c, pb * pc) for (b, pb) in f.kernel[a].support
                               for (c, pc) in g.kernel[b].support)
        for a in f.dom)
    StochMap(f.dom, g.cod, kernel)
end

"""`is_deterministic(f)` — does every kernel concentrate on a single outcome?"""
is_deterministic(f::StochMap) = all(length(f.kernel[a].support) == 1 for a in f.dom)

"""`markov_copy(A)` — the comonoid copy `A → A×A`, `a ↦ δ_{(a,a)}` (a Markov-category structure map)."""
markov_copy(A) = StochMap(A, [(a, a) for a in A], Dict{Any, Dist}(a => dirac((a, a)) for a in A))

"""`markov_discard(A)` — the counit `A → 1`, `a ↦ δ_{()}` (every Markov morphism is discardable)."""
markov_discard(A) = StochMap(A, [()], Dict{Any, Dist}(a => dirac(()) for a in A))

"""`markov_tensor(f, g)` — the monoidal product `f ⊗ g : A×C → B×D` (independent kernels)."""
function markov_tensor(f::StochMap, g::StochMap)
    dom = [(a, c) for a in f.dom for c in g.dom]
    cod = [(b, d) for b in f.cod for d in g.cod]
    kernel = Dict{Any, Dist}(
        (a, c) => Dist(((b, d), pb * pd) for (b, pb) in f.kernel[a].support
                                         for (d, pd) in g.kernel[c].support)
        for (a, c) in dom)
    StochMap(dom, cod, kernel)
end

"""
    markov_laws(maps) -> Bool

Check the Markov-category (Kleisli) laws on a set of stochastic maps: identity
and associativity of composition.
"""
function markov_laws(maps::AbstractVector{StochMap})
    for f in maps
        markov_compose(markov_id(f.dom), f) == f || return false
        markov_compose(f, markov_id(f.cod)) == f || return false
    end
    for f in maps, g in maps
        Set(f.cod) == Set(g.dom) || continue
        for h in maps
            Set(g.cod) == Set(h.dom) || continue
            markov_compose(markov_compose(f, g), h) == markov_compose(f, markov_compose(g, h)) || return false
        end
    end
    true
end

# ----------------------------------------------------------------------------
# Bayesian inference in the Markov category
# ----------------------------------------------------------------------------

"""
    bayes_update(prior::Dist, likelihood::StochMap, observation) -> Dist

Bayesian updating as disintegration in the Markov category: the posterior over
the hidden variable given `observation`, `P(x | obs) ∝ prior(x)·likelihood(x)(obs)`.
"""
function bayes_update(prior::Dist, likelihood::StochMap, observation)
    weights = Dict{Any, Rational{Int}}()
    for (x, px) in prior.support
        w = px * prob(likelihood.kernel[x], observation)
        w == 0 // 1 || (weights[x] = w)
    end
    z = sum(values(weights); init=0 // 1)
    z == 0 // 1 && throw(ArgumentError("observation has probability zero under the model"))
    Dist(Dict{Any, Rational{Int}}(x => w / z for (x, w) in weights))
end
