# ============================================================================
# heyting.jl — Heyting algebras: the intuitionistic internal logic of a topos
# (included into module Cat)
#
# The truth values of a presheaf topos form a Heyting algebra, not a Boolean
# one: the subobject lattice (and, pointwise, each `Ω(c)` of cosieves) carries
# `∧, ∨, ⇒, ¬, ⊤, ⊥` satisfying the Heyting adjunction `z∧x ≤ y ⇔ z ≤ (x⇒y)`.
# This is the home of neuro-symbolic / intuitionistic reasoning over the topos
# built in `topos.jl`. Everything is finite, hence decidable and Lean-certifiable.
# ============================================================================

"""
    HeytingAlgebra(elements, leq)

A finite lattice (given by its order `leq`) from which the Heyting operations
are derived; `is_heyting_algebra` checks it really is one.
"""
struct HeytingAlgebra
    elements::Vector{Any}
    leq::Dict{Tuple{Any,Any}, Bool}
    function HeytingAlgebra(elements, leq)
        els = collect(elements)
        d = Dict{Tuple{Any,Any}, Bool}((a, b) => Bool(v) for ((a, b), v) in leq)
        for x in els, y in els
            haskey(d, (x, y)) || throw(ArgumentError("missing order ($x, $y)"))
        end
        new(els, d)
    end
end

hle(H::HeytingAlgebra, x, y) = H.leq[(x, y)]

function hmeet(H::HeytingAlgebra, x, y)
    lbs = [w for w in H.elements if hle(H, w, x) && hle(H, w, y)]
    for m in lbs
        all(hle(H, w, m) for w in lbs) && return m
    end
    error("no greatest lower bound for $x, $y (not a lattice)")
end

function hjoin(H::HeytingAlgebra, x, y)
    ubs = [w for w in H.elements if hle(H, x, w) && hle(H, y, w)]
    for j in ubs
        all(hle(H, j, w) for w in ubs) && return j
    end
    error("no least upper bound for $x, $y (not a lattice)")
end

# The unique element satisfying `pred`, or `error(what)` if there is none.
function _unique_bound(H::HeytingAlgebra, pred, what::AbstractString)
    i = findfirst(pred, H.elements)
    i === nothing && error("no $what")
    H.elements[i]
end

"""`htop(H)` — the top element `⊤` (above every element)."""
htop(H::HeytingAlgebra) = _unique_bound(H, t -> all(hle(H, x, t) for x in H.elements), "top")
"""`hbot(H)` — the bottom element `⊥` (below every element)."""
hbot(H::HeytingAlgebra) = _unique_bound(H, b -> all(hle(H, b, x) for x in H.elements), "bottom")

"""`himply(H, x, y)` — Heyting implication `x ⇒ y`: the greatest `z` with `z∧x ≤ y`."""
function himply(H::HeytingAlgebra, x, y)
    cands = [z for z in H.elements if hle(H, hmeet(H, z, x), y)]
    for g in cands
        all(hle(H, z, g) for z in cands) && return g
    end
    error("no Heyting implication $x ⇒ $y")
end

"""`hneg(H, x)` — intuitionistic negation `¬x = (x ⇒ ⊥)`."""
hneg(H::HeytingAlgebra, x) = himply(H, x, hbot(H))

"""
    is_heyting_algebra(H) -> Bool

Verify `H` is a Heyting algebra: it is a (bounded) lattice and the Heyting
adjunction `z∧x ≤ y ⇔ z ≤ (x⇒y)` holds for all `x, y, z`.
"""
function is_heyting_algebra(H::HeytingAlgebra)
    try
        htop(H); hbot(H)
        for x in H.elements, y in H.elements
            hmeet(H, x, y); hjoin(H, x, y); himply(H, x, y)
        end
        for x in H.elements, y in H.elements, z in H.elements
            hle(H, hmeet(H, z, x), y) == hle(H, z, himply(H, x, y)) || return false
        end
        true
    catch
        false
    end
end

"""
    cosieve_heyting(C, c) -> HeytingAlgebra

The Heyting algebra of cosieves on `c` (ordered by inclusion) — the
intuitionistic truth-value algebra `Ω(c)` of the topos `[C, Set]` at `c`.
"""
function cosieve_heyting(C::AbstractCategory, c)
    cos = _cosieves(C, c)
    leq = Dict{Tuple{Any,Any}, Bool}()
    for R in cos, S in cos
        leq[(R, S)] = Set(R) ⊆ Set(S)
    end
    HeytingAlgebra(cos, leq)
end
