# ============================================================================
# galois.jl — Galois connections & formal concept analysis
# (included into module Cat)
#
# A Galois connection `f ⊣ g` between posets (`f p ≤ q ⇔ p ≤ g q`) is an
# adjunction between thin categories — the simplest, most ubiquitous adjunction.
# Formal Concept Analysis (concept learning) is exactly the Galois connection
# induced by a formal context (objects × attributes): the concept lattice is the
# fixed points. Finite, decidable, Lean-certifiable.
# ============================================================================

"""
    Poset(elements, leq)

A finite poset given by its order relation `leq`.
"""
struct Poset
    elements::Vector{Any}
    leq::Dict{Tuple{Any,Any}, Bool}
    function Poset(elements, leq)
        els = collect(elements)
        d = Dict{Tuple{Any,Any}, Bool}((a, b) => Bool(v) for ((a, b), v) in leq)
        for x in els, y in els
            haskey(d, (x, y)) || throw(ArgumentError("missing order ($x, $y)"))
        end
        new(els, d)
    end
end

ple(P::Poset, x, y) = P.leq[(x, y)]

"""`is_poset(P)` — reflexive, antisymmetric, transitive."""
function is_poset(P::Poset)
    for x in P.elements
        ple(P, x, x) || return false
    end
    for x in P.elements, y in P.elements
        (ple(P, x, y) && ple(P, y, x)) && x != y && return false
        for z in P.elements
            (ple(P, x, y) && ple(P, y, z) && !ple(P, x, z)) && return false
        end
    end
    true
end

"""
    is_galois_connection(P, Q, f, g) -> Bool

Is `(f : P→Q, g : Q→P)` a (monotone) Galois connection `f ⊣ g`, i.e.
`f(p) ≤ q ⇔ p ≤ g(q)` for all `p, q`.
"""
function is_galois_connection(P::Poset, Q::Poset, f::AbstractDict, g::AbstractDict)
    fd = Dict(k => v for (k, v) in f); gd = Dict(k => v for (k, v) in g)
    for p in P.elements, q in Q.elements
        ple(Q, fd[p], q) == ple(P, p, gd[q]) || return false
    end
    true
end

# ----------------------------------------------------------------------------
# Formal Concept Analysis: the Galois connection of a formal context
# ----------------------------------------------------------------------------

"""
    formal_concepts(objects, attributes, incidence) -> Vector{NamedTuple}

The formal concepts `(extent, intent)` of a context, where `incidence` is the
set of `(object, attribute)` pairs. A concept is a pair of an object set `A` and
an attribute set `B` with `A' = B` and `B' = A` under the derivation operators
(the Galois connection): `A' = {attrs shared by all of A}`,
`B' = {objects having all of B}`. These are the fixed points (closed sets).
"""
function formal_concepts(objects, attributes, incidence)
    O = collect(objects); At = collect(attributes)
    inc = Set(incidence)
    # derivation operators
    up(A) = Set(m for m in At if all((o, m) in inc for o in A))           # objects → shared attributes
    down(B) = Set(o for o in O if all((o, m) in inc for m in B))          # attributes → common objects
    concepts = NamedTuple[]
    seen = Set{Any}()
    # every concept's extent is down(B) for some attribute set; generate from attribute-closed sets
    for o in O
        ext = down(up(Set([o])))   # closure of each singleton object extent
        key = sort(collect(ext); by=string)
        key in seen && continue
        push!(seen, key)
        push!(concepts, (extent=ext, intent=up(ext)))
    end
    # also the top concept (all objects) and bottom (all attributes)
    for ext in (down(Set(At)), Set(O))
        key = sort(collect(ext); by=string)
        key in seen || (push!(seen, key); push!(concepts, (extent=ext, intent=up(ext))))
    end
    concepts
end

"""
    is_formal_concept(extent, intent, objects, attributes, incidence) -> Bool

Check `(extent, intent)` is a formal concept (a fixed point of the Galois
connection): `extent' = intent` and `intent' = extent`.
"""
function is_formal_concept(extent, intent, objects, attributes, incidence)
    O = collect(objects); At = collect(attributes); inc = Set(incidence)
    up(A) = Set(m for m in At if all((o, m) in inc for o in A))
    down(B) = Set(o for o in O if all((o, m) in inc for m in B))
    Set(up(extent)) == Set(intent) && Set(down(intent)) == Set(extent)
end
