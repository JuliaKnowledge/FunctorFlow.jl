# ============================================================================
# Cat.jl — FunctorFlow's verified categorical kernel
#
# A small, pure-Julia (no Catlab dependency) layer of *concrete, finite*
# category theory: finite sets and functions, free categories on finite DAGs,
# functors, Set-valued functors (C-Sets), natural transformations, and the
# Yoneda machinery. "Concrete and finite" is the point — every law
# (associativity, identity, functoriality, naturality, the Yoneda bijection)
# is actually *checkable* by enumeration, so this is a foundation with teeth
# rather than categorical vocabulary over plain structs.
#
# Layering:
#   FinSet / FinFunction                  — the category FinSet (the semantic target)
#   AbstractCategory / FreeCat            — small categories (free on a DAG)
#   FinFunctor                            — functors between small categories
#   SetFunctor                            — copresheaves C → FinSet (i.e. C-Sets)
#   CatNatTrans                           — natural transformations of SetFunctors
#   representable_functor / yoneda_*      — the Yoneda embedding & lemma (yoneda.jl)
#
# Interface verbs (`compose`, `id`, `dom`, `cod`, `ob`, `homset`) live in this
# submodule to avoid clashing with Base / Catlab; access them as `Cat.compose`.
# ============================================================================

module Cat

export FinSet, FinFunction
export AbstractCategory, FreeCat, PathMor
export FinFunctor, SetFunctor, CatNatTrans
export objects, homset, compose, id, dom, cod, ob, hommap
export is_functorial, check_category_laws, is_natural, hom_cardinality
# Yoneda layer (defined in yoneda.jl)
export representable_functor, representable_presheaf
export yoneda_map, yoneda_inverse, yoneda_lemma_holds, is_representable

# ----------------------------------------------------------------------------
# FinSet & FinFunction — the category FinSet
# ----------------------------------------------------------------------------

"""
    FinSet(elements)

A finite set, carried as the vector of its (hashable) elements. Equality is
by underlying set, so element order is irrelevant.
"""
struct FinSet
    elements::Vector{Any}
    FinSet(xs) = new(collect(xs))
end

Base.length(s::FinSet) = length(s.elements)
Base.in(x, s::FinSet) = x in s.elements
Base.collect(s::FinSet) = copy(s.elements)
Base.isempty(s::FinSet) = isempty(s.elements)
Base.:(==)(a::FinSet, b::FinSet) = Set(a.elements) == Set(b.elements)
Base.hash(s::FinSet, h::UInt) = hash(Set(s.elements), h)
Base.show(io::IO, s::FinSet) = print(io, "FinSet(", s.elements, ")")

"""
    FinFunction(dom::FinSet, cod::FinSet, pairs)

A total function `dom → cod`, given by an element-to-element mapping. Construction
validates totality and that every image lands in `cod`.
"""
struct FinFunction
    dom::FinSet
    cod::FinSet
    map::Dict{Any, Any}
end

function FinFunction(dom::FinSet, cod::FinSet, pairs)
    m = Dict{Any, Any}()
    for (k, v) in pairs
        m[k] = v
    end
    for x in dom.elements
        haskey(m, x) || throw(ArgumentError("FinFunction is not total: missing image for $(repr(x))"))
        m[x] in cod || throw(ArgumentError("FinFunction image $(repr(m[x])) of $(repr(x)) ∉ codomain"))
    end
    FinFunction(dom, cod, m)
end

(f::FinFunction)(x) = f.map[x]
dom(f::FinFunction) = f.dom
cod(f::FinFunction) = f.cod

"""`id(s::FinSet)` — the identity function on a finite set."""
id(s::FinSet) = FinFunction(s, s, Dict{Any,Any}(x => x for x in s.elements))

"""`compose(f, g)` — diagrammatic composition `f` then `g` (`g ∘ f`)."""
function compose(f::FinFunction, g::FinFunction)
    f.cod == g.dom || throw(ArgumentError("FinFunctions not composable: cod(f) ≠ dom(g)"))
    FinFunction(f.dom, g.cod, Dict{Any,Any}(x => g.map[f.map[x]] for x in f.dom.elements))
end

Base.:(==)(f::FinFunction, g::FinFunction) =
    f.dom == g.dom && f.cod == g.cod && all(f.map[x] == g.map[x] for x in f.dom.elements)

"""
    _uf_find!(parent::Vector{Int}, i::Int) -> Int

Root of `i` in an integer union-find forest, with path-halving (mutates
`parent`). Shared by the quotient/colimit constructions (`coequalizer`,
`colimit`, `left_kan`) that glue tagged elements into equivalence classes.
"""
function _uf_find!(parent::Vector{Int}, i::Int)
    while parent[i] != i
        parent[i] = parent[parent[i]]
        i = parent[i]
    end
    i
end

# ----------------------------------------------------------------------------
# Small categories: abstract interface + free category on a finite DAG
# ----------------------------------------------------------------------------

"""Supertype for small categories exposing `objects`, `homset`, `compose`, `id`."""
abstract type AbstractCategory end

"""
    PathMor(dom, cod, edges)

A morphism of a [`FreeCat`](@ref): a directed path `dom → cod` recorded as its
sequence of generating-edge names (empty = identity).
"""
struct PathMor
    dom::Symbol
    cod::Symbol
    edges::Vector{Symbol}
end

dom(f::PathMor) = f.dom
cod(f::PathMor) = f.cod
Base.:(==)(a::PathMor, b::PathMor) = a.dom == b.dom && a.cod == b.cod && a.edges == b.edges
Base.hash(p::PathMor, h::UInt) = hash((p.dom, p.cod, p.edges), h)
Base.show(io::IO, p::PathMor) =
    print(io, "PathMor(", p.dom, "→", p.cod, isempty(p.edges) ? " id" : " via " * join(p.edges, "·"), ")")

"""
    FreeCat(objects, edges)

The free category on a finite directed graph: objects are the nodes and
morphisms are directed paths (composition is path concatenation, identities are
empty paths). `edges` is a vector of `(name, src, tgt)`. The graph must be
**acyclic** so that every hom-set is finite (a precondition for the Yoneda
machinery); a directed cycle raises an error.
"""
struct FreeCat <: AbstractCategory
    objects::Vector{Symbol}
    edges::Vector{Tuple{Symbol, Symbol, Symbol}}   # (name, src, tgt)
    # Inner constructor: normalise, validate endpoints, and reject cycles.
    function FreeCat(objects, edges)
        objs = Symbol[Symbol(o) for o in objects]
        es = Tuple{Symbol,Symbol,Symbol}[(Symbol(n), Symbol(s), Symbol(t)) for (n, s, t) in edges]
        objset = Set(objs)
        for (n, s, t) in es
            s in objset || throw(ArgumentError("edge $n has unknown source $s"))
            t in objset || throw(ArgumentError("edge $n has unknown target $t"))
        end
        C = new(objs, es)
        _assert_acyclic(C)
        C
    end
end

function _assert_acyclic(C::FreeCat)
    indeg = Dict{Symbol, Int}(o => 0 for o in C.objects)
    for (_, _, t) in C.edges
        indeg[t] += 1
    end
    queue = Symbol[o for o in C.objects if indeg[o] == 0]
    seen = 0
    while !isempty(queue)
        v = popfirst!(queue); seen += 1
        for (_, s, t) in C.edges
            s == v || continue
            indeg[t] -= 1
            indeg[t] == 0 && push!(queue, t)
        end
    end
    seen == length(C.objects) ||
        throw(ArgumentError("FreeCat generators contain a directed cycle ⇒ infinite hom-sets; " *
                            "the kernel currently supports free categories on finite DAGs"))
    nothing
end

Base.:(==)(a::FreeCat, b::FreeCat) = a.objects == b.objects && a.edges == b.edges

objects(C::FreeCat) = copy(C.objects)
ob(C::FreeCat) = objects(C)
dom(::FreeCat, f::PathMor) = f.dom
cod(::FreeCat, f::PathMor) = f.cod
id(::FreeCat, a::Symbol) = PathMor(Symbol(a), Symbol(a), Symbol[])

function compose(C::FreeCat, f::PathMor, g::PathMor)
    f.cod == g.dom || throw(ArgumentError("not composable in FreeCat: cod $(f.cod) ≠ dom $(g.dom)"))
    PathMor(f.dom, g.cod, vcat(f.edges, g.edges))
end

"""
    homset(C::FreeCat, a, b) -> Vector{PathMor}

All morphisms `a → b` — i.e. all directed paths, enumerated (finite by acyclicity).
"""
function homset(C::FreeCat, a, b)
    a = Symbol(a); b = Symbol(b)
    adj = Dict{Symbol, Vector{Tuple{Symbol,Symbol}}}()
    for (n, s, t) in C.edges
        push!(get!(adj, s, Tuple{Symbol,Symbol}[]), (n, t))
    end
    out = PathMor[]
    function dfs(cur, path)
        cur == b && push!(out, PathMor(a, b, copy(path)))
        for (n, t) in get(adj, cur, Tuple{Symbol,Symbol}[])
            push!(path, n); dfs(t, path); pop!(path)
        end
    end
    dfs(a, Symbol[])
    out
end

"""`hom_cardinality(C, a, b)` — `|Hom(a, b)|`."""
hom_cardinality(C::FreeCat, a, b) = length(homset(C, a, b))

"""
    check_category_laws(C::FreeCat) -> Bool

Verify the category axioms by enumeration: left/right identity for every
morphism, and associativity for every composable triple. (Holds by
construction for free categories — this exercises the implementation.)
"""
function check_category_laws(C::AbstractCategory)
    objs = objects(C)
    allmors = PathMor[]
    for a in objs, b in objs
        append!(allmors, homset(C, a, b))
    end
    # identity laws
    for f in allmors
        compose(C, id(C, f.dom), f) == f || return false
        compose(C, f, id(C, f.cod)) == f || return false
    end
    # associativity
    for f in allmors, g in allmors
        f.cod == g.dom || continue
        for h in allmors
            g.cod == h.dom || continue
            lhs = compose(C, compose(C, f, g), h)
            rhs = compose(C, f, compose(C, g, h))
            lhs == rhs || return false
        end
    end
    true
end

include("presented.jl")
include("functor.jl")
include("yoneda.jl")
include("topos.jl")
include("limits.jl")
include("adjunction.jl")
include("kan.jl")
include("kan_general.jl")
include("monads.jl")
include("comonad.jl")
include("learn.jl")
include("coalg.jl")
include("markov.jl")
include("enriched.jl")
include("optics.jl")
include("heyting.jl")
include("galois.jl")
include("grothendieck.jl")
include("sheaf.jl")
include("rel.jl")
include("poly.jl")
include("falgebra.jl")
include("coend.jl")
include("operad.jl")
include("twocat.jl")

# Finitely-presented categories (with relations)
export FinPresentedCat, normalize, commutative_square
# Subobject classifier (presheaf topos)
export subobject_classifier, omega_true, classify, is_subfunctor, verify_classifies

# Limits / colimits in FinSet
export ProductCone, CoproductCocone, EqualizerCone, CoequalizerCocone, PullbackCone, PushoutCocone
export product, coproduct, equalizer, coequalizer, pullback, pushout
export mediate, comediate
export verify_product, verify_coproduct, verify_equalizer, verify_coequalizer,
       verify_pullback, verify_pushout
# Adjunctions
export identity_functor, terminal_category, FunctorNatTrans, Adjunction
export is_adjunction, initial_object_adjunction, restrict
# Kan extensions along the terminal functor (colimit / limit)
export ColimitCocone, LimitCone, colimit, limit, verify_colimit, verify_limit
export left_kan_along_terminal, right_kan_along_terminal
# Kan extensions along an arbitrary functor
export left_kan, right_kan
# Monads / Kleisli
export Monad, KleisliMor, is_monad, kleisli_hom, kleisli_id, kleisli_compose, check_kleisli_laws
export monad_from_adjunction, identity_monad, closure_monad
# Comonads (context-dependent computation)
export Comonad, is_comonad, identity_comonad, comonad_from_adjunction
# Categorical deep learning: backprop as a functor (FinVect_n)
export LinMap, forward, backward, lin_id, lin_compose, lin_transpose, reverse_derivative
export transpose_is_functorial, finvect_category_laws, backprop_demo
# Coalgebras / automata (state machines & RNNs as F-coalgebras)
export MooreMachine, moore_step, moore_run, is_bisimulation, bisimilar, minimize, coalgebra_morphism
# Markov categories (probability & causality)
export Dist, dirac, prob, StochMap, markov_id, markov_compose, is_deterministic
export markov_copy, markov_discard, markov_tensor, markov_laws
export bayes_update
# Enriched categories / metric spaces (representation learning)
export MetricCat, metric_dist, is_lawvere_metric, is_enriched_functor, embedding_metric
# Lenses & Para (gradient-learning foundations)
export Lens, lens_id, lens_compose, lens_get_put, lens_put_get, lens_put_put
export is_very_well_behaved, record_lens
export ParaMap, para_id, para_compose, para_apply
# Heyting algebras / intuitionistic internal logic
export HeytingAlgebra, hle, hmeet, hjoin, htop, hbot, himply, hneg, is_heyting_algebra, cosieve_heyting
# Galois connections / formal concept analysis
export Poset, ple, is_poset, is_galois_connection, formal_concepts, is_formal_concept
# Grothendieck construction (category of elements)
export category_of_elements, elements_projection
# Grothendieck (co)topologies, sheaf condition, sheafification (sheaf.jl)
export Coverage, covering_sieves, is_grothendieck_topology
export matching_families, amalgamations, is_separated, is_sheaf
export separated_reflection
export span_site, span_sheaf, span_non_sheaf
# Rel + powerset (nondeterminism) monad
export RelMap, rel_id, rel_compose, rel_dagger, rel_laws
export powerset_unit, powerset_mult, kleisli_to_rel
# Polynomial functors (interfaces & dynamics)
export Poly, monomial, PolyMap, is_poly_morphism, poly_id, poly_compose, moore_to_poly
# F-algebras & catamorphisms (folds / recursion schemes)
export Signature, Term, terms_upto, FAlgebra, cata, cata_is_homomorphism, arithmetic_signature
# Coends & profunctors (attention-as-a-coend; coend = coequalizer of dinaturality)
export Profunctor, profunctor_diag
export CoendCocone, coend, coend_class, verify_coend
export EndCone, end_
# Operads / multicategories (compositional architectures & wiring)
export Operad, operad_ops, operad_arity, operad_id, operad_compose, operad_laws
export operad_act, operad_symmetry_laws, unary_monoid
export commutative_operad, associative_operad, wiring_operad, little_intervals_operad
# Strict 2-categories / bicategories
export TwoCategory, OneCell, TwoCell
export zerocells, vcomp, hcomp
export check_vertical_category_laws, check_horizontal_category_laws,
       check_interchange_law, check_two_category_laws
export deloop_monoid, cat_two_category
export vcompose, hcompose, identity_nat
export para_reparam_two_cell, para_is_bicategory_note

end # module Cat
