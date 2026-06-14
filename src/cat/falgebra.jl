# ============================================================================
# falgebra.jl — F-algebras & catamorphisms (folds / recursion schemes)
# (included into module Cat)
#
# The dual of the coalgebra/automaton story (`coalg.jl`). For a signature
# functor `F`, an F-algebra `(carrier, ops)` interprets the constructors; the
# **initial algebra** is the term algebra, and the unique map out of it is the
# **catamorphism** (a fold). This is the categorical home of structural
# recursion: an evaluator, a tree-RNN, or any fold *is* a catamorphism. (Folds =
# induction = initial algebras; the Moore-machine unfolds of `coalg.jl` =
# coinduction = final coalgebras.)
# ============================================================================

"""A finite algebraic signature: constructor name ↦ arity."""
struct Signature
    constructors::Dict{Symbol, Int}
end
Signature(pairs...) = Signature(Dict{Symbol, Int}(Symbol(k) => Int(v) for (k, v) in pairs))

"""A term of a signature (a node of the initial/term algebra)."""
struct Term
    con::Symbol
    args::Vector{Term}
end
Term(con) = Term(Symbol(con), Term[])

"""
    terms_upto(sig, depth) -> Vector{Term}

All terms of `sig` of height ≤ `depth` (a finite approximation of the initial
algebra — finite because the depth is bounded).
"""
function terms_upto(sig::Signature, depth::Integer)
    base = Term[Term(c) for (c, a) in sig.constructors if a == 0]
    levels = Vector{Term}[base]
    for _ in 2:Int(depth)
        prev = vcat(levels...)
        nxt = Term[]
        for (c, a) in sig.constructors
            a == 0 && continue
            for combo in Iterators.product((prev for _ in 1:a)...)
                push!(nxt, Term(c, collect(combo)))
            end
        end
        push!(levels, nxt)
    end
    vcat(levels...)
end

"""
    FAlgebra(carrier, ops)

An F-algebra: a `carrier` set and, for each constructor, an operation
`ops[con] : carrier^arity → carrier` (taking the vector of child values).
"""
struct FAlgebra
    carrier::Vector{Any}
    ops::Dict{Symbol, Function}
end

"""
    cata(alg, term)

The catamorphism (fold): the unique F-algebra homomorphism from the initial
(term) algebra into `alg`. Recursively folds children, then applies `ops`.
"""
function cata(alg::FAlgebra, t::Term)
    alg.ops[t.con]([cata(alg, s) for s in t.args])
end

"""
    cata_is_homomorphism(alg, terms) -> Bool

Verify the catamorphism is an F-algebra homomorphism on a set of terms:
`cata(c(t₁,…,tₙ)) = ops[c](cata t₁, …, cata tₙ)` (the universal-property square).
"""
function cata_is_homomorphism(alg::FAlgebra, terms)
    for t in terms
        cata(alg, t) == alg.ops[t.con]([cata(alg, s) for s in t.args]) || return false
    end
    true
end

"""
    arithmetic_signature() -> Signature

The signature `zero | one | add(·,·) | mul(·,·)` — the syntax functor of
arithmetic expressions, for catamorphism demos (evaluation = a fold).
"""
arithmetic_signature() = Signature(:zero => 0, :one => 0, :add => 2, :mul => 2)
