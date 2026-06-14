# ============================================================================
# operad.jl — finite one-colored (symmetric) operads / multicategories
# (included into module Cat)
#
# An operad is the algebra of *composition with many inputs and one output* —
# the natural home of compositional architectures and wiring. Where a category
# has morphisms `a → b` (one input, one output), an operad `O` has operations
# `θ ∈ O(n)` with `n` inputs and one output, an identity `1 ∈ O(1)`, and a
# substitution / composition
#
#     γ : O(n) × O(k₁) × … × O(kₙ) → O(k₁ + … + kₙ)
#
# that plugs an operation into each input of an `n`-ary operation. (One-colored
# = there is a single object/type; the multi-colored version is a
# *multicategory*, with typed inputs/outputs.) The axioms — associativity of
# `γ` and the left/right unit laws — say exactly that nested wiring can be
# re-bracketed and that wiring in the identity does nothing.
#
# As with the rest of the kernel everything is *concrete and finite*: `O(n)` is
# a finite set for each arity up to a bound, `γ` is given as data, and the
# operad axioms are *checked by enumeration* (`operad_laws`). A symmetric
# operad additionally carries an action of the symmetric group `Sₙ` on `O(n)`
# (permuting the inputs) satisfying equivariance laws; this is optional and
# checked by `operad_symmetry_laws`.
#
# Why this lives in FunctorFlow: a FunctorFlow `Diagram` / wiring diagram is an
# operation in the operad of wiring diagrams — boxes plugged into boxes. Operad
# composition `γ` is exactly "substitute a sub-architecture for a box", and the
# operad laws are the guarantee that nesting sub-architectures is associative
# and that the trivial box is a unit. See `wiring_operad` for the worked example.
# ============================================================================

"""
    Operad(ops, identity, gamma; symmetry=nothing, max_arity=…)

A finite one-colored operad.

- `ops :: Dict{Int, Vector{Any}}` — the operations by arity, i.e. `ops[n] = O(n)`.
  Every arity from `0` up to `max_arity` that the laws will range over must be
  present (possibly as an empty vector).
- `identity` — the distinguished operation `1 ∈ O(1)`.
- `gamma :: Function` — the composition `γ(θ, [φ₁, …, φₙ])` where `θ ∈ O(n)` and
  `φᵢ ∈ O(kᵢ)`, returning an element of `O(k₁+…+kₙ)`. Given as *data* so that
  non-associative / non-unital candidates can be supplied to the law checker as
  negative controls.
- `symmetry :: Union{Nothing, Function}` — optional symmetric-group action
  `σ ⋅ θ` written `symmetry(perm, θ)`, where `perm :: Vector{Int}` is a
  permutation of `1:n` (in one-line notation) and `θ ∈ O(n)`. `nothing` ⇒ a
  plain (non-symmetric) operad.

The arity of an operation is recovered structurally via [`operad_arity`](@ref);
operations therefore need only be `==`/`hash`-comparable values.
"""
struct Operad
    ops::Dict{Int, Vector{Any}}
    identity::Any
    gamma::Function
    symmetry::Union{Nothing, Function}
    max_arity::Int
end

function Operad(ops::AbstractDict, identity, gamma::Function;
               symmetry::Union{Nothing,Function}=nothing,
               max_arity::Int=maximum(keys(ops)))
    opsd = Dict{Int, Vector{Any}}()
    for (n, xs) in ops
        opsd[Int(n)] = collect(Any, xs)
    end
    # Ensure every arity 0..max_arity is represented (default empty).
    for n in 0:max_arity
        haskey(opsd, n) || (opsd[n] = Any[])
    end
    haskey(opsd, 1) && identity in opsd[1] ||
        throw(ArgumentError("the operad identity must be an element of O(1)"))
    Operad(opsd, identity, gamma, symmetry, max_arity)
end

"""`operad_ops(O, n)` — the set `O(n)` of `n`-ary operations."""
operad_ops(O::Operad, n::Integer) = get(O.ops, Int(n), Any[])

"""
    operad_arity(O, θ) -> Int

The arity of operation `θ`, i.e. the unique `n` with `θ ∈ O(n)`. Throws if `θ`
is not an operation of `O`.
"""
function operad_arity(O::Operad, θ)
    for n in 0:O.max_arity
        θ in O.ops[n] && return n
    end
    throw(ArgumentError("operation $(repr(θ)) is not in any O(n) of this operad"))
end

"""`operad_id(O)` — the identity operation `1 ∈ O(1)`."""
operad_id(O::Operad) = O.identity

"""
    operad_compose(O, θ, φs) -> operation

Operadic composition `γ(θ; φ₁,…,φₙ)`: substitute the operation `φᵢ` into the
`i`-th input of `θ ∈ O(n)`. If `θ` is `n`-ary then `φs` must have length `n`,
and the result lies in `O(k₁+…+kₙ)` where `kᵢ = arity(φᵢ)`. Validates arities
on the way in and out.
"""
function operad_compose(O::Operad, θ, φs::AbstractVector)
    n = operad_arity(O, θ)
    length(φs) == n ||
        throw(ArgumentError("γ: θ is $n-ary but got $(length(φs)) inner operations"))
    result = O.gamma(θ, collect(φs))
    expected = isempty(φs) ? 0 : sum(operad_arity(O, φ) for φ in φs)
    got = operad_arity(O, result)
    got == expected ||
        throw(ArgumentError("γ produced an operation of arity $got, expected $expected"))
    result
end

# A 2-argument convenience: γ(θ, φ₁, φ₂, …).
operad_compose(O::Operad, θ, φs...) = operad_compose(O, θ, collect(φs))

"""
    operad_laws(O; verbose=false) -> Bool

Verify the operad axioms **by enumeration** over all operations up to
`O.max_arity`:

1. **Right unit**: `γ(θ; 1, …, 1) = θ` for every `θ ∈ O(n)`.
2. **Left unit**: `γ(1; θ) = θ` for every `θ ∈ O(n)`.
3. **Associativity**: for `θ ∈ O(n)`, `φⱼ ∈ O(kⱼ)` and `ψ ∈ O(·)`, the two
   ways of substituting agree — substituting the `ψ`'s into the `φ`'s first and
   then into `θ`, versus substituting the `φ`'s into `θ` first and then the
   `ψ`'s. Concretely

       γ(γ(θ; φ₁,…,φₙ); ψ₁,…,ψ_K)
         = γ(θ; γ(φ₁; ψ₁..), …, γ(φₙ; ψ..)),

   where the flat list `ψ₁,…,ψ_K` is partitioned to match the arities of the
   `φⱼ`. (This is the standard "operad associativity" / parallel-then-serial
   coherence.)

The enumeration is bounded so that the result arities never exceed
`O.max_arity` (otherwise `γ` could land outside the represented operations).
Set `verbose=true` to print the first violating witness.
"""
function operad_laws(O::Operad; verbose::Bool=false)
    ident = O.identity
    M = O.max_arity

    # --- unit laws ---
    for n in 0:M
        for θ in O.ops[n]
            # right unit: γ(θ; 1,…,1) = θ
            if n <= M
                rhs = O.gamma(θ, Any[ident for _ in 1:n])
                if rhs != θ
                    verbose && @info "right-unit violated" θ rhs
                    return false
                end
            end
            # left unit: γ(1; θ) = θ
            lhs = O.gamma(ident, Any[θ])
            if lhs != θ
                verbose && @info "left-unit violated" θ lhs
                return false
            end
        end
    end

    # --- associativity ---
    # θ ∈ O(n); for each input i choose φᵢ ∈ O(kᵢ); for each input of each φᵢ
    # choose a ψ. Bound everything by max_arity.
    for n in 0:M
        for θ in O.ops[n]
            # choose φ₁..φₙ, each of arity ≤ M and with Σkᵢ ≤ M
            for φs in _bounded_tuples(O, n, M)
                ks = Int[operad_arity(O, φ) for φ in φs]
                K = isempty(ks) ? 0 : sum(ks)
                K <= M || continue
                # the middle composite has arity K (≤ M), so the outer ψ-stage
                # is itself bounded by M.
                for ψs in _bounded_tuples(O, K, M)
                    # left side: γ(γ(θ; φ); ψ)
                    inner = O.gamma(θ, collect(φs))
                    left = O.gamma(inner, collect(ψs))
                    # right side: partition ψ to match the kᵢ, compose into φᵢ
                    parts = _partition(collect(ψs), ks)
                    composed_φ = Any[O.gamma(φs[j], parts[j]) for j in 1:n]
                    right = O.gamma(θ, composed_φ)
                    if left != right
                        verbose && @info "associativity violated" θ φs ψs left right
                        return false
                    end
                end
            end
        end
    end
    true
end

# All length-`n` tuples of operations whose total arity is ≤ `bound`.
function _bounded_tuples(O::Operad, n::Int, bound::Int)
    n == 0 && return Vector{Vector{Any}}([Any[]])
    # candidate operations: those whose arity alone fits the bound
    cands = Any[]
    for k in 0:bound
        append!(cands, O.ops[k])
    end
    out = Vector{Vector{Any}}()
    for combo in Iterators.product((cands for _ in 1:n)...)
        tup = collect(Any, combo)
        tot = sum(operad_arity(O, x) for x in tup)
        tot <= bound && push!(out, tup)
    end
    out
end

# Split a flat list into consecutive chunks of the given sizes.
function _partition(xs::Vector, sizes::Vector{Int})
    out = Vector{Vector{Any}}()
    i = 1
    for s in sizes
        push!(out, Any[xs[i + t] for t in 0:(s-1)])
        i += s
    end
    out
end

# ----------------------------------------------------------------------------
# Symmetric structure (optional)
# ----------------------------------------------------------------------------

"""`operad_act(O, perm, θ)` — the symmetric-group action `σ ⋅ θ` (requires a symmetric operad)."""
function operad_act(O::Operad, perm::AbstractVector{<:Integer}, θ)
    O.symmetry === nothing && throw(ArgumentError("this operad has no symmetric structure"))
    O.symmetry(collect(Int, perm), θ)
end

# all permutations of 1:n in one-line notation
function _perms(n::Int)
    n == 0 && return Vector{Vector{Int}}([Int[]])
    out = Vector{Vector{Int}}()
    function go(rem, acc)
        isempty(rem) && (push!(out, copy(acc)); return)
        for x in rem
            push!(acc, x)
            go(filter(!=(x), rem), acc)
            pop!(acc)
        end
    end
    go(collect(1:n), Int[])
    out
end

# compose permutations (one-line): (σ∘τ)(i) = σ(τ(i))
_perm_compose(σ::Vector{Int}, τ::Vector{Int}) = Int[σ[τ[i]] for i in eachindex(τ)]

"""
    operad_symmetry_laws(O) -> Bool

For a **symmetric** operad, verify by enumeration that the `Sₙ`-action is a
group action (identity permutation acts trivially; `σ ⋅ (τ ⋅ θ) = (σ∘τ) ⋅ θ`).
The full equivariance of `γ` w.r.t. the actions is a further axiom; here we
check the (already nontrivial) action laws, plus that the action preserves
arity. Returns `true` for a non-symmetric operad (vacuously).
"""
function operad_symmetry_laws(O::Operad)
    O.symmetry === nothing && return true
    for n in 0:O.max_arity
        idperm = collect(1:n)
        for θ in O.ops[n]
            # identity acts trivially
            operad_act(O, idperm, θ) == θ || return false
            for σ in _perms(n), τ in _perms(n)
                lhs = operad_act(O, σ, operad_act(O, τ, θ))
                rhs = operad_act(O, _perm_compose(σ, τ), θ)
                lhs == rhs || return false
                # arity preserved
                operad_arity(O, operad_act(O, σ, θ)) == n || return false
            end
        end
    end
    true
end

# ----------------------------------------------------------------------------
# Worked examples
# ----------------------------------------------------------------------------

"""
    commutative_operad(; max_arity=3) -> Operad

The **commutative operad** `Com`, the terminal one-colored operad: `Com(n)` is a
one-element set for every `n` (here the symbol `Symbol("•", n)`), so there is a
unique `n`-ary operation. `γ` is forced (there is nowhere else to land) and the
symmetric action is trivial. Algebras over `Com` are commutative monoids, which
is why it is the canonical "combine `n` things into one, order-independent"
operad — e.g. a permutation-invariant pooling / aggregation primitive.
"""
function commutative_operad(; max_arity::Int=3)
    star(n) = Symbol("•", n)
    ops = Dict{Int, Vector{Any}}(n => Any[star(n)] for n in 0:max_arity)
    γ(θ, φs) = star(isempty(φs) ? 0 : sum(_star_arity(φ) for φ in φs))
    sym(_perm, θ) = θ                         # unique op ⇒ trivial action
    Operad(ops, star(1), γ; symmetry=sym, max_arity=max_arity)
end
_star_arity(s::Symbol) = parse(Int, String(s)[nextind(String(s), 1):end])

"""
    associative_operad(; max_arity=3) -> Operad

The **associative operad** `Assoc`: `Assoc(n)` is the set of linear orders on
`n` inputs, i.e. the `n!` permutations of `1:n` (in one-line notation), with the
identity `[1]` in arity 1. `γ` substitutes ordered lists into an ordered list,
relabelling. Algebras over `Assoc` are (not-necessarily-commutative) monoids —
the operad of *sequencing*, the backbone of pipelines and serial composition.

This is included as the non-trivial *symmetric* example: `Sₙ` acts on the
orders by permuting input labels.
"""
function associative_operad(; max_arity::Int=3)
    ops = Dict{Int, Vector{Any}}(n => Any[p for p in _perms(n)] for n in 0:max_arity)
    # γ(θ; φ₁..φₙ): θ is an order on n blocks; block i contributes its kᵢ inputs.
    # Result order: read blocks in the order given by θ, and within block θ(j)
    # emit that block's inputs in φ_{θ(j)}'s order, with global relabelling.
    function γ(θ::Vector{Int}, φs)
        n = length(θ)
        ks = Int[length(φs[i]) for i in 1:n]
        offsets = zeros(Int, n)            # global start index (0-based) of block i
        acc = 0
        for i in 1:n
            offsets[i] = acc; acc += ks[i]
        end
        out = Int[]
        for blk in θ                        # blocks in θ's order
            for local_idx in φs[blk]        # inputs of that block in φ's order
                push!(out, offsets[blk] + local_idx)
            end
        end
        out
    end
    # symmetric action: relabel the inputs by σ (σ⋅θ permutes which label sits
    # in each position): (σ⋅θ)[j] = σ[θ[j]].
    sym(σ::Vector{Int}, θ::Vector{Int}) = Int[σ[θ[j]] for j in eachindex(θ)]
    Operad(ops, [1], γ; symmetry=sym, max_arity=max_arity)
end

"""
    wiring_operad(; max_arity=3) -> Operad

A small **operad of wirings** modelling compositional architectures. An `n`-ary
operation is a *wiring* of `n` boxes into one outer box, recorded here as the
tuple of input fan-ins `(w₁,…,wₙ)` where `wᵢ ∈ {1}` is the (single, one-colored)
wire feeding box `i`; equivalently `O(n)` is again a singleton per arity, but we
*tag* operations with their arity so that `γ` and the laws exercise real
substitution rather than a degenerate identity. Composition `γ` nests a wiring
into each box (substitution of sub-architectures), and the laws certify that
*nesting sub-architectures is associative and that the trivial box is a unit* —
the categorical guarantee underwriting modular/compositional model design and
FunctorFlow's own wiring diagrams.

Concretely this is isomorphic to [`commutative_operad`](@ref); it is provided
under an architecture-facing name (and with a `box`/`wire` vocabulary in the
docstring) to make the connection to compositional wiring explicit.
"""
wiring_operad(; max_arity::Int=3) = commutative_operad(; max_arity=max_arity)

"""
    little_intervals_operad(; max_arity=3) -> Operad

A finite, combinatorial stand-in for the **little 1-cubes (intervals) operad**:
`O(n)` is the single order-preserving way to place `n` disjoint sub-intervals in
order inside the unit interval, represented as the identity order `[1,…,n]`.
This is the *planar* (non-symmetric) operad of *nesting intervals*: `γ` rescales
and inserts. Algebras are (up to the usual homotopy story) `A∞`/monoid-like
structures; here it serves as a second associativity-flavoured example whose
operations carry geometric intent (sequential composition of sub-processes over
"time"). The symmetric action is dropped (intervals are ordered).
"""
function little_intervals_operad(; max_arity::Int=3)
    ops = Dict{Int, Vector{Any}}(n => Any[collect(1:n)] for n in 0:max_arity)
    γ(θ, φs) = collect(1:(isempty(φs) ? 0 : sum(length(φ) for φ in φs)))
    Operad(ops, [1], γ; symmetry=nothing, max_arity=max_arity)
end

# ----------------------------------------------------------------------------
# Operad → category: the underlying category of unary operations
# ----------------------------------------------------------------------------

"""
    unary_monoid(O) -> (ops, mul, unit)

The underlying **monoid** of unary operations `O(1)`: an operad restricted to
arity 1 is exactly a monoid, with multiplication `θ·φ = γ(θ; φ)` and unit the
operad identity. Returns `(O(1), mul, unit)` so callers can check the monoid
laws directly. (This is the operad ↔ category bridge: a one-object category =
a monoid = the unary part of an operad.)
"""
function unary_monoid(O::Operad)
    ops = collect(operad_ops(O, 1))
    mul(θ, φ) = operad_compose(O, θ, Any[φ])
    (ops, mul, O.identity)
end
