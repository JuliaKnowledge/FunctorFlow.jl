# ============================================================================
# learn.jl — categorical deep learning: backpropagation as a functor
# (included into module Cat)
#
# The deepest bridge between category theory and AI: gradient backpropagation
# is the *reverse-derivative functor*, and the **chain rule is functoriality**.
# Concretely, in the category `FinVect_n` of linear maps over ℤ_n, a layer is a
# matrix, a network is a composite, the forward pass applies the matrix, and the
# backward pass (the vector–Jacobian product) applies the **transpose**. The
# transpose is a contravariant functor — `(g∘f)ᵀ = fᵀ∘gᵀ` — which is exactly the
# statement that backprop reverses the network and accumulates by the chain
# rule. Over ℤ_n this is exact and finite, hence runtime-checkable *and*
# Lean-certifiable (see `render_backprop_certificate` / `Learn.lean`).
# ============================================================================

"""
    LinMap(modulus, indim, outdim, matrix)

A morphism of `FinVect_n`: a linear map `ℤ_n^indim → ℤ_n^outdim` given by an
`outdim × indim` matrix (entries reduced mod `n`). A neural layer with no bias.
"""
struct LinMap
    modulus::Int
    indim::Int
    outdim::Int
    matrix::Matrix{Int}
    function LinMap(modulus::Integer, indim::Integer, outdim::Integer, matrix::AbstractMatrix)
        size(matrix) == (outdim, indim) ||
            throw(ArgumentError("matrix must be $(outdim)×$(indim), got $(size(matrix))"))
        new(Int(modulus), Int(indim), Int(outdim), Int.(mod.(matrix, modulus)))
    end
end

Base.:(==)(f::LinMap, g::LinMap) =
    f.modulus == g.modulus && f.indim == g.indim && f.outdim == g.outdim && f.matrix == g.matrix

_matmul(A::Matrix{Int}, B::Matrix{Int}, n::Int) = begin
    ra, ca = size(A); rb, cb = size(B)
    ca == rb || throw(ArgumentError("inner dimensions disagree: $ca ≠ $rb"))
    C = zeros(Int, ra, cb)
    for i in 1:ra, j in 1:cb
        s = 0
        for k in 1:ca
            s += A[i, k] * B[k, j]
        end
        C[i, j] = mod(s, n)
    end
    C
end

"""`forward(f, x)` — the forward pass `x ↦ f.matrix · x` (mod n)."""
function forward(f::LinMap, x::AbstractVector{<:Integer})
    length(x) == f.indim || throw(ArgumentError("input length $(length(x)) ≠ indim $(f.indim)"))
    [mod(sum(f.matrix[i, k] * x[k] for k in 1:f.indim; init=0), f.modulus) for i in 1:f.outdim]
end

"""`lin_id(n, d)` — the identity layer on `ℤ_n^d`."""
lin_id(n::Integer, d::Integer) =
    LinMap(n, d, d, [i == j ? 1 : 0 for i in 1:d, j in 1:d])

"""`lin_compose(f, g)` — layer composition `g ∘ f` (diagrammatic: `f` then `g`)."""
function lin_compose(f::LinMap, g::LinMap)
    f.modulus == g.modulus || throw(ArgumentError("different moduli"))
    f.outdim == g.indim || throw(ArgumentError("not composable: $(f.outdim) ≠ $(g.indim)"))
    LinMap(f.modulus, f.indim, g.outdim, _matmul(g.matrix, f.matrix, f.modulus))
end

"""
    lin_transpose(f) -> LinMap

The reverse-derivative of a linear layer: the **transpose** `fᵀ : ℤ_n^outdim →
ℤ_n^indim`. This is the backward pass (vector–Jacobian product) of `f`.
"""
lin_transpose(f::LinMap) = LinMap(f.modulus, f.outdim, f.indim, permutedims(f.matrix))

"""`backward(f, cotangent)` — the backward pass: apply the transpose to a cotangent."""
backward(f::LinMap, cotangent::AbstractVector{<:Integer}) = forward(lin_transpose(f), cotangent)

# `reverse_derivative` is the categorical name for the backward pass.
const reverse_derivative = lin_transpose

"""
    transpose_is_functorial(f, g) -> Bool

The crux: `(g∘f)ᵀ = fᵀ∘gᵀ`. Backpropagation reverses the order of composition —
this is the chain rule expressed as the (contravariant) functoriality of the
reverse-derivative.
"""
transpose_is_functorial(f::LinMap, g::LinMap) =
    lin_transpose(lin_compose(f, g)) == lin_compose(lin_transpose(g), lin_transpose(f))

"""
    finvect_category_laws(maps) -> Bool

Check the `FinVect_n` category axioms on a set of layers: identity laws and
associativity of composition (where composable).
"""
function finvect_category_laws(maps::AbstractVector{LinMap})
    for f in maps
        lin_compose(lin_id(f.modulus, f.indim), f) == f || return false
        lin_compose(f, lin_id(f.modulus, f.outdim)) == f || return false
    end
    for f in maps, g in maps
        f.outdim == g.indim || continue
        for h in maps
            g.outdim == h.indim || continue
            lin_compose(lin_compose(f, g), h) == lin_compose(f, lin_compose(g, h)) || return false
        end
    end
    true
end

"""
    backprop_demo(; modulus=7) -> NamedTuple

A 2-layer linear network `ℤ_n³ → ℤ_n² → ℤ_n¹`: run the forward pass, then
backprop a cotangent two ways — through the composite network's transpose, and
layer-by-layer — and confirm they agree (the chain rule / functoriality of the
backward pass). Returns the forward output, the input gradient, and the check.
"""
function backprop_demo(; modulus::Integer=7)
    W1 = LinMap(modulus, 3, 2, [1 0 2; 0 1 1])    # layer 1: ℤ³ → ℤ²
    W2 = LinMap(modulus, 2, 1, reshape([1, 1], 1, 2))  # layer 2: ℤ² → ℤ¹
    net = lin_compose(W1, W2)                      # the network ℤ³ → ℤ¹
    x = [1, 2, 3]
    y = forward(net, x)
    cot = [1]                                      # cotangent at the output
    grad_whole = backward(net, cot)               # via the composite transpose
    grad_layered = backward(W1, backward(W2, cot))# via the chain rule, layer by layer
    (forward=y, input_gradient=grad_whole,
     chain_rule_ok=grad_whole == grad_layered,
     transpose_functorial=transpose_is_functorial(W1, W2),
     category_laws=finvect_category_laws([W1, W2, net, lin_id(modulus, 3)]))
end
