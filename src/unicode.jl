# ============================================================================
# unicode.jl — Unicode operators for FunctorFlow
#
# Provides mathematical notation for categorical operations:
#   Σ  — left Kan extension (pushforward / universal aggregation)
#   Δ  — right Kan extension (pullback / universal completion)
#   ⋅  — morphism composition
#   ⊗  — product of diagrams
#   ⊕  — coproduct of diagrams
# ============================================================================

# ---------------------------------------------------------------------------
# Σ — Left Kan Extension (universal aggregation)
# ---------------------------------------------------------------------------

"""
    Σ(D, source; along, name=nothing, target=nothing, reducer=:sum, kwargs...)

Left Kan extension (universal aggregation / pushforward).

Covers: attention, pooling, neighborhood message passing, context fusion,
plan-fragment integration.

# Examples
```julia
D = Diagram(:MyKET)
add_object!(D, :Tokens, kind=:messages)
add_object!(D, :Nbrs, kind=:relation)
Σ(D, :Tokens; along=:Nbrs, reducer=:sum, name=:aggregate)
```

The name `Σ` reflects the categorical semantics: left Kan extensions are
computed as colimits (generalized sums/coproducts).
"""
function Σ(D, source::Union{Symbol, AbstractString};
           along::Union{Symbol, AbstractString},
           name::Union{Nothing, Symbol, AbstractString}=nothing,
           target::Union{Nothing, Symbol, AbstractString}=nothing,
           reducer::Union{Symbol, AbstractString}=:sum,
           description::AbstractString="",
           metadata::Dict=Dict{Symbol, Any}())
    kan_name = name === nothing ? Symbol(:Σ_, source, :_along_, along) : Symbol(name)
    add_left_kan!(D, kan_name; source, along, target, reducer, description, metadata)
end

"""
    left_kan(D, source; along, kwargs...)

ASCII alias for [`Σ`](@ref). Identical behavior.
"""
const left_kan = Σ

# ---------------------------------------------------------------------------
# Δ — Right Kan Extension (universal completion)
# ---------------------------------------------------------------------------

"""
    Δ(D, source; along, name=nothing, target=nothing, reducer=:first_non_null, kwargs...)

Right Kan extension (universal completion / repair).

Covers: denoising, masked completion, plan repair, partial-view
reconciliation.

# Examples
```julia
D = Diagram(:Repair)
add_object!(D, :Partial, kind=:state)
add_object!(D, :Structure, kind=:relation)
Δ(D, :Partial; along=:Structure, name=:complete)
```

The name `Δ` reflects the categorical semantics: right Kan extensions are
computed as limits (generalized products/pullbacks). In the adjunction
Σ ⊣ Δ, left and right Kan form an adjoint pair.
"""
function Δ(D, source::Union{Symbol, AbstractString};
           along::Union{Symbol, AbstractString},
           name::Union{Nothing, Symbol, AbstractString}=nothing,
           target::Union{Nothing, Symbol, AbstractString}=nothing,
           reducer::Union{Symbol, AbstractString}=:first_non_null,
           description::AbstractString="",
           metadata::Dict=Dict{Symbol, Any}())
    kan_name = name === nothing ? Symbol(:Δ_, source, :_along_, along) : Symbol(name)
    add_right_kan!(D, kan_name; source, along, target, reducer, description, metadata)
end

"""
    right_kan(D, source; along, kwargs...)

ASCII alias for [`Δ`](@ref). Identical behavior.
"""
const right_kan = Δ

# ---------------------------------------------------------------------------
# ⋅ — Morphism composition
# ---------------------------------------------------------------------------

"""
    f ⋅ g

Compose two model morphisms: `g ∘ f` (diagrammatic order, left to right).

# Example
```julia
f = ModelMorphism(:f, :A, :B; functor_data=x -> x * 2)
g = ModelMorphism(:g, :B, :C; functor_data=x -> x + 1)
h = f ⋅ g   # h : A → C
```
"""
function ⋅(f::ModelMorphism, g::ModelMorphism)
    compose(f, g)
end

# ---------------------------------------------------------------------------
# ⊗ — Product of diagrams
# ---------------------------------------------------------------------------

"""
    D1 ⊗ D2

Product (parallel composition) of two diagrams.
Equivalent to `product(D1, D2)`.

# Example
```julia
encoder = ket_block(; name=:Encoder)
decoder = completion_block(; name=:Decoder)
combined = encoder ⊗ decoder
```
"""
function ⊗(D1, D2)
    product(D1, D2; name=Symbol(D1.name, Symbol("_⊗_"), D2.name))
end

# ---------------------------------------------------------------------------
# ⊕ — Coproduct of diagrams
# ---------------------------------------------------------------------------

"""
    D1 ⊕ D2

Coproduct (disjoint union / ensemble) of two diagrams.
Equivalent to `coproduct(D1, D2)`.

# Example
```julia
model_a = ket_block(; name=:ModelA)
model_b = ket_block(; name=:ModelB)
ensemble = model_a ⊕ model_b
```
"""
function ⊕(D1, D2)
    coproduct(D1, D2; name=Symbol(D1.name, Symbol("_⊕_"), D2.name))
end

# ---------------------------------------------------------------------------
# → — Morphism arrow (for display, not construction)
# ---------------------------------------------------------------------------

"""
    →(source::Symbol, target::Symbol) -> Tuple{Symbol, Symbol}

Create a source-target pair for morphism specification.
Used in the `@functorflow` macro: `f = A → B`.
"""
→(a::Symbol, b::Symbol) = (a, b)
