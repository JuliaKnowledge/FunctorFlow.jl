
# ============================================================================
# Layer types
# ============================================================================

"""
    DiagramDenseLayer(in_dims, out_dims; activation=identity, name=:dense)

A dense (fully-connected) morphism layer for use inside FunctorFlow diagrams.
Wraps `Lux.Dense` with FunctorFlow metadata.
"""
struct DiagramDenseLayer <: LuxCore.AbstractLuxLayer
    name::Symbol
    in_dims::Int
    out_dims::Int
    activation::Any
end

function DiagramDenseLayer(in_dims::Int, out_dims::Int;
                           activation=identity, name::Symbol=:dense)
    DiagramDenseLayer(name, in_dims, out_dims, activation)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::DiagramDenseLayer)
    scale = sqrt(2.0f0 / l.in_dims)
    weight = randn(rng, Float32, l.out_dims, l.in_dims) .* scale
    bias = zeros(Float32, l.out_dims)
    (; weight, bias)
end

LuxCore.initialstates(::AbstractRNG, ::DiagramDenseLayer) = NamedTuple()

function (l::DiagramDenseLayer)(x::AbstractArray, ps, st)
    # Handle batched (3D+) inputs: reshape to 2D, multiply, reshape back
    sz = size(x)
    x2d = reshape(x, sz[1], :)
    y2d = ps.weight * x2d .+ ps.bias
    y = reshape(y2d, size(y2d, 1), sz[2:end]...)
    return l.activation.(y), st
end

"""
    DiagramChainLayer(layers...; name=:chain)

A sequential composition of Lux layers, corresponding to a FunctorFlow
`Composition`. Each layer's output feeds into the next.
"""
struct DiagramChainLayer <: LuxCore.AbstractLuxLayer
    name::Symbol
    layers::Vector{LuxCore.AbstractLuxLayer}
    layer_names::Vector{Symbol}
end

function DiagramChainLayer(layers::Vector{<:LuxCore.AbstractLuxLayer};
                           name::Symbol=:chain,
                           layer_names::Union{Nothing, Vector{Symbol}}=nothing)
    names = layer_names === nothing ? [Symbol("layer_$i") for i in 1:length(layers)] : layer_names
    DiagramChainLayer(name, collect(LuxCore.AbstractLuxLayer, layers), names)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::DiagramChainLayer)
    params = NamedTuple()
    for (i, layer) in enumerate(l.layers)
        lname = l.layer_names[i]
        lps = LuxCore.initialparameters(rng, layer)
        params = merge(params, NamedTuple{(lname,)}((lps,)))
    end
    params
end

function LuxCore.initialstates(rng::AbstractRNG, l::DiagramChainLayer)
    states = NamedTuple()
    for (i, layer) in enumerate(l.layers)
        lname = l.layer_names[i]
        lst = LuxCore.initialstates(rng, layer)
        states = merge(states, NamedTuple{(lname,)}((lst,)))
    end
    states
end

function (l::DiagramChainLayer)(x::AbstractArray, ps, st)
    current = x
    new_st = st
    for (i, layer) in enumerate(l.layers)
        lname = l.layer_names[i]
        lps = getfield(ps, lname)
        lst = getfield(st, lname)
        current, lst_new = layer(current, lps, lst)
        new_st = merge(new_st, NamedTuple{(lname,)}((lst_new,)))
    end
    current, new_st
end

"""
    KETAttentionLayer(d_model; n_heads=1, dropout=0.0f0, name=:ket_attention)

Learnable Kan Extension Transformer (KET) reducer as a Lux layer.
Implements scaled multi-head dot-product attention:

    Attention(Q, K, V) = softmax(QKᵀ / √d_k ⊙ mask) V

where:
- Q, K, V are learned linear projections of the source values
- mask comes from the `along` relation (incidence geometry)
- The output is projected back to `d_model` dimensions

This is the neural implementation of a left-Kan extension: universal
aggregation via attention over an incidence relation.
"""
struct KETAttentionLayer <: LuxCore.AbstractLuxLayer
    name::Symbol
    d_model::Int
    n_heads::Int
    d_k::Int
    dropout_rate::Float32
end

function KETAttentionLayer(d_model::Int;
                           n_heads::Int=1,
                           dropout::Real=0.0f0,
                           name::Symbol=:ket_attention)
    @assert d_model % n_heads == 0 "d_model ($d_model) must be divisible by n_heads ($n_heads)"
    KETAttentionLayer(name, d_model, n_heads, d_model ÷ n_heads, Float32(dropout))
end

function LuxCore.initialparameters(rng::AbstractRNG, l::KETAttentionLayer)
    scale = sqrt(2.0f0 / l.d_model)
    W_q = randn(rng, Float32, l.d_model, l.d_model) .* scale
    W_k = randn(rng, Float32, l.d_model, l.d_model) .* scale
    W_v = randn(rng, Float32, l.d_model, l.d_model) .* scale
    W_o = randn(rng, Float32, l.d_model, l.d_model) .* scale
    b_q = zeros(Float32, l.d_model)
    b_k = zeros(Float32, l.d_model)
    b_v = zeros(Float32, l.d_model)
    b_o = zeros(Float32, l.d_model)
    (; W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o)
end

function LuxCore.initialstates(rng::AbstractRNG, l::KETAttentionLayer)
    (; training=Lux.Val(true), rng=Lux.replicate(rng))
end

function (l::KETAttentionLayer)(x::AbstractArray, ps, st)
    _ket_attention_forward(l, x, nothing, ps, st)
end

"""
    (l::KETAttentionLayer)((source, mask), ps, st)

Forward pass with an explicit attention mask (the `along` relation).
`source` is (d_model, seq_len[, batch]) and `mask` is (seq_len, seq_len[, batch])
where `true` entries allow attention.
"""
function (l::KETAttentionLayer)((source, mask)::Tuple{AbstractArray, AbstractArray}, ps, st)
    _ket_attention_forward(l, source, mask, ps, st)
end

function _ket_attention_forward(l::KETAttentionLayer, source::AbstractArray,
                                mask::Union{Nothing, AbstractArray}, ps, st)
    d_model = l.d_model
    n_heads = l.n_heads
    d_k = l.d_k

    # GPU-compatible linear projection: W * x for x with batch dims
    function _proj(W, b, x)
        sz = size(x)
        x2d = reshape(x, sz[1], :)
        y2d = W * x2d .+ b
        reshape(y2d, size(y2d, 1), sz[2:end]...)
    end

    # GPU + AD compatible batched matmul (no mutation, Zygote-safe)
    function _bmm(A, B)
        if ndims(A) == 2
            return A * B
        end
        batch = size(A, 3)
        # Stack slices without mutation
        slices = [reshape(A[:, :, i] * B[:, :, i], :, 1) for i in 1:batch]
        r = size(A, 1)
        c = size(B, 2)
        reshape(hcat(slices...), r, c, batch)
    end

    # Batched transpose-matmul: A^T * B per batch slice
    function _btmm(A, B)
        if ndims(A) == 2
            return A' * B
        end
        batch = size(A, 3)
        s = size(A, 2)
        c = size(B, 2)
        slices = [reshape(A[:, :, i]' * B[:, :, i], :, 1) for i in 1:batch]
        reshape(hcat(slices...), s, c, batch)
    end

    # Linear projections: Q, K, V
    Q = _proj(ps.W_q, ps.b_q, source)
    K = _proj(ps.W_k, ps.b_k, source)
    V = _proj(ps.W_v, ps.b_v, source)

    if n_heads == 1
        scale = Float32(sqrt(d_k))
        scores = _btmm(K, Q) ./ scale

        if mask !== nothing
            if ndims(scores) > ndims(mask)
                mask_nd = reshape(mask, size(mask)..., ntuple(_ -> 1, ndims(scores) - ndims(mask))...)
            else
                mask_nd = mask
            end
            # Use ifelse to avoid 0 * -Inf = NaN
            scores = scores .+ ifelse.(mask_nd .> 0.5f0, 0.0f0, typemin(Float32))
        end

        weights = _softmax_cols(scores)
        output = _bmm(V, weights)
    else
        seq_len = size(source, 2)
        batch_dims = size(source)[3:end]

        Q_mh = reshape(Q, d_k, n_heads, seq_len, batch_dims...)
        K_mh = reshape(K, d_k, n_heads, seq_len, batch_dims...)
        V_mh = reshape(V, d_k, n_heads, seq_len, batch_dims...)

        scale = Float32(sqrt(d_k))
        heads = map(1:n_heads) do h
            Qh = copy(selectdim(Q_mh, 2, h))
            Kh = copy(selectdim(K_mh, 2, h))
            Vh = copy(selectdim(V_mh, 2, h))
            scores_h = _btmm(Kh, Qh) ./ scale
            if mask !== nothing
                if ndims(scores_h) > ndims(mask)
                    mask_nd = reshape(mask, size(mask)..., ntuple(_ -> 1, ndims(scores_h) - ndims(mask))...)
                else
                    mask_nd = mask
                end
                scores_h = scores_h .+ ifelse.(mask_nd .> 0.5f0, 0.0f0, typemin(Float32))
            end
            w_h = _softmax_cols(scores_h)
            _bmm(Vh, w_h)
        end
        output = cat(heads...; dims=1)
    end

    result = _proj(ps.W_o, ps.b_o, output)
    result, st
end

"""Column-wise softmax (along first dimension of a 2D+ array)."""
function _softmax_cols(x::AbstractArray)
    mx = maximum(x; dims=1)
    ex = exp.(x .- mx)
    ex ./ sum(ex; dims=1)
end

"""
    predict_detach_source(logits, embedding_weights; position_bias=nothing)

Project logits back into embedding space while stopping gradients through the
prediction path. This is the reusable helper behind the "predict-detach"
pattern used in the vignettes.

- `logits` has shape `(vocab, seq_len[, batch])`
- `embedding_weights` has shape `(d_model, vocab)`
- `position_bias`, when provided, is added *after* the detach boundary so it
  remains differentiable
"""
function predict_detach_source(logits::AbstractArray,
                               embedding_weights::AbstractMatrix;
                               position_bias::Union{Nothing, AbstractArray}=nothing)
    detached_prediction = ignore_derivatives() do
        probs = _softmax_cols(logits)
        probs_2d = reshape(probs, size(probs, 1), :)
        pred_2d = embedding_weights * probs_2d
        reshape(pred_2d, size(embedding_weights, 1), size(logits)[2:end]...)
    end
    position_bias === nothing ? detached_prediction : detached_prediction .+ position_bias
end

# ============================================================================
# Neural comparators (differentiable)
# ============================================================================

"""
    neural_l2_comparator(a, b)

Differentiable L2 distance between tensors. For use as an obstruction loss
comparator in neural diagrams.
"""
function neural_l2_comparator(a::AbstractArray, b::AbstractArray)
    sqrt(sum((a .- b) .^ 2))
end

"""
    neural_l1_comparator(a, b)

Differentiable L1 distance between tensors.
"""
function neural_l1_comparator(a::AbstractArray, b::AbstractArray)
    sum(abs.(a .- b))
end

"""
    neural_cosine_comparator(a, b)

Differentiable cosine distance between tensors: `1 - cos(a, b)`.
"""
function neural_cosine_comparator(a::AbstractArray, b::AbstractArray)
    a_flat = reshape(a, :)
    b_flat = reshape(b, :)
    dot_ab = sum(a_flat .* b_flat)
    norm_a = sqrt(sum(a_flat .^ 2) + 1.0f-8)
    norm_b = sqrt(sum(b_flat .^ 2) + 1.0f-8)
    1.0f0 - dot_ab / (norm_a * norm_b)
end

# ============================================================================
# LuxDiagramModel — top-level Lux layer wrapping a FunctorFlow diagram
# ============================================================================

"""
    LuxDiagramModel(diagram; morphism_layers=Dict(), reducer_layers=Dict(),
                    comparator_layers=Dict())

A Lux model compiled from a FunctorFlow `Diagram`. This is the Lux equivalent
of `TorchCompiledDiagram` in the Python implementation.

## How it works

1. Morphisms bound as Lux layers participate in the autograd graph
2. KET reducers bound as `KETAttentionLayer` provide learnable attention
3. Obstruction losses use neural comparators for differentiable constraints
4. Non-neural operations (symbolic reducers, etc.) pass through unchanged

## Example

```julia
using FunctorFlow, Lux, Random

D = ket_block(; name=:MyKET, reducer=:ket_attention)
model = compile_to_lux(D;
    reducer_layers=Dict(:ket_attention => KETAttentionLayer(64)))

rng = Random.default_rng()
ps, st = Lux.setup(rng, model)

# source: (d_model, seq_len), mask: (seq_len, seq_len)
inputs = Dict(
    :Values => randn(Float32, 64, 10),
    :Incidence => Float32.(ones(10, 10))
)
output, st = model(inputs, ps, st)
```
"""
struct LuxDiagramModel <: LuxCore.AbstractLuxLayer
    diagram::Diagram
    compiled::CompiledDiagram
    morphism_layers::Dict{Symbol, LuxCore.AbstractLuxLayer}
    reducer_layers::Dict{Symbol, LuxCore.AbstractLuxLayer}
    comparator_layers::Dict{Symbol, LuxCore.AbstractLuxLayer}
    morphism_names::Vector{Symbol}
    reducer_names::Vector{Symbol}
    comparator_names::Vector{Symbol}
end

function LuxDiagramModel(D::Diagram;
                         morphism_layers::Dict=Dict{Symbol, LuxCore.AbstractLuxLayer}(),
                         reducer_layers::Dict=Dict{Symbol, LuxCore.AbstractLuxLayer}(),
                         comparator_layers::Dict=Dict{Symbol, LuxCore.AbstractLuxLayer}(),
                         morphisms::Union{Nothing, Dict}=nothing,
                         reducers::Union{Nothing, Dict}=nothing,
                         comparators::Union{Nothing, Dict}=nothing)
    # Build the base compiled diagram with any non-neural implementations
    compiled = compile_to_callable(D; morphisms=morphisms, reducers=reducers, comparators=comparators)

    ml = Dict{Symbol, LuxCore.AbstractLuxLayer}(Symbol(k) => v for (k, v) in morphism_layers)
    rl = Dict{Symbol, LuxCore.AbstractLuxLayer}(Symbol(k) => v for (k, v) in reducer_layers)
    cl = Dict{Symbol, LuxCore.AbstractLuxLayer}(Symbol(k) => v for (k, v) in comparator_layers)

    LuxDiagramModel(
        D, compiled, ml, rl, cl,
        sort(collect(keys(ml))),
        sort(collect(keys(rl))),
        sort(collect(keys(cl)))
    )
end

function LuxCore.initialparameters(rng::AbstractRNG, m::LuxDiagramModel)
    params = NamedTuple()
    for name in m.morphism_names
        lps = LuxCore.initialparameters(rng, m.morphism_layers[name])
        params = merge(params, NamedTuple{(Symbol("morph_", name),)}((lps,)))
    end
    for name in m.reducer_names
        lps = LuxCore.initialparameters(rng, m.reducer_layers[name])
        params = merge(params, NamedTuple{(Symbol("red_", name),)}((lps,)))
    end
    for name in m.comparator_names
        lps = LuxCore.initialparameters(rng, m.comparator_layers[name])
        params = merge(params, NamedTuple{(Symbol("comp_", name),)}((lps,)))
    end
    params
end

function LuxCore.initialstates(rng::AbstractRNG, m::LuxDiagramModel)
    states = NamedTuple()
    for name in m.morphism_names
        lst = LuxCore.initialstates(rng, m.morphism_layers[name])
        states = merge(states, NamedTuple{(Symbol("morph_", name),)}((lst,)))
    end
    for name in m.reducer_names
        lst = LuxCore.initialstates(rng, m.reducer_layers[name])
        states = merge(states, NamedTuple{(Symbol("red_", name),)}((lst,)))
    end
    for name in m.comparator_names
        lst = LuxCore.initialstates(rng, m.comparator_layers[name])
        states = merge(states, NamedTuple{(Symbol("comp_", name),)}((lst,)))
    end
    states
end

function (m::LuxDiagramModel)(inputs::AbstractDict, ps, st)
    env = Dict{Symbol, Any}(Symbol(k) => v for (k, v) in inputs)
    new_st = st

    # Execute operations in insertion order
    # Store results under both op.name and op.target (when safe) so
    # downstream ops can reference either the operation or the target object.
    for op in values(m.diagram.operations)
        if op isa Morphism
            val = _lux_execute_morphism(m, op, env, ps, new_st)
            env[op.name] = val
            if op.target != op.source && !haskey(env, op.target)
                env[op.target] = val
            end
        elseif op isa Composition
            val = _lux_execute_composition(m, op, env, ps, new_st)
            env[op.name] = val
            if op.target != op.source && !haskey(env, op.target)
                env[op.target] = val
            end
        elseif op isa KanExtension
            val, new_st = _lux_execute_kan(m, op, env, ps, new_st)
            env[op.name] = val
            if op.target !== nothing && !haskey(env, op.target)
                env[op.target] = val
            end
        end
    end

    # Compute losses
    losses = Dict{Symbol, Any}()
    for loss in values(m.diagram.losses)
        losses[loss.name] = _lux_execute_loss(m, loss, env, ps, new_st)
    end

    result = Dict{Symbol, Any}(:values => env, :losses => losses)
    result, new_st
end

# ---------------------------------------------------------------------------
# Internal execution helpers
# ---------------------------------------------------------------------------

function _lux_execute_morphism(m::LuxDiagramModel, morph::Morphism,
                               env::Dict{Symbol, Any}, ps, st)
    haskey(env, morph.source) || error("Missing source :$(morph.source) for morphism :$(morph.name)")
    input = env[morph.source]

    # Check for Lux layer first
    lux_name = morph.name
    impl_key = morph.implementation_key !== nothing ? morph.implementation_key : morph.name
    if haskey(m.morphism_layers, lux_name)
        layer = m.morphism_layers[lux_name]
        ps_key = Symbol("morph_", lux_name)
        lps = getfield(ps, ps_key)
        lst = getfield(st, ps_key)
        output, _ = layer(input, lps, lst)
        return output
    elseif haskey(m.morphism_layers, impl_key)
        layer = m.morphism_layers[impl_key]
        ps_key = Symbol("morph_", impl_key)
        lps = getfield(ps, ps_key)
        lst = getfield(st, ps_key)
        output, _ = layer(input, lps, lst)
        return output
    end

    # Fall back to compiled (non-neural) implementation
    fn = get(m.compiled.morphisms, morph.name, nothing)
    if fn === nothing && morph.implementation_key !== nothing
        fn = get(m.compiled.morphisms, morph.implementation_key, nothing)
    end
    fn === nothing && error("No implementation (Lux or callable) for morphism :$(morph.name)")
    fn(input)
end

function _lux_execute_composition(m::LuxDiagramModel, comp::Composition,
                                  env::Dict{Symbol, Any}, ps, st)
    haskey(env, comp.source) || error("Missing source :$(comp.source) for composition :$(comp.name)")
    current = env[comp.source]

    for morph_name in comp.chain
        # Check for Lux layer
        if haskey(m.morphism_layers, morph_name)
            layer = m.morphism_layers[morph_name]
            ps_key = Symbol("morph_", morph_name)
            lps = getfield(ps, ps_key)
            lst = getfield(st, ps_key)
            current, _ = layer(current, lps, lst)
        else
            fn = get(m.compiled.morphisms, morph_name, nothing)
            fn === nothing && error("No implementation for morphism :$(morph_name) in composition :$(comp.name)")
            current = fn(current)
        end
    end
    current
end

function _lux_execute_kan(m::LuxDiagramModel, kan::KanExtension,
                          env::Dict{Symbol, Any}, ps, st)
    haskey(env, kan.source) || error("Missing source :$(kan.source) for Kan extension :$(kan.name)")
    haskey(env, kan.along) || error("Missing relation :$(kan.along) for Kan extension :$(kan.name)")

    source_val = env[kan.source]
    along_val = env[kan.along]

    # Check for Lux reducer layer
    if haskey(m.reducer_layers, kan.reducer)
        layer = m.reducer_layers[kan.reducer]
        ps_key = Symbol("red_", kan.reducer)
        lps = getfield(ps, ps_key)
        lst = getfield(st, ps_key)
        result, new_lst = layer((source_val, along_val), lps, lst)
        new_st = merge(st, NamedTuple{(ps_key,)}((new_lst,)))
        return result, new_st
    end

    # Fall back to compiled reducer
    reducer = get(m.compiled.reducers, kan.reducer, nothing)
    reducer === nothing && error("No reducer :$(kan.reducer) for Kan extension :$(kan.name)")
    metadata = Dict{String, Any}("direction" => string(kan.direction))
    merge!(metadata, Dict{String, Any}(string(k) => v for (k, v) in kan.metadata))
    result = reducer(source_val, along_val, metadata)
    result, st
end

function _lux_execute_loss(m::LuxDiagramModel, loss::ObstructionLoss,
                           env::Dict{Symbol, Any}, ps, st)
    # Use neural comparator if available, otherwise fall back
    if haskey(m.comparator_layers, loss.comparator)
        layer = m.comparator_layers[loss.comparator]
        ps_key = Symbol("comp_", loss.comparator)
        lps = getfield(ps, ps_key)
        lst = getfield(st, ps_key)
        total = 0.0f0
        for (left_name, right_name) in loss.paths
            haskey(env, left_name) || error("Missing :$(left_name) for loss :$(loss.name)")
            haskey(env, right_name) || error("Missing :$(right_name) for loss :$(loss.name)")
            val, _ = layer((env[left_name], env[right_name]), lps, lst)
            total = total + val
        end
        return loss.weight * total
    end

    # Neural comparator functions (no parameters)
    comparator = if loss.comparator == :l2
        neural_l2_comparator
    elseif loss.comparator == :l1
        neural_l1_comparator
    elseif loss.comparator == :cosine
        neural_cosine_comparator
    else
        get(m.compiled.comparators, loss.comparator, nothing)
    end

    comparator === nothing && error("No comparator :$(loss.comparator) for loss :$(loss.name)")
    total = 0.0f0
    for (left_name, right_name) in loss.paths
        haskey(env, left_name) || error("Missing :$(left_name) for loss :$(loss.name)")
        haskey(env, right_name) || error("Missing :$(right_name) for loss :$(loss.name)")
        lv = env[left_name]
        rv = env[right_name]
        if lv isa AbstractArray && rv isa AbstractArray
            total = total + comparator(lv, rv)
        else
            total = total + Float32(m.compiled.comparators[loss.comparator](lv, rv))
        end
    end
    loss.weight * total
end

# ============================================================================
# compile_to_lux — main entry point
# ============================================================================

"""
    compile_to_lux(D::Diagram; morphism_layers=Dict(), reducer_layers=Dict(),
                   comparator_layers=Dict(), morphisms=nothing, reducers=nothing,
                   comparators=nothing) -> LuxDiagramModel

Compile a FunctorFlow `Diagram` to a Lux model for differentiable execution.

Lux layers override callable implementations: any morphism or reducer bound
as a `LuxCore.AbstractLuxLayer` will participate in the autograd graph, while
non-neural operations pass through unchanged.

## Arguments
- `D`: The FunctorFlow diagram to compile
- `morphism_layers`: Dict mapping morphism names to Lux layers
- `reducer_layers`: Dict mapping reducer names to Lux layers (e.g., `KETAttentionLayer`)
- `comparator_layers`: Dict mapping comparator names to Lux layers
- `morphisms`, `reducers`, `comparators`: Non-neural callable overrides

## Example

```julia
using FunctorFlow, Lux, Random

# Build a KET block with learned attention
D = ket_block(; name=:MyKET, reducer=:ket_attention)
model = compile_to_lux(D;
    reducer_layers=Dict(:ket_attention => KETAttentionLayer(64; n_heads=4)))

# Initialize parameters
rng = Random.default_rng()
ps, st = Lux.setup(rng, model)

# Forward pass
inputs = Dict(:Values => randn(Float32, 64, 10),
              :Incidence => Float32.(ones(10, 10)))
result, st = model(inputs, ps, st)
contextualized = result[:values][:aggregate]
```
"""
function compile_to_lux(D::Diagram;
                        morphism_layers::Dict=Dict{Symbol, LuxCore.AbstractLuxLayer}(),
                        reducer_layers::Dict=Dict{Symbol, LuxCore.AbstractLuxLayer}(),
                        comparator_layers::Dict=Dict{Symbol, LuxCore.AbstractLuxLayer}(),
                        morphisms::Union{Nothing, Dict}=nothing,
                        reducers::Union{Nothing, Dict}=nothing,
                        comparators::Union{Nothing, Dict}=nothing)
    LuxDiagramModel(D;
                    morphism_layers=morphism_layers,
                    reducer_layers=reducer_layers,
                    comparator_layers=comparator_layers,
                    morphisms=morphisms,
                    reducers=reducers,
                    comparators=comparators)
end

# ============================================================================
# Convenience builders for KET language model patterns
# ============================================================================

"""
    build_ket_lux_model(d_model; n_heads=4, reducer=:ket_attention, kwargs...)

Build a KET block with a learned attention reducer as a Lux model.
This is the standard pattern for a Kan Extension Transformer head.

Returns `(model, diagram)` where `model` is a `LuxDiagramModel`.
"""
function build_ket_lux_model(d_model::Int;
                             n_heads::Int=4,
                             dropout::Real=0.0f0,
                             reducer::Symbol=:ket_attention,
                             kwargs...)
    D = ket_block(; reducer=reducer, kwargs...)
    attn = KETAttentionLayer(d_model; n_heads=n_heads, dropout=dropout, name=reducer)
    model = compile_to_lux(D; reducer_layers=Dict(reducer => attn))
    (model, D)
end

"""
    build_db_lux_model(d_model; comparator=:l2, kwargs...)

Build a DB square with neural morphisms as a Lux model.
The two morphisms `f` and `g` are `DiagramDenseLayer`s, and the obstruction
loss measures non-commutativity of `f∘g` vs `g∘f` using a neural comparator.

Returns `(model, diagram)`.
"""
function build_db_lux_model(d_model::Int;
                            comparator::Symbol=:l2,
                            kwargs...)
    D = db_square(; kwargs...)
    cfg = DBSquareConfig(; kwargs...)
    f_layer = DiagramDenseLayer(d_model, d_model; name=cfg.first_morphism)
    g_layer = DiagramDenseLayer(d_model, d_model; name=cfg.second_morphism)
    model = compile_to_lux(D;
                           morphism_layers=Dict(cfg.first_morphism => f_layer,
                                                cfg.second_morphism => g_layer))
    (model, D)
end

"""
    build_gt_lux_model(d_model; n_heads=4, kwargs...)

Build a GT (Graph Transformer) neighborhood block as a Lux model.
The lift morphism is a `DiagramDenseLayer` and the aggregation uses
`KETAttentionLayer`.

Returns `(model, diagram)`.
"""
function build_gt_lux_model(d_model::Int;
                            n_heads::Int=4,
                            dropout::Real=0.0f0,
                            reducer::Symbol=:ket_attention,
                            kwargs...)
    D = gt_neighborhood_block(; reducer=reducer, kwargs...)
    cfg = GTNeighborhoodConfig(; kwargs...)
    lift_layer = DiagramDenseLayer(d_model, d_model; name=cfg.lift_name)
    attn = KETAttentionLayer(d_model; n_heads=n_heads, dropout=dropout, name=reducer)
    model = compile_to_lux(D;
                           morphism_layers=Dict(cfg.lift_name => lift_layer),
                           reducer_layers=Dict(reducer => attn))
    (model, D)
end

"""
    build_basket_rocket_lux_model(d_model; n_heads=4, kwargs...)

Build a Lux-backed `BASKET → ROCKET` planner. Both the drafting and repair
stages are instantiated as learnable attention reducers, and the
draft/repair consistency loss is switched to a differentiable comparator.

Returns `(model, diagram)`.
"""
function build_basket_rocket_lux_model(d_model::Int;
                                       n_heads::Int=4,
                                       dropout::Real=0.0f0,
                                       draft_reducer::Symbol=:draft_attention,
                                       repair_reducer::Symbol=:repair_attention,
                                       comparator::Symbol=:l2,
                                       kwargs...)
    basket_cfg = BASKETWorkflowConfig(; reducer=draft_reducer)
    rocket_cfg = ROCKETRepairConfig(; reducer=repair_reducer)
    pipeline_cfg = BasketRocketPipelineConfig(;
        basket_config=basket_cfg,
        rocket_config=rocket_cfg,
        consistency_comparator=comparator
    )
    D = basket_rocket_pipeline(; config=pipeline_cfg, kwargs...)
    draft_layer = KETAttentionLayer(d_model; n_heads=n_heads, dropout=dropout, name=draft_reducer)
    repair_layer = KETAttentionLayer(d_model; n_heads=n_heads, dropout=dropout, name=repair_reducer)
    model = compile_to_lux(D;
                           reducer_layers=Dict(
                               draft_reducer => draft_layer,
                               repair_reducer => repair_layer
                           ))
    (model, D)
end
