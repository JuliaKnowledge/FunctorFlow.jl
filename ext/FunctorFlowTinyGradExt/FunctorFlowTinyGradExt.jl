# ============================================================================
# FunctorFlowTinyGradExt — TinyGrad backend extension for FunctorFlow
# ============================================================================
#
# Two execution backends:
#
# 1. TinyGradBackend    — array round-trip (compatible with all morphisms).
#                         Each morphism is wrapped so that TinyTensor inputs
#                         are materialised, the user function runs on plain
#                         Julia arrays / dicts, and the result is wrapped
#                         back into a TinyTensor.  This mirrors CatNet.jl's
#                         CatNetTinyGradExt.TinyGradBackend.
#
# 2. UOpCompiledBackend — native UOp tracing.  Morphisms whose
#                         implementation is expressed in TinyGrad tensor
#                         operations (`+`, `*`, `tensor_relu`,
#                         `tensor_matmul`, …) are traced through TinyGrad's
#                         lazy tensor system into a fused UOp DAG that is
#                         simplified + DCE-d once during `lower()` and
#                         re-interpreted on each `realize()`.  Morphisms
#                         that fall back to Julia broadcasting or that
#                         operate on dicts (the common FunctorFlow KET
#                         pattern) cannot be traced; those morphisms are
#                         re-executed opaquely at the boundary, exactly as
#                         they would be under TinyGradBackend.
#
# The parent module exposes shim functions
# (`compile_to_tinygrad`, `tinygrad_backend`, `uop_compiled_backend`)
# that resolve to the definitions here via `Base.get_extension`.
# ============================================================================

module FunctorFlowTinyGradExt

using FunctorFlow
using FunctorFlow: Diagram, Morphism, Composition, KanExtension, ObstructionLoss,
                   CompiledDiagram, compile_to_callable,
                   AbstractFunctorFlowBackend,
                   BUILTIN_REDUCERS, BUILTIN_COMPARATORS,
                   ExecutionResult
using TinyGrad
using OrderedCollections: OrderedDict

import FunctorFlow: lower, realize, backend_name, supports_dtype,
                    compile_to_tinygrad

# ════════════════════════════════════════════════════════════════════════
# Shared utilities
# ════════════════════════════════════════════════════════════════════════

"""Convert a Julia array → TinyTensor (dtype inferred from eltype)."""
function to_tiny_tensor(x::AbstractArray{T}) where T
    dt = T == Float32 ? TinyGrad.dtypes.float32 :
         T == Bool ? TinyGrad.dtypes.bool_ :
         T == Float64 ? TinyGrad.dtypes.float64 :
         nothing
    dt === nothing && error("TinyGradBackend does not support Julia array eltype $(T)")
    TinyGrad.TinyTensor(x; dtype=dt)
end

"""Convert a TinyTensor → Julia Array."""
from_tiny_tensor(t::TinyGrad.TinyTensor) = TinyGrad.to_array(t)

"""Coerce a value to a Julia array (passes through non-tensors)."""
_to_array(x) = x isa TinyGrad.TinyTensor ? from_tiny_tensor(x) : x

"""
    _wrap_for_tinygrad(fn)

Wrap a Julia morphism implementation so it round-trips TinyTensors via
plain arrays.  If the input is not a TinyTensor (e.g. a Dict for KET-style
morphisms) the function is called as-is and the result is left untouched.
"""
function _wrap_for_tinygrad(fn)
    function wrapped(x)
        if x isa TinyGrad.TinyTensor
            arr = TinyGrad.to_array(x)
            result = fn(arr)
            return result isa AbstractArray ? to_tiny_tensor(result) : result
        else
            return fn(x)
        end
    end
    return wrapped
end

# Resolve a morphism implementation, honouring `implementation_key`.
function _morphism_fn(morphisms::Dict, m::Morphism)
    fn = get(morphisms, m.name, nothing)
    if fn === nothing && m.implementation_key !== nothing
        fn = get(morphisms, m.implementation_key, nothing)
    end
    return fn
end

# ════════════════════════════════════════════════════════════════════════
# TinyGradBackend — array round-trip (fully compatible)
# ════════════════════════════════════════════════════════════════════════

"""
    TinyGradBackend <: AbstractFunctorFlowBackend

Backend that executes FunctorFlow diagrams using TinyGrad TinyTensor
storage with array round-trip at every morphism boundary.  Compatible
with every FunctorFlow morphism (whether tensor-pure or dict-based).
"""
struct TinyGradBackend <: AbstractFunctorFlowBackend end

backend_name(::TinyGradBackend) = "tinygrad"
supports_dtype(::TinyGradBackend, ::Type{Float32}) = true
supports_dtype(::TinyGradBackend, ::Type{Float64}) = true
supports_dtype(::TinyGradBackend, ::Type{Bool}) = true

# Lowering for the round-trip backend is a no-op — execution simply
# replays the diagram with TinyTensor storage.
function lower(::TinyGradBackend, D::Diagram, params::Dict=Dict(), inputs::Dict=Dict())
    return D
end

function realize(::TinyGradBackend, D::Diagram, inputs::AbstractDict;
                 morphisms::Dict=D.implementations,
                 reducers::Dict=BUILTIN_REDUCERS,
                 comparators::Dict=BUILTIN_COMPARATORS)
    env = Dict{Symbol, Any}()
    for (name, val) in inputs
        sym = Symbol(name)
        env[sym] = val isa AbstractArray ? to_tiny_tensor(val) : val
    end

    for op in values(D.operations)
        _execute_op_roundtrip!(env, op, D, morphisms, reducers)
    end

    # Materialise final values as Julia arrays (unwrap TinyTensors)
    values_out = Dict{Symbol, Any}()
    for (k, v) in env
        values_out[k] = v isa TinyGrad.TinyTensor ? from_tiny_tensor(v) : v
    end

    losses = _compute_losses(D, values_out, comparators)
    return ExecutionResult(values_out, losses)
end

function _execute_op_roundtrip!(env, op::Morphism, D::Diagram, morphisms, reducers)
    haskey(env, op.source) || error("Missing source value :$(op.source) for morphism :$(op.name)")
    fn = _morphism_fn(morphisms, op)
    fn === nothing && error("No implementation bound for morphism :$(op.name)")
    wrapped = _wrap_for_tinygrad(fn)
    val = wrapped(env[op.source])
    env[op.name] = val
    if op.target != op.source
        env[op.target] = val
    end
end

function _execute_op_roundtrip!(env, op::Composition, D::Diagram, morphisms, reducers)
    haskey(env, op.source) || error("Missing source value :$(op.source) for composition :$(op.name)")
    current = env[op.source]
    for morph_name in op.chain
        m = D.operations[morph_name]
        m isa Morphism || error("Composition :$(op.name) entry :$(morph_name) is not a Morphism")
        fn = _morphism_fn(morphisms, m)
        fn === nothing && error("No implementation bound for morphism :$(morph_name)")
        wrapped = _wrap_for_tinygrad(fn)
        current = wrapped(current)
    end
    env[op.name] = current
end

function _execute_op_roundtrip!(env, op::KanExtension, D::Diagram, morphisms, reducers)
    haskey(env, op.source) || error("Missing source value :$(op.source) for Kan :$(op.name)")
    haskey(env, op.along)  || error("Missing relation value :$(op.along) for Kan :$(op.name)")
    reducer = get(reducers, op.reducer, nothing)
    reducer === nothing && error("No reducer :$(op.reducer) bound for Kan :$(op.name)")
    metadata = Dict{String, Any}("direction" => string(op.direction))
    for (k, v) in op.metadata
        metadata[string(k)] = v
    end
    src = _to_array(env[op.source])
    along = _to_array(env[op.along])
    raw = reducer(src, along, metadata)
    val = raw isa AbstractArray ? to_tiny_tensor(raw) : raw
    env[op.name] = val
    if op.target !== nothing && op.target != op.name
        env[op.target] = val
    end
end

function _compute_losses(D::Diagram, values_out::Dict, comparators::Dict)
    losses = Dict{Symbol, Float64}()
    for loss in values(D.losses)
        comparator = get(comparators, loss.comparator, nothing)
        comparator === nothing && error("No comparator :$(loss.comparator) for loss :$(loss.name)")
        total = 0.0
        for (a, b) in loss.paths
            haskey(values_out, a) || error("Missing path :$a for loss :$(loss.name)")
            haskey(values_out, b) || error("Missing path :$b for loss :$(loss.name)")
            total += Float64(comparator(values_out[a], values_out[b]))
        end
        losses[loss.name] = loss.weight * total
    end
    return losses
end

# ════════════════════════════════════════════════════════════════════════
# UOpCompiledBackend — native UOp tracing (with opaque fallback)
# ════════════════════════════════════════════════════════════════════════

"""
    UOpCompiledBackend <: AbstractFunctorFlowBackend

Backend that traces FunctorFlow diagrams through TinyGrad's lazy tensor
system into a fused UOp DAG.  Tensor-pure morphisms (those using `+`,
`*`, `tensor_relu`, `tensor_matmul`, …) are fused across morphism
boundaries.  Morphisms that cannot be traced (Julia broadcasting,
dict-based reducers, etc.) fall back to opaque array round-trip — the
same semantics as `TinyGradBackend` for those individual ops.
"""
struct UOpCompiledBackend <: AbstractFunctorFlowBackend end

backend_name(::UOpCompiledBackend) = "uop_compiled"
supports_dtype(::UOpCompiledBackend, ::Type{Float32}) = true
supports_dtype(::UOpCompiledBackend, ::Type{Float64}) = true
supports_dtype(::UOpCompiledBackend, ::Type{Bool}) = true

# ── Compiled representation ──────────────────────────────────────────

"""Metadata for one diagram input: the BUFFER UOp to fill plus shape/dtype."""
struct InputSlot
    buf_uop::TinyGrad.UOp
    shape::Tuple{Vararg{Int}}
    jl_dtype::Type
    tg_dtype::TinyGrad.DType
end

"""One operation in the compiled execution plan."""
struct CompiledOp
    name::Symbol
    kind::Symbol         # :traced, :opaque, :traced_kan
    op_kind::Symbol      # :morphism, :composition, :left_kan, :right_kan
    spec::Any            # the original FunctorFlow op
    output_uop::Union{Nothing, TinyGrad.UOp}
    output_shape::Tuple{Vararg{Int}}
    output_dtype::Type
end

"""
    UOpCompiledDiagram

Pre-compiled FunctorFlow diagram: input slot table, per-op execution
plan, and optimized terminal UOps for each value in the diagram.
Returned by `lower(::UOpCompiledBackend, ...)`.
"""
struct UOpCompiledDiagram
    name::Symbol
    diagram::Diagram
    input_slots::OrderedDict{Symbol, InputSlot}
    compiled_ops::Vector{CompiledOp}
    terminal_uops::OrderedDict{Symbol, TinyGrad.UOp}
    fully_traced::Bool
    stats::NamedTuple
    morphisms::Dict
    reducers::Dict
    comparators::Dict
    _refs::Vector{Any}  # GC anchor for traced TinyTensors
end

# ── Tracing helpers ──────────────────────────────────────────────────

function _find_buffer_uop(u::TinyGrad.UOp)
    u.op === TinyGrad.BUFFER && return u
    u.op === TinyGrad.RESHAPE && length(u.src) == 1 && return _find_buffer_uop(u.src[1])
    return nothing
end

function _try_trace(fn, input::TinyGrad.TinyTensor)
    try
        result = fn(input)
        result isa TinyGrad.TinyTensor && return result
        return nothing
    catch
        return nothing
    end
end

function _try_trace_kan_sum(source::TinyGrad.TinyTensor, along::TinyGrad.TinyTensor)
    try
        return TinyGrad.tensor_matmul(TinyGrad.tensor_transpose(along), source)
    catch
        return nothing
    end
end

function _try_trace_kan_mean(source::TinyGrad.TinyTensor, along::TinyGrad.TinyTensor)
    try
        agg = TinyGrad.tensor_matmul(TinyGrad.tensor_transpose(along), source)
        col_sums = TinyGrad.tensor_sum(along; axis=0, keepdims=true)
        col_sums_t = TinyGrad.tensor_transpose(col_sums)
        denom = TinyGrad.tensor_maximum(col_sums_t, TinyGrad.TinyTensor(1.0f0))
        return TinyGrad.tensor_div(agg, denom)
    catch
        return nothing
    end
end

_count_uop_nodes(u::TinyGrad.UOp) = TinyGrad.count_nodes(u)

# Try to discover the (shape, dtype) of an input object from the runtime
# `inputs` dict provided to `lower()`.  FF's FFObject does not carry
# explicit shape/dtype, so we infer them from concrete dummy inputs.
function _infer_input_shape_dtype(val)
    if val isa AbstractArray
        T = eltype(val)
        dt = T == Float32 ? TinyGrad.dtypes.float32 :
             T == Float64 ? TinyGrad.dtypes.float64 :
             T == Bool    ? TinyGrad.dtypes.bool_   :
             nothing
        dt === nothing && return nothing
        return (Tuple(size(val)), T, dt)
    end
    return nothing
end

# ── lower: trace + optimize ──────────────────────────────────────────

function lower(::UOpCompiledBackend, D::Diagram, params::Dict=Dict(), inputs::Dict=Dict();
               morphisms::Dict=D.implementations,
               reducers::Dict=BUILTIN_REDUCERS,
               comparators::Dict=BUILTIN_COMPARATORS)
    refs = Any[]
    input_slots = OrderedDict{Symbol, InputSlot}()
    compiled_ops = CompiledOp[]
    env = OrderedDict{Symbol, Any}()

    # Phase 1: build TinyTensor placeholders for inputs we have shape info for.
    for (name, val) in inputs
        sym = Symbol(name)
        info = _infer_input_shape_dtype(val)
        if info === nothing
            env[sym] = val
            continue
        end
        shape, jl_T, dt = info
        dummy = zeros(jl_T, shape...)
        tt = TinyGrad.TinyTensor(dummy; dtype=dt, requires_grad=true)
        env[sym] = tt
        push!(refs, tt)
        buf = _find_buffer_uop(tt.uop)
        if buf !== nothing
            input_slots[sym] = InputSlot(buf, shape, jl_T, dt)
        end
    end

    n_traced = 0
    n_opaque = 0

    # Phase 2: trace operations in insertion order
    for op in values(D.operations)
        traced_count, opaque_count = _trace_op!(op, env, D, morphisms, reducers,
                                                compiled_ops, refs)
        n_traced += traced_count
        n_opaque += opaque_count
    end

    # Phase 3: optimize terminal UOps
    n_nodes_before = 0
    n_nodes_after = 0
    terminal_uops = OrderedDict{Symbol, TinyGrad.UOp}()
    for (name, val) in env
        val isa TinyGrad.TinyTensor || continue
        u = val.uop
        n_before = _count_uop_nodes(u)
        opt = TinyGrad.optimize(u)
        n_after = _count_uop_nodes(opt)
        n_nodes_before += n_before
        n_nodes_after += n_after
        terminal_uops[name] = opt
        push!(refs, opt)
    end

    stats = (n_traced=n_traced, n_opaque=n_opaque,
             n_nodes_before=n_nodes_before, n_nodes_after=n_nodes_after)

    return UOpCompiledDiagram(
        D.name, D, input_slots, compiled_ops, terminal_uops,
        n_opaque == 0, stats,
        morphisms, reducers, comparators, refs)
end

function _trace_op!(op::Morphism, env, D, morphisms, reducers, compiled_ops, refs)
    fn = _morphism_fn(morphisms, op)
    fn === nothing && error("No implementation bound for morphism :$(op.name)")
    src_val = get(env, op.source, nothing)
    src_val === nothing && error("Missing source :$(op.source) for morphism :$(op.name)")

    traced = src_val isa TinyGrad.TinyTensor ? _try_trace(fn, src_val) : nothing
    if traced !== nothing
        env[op.name] = traced
        if op.target != op.source
            env[op.target] = traced
        end
        push!(refs, traced)
        push!(compiled_ops, CompiledOp(
            op.name, :traced, :morphism, op,
            traced.uop, TinyGrad.tensor_shape(traced),
            TinyGrad.julia_eltype(TinyGrad.tensor_dtype(traced))))
        return (1, 0)
    else
        # Opaque: materialize, call, wrap back
        arr = src_val isa TinyGrad.TinyTensor ? TinyGrad.to_array(src_val) : src_val
        result = fn(arr)
        if result isa AbstractArray
            tt = to_tiny_tensor(result)
            env[op.name] = tt
            if op.target != op.source
                env[op.target] = tt
            end
            push!(refs, tt)
            push!(compiled_ops, CompiledOp(
                op.name, :opaque, :morphism, op,
                nothing, Tuple(size(result)), eltype(result)))
        else
            env[op.name] = result
            if op.target != op.source
                env[op.target] = result
            end
            push!(compiled_ops, CompiledOp(
                op.name, :opaque, :morphism, op,
                nothing, (), typeof(result)))
        end
        return (0, 1)
    end
end

function _trace_op!(op::Composition, env, D, morphisms, reducers, compiled_ops, refs)
    src_val = get(env, op.source, nothing)
    src_val === nothing && error("Missing source :$(op.source) for composition :$(op.name)")
    current = src_val
    n_traced = 0
    n_opaque = 0
    for morph_name in op.chain
        m = D.operations[morph_name]
        m isa Morphism || error("Composition entry :$morph_name is not a Morphism")
        fn = _morphism_fn(morphisms, m)
        fn === nothing && error("No implementation bound for morphism :$morph_name")
        traced = current isa TinyGrad.TinyTensor ? _try_trace(fn, current) : nothing
        if traced !== nothing
            current = traced
            push!(refs, traced)
            n_traced += 1
        else
            arr = current isa TinyGrad.TinyTensor ? TinyGrad.to_array(current) : current
            result = fn(arr)
            current = result isa AbstractArray ? to_tiny_tensor(result) : result
            current isa TinyGrad.TinyTensor && push!(refs, current)
            n_opaque += 1
        end
    end
    env[op.name] = current
    if current isa TinyGrad.TinyTensor
        push!(compiled_ops, CompiledOp(
            op.name, n_opaque == 0 ? :traced : :opaque, :composition, op,
            n_opaque == 0 ? current.uop : nothing,
            TinyGrad.tensor_shape(current),
            TinyGrad.julia_eltype(TinyGrad.tensor_dtype(current))))
    else
        push!(compiled_ops, CompiledOp(
            op.name, :opaque, :composition, op,
            nothing, (), typeof(current)))
    end
    return (n_traced, n_opaque)
end

function _trace_op!(op::KanExtension, env, D, morphisms, reducers, compiled_ops, refs)
    src = get(env, op.source, nothing)
    along = get(env, op.along, nothing)
    src === nothing && error("Missing source :$(op.source) for Kan :$(op.name)")
    along === nothing && error("Missing along :$(op.along) for Kan :$(op.name)")

    traced = nothing
    if src isa TinyGrad.TinyTensor && along isa TinyGrad.TinyTensor
        if op.direction == FunctorFlow.LEFT
            if op.reducer === :sum
                traced = _try_trace_kan_sum(src, along)
            elseif op.reducer === :mean
                traced = _try_trace_kan_mean(src, along)
            end
        elseif op.direction == FunctorFlow.RIGHT
            along_t = TinyGrad.tensor_transpose(along)
            if op.reducer === :sum
                traced = _try_trace_kan_sum(src, along_t)
            elseif op.reducer === :mean
                traced = _try_trace_kan_mean(src, along_t)
            end
        end
    end

    if traced !== nothing
        env[op.name] = traced
        if op.target !== nothing && op.target != op.name
            env[op.target] = traced
        end
        push!(refs, traced)
        op_kind = op.direction == FunctorFlow.LEFT ? :left_kan : :right_kan
        push!(compiled_ops, CompiledOp(
            op.name, :traced_kan, op_kind, op,
            traced.uop, TinyGrad.tensor_shape(traced),
            TinyGrad.julia_eltype(TinyGrad.tensor_dtype(traced))))
        return (1, 0)
    else
        # Opaque Kan fallback via FF reducer registry (operates on Julia values)
        reducer = get(reducers, op.reducer, nothing)
        reducer === nothing && error("No reducer :$(op.reducer) for Kan :$(op.name)")
        metadata = Dict{String, Any}("direction" => string(op.direction))
        for (k, v) in op.metadata
            metadata[string(k)] = v
        end
        src_arr = src isa TinyGrad.TinyTensor ? TinyGrad.to_array(src) : src
        along_arr = along isa TinyGrad.TinyTensor ? TinyGrad.to_array(along) : along
        result = reducer(src_arr, along_arr, metadata)
        val = result isa AbstractArray ? to_tiny_tensor(result) : result
        env[op.name] = val
        if op.target !== nothing && op.target != op.name
            env[op.target] = val
        end
        val isa TinyGrad.TinyTensor && push!(refs, val)
        op_kind = op.direction == FunctorFlow.LEFT ? :left_kan : :right_kan
        push!(compiled_ops, CompiledOp(
            op.name, :opaque, op_kind, op,
            nothing,
            val isa TinyGrad.TinyTensor ? TinyGrad.tensor_shape(val) : (),
            val isa TinyGrad.TinyTensor ?
                TinyGrad.julia_eltype(TinyGrad.tensor_dtype(val)) : typeof(val)))
        return (0, 1)
    end
end

# ── realize: swap data + interpret ──────────────────────────────────

function realize(::UOpCompiledBackend, compiled::UOpCompiledDiagram, inputs::AbstractDict)
    if compiled.fully_traced
        return _realize_fully_traced(compiled, inputs)
    else
        return _realize_hybrid(compiled, inputs)
    end
end

function _clear_uop_cache!(compiled::UOpCompiledDiagram)
    seen = Set{UInt}()
    for (_, opt_uop) in compiled.terminal_uops
        _walk_and_clear!(opt_uop, seen)
    end
end

function _walk_and_clear!(u::TinyGrad.UOp, seen::Set{UInt})
    h = objectid(u)
    h in seen && return
    push!(seen, h)
    delete!(TinyGrad._UOP_ARRAY_CACHE, u)
    for s in u.src
        _walk_and_clear!(s, seen)
    end
end

function _swap_inputs!(compiled::UOpCompiledDiagram, inputs::AbstractDict)
    for (name, slot) in compiled.input_slots
        sym = Symbol(name)
        haskey(inputs, sym) || haskey(inputs, name) || continue
        val = get(inputs, sym, get(inputs, name, nothing))
        arr = val isa AbstractArray ? val :
              error("expected Array for input :$(name)")
        jl_T = TinyGrad.julia_eltype(slot.tg_dtype)
        typed = jl_T === eltype(arr) ? arr : jl_T.(arr)
        flat = TinyGrad.row_major_vec(typed)
        TinyGrad._TENSOR_DATA[slot.buf_uop] = flat
    end
end

function _interpret_terminal(opt_uop::TinyGrad.UOp)
    shape = TinyGrad.uop_shape(opt_uop)
    T = TinyGrad.julia_eltype(opt_uop.dtype)
    raw = TinyGrad.interpret(opt_uop)
    if isempty(shape)
        return raw isa Number ? T(raw) : T(first(raw))
    elseif raw isa Number
        return fill(T(raw), shape...)
    elseif raw isa AbstractArray && Base.size(raw) == shape && eltype(raw) === T
        return raw
    elseif raw isa Vector && length(raw) == prod(shape) && length(shape) > 1
        return reshape(T.(raw), shape...)
    else
        result = raw isa AbstractArray ? raw : [raw]
        if Base.size(result) != shape
            result = reshape(collect(T, Iterators.flatten(result)), shape...)
        end
        return eltype(result) === T ? result : T.(result)
    end
end

function _realize_fully_traced(compiled::UOpCompiledDiagram, inputs::AbstractDict)
    _swap_inputs!(compiled, inputs)
    _clear_uop_cache!(compiled)

    values_out = Dict{Symbol, Any}()
    # Forward any non-tensor inputs unchanged.
    for (name, val) in inputs
        sym = Symbol(name)
        if !haskey(compiled.input_slots, sym)
            values_out[sym] = val
        end
    end
    # Inputs that were realized as buffers: re-expose the array form.
    for (name, slot) in compiled.input_slots
        if haskey(inputs, name) || haskey(inputs, Symbol(name))
            values_out[name] = get(inputs, Symbol(name), get(inputs, name, nothing))
        end
    end

    for (name, opt_uop) in compiled.terminal_uops
        values_out[name] = _interpret_terminal(opt_uop)
    end

    losses = _compute_losses(compiled.diagram, values_out, compiled.comparators)
    return ExecutionResult(values_out, losses)
end

# Hybrid: at least one opaque op — re-execute the diagram with real inputs
# using the array round-trip semantics, but we still got the benefit of
# tracing-time validation + a simplified terminal graph for any traced
# subgraphs.
function _realize_hybrid(compiled::UOpCompiledDiagram, inputs::AbstractDict)
    return realize(TinyGradBackend(), compiled.diagram, inputs;
                   morphisms=compiled.morphisms,
                   reducers=compiled.reducers,
                   comparators=compiled.comparators)
end

# ════════════════════════════════════════════════════════════════════════
# FFTinyGradModel — user-facing compile_to_tinygrad result
# ════════════════════════════════════════════════════════════════════════

"""
    FFTinyGradModel

Result of `compile_to_tinygrad`.  Callable: `model(inputs)` returns an
`ExecutionResult`.  Holds the chosen backend, the source diagram, the
resolved morphism / reducer / comparator dicts, and (for the
`:uop_compiled` backend) a pre-lowered `UOpCompiledDiagram`.

The model can be re-lowered against new dummy inputs via `lower!(model,
inputs)` if the input shapes change.
"""
mutable struct FFTinyGradModel
    backend::AbstractFunctorFlowBackend
    diagram::Diagram
    morphisms::Dict{Symbol, Any}
    reducers::Dict{Symbol, Any}
    comparators::Dict{Symbol, Any}
    compiled::Union{Nothing, UOpCompiledDiagram}
end

backend_name(m::FFTinyGradModel) = backend_name(m.backend)

function (m::FFTinyGradModel)(inputs::AbstractDict)
    if m.backend isa UOpCompiledBackend
        if m.compiled === nothing
            # Lazy lowering on first call using these inputs as shape hints.
            m.compiled = lower(m.backend, m.diagram, Dict(), Dict(inputs);
                               morphisms=m.morphisms,
                               reducers=m.reducers,
                               comparators=m.comparators)
        end
        return realize(m.backend, m.compiled, inputs)
    else
        return realize(m.backend, m.diagram, inputs;
                       morphisms=m.morphisms,
                       reducers=m.reducers,
                       comparators=m.comparators)
    end
end

"""
    compile_to_tinygrad(D::Diagram;
                        backend::Symbol=:array_roundtrip,
                        morphisms=nothing, reducers=nothing, comparators=nothing,
                        inputs=nothing) -> FFTinyGradModel

Build a TinyGrad-backed executable model from a FunctorFlow diagram.

# Arguments
- `D::Diagram`           — FunctorFlow diagram to compile.
- `backend::Symbol`      — `:array_roundtrip` (default, `TinyGradBackend`) or
                           `:uop_compiled` (`UOpCompiledBackend`).
- `morphisms`            — optional override Dict of morphism implementations
                           (merged on top of `D.implementations`).
- `reducers`             — optional override Dict of Kan reducers (merged on
                           top of `BUILTIN_REDUCERS`).
- `comparators`          — optional override Dict of comparators (merged on top
                           of `BUILTIN_COMPARATORS`).
- `inputs`               — for `:uop_compiled`, dummy inputs used to seed the
                           tracing pass.  If omitted, the model lowers lazily
                           on the first call.

Returns an [`FFTinyGradModel`](@ref) callable.
"""
function compile_to_tinygrad(D::Diagram;
                             backend::Symbol = :array_roundtrip,
                             morphisms::Union{Nothing, Dict} = nothing,
                             reducers::Union{Nothing, Dict} = nothing,
                             comparators::Union{Nothing, Dict} = nothing,
                             inputs::Union{Nothing, AbstractDict} = nothing)
    morph_dict = merge(D.implementations,
        morphisms === nothing ? Dict{Symbol, Any}() :
            Dict{Symbol, Any}(Symbol(k) => v for (k, v) in morphisms))
    reducer_dict = merge(BUILTIN_REDUCERS, D.reducers,
        reducers === nothing ? Dict{Symbol, Any}() :
            Dict{Symbol, Any}(Symbol(k) => v for (k, v) in reducers))
    comparator_dict = merge(BUILTIN_COMPARATORS, D.comparators,
        comparators === nothing ? Dict{Symbol, Any}() :
            Dict{Symbol, Any}(Symbol(k) => v for (k, v) in comparators))

    backend_obj = if backend === :array_roundtrip
        TinyGradBackend()
    elseif backend === :uop_compiled
        UOpCompiledBackend()
    else
        error("Unknown TinyGrad backend :$(backend) — expected :array_roundtrip or :uop_compiled")
    end

    compiled = nothing
    if backend_obj isa UOpCompiledBackend && inputs !== nothing
        compiled = lower(backend_obj, D, Dict(), Dict(inputs);
                         morphisms=morph_dict,
                         reducers=reducer_dict,
                         comparators=comparator_dict)
    end

    return FFTinyGradModel(backend_obj, D, morph_dict, reducer_dict,
                           comparator_dict, compiled)
end

# ── Display ──────────────────────────────────────────────────────────

function Base.show(io::IO, c::UOpCompiledDiagram)
    s = c.stats
    pct = s.n_nodes_before > 0 ?
        round(100 * (1 - s.n_nodes_after / s.n_nodes_before); digits=1) : 0.0
    print(io, "UOpCompiledDiagram(:$(c.name), ",
          "$(s.n_traced) traced + $(s.n_opaque) opaque, ",
          "$(s.n_nodes_before)→$(s.n_nodes_after) nodes ($(pct)% reduction)",
          c.fully_traced ? ", fully fused" : "",
          ")")
end

function Base.show(io::IO, m::FFTinyGradModel)
    print(io, "FFTinyGradModel(diagram=:$(m.diagram.name), backend=$(backend_name(m.backend))")
    if m.compiled !== nothing
        print(io, ", compiled=", m.compiled)
    end
    print(io, ")")
end

# ── Exports ──────────────────────────────────────────────────────────

export TinyGradBackend, UOpCompiledBackend, UOpCompiledDiagram, FFTinyGradModel
export compile_to_tinygrad, to_tiny_tensor, from_tiny_tensor

end # module
