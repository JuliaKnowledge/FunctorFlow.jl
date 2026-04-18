# ============================================================================
# FunctorFlowLuxTrainExt — high-level training helpers for the Lux backend
#
# This extension is **layered on top of** FunctorFlowLuxExt: it pulls in
# Optimisers + Zygote and adds `train_diagram!`, a CatNet.jl-compatible
# wrapper around `LuxDiagramModel`. Loaded automatically when all four of
# Lux, LuxCore, Optimisers, and Zygote are present in the active project.
#
# Splitting this out keeps FunctorFlowLuxExt itself dep-light: users who
# only want `compile_to_lux` and to drive their own training loop don't
# need Optimisers / Zygote in their environment.
# ============================================================================

module FunctorFlowLuxTrainExt

using FunctorFlow
using Lux
using LuxCore
using Optimisers
using Zygote
using ChainRulesCore

# Reach into the sister extension to grab the LuxDiagramModel type. The
# extension is guaranteed to be loaded at this point because
# FunctorFlowLuxTrainExt's trigger set is a superset of FunctorFlowLuxExt's.
const _LuxExt = let
    ext = Base.get_extension(FunctorFlow, :FunctorFlowLuxExt)
    ext === nothing && error("FunctorFlowLuxExt must be loaded before FunctorFlowLuxTrainExt")
    ext
end

const LuxDiagramModel = _LuxExt.LuxDiagramModel

import FunctorFlow: train_diagram!

const _BUILTIN_LOSSES = Dict{Symbol, Function}(
    :mse => (pred, tgt) -> begin
        diff = pred .- tgt
        sum(diff .* diff) / length(diff)
    end,
    :mae => (pred, tgt) -> sum(abs.(pred .- tgt)) / length(pred),
)

function _resolve_loss_fn(loss_fn)
    loss_fn isa Function && return loss_fn
    loss_fn isa Symbol || error("loss_fn must be a Symbol or Function, got $(typeof(loss_fn))")
    fn = get(_BUILTIN_LOSSES, loss_fn, nothing)
    fn === nothing && error("unknown loss_fn :$loss_fn (available: $(collect(keys(_BUILTIN_LOSSES))))")
    return fn
end

function _resolve_output_keys(output_keys, targets)
    output_keys === nothing && return Tuple(collect(keys(targets)))
    return Tuple(Symbol.(output_keys))
end

function _data_loss_dict(values_dict::AbstractDict, targets, keys_to_use, loss_fn_resolved)
    isempty(keys_to_use) && return 0.0
    total = 0.0
    for k in keys_to_use
        haskey(values_dict, k) || error("output key :$k not produced by diagram")
        haskey(targets, k) || error("target :$k missing from batch")
        total += loss_fn_resolved(values_dict[k], targets[k])
    end
    return total / length(keys_to_use)
end

function _obstruction_total_dict(losses)
    total = 0.0
    for v in values(losses)
        total += v
    end
    return total
end

"""
    train_diagram!(model::LuxDiagramModel, ps, st, data_loader;
                   optimizer = Optimisers.Adam(1e-3),
                   n_epochs::Integer = 1,
                   loss_fn = :mse,
                   obstruction_weight::Real = 1.0,
                   on_step = nothing,
                   output_keys = nothing)
        -> (ps, st, history::Vector{NamedTuple})

Train a `LuxDiagramModel` end-to-end with Zygote + Optimisers. This is a
direct mirror of CatNet.jl's `train_diagram!` — same kwargs, same return
shape — so code can move between the two packages without learning a new
training API.

`data_loader` is any iterable whose elements are `(inputs::Dict,
targets::Dict)` tuples. Keys of `targets` (or those in `output_keys`)
must be entries in `result[:values]` produced by the diagram.

# Keyword arguments
- `optimizer`: Optimisers.jl rule. Defaults to `Optimisers.Adam(1e-3)`.
- `n_epochs`: number of full passes over `data_loader`.
- `loss_fn`: `:mse`, `:mae`, or a function `(pred, target) -> scalar`.
- `obstruction_weight`: multiplier on the sum of obstruction-loss values.
- `on_step`: optional callback `(step, total_loss, ps) -> Any`.
- `output_keys`: env keys compared to targets (default: `keys(targets)`).

Returns `(ps, st, history)` where `history` is a `Vector{NamedTuple}` of
`(step, epoch, data_loss, obstruction_loss, total_loss)` per minibatch.

# Notes vs. CatNet.jl

CatNet ships `:crossentropy` as a built-in loss; FunctorFlow does not,
because `LuxDiagramModel` does not normalise its outputs in a single
canonical way. Pass a function for cross-entropy or any other loss.

Mixed-precision training, distributed training, learning-rate schedules,
and gradient accumulation are out of scope of this shim — drop down to
the underlying `Optimisers` / `Zygote` loop or use a higher-level
training framework.
"""
function train_diagram!(model::LuxDiagramModel, ps, st, data_loader;
                        optimizer = Optimisers.Adam(1e-3),
                        n_epochs::Integer = 1,
                        loss_fn = :mse,
                        obstruction_weight::Real = 1.0,
                        on_step = nothing,
                        output_keys = nothing)
    loss_fn_resolved = _resolve_loss_fn(loss_fn)
    opt_state = Optimisers.setup(optimizer, ps)
    history = NamedTuple[]
    step = 0
    st_ref = st

    for epoch in 1:n_epochs
        for (inputs, targets) in data_loader
            step += 1
            keys_to_use = _resolve_output_keys(output_keys, targets)
            local data_loss_val = 0.0
            local obs_loss_val = 0.0

            function loss_closure(p)
                (result, new_st) = model(inputs, p, st_ref)
                values_dict = result[:values]::AbstractDict
                losses_dict = result[:losses]::AbstractDict
                dl = _data_loss_dict(values_dict, targets, keys_to_use, loss_fn_resolved)
                ol = _obstruction_total_dict(losses_dict)
                ChainRulesCore.@ignore_derivatives begin
                    data_loss_val = dl
                    obs_loss_val = ol
                    st_ref = new_st
                end
                return dl + obstruction_weight * ol
            end

            total_loss, back = Zygote.pullback(loss_closure, ps)
            grads = back(one(total_loss))[1]
            opt_state, ps = Optimisers.update!(opt_state, ps, grads)

            push!(history, (step=step, epoch=epoch,
                            data_loss=float(data_loss_val),
                            obstruction_loss=float(obs_loss_val),
                            total_loss=float(total_loss)))
            on_step === nothing || on_step(step, total_loss, ps)
        end
    end

    return ps, st_ref, history
end

end # module FunctorFlowLuxTrainExt
