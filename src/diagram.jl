# ============================================================================
# diagram.jl — Diagram construction and manipulation
# ============================================================================

using OrderedCollections: OrderedDict

"""
    Diagram(name)

A FunctorFlow diagram: the primary user-facing artifact. Diagrams declare
objects, morphisms, composed paths, Kan operators, and structural losses.
This is the level at which users design architectures.
"""
mutable struct Diagram
    name::Symbol
    objects::OrderedDict{Symbol, FFObject}
    operations::OrderedDict{Symbol, AbstractFFOperation}
    losses::OrderedDict{Symbol, ObstructionLoss}
    ports::OrderedDict{Symbol, Port}
    implementations::Dict{Symbol, Any}
    reducers::Dict{Symbol, Any}
    comparators::Dict{Symbol, Any}
    adapters::Dict{Tuple{Symbol, Symbol}, Adapter}
    adapter_implementations::Dict{Symbol, Any}
end

function Diagram(name::Union{Symbol, AbstractString})
    Diagram(Symbol(name),
            OrderedDict{Symbol, FFObject}(),
            OrderedDict{Symbol, AbstractFFOperation}(),
            OrderedDict{Symbol, ObstructionLoss}(),
            OrderedDict{Symbol, Port}(),
            Dict{Symbol, Any}(),
            Dict{Symbol, Any}(),
            Dict{Symbol, Any}(),
            Dict{Tuple{Symbol, Symbol}, Adapter}(),
            Dict{Symbol, Any}())
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

function _ref(x::Union{Symbol, AbstractString})
    Symbol(x)
end
function _ref(x::FFObject)
    x.name
end
function _ref(x::AbstractFFOperation)
    x.name
end

function _ensure_object!(D::Diagram, name::Symbol)
    if !haskey(D.objects, name)
        D.objects[name] = FFObject(name)
    end
    D.objects[name]
end

function _require_morphism(D::Diagram, name::Symbol)
    op = get(D.operations, name, nothing)
    op isa Morphism || error("Operation :$name is not a Morphism")
    op
end

function _infer_chain_endpoints(D::Diagram, chain::Vector{Symbol})
    isempty(chain) && error("Composition chain must not be empty")
    # Validate chain compatibility
    for i in 1:(length(chain) - 1)
        left = D.operations[chain[i]]
        right = D.operations[chain[i + 1]]
        left_target = left isa Morphism ? left.target : left isa Composition ? left.target : error("Chain element :$(chain[i]) is not a morphism or composition")
        right_source = right isa Morphism ? right.source : right isa Composition ? right.source : error("Chain element :$(chain[i+1]) is not a morphism or composition")
        left_target == right_source || error("Chain incompatible: :$(chain[i]) target :$left_target != :$(chain[i+1]) source :$right_source")
    end
    first_op = D.operations[chain[1]]
    last_op = D.operations[chain[end]]
    source = first_op isa Morphism ? first_op.source : first_op.source
    target = last_op isa Morphism ? last_op.target : last_op.target
    (source, target)
end

# ---------------------------------------------------------------------------
# Public API — add elements to a diagram
# ---------------------------------------------------------------------------

"""
    add_object!(D, name; kind=:object, shape=nothing, description="", metadata=Dict())

Add an object to the diagram. If a placeholder with the same name exists, it
is replaced with the fully specified object.
"""
function add_object!(D::Diagram, name::Union{Symbol, AbstractString};
                     kind::Union{Symbol, AbstractString}=:object,
                     shape::Union{Nothing, String}=nothing,
                     description::AbstractString="",
                     metadata::Dict=Dict{Symbol, Any}())
    obj = FFObject(name; kind, shape, description, metadata)
    D.objects[obj.name] = obj
    obj
end

"""
    add_morphism!(D, name, source, target; implementation=nothing, ...)

Add a morphism (typed arrow) between two objects. Source and target objects
are auto-created as placeholders if they don't yet exist.
"""
function add_morphism!(D::Diagram, name, source, target;
                       implementation::Union{Nothing, Any}=nothing,
                       implementation_key::Union{Nothing, Symbol, AbstractString}=nothing,
                       description::AbstractString="",
                       metadata::Dict=Dict{Symbol, Any}())
    src = _ref(source)
    tgt = _ref(target)
    _ensure_object!(D, src)
    _ensure_object!(D, tgt)
    m = Morphism(name, src, tgt;
                 implementation_key=implementation_key,
                 description=description, metadata=metadata)
    D.operations[m.name] = m
    if implementation !== nothing
        bind_morphism!(D, m.name, implementation)
    end
    m
end

"""
    compose!(D, chain...; name, description="", metadata=Dict())

Create a composition of morphisms. Validates that the chain is compatible
(each morphism's target matches the next's source) and infers endpoints.
"""
function compose!(D::Diagram, chain::Union{Symbol, AbstractString, Morphism, Composition}...;
                  name::Union{Symbol, AbstractString},
                  description::AbstractString="",
                  metadata::Dict=Dict{Symbol, Any}())
    chain_syms = Symbol[_ref(c) for c in chain]
    for s in chain_syms
        haskey(D.operations, s) || error("Operation :$s not found in diagram :$(D.name)")
    end
    source, target = _infer_chain_endpoints(D, chain_syms)
    comp = Composition(name, chain_syms, source, target;
                       description=description, metadata=metadata)
    D.operations[comp.name] = comp
    comp
end

"""
    add_left_kan!(D, name; source, along, target=nothing, reducer=:sum, ...)

Add a left Kan extension (universal aggregation). Covers attention, pooling,
neighborhood message passing, context fusion, plan-fragment integration.
"""
function add_left_kan!(D::Diagram, name::Union{Symbol, AbstractString};
                       source::Union{Symbol, AbstractString},
                       along::Union{Symbol, AbstractString},
                       target::Union{Nothing, Symbol, AbstractString}=nothing,
                       reducer::Union{Symbol, AbstractString}=:sum,
                       description::AbstractString="",
                       metadata::Dict=Dict{Symbol, Any}())
    src = Symbol(source)
    alg = Symbol(along)
    _ensure_object!(D, src)
    _ensure_object!(D, alg)
    if target !== nothing
        _ensure_object!(D, Symbol(target))
    end
    kan = KanExtension(name, LEFT, src, alg;
                       target=target, reducer=reducer,
                       description=description, metadata=metadata)
    D.operations[kan.name] = kan
    kan
end

"""
    add_right_kan!(D, name; source, along, target=nothing, reducer=:first_non_null, ...)

Add a right Kan extension (universal completion / repair). Covers denoising,
masked completion, plan repair, partial-view reconciliation.
"""
function add_right_kan!(D::Diagram, name::Union{Symbol, AbstractString};
                        source::Union{Symbol, AbstractString},
                        along::Union{Symbol, AbstractString},
                        target::Union{Nothing, Symbol, AbstractString}=nothing,
                        reducer::Union{Symbol, AbstractString}=:first_non_null,
                        description::AbstractString="",
                        metadata::Dict=Dict{Symbol, Any}())
    src = Symbol(source)
    alg = Symbol(along)
    _ensure_object!(D, src)
    _ensure_object!(D, alg)
    if target !== nothing
        _ensure_object!(D, Symbol(target))
    end
    kan = KanExtension(name, RIGHT, src, alg;
                       target=target, reducer=reducer,
                       description=description, metadata=metadata)
    D.operations[kan.name] = kan
    kan
end

"""
    add_obstruction_loss!(D, name; paths, comparator=:l2, weight=1.0, ...)

Add an obstruction loss measuring non-commutativity between diagram paths.
This is the native home for Diagrammatic Backpropagation (DB).
"""
function add_obstruction_loss!(D::Diagram, name::Union{Symbol, AbstractString};
                               paths::Vector,
                               comparator::Union{Symbol, AbstractString}=:l2,
                               weight::Real=1.0,
                               description::AbstractString="",
                               metadata::Dict=Dict{Symbol, Any}())
    loss = ObstructionLoss(name, paths;
                           comparator=comparator, weight=weight,
                           description=description, metadata=metadata)
    D.losses[loss.name] = loss
    loss
end

"""
    add!(D, element)

Add a pre-constructed element (FFObject, Morphism, etc.) to the diagram.
"""
function add!(D::Diagram, obj::FFObject)
    D.objects[obj.name] = obj
    obj
end

function add!(D::Diagram, m::Morphism)
    _ensure_object!(D, m.source)
    _ensure_object!(D, m.target)
    D.operations[m.name] = m
    m
end

function add!(D::Diagram, k::KanExtension)
    _ensure_object!(D, k.source)
    _ensure_object!(D, k.along)
    if k.target !== nothing
        _ensure_object!(D, k.target)
    end
    D.operations[k.name] = k
    k
end

function add!(D::Diagram, c::Composition)
    D.operations[c.name] = c
    c
end

function add!(D::Diagram, loss::ObstructionLoss)
    D.losses[loss.name] = loss
    loss
end

# ---------------------------------------------------------------------------
# Binding implementations
# ---------------------------------------------------------------------------

"""Bind a callable implementation to a morphism."""
function bind_morphism!(D::Diagram, name::Union{Symbol, AbstractString}, impl)
    D.implementations[Symbol(name)] = impl
end

"""Bind a callable reducer for Kan extensions."""
function bind_reducer!(D::Diagram, name::Union{Symbol, AbstractString}, impl)
    D.reducers[Symbol(name)] = impl
end

"""Bind a callable comparator for obstruction losses."""
function bind_comparator!(D::Diagram, name::Union{Symbol, AbstractString}, impl)
    D.comparators[Symbol(name)] = impl
end

"""Return the current summary of the diagram as a string."""
function summary(D::Diagram)
    io = IOBuffer()
    println(io, "Diagram: $(D.name)")
    println(io, "  Objects ($(length(D.objects))): ", join(keys(D.objects), ", "))
    println(io, "  Operations ($(length(D.operations))): ", join(keys(D.operations), ", "))
    println(io, "  Losses ($(length(D.losses))): ", join(keys(D.losses), ", "))
    println(io, "  Ports ($(length(D.ports))): ", join(keys(D.ports), ", "))
    String(take!(io))
end
