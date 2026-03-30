# ============================================================================
# types.jl — Core type definitions for FunctorFlow.jl
# ============================================================================

"""Direction of a Kan extension."""
@enum KanDirection LEFT RIGHT

"""Direction of a port relative to its diagram."""
@enum PortDirection INPUT OUTPUT INTERNAL

# ---------------------------------------------------------------------------
# Abstract hierarchy
# ---------------------------------------------------------------------------

"""Supertype for all FunctorFlow elements."""
abstract type AbstractFFElement end

"""Supertype for objects (typed interfaces) in a diagram."""
abstract type AbstractFFObject <: AbstractFFElement end

"""Supertype for operations (morphisms, compositions, Kan extensions)."""
abstract type AbstractFFOperation <: AbstractFFElement end

# ---------------------------------------------------------------------------
# FFObject — typed interface in a diagram
# ---------------------------------------------------------------------------

"""
    FFObject(name; kind=:object, shape=nothing, description="", metadata=Dict())

An object in a FunctorFlow diagram representing a typed semantic interface.
Objects name the kinds of states a diagram manipulates: token collections,
neighborhood indices, local fragments, global latent states, plan states, etc.
"""
struct FFObject <: AbstractFFObject
    name::Symbol
    kind::Symbol
    shape::Union{Nothing, String}
    description::String
    metadata::Dict{Symbol, Any}
end

function FFObject(name::Union{Symbol, AbstractString};
                  kind::Union{Symbol, AbstractString}=:object,
                  shape::Union{Nothing, String}=nothing,
                  description::AbstractString="",
                  metadata::Dict=Dict{Symbol, Any}())
    FFObject(Symbol(name), Symbol(kind), shape, String(description),
             Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

_is_placeholder(obj::FFObject) = obj.kind == :object && isempty(obj.description) && isempty(obj.metadata)

# ---------------------------------------------------------------------------
# Morphism — typed computational arrow
# ---------------------------------------------------------------------------

"""
    Morphism(name, source, target; implementation_key=nothing, description="", metadata=Dict())

A morphism in a FunctorFlow diagram: a typed transformation between objects.
Morphisms can represent neural layers, geometric lifts, projection maps,
repair operators, or workflow transitions.
"""
struct Morphism <: AbstractFFOperation
    name::Symbol
    source::Symbol
    target::Symbol
    implementation_key::Union{Nothing, Symbol}
    description::String
    metadata::Dict{Symbol, Any}
end

function Morphism(name, source, target;
                  implementation_key::Union{Nothing, Symbol, AbstractString}=nothing,
                  description::AbstractString="",
                  metadata::Dict=Dict{Symbol, Any}())
    Morphism(Symbol(name), Symbol(source), Symbol(target),
             implementation_key === nothing ? nothing : Symbol(implementation_key),
             String(description),
             Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

# ---------------------------------------------------------------------------
# Composition — sequential chaining of morphisms
# ---------------------------------------------------------------------------

"""
    Composition(name, chain, source, target; description="", metadata=Dict())

A composition of morphisms applied sequentially. The chain is validated such
that each morphism's target matches the next morphism's source.
"""
struct Composition <: AbstractFFOperation
    name::Symbol
    chain::Vector{Symbol}
    source::Symbol
    target::Symbol
    description::String
    metadata::Dict{Symbol, Any}
end

function Composition(name, chain, source, target;
                     description::AbstractString="",
                     metadata::Dict=Dict{Symbol, Any}())
    Composition(Symbol(name), Symbol.(chain), Symbol(source), Symbol(target),
                String(description),
                Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

# ---------------------------------------------------------------------------
# KanExtension — universal aggregation / completion
# ---------------------------------------------------------------------------

"""
    KanExtension(name, direction, source, along; target=nothing, reducer=:sum, ...)

A Kan extension operation in a diagram.

- **Left Kan** (`LEFT`): universal aggregation / pushforward. Covers attention,
  pooling, neighborhood message passing, context fusion.
- **Right Kan** (`RIGHT`): universal completion / repair. Covers denoising,
  masked completion, plan repair, partial-view reconciliation.
"""
struct KanExtension <: AbstractFFOperation
    name::Symbol
    direction::KanDirection
    source::Symbol
    along::Symbol
    target::Union{Nothing, Symbol}
    reducer::Symbol
    description::String
    metadata::Dict{Symbol, Any}
end

function KanExtension(name, direction::KanDirection, source, along;
                      target::Union{Nothing, Symbol, AbstractString}=nothing,
                      reducer::Union{Symbol, AbstractString}=:sum,
                      description::AbstractString="",
                      metadata::Dict=Dict{Symbol, Any}())
    KanExtension(Symbol(name), direction, Symbol(source), Symbol(along),
                 target === nothing ? nothing : Symbol(target),
                 Symbol(reducer), String(description),
                 Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

# ---------------------------------------------------------------------------
# ObstructionLoss — measures non-commutativity of diagram paths
# ---------------------------------------------------------------------------

"""
    ObstructionLoss(name, paths; comparator=:l2, weight=1.0, ...)

An obstruction loss measuring the degree to which named paths in a diagram
fail to commute. This gives Diagrammatic Backpropagation (DB) a native home
in the FunctorFlow language.
"""
struct ObstructionLoss <: AbstractFFElement
    name::Symbol
    paths::Vector{Tuple{Symbol, Symbol}}
    comparator::Symbol
    weight::Float64
    description::String
    metadata::Dict{Symbol, Any}
end

function ObstructionLoss(name, paths;
                         comparator::Union{Symbol, AbstractString}=:l2,
                         weight::Real=1.0,
                         description::AbstractString="",
                         metadata::Dict=Dict{Symbol, Any}())
    ObstructionLoss(Symbol(name),
                    [(Symbol(a), Symbol(b)) for (a, b) in paths],
                    Symbol(comparator), Float64(weight),
                    String(description),
                    Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

# ---------------------------------------------------------------------------
# Port — semantic interface of a diagram
# ---------------------------------------------------------------------------

"""
    Port(name, ref; kind=:object, port_type=:any, direction=INTERNAL, ...)

A port exposes a semantic interface on a diagram that survives composition.
Ports enable typed wiring of sub-diagrams through stable contracts rather
than raw internal operation names.
"""
struct Port <: AbstractFFElement
    name::Symbol
    ref::Symbol
    kind::Symbol
    port_type::Symbol
    direction::PortDirection
    description::String
    metadata::Dict{Symbol, Any}
end

function Port(name, ref;
              kind::Union{Symbol, AbstractString}=:object,
              port_type::Union{Symbol, AbstractString}=:any,
              direction::PortDirection=INTERNAL,
              description::AbstractString="",
              metadata::Dict=Dict{Symbol, Any}())
    Port(Symbol(name), Symbol(ref), Symbol(kind), Symbol(port_type),
         direction, String(description),
         Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

# ---------------------------------------------------------------------------
# Adapter — type coercion between port types
# ---------------------------------------------------------------------------

"""
    Adapter(name, source_type, target_type; description="", metadata=Dict())

An adapter declares a legal bridge between two port types. When two blocks
need connection but have incompatible port types, an adapter provides a
principled coercion rather than silently weakening the type system.
"""
struct Adapter <: AbstractFFElement
    name::Symbol
    source_type::Symbol
    target_type::Symbol
    description::String
    metadata::Dict{Symbol, Any}
end

function Adapter(name, source_type, target_type;
                 description::AbstractString="",
                 metadata::Dict=Dict{Symbol, Any}())
    Adapter(Symbol(name), Symbol(source_type), Symbol(target_type),
            String(description),
            Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

# ---------------------------------------------------------------------------
# IncludedDiagram — result of including a sub-diagram
# ---------------------------------------------------------------------------

"""
    IncludedDiagram

Result of including a sub-diagram via `include!()`. Provides namespace-aware
lookup of objects, operations, and ports from the included diagram.
"""
struct IncludedDiagram
    namespace::Symbol
    diagram_name::Symbol
    object_map::Dict{Symbol, Symbol}
    operation_map::Dict{Symbol, Symbol}
    loss_map::Dict{Symbol, Symbol}
    port_specs::Dict{Symbol, Port}
end

"""Get the namespaced name of an object from an included diagram."""
function object_ref(inc::IncludedDiagram, name::Union{Symbol, AbstractString})
    get(inc.object_map, Symbol(name)) do
        error("Object :$(name) not found in included diagram :$(inc.diagram_name)")
    end
end

"""Get the namespaced name of an operation from an included diagram."""
function operation_ref(inc::IncludedDiagram, name::Union{Symbol, AbstractString})
    get(inc.operation_map, Symbol(name)) do
        error("Operation :$(name) not found in included diagram :$(inc.diagram_name)")
    end
end

"""Get the port spec from an included diagram."""
function port_spec(inc::IncludedDiagram, name::Union{Symbol, AbstractString})
    get(inc.port_specs, Symbol(name)) do
        error("Port :$(name) not found in included diagram :$(inc.diagram_name)")
    end
end
