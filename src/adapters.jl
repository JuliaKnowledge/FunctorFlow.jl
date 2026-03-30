# ============================================================================
# adapters.jl — Adapter registration, coercion, and libraries
# ============================================================================

"""
    AdapterSpec(name, source_type, target_type, implementation; description="")

Specification for an adapter in a library.
"""
struct AdapterSpec
    name::Symbol
    source_type::Symbol
    target_type::Symbol
    implementation::Any
    description::String
end

function AdapterSpec(name, source_type, target_type, implementation=identity;
                     description::AbstractString="")
    AdapterSpec(Symbol(name), Symbol(source_type), Symbol(target_type),
                implementation, String(description))
end

"""
    AdapterLibrary(name, adapters; description="")

A named collection of adapter specifications.
"""
struct AdapterLibrary
    name::Symbol
    adapters::Vector{AdapterSpec}
    description::String
end

function AdapterLibrary(name, adapters; description::AbstractString="")
    AdapterLibrary(Symbol(name), adapters, String(description))
end

"""Standard adapter library shipping with FunctorFlow."""
const STANDARD_ADAPTER_LIBRARY = AdapterLibrary(:standard, [
    AdapterSpec(:context_to_candidates, :contextualized_messages, :plan_candidates, identity;
                description="Treat contextualized message collections as candidate plan pools."),
    AdapterSpec(:plan_candidates_to_plan, :plan_candidates, :plan, identity;
                description="Collapse a selected candidate pool into a concrete plan representation."),
    AdapterSpec(:string_plan_to_plan_steps, :plan, :plan_steps,
                x -> x isa AbstractString ? split(strip(x), r"\n+") : x;
                description="Convert string-serialized plans into tokenized step lists."),
]; description="Standard adapter library for FunctorFlow.")

const ADAPTER_LIBRARIES = Dict{Symbol, AdapterLibrary}(
    :standard => STANDARD_ADAPTER_LIBRARY,
)

"""
    get_adapter_library(name) -> AdapterLibrary

Look up a registered adapter library by name.
"""
function get_adapter_library(name::Union{Symbol, AbstractString})
    get(ADAPTER_LIBRARIES, Symbol(name)) do
        error("Adapter library :$(name) not found")
    end
end

"""
    register_adapter!(D, name; source_type, target_type, implementation=identity, ...)

Register an adapter on a diagram for bridging between two port types.
"""
function register_adapter!(D::Diagram, name::Union{Symbol, AbstractString};
                           source_type::Union{Symbol, AbstractString},
                           target_type::Union{Symbol, AbstractString},
                           implementation=identity,
                           description::AbstractString="",
                           metadata::Dict=Dict{Symbol, Any}())
    adapter = Adapter(name, source_type, target_type;
                      description=description, metadata=metadata)
    D.adapters[(adapter.source_type, adapter.target_type)] = adapter
    D.adapter_implementations[adapter.name] = implementation
    adapter
end

"""
    use_adapter_library!(D, name_or_library)

Install all adapters from a library into the diagram.
"""
function use_adapter_library!(D::Diagram, name::Union{Symbol, AbstractString})
    lib = get_adapter_library(name)
    use_adapter_library!(D, lib)
end

function use_adapter_library!(D::Diagram, lib::AdapterLibrary)
    for spec in lib.adapters
        register_adapter!(D, spec.name;
                          source_type=spec.source_type,
                          target_type=spec.target_type,
                          implementation=spec.implementation,
                          description=spec.description)
    end
    nothing
end

"""
    coerce!(D, ref; to_type, from_type=nothing, name=nothing, ...)

Create a coercion morphism that adapts `ref` from one port type to another.
Looks up a registered adapter for the type bridge. Returns the name of the
generated coercion morphism.
"""
function coerce!(D::Diagram, ref::Union{Symbol, AbstractString};
                 to_type::Union{Symbol, AbstractString},
                 from_type::Union{Nothing, Symbol, AbstractString}=nothing,
                 name::Union{Nothing, Symbol, AbstractString}=nothing,
                 description::AbstractString="",
                 metadata::Dict=Dict{Symbol, Any}())
    ref_sym = _ref(ref)
    actual_from = from_type === nothing ? _infer_ref_port_type(D, ref_sym) : Symbol(from_type)
    to_sym = Symbol(to_type)

    adapter_key = (actual_from, to_sym)
    haskey(D.adapters, adapter_key) || error("No adapter registered for $(actual_from) → $(to_sym)")

    adapter = D.adapters[adapter_key]
    impl = get(D.adapter_implementations, adapter.name, identity)

    coerce_name = name === nothing ? Symbol(:_coerce_, ref_sym, :_to_, to_sym) : Symbol(name)
    out_obj_name = Symbol(coerce_name, :_output)
    add_object!(D, out_obj_name; kind=to_sym)
    add_morphism!(D, coerce_name, ref_sym, out_obj_name; implementation=impl,
                  description=description, metadata=metadata)
    expose_port!(D, Symbol(:_coerced_, ref_sym), coerce_name;
                 direction=OUTPUT, port_type=to_sym)
    coerce_name
end
