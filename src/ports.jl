# ============================================================================
# ports.jl — Port system and type checking
# ============================================================================

"""
    expose_port!(D, name, ref; kind=nothing, port_type=nothing, direction=INTERNAL, ...)

Expose a semantic interface on the diagram. Ports survive composition and
enable typed wiring of sub-diagrams through stable contracts.

`kind` and `port_type` are inferred from the referenced element if not given.
"""
function expose_port!(D::Diagram, name::Union{Symbol, AbstractString},
                      ref::Union{Symbol, AbstractString, AbstractFFElement};
                      kind::Union{Nothing, Symbol, AbstractString}=nothing,
                      port_type::Union{Nothing, Symbol, AbstractString}=nothing,
                      direction::PortDirection=INTERNAL,
                      description::AbstractString="",
                      metadata::Dict=Dict{Symbol, Any}())
    ref_sym = _ref(ref)
    inferred_kind = _infer_ref_kind(D, ref_sym)
    inferred_port_type = _infer_ref_port_type(D, ref_sym)
    actual_kind = kind === nothing ? inferred_kind : Symbol(kind)
    actual_port_type = port_type === nothing ? inferred_port_type : Symbol(port_type)
    p = Port(name, ref_sym;
             kind=actual_kind, port_type=actual_port_type,
             direction=direction, description=description, metadata=metadata)
    D.ports[p.name] = p
    p
end

"""Look up a port by name."""
function get_port(D::Diagram, name::Union{Symbol, AbstractString})
    get(D.ports, Symbol(name)) do
        error("Port :$(name) not found in diagram :$(D.name)")
    end
end

function _infer_ref_kind(D::Diagram, name::Symbol)
    haskey(D.objects, name) && return :object
    haskey(D.operations, name) && return :operation
    haskey(D.losses, name) && return :loss
    :object
end

function _infer_ref_port_type(D::Diagram, name::Symbol)
    if haskey(D.objects, name)
        return D.objects[name].kind
    elseif haskey(D.operations, name)
        op = D.operations[name]
        if op isa Morphism
            return haskey(D.objects, op.target) ? D.objects[op.target].kind : :any
        elseif op isa KanExtension
            tgt = op.target
            return tgt !== nothing && haskey(D.objects, tgt) ? D.objects[tgt].kind : :any
        end
    elseif haskey(D.losses, name)
        return :loss
    end
    :any
end

function _port_types_compatible(actual::Symbol, expected::Symbol)
    actual == expected || actual == :any || expected == :any
end
