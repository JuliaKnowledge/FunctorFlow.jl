# ============================================================================
# ir.jl — Intermediate representation and serialization
# ============================================================================

using JSON3

"""
    DiagramIR(name, objects, operations, losses, ports)

The normalized intermediate representation of a FunctorFlow diagram.
Operations are ordered (execution order preserved). Serializable to JSON.
"""
struct DiagramIR
    name::Symbol
    objects::Vector{FFObject}
    operations::Vector{Dict{Symbol, Any}}
    losses::Vector{ObstructionLoss}
    ports::Vector{Port}
end

"""Convert an operation to a serializable dictionary."""
function _operation_dict(op::Morphism)
    Dict{Symbol, Any}(
        :kind => :morphism,
        :name => op.name,
        :source => op.source,
        :target => op.target,
        :implementation_key => op.implementation_key,
        :description => op.description,
        :metadata => op.metadata
    )
end

function _operation_dict(op::Composition)
    Dict{Symbol, Any}(
        :kind => :composition,
        :name => op.name,
        :chain => op.chain,
        :source => op.source,
        :target => op.target,
        :description => op.description,
        :metadata => op.metadata
    )
end

function _operation_dict(op::KanExtension)
    Dict{Symbol, Any}(
        :kind => :kanextension,
        :name => op.name,
        :direction => op.direction == LEFT ? :left : :right,
        :source => op.source,
        :along => op.along,
        :target => op.target,
        :reducer => op.reducer,
        :description => op.description,
        :metadata => op.metadata
    )
end

"""
    to_ir(D::Diagram) -> DiagramIR

Convert a diagram to its intermediate representation.
"""
function to_ir(D::Diagram)
    ops = [_operation_dict(op) for op in values(D.operations)]
    DiagramIR(D.name,
              collect(values(D.objects)),
              ops,
              collect(values(D.losses)),
              collect(values(D.ports)))
end

"""
    as_dict(ir::DiagramIR) -> Dict

Convert a DiagramIR to a plain dictionary for JSON serialization.
"""
function as_dict(ir::DiagramIR)
    Dict{String, Any}(
        "name" => String(ir.name),
        "objects" => [Dict("name" => String(o.name), "kind" => String(o.kind),
                           "shape" => o.shape, "description" => o.description)
                      for o in ir.objects],
        "operations" => [Dict(String(k) => _serialize_value(v) for (k, v) in op)
                         for op in ir.operations],
        "losses" => [Dict("name" => String(l.name),
                          "paths" => [(String(a), String(b)) for (a, b) in l.paths],
                          "comparator" => String(l.comparator),
                          "weight" => l.weight)
                     for l in ir.losses],
        "ports" => [Dict("name" => String(p.name), "ref" => String(p.ref),
                         "kind" => String(p.kind), "port_type" => String(p.port_type),
                         "direction" => lowercase(string(p.direction)))
                    for p in ir.ports]
    )
end

function _serialize_value(v::Symbol)
    String(v)
end
function _serialize_value(v::Vector{Symbol})
    String.(v)
end
function _serialize_value(v::PortDirection)
    lowercase(string(v))
end
function _serialize_value(v::KanDirection)
    v == LEFT ? "left" : "right"
end
function _serialize_value(v::Vector{Tuple{Symbol, Symbol}})
    [(String(a), String(b)) for (a, b) in v]
end
function _serialize_value(v::Nothing)
    nothing
end
function _serialize_value(v::Dict)
    Dict(String(k) => _serialize_value(val) for (k, val) in v)
end
function _serialize_value(v)
    v
end

"""Serialize a DiagramIR to a JSON string."""
function to_json(ir::DiagramIR)
    JSON3.write(as_dict(ir))
end

"""Serialize a Diagram directly to JSON via its IR."""
function to_json(D::Diagram)
    to_json(to_ir(D))
end
