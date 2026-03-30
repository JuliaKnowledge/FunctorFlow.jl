# ============================================================================
# composition.jl — Diagram inclusion, namespacing, and wiring
# ============================================================================

"""
    include!(parent, child; namespace, object_aliases=nothing)

Include a child diagram into a parent diagram under the given namespace.
All objects and operations from the child are prefixed with `namespace__`.
Object aliases allow wiring external objects into sub-diagram slots.

Returns an `IncludedDiagram` for accessing namespaced references.
"""
function include!(parent::Diagram, child::Diagram;
                  namespace::Union{Symbol, AbstractString},
                  object_aliases::Union{Nothing, Dict}=nothing)
    ns = Symbol(namespace)
    prefix = Symbol(ns, :__)

    obj_map = Dict{Symbol, Symbol}()
    op_map = Dict{Symbol, Symbol}()
    loss_map = Dict{Symbol, Symbol}()

    aliases = object_aliases === nothing ? Dict{Symbol, Symbol}() :
        Dict{Symbol, Symbol}(Symbol(k) => Symbol(v) for (k, v) in object_aliases)

    # Copy objects (with aliasing)
    for (name, obj) in child.objects
        if haskey(aliases, name)
            obj_map[name] = aliases[name]
        else
            ns_name = Symbol(prefix, name)
            obj_map[name] = ns_name
            parent.objects[ns_name] = FFObject(ns_name;
                kind=obj.kind, shape=obj.shape,
                description=obj.description,
                metadata=merge(obj.metadata, Dict{Symbol, Any}(:_namespace => ns, :_original => name)))
        end
    end

    # Helper to remap a symbol through the object map
    _remap(s::Symbol) = get(obj_map, s, Symbol(prefix, s))
    _remap(::Nothing) = nothing

    # Copy operations
    for (name, op) in child.operations
        ns_name = Symbol(prefix, name)
        op_map[name] = ns_name
        if op isa Morphism
            parent.operations[ns_name] = Morphism(ns_name, _remap(op.source), _remap(op.target);
                implementation_key=op.implementation_key,
                description=op.description,
                metadata=merge(op.metadata, Dict{Symbol, Any}(:_namespace => ns)))
        elseif op isa Composition
            parent.operations[ns_name] = Composition(ns_name,
                [Symbol(prefix, c) for c in op.chain],
                _remap(op.source), _remap(op.target);
                description=op.description,
                metadata=merge(op.metadata, Dict{Symbol, Any}(:_namespace => ns)))
        elseif op isa KanExtension
            parent.operations[ns_name] = KanExtension(ns_name, op.direction,
                _remap(op.source), _remap(op.along);
                target=_remap(op.target),
                reducer=op.reducer,
                description=op.description,
                metadata=merge(op.metadata, Dict{Symbol, Any}(:_namespace => ns)))
        end
    end

    # Copy losses
    for (name, loss) in child.losses
        ns_name = Symbol(prefix, name)
        loss_map[name] = ns_name
        ns_paths = [(Symbol(prefix, a), Symbol(prefix, b)) for (a, b) in loss.paths]
        parent.losses[ns_name] = ObstructionLoss(ns_name, ns_paths;
            comparator=loss.comparator, weight=loss.weight,
            description=loss.description,
            metadata=merge(loss.metadata, Dict{Symbol, Any}(:_namespace => ns)))
    end

    # Copy implementations
    for (name, impl) in child.implementations
        parent.implementations[Symbol(prefix, name)] = impl
    end
    for (name, impl) in child.reducers
        parent.reducers[name] = impl
    end
    for (name, impl) in child.comparators
        parent.comparators[name] = impl
    end

    # Build port specs (remap refs)
    port_specs = Dict{Symbol, Port}()
    for (name, port) in child.ports
        ns_ref = get(op_map, port.ref, get(obj_map, port.ref, Symbol(prefix, port.ref)))
        port_specs[name] = Port(name, ns_ref;
            kind=port.kind, port_type=port.port_type,
            direction=port.direction,
            description=port.description, metadata=port.metadata)
    end

    # Check port type compatibility with aliases and auto-insert adapters
    for (name, port) in child.ports
        if port.direction == INPUT && haskey(aliases, port.ref)
            alias_target = aliases[port.ref]
            if haskey(parent.objects, alias_target)
                actual_type = parent.objects[alias_target].kind
                if !_port_types_compatible(actual_type, port.port_type)
                    adapter_key = (actual_type, port.port_type)
                    if haskey(parent.adapters, adapter_key)
                        adapter = parent.adapters[adapter_key]
                        adapt_name = Symbol(prefix, :_adapt_, name)
                        adapted_obj_name = Symbol(prefix, :_adapted_, name)
                        parent.objects[adapted_obj_name] = FFObject(adapted_obj_name; kind=port.port_type)
                        impl = get(parent.adapter_implementations, adapter.name, identity)
                        parent.operations[adapt_name] = Morphism(adapt_name, alias_target, adapted_obj_name)
                        parent.implementations[adapt_name] = impl
                        obj_map[port.ref] = adapted_obj_name
                    end
                end
            end
        end
    end

    IncludedDiagram(ns, child.name, obj_map, op_map, loss_map, port_specs)
end
