# ============================================================================
# show.jl — Pretty-printing for FunctorFlow types (unicode-aware)
# ============================================================================

function Base.show(io::IO, obj::FFObject)
    print(io, "$(obj.name)::$(obj.kind)")
    obj.shape !== nothing && print(io, " [$(obj.shape)]")
end

function Base.show(io::IO, m::Morphism)
    print(io, "$(m.name): $(m.source) → $(m.target)")
end

function Base.show(io::IO, c::Composition)
    chain_str = join([string(s) for s in c.chain], " ⋅ ")
    print(io, "$(c.name) = $chain_str")
end

function Base.show(io::IO, k::KanExtension)
    op = k.direction == LEFT ? "Σ" : "Δ"
    print(io, "$(k.name) = $(op)($(k.source), along=$(k.along)")
    k.target !== nothing && print(io, ", target=$(k.target)")
    print(io, ", reducer=:$(k.reducer))")
end

function Base.show(io::IO, l::ObstructionLoss)
    paths_str = join(["‖$(a), $(b)‖" for (a, b) in l.paths], " + ")
    print(io, "$(l.name) = $paths_str")
    l.comparator != :l2 && print(io, " [$(l.comparator)]")
    l.weight != 1.0 && print(io, " ×$(l.weight)")
end

function Base.show(io::IO, p::Port)
    dir = p.direction == INPUT ? "→" : p.direction == OUTPUT ? "←" : "↔"
    print(io, "$(dir) $(p.name) ($(p.port_type))")
end

function Base.show(io::IO, a::Adapter)
    print(io, "Adapter(:$(a.name), $(a.source_type) → $(a.target_type))")
end

function Base.show(io::IO, inc::IncludedDiagram)
    print(io, "IncludedDiagram(:$(inc.namespace), from=:$(inc.diagram_name), ",
          "objects=$(length(inc.object_map)), ops=$(length(inc.operation_map)), ",
          "ports=$(length(inc.port_specs)))")
end

function Base.show(io::IO, D::Diagram)
    n_kans = count(op -> op isa KanExtension, values(D.operations))
    n_morphisms = count(op -> op isa Morphism, values(D.operations))
    print(io, "Diagram :$(D.name)")
    print(io, " ⟨$(length(D.objects)) objects, $(n_morphisms) morphisms, $(n_kans) Kan, $(length(D.losses)) losses⟩")
end

function Base.show(io::IO, ::MIME"text/plain", D::Diagram)
    println(io, "Diagram :$(D.name)")
    if !isempty(D.objects)
        println(io, "  Objects:")
        for (_, obj) in D.objects
            println(io, "    ", obj)
        end
    end
    if !isempty(D.operations)
        println(io, "  Operations:")
        for (_, op) in D.operations
            println(io, "    ", op)
        end
    end
    if !isempty(D.losses)
        println(io, "  Losses:")
        for (_, loss) in D.losses
            println(io, "    ", loss)
        end
    end
    if !isempty(D.ports)
        println(io, "  Ports:")
        for (_, p) in D.ports
            println(io, "    ", p)
        end
    end
end

function Base.show(io::IO, ir::DiagramIR)
    print(io, "DiagramIR :$(ir.name) ⟨$(length(ir.objects)) objects, ",
          "$(length(ir.operations)) operations, $(length(ir.losses)) losses⟩")
end

function Base.show(io::IO, r::ExecutionResult)
    print(io, "ExecutionResult($(length(r.values)) values, $(length(r.losses)) losses)")
end

function Base.show(io::IO, c::CompiledDiagram)
    print(io, "CompiledDiagram :$(c.diagram.name) ⟨",
          "$(length(c.morphisms)) morphisms, ",
          "$(length(c.reducers)) reducers, ",
          "$(length(c.comparators)) comparators⟩")
end
