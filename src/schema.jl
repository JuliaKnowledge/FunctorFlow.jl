# ============================================================================
# schema.jl — ACSet schema for FunctorFlow diagrams (Catlab integration)
# ============================================================================
#
# Defines the categorical structure of FunctorFlow diagrams using Catlab's
# @present / @acset_type machinery. This gives FunctorFlow diagrams a
# rigorous mathematical foundation and enables interop with the full
# AlgebraicJulia ecosystem.
#
# The schema has three object types:
#   Node — objects in the diagram (typed interfaces)
#   Edge — morphisms / arrows between objects
#   Kan  — Kan extension operations (left or right)
#
# Compositions, losses, and ports live in Julia-level data structures
# alongside the ACSet (they annotate the graph but aren't part of the
# core categorical structure).

using Catlab
using Catlab.Theories: FreeSchema
using Catlab.CategoricalAlgebra

"""
Schema for FunctorFlow diagrams as ACSets.

- `Node`: objects (typed interfaces) in the diagram
- `Edge`: morphisms (typed arrows between objects)
- `Kan`: Kan extension operations (aggregation / completion)

Morphisms encode the graph topology:
- `src`, `tgt`: source and target nodes of each edge
- `kan_src`, `kan_along`: source and relation nodes of each Kan extension

All names and kinds are stored as `Label` attributes (concretely `Symbol`).
"""
@present SchFunctorFlow(FreeSchema) begin
    (Node, Edge, Kan)::Ob

    # Edge topology
    src::Hom(Edge, Node)
    tgt::Hom(Edge, Node)

    # Kan extension topology
    kan_src::Hom(Kan, Node)
    kan_along::Hom(Kan, Node)

    # Single attribute type for all symbolic labels
    Label::AttrType

    # Node attributes
    node_name::Attr(Node, Label)
    node_kind::Attr(Node, Label)

    # Edge attributes
    edge_name::Attr(Edge, Label)
    edge_kind::Attr(Edge, Label)   # :morphism or :composition

    # Kan extension attributes
    kan_name::Attr(Kan, Label)
    kan_dir::Attr(Kan, Label)      # :left or :right
    kan_reducer::Attr(Kan, Label)  # reducer name (:sum, :mean, etc.)
end

"""Abstract supertype for FunctorFlow ACSet types."""
@abstract_acset_type AbstractFunctorFlowGraph

"""
    FunctorFlowGraph{Label}

Concrete ACSet type for FunctorFlow diagrams. Instantiate with `Symbol`:

```julia
g = FunctorFlowGraph{Symbol}()
add_part!(g, :Node; node_name=:Tokens, node_kind=:messages)
```
"""
@acset_type FunctorFlowGraph(SchFunctorFlow,
    index=[:src, :tgt, :kan_src, :kan_along,
           :node_name, :edge_name, :kan_name]) <: AbstractFunctorFlowGraph

# ---------------------------------------------------------------------------
# Conversion: Diagram → ACSet
# ---------------------------------------------------------------------------

"""
    to_acset(D::Diagram) -> FunctorFlowGraph{Symbol}

Convert a FunctorFlow Diagram to its ACSet representation.

Each object becomes a Node, each morphism becomes an Edge, each
Kan extension becomes a Kan part. Compositions are flattened to
single edges with `edge_kind=:composition`.
"""
function to_acset(D)
    acs = FunctorFlowGraph{Symbol}()
    node_idx = Dict{Symbol, Int}()

    # Add objects as nodes
    for (name, obj) in D.objects
        nid = add_part!(acs, :Node; node_name=name, node_kind=obj.kind)
        node_idx[name] = nid
    end

    # Add operations
    for (name, op) in D.operations
        if op isa Morphism
            sid = get(node_idx, op.source, nothing)
            tid = get(node_idx, op.target, nothing)
            (sid === nothing || tid === nothing) && continue
            add_part!(acs, :Edge; src=sid, tgt=tid,
                      edge_name=name, edge_kind=:morphism)
        elseif op isa Composition
            sid = get(node_idx, op.source, nothing)
            tid = get(node_idx, op.target, nothing)
            (sid === nothing || tid === nothing) && continue
            add_part!(acs, :Edge; src=sid, tgt=tid,
                      edge_name=name, edge_kind=:composition)
        elseif op isa KanExtension
            sid = get(node_idx, op.source, nothing)
            aid = get(node_idx, op.along, nothing)
            (sid === nothing || aid === nothing) && continue
            dir = op.direction == LEFT ? :left : :right
            add_part!(acs, :Kan; kan_src=sid, kan_along=aid,
                      kan_name=name, kan_dir=dir, kan_reducer=op.reducer)
        end
    end

    acs
end

# ---------------------------------------------------------------------------
# Conversion: ACSet → Diagram
# ---------------------------------------------------------------------------

"""
    from_acset(acs::FunctorFlowGraph; name=:Imported) -> Diagram

Reconstruct a FunctorFlow Diagram from its ACSet representation.
"""
function from_acset(acs; name::Union{Symbol,AbstractString}=:Imported)
    D = Diagram(Symbol(name))

    # Reconstruct objects
    names = subpart(acs, :node_name)
    kinds = subpart(acs, :node_kind)
    for i in 1:nparts(acs, :Node)
        add_object!(D, names[i]; kind=kinds[i])
    end

    # Reconstruct morphisms/compositions
    for i in 1:nparts(acs, :Edge)
        ename = subpart(acs, i, :edge_name)
        ekind = subpart(acs, i, :edge_kind)
        src_name = names[subpart(acs, i, :src)]
        tgt_name = names[subpart(acs, i, :tgt)]
        add_morphism!(D, ename, src_name, tgt_name)
    end

    # Reconstruct Kan extensions
    for i in 1:nparts(acs, :Kan)
        kname = subpart(acs, i, :kan_name)
        kdir = subpart(acs, i, :kan_dir)
        kreducer = subpart(acs, i, :kan_reducer)
        src_name = names[subpart(acs, i, :kan_src)]
        along_name = names[subpart(acs, i, :kan_along)]
        if kdir == :left
            add_left_kan!(D, kname; source=src_name, along=along_name, reducer=kreducer)
        else
            add_right_kan!(D, kname; source=src_name, along=along_name, reducer=kreducer)
        end
    end

    D
end

# ---------------------------------------------------------------------------
# Symbolic Catlab representation
# ---------------------------------------------------------------------------

using Catlab.Theories: FreeCategory, Ob, Hom, dom, codom

"""
    to_presentation(D::Diagram) -> Presentation

Convert a FunctorFlow Diagram into a Catlab Presentation (free category).
Each object becomes a generator of sort Ob, each morphism a generator of
sort Hom. Compositions become composed Hom expressions.
"""
function to_presentation(D)
    pres = Catlab.Theories.Presentation(FreeCategory)
    ob_gens = Dict{Symbol, Any}()

    for (name, _) in D.objects
        gen = Ob(FreeCategory, name)
        Catlab.Theories.add_generator!(pres, gen)
        ob_gens[name] = gen
    end

    hom_names = Set{Symbol}()
    for (name, op) in D.operations
        if op isa Morphism && name ∉ hom_names
            s = get(ob_gens, op.source, nothing)
            t = get(ob_gens, op.target, nothing)
            (s === nothing || t === nothing) && continue
            Catlab.Theories.add_generator!(pres, Hom(name, s, t))
            push!(hom_names, name)
        end
    end

    pres
end

"""
    to_symbolic(D::Diagram) -> NamedTuple

Convert a FunctorFlow Diagram into symbolic Catlab category elements.

Returns `(objects, morphisms, compositions)` where each is a Dict
mapping names to FreeCategory expressions.
"""
function to_symbolic(D)
    obs = Dict{Symbol, Any}()
    for (name, _) in D.objects
        obs[name] = Ob(FreeCategory, name)
    end

    homs = Dict{Symbol, Any}()
    for (name, op) in D.operations
        if op isa Morphism
            s = get(obs, op.source, nothing)
            t = get(obs, op.target, nothing)
            (s === nothing || t === nothing) && continue
            homs[name] = Hom(name, s, t)
        end
    end

    comps = Dict{Symbol, Any}()
    for (name, op) in D.operations
        if op isa Composition && length(op.chain) >= 2
            parts = [get(homs, c, nothing) for c in op.chain]
            all(!isnothing, parts) || continue
            comps[name] = foldl(Catlab.Theories.compose, parts)
        end
    end

    (objects=obs, morphisms=homs, compositions=comps)
end
