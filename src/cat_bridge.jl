# ============================================================================
# cat_bridge.jl — bridge the FunctorFlow Diagram DSL to the Cat kernel
#
# A `Diagram`'s *shape* is a category: objects are objects, morphisms and
# (targeted) Kan extensions are generating arrows, and composed paths are the
# composites. `diagram_freecat` exposes that as a genuine `Cat.FreeCat`, and
# `diagram_setfunctor` realises a concrete instance as a functor to Set
# (a C-Set), making "a Diagram instance is a functor to Set" literal and
# law-checkable rather than a slogan.
# ============================================================================

"""
    diagram_freecat(D::Diagram) -> Cat.FreeCat

The free schema category of a diagram: nodes are `D`'s objects; generating
edges are its morphisms and its Kan extensions that carry an explicit target.
Compositions are *derived* (paths) and so are not generators. The diagram's
generating graph must be acyclic (a requirement of the finite kernel).
"""
function diagram_freecat(D::Diagram)
    objs = collect(keys(D.objects))
    edges = Tuple{Symbol, Symbol, Symbol}[]
    for (name, op) in D.operations
        if op isa Morphism
            push!(edges, (name, op.source, op.target))
        elseif op isa KanExtension && op.target !== nothing
            push!(edges, (name, op.source, op.target))
        end
    end
    # ensure all endpoints are declared objects (auto-add placeholders if needed)
    known = Set(objs)
    for (_, s, t) in edges
        for v in (s, t)
            v in known || (push!(objs, v); push!(known, v))
        end
    end
    Cat.FreeCat(objs, edges)
end

"""
    diagram_setfunctor(D::Diagram; sets, functions) -> Cat.SetFunctor

Realise a concrete instance of a diagram as a functor `schema → FinSet`
(a C-Set). `sets` maps each object to its carrier (a collection of elements);
`functions` maps each generating edge name to the element-level action (an
iterable of `elem => image` pairs). The result is validated for functoriality
on construction.
"""
function diagram_setfunctor(D::Diagram; sets::AbstractDict, functions::AbstractDict)
    C = diagram_freecat(D)
    ob_map = Dict{Symbol, Cat.FinSet}(Symbol(k) => Cat.FinSet(v) for (k, v) in sets)
    edge_map = Dict{Symbol, Cat.FinFunction}()
    for (n, s, t) in C.edges
        haskey(functions, n) || haskey(functions, String(n)) ||
            throw(ArgumentError("diagram_setfunctor: no element map supplied for edge $n"))
        pairs = haskey(functions, n) ? functions[n] : functions[String(n)]
        edge_map[n] = Cat.FinFunction(ob_map[s], ob_map[t], pairs)
    end
    Cat.SetFunctor(C; ob_map=ob_map, edge_map=edge_map)
end
