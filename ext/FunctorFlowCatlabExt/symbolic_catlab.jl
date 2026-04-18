# ============================================================================
# symbolic_catlab.jl — Symbolic Catlab (FreeCategory) projection of a Diagram
# ============================================================================
#
# Loaded by `FunctorFlowCatlabExt` when `using Catlab` is active alongside
# FunctorFlow. The parent extension module already provides the relevant
# Catlab.Theories names (FreeCategory, Ob, Hom, dom, codom).
#
# This file provides a pure-Catlab symbolic view of a FunctorFlow Diagram.
# It is independent of the ACSet schema (which lives in
# CategoricalDiagramSchema.jl and is wired in via FunctorFlowSchemaExt).

"""
    to_presentation(D::Diagram) -> Presentation

Convert a FunctorFlow Diagram into a Catlab Presentation (free category).
Each object becomes a generator of sort Ob, each morphism a generator of
sort Hom. Compositions become composed Hom expressions.
"""
function FunctorFlow.to_presentation(D)
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
function FunctorFlow.to_symbolic(D)
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
