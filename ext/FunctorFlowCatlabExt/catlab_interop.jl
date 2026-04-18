# ============================================================================
# catlab_interop.jl — Catlab-backed methods for FunctorFlow categorical models
# ============================================================================
#
# Loaded by `FunctorFlowCatlabExt`. Provides the Catlab-using `define_theory`
# method. The struct types (CategoricalModelObject, ModelMorphism,
# NaturalTransformation) and their pure-Julia operations live in the main
# FunctorFlow module (`src/categorical_model.jl`).

"""
    define_theory(objects::AbstractVector; name=:FunctorFlowTheory) -> Presentation

Build a Catlab Presentation (free category) from CategoricalModelObject instances.
Each model object → generator of sort Ob, each boundary map → generator of sort Hom.

Requires `using Catlab` to be active alongside FunctorFlow.
"""
function FunctorFlow.define_theory(objects::AbstractVector;
                                   name::Union{Symbol,AbstractString}=:FunctorFlowTheory)
    pres = Catlab.Theories.Presentation(Catlab.Theories.FreeCategory)
    ob_gens = Dict{Symbol, Any}()

    for obj in objects
        gen = Catlab.Theories.Ob(Catlab.Theories.FreeCategory, obj.name)
        Catlab.Theories.add_generator!(pres, gen)
        ob_gens[obj.name] = gen
    end

    hom_names = Set{Symbol}()
    for obj in objects
        for bm in obj.boundary_maps
            if !haskey(ob_gens, bm.source)
                s = Catlab.Theories.Ob(Catlab.Theories.FreeCategory, bm.source)
                Catlab.Theories.add_generator!(pres, s)
                ob_gens[bm.source] = s
            end
            if !haskey(ob_gens, bm.target)
                t = Catlab.Theories.Ob(Catlab.Theories.FreeCategory, bm.target)
                Catlab.Theories.add_generator!(pres, t)
                ob_gens[bm.target] = t
            end
            if bm.name ∉ hom_names
                s = ob_gens[bm.source]
                t = ob_gens[bm.target]
                Catlab.Theories.add_generator!(pres, Catlab.Theories.Hom(bm.name, s, t))
                push!(hom_names, bm.name)
            end
        end
    end

    pres
end
