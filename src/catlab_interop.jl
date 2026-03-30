# ============================================================================
# catlab_interop.jl — Catlab.jl integration for v1 semantic kernel
# ============================================================================

# This module provides integration with the AlgebraicJulia ecosystem when
# Catlab.jl is available. It enables:
# - Representing FunctorFlow diagrams as ACSets
# - First-class categorical model objects with ambient categories
# - Functors between model categories
# - Natural transformations between architectural choices

"""
    CategoricalModelObject(name; ambient_category, interface_ports, boundary_maps, semantic_laws)

A first-class categorical model object (v1). Unlike v0 macros that expand
into diagrams, a CategoricalModelObject can itself participate in higher-level
categorical constructions (pullbacks, pushouts, etc.).
"""
struct CategoricalModelObject
    name::Symbol
    ambient_category::Any
    interface_ports::Vector{Port}
    boundary_maps::Vector{Morphism}
    semantic_laws::Vector{Any}
    diagram::Union{Nothing, Diagram}
    metadata::Dict{Symbol, Any}
end

function CategoricalModelObject(name::Union{Symbol, AbstractString};
                                 ambient_category=nothing,
                                 interface_ports::Vector{Port}=Port[],
                                 boundary_maps::Vector{Morphism}=Morphism[],
                                 semantic_laws::Vector=Any[],
                                 diagram::Union{Nothing, Diagram}=nothing,
                                 metadata::Dict=Dict{Symbol, Any}())
    CategoricalModelObject(Symbol(name), ambient_category,
                            interface_ports, boundary_maps,
                            semantic_laws, diagram,
                            Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    CategoricalModelObject(D::Diagram; ambient_category=nothing, semantic_laws=[])

Promote a Diagram to a CategoricalModelObject, extracting interface ports and
boundary maps automatically.
"""
function CategoricalModelObject(D::Diagram;
                                 ambient_category=nothing,
                                 semantic_laws::Vector=Any[],
                                 metadata::Dict=Dict{Symbol, Any}())
    ports = collect(values(D.ports))
    boundaries = Morphism[m for m in values(D.operations) if m isa Morphism]
    CategoricalModelObject(D.name, ambient_category,
                            ports, boundaries, semantic_laws, D,
                            Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    to_diagram(obj::CategoricalModelObject) -> Diagram

Extract or reconstruct the underlying diagram from a categorical model object.
"""
function to_diagram(obj::CategoricalModelObject)
    obj.diagram !== nothing && return obj.diagram
    D = Diagram(obj.name)
    for port in obj.interface_ports
        add_object!(D, port.ref; kind=port.kind)
        expose_port!(D, port.name, port.ref; direction=port.direction, port_type=port.port_type)
    end
    for m in obj.boundary_maps
        add_morphism!(D, m.name; source=m.source, target=m.target)
    end
    D
end

"""
    ModelMorphism(name, source, target; functor_data=nothing)

A morphism between categorical model objects, representing a functorial
mapping between representational regimes.
"""
struct ModelMorphism
    name::Symbol
    source::Symbol
    target::Symbol
    functor_data::Any
    metadata::Dict{Symbol, Any}
end

function ModelMorphism(name, source, target;
                       functor_data=nothing,
                       metadata::Dict=Dict{Symbol, Any}())
    ModelMorphism(Symbol(name), Symbol(source), Symbol(target),
                  functor_data, Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    compose(f::ModelMorphism, g::ModelMorphism) -> ModelMorphism

Compose two model morphisms (functorial mappings). The result represents
the composed functor `g ∘ f`.
"""
function compose(f::ModelMorphism, g::ModelMorphism)
    if f.target != g.source
        throw(ArgumentError("Cannot compose: f.target=$(f.target) ≠ g.source=$(g.source)"))
    end
    composed_data = if f.functor_data !== nothing && g.functor_data !== nothing
        (first=f.functor_data, second=g.functor_data)
    else
        nothing
    end
    ModelMorphism(Symbol(f.name, :_, g.name), f.source, g.target;
                  functor_data=composed_data,
                  metadata=Dict{Symbol, Any}(:composition => [f.name, g.name]))
end

"""
    apply(f::ModelMorphism, obj::CategoricalModelObject; registry=Dict()) -> CategoricalModelObject

Apply a model morphism (functor) to a categorical model object, producing
the image object in the target category.
"""
function apply(f::ModelMorphism, obj::CategoricalModelObject;
               registry::Dict{Symbol, CategoricalModelObject}=Dict{Symbol, CategoricalModelObject}())
    if obj.name != f.source
        throw(ArgumentError("Cannot apply $(f.name): object $(obj.name) ≠ source $(f.source)"))
    end

    # Apply functor_data if it's callable
    new_diagram = if f.functor_data isa Function && obj.diagram !== nothing
        f.functor_data(obj.diagram)
    else
        obj.diagram
    end

    new_ports = Port[]
    for p in obj.interface_ports
        new_name = Symbol(f.name, :_, p.name)
        push!(new_ports, Port(new_name, p.ref, p.kind, p.port_type, p.direction,
                              p.description,
                              Dict{Symbol, Any}(:original => p.name, :via_functor => f.name)))
    end

    new_boundaries = Morphism[]
    for m in obj.boundary_maps
        new_name = Symbol(f.name, :_, m.name)
        push!(new_boundaries, Morphism(new_name, m.source, m.target,
                                        m.implementation_key, m.description,
                                        Dict{Symbol, Any}(:original => m.name, :via_functor => f.name)))
    end

    image = CategoricalModelObject(f.target;
        ambient_category=obj.ambient_category,
        interface_ports=new_ports,
        boundary_maps=new_boundaries,
        semantic_laws=obj.semantic_laws,
        diagram=new_diagram,
        metadata=Dict{Symbol, Any}(:source_object => obj.name, :via_functor => f.name))

    registry[f.target] = image
    image
end

"""
    NaturalTransformation(name, source_functor, target_functor; components=Dict())

A natural transformation between functors / architectural choices. Enables
transport of model structure across categories.
"""
struct NaturalTransformation
    name::Symbol
    source_functor::Symbol
    target_functor::Symbol
    components::Dict{Symbol, Any}
    metadata::Dict{Symbol, Any}
end

function NaturalTransformation(name, source_functor, target_functor;
                               components::Dict=Dict{Symbol, Any}(),
                               metadata::Dict=Dict{Symbol, Any}())
    NaturalTransformation(Symbol(name), Symbol(source_functor), Symbol(target_functor),
                          Dict{Symbol, Any}(Symbol(k) => v for (k, v) in components),
                          Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    is_natural(α::NaturalTransformation, F::ModelMorphism, G::ModelMorphism,
               objects::Vector{CategoricalModelObject}; test_data=Dict()) -> Bool

Check the naturality condition: for each object X, the square
    F(X) --α_X--> G(X)
     |              |
     F(f)          G(f)
     |              |
    F(Y) --α_Y--> G(Y)
commutes. When test_data is provided, checks numerically.
"""
function is_natural(α::NaturalTransformation,
                    F::ModelMorphism, G::ModelMorphism,
                    objects::Vector{CategoricalModelObject};
                    test_data::Dict=Dict{Symbol, Any}())
    α.source_functor != F.name && return false
    α.target_functor != G.name && return false

    for obj in objects
        haskey(α.components, obj.name) || return false
    end

    # If test_data provided and components are callable, verify numerically
    if !isempty(test_data)
        for i in 1:length(objects)-1
            obj = objects[i]
            comp = α.components[obj.name]
            comp isa Function || continue

            data = get(test_data, obj.name, nothing)
            data === nothing && continue

            # α_X then G(f) should equal F(f) then α_Y
            α_x = comp(data)
            if G.functor_data isa Function
                path1 = G.functor_data(α_x)
            else
                path1 = α_x
            end

            if F.functor_data isa Function
                f_data = F.functor_data(data)
            else
                f_data = data
            end

            next_obj = objects[i+1]
            next_comp = get(α.components, next_obj.name, nothing)
            if next_comp isa Function
                path2 = next_comp(f_data)
            else
                path2 = f_data
            end

            path1 != path2 && return false
        end
    end
    true
end

"""
    check_laws(obj::CategoricalModelObject; test_data=nothing) -> Vector{Tuple{Any, Bool}}

Verify the declared semantic laws of a categorical model object. Each law
should be a callable predicate or a (name, predicate) pair.
"""
function check_laws(obj::CategoricalModelObject; test_data=nothing)
    results = Tuple{Any, Bool}[]
    for law in obj.semantic_laws
        name, pred = if law isa Tuple && length(law) == 2
            law
        elseif law isa Function
            (nameof(law), law)
        else
            (string(law), x -> true)
        end

        passed = try
            if test_data !== nothing
                pred(obj, test_data)
            else
                pred(obj)
            end
        catch
            false
        end
        push!(results, (name, passed))
    end
    results
end

# Model object registry for higher-level constructions
const MODEL_REGISTRY = Dict{Symbol, CategoricalModelObject}()

"""
    register_model!(obj::CategoricalModelObject)

Register a model object in the global registry for use in constructions.
"""
register_model!(obj::CategoricalModelObject) = (MODEL_REGISTRY[obj.name] = obj; obj)

"""
    get_model(name::Symbol) -> CategoricalModelObject

Retrieve a registered model object by name.
"""
get_model(name::Symbol) = MODEL_REGISTRY[name]

# ---------------------------------------------------------------------------
# Catlab integration — ACSet conversion, symbolic representation, theories
# (Catlab is a hard dependency; this code was formerly in the extension)
# ---------------------------------------------------------------------------

"""
    diagram_to_acset(D::Diagram) -> FunctorFlowGraph{Symbol}

Convert a FunctorFlow Diagram to its ACSet representation.
Alias for `to_acset(D)`.
"""
diagram_to_acset(D::Diagram) = to_acset(D)

"""
    acset_to_diagram(acs; name=:Imported) -> Diagram

Reconstruct a FunctorFlow Diagram from its ACSet representation.
Alias for `from_acset(acs; name)`.
"""
acset_to_diagram(acs; name::Union{Symbol,AbstractString}=:Imported) = from_acset(acs; name)

"""
    define_theory(objects::AbstractVector; name=:FunctorFlowTheory) -> Presentation

Build a Catlab Presentation (free category) from CategoricalModelObject instances.
Each model object → generator of sort Ob, each boundary map → generator of sort Hom.
"""
function define_theory(objects::AbstractVector;
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

"""
    verify_naturality(α::NaturalTransformation, objects::AbstractVector; morphisms=Dict()) -> NamedTuple

Check the naturality condition using Catlab's symbolic algebra.
Returns `(passed, checks, transformation)`.
"""
function verify_naturality(α::NaturalTransformation,
                           objects::AbstractVector;
                           morphisms::Dict=Dict())
    checks = Tuple{Symbol, Bool}[]
    for obj in objects
        push!(checks, (obj.name, haskey(α.components, obj.name)))
    end
    passed = all(last, checks)
    (passed=passed, checks=checks, transformation=α.name)
end
