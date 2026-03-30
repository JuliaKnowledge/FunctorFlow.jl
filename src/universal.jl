# ============================================================================
# universal.jl — Universal constructions for v1 model assembly
# ============================================================================

# Universal constructions make compositionality real: not just composition of
# named operations, but composition through universal properties.

"""Supertype for results of universal constructions."""
abstract type UniversalConstruction end

"""
    PullbackResult(name, cone, projection1, projection2, universal_map)

Result of a pullback construction. Represents the most general object that
maps compatibly into two given objects over a shared base.

In FunctorFlow, pullbacks build joint constraint-compatible models: given two
KET models with shared interface morphisms, the pullback is the model that
satisfies both sets of constraints simultaneously.
"""
struct PullbackResult <: UniversalConstruction
    name::Symbol
    cone::Diagram
    projection1::Symbol
    projection2::Symbol
    shared_object::Symbol
    interface_morphisms::Vector{Symbol}
    universal_map::Any
    metadata::Dict{Symbol, Any}
end

"""
    PushoutResult(name, cocone, injection1, injection2, universal_map)

Result of a pushout construction. Represents gluing two objects along a
shared sub-object.

In FunctorFlow, pushouts merge partial models while preserving interfaces.
"""
struct PushoutResult <: UniversalConstruction
    name::Symbol
    cocone::Diagram
    injection1::Symbol
    injection2::Symbol
    shared_object::Symbol
    interface_morphisms::Vector{Symbol}
    universal_map::Any
    metadata::Dict{Symbol, Any}
end

"""
    ProductResult(name, product_diagram, projections)

Result of a product construction. Combines independent models.
"""
struct ProductResult <: UniversalConstruction
    name::Symbol
    product_diagram::Diagram
    projections::Vector{Symbol}
    metadata::Dict{Symbol, Any}
end

"""
    CoproductResult(name, coproduct_diagram, injections)

Result of a coproduct construction. Hypothesis aggregation / ensemble.
"""
struct CoproductResult <: UniversalConstruction
    name::Symbol
    coproduct_diagram::Diagram
    injections::Vector{Symbol}
    metadata::Dict{Symbol, Any}
end

"""
    EqualizerResult(name, equalizer_diagram, equalizer_map)

Result of an equalizer construction. Enforces that two morphisms agree.
"""
struct EqualizerResult <: UniversalConstruction
    name::Symbol
    equalizer_diagram::Diagram
    equalizer_map::Symbol
    metadata::Dict{Symbol, Any}
end

"""
    CoequalizerResult(name, coequalizer_diagram, coequalizer_map)

Result of a coequalizer construction. Given two parallel morphisms `f, g : A → B`,
the coequalizer is the quotient object `Q` with a map `q : B → Q` such that
`q ∘ f = q ∘ g`. This identifies elements of B that are related by f and g.

Dually to the equalizer (which finds where f and g agree), the coequalizer
*forces* f and g to agree by quotienting B.
"""
struct CoequalizerResult <: UniversalConstruction
    name::Symbol
    coequalizer_diagram::Diagram
    coequalizer_map::Symbol           # q : B → Q
    quotient_object::Symbol           # Q
    metadata::Dict{Symbol, Any}
end

# ---------------------------------------------------------------------------
# Construction functions
# ---------------------------------------------------------------------------

"""
    pullback(D1, D2; over, name=:Pullback) -> PullbackResult

Construct the pullback of two diagrams over a shared interface. The resulting
diagram contains both sub-diagrams plus interface morphisms (projections)
into a shared base object. The pullback is the universal cone: any other
diagram mapping compatibly into D1 and D2 factors uniquely through it.
"""
function pullback(D1::Diagram, D2::Diagram;
                  over::Union{Symbol, AbstractString},
                  name::Union{Symbol, AbstractString}=:Pullback,
                  metadata::Dict=Dict{Symbol, Any}())
    n = Symbol(name)
    result = Diagram(n)

    inc1 = include!(result, D1; namespace=:left)
    inc2 = include!(result, D2; namespace=:right)

    shared = Symbol(over)
    add_object!(result, shared; kind=:shared_interface,
                metadata=Dict{Symbol, Any}(:construction => :pullback_base))

    # Create interface morphisms (projections from cone to factors)
    proj1_name = Symbol(:π₁)
    proj2_name = Symbol(:π₂)
    interface_morphisms = Symbol[]

    # Find output ports from each factor to create projections
    for (pname, port) in D1.ports
        if port.direction == OUTPUT
            morphism_name = Symbol(:proj_left_, pname)
            add_morphism!(result, morphism_name,
                         Symbol(:left__, port.ref), shared;
                         metadata=Dict{Symbol, Any}(:role => :pullback_projection, :factor => :left))
            push!(interface_morphisms, morphism_name)
        end
    end
    for (pname, port) in D2.ports
        if port.direction == OUTPUT
            morphism_name = Symbol(:proj_right_, pname)
            add_morphism!(result, morphism_name,
                         Symbol(:right__, port.ref), shared;
                         metadata=Dict{Symbol, Any}(:role => :pullback_projection, :factor => :right))
            push!(interface_morphisms, morphism_name)
        end
    end

    # Add commuting constraint: projections should agree on the shared interface
    if length(interface_morphisms) >= 2
        add_obstruction_loss!(result, Symbol(n, :_commuting);
                             paths=[(interface_morphisms[1], interface_morphisms[2])],
                             comparator=:l2, weight=1.0,
                             metadata=Dict{Symbol, Any}(:role => :pullback_constraint))
    end

    PullbackResult(n, result, :left, :right, shared, interface_morphisms, nothing,
                   Dict{Symbol, Any}(:over => shared, metadata...))
end

"""
    pushout(D1, D2; along, name=:Pushout) -> PushoutResult

Construct the pushout of two diagrams along a shared sub-object. Creates
injection morphisms from each factor into the merged result.
"""
function pushout(D1::Diagram, D2::Diagram;
                 along::Union{Symbol, AbstractString},
                 name::Union{Symbol, AbstractString}=:Pushout,
                 metadata::Dict=Dict{Symbol, Any}())
    n = Symbol(name)
    result = Diagram(n)

    inc1 = include!(result, D1; namespace=:left)
    inc2 = include!(result, D2; namespace=:right)

    shared = Symbol(along)
    add_object!(result, shared; kind=:shared_subobject,
                metadata=Dict{Symbol, Any}(:construction => :pushout_glue))

    # Create injection morphisms (from shared into factors)
    interface_morphisms = Symbol[]
    for (pname, port) in D1.ports
        if port.direction == INPUT
            morphism_name = Symbol(:ι_left_, pname)
            add_morphism!(result, morphism_name,
                         shared, Symbol(:left__, port.ref);
                         metadata=Dict{Symbol, Any}(:role => :pushout_injection, :factor => :left))
            push!(interface_morphisms, morphism_name)
        end
    end
    for (pname, port) in D2.ports
        if port.direction == INPUT
            morphism_name = Symbol(:ι_right_, pname)
            add_morphism!(result, morphism_name,
                         shared, Symbol(:right__, port.ref);
                         metadata=Dict{Symbol, Any}(:role => :pushout_injection, :factor => :right))
            push!(interface_morphisms, morphism_name)
        end
    end

    PushoutResult(n, result, :left, :right, shared, interface_morphisms, nothing,
                  Dict{Symbol, Any}(:along => shared, metadata...))
end

"""
    product(diagrams...; name=:Product) -> ProductResult

Construct the product of multiple diagrams. Combines independent models
with projection morphisms into each factor.
"""
function product(diagrams::Diagram...; name::Union{Symbol, AbstractString}=:Product,
                 metadata::Dict=Dict{Symbol, Any}())
    n = Symbol(name)
    result = Diagram(n)
    projections = Symbol[]

    for (i, D) in enumerate(diagrams)
        ns = Symbol(:factor_, i)
        include!(result, D; namespace=ns)
        push!(projections, ns)
    end

    ProductResult(n, result, projections,
                  Dict{Symbol, Any}(:n_factors => length(diagrams), metadata...))
end

"""
    coproduct(diagrams...; name=:Coproduct) -> CoproductResult

Construct the coproduct of multiple diagrams. Ensemble / hypothesis aggregation.
"""
function coproduct(diagrams::Diagram...; name::Union{Symbol, AbstractString}=:Coproduct,
                   metadata::Dict=Dict{Symbol, Any}())
    n = Symbol(name)
    result = Diagram(n)
    injections = Symbol[]

    for (i, D) in enumerate(diagrams)
        ns = Symbol(:summand_, i)
        include!(result, D; namespace=ns)
        push!(injections, ns)
    end

    CoproductResult(n, result, injections,
                    Dict{Symbol, Any}(:n_summands => length(diagrams), metadata...))
end

"""
    equalizer(D, f_name, g_name; name=:Equalizer) -> EqualizerResult

Construct the equalizer of two morphisms in a diagram. Enforces `f = g`.
This is closely related to the DB obstruction loss: the equalizer is the
sub-object where the two paths agree exactly.
"""
function equalizer(D::Diagram, f_name::Union{Symbol, AbstractString},
                   g_name::Union{Symbol, AbstractString};
                   name::Union{Symbol, AbstractString}=:Equalizer,
                   metadata::Dict=Dict{Symbol, Any}())
    n = Symbol(name)
    result = Diagram(n)
    inc = include!(result, D; namespace=:base)

    # Add equalizer constraint as an obstruction loss with zero tolerance
    add_obstruction_loss!(result, Symbol(n, :_eq_loss);
                          paths=[(Symbol(:base__, Symbol(f_name)),
                                  Symbol(:base__, Symbol(g_name)))],
                          comparator=:l2, weight=1.0)

    EqualizerResult(n, result, Symbol(:base__, Symbol(f_name)),
                    Dict{Symbol, Any}(:f => Symbol(f_name), :g => Symbol(g_name), metadata...))
end

"""
    coequalizer(D, f_name, g_name; name=:Coequalizer) -> CoequalizerResult

Construct the coequalizer of two parallel morphisms `f, g : A → B` in a diagram.

The coequalizer is the universal quotient object `Q` with a map `q : B → Q` such
that `q ∘ f = q ∘ g`. It identifies elements of B that are related through f and g.

Where an equalizer finds the subobject of A where f and g agree (a limit),
the coequalizer quotients B by forcing f and g to agree (a colimit).

# AI interpretation
- **Equivalence classes**: collapse representations that two maps agree should be identified
- **Symmetry quotienting**: remove redundant structure by identifying symmetric states
- **Consensus merging**: merge outputs that multiple processing paths declare equivalent
"""
function coequalizer(D::Diagram, f_name::Union{Symbol, AbstractString},
                     g_name::Union{Symbol, AbstractString};
                     name::Union{Symbol, AbstractString}=:Coequalizer,
                     metadata::Dict=Dict{Symbol, Any}())
    n = Symbol(name)
    f_sym = Symbol(f_name)
    g_sym = Symbol(g_name)

    # Validate: f and g must exist and share source and target
    haskey(D.operations, f_sym) || error("Morphism :$f_sym not found in diagram")
    haskey(D.operations, g_sym) || error("Morphism :$g_sym not found in diagram")
    f_op = D.operations[f_sym]
    g_op = D.operations[g_sym]
    f_op isa Morphism || error(":$f_sym is not a morphism")
    g_op isa Morphism || error(":$g_sym is not a morphism")
    f_op.source == g_op.source || error("Morphisms must share source: $(f_op.source) ≠ $(g_op.source)")
    f_op.target == g_op.target || error("Morphisms must share target: $(f_op.target) ≠ $(g_op.target)")

    result = Diagram(n)
    inc = include!(result, D; namespace=:base)

    # Create the quotient object Q
    quotient_obj = Symbol(n, :_Quotient)
    add_object!(result, quotient_obj;
                kind=:quotient,
                description="Coequalizer quotient of $(f_op.target) by $f_sym ~ $g_sym")

    # Create the coequalizer map q : B → Q
    coeq_map = Symbol(n, :_q)
    base_target = Symbol(:base__, f_op.target)
    add_morphism!(result, coeq_map, base_target, quotient_obj;
                  description="Coequalizer map: quotient by $f_sym ~ $g_sym",
                  metadata=Dict{Symbol, Any}(:role => :coequalizer_map))

    # Add compositions q∘f and q∘g
    base_f = Symbol(:base__, f_sym)
    base_g = Symbol(:base__, g_sym)
    qf_name = Symbol(n, :_qf)
    qg_name = Symbol(n, :_qg)
    compose!(result, base_f, coeq_map; name=qf_name)
    compose!(result, base_g, coeq_map; name=qg_name)

    # Coequalizer constraint: q ∘ f = q ∘ g (obstruction loss)
    add_obstruction_loss!(result, Symbol(n, :_coeq_loss);
                          paths=[(qf_name, qg_name)],
                          comparator=:l2, weight=1.0)

    CoequalizerResult(n, result, coeq_map, quotient_obj,
                      Dict{Symbol, Any}(:f => f_sym, :g => g_sym,
                                        :source => f_op.source, :target => f_op.target,
                                        metadata...))
end

# ---------------------------------------------------------------------------
# Verification functions
# ---------------------------------------------------------------------------

"""
    verify(uc::PullbackResult; test_data=nothing) -> NamedTuple

Verify the pullback universal property:
1. The cone diagram is well-formed (has both factors + shared object)
2. Interface morphisms exist connecting factors to shared base
3. If test_data provided, check the commuting square numerically
"""
function verify(pb::PullbackResult; test_data::Union{Nothing, Dict}=nothing)
    checks = Dict{Symbol, Bool}()

    # Check structure: cone has both factor namespaces
    checks[:has_left_factor] = any(startswith(String(k), "left__") for k in keys(pb.cone.objects))
    checks[:has_right_factor] = any(startswith(String(k), "right__") for k in keys(pb.cone.objects))
    checks[:has_shared_object] = haskey(pb.cone.objects, pb.shared_object)
    checks[:has_interface_morphisms] = !isempty(pb.interface_morphisms)

    # Check commuting constraint exists
    checks[:has_commuting_constraint] = any(
        haskey(loss.metadata, :role) && loss.metadata[:role] == :pullback_constraint
        for loss in values(pb.cone.losses)
    )

    # Numerical verification if test data provided
    if test_data !== nothing && checks[:has_commuting_constraint]
        compiled = compile_to_callable(pb.cone)
        result = FunctorFlow.run(compiled, test_data)
        # Commuting check: loss should be small for compatible inputs
        loss_val = isempty(result.losses) ? 0.0 : first(values(result.losses))
        checks[:commutes_numerically] = loss_val < 1e-6
    end

    all_passed = all(values(checks))
    (passed=all_passed, checks=checks, construction=:pullback)
end

"""
    verify(po::PushoutResult; test_data=nothing) -> NamedTuple

Verify the pushout universal property:
1. The cocone has both factors + shared sub-object
2. Injection morphisms exist from shared into factors
"""
function verify(po::PushoutResult; test_data::Union{Nothing, Dict}=nothing)
    checks = Dict{Symbol, Bool}()

    checks[:has_left_factor] = any(startswith(String(k), "left__") for k in keys(po.cocone.objects))
    checks[:has_right_factor] = any(startswith(String(k), "right__") for k in keys(po.cocone.objects))
    checks[:has_shared_object] = haskey(po.cocone.objects, po.shared_object)
    checks[:has_interface_morphisms] = !isempty(po.interface_morphisms)

    # Check injections go from shared into factors
    for morph_name in po.interface_morphisms
        if haskey(po.cocone.operations, morph_name)
            m = po.cocone.operations[morph_name]
            checks[Symbol(:injection_, morph_name)] = m.source == po.shared_object
        end
    end

    all_passed = all(values(checks))
    (passed=all_passed, checks=checks, construction=:pushout)
end

"""
    verify(prod::ProductResult) -> NamedTuple

Verify the product has all factor projections.
"""
function verify(prod::ProductResult)
    checks = Dict{Symbol, Bool}()
    for proj in prod.projections
        checks[Symbol(:has_factor_, proj)] = any(
            startswith(String(k), String(proj) * "__") for k in keys(prod.product_diagram.objects))
    end
    all_passed = all(values(checks))
    (passed=all_passed, checks=checks, construction=:product)
end

"""
    verify(coprod::CoproductResult) -> NamedTuple

Verify the coproduct has all summand injections.
"""
function verify(coprod::CoproductResult)
    checks = Dict{Symbol, Bool}()
    for inj in coprod.injections
        checks[Symbol(:has_summand_, inj)] = any(
            startswith(String(k), String(inj) * "__") for k in keys(coprod.coproduct_diagram.objects))
    end
    all_passed = all(values(checks))
    (passed=all_passed, checks=checks, construction=:coproduct)
end

"""
    verify(eq::EqualizerResult) -> NamedTuple

Verify the equalizer has a constraining obstruction loss.
"""
function verify(eq::EqualizerResult)
    checks = Dict{Symbol, Bool}()
    checks[:has_equalizer_map] = haskey(eq.equalizer_diagram.operations, eq.equalizer_map)
    checks[:has_eq_loss] = !isempty(eq.equalizer_diagram.losses)
    all_passed = all(values(checks))
    (passed=all_passed, checks=checks, construction=:equalizer)
end

"""
    verify(coeq::CoequalizerResult) -> NamedTuple

Verify the coequalizer has a quotient object, coequalizer map, and constraining loss.
"""
function verify(coeq::CoequalizerResult)
    checks = Dict{Symbol, Bool}()
    checks[:has_quotient_object] = haskey(coeq.coequalizer_diagram.objects, coeq.quotient_object)
    checks[:has_coequalizer_map] = haskey(coeq.coequalizer_diagram.operations, coeq.coequalizer_map)
    checks[:has_coeq_loss] = !isempty(coeq.coequalizer_diagram.losses)
    # Check that the map targets the quotient object
    if checks[:has_coequalizer_map]
        op = coeq.coequalizer_diagram.operations[coeq.coequalizer_map]
        checks[:map_targets_quotient] = op isa Morphism && op.target == coeq.quotient_object
    else
        checks[:map_targets_quotient] = false
    end
    all_passed = all(values(checks))
    (passed=all_passed, checks=checks, construction=:coequalizer)
end

"""
    compile_construction(uc::UniversalConstruction) -> CompiledDiagram

Lower a universal construction to an executable CompiledDiagram.
"""
function compile_construction(uc::PullbackResult)
    compile_to_callable(uc.cone)
end
function compile_construction(uc::PushoutResult)
    compile_to_callable(uc.cocone)
end
function compile_construction(uc::ProductResult)
    compile_to_callable(uc.product_diagram)
end
function compile_construction(uc::CoproductResult)
    compile_to_callable(uc.coproduct_diagram)
end
function compile_construction(uc::EqualizerResult)
    compile_to_callable(uc.equalizer_diagram)
end
function compile_construction(uc::CoequalizerResult)
    compile_to_callable(uc.coequalizer_diagram)
end

"""
    universal_morphism(pb::PullbackResult, from::Diagram; name=:mediating) -> Diagram

Compute the universal (mediating) morphism from a cone `from` into the
pullback. For the pullback to satisfy the universal property, any cone
mapping compatibly into both factors must factor uniquely through the pullback.
"""
function universal_morphism(pb::PullbackResult, from::Diagram;
                            name::Union{Symbol, AbstractString}=:mediating)
    D = Diagram(name)
    inc_from = include!(D, from; namespace=:cone)
    inc_pb = include!(D, pb.cone; namespace=:pullback)

    # The mediating morphism maps from the external cone into the pullback cone
    for (pname, port) in from.ports
        if port.direction == OUTPUT
            med_name = Symbol(:mediate_, pname)
            # Map from cone's output to pullback's corresponding input
            add_morphism!(D, med_name,
                         Symbol(:cone__, port.ref),
                         Symbol(:pullback__, pb.shared_object);
                         metadata=Dict{Symbol, Any}(:role => :mediating_morphism))
        end
    end
    D
end
