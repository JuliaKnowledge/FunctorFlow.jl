# ============================================================================
# topos.jl — Topos-theoretic foundations (v1)
# ============================================================================

# Topos-theoretic reasoning for FunctorFlow. This module provides the
# foundations for internal logic, subobject classifiers, and sheaf coherence.
# Full topos support is a v2 goal; v1 provides operational foundations.

"""
    SubobjectClassifier(name; truth_object, true_map)

A subobject classifier Ω with truth morphism true: 1 → Ω. In FunctorFlow,
this enables internal predicates over model states and constructions.
"""
struct SubobjectClassifier
    name::Symbol
    truth_object::Symbol
    true_map::Symbol
    truth_values::Set{Symbol}
    metadata::Dict{Symbol, Any}
end

function SubobjectClassifier(name::Union{Symbol, AbstractString};
                             truth_object::Union{Symbol, AbstractString}=:Omega,
                             true_map::Union{Symbol, AbstractString}=:true_map,
                             truth_values::Set{Symbol}=Set{Symbol}([Symbol("true"), Symbol("false")]),
                             metadata::Dict=Dict{Symbol, Any}())
    SubobjectClassifier(Symbol(name), Symbol(truth_object), Symbol(true_map),
                        truth_values,
                        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    SheafSection(name; base_space, section_data, domain)

A local section of a sheaf over a base space. Sections represent local
knowledge fragments that must be glued into a global section.
"""
struct SheafSection
    name::Symbol
    base_space::Symbol
    section_data::Any
    domain::Any
    metadata::Dict{Symbol, Any}
end

function SheafSection(name::Union{Symbol, AbstractString};
                      base_space::Union{Symbol, AbstractString}=:BaseSpace,
                      section_data=nothing,
                      domain=nothing,
                      metadata::Dict=Dict{Symbol, Any}())
    SheafSection(Symbol(name), Symbol(base_space), section_data, domain,
                 Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    SheafCoherenceCheck(sections; overlap_checker, gluing_condition)

Check that local sections satisfy the sheaf gluing axioms:
- Locality: sections determined by local neighborhoods
- Gluing: local agreements extend to global section
- Stability: regime-level coherence (cross-document consistency)
"""
struct SheafCoherenceCheck
    sections::Vector{SheafSection}
    overlap_checker::Any
    gluing_condition::Any
    stability_penalty::Float64
    metadata::Dict{Symbol, Any}
end

function SheafCoherenceCheck(sections::Vector{SheafSection};
                             overlap_checker=nothing,
                             gluing_condition=nothing,
                             stability_penalty::Real=0.0,
                             metadata::Dict=Dict{Symbol, Any}())
    SheafCoherenceCheck(sections, overlap_checker, gluing_condition,
                        Float64(stability_penalty),
                        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    check_coherence(check::SheafCoherenceCheck) -> NamedTuple

Run the sheaf gluing axiom checks on sections:
- **Locality**: each section has non-empty domain
- **Gluing**: overlapping sections agree on their intersection
- **Stability**: total coherence penalty is bounded

Returns (passed=Bool, locality=Bool, gluing=Bool, stability=Bool, details=Dict).
"""
function check_coherence(check::SheafCoherenceCheck)
    details = Dict{Symbol, Any}()

    # Locality: each section has defined domain and data
    locality_results = Bool[]
    for sec in check.sections
        has_domain = sec.domain !== nothing
        has_data = sec.section_data !== nothing
        push!(locality_results, has_domain && has_data)
        details[Symbol(:locality_, sec.name)] = has_domain && has_data
    end
    locality = all(locality_results)

    # Gluing: check overlap agreement between section pairs
    gluing_results = Bool[]
    if check.overlap_checker !== nothing
        for i in 1:length(check.sections)
            for j in (i+1):length(check.sections)
                s1 = check.sections[i]
                s2 = check.sections[j]
                try
                    agrees = check.overlap_checker(s1, s2)
                    push!(gluing_results, agrees)
                    details[Symbol(:gluing_, s1.name, :_, s2.name)] = agrees
                catch e
                    push!(gluing_results, false)
                    details[Symbol(:gluing_, s1.name, :_, s2.name)] = false
                    details[Symbol(:gluing_error_, s1.name, :_, s2.name)] = string(e)
                end
            end
        end
    end
    gluing = isempty(gluing_results) ? true : all(gluing_results)

    # Stability: coherence penalty within tolerance
    stability = check.stability_penalty >= 0.0
    if check.gluing_condition !== nothing
        try
            penalty = check.gluing_condition(check.sections)
            details[:computed_penalty] = penalty
            stability = penalty <= check.stability_penalty
        catch e
            stability = false
            details[:stability_error] = string(e)
        end
    end

    passed = locality && gluing && stability
    (passed=passed, locality=locality, gluing=gluing, stability=stability, details=details)
end

"""
    InternalPredicate(name; classifier, characteristic_map)

An internal predicate in the FunctorFlow topos, represented via a
characteristic morphism into the subobject classifier.
"""
struct InternalPredicate
    name::Symbol
    classifier::SubobjectClassifier
    characteristic_map::Any
    metadata::Dict{Symbol, Any}
end

function InternalPredicate(name::Union{Symbol, AbstractString},
                           classifier::SubobjectClassifier;
                           characteristic_map=nothing,
                           metadata::Dict=Dict{Symbol, Any}())
    InternalPredicate(Symbol(name), classifier, characteristic_map,
                      Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    evaluate_predicate(pred::InternalPredicate, data) -> Symbol

Evaluate an internal predicate on data, returning a truth value from
the subobject classifier.
"""
function evaluate_predicate(pred::InternalPredicate, data)
    pred.characteristic_map === nothing && throw(ArgumentError(
        "InternalPredicate $(pred.name) has no characteristic map"))
    result = pred.characteristic_map(data)
    # Normalize to a truth value in the classifier
    if result isa Bool
        result ? Symbol("true") : Symbol("false")
    elseif result isa Symbol && result in pred.classifier.truth_values
        result
    else
        result ? Symbol("true") : Symbol("false")
    end
end

"""
    classify_subobject(classifier::SubobjectClassifier, inclusion_fn, data) -> Dict

Compute the characteristic morphism for a subobject inclusion.
For each element in data, determines whether it belongs to the subobject.
Returns a Dict mapping elements to truth values.
"""
function classify_subobject(classifier::SubobjectClassifier,
                            inclusion_fn::Function, data)
    result = Dict{Any, Symbol}()
    if data isa Dict
        for (k, v) in data
            try
                result[k] = inclusion_fn(v) ? Symbol("true") : Symbol("false")
            catch
                result[k] = Symbol("false")
            end
        end
    elseif data isa AbstractVector
        for (i, v) in enumerate(data)
            try
                result[i] = inclusion_fn(v) ? Symbol("true") : Symbol("false")
            catch
                result[i] = Symbol("false")
            end
        end
    end
    result
end

"""
    internal_and(p1::InternalPredicate, p2::InternalPredicate;
                 classifier=p1.classifier, name=:and) -> InternalPredicate

Conjunction of two internal predicates in the topos.
"""
function internal_and(p1::InternalPredicate, p2::InternalPredicate;
                      classifier::SubobjectClassifier=p1.classifier,
                      name::Union{Symbol, AbstractString}=Symbol(p1.name, :_and_, p2.name))
    combined_map = if p1.characteristic_map !== nothing && p2.characteristic_map !== nothing
        x -> p1.characteristic_map(x) && p2.characteristic_map(x)
    else
        nothing
    end
    InternalPredicate(name, classifier; characteristic_map=combined_map,
                      metadata=Dict{Symbol, Any}(:op => :and, :left => p1.name, :right => p2.name))
end

"""
    internal_or(p1::InternalPredicate, p2::InternalPredicate;
                classifier=p1.classifier, name=:or) -> InternalPredicate

Disjunction of two internal predicates in the topos.
"""
function internal_or(p1::InternalPredicate, p2::InternalPredicate;
                     classifier::SubobjectClassifier=p1.classifier,
                     name::Union{Symbol, AbstractString}=Symbol(p1.name, :_or_, p2.name))
    combined_map = if p1.characteristic_map !== nothing && p2.characteristic_map !== nothing
        x -> p1.characteristic_map(x) || p2.characteristic_map(x)
    else
        nothing
    end
    InternalPredicate(name, classifier; characteristic_map=combined_map,
                      metadata=Dict{Symbol, Any}(:op => :or, :left => p1.name, :right => p2.name))
end

"""
    internal_not(p::InternalPredicate; classifier=p.classifier) -> InternalPredicate

Negation of an internal predicate in the topos.
"""
function internal_not(p::InternalPredicate;
                      classifier::SubobjectClassifier=p.classifier,
                      name::Union{Symbol, AbstractString}=Symbol(:not_, p.name))
    negated_map = p.characteristic_map !== nothing ? (x -> !p.characteristic_map(x)) : nothing
    InternalPredicate(name, classifier; characteristic_map=negated_map,
                      metadata=Dict{Symbol, Any}(:op => :not, :operand => p.name))
end

"""
    build_sheaf_diagram(sections; name=:SheafGluing, overlap_relation=:Overlap) -> Diagram

Build a FunctorFlow diagram representing sheaf gluing from local sections.
Uses the Democritus-style right-Kan gluing construction.
"""
function build_sheaf_diagram(sections::Vector{SheafSection};
                             name::Union{Symbol, AbstractString}=:SheafGluing,
                             overlap_relation::Union{Symbol, AbstractString}=:Overlap,
                             reducer::Union{Symbol, AbstractString}=:set_union)
    D = Diagram(name)
    add_object!(D, :LocalSections; kind=:local_claims,
                metadata=Dict{Symbol, Any}(:n_sections => length(sections),
                                            :section_names => [s.name for s in sections]))
    add_object!(D, overlap_relation; kind=:overlap_region)
    add_object!(D, :GlobalSection; kind=:global_state)

    add_right_kan!(D, :glue;
                   source=:LocalSections, along=overlap_relation,
                   target=:GlobalSection, reducer=reducer,
                   metadata=Dict{Symbol, Any}(:topos => :sheaf_gluing,
                                               :n_sections => length(sections)))

    expose_port!(D, :sections, :LocalSections; direction=INPUT, port_type=:local_claims)
    expose_port!(D, :overlaps, overlap_relation; direction=INPUT, port_type=:overlap_region)
    expose_port!(D, :global, :glue; direction=OUTPUT, port_type=:global_state)
    D
end
