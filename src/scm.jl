# ============================================================================
# scm.jl — Structural causal model semantics
# ============================================================================

"""
    SCMLocalFunctionSpec(name, target_variable; exogenous_parents=Symbol[], endogenous_parents=Symbol[], expression=nothing, metadata=Dict())

A local structural equation for one endogenous variable in an SCM.
"""
struct SCMLocalFunctionSpec
    name::Symbol
    target_variable::Symbol
    exogenous_parents::Vector{Symbol}
    endogenous_parents::Vector{Symbol}
    expression::Union{Nothing, String}
    metadata::Dict{Symbol, Any}
end

function SCMLocalFunctionSpec(name, target_variable;
                              exogenous_parents::Vector{Symbol}=Symbol[],
                              endogenous_parents::Vector{Symbol}=Symbol[],
                              expression::Union{Nothing, AbstractString}=nothing,
                              metadata::Dict=Dict{Symbol, Any}())
    SCMLocalFunctionSpec(Symbol(name), Symbol(target_variable),
                         copy(exogenous_parents), copy(endogenous_parents),
                         expression === nothing ? nothing : String(expression),
                         Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    SCMObjectSpec(name, exogenous_variables, endogenous_variables, local_functions)

Declarative specification for an SCM object.
"""
struct SCMObjectSpec
    name::Symbol
    exogenous_variables::Vector{Symbol}
    endogenous_variables::Vector{Symbol}
    local_functions::Vector{SCMLocalFunctionSpec}
end

function SCMObjectSpec(name, exogenous_variables::Vector{Symbol},
                       endogenous_variables::Vector{Symbol},
                       local_functions::Vector{SCMLocalFunctionSpec})
    SCMObjectSpec(Symbol(name), copy(exogenous_variables), copy(endogenous_variables), copy(local_functions))
end

"""
    SCMModelObject

A first-class structural causal model object with explicit exogenous variables,
endogenous variables, and local mechanisms.
"""
struct SCMModelObject
    object::CategoricalModelObject
    exogenous_variables::Vector{Symbol}
    endogenous_variables::Vector{Symbol}
    local_functions::Vector{SCMLocalFunctionSpec}
end

function Base.getproperty(obj::SCMModelObject, sym::Symbol)
    if sym === :name
        return getfield(getfield(obj, :object), :name)
    elseif sym === :category
        return getfield(getfield(obj, :object), :ambient_category)
    elseif sym === :interfaces
        return getfield(getfield(obj, :object), :interface_ports)
    else
        return getfield(obj, sym)
    end
end

function Base.propertynames(::SCMModelObject, private::Bool=false)
    props = (:object, :exogenous_variables, :endogenous_variables, :local_functions,
             :name, :category, :interfaces)
    private ? props : props
end

"""
    SCMMorphism

A structure-preserving morphism between two SCM model objects.
"""
struct SCMMorphism
    morphism::ModelMorphism
    source_scm::SCMModelObject
    target_scm::SCMModelObject
    exogenous_variable_map::Vector{Tuple{Symbol, Symbol}}
    endogenous_variable_map::Vector{Tuple{Symbol, Symbol}}
    local_function_map::Vector{Tuple{Symbol, Symbol}}
end

function Base.getproperty(m::SCMMorphism, sym::Symbol)
    if sym === :name
        return getfield(getfield(m, :morphism), :name)
    elseif sym === :source
        return getfield(getfield(m, :morphism), :source)
    elseif sym === :target
        return getfield(getfield(m, :morphism), :target)
    elseif sym === :category
        return getproperty(getfield(m, :source_scm), :category)
    elseif sym === :metadata
        return getfield(getfield(m, :morphism), :metadata)
    else
        return getfield(m, sym)
    end
end

function Base.propertynames(::SCMMorphism, private::Bool=false)
    props = (:morphism, :source_scm, :target_scm, :exogenous_variable_map,
             :endogenous_variable_map, :local_function_map, :name, :source,
             :target, :category, :metadata)
    private ? props : props
end

"""
    SCMPredicateClause(name, statement; clause_kind=:constraint, metadata=Dict())

A single intuitionistic clause over an SCM.
"""
struct SCMPredicateClause
    name::Symbol
    statement::String
    clause_kind::Symbol
    metadata::Dict{Symbol, Any}
end

function SCMPredicateClause(name, statement;
                            clause_kind::Union{Symbol, AbstractString}=:constraint,
                            metadata::Dict=Dict{Symbol, Any}())
    SCMPredicateClause(Symbol(name), String(statement), Symbol(clause_kind),
                       Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    SCMMonomorphism

A monic SCM morphism witnessing subobject inclusion.
"""
struct SCMMonomorphism
    morphism::SCMMorphism
    ambient_scm::SCMModelObject
    metadata::Dict{Symbol, Any}
end

function Base.getproperty(m::SCMMonomorphism, sym::Symbol)
    if sym === :name
        return getproperty(getfield(m, :morphism), :name)
    elseif sym === :source_scm
        return getfield(getfield(m, :morphism), :source_scm)
    elseif sym === :target_scm
        return getfield(getfield(m, :morphism), :target_scm)
    else
        return getfield(m, sym)
    end
end

"""
    SCMSubobject

A constrained subobject of an ambient SCM.
"""
struct SCMSubobject
    name::Symbol
    object_scm::SCMModelObject
    ambient_scm::SCMModelObject
    inclusion::SCMMonomorphism
    clauses::Vector{SCMPredicateClause}
    metadata::Dict{Symbol, Any}
end

"""
    SCMPredicate

An internal predicate over an SCM, represented by a constrained subobject.
"""
struct SCMPredicate
    name::Symbol
    ambient_scm::SCMModelObject
    subobject::SCMSubobject
    clauses::Vector{SCMPredicateClause}
    metadata::Dict{Symbol, Any}
end

"""
    SCMTruthValue(name, meaning; metadata=Dict())

A provisional intuitionistic truth value in `OmegaSCM`.
"""
struct SCMTruthValue
    name::Symbol
    meaning::String
    metadata::Dict{Symbol, Any}
end

function SCMTruthValue(name, meaning; metadata::Dict=Dict{Symbol, Any}())
    SCMTruthValue(Symbol(name), String(meaning), Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    OmegaSCM

A provisional truth object for SCM predicates.
"""
struct OmegaSCM
    object::CategoricalModelObject
    truth_values::Vector{SCMTruthValue}
    metadata::Dict{Symbol, Any}
end

function Base.getproperty(ω::OmegaSCM, sym::Symbol)
    if sym === :name
        return getfield(getfield(ω, :object), :name)
    elseif sym === :category
        return getfield(getfield(ω, :object), :ambient_category)
    elseif sym === :interfaces
        return getfield(getfield(ω, :object), :interface_ports)
    else
        return getfield(ω, sym)
    end
end

"""
    SCMCharacteristicMap

A characteristic map from an ambient SCM into an `OmegaSCM` truth object.
"""
struct SCMCharacteristicMap
    morphism::ModelMorphism
    ambient_scm::SCMModelObject
    predicate::SCMPredicate
    omega::OmegaSCM
    classifying_truth_value::SCMTruthValue
    false_truth_value::SCMTruthValue
    metadata::Dict{Symbol, Any}
end

function Base.getproperty(χ::SCMCharacteristicMap, sym::Symbol)
    if sym === :name
        return getfield(getfield(χ, :morphism), :name)
    else
        return getfield(χ, sym)
    end
end

scm_interfaces() = (
    Port(:exogenous, :Exogenous; port_type=:exogenous_variables, direction=OUTPUT),
    Port(:endogenous, :Endogenous; port_type=:endogenous_variables, direction=OUTPUT),
    Port(:mechanisms, :Mechanisms; port_type=:local_functions, direction=OUTPUT),
    Port(:causal_signature, :CausalSignature; port_type=:causal_signature, direction=OUTPUT),
)

function _unique_symbols(values::Vector{Symbol}, label::AbstractString)
    length(unique(values)) == length(values) || throw(ArgumentError("$label must not contain duplicates"))
end

function _validate_mapping_pairs(pairs::Vector{Tuple{Symbol, Symbol}};
                                 source_names::Set{Symbol},
                                 target_names::Set{Symbol},
                                 label::AbstractString)
    seen_source = Set{Symbol}()
    seen_target = Set{Symbol}()
    for (src, tgt) in pairs
        src in source_names || throw(ArgumentError("$label references unknown source name $src"))
        tgt in target_names || throw(ArgumentError("$label references unknown target name $tgt"))
        src ∉ seen_source || throw(ArgumentError("$label repeats source name $src"))
        tgt ∉ seen_target || throw(ArgumentError("$label repeats target name $tgt"))
        push!(seen_source, src)
        push!(seen_target, tgt)
    end
    nothing
end

"""
    validate_scm_spec(spec)

Validate that an SCM object specification is well-formed.
"""
function validate_scm_spec(spec::SCMObjectSpec)
    _unique_symbols(spec.exogenous_variables, "Exogenous variables")
    _unique_symbols(spec.endogenous_variables, "Endogenous variables")

    targets = [f.target_variable for f in spec.local_functions]
    _unique_symbols(targets, "SCM local-function targets")
    Set(targets) == Set(spec.endogenous_variables) ||
        throw(ArgumentError("SCM local functions must define exactly one mechanism for each endogenous variable"))

    exogenous_set = Set(spec.exogenous_variables)
    endogenous_set = Set(spec.endogenous_variables)
    for fn in spec.local_functions
        fn.target_variable in endogenous_set ||
            throw(ArgumentError("Local function $(fn.name) target must be endogenous"))
        fn.target_variable ∉ fn.endogenous_parents ||
            throw(ArgumentError("Local function $(fn.name) cannot list its own target as an endogenous parent"))
        Set(fn.exogenous_parents) ⊆ exogenous_set ||
            throw(ArgumentError("Local function $(fn.name) references unknown exogenous parents"))
        Set(fn.endogenous_parents) ⊆ endogenous_set ||
            throw(ArgumentError("Local function $(fn.name) references unknown endogenous parents"))
    end
    true
end

function local_function_named(scm::SCMModelObject, name::Union{Symbol, AbstractString})
    sym = Symbol(name)
    for function_spec in scm.local_functions
        function_spec.name == sym && return function_spec
    end
    throw(KeyError("SCM $(scm.name) has no local function named $sym"))
end

function local_function_for_target(scm::SCMModelObject, variable::Union{Symbol, AbstractString})
    sym = Symbol(variable)
    for function_spec in scm.local_functions
        function_spec.target_variable == sym && return function_spec
    end
    throw(KeyError("SCM $(scm.name) has no local function targeting $sym"))
end

function _build_scm_diagram(spec::SCMObjectSpec; metadata::Dict{Symbol, Any}=Dict{Symbol, Any}())
    D = Diagram(spec.name)
    add_object!(D, :Exogenous; kind=:exogenous_variables,
                metadata=Dict(:variables => copy(spec.exogenous_variables)))
    add_object!(D, :Endogenous; kind=:endogenous_variables,
                metadata=Dict(:variables => copy(spec.endogenous_variables)))
    add_object!(D, :Mechanisms; kind=:local_functions,
                metadata=Dict(:functions => [f.name for f in spec.local_functions]))
    add_object!(D, :CausalSignature; kind=:causal_signature, metadata=copy(metadata))

    add_morphism!(D, :exogenous_signature, :Exogenous, :CausalSignature;
                  metadata=Dict(:semantic_role => :scm_exogenous_signature))
    add_morphism!(D, :endogenous_signature, :Endogenous, :CausalSignature;
                  metadata=Dict(:semantic_role => :scm_endogenous_signature))
    add_morphism!(D, :mechanism_signature, :Mechanisms, :CausalSignature;
                  metadata=Dict(:semantic_role => :scm_mechanism_signature))

    for fn in spec.local_functions
        add_morphism!(D, fn.name, :Mechanisms, :Endogenous;
                      metadata=Dict(
                          :semantic_role => :scm_local_function,
                          :target_variable => fn.target_variable,
                          :exogenous_parents => copy(fn.exogenous_parents),
                          :endogenous_parents => copy(fn.endogenous_parents),
                          :expression => fn.expression,
                      ))
    end

    for port in scm_interfaces()
        expose_port!(D, port.name, port.ref; direction=port.direction, port_type=port.port_type)
    end
    D
end

"""
    build_scm_model_object(spec; category, metadata=Dict())

Build a first-class SCM object with explicit U, V, and F data.
"""
function build_scm_model_object(spec::SCMObjectSpec; category, metadata::Dict=Dict{Symbol, Any}())
    validate_scm_spec(spec)
    D = _build_scm_diagram(spec; metadata=Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
    object = CategoricalModelObject(D;
        ambient_category=category,
        metadata=Dict{Symbol, Any}(
            :family => :SCM,
            :semantic_role => :structural_causal_model,
            :exogenous_variables => copy(spec.exogenous_variables),
            :endogenous_variables => copy(spec.endogenous_variables),
            :local_functions => [fn.name for fn in spec.local_functions],
            Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata)...,
        ))
    SCMModelObject(object, copy(spec.exogenous_variables), copy(spec.endogenous_variables), copy(spec.local_functions))
end

function _validate_scm_morphism(morphism::SCMMorphism)
    morphism.morphism.source == morphism.source_scm.name ||
        throw(ArgumentError("Base model morphism source must match the source SCM"))
    morphism.morphism.target == morphism.target_scm.name ||
        throw(ArgumentError("Base model morphism target must match the target SCM"))

    _validate_mapping_pairs(morphism.exogenous_variable_map;
        source_names=Set(morphism.source_scm.exogenous_variables),
        target_names=Set(morphism.target_scm.exogenous_variables),
        label="SCM exogenous-variable map")
    _validate_mapping_pairs(morphism.endogenous_variable_map;
        source_names=Set(morphism.source_scm.endogenous_variables),
        target_names=Set(morphism.target_scm.endogenous_variables),
        label="SCM endogenous-variable map")
    _validate_mapping_pairs(morphism.local_function_map;
        source_names=Set(fn.name for fn in morphism.source_scm.local_functions),
        target_names=Set(fn.name for fn in morphism.target_scm.local_functions),
        label="SCM local-function map")

    exogenous_map = Dict(morphism.exogenous_variable_map)
    endogenous_map = Dict(morphism.endogenous_variable_map)
    covered_target_functions = Set{Symbol}()
    for (source_name, target_name) in morphism.local_function_map
        source_function = local_function_named(morphism.source_scm, source_name)
        target_function = local_function_named(morphism.target_scm, target_name)
        push!(covered_target_functions, target_name)

        get(endogenous_map, source_function.target_variable, nothing) == target_function.target_variable ||
            throw(ArgumentError("SCM local-function transport must preserve target variables"))

        mapped_exogenous = Set(get(exogenous_map, p, nothing) for p in source_function.exogenous_parents if haskey(exogenous_map, p))
        Set(target_function.exogenous_parents) ⊆ mapped_exogenous ||
            throw(ArgumentError("SCM local-function transport must cover target exogenous parents"))

        mapped_endogenous = Set(get(endogenous_map, p, nothing) for p in source_function.endogenous_parents if haskey(endogenous_map, p))
        Set(target_function.endogenous_parents) ⊆ mapped_endogenous ||
            throw(ArgumentError("SCM local-function transport must cover target endogenous parents"))
    end

    required_target_functions = Set(local_function_for_target(morphism.target_scm, target).name
                                    for (_, target) in morphism.endogenous_variable_map)
    required_target_functions ⊆ covered_target_functions ||
        throw(ArgumentError("SCM local-function map must cover mechanisms for mapped endogenous variables"))
    true
end

"""
    build_scm_morphism(; name, source_scm, target_scm, exogenous_variable_map, endogenous_variable_map, local_function_map, metadata=Dict())

Build a structure-preserving morphism between two SCM objects.
"""
function build_scm_morphism(; name,
                             source_scm::SCMModelObject,
                             target_scm::SCMModelObject,
                             exogenous_variable_map::Vector{Tuple{Symbol, Symbol}},
                             endogenous_variable_map::Vector{Tuple{Symbol, Symbol}},
                             local_function_map::Vector{Tuple{Symbol, Symbol}},
                             metadata::Dict=Dict{Symbol, Any}())
    morphism = ModelMorphism(name, source_scm.name, target_scm.name;
        metadata=Dict{Symbol, Any}(
            :semantic_role => :scm_morphism,
            :exogenous_variable_map => copy(exogenous_variable_map),
            :endogenous_variable_map => copy(endogenous_variable_map),
            :local_function_map => copy(local_function_map),
            Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata)...,
        ))
    result = SCMMorphism(morphism, source_scm, target_scm,
                         copy(exogenous_variable_map), copy(endogenous_variable_map), copy(local_function_map))
    _validate_scm_morphism(result)
    result
end

as_model_object(value::SCMModelObject) = value.object
as_model_object(value::CategoricalModelObject) = value
as_base_morphism(value::SCMMorphism) = value.morphism
as_base_morphism(value::ModelMorphism) = value

"""
    scm_to_shared_context(scm_object, shared_context; shared_exogenous=Symbol[], shared_endogenous=Symbol[], name=nothing)

Build a signature-preserving morphism from an SCM object into a shared SCM
context.
"""
function scm_to_shared_context(scm_object::SCMModelObject, shared_context::SCMModelObject;
                               shared_exogenous::Vector{Symbol}=Symbol[],
                               shared_endogenous::Vector{Symbol}=Symbol[],
                               name=nothing)
    local_function_map = [
        (local_function_for_target(scm_object, variable).name,
         local_function_for_target(shared_context, variable).name)
        for variable in shared_endogenous
    ]
    build_scm_morphism(
        name=name === nothing ? Symbol(scm_object.name, :__to__, shared_context.name) : Symbol(name),
        source_scm=scm_object,
        target_scm=shared_context,
        exogenous_variable_map=[(var, var) for var in shared_exogenous],
        endogenous_variable_map=[(var, var) for var in shared_endogenous],
        local_function_map=local_function_map,
        metadata=Dict(
            :semantic_role => :scm_signature_projection,
            :shared_exogenous => copy(shared_exogenous),
            :shared_endogenous => copy(shared_endogenous),
        ),
    )
end

"""
    compose_scm_pullback(; name, left_scm, right_scm, shared_context, left_to_context, right_to_context)

Compose two SCM objects by pullback over a shared SCM context.
"""
function compose_scm_pullback(; name,
                              left_scm::Union{CategoricalModelObject, SCMModelObject},
                              right_scm::Union{CategoricalModelObject, SCMModelObject},
                              shared_context::Union{CategoricalModelObject, SCMModelObject},
                              left_to_context::Union{ModelMorphism, SCMMorphism},
                              right_to_context::Union{ModelMorphism, SCMMorphism})
    left_obj = as_model_object(left_scm)
    right_obj = as_model_object(right_scm)
    shared_obj = as_model_object(shared_context)
    pullback(to_diagram(left_obj), to_diagram(right_obj);
             over=shared_obj.name,
             name=name,
             metadata=Dict(
                 :family => :SCM,
                 :semantic_role => :scm_pullback_composition,
                 :left_to_context => as_base_morphism(left_to_context).name,
                 :right_to_context => as_base_morphism(right_to_context).name,
             ))
end

default_omega_truth_values() = [
    SCMTruthValue(:bottom, "predicate is refuted or unsatisfied"),
    SCMTruthValue(:admissible, "predicate is admissible in the current SCM context"),
    SCMTruthValue(:compatible, "predicate is compatibility-preserving over a shared interface"),
    SCMTruthValue(:invariant, "predicate is stable under the intended transport or intervention"),
    SCMTruthValue(:top, "predicate is satisfied"),
]

function truth_value_named(ω::OmegaSCM, name::Union{Symbol, AbstractString})
    sym = Symbol(name)
    for value in ω.truth_values
        value.name == sym && return value
    end
    throw(KeyError("Omega object $(ω.name) has no truth value named $sym"))
end

"""
    build_scm_monomorphism(; name, constrained_scm, ambient_scm, metadata=Dict())

Build a placeholder monomorphism for a constrained SCM inclusion.
"""
function build_scm_monomorphism(; name,
                                 constrained_scm::SCMModelObject,
                                 ambient_scm::SCMModelObject,
                                 metadata::Dict=Dict{Symbol, Any}())
    constrained_scm.category == ambient_scm.category ||
        throw(ArgumentError("SCM monomorphism objects must share a category"))
    constrained_scm.exogenous_variables == ambient_scm.exogenous_variables ||
        throw(ArgumentError("Current SCM monomorphism scaffold expects matching exogenous signatures"))
    constrained_scm.endogenous_variables == ambient_scm.endogenous_variables ||
        throw(ArgumentError("Current SCM monomorphism scaffold expects matching endogenous signatures"))
    [fn.target_variable for fn in constrained_scm.local_functions] ==
        [fn.target_variable for fn in ambient_scm.local_functions] ||
        throw(ArgumentError("Current SCM monomorphism scaffold expects matching local-function targets"))

    morphism = build_scm_morphism(
        name=Symbol(name),
        source_scm=constrained_scm,
        target_scm=ambient_scm,
        exogenous_variable_map=[(name, name) for name in constrained_scm.exogenous_variables],
        endogenous_variable_map=[(name, name) for name in constrained_scm.endogenous_variables],
        local_function_map=[(fn.name, fn.name) for fn in constrained_scm.local_functions],
        metadata=merge(Dict{Symbol, Any}(:semantic_role => :scm_monomorphism), metadata),
    )
    SCMMonomorphism(morphism, ambient_scm, Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    build_scm_subobject(; name, ambient_scm, clauses, metadata=Dict())

Build a constrained SCM subobject over an ambient SCM.
"""
function build_scm_subobject(; name,
                              ambient_scm::SCMModelObject,
                              clauses::Vector{SCMPredicateClause},
                              metadata::Dict=Dict{Symbol, Any}())
    constrained = build_scm_model_object(
        SCMObjectSpec(Symbol(name, :__object),
                      copy(ambient_scm.exogenous_variables),
                      copy(ambient_scm.endogenous_variables),
                      copy(ambient_scm.local_functions));
        category=ambient_scm.category,
        metadata=merge(Dict{Symbol, Any}(
            :semantic_role => :scm_subobject,
            :predicate_clauses => [clause.name for clause in clauses],
        ), metadata),
    )
    inclusion = build_scm_monomorphism(
        name=Symbol(name, :__inclusion),
        constrained_scm=constrained,
        ambient_scm=ambient_scm,
        metadata=merge(Dict{Symbol, Any}(
            :predicate_clauses => [clause.name for clause in clauses],
        ), metadata),
    )
    SCMSubobject(Symbol(name), constrained, ambient_scm, inclusion, copy(clauses),
                 Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    build_scm_predicate(; name, ambient_scm, clauses, metadata=Dict())

Build an internal predicate over an ambient SCM via a constrained subobject.
"""
function build_scm_predicate(; name,
                              ambient_scm::SCMModelObject,
                              clauses::Vector{SCMPredicateClause},
                              metadata::Dict=Dict{Symbol, Any}())
    subobject = build_scm_subobject(name=Symbol(name, :__subobject),
                                    ambient_scm=ambient_scm,
                                    clauses=clauses,
                                    metadata=metadata)
    SCMPredicate(Symbol(name), ambient_scm, subobject, copy(clauses),
                 Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    build_omega_scm(; category, name=:Omega_SCM, truth_values=default_omega_truth_values(), metadata=Dict())

Build a provisional truth object for SCM predicates.
"""
function build_omega_scm(; category,
                          name::Union{Symbol, AbstractString}=:Omega_SCM,
                          truth_values::Vector{SCMTruthValue}=default_omega_truth_values(),
                          metadata::Dict=Dict{Symbol, Any}())
    D = Diagram(name)
    add_object!(D, :TruthValues; kind=:intuitionistic_truth_space,
                metadata=Dict(:truth_values => [value.name for value in truth_values]))
    add_object!(D, :ClassifiedPredicates; kind=:predicate_classifier)
    expose_port!(D, :truth_values, :TruthValues; direction=OUTPUT, port_type=:intuitionistic_truth_space)
    expose_port!(D, :classified_predicates, :ClassifiedPredicates; direction=OUTPUT, port_type=:predicate_classifier)

    object = CategoricalModelObject(D;
        ambient_category=category,
        metadata=merge(Dict{Symbol, Any}(
            :semantic_role => :omega_scm,
            :truth_values => [value.name for value in truth_values],
        ), Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata)))
    OmegaSCM(object, copy(truth_values), Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    build_scm_characteristic_map(; name, ambient_scm, predicate, omega, classifying_truth_value=:top, false_truth_value=:bottom, metadata=Dict())

Build a characteristic map classifying an SCM predicate into `OmegaSCM`.
"""
function build_scm_characteristic_map(; name,
                                       ambient_scm::SCMModelObject,
                                       predicate::SCMPredicate,
                                       omega::OmegaSCM,
                                       classifying_truth_value::Union{Symbol, AbstractString}=:top,
                                       false_truth_value::Union{Symbol, AbstractString}=:bottom,
                                       metadata::Dict=Dict{Symbol, Any}())
    ambient_scm.category == omega.category ||
        throw(ArgumentError("SCM characteristic map requires the ambient SCM and Omega to share a category"))
    predicate.ambient_scm == ambient_scm ||
        throw(ArgumentError("SCM characteristic map predicate must live over the ambient SCM"))

    classifying_value = truth_value_named(omega, classifying_truth_value)
    false_value = truth_value_named(omega, false_truth_value)
    morphism = ModelMorphism(name, ambient_scm.name, omega.name;
        metadata=merge(Dict{Symbol, Any}(
            :semantic_role => :scm_characteristic_map,
            :predicate => predicate.name,
            :classifying_truth_value => classifying_value.name,
            :false_truth_value => false_value.name,
        ), Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata)))
    SCMCharacteristicMap(morphism, ambient_scm, predicate, omega, classifying_value, false_value,
                         Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    conjoin_scm_predicates(; name, left, right, metadata=Dict())

Conjoin two SCM predicates over the same ambient SCM.
"""
function conjoin_scm_predicates(; name,
                                 left::SCMPredicate,
                                 right::SCMPredicate,
                                 metadata::Dict=Dict{Symbol, Any}())
    left.ambient_scm == right.ambient_scm ||
        throw(ArgumentError("SCM predicate conjunction requires a shared ambient SCM"))
    build_scm_predicate(
        name=name,
        ambient_scm=left.ambient_scm,
        clauses=vcat(left.clauses, right.clauses),
        metadata=merge(Dict{Symbol, Any}(
            :logical_connective => :and,
            :left_predicate => left.name,
            :right_predicate => right.name,
        ), metadata),
    )
end

"""
    scm_subobject_classifier(omega::OmegaSCM) -> SubobjectClassifier

Bridge the SCM truth object into the generic topos-level subobject classifier.
"""
function scm_subobject_classifier(omega::OmegaSCM)
    SubobjectClassifier(Symbol(omega.name, :__classifier);
                        truth_object=omega.name,
                        true_map=:true_map,
                        truth_values=Set(value.name for value in omega.truth_values),
                        metadata=Dict(:family => :SCM, :omega => omega.name))
end

"""
    as_internal_predicate(map::SCMCharacteristicMap) -> InternalPredicate

Convert an SCM characteristic map into the generic topos-level internal
predicate representation.
"""
function as_internal_predicate(map::SCMCharacteristicMap)
    classifier = scm_subobject_classifier(map.omega)
    characteristic = function(value)
        if value isa Symbol
            value in classifier.truth_values && return value
            return value == map.classifying_truth_value.name ? value : map.false_truth_value.name
        elseif value isa Bool
            return value ? map.classifying_truth_value.name : map.false_truth_value.name
        else
            return value ? map.classifying_truth_value.name : map.false_truth_value.name
        end
    end
    InternalPredicate(map.predicate.name, classifier;
                      characteristic_map=characteristic,
                      metadata=Dict(
                          :family => :SCM,
                          :predicate => map.predicate.name,
                          :characteristic_map => map.name,
                      ))
end
