# ============================================================================
# scm_examples.jl — Canonical SCM parity examples
# ============================================================================

"""
    build_transport_scm_pullback_example() -> Dict{Symbol,Any}

Build a small pullback of two SCMs over a shared transport SCM interface.
"""
function build_transport_scm_pullback_example()
    category = :StructuralCausalModels

    shared_scm = build_scm_model_object(
        SCMObjectSpec(:TrafficWeatherSCM,
            [:u_weather],
            [:traffic],
            [
                SCMLocalFunctionSpec(:f_traffic_shared, :traffic;
                    exogenous_parents=[:u_weather],
                    expression="traffic := traffic_from_weather(u_weather)")
            ]);
        category=category,
    )

    left_scm = build_scm_model_object(
        SCMObjectSpec(:CommuteSCM,
            [:u_weather, :u_event],
            [:traffic, :commute_time],
            [
                SCMLocalFunctionSpec(:f_traffic_commute, :traffic;
                    exogenous_parents=[:u_weather, :u_event],
                    expression="traffic := traffic_from_weather_and_event(u_weather, u_event)"),
                SCMLocalFunctionSpec(:f_commute_time, :commute_time;
                    exogenous_parents=[:u_weather],
                    endogenous_parents=[:traffic],
                    expression="commute_time := commute_from_weather_and_traffic(u_weather, traffic)"),
            ]);
        category=category,
    )

    right_scm = build_scm_model_object(
        SCMObjectSpec(:SafetySCM,
            [:u_weather, :u_incident],
            [:traffic, :delay_risk],
            [
                SCMLocalFunctionSpec(:f_traffic_safety, :traffic;
                    exogenous_parents=[:u_weather, :u_incident],
                    expression="traffic := traffic_from_weather_and_incident(u_weather, u_incident)"),
                SCMLocalFunctionSpec(:f_delay_risk, :delay_risk;
                    exogenous_parents=[:u_weather],
                    endogenous_parents=[:traffic],
                    expression="delay_risk := risk_from_weather_and_traffic(u_weather, traffic)"),
            ]);
        category=category,
    )

    left_to_shared = scm_to_shared_context(left_scm, shared_scm;
                                           shared_exogenous=[:u_weather],
                                           shared_endogenous=[:traffic])
    right_to_shared = scm_to_shared_context(right_scm, shared_scm;
                                            shared_exogenous=[:u_weather],
                                            shared_endogenous=[:traffic])

    pullback_result = compose_scm_pullback(
        name=:IntegratedTransportSCM,
        left_scm=left_scm,
        right_scm=right_scm,
        shared_context=shared_scm,
        left_to_context=left_to_shared,
        right_to_context=right_to_shared,
    )

    Dict{Symbol, Any}(
        :category => category,
        :shared_scm => shared_scm,
        :left_scm => left_scm,
        :right_scm => right_scm,
        :left_to_shared => left_to_shared,
        :right_to_shared => right_to_shared,
        :pullback => pullback_result,
    )
end

function build_transport_scm_pullback_proof_bundle(example::Union{Nothing, Dict{Symbol, Any}}=nothing)
    example = example === nothing ? build_transport_scm_pullback_example() : example
    bundle_proof_shapes(:IntegratedTransportSCMProofBundle,
                        prove_pullback_shape(example[:pullback]).claim)
end

function build_transport_scm_pullback_compilation_plan(example::Union{Nothing, Dict{Symbol, Any}}=nothing)
    example = example === nothing ? build_transport_scm_pullback_example() : example
    compile_plan(:IntegratedTransportSCMCompilationPlan,
                 example[:shared_scm],
                 example[:left_scm],
                 example[:right_scm],
                 example[:left_to_shared],
                 example[:right_to_shared],
                 example[:pullback];
                 metadata=Dict(:example => "transport_scm_pullback"))
end

build_transport_scm_pullback_executable_ir(example::Union{Nothing, Dict{Symbol, Any}}=nothing) =
    lower_plan_to_executable_ir(build_transport_scm_pullback_compilation_plan(example))

execute_transport_scm_pullback_example(example::Union{Nothing, Dict{Symbol, Any}}=nothing) =
    execute_placeholder_ir(build_transport_scm_pullback_executable_ir(example))

"""
    build_transport_scm_predicate_example() -> Dict{Symbol,Any}

Build a first SCM internal-logic predicate example over the shared transport
SCM.
"""
function build_transport_scm_predicate_example(example::Union{Nothing, Dict{Symbol, Any}}=nothing)
    example = example === nothing ? build_transport_scm_pullback_example() : example
    ambient_scm = example[:shared_scm]

    monotonicity = build_scm_predicate(
        name=:TrafficMonotonicity,
        ambient_scm=ambient_scm,
        clauses=[
            SCMPredicateClause(:traffic_weather_monotone,
                               "traffic is monotone in weather severity";
                               clause_kind=:monotonicity)
        ],
        metadata=Dict(:logical_role => :predicate),
    )
    consistency = build_scm_predicate(
        name=:TrafficObservationalConsistency,
        ambient_scm=ambient_scm,
        clauses=[
            SCMPredicateClause(:traffic_observational_consistency,
                               "traffic respects the shared observational interface";
                               clause_kind=:consistency)
        ],
        metadata=Dict(:logical_role => :predicate),
    )
    conjunction = conjoin_scm_predicates(
        name=:TrafficMonotoneAndConsistent,
        left=monotonicity,
        right=consistency,
        metadata=Dict(:logical_role => :conjunction),
    )

    Dict{Symbol, Any}(
        :ambient_scm => ambient_scm,
        :monotonicity_predicate => monotonicity,
        :consistency_predicate => consistency,
        :conjunction_predicate => conjunction,
    )
end

function build_transport_scm_predicate_compilation_plan(example::Union{Nothing, Dict{Symbol, Any}}=nothing)
    example = example === nothing ? build_transport_scm_predicate_example() : example
    monotonicity = example[:monotonicity_predicate]
    consistency = example[:consistency_predicate]
    conjunction = example[:conjunction_predicate]
    compile_plan(:TransportSCMPredicateCompilationPlan,
                 example[:ambient_scm],
                 monotonicity.subobject.object_scm,
                 monotonicity.subobject.inclusion,
                 monotonicity.subobject,
                 monotonicity,
                 consistency.subobject.object_scm,
                 consistency.subobject.inclusion,
                 consistency.subobject,
                 consistency,
                 conjunction.subobject.object_scm,
                 conjunction.subobject.inclusion,
                 conjunction.subobject,
                 conjunction;
                 metadata=Dict(:example => "transport_scm_predicates"))
end

build_transport_scm_predicate_executable_ir(example::Union{Nothing, Dict{Symbol, Any}}=nothing) =
    lower_plan_to_executable_ir(build_transport_scm_predicate_compilation_plan(example))

execute_transport_scm_predicate_example(example::Union{Nothing, Dict{Symbol, Any}}=nothing) =
    execute_placeholder_ir(build_transport_scm_predicate_executable_ir(example))

"""
    build_transport_scm_omega_example() -> Dict{Symbol,Any}

Build an `OmegaSCM` example classifying transport SCM predicates.
"""
function build_transport_scm_omega_example(example::Union{Nothing, Dict{Symbol, Any}}=nothing)
    predicate_example = example === nothing ? build_transport_scm_predicate_example() : example
    ambient_scm = predicate_example[:ambient_scm]
    monotonicity = predicate_example[:monotonicity_predicate]
    consistency = predicate_example[:consistency_predicate]
    conjunction = predicate_example[:conjunction_predicate]

    omega = build_omega_scm(category=ambient_scm.category, metadata=Dict(:logical_role => :truth_object))
    monotonicity_classifier = build_scm_characteristic_map(
        name=:chi_TrafficMonotonicity,
        ambient_scm=ambient_scm,
        predicate=monotonicity,
        omega=omega,
        classifying_truth_value=:top,
        false_truth_value=:bottom,
        metadata=Dict(:classified_predicate => monotonicity.name),
    )
    consistency_classifier = build_scm_characteristic_map(
        name=:chi_TrafficObservationalConsistency,
        ambient_scm=ambient_scm,
        predicate=consistency,
        omega=omega,
        classifying_truth_value=:compatible,
        false_truth_value=:bottom,
        metadata=Dict(:classified_predicate => consistency.name),
    )
    conjunction_classifier = build_scm_characteristic_map(
        name=:chi_TrafficMonotoneAndConsistent,
        ambient_scm=ambient_scm,
        predicate=conjunction,
        omega=omega,
        classifying_truth_value=:compatible,
        false_truth_value=:bottom,
        metadata=Dict(:classified_predicate => conjunction.name),
    )

    result = copy(predicate_example)
    result[:omega] = omega
    result[:monotonicity_classifier] = monotonicity_classifier
    result[:consistency_classifier] = consistency_classifier
    result[:conjunction_classifier] = conjunction_classifier
    result
end

function build_transport_scm_omega_compilation_plan(example::Union{Nothing, Dict{Symbol, Any}}=nothing)
    example = example === nothing ? build_transport_scm_omega_example() : example
    monotonicity = example[:monotonicity_predicate]
    consistency = example[:consistency_predicate]
    conjunction = example[:conjunction_predicate]
    compile_plan(:TransportSCMOmegaCompilationPlan,
                 example[:ambient_scm],
                 example[:omega],
                 monotonicity.subobject.object_scm,
                 monotonicity.subobject.inclusion,
                 monotonicity.subobject,
                 monotonicity,
                 example[:monotonicity_classifier],
                 consistency.subobject.object_scm,
                 consistency.subobject.inclusion,
                 consistency.subobject,
                 consistency,
                 example[:consistency_classifier],
                 conjunction.subobject.object_scm,
                 conjunction.subobject.inclusion,
                 conjunction.subobject,
                 conjunction,
                 example[:conjunction_classifier];
                 metadata=Dict(:example => "transport_scm_omega"))
end

build_transport_scm_omega_executable_ir(example::Union{Nothing, Dict{Symbol, Any}}=nothing) =
    lower_plan_to_executable_ir(build_transport_scm_omega_compilation_plan(example))

execute_transport_scm_omega_example(example::Union{Nothing, Dict{Symbol, Any}}=nothing) =
    execute_placeholder_ir(build_transport_scm_omega_executable_ir(example))
