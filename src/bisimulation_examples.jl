# ============================================================================
# bisimulation_examples.jl — Concrete behavioral quotient examples
# ============================================================================

_priority_bucket(score::Real) = score >= 0.85 ? 2.0 : score >= 0.55 ? 1.0 : 0.0
_bool_flag(value) = value ? 1.0 : 0.0

function _left_bisimulation_behavior(state)
    reroute = _bool_flag(Bool(_payload_lookup(state, :blocked, false)))
    priority = _priority_bucket(Float64(_payload_lookup(state, :acuity, 0.0)))
    (reroute, priority)
end

function _right_bisimulation_behavior(state)
    reroute = _bool_flag(Bool(_payload_lookup(state, :reroute, false)))
    priority = _priority_bucket(Float64(_payload_lookup(state, :alert_level, 0.0)))
    (reroute, priority)
end

function _project_left_states(relation_pairs::AbstractDict)
    Dict(pair => _payload_lookup(payload, :left) for (pair, payload) in relation_pairs)
end

function _project_right_states(relation_pairs::AbstractDict)
    Dict(pair => _payload_lookup(payload, :right) for (pair, payload) in relation_pairs)
end

function _observe_left_behavior(states::AbstractDict)
    Dict(pair => _left_bisimulation_behavior(state) for (pair, state) in states)
end

function _observe_right_behavior(states::AbstractDict)
    Dict(pair => _right_bisimulation_behavior(state) for (pair, state) in states)
end

function _behavioral_quotient_codes(behaviors::AbstractDict)
    Dict(pair => signature[2] * 10.0 + signature[1] for (pair, signature) in behaviors)
end

function _behavior_label(code::Real)
    if code ≈ 21.0
        "acute_reroute"
    elseif code ≈ 10.0
        "watchful_recovery"
    elseif code ≈ 0.0
        "steady_progress"
    else
        string("class_", Int(round(code)))
    end
end

"""
    build_bisimulation_quotient_example()

Construct a concrete behavioral-quotient example with two distinct latent state
spaces that induce the same observable control policy on paired states. The
resulting coequalizer quotient collapses those paired states into shared
behavior classes.
"""
function build_bisimulation_quotient_example()
    relation_pairs = Dict(
        :acute_reroute => (
            left=(acuity=0.92, blocked=true, ward=:respiratory),
            right=(alert_level=0.93, reroute=true, queue=:red),
        ),
        :watchful_recovery => (
            left=(acuity=0.63, blocked=false, ward=:stepdown),
            right=(alert_level=0.68, reroute=false, queue=:amber),
        ),
        :steady_progress => (
            left=(acuity=0.30, blocked=false, ward=:routine),
            right=(alert_level=0.28, reroute=false, queue=:green),
        ),
    )

    config = BisimulationQuotientConfig(
        name=:BisimulationClinicalQuotient,
        relation_object=:PairedLatentStates,
        state_a_object=:ControllerAState,
        state_b_object=:ControllerBState,
        behavior_object=:SharedBehavior,
        left_projection=:project_controller_a,
        right_projection=:project_controller_b,
        observe_a=:observe_controller_a,
        observe_b=:observe_controller_b,
        left_path=:controller_a_behavior,
        right_path=:controller_b_behavior,
        quotient_name=:behavioral_class,
    )

    diagram = bisimulation_quotient_block(;
        config,
        left_projection_impl=_project_left_states,
        right_projection_impl=_project_right_states,
        observe_a_impl=_observe_left_behavior,
        observe_b_impl=_observe_right_behavior,
    )

    quotient_map = diagram.ports[:output].ref
    bind_morphism!(diagram, quotient_map, _behavioral_quotient_codes)
    add_coalgebra!(diagram, :controller_a;
        state=Symbol(:base__, config.state_a_object),
        transition=Symbol(:base__, config.observe_a),
        functor_type=:observation,
        description="Observable behavior coalgebra for controller A")
    add_coalgebra!(diagram, :controller_b;
        state=Symbol(:base__, config.state_b_object),
        transition=Symbol(:base__, config.observe_b),
        functor_type=:observation,
        description="Observable behavior coalgebra for controller B")
    add_bisimulation!(diagram, :controller_alignment;
        coalgebra_a=:controller_a,
        coalgebra_b=:controller_b,
        relation=diagram.ports[:relation].ref,
        description="Paired controller states with identical observable routing behavior")

    Dict{Symbol, Any}(
        :diagram => diagram,
        :relation_pairs => relation_pairs,
        :metadata => Dict(
            :domain => "clinical_controller_alignment",
            :quotient_name => config.quotient_name,
            :left_quotient_name => Symbol(config.quotient_name, :_qf),
            :right_quotient_name => Symbol(config.quotient_name, :_qg),
            :loss_name => Symbol(config.quotient_name, :_coeq_loss),
        ),
    )
end

function execute_bisimulation_quotient_example(example::Union{Nothing, Dict{Symbol, Any}}=nothing)
    example = example === nothing ? build_bisimulation_quotient_example() : example
    diagram = example[:diagram]
    run(diagram, Dict(
        diagram.ports[:relation].ref => example[:relation_pairs],
    ))
end

function summarize_bisimulation_quotient_example(example::Union{Nothing, Dict{Symbol, Any}}=nothing)
    example = example === nothing ? build_bisimulation_quotient_example() : example
    diagram = example[:diagram]
    executed = execute_bisimulation_quotient_example(example)
    left_behaviors = executed.values[diagram.ports[:left_behavior].ref]
    right_behaviors = executed.values[diagram.ports[:right_behavior].ref]
    quotient_codes = executed.values[diagram.ports[:output].ref]
    bisimulations = get_bisimulations(diagram)

    Dict(
        "domain" => String(example[:metadata][:domain]),
        "counts" => Dict(
            "paired_states" => length(example[:relation_pairs]),
            "quotient_classes" => length(Set(values(quotient_codes))),
            "declared_bisimulations" => length(bisimulations),
        ),
        "left_behaviors" => Dict(
            String(pair) => collect(signature)
            for (pair, signature) in left_behaviors
        ),
        "right_behaviors" => Dict(
            String(pair) => collect(signature)
            for (pair, signature) in right_behaviors
        ),
        "quotient_codes" => Dict(
            String(pair) => Float64(code)
            for (pair, code) in quotient_codes
        ),
        "quotient_labels" => Dict(
            String(pair) => _behavior_label(code)
            for (pair, code) in quotient_codes
        ),
        "declared_bisimulations" => sort(String.(collect(keys(bisimulations)))),
        "coequalizer_loss" => executed.losses[example[:metadata][:loss_name]],
    )
end
