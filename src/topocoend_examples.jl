# ============================================================================
# topocoend_examples.jl — Concrete TopoCoend learned-cover examples
# ============================================================================

function _payload_lookup(payload, key::Symbol, default=nothing)
    if payload isa AbstractDict
        if haskey(payload, key)
            return payload[key]
        end
        string_key = String(key)
        return haskey(payload, string_key) ? payload[string_key] : default
    elseif payload isa NamedTuple
        return hasproperty(payload, key) ? getproperty(payload, key) : default
    else
        return hasproperty(payload, key) ? getproperty(payload, key) : default
    end
end

function _string_set(value)::Set{String}
    result = Set{String}()
    _string_set!(result, value)
    result
end

_string_set!(result::Set{String}, value::Nothing) = result

function _string_set!(result::Set{String}, value::Union{AbstractString, Symbol})
    push!(result, String(value))
    result
end

function _string_set!(result::Set{String}, value::Union{AbstractVector, Tuple, AbstractSet})
    for item in value
        _string_set!(result, item)
    end
    result
end

function infer_topocoend_cover(token_payloads::AbstractDict;
                               context_prototypes::AbstractDict,
                               min_overlap::Integer=1)
    min_overlap >= 1 || error("min_overlap must be at least 1")
    relation = Dict{Any, Vector{Any}}()
    for (context, prototype) in context_prototypes
        prototype_tags = _string_set(prototype)
        members = Any[]
        for (token, payload) in token_payloads
            tags = _string_set(_payload_lookup(payload, :tags, String[]))
            length(intersect(tags, prototype_tags)) >= min_overlap || continue
            push!(members, token)
        end
        isempty(members) || (relation[context] = members)
    end
    relation
end

function lift_topocoend_scores(token_payloads::AbstractDict; score_key::Symbol=:priority)
    Dict(
        token => begin
            score = _payload_lookup(payload, score_key, nothing)
            score isa Real || error("Expected token $(repr(token)) to carry a numeric $(repr(score_key)) field")
            Float64(score)
        end
        for (token, payload) in token_payloads
    )
end

"""
    build_topocoend_triage_example()

Construct a concrete TopoCoend example in which local clinical signals induce a
learned cover over latent triage contexts before Kan aggregation produces
context-level risk summaries.
"""
function build_topocoend_triage_example()
    token_payloads = Dict(
        :chief_complaint => (text="shortness of breath", tags=["respiratory", "symptom", "infection"], priority=0.95),
        :oxygen_saturation => (text="SpO₂ 88%", tags=["respiratory", "vitals", "oxygen"], priority=0.90),
        :blood_pressure => (text="BP 92/60", tags=["cardio", "vitals", "circulation"], priority=0.55),
        :glucose => (text="glucose 245", tags=["metabolic", "labs"], priority=0.70),
        :history => (text="recent pneumonia", tags=["respiratory", "history"], priority=0.60),
        :lactate => (text="lactate 3.8", tags=["metabolic", "labs", "circulation"], priority=0.85),
    )
    context_prototypes = Dict(
        :respiratory_focus => Set(["respiratory", "oxygen", "infection", "symptom"]),
        :hemodynamic_focus => Set(["cardio", "vitals", "circulation"]),
        :metabolic_focus => Set(["metabolic", "labs"]),
    )

    config = TopoCoendConfig(
        name=:TopoCoendTriage,
        token_object=:ClinicalSignals,
        neighborhood_object=:LearnedCover,
        local_object=:SignalPriority,
        target_object=:ContextRisk,
        infer_neighborhood_name=:infer_context_cover,
        lift_name=:lift_signal_priority,
        aggregate_name=:aggregate_context_risk,
        reducer=:mean,
    )

    diagram = topocoend_block(;
        config,
        infer_neighborhood_impl=tokens -> infer_topocoend_cover(tokens;
            context_prototypes=context_prototypes,
            min_overlap=1),
        lift_impl=tokens -> lift_topocoend_scores(tokens; score_key=:priority),
    )

    Dict{Symbol, Any}(
        :diagram => diagram,
        :token_payloads => token_payloads,
        :context_prototypes => context_prototypes,
        :metadata => Dict(
            :domain => "clinical_triage",
            :min_overlap => 1,
            :score_key => :priority,
        ),
    )
end

function execute_topocoend_triage_example(example::Union{Nothing, Dict{Symbol, Any}}=nothing)
    example = example === nothing ? build_topocoend_triage_example() : example
    diagram = example[:diagram]
    run(diagram, Dict(
        diagram.ports[:input].ref => example[:token_payloads],
    ))
end

function summarize_topocoend_triage_example(example::Union{Nothing, Dict{Symbol, Any}}=nothing)
    example = example === nothing ? build_topocoend_triage_example() : example
    diagram = example[:diagram]
    executed = execute_topocoend_triage_example(example)
    lifted_scores = lift_topocoend_scores(example[:token_payloads];
        score_key=example[:metadata][:score_key])
    learned_relation = executed.values[diagram.ports[:learned_relation].ref]
    context_risks = Dict(
        String(context) => Float64(score)
        for (context, score) in executed.values[diagram.ports[:output].ref]
    )
    ranked_contexts = sort(collect(context_risks); by=last, rev=true)

    Dict(
        "domain" => String(example[:metadata][:domain]),
        "counts" => Dict(
            "signals" => length(example[:token_payloads]),
            "contexts" => length(learned_relation),
            "cover_edges" => sum(length(members) for members in values(learned_relation)),
        ),
        "context_prototypes" => Dict(
            String(context) => sort(collect(_string_set(prototype)))
            for (context, prototype) in example[:context_prototypes]
        ),
        "learned_relation" => Dict(
            String(context) => sort(String.(members))
            for (context, members) in learned_relation
        ),
        "signal_priorities" => Dict(
            String(token) => lifted_scores[token]
            for token in sort(collect(keys(lifted_scores)); by=String)
        ),
        "context_risks" => context_risks,
        "highest_priority_context" => first(ranked_contexts).first,
    )
end
