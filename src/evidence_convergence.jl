# ============================================================================
# evidence_convergence.jl — Evidence convergence control for agentic workflows
# ============================================================================

abstract type AbstractEvidenceConvergenceAdapter end

struct EvidenceConvergencePolicy
    min_evidence::Int
    stability_threshold::Float64
    required_stable_passes::Int
    max_evidence::Int
    metadata::Dict{Symbol, Any}
end

function EvidenceConvergencePolicy(min_evidence::Integer;
                                   stability_threshold=1.0,
                                   required_stable_passes::Integer=1,
                                   max_evidence::Integer=0,
                                   metadata::Dict=Dict{Symbol, Any}())
    min_evidence >= 1 || throw(ArgumentError("min_evidence must be at least 1"))
    required_stable_passes >= 1 || throw(ArgumentError("required_stable_passes must be at least 1"))
    max_evidence >= 0 || throw(ArgumentError("max_evidence must be non-negative"))
    EvidenceConvergencePolicy(
        Int(min_evidence),
        Float64(stability_threshold),
        Int(required_stable_passes),
        Int(max_evidence),
        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata),
    )
end

struct FunctionEvidenceConvergenceAdapter{F, G} <: AbstractEvidenceConvergenceAdapter
    similarity_fn::F
    describe_fn::G
end

FunctionEvidenceConvergenceAdapter(similarity_fn; describe_fn=snapshot -> string(snapshot)) =
    FunctionEvidenceConvergenceAdapter(similarity_fn, describe_fn)

convergence_similarity(adapter::FunctionEvidenceConvergenceAdapter, previous, current; policy::EvidenceConvergencePolicy) =
    Float64(adapter.similarity_fn(previous, current; policy=policy))

convergence_description(adapter::FunctionEvidenceConvergenceAdapter, snapshot) =
    String(adapter.describe_fn(snapshot))

struct EvidenceConvergenceAssessment{T}
    snapshot::T
    evidence_count::Int
    iteration::Int
    similarity::Union{Nothing, Float64}
    comparable::Bool
    stability_threshold::Float64
    stable_passes::Int
    required_stable_passes::Int
    remaining_stable_passes::Int
    min_evidence_remaining::Int
    stable::Bool
    stop::Bool
    stop_trigger::String
    reason::String
end

mutable struct EvidenceConvergenceTracker{A<:AbstractEvidenceConvergenceAdapter}
    policy::EvidenceConvergencePolicy
    adapter::A
    previous_snapshot::Any
    last_assessment::Union{Nothing, EvidenceConvergenceAssessment}
    stable_passes::Int
    iteration::Int
end

EvidenceConvergenceTracker(; policy::EvidenceConvergencePolicy, adapter::A) where {A<:AbstractEvidenceConvergenceAdapter} =
    EvidenceConvergenceTracker{A}(policy, adapter, nothing, nothing, 0, 0)

function last_assessment(tracker::EvidenceConvergenceTracker)
    tracker.last_assessment === nothing && throw(ArgumentError("No convergence assessment has been recorded yet"))
    tracker.last_assessment
end

function assess!(tracker::EvidenceConvergenceTracker, snapshot; evidence_count::Integer)
    tracker.iteration += 1
    similarity = nothing
    comparable = false
    stable = false
    required_stable_passes = max(1, tracker.policy.required_stable_passes)
    evidence_count = Int(evidence_count)
    min_evidence_remaining = max(0, tracker.policy.min_evidence - evidence_count)

    if evidence_count >= tracker.policy.min_evidence && tracker.previous_snapshot !== nothing
        comparable = true
        similarity = convergence_similarity(tracker.adapter, tracker.previous_snapshot, snapshot; policy=tracker.policy)
        stable = similarity >= tracker.policy.stability_threshold
    end

    if stable
        tracker.stable_passes += 1
    else
        tracker.stable_passes = 0
    end

    stop = false
    stop_trigger = "pending"
    if evidence_count >= tracker.policy.min_evidence && tracker.stable_passes >= required_stable_passes
        stop = true
        stop_trigger = "stability"
        reason = "Evidence stabilized after $(evidence_count) items (similarity=$(round(similarity; digits=3))); $(convergence_description(tracker.adapter, snapshot))"
    elseif tracker.policy.max_evidence > 0 && evidence_count >= tracker.policy.max_evidence
        stop = true
        stop_trigger = "max_evidence"
        reason = "Reached max evidence budget of $(tracker.policy.max_evidence); $(convergence_description(tracker.adapter, snapshot))"
    elseif evidence_count < tracker.policy.min_evidence
        stop_trigger = "min_evidence"
        remaining = tracker.policy.min_evidence - evidence_count
        reason = "Need $(remaining) more evidence item(s) before convergence checks; $(convergence_description(tracker.adapter, snapshot))"
    elseif similarity === nothing
        stop_trigger = "baseline_pending"
        reason = "Need one more post-floor update before judging stability; $(convergence_description(tracker.adapter, snapshot))"
    else
        stop_trigger = "stability_pending"
        reason = "Evidence not yet stable (similarity=$(round(similarity; digits=3)), threshold=$(round(tracker.policy.stability_threshold; digits=3))); $(convergence_description(tracker.adapter, snapshot))"
    end

    remaining_stable_passes = stop ? 0 : max(0, required_stable_passes - tracker.stable_passes)
    tracker.previous_snapshot = snapshot
    assessment = EvidenceConvergenceAssessment(
        snapshot,
        evidence_count,
        tracker.iteration,
        similarity,
        comparable,
        tracker.policy.stability_threshold,
        tracker.stable_passes,
        required_stable_passes,
        remaining_stable_passes,
        min_evidence_remaining,
        stable,
        stop,
        stop_trigger,
        reason,
    )
    tracker.last_assessment = assessment
    assessment
end

function as_dict(policy::EvidenceConvergencePolicy)
    Dict(
        "min_evidence" => policy.min_evidence,
        "stability_threshold" => policy.stability_threshold,
        "required_stable_passes" => policy.required_stable_passes,
        "max_evidence" => policy.max_evidence,
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in policy.metadata),
    )
end

function as_dict(assessment::EvidenceConvergenceAssessment)
    Dict(
        "snapshot" => _serialize_value(assessment.snapshot),
        "evidence_count" => assessment.evidence_count,
        "iteration" => assessment.iteration,
        "similarity" => assessment.similarity,
        "comparable" => assessment.comparable,
        "stability_threshold" => assessment.stability_threshold,
        "stable_passes" => assessment.stable_passes,
        "required_stable_passes" => assessment.required_stable_passes,
        "remaining_stable_passes" => assessment.remaining_stable_passes,
        "min_evidence_remaining" => assessment.min_evidence_remaining,
        "stable" => assessment.stable,
        "stop" => assessment.stop,
        "stop_trigger" => assessment.stop_trigger,
        "reason" => assessment.reason,
    )
end

function as_dict(tracker::EvidenceConvergenceTracker)
    Dict(
        "policy" => as_dict(tracker.policy),
        "iteration" => tracker.iteration,
        "stable_passes" => tracker.stable_passes,
        "previous_snapshot" => _serialize_value(tracker.previous_snapshot),
        "last_assessment" => tracker.last_assessment === nothing ? nothing : as_dict(tracker.last_assessment),
    )
end

to_json(policy::EvidenceConvergencePolicy) = JSON3.write(as_dict(policy))
to_json(assessment::EvidenceConvergenceAssessment) = JSON3.write(as_dict(assessment))
to_json(tracker::EvidenceConvergenceTracker) = JSON3.write(as_dict(tracker))
