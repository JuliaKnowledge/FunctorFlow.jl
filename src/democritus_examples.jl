# ============================================================================
# democritus_examples.jl — Democritus-style local causal graph assembly
# ============================================================================

_canonical_claim_text(claim::AbstractString) = replace(strip(String(claim)), r"\s*->\s*" => " -> ")

function _claim_endpoints(claim::AbstractString)
    parts = split(_canonical_claim_text(claim), " -> ")
    length(parts) == 2 || error("Expected a causal claim of the form `source -> target`, got: $(repr(claim))")
    strip(parts[1]), strip(parts[2])
end

_claim_nodes(claim::AbstractString) = Set(_claim_endpoints(claim))
_format_claim(source::AbstractString, target::AbstractString) = string(strip(source), " -> ", strip(target))

function _claim_set(value)::Set{String}
    result = Set{String}()
    _claim_set!(result, value)
    result
end

function _claim_set!(result::Set{String}, value::Nothing)
    result
end

function _claim_set!(result::Set{String}, value::AbstractString)
    push!(result, _canonical_claim_text(value))
    result
end

function _claim_set!(result::Set{String}, value::Union{AbstractVector, Tuple, AbstractSet})
    for item in value
        _claim_set!(result, item)
    end
    result
end

function _claim_set!(result::Set{String}, value::AbstractDict)
    for item in values(value)
        _claim_set!(result, item)
    end
    result
end

function infer_democritus_repairs(claims; max_hops::Integer=2)
    repaired = _claim_set(claims)
    max_hops <= 1 && return repaired
    for _ in 2:Int(max_hops)
        additions = Set{String}()
        current = collect(repaired)
        for left in current, right in current
            src1, dst1 = _claim_endpoints(left)
            src2, dst2 = _claim_endpoints(right)
            dst1 == src2 || continue
            src1 == dst2 && continue
            candidate = _format_claim(src1, dst2)
            candidate in repaired || push!(additions, candidate)
        end
        isempty(additions) && break
        union!(repaired, additions)
    end
    repaired
end

"""
    democritus_repair_reducer(source, relation, metadata)

Reducer for Democritus-style local-to-global assembly. It unions local claim
sets inside each overlap region and then adds repaired claims by taking a small
transitive closure over the resulting causal graph.
"""
function democritus_repair_reducer(source, relation, metadata)
    grouped = group_values(source, relation)
    max_hops = Int(get(metadata, "closure_depth", 2))
    Dict(region => infer_democritus_repairs(values; max_hops=max_hops) for (region, values) in grouped)
end

"""
    democritus_claim_distance(a, b)

Edge-aware distance between two local-claim collections. Unlike the generic
token-overlap comparator, this treats whole causal claims as the semantic unit.
"""
function democritus_claim_distance(a, b)
    sa = _claim_set(a)
    sb = _claim_set(b)
    union_size = length(union(sa, sb))
    union_size == 0 && return 0.0
    length(symdiff(sa, sb)) / union_size
end

"""
    build_democritus_restrictor(local_claims; fragment_focus=nothing)

Build a regrounding map from a glued global section back to local fragments.
Claims are assigned back to fragments whose focus vocabulary contains both
endpoints of the repaired claim.
"""
function build_democritus_restrictor(local_claims::AbstractDict; fragment_focus::Union{Nothing, AbstractDict}=nothing)
    local_keys = collect(keys(local_claims))
    focus_map = Dict{Any, Set{String}}()
    for key in local_keys
        focus_map[key] = fragment_focus === nothing ?
            reduce(union!, (_claim_nodes(claim) for claim in _claim_set(local_claims[key])); init=Set{String}()) :
            Set{String}(String.(fragment_focus[key]))
    end

    function restrictor(global_sections)
        global_claims = _claim_set(global_sections)
        Dict(
            key => Set{String}(
                claim for claim in global_claims
                if all(node -> node in focus_map[key], _claim_endpoints(claim))
            )
            for key in local_keys
        )
    end
end

"""
    build_democritus_assembly_example()

Construct a concrete Democritus-style local causal graph assembly example with
repair-aware gluing and regrounding. The example returns the diagram together
with its local claim fragments, cover relation, and focus vocabularies.
"""
function build_democritus_assembly_example()
    local_claims = Dict(
        :policy => Set([
            "minimum wage -> earnings",
            "minimum wage -> labor costs",
        ]),
        :household => Set([
            "earnings -> demand",
            "demand -> employment",
        ]),
        :labor => Set([
            "labor costs -> employment",
            "employment -> job quality",
        ]),
    )
    fragment_focus = Dict(
        :policy => Set(["minimum wage", "earnings", "labor costs", "employment"]),
        :household => Set(["minimum wage", "earnings", "demand", "employment"]),
        :labor => Set(["minimum wage", "labor costs", "employment", "job quality"]),
    )
    overlap_relation = Dict(
        :labor_market => [:policy, :household, :labor],
        :policy_household => [:policy, :household],
        :policy_labor => [:policy, :labor],
    )

    config = DemocritusAssemblyConfig(
        name=:DemocritusCausalAssembly,
        gluing_config=DemocritusGluingConfig(
            name=:DemocritusCausalGlue,
            claim_object=:LocalCausalClaims,
            overlap_relation=:ContextCover,
            global_object=:GlobalCausalGraph,
            glue_name=:assemble_global_section,
            reducer=:causal_transitive_closure,
        ),
        restrict_name=:reground_repaired_claims,
        coherence_loss_name=:section_repair_loss,
        comparator=:claim_delta,
    )

    diagram = democritus_assembly_pipeline(;
        config,
        restrict_impl=build_democritus_restrictor(local_claims; fragment_focus),
    )
    bind_reducer!(diagram, :causal_transitive_closure, democritus_repair_reducer)
    bind_comparator!(diagram, :claim_delta, democritus_claim_distance)

    Dict{Symbol, Any}(
        :diagram => diagram,
        :local_claims => local_claims,
        :fragment_focus => fragment_focus,
        :overlap_relation => overlap_relation,
        :metadata => Dict(
            :domain => "labor_market_policy",
            :closure_depth => 2,
        ),
    )
end

function execute_democritus_assembly_example(example::Union{Nothing, Dict{Symbol, Any}}=nothing)
    example = example === nothing ? build_democritus_assembly_example() : example
    diagram = example[:diagram]
    run(diagram, Dict(
        diagram.ports[:input].ref => example[:local_claims],
        diagram.ports[:relation].ref => example[:overlap_relation],
    ))
end

function summarize_democritus_assembly_example(example::Union{Nothing, Dict{Symbol, Any}}=nothing)
    example = example === nothing ? build_democritus_assembly_example() : example
    diagram = example[:diagram]
    executed = execute_democritus_assembly_example(example)
    seed_claims = _claim_set(example[:local_claims])
    global_sections = executed.values[diagram.ports[:global_output].ref]
    global_claims = _claim_set(global_sections)
    regrounded_claims = executed.values[diagram.ports[:regrounded_claims].ref]
    fragment_repairs = Dict(
        String(fragment) => sort(collect(setdiff(_claim_set(regrounded_claims[fragment]), _claim_set(example[:local_claims][fragment]))))
        for fragment in keys(example[:local_claims])
    )

    Dict(
        "domain" => String(example[:metadata][:domain]),
        "counts" => Dict(
            "local_fragments" => length(example[:local_claims]),
            "seed_claims" => length(seed_claims),
            "global_claims" => length(global_claims),
            "repaired_claims" => length(setdiff(global_claims, seed_claims)),
        ),
        "overlap_regions" => Dict(String(region) => copy(String.(fragments)) for (region, fragments) in example[:overlap_relation]),
        "fragment_focus" => Dict(String(fragment) => sort(collect(String.(focus))) for (fragment, focus) in example[:fragment_focus]),
        "global_sections" => Dict(String(region) => sort(collect(_claim_set(claims))) for (region, claims) in global_sections),
        "global_inferred_claims" => sort(collect(setdiff(global_claims, seed_claims))),
        "fragment_repairs" => fragment_repairs,
        "coherence_loss" => executed.losses[diagram.ports[:coherence_loss].ref],
    )
end
