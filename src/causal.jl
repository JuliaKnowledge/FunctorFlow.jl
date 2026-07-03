# ============================================================================
# causal.jl — RN-Kan-Do-Calculus and causal semantics (v1)
# ============================================================================

# The causal interpretation of FunctorFlow's Kan primitives:
# - Right Kan ↔ conditioning (observational, compatible completion)
# - Left Kan  ↔ intervention (do-calculus, pushforward aggregation)
# - RN layer  ↔ density ratio estimation (interventional vs observational)

"""
    CausalContext(name; observational_regime, interventional_regime)

A causal context specifying the regimes for RN-Kan-Do reasoning.
"""
struct CausalContext
    name::Symbol
    observational_regime::Symbol
    interventional_regime::Symbol
    metadata::Dict{Symbol, Any}
end

function CausalContext(name::Union{Symbol, AbstractString};
                       observational_regime::Union{Symbol, AbstractString}=:obs,
                       interventional_regime::Union{Symbol, AbstractString}=:do,
                       metadata::Dict=Dict{Symbol, Any}())
    CausalContext(Symbol(name), Symbol(observational_regime),
                  Symbol(interventional_regime),
                  Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    CausalDiagram(name, context; conditioning_kan, intervention_kan)

A diagram with explicit causal semantics: right-Kan for conditioning
(observational) and left-Kan for intervention (do-calculus).
"""
struct CausalDiagram
    name::Symbol
    context::CausalContext
    base_diagram::Diagram
    conditioning_kan::Union{Nothing, Symbol}
    intervention_kan::Union{Nothing, Symbol}
    metadata::Dict{Symbol, Any}
end

"""
    build_causal_diagram(name; context=CausalContext(:default), kwargs...) -> CausalDiagram

Build a diagram with explicit causal Kan semantics.
"""
function build_causal_diagram(name::Union{Symbol, AbstractString};
                              context::CausalContext=CausalContext(:default),
                              observation_source::Union{Symbol, AbstractString}=:Observations,
                              causal_relation::Union{Symbol, AbstractString}=:CausalStructure,
                              intervention_target::Union{Symbol, AbstractString}=:InterventionalState,
                              conditioning_target::Union{Symbol, AbstractString}=:ConditionalState,
                              metadata::Dict=Dict{Symbol, Any}())
    D = Diagram(name)
    add_object!(D, observation_source; kind=:observations)
    add_object!(D, causal_relation; kind=:causal_structure)
    add_object!(D, intervention_target; kind=:interventional_state)
    add_object!(D, conditioning_target; kind=:conditional_state)

    # Left-Kan: intervention (do-calculus pushforward)
    intervention = add_left_kan!(D, :intervene;
        source=observation_source, along=causal_relation,
        target=intervention_target, reducer=:sum,
        metadata=Dict{Symbol, Any}(:causal_role => :intervention,
                                    :regime => context.interventional_regime))

    # Right-Kan: conditioning (observational completion)
    conditioning = add_right_kan!(D, :condition;
        source=observation_source, along=causal_relation,
        target=conditioning_target, reducer=:first_non_null,
        metadata=Dict{Symbol, Any}(:causal_role => :conditioning,
                                    :regime => context.observational_regime))

    expose_port!(D, :observations, observation_source; direction=INPUT, port_type=:observations)
    expose_port!(D, :causal_structure, causal_relation; direction=INPUT, port_type=:causal_structure)
    expose_port!(D, :intervention, :intervene; direction=OUTPUT, port_type=:interventional_state)
    expose_port!(D, :conditioning, :condition; direction=OUTPUT, port_type=:conditional_state)

    CausalDiagram(Symbol(name), context, D, :condition, :intervene,
                  Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    causal_transport(source_diagram, target_diagram; density_ratio, name)

Construct a causal transport between two regimes using density ratio
reweighting (RN layer). The density ratio ρ = p_do(y)/p_obs(y) enables
computing interventional expectations from observational data.
"""
function causal_transport(source::CausalDiagram, target::CausalDiagram;
                          density_ratio::Any=nothing,
                          name::Union{Symbol, AbstractString}=:CausalTransport,
                          metadata::Dict=Dict{Symbol, Any}())
    D = Diagram(name)

    src_inc = include!(D, source.base_diagram; namespace=:source_regime)
    tgt_inc = include!(D, target.base_diagram; namespace=:target_regime)

    # Add RN density ratio morphism if provided
    if density_ratio !== nothing
        add_object!(D, :DensityRatio; kind=:density_ratio)
        add_morphism!(D, :rn_reweight,
                      :source_regime__Observations, :DensityRatio;
                      implementation=density_ratio,
                      metadata=Dict{Symbol, Any}(:causal_role => :rn_layer))

        # Connect RN output to target regime for reweighted estimation
        add_object!(D, :ReweightedEstimate; kind=:reweighted)
        add_morphism!(D, :apply_weights,
                      :DensityRatio, :ReweightedEstimate;
                      metadata=Dict{Symbol, Any}(:causal_role => :reweighting))

        expose_port!(D, :density_ratio, :DensityRatio; direction=OUTPUT, port_type=:density_ratio)
        expose_port!(D, :reweighted, :ReweightedEstimate; direction=OUTPUT, port_type=:reweighted)
    end

    expose_port!(D, :source_obs, :source_regime__Observations; direction=INPUT, port_type=:observations)
    expose_port!(D, :target_obs, :target_regime__Observations; direction=INPUT, port_type=:observations)

    D
end

"""
    interventional_expectation(cd::CausalDiagram, obs_data::Dict;
                               density_ratio_fn=nothing) -> Dict

Compute E_do[Y] from observational data using importance weighting.
If density_ratio_fn is provided, it computes ρ(y) = p_do(y)/p_obs(y)
for each observation.
"""
function interventional_expectation(cd::CausalDiagram, obs_data::Dict;
                                    density_ratio_fn::Union{Nothing, Function}=nothing)
    diagram = density_ratio_fn === nothing ? cd.base_diagram : deepcopy(cd.base_diagram)

    # Bind density ratio as the intervention reducer if provided
    if density_ratio_fn !== nothing
        bind_reducer!(diagram, :sum,
            function(data, relation, meta)
                if data isa Dict
                    result = Dict{Any, Any}()
                    for (k, v) in data
                        weight = density_ratio_fn(v)
                        result[k] = v * weight
                    end
                    result
                else
                    data
                end
            end)
    end

    compiled = compile_to_callable(diagram)
    result = FunctorFlow.run(compiled, obs_data)

    Dict{Symbol, Any}(
        :intervention => get(result.values, :intervene, nothing),
        :conditioning => get(result.values, :condition, nothing),
        :all_values => result.values
    )
end

"""
    is_identifiable(cd::CausalDiagram, target::Symbol;
                    observed::Vector{Symbol}=Symbol[]) -> NamedTuple

Check whether a causal effect is identifiable from observational data.
Uses the structure of the causal diagram to determine if do-calculus
rules can eliminate all do-operators.

Returns (identifiable=Bool, rule=Symbol, reasoning=String).
"""
function is_identifiable(cd::CausalDiagram, target::Symbol;
                         observed::Vector{Symbol}=Symbol[])
    D = cd.base_diagram

    haskey(D.objects, target) || return (
        identifiable=false,
        rule=:invalid_target,
        reasoning="Target $target is not a declared object in diagram :$(D.name); " *
                  "this high-level API only reasons structurally over declared diagram objects."
    )

    # Check if the target is reachable from interventional state
    has_intervention = haskey(D.operations, cd.intervention_kan)

    if !has_intervention
        return (identifiable=false, rule=:none, reasoning="No intervention Kan extension in diagram")
    end

    intervene_op = D.operations[cd.intervention_kan]

    adjacency = Dict{Symbol, Vector{Symbol}}()
    function add_edge!(src::Union{Nothing, Symbol}, dst::Union{Nothing, Symbol})
        src === nothing && return
        dst === nothing && return
        push!(get!(adjacency, src, Symbol[]), dst)
    end
    for op in values(D.operations)
        if op isa Morphism || op isa Composition
            add_edge!(op.source, op.target)
        elseif op isa KanExtension
            add_edge!(op.source, op.target)
            add_edge!(op.along, op.target)
        end
    end

    reachable = Set{Symbol}()
    frontier = Symbol[]
    intervene_op.target !== nothing && push!(frontier, intervene_op.target)
    push!(frontier, intervene_op.source)
    while !isempty(frontier)
        node = pop!(frontier)
        node in reachable && continue
        push!(reachable, node)
        for nxt in get(adjacency, node, Symbol[])
            nxt in reachable || push!(frontier, nxt)
        end
    end

    if !(target in reachable)
        return (identifiable=true, rule=:no_causal_path,
                reasoning="No left-Kan (interventional) path reaches target $target")
    end

    (identifiable=false, rule=:structural_stub,
     reasoning="This CausalDiagram method is only a structural heuristic and does not run " *
               "the Shpitser-Pearl ID algorithm. Build an explicit CausalDAG and call " *
               "`identify_effect`/`is_identifiable(::CausalDAG, ...)` for sound identifiability results.")
end
