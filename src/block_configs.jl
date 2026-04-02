# ============================================================================
# block_configs.jl — Configuration types for named block builders
# ============================================================================

"""Configuration for the KET (Kan Extension Template) block."""
Base.@kwdef struct KETBlockConfig
    name::Symbol = :KETBlock
    source_object::Symbol = :Values
    relation_object::Symbol = :Incidence
    target_object::Symbol = :ContextualizedValues
    aggregate_name::Symbol = :aggregate
    reducer::Symbol = :sum
end

"""Configuration for the DB (Diagrammatic Backpropagation) square."""
Base.@kwdef struct DBSquareConfig
    name::Symbol = :DBSquare
    state_object::Symbol = :State
    first_morphism::Symbol = :f
    second_morphism::Symbol = :g
    left_path::Symbol = :p1
    right_path::Symbol = :p2
    comparator::Symbol = :l2
    loss_name::Symbol = :obstruction
end

"""Configuration for the GT (Graph Transformer) neighborhood block."""
Base.@kwdef struct GTNeighborhoodConfig
    name::Symbol = :GTNeighborhood
    token_object::Symbol = :Tokens
    edge_object::Symbol = :EdgeMessages
    relation_object::Symbol = :Incidence
    target_object::Symbol = :ContextualizedTokens
    lift_name::Symbol = :lift
    aggregate_name::Symbol = :aggregate
    reducer::Symbol = :sum
end

"""Configuration for the right-Kan completion block."""
Base.@kwdef struct CompletionBlockConfig
    name::Symbol = :CompletionBlock
    source_object::Symbol = :PartialValues
    relation_object::Symbol = :Compatibility
    target_object::Symbol = :CompletedValues
    complete_name::Symbol = :complete
    reducer::Symbol = :first_non_null
end

"""Configuration for the BASKET workflow block."""
Base.@kwdef struct BASKETWorkflowConfig
    name::Symbol = :BASKETWorkflow
    fragment_object::Symbol = :PlanFragments
    relation_object::Symbol = :WorkflowRelation
    target_object::Symbol = :ComposedPlan
    compose_name::Symbol = :compose_fragments
    reducer::Symbol = :concat
end

"""Configuration for the ROCKET repair block."""
Base.@kwdef struct ROCKETRepairConfig
    name::Symbol = :ROCKETRepair
    candidate_object::Symbol = :Candidates
    edit_relation::Symbol = :EditNeighborhood
    repaired_object::Symbol = :RepairedPlan
    repair_name::Symbol = :repair
    reducer::Symbol = :first_non_null
end

"""Configuration for the structured LM duality (left-Kan + right-Kan)."""
Base.@kwdef struct StructuredLMDualityConfig
    name::Symbol = :StructuredLMDuality
    predict_config::KETBlockConfig = KETBlockConfig(name=:Predict)
    repair_config::CompletionBlockConfig = CompletionBlockConfig(name=:Repair)
    shared_input::Symbol = :SharedInput
    shared_input_kind::Symbol = :hidden_state
end

"""Configuration for the Democritus gluing block."""
Base.@kwdef struct DemocritusGluingConfig
    name::Symbol = :DemocritusGluing
    claim_object::Symbol = :LocalClaims
    overlap_relation::Symbol = :OverlapRegion
    global_object::Symbol = :GlobalState
    glue_name::Symbol = :glue
    reducer::Symbol = :set_union
end

"""Configuration for the BASKET-ROCKET pipeline."""
Base.@kwdef struct BasketRocketPipelineConfig
    name::Symbol = :BasketRocketPipeline
    basket_config::BASKETWorkflowConfig = BASKETWorkflowConfig()
    rocket_config::ROCKETRepairConfig = ROCKETRepairConfig()
    consistency_loss_name::Union{Nothing, Symbol} = :draft_repair_consistency
    consistency_comparator::Symbol = :jaccard
end

"""Configuration for a Democritus gluing + regrounding assembly pipeline."""
Base.@kwdef struct DemocritusAssemblyConfig
    name::Symbol = :DemocritusAssembly
    gluing_config::DemocritusGluingConfig = DemocritusGluingConfig()
    regrounded_object::Symbol = :RegroundedClaims
    restrict_name::Symbol = :restrict_to_local
    coherence_loss_name::Symbol = :section_coherence
    comparator::Symbol = :jaccard
end

"""Configuration for a learned-neighborhood TopoCoend-style block."""
Base.@kwdef struct TopoCoendConfig
    name::Symbol = :TopoCoend
    token_object::Symbol = :Tokens
    neighborhood_object::Symbol = :LearnedNeighborhood
    local_object::Symbol = :LocalContexts
    target_object::Symbol = :GlobalContext
    infer_neighborhood_name::Symbol = :infer_neighborhood
    lift_name::Symbol = :lift_local
    aggregate_name::Symbol = :coend_aggregate
    reducer::Symbol = :mean
end

"""Configuration for a 2-simplex horn filling obstruction block."""
Base.@kwdef struct HornObstructionConfig
    name::Symbol = :HornObstruction
    source_object::Symbol = :Vertex0
    middle_object::Symbol = :Vertex1
    target_object::Symbol = :Vertex2
    first_face::Symbol = :d01
    second_face::Symbol = :d12
    filler_face::Symbol = :d02
    boundary_path::Symbol = :horn_boundary
    comparator::Symbol = :l2
    loss_name::Symbol = :horn_obstruction
end

"""Configuration for a higher-order horn regularization block."""
Base.@kwdef struct HigherHornConfig
    name::Symbol = :HigherHorn
    object_names::Vector{Symbol} = [:Vertex0, :Vertex1, :Vertex2, :Vertex3]
    boundary_faces::Vector{Symbol} = [:d01, :d12, :d23]
    filler_faces::Vector{Symbol} = [:d03]
    boundary_path::Symbol = :higher_horn_boundary
    comparator::Symbol = :l2
    loss_name::Symbol = :higher_horn_obstruction
end

"""Configuration for a bisimulation-inspired behavioral quotient block."""
Base.@kwdef struct BisimulationQuotientConfig
    name::Symbol = :BisimulationQuotient
    relation_object::Symbol = :BisimRelation
    state_a_object::Symbol = :StateA
    state_b_object::Symbol = :StateB
    behavior_object::Symbol = :Behavior
    left_projection::Symbol = :proj_left
    right_projection::Symbol = :proj_right
    observe_a::Symbol = :observe_a
    observe_b::Symbol = :observe_b
    left_path::Symbol = :left_behavior
    right_path::Symbol = :right_behavior
    quotient_name::Symbol = :behavior_quotient
end
