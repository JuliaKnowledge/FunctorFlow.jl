# ============================================================================
# blocks.jl — Named block builders (macro library)
# ============================================================================

"""
    ket_block(; config=KETBlockConfig(), kwargs...) -> Diagram

Build a KET (Kan Extension Template) block: left-Kan aggregation over an
incidence relation. The fundamental aggregation pattern covering attention,
pooling, neighborhood message passing, and context fusion.
"""
function ket_block(; config::KETBlockConfig=KETBlockConfig(), kwargs...)
    cfg = _apply_overrides(config, kwargs)
    D = Diagram(cfg.name)
    add_object!(D, cfg.source_object; kind=:messages)
    add_object!(D, cfg.relation_object; kind=:relation)
    add_object!(D, cfg.target_object; kind=:contextualized_messages)
    add_left_kan!(D, cfg.aggregate_name;
                  source=cfg.source_object, along=cfg.relation_object,
                  target=cfg.target_object, reducer=cfg.reducer,
                  metadata=Dict{Symbol, Any}(:macro => :KETBlock))
    expose_port!(D, :input, cfg.source_object; direction=INPUT, port_type=:messages)
    expose_port!(D, :relation, cfg.relation_object; direction=INPUT, port_type=:relation)
    expose_port!(D, :output, cfg.aggregate_name; direction=OUTPUT, port_type=:contextualized_messages)
    D
end

"""
    db_square(; config=DBSquareConfig(), kwargs...) -> Diagram

Build a DB (Diagrammatic Backpropagation) square: measures obstruction to
commutativity via `||f∘g - g∘f||`. The native FunctorFlow pattern for
consistency-aware learning.
"""
function db_square(; config::DBSquareConfig=DBSquareConfig(),
                   first_impl=nothing, second_impl=nothing, kwargs...)
    cfg = _apply_overrides(config, kwargs)
    D = Diagram(cfg.name)
    add_object!(D, cfg.state_object; kind=:state)
    add_morphism!(D, cfg.first_morphism, cfg.state_object, cfg.state_object;
                  implementation=first_impl)
    add_morphism!(D, cfg.second_morphism, cfg.state_object, cfg.state_object;
                  implementation=second_impl)
    compose!(D, cfg.first_morphism, cfg.second_morphism; name=cfg.left_path)
    compose!(D, cfg.second_morphism, cfg.first_morphism; name=cfg.right_path)
    add_obstruction_loss!(D, cfg.loss_name;
                          paths=[(cfg.left_path, cfg.right_path)],
                          comparator=cfg.comparator,
                          metadata=Dict{Symbol, Any}(:macro => :DBSquare))
    expose_port!(D, :input, cfg.state_object; direction=INPUT, port_type=:state)
    expose_port!(D, :left_path, cfg.left_path; direction=OUTPUT, port_type=:state)
    expose_port!(D, :right_path, cfg.right_path; direction=OUTPUT, port_type=:state)
    expose_port!(D, :loss, cfg.loss_name; kind=:loss, direction=OUTPUT, port_type=:loss)
    D
end

"""
    gt_neighborhood_block(; config=GTNeighborhoodConfig(), kwargs...) -> Diagram

Build a GT (Graph Transformer) neighborhood block: lifts tokens to edge
messages, then aggregates via left-Kan over the incidence geometry.
"""
function gt_neighborhood_block(; config::GTNeighborhoodConfig=GTNeighborhoodConfig(), kwargs...)
    cfg = _apply_overrides(config, kwargs)
    D = Diagram(cfg.name)
    add_object!(D, cfg.token_object; kind=:messages)
    add_object!(D, cfg.edge_object; kind=:edge_messages)
    add_object!(D, cfg.relation_object; kind=:relation)
    add_object!(D, cfg.target_object; kind=:contextualized_messages)
    add_morphism!(D, cfg.lift_name, cfg.token_object, cfg.edge_object;
                  metadata=Dict{Symbol, Any}(:macro => :GTNeighborhood))
    add_left_kan!(D, cfg.aggregate_name;
                  source=cfg.edge_object, along=cfg.relation_object,
                  target=cfg.target_object, reducer=cfg.reducer,
                  metadata=Dict{Symbol, Any}(:macro => :GTNeighborhood))
    expose_port!(D, :input, cfg.token_object; direction=INPUT, port_type=:messages)
    expose_port!(D, :relation, cfg.relation_object; direction=INPUT, port_type=:relation)
    expose_port!(D, :output, cfg.aggregate_name; direction=OUTPUT, port_type=:contextualized_messages)
    D
end

"""
    completion_block(; config=CompletionBlockConfig(), kwargs...) -> Diagram

Build a generic right-Kan completion block for denoising, masked completion,
plan repair, or partial-view reconciliation.
"""
function completion_block(; config::CompletionBlockConfig=CompletionBlockConfig(), kwargs...)
    cfg = _apply_overrides(config, kwargs)
    D = Diagram(cfg.name)
    add_object!(D, cfg.source_object; kind=:partial_values)
    add_object!(D, cfg.relation_object; kind=:compatibility)
    add_object!(D, cfg.target_object; kind=:completed_values)
    add_right_kan!(D, cfg.complete_name;
                   source=cfg.source_object, along=cfg.relation_object,
                   target=cfg.target_object, reducer=cfg.reducer,
                   metadata=Dict{Symbol, Any}(:macro => :CompletionBlock))
    expose_port!(D, :input, cfg.source_object; direction=INPUT, port_type=:partial_values)
    expose_port!(D, :relation, cfg.relation_object; direction=INPUT, port_type=:compatibility)
    expose_port!(D, :output, cfg.complete_name; direction=OUTPUT, port_type=:completed_values)
    D
end

"""
    basket_workflow_block(; config=BASKETWorkflowConfig(), kwargs...) -> Diagram

Build a BASKET workflow block: left-Kan with concat reducer to compose local
plan fragments into a composed plan.
"""
function basket_workflow_block(; config::BASKETWorkflowConfig=BASKETWorkflowConfig(), kwargs...)
    cfg = _apply_overrides(config, kwargs)
    D = Diagram(cfg.name)
    add_object!(D, cfg.fragment_object; kind=:plan_fragments)
    add_object!(D, cfg.relation_object; kind=:workflow_relation)
    add_object!(D, cfg.target_object; kind=:plan)
    add_left_kan!(D, cfg.compose_name;
                  source=cfg.fragment_object, along=cfg.relation_object,
                  target=cfg.target_object, reducer=cfg.reducer,
                  metadata=Dict{Symbol, Any}(:macro => :BASKETWorkflow))
    expose_port!(D, :input, cfg.fragment_object; direction=INPUT, port_type=:plan_fragments)
    expose_port!(D, :relation, cfg.relation_object; direction=INPUT, port_type=:workflow_relation)
    expose_port!(D, :output, cfg.compose_name; direction=OUTPUT, port_type=:plan)
    D
end

"""
    rocket_repair_block(; config=ROCKETRepairConfig(), kwargs...) -> Diagram

Build a ROCKET repair block: right-Kan completion to repair candidates using
edit neighborhoods.
"""
function rocket_repair_block(; config::ROCKETRepairConfig=ROCKETRepairConfig(), kwargs...)
    cfg = _apply_overrides(config, kwargs)
    D = Diagram(cfg.name)
    add_object!(D, cfg.candidate_object; kind=:plan_candidates)
    add_object!(D, cfg.edit_relation; kind=:edit_neighborhood)
    add_object!(D, cfg.repaired_object; kind=:repaired_plan)
    add_right_kan!(D, cfg.repair_name;
                   source=cfg.candidate_object, along=cfg.edit_relation,
                   target=cfg.repaired_object, reducer=cfg.reducer,
                   metadata=Dict{Symbol, Any}(:macro => :ROCKETRepair))
    expose_port!(D, :input, cfg.candidate_object; direction=INPUT, port_type=:plan_candidates)
    expose_port!(D, :relation, cfg.edit_relation; direction=INPUT, port_type=:edit_neighborhood)
    expose_port!(D, :output, cfg.repair_name; direction=OUTPUT, port_type=:repaired_plan)
    D
end

"""
    structured_lm_duality(; config=StructuredLMDualityConfig(), kwargs...) -> Diagram

Build a structured LM duality diagram: parallel left-Kan (prediction) and
right-Kan (completion/repair) sub-diagrams sharing a common input.
"""
function structured_lm_duality(; config::StructuredLMDualityConfig=StructuredLMDualityConfig(), kwargs...)
    cfg = _apply_overrides(config, kwargs)
    D = Diagram(cfg.name)
    add_object!(D, cfg.shared_input; kind=cfg.shared_input_kind)

    predict_diag = ket_block(; config=cfg.predict_config)
    repair_diag = completion_block(; config=cfg.repair_config)

    predict_inc = include!(D, predict_diag; namespace=:predict,
                           object_aliases=Dict(cfg.predict_config.source_object => cfg.shared_input))
    repair_inc = include!(D, repair_diag; namespace=:repair,
                          object_aliases=Dict(cfg.repair_config.source_object => cfg.shared_input))

    expose_port!(D, :input, cfg.shared_input; direction=INPUT, port_type=cfg.shared_input_kind)
    expose_port!(D, :predict_output, operation_ref(predict_inc, cfg.predict_config.aggregate_name);
                 direction=OUTPUT, port_type=:contextualized_messages)
    expose_port!(D, :repair_output, operation_ref(repair_inc, cfg.repair_config.complete_name);
                 direction=OUTPUT, port_type=:completed_values)
    D
end

"""
    democritus_gluing_block(; config=DemocritusGluingConfig(), kwargs...) -> Diagram

Build a Democritus gluing block: right-Kan with set_union reducer to glue
local causal claims into a global relational state. This is the sheaf-theoretic
local-to-global construction.
"""
function democritus_gluing_block(; config::DemocritusGluingConfig=DemocritusGluingConfig(), kwargs...)
    cfg = _apply_overrides(config, kwargs)
    D = Diagram(cfg.name)
    add_object!(D, cfg.claim_object; kind=:local_claims)
    add_object!(D, cfg.overlap_relation; kind=:overlap_region)
    add_object!(D, cfg.global_object; kind=:global_state)
    add_right_kan!(D, cfg.glue_name;
                   source=cfg.claim_object, along=cfg.overlap_relation,
                   target=cfg.global_object, reducer=cfg.reducer,
                   metadata=Dict{Symbol, Any}(:macro => :DemocritusGluing))
    expose_port!(D, :input, cfg.claim_object; direction=INPUT, port_type=:local_claims)
    expose_port!(D, :relation, cfg.overlap_relation; direction=INPUT, port_type=:overlap_region)
    expose_port!(D, :output, cfg.glue_name; direction=OUTPUT, port_type=:global_state)
    D
end

"""
    basket_rocket_pipeline(; config=BasketRocketPipelineConfig(), kwargs...) -> Diagram

Build a two-stage BASKET→ROCKET pipeline: drafts a plan via left-Kan
aggregation, then repairs it via right-Kan completion.
"""
function basket_rocket_pipeline(; config::BasketRocketPipelineConfig=BasketRocketPipelineConfig(), kwargs...)
    cfg = _apply_overrides(config, kwargs)
    D = Diagram(cfg.name)

    draft = basket_workflow_block(; config=cfg.basket_config)
    repair = rocket_repair_block(; config=cfg.rocket_config)

    draft_inc = include!(D, draft; namespace=:draft)
    repair_inc = include!(D, repair; namespace=:repair,
                          object_aliases=Dict(cfg.rocket_config.candidate_object =>
                                              operation_ref(draft_inc, cfg.basket_config.compose_name)))

    expose_port!(D, :input, object_ref(draft_inc, cfg.basket_config.fragment_object);
                 direction=INPUT, port_type=:plan_fragments)
    expose_port!(D, :draft_relation, object_ref(draft_inc, cfg.basket_config.relation_object);
                 direction=INPUT, port_type=:workflow_relation)
    expose_port!(D, :repair_relation, object_ref(repair_inc, cfg.rocket_config.edit_relation);
                 direction=INPUT, port_type=:edit_neighborhood)
    expose_port!(D, :draft, operation_ref(draft_inc, cfg.basket_config.compose_name);
                 direction=OUTPUT, port_type=:plan)
    if cfg.consistency_loss_name !== nothing
        add_obstruction_loss!(D, cfg.consistency_loss_name;
                              paths=[(operation_ref(draft_inc, cfg.basket_config.compose_name),
                                      operation_ref(repair_inc, cfg.rocket_config.repair_name))],
                              comparator=cfg.consistency_comparator,
                              metadata=Dict{Symbol, Any}(:macro => :BasketRocketPipeline,
                                                         :role => :repair_consistency))
        expose_port!(D, :consistency_loss, cfg.consistency_loss_name;
                     kind=:loss, direction=OUTPUT, port_type=:loss)
    end
    expose_port!(D, :output, operation_ref(repair_inc, cfg.rocket_config.repair_name);
                 direction=OUTPUT, port_type=:repaired_plan)
    D
end

"""
    democritus_assembly_pipeline(; config=DemocritusAssemblyConfig(), restrict_impl=nothing, kwargs...) -> Diagram

Build a Democritus local-to-global pipeline with an explicit regrounding map
back to local claims. The resulting obstruction loss measures whether the
global section remains compatible with the original fragments.
"""
function democritus_assembly_pipeline(; config::DemocritusAssemblyConfig=DemocritusAssemblyConfig(),
                                      restrict_impl=nothing, kwargs...)
    cfg = _apply_overrides(config, kwargs)
    D = Diagram(cfg.name)

    glue = democritus_gluing_block(; config=cfg.gluing_config)
    glue_inc = include!(D, glue; namespace=:glue)

    glue_op = operation_ref(glue_inc, cfg.gluing_config.glue_name)
    global_obj = object_ref(glue_inc, cfg.gluing_config.global_object)
    claim_obj = object_ref(glue_inc, cfg.gluing_config.claim_object)
    add_object!(D, cfg.regrounded_object; kind=:local_claims)

    add_morphism!(D, cfg.restrict_name, global_obj, cfg.regrounded_object;
                  implementation=restrict_impl,
                  metadata=Dict{Symbol, Any}(:macro => :DemocritusAssembly,
                                             :role => :regrounding))
    add_obstruction_loss!(D, cfg.coherence_loss_name;
                          paths=[(cfg.restrict_name, claim_obj)],
                          comparator=cfg.comparator,
                          metadata=Dict{Symbol, Any}(:macro => :DemocritusAssembly,
                                                     :role => :section_coherence))

    expose_port!(D, :input, claim_obj; direction=INPUT, port_type=:local_claims)
    expose_port!(D, :relation, object_ref(glue_inc, cfg.gluing_config.overlap_relation);
                 direction=INPUT, port_type=:overlap_region)
    expose_port!(D, :global_output, glue_op; direction=OUTPUT, port_type=:global_state)
    expose_port!(D, :regrounded_claims, cfg.restrict_name; direction=OUTPUT, port_type=:local_claims)
    expose_port!(D, :coherence_loss, cfg.coherence_loss_name;
                 kind=:loss, direction=OUTPUT, port_type=:loss)
    D
end

"""
    topocoend_block(; config=TopoCoendConfig(), infer_neighborhood_impl=nothing, lift_impl=nothing, kwargs...) -> Diagram

Build a TopoCoend-style learned neighborhood block. A morphism first infers a
relation from tokens, another morphism lifts tokens into local contexts, and a
left-Kan aggregates those local contexts into a global contextualized state.
"""
function topocoend_block(; config::TopoCoendConfig=TopoCoendConfig(),
                         infer_neighborhood_impl=nothing, lift_impl=nothing, kwargs...)
    cfg = _apply_overrides(config, kwargs)
    D = Diagram(cfg.name)
    add_object!(D, cfg.token_object; kind=:messages)
    add_object!(D, cfg.neighborhood_object; kind=:relation)
    add_object!(D, cfg.local_object; kind=:local_context)
    add_object!(D, cfg.target_object; kind=:contextualized_messages)
    add_morphism!(D, cfg.infer_neighborhood_name, cfg.token_object, cfg.neighborhood_object;
                  implementation=infer_neighborhood_impl,
                  metadata=Dict{Symbol, Any}(:macro => :TopoCoend, :role => :learn_relation))
    add_morphism!(D, cfg.lift_name, cfg.token_object, cfg.local_object;
                  implementation=lift_impl,
                  metadata=Dict{Symbol, Any}(:macro => :TopoCoend, :role => :lift_local))
    add_left_kan!(D, cfg.aggregate_name;
                  source=cfg.local_object, along=cfg.neighborhood_object,
                  target=cfg.target_object, reducer=cfg.reducer,
                  metadata=Dict{Symbol, Any}(:macro => :TopoCoend))
    expose_port!(D, :input, cfg.token_object; direction=INPUT, port_type=:messages)
    expose_port!(D, :learned_relation, cfg.infer_neighborhood_name;
                 direction=OUTPUT, port_type=:relation)
    expose_port!(D, :output, cfg.aggregate_name;
                 direction=OUTPUT, port_type=:contextualized_messages)
    D
end

"""
    horn_fill_block(; config=HornObstructionConfig(), first_face_impl=nothing, second_face_impl=nothing, filler_impl=nothing, kwargs...) -> Diagram

Build a 2-simplex horn filling block. The composed boundary path `d12 ∘ d01` is
compared against the direct filler `d02`, turning simplicial coherence into a
first-class obstruction loss.
"""
function horn_fill_block(; config::HornObstructionConfig=HornObstructionConfig(),
                         first_face_impl=nothing, second_face_impl=nothing, filler_impl=nothing,
                         kwargs...)
    cfg = _apply_overrides(config, kwargs)
    D = Diagram(cfg.name)
    add_object!(D, cfg.source_object; kind=:state)
    add_object!(D, cfg.middle_object; kind=:state)
    add_object!(D, cfg.target_object; kind=:state)
    add_morphism!(D, cfg.first_face, cfg.source_object, cfg.middle_object;
                  implementation=first_face_impl,
                  metadata=Dict{Symbol, Any}(:macro => :HornObstruction, :role => :horn_face))
    add_morphism!(D, cfg.second_face, cfg.middle_object, cfg.target_object;
                  implementation=second_face_impl,
                  metadata=Dict{Symbol, Any}(:macro => :HornObstruction, :role => :horn_face))
    add_morphism!(D, cfg.filler_face, cfg.source_object, cfg.target_object;
                  implementation=filler_impl,
                  metadata=Dict{Symbol, Any}(:macro => :HornObstruction, :role => :horn_filler))
    compose!(D, cfg.first_face, cfg.second_face; name=cfg.boundary_path)
    add_obstruction_loss!(D, cfg.loss_name;
                          paths=[(cfg.boundary_path, cfg.filler_face)],
                          comparator=cfg.comparator,
                          metadata=Dict{Symbol, Any}(:macro => :HornObstruction))
    expose_port!(D, :input, cfg.source_object; direction=INPUT, port_type=:state)
    expose_port!(D, :boundary, cfg.boundary_path; direction=OUTPUT, port_type=:state)
    expose_port!(D, :filler, cfg.filler_face; direction=OUTPUT, port_type=:state)
    expose_port!(D, :loss, cfg.loss_name; kind=:loss, direction=OUTPUT, port_type=:loss)
    D
end

"""
    higher_horn_block(; config=HigherHornConfig(), boundary_face_impls=nothing, filler_impls=nothing, kwargs...) -> Diagram

Build a higher-order horn regularization block. An arbitrary boundary chain is
composed from `boundary_faces`, then compared against one or more direct filler
maps from the first object to the last. The resulting loss sums the obstruction
across all registered fillers.
"""
function higher_horn_block(; config::HigherHornConfig=HigherHornConfig(),
                           boundary_face_impls=nothing, filler_impls=nothing, kwargs...)
    cfg = _apply_overrides(config, kwargs)
    length(cfg.object_names) >= 3 ||
        error("Higher horns require at least three objects")
    length(cfg.boundary_faces) == length(cfg.object_names) - 1 ||
        error("Higher horn boundary faces must have one fewer element than object_names")
    isempty(cfg.filler_faces) &&
        error("Higher horns require at least one filler face")

    boundary_impl_vec = boundary_face_impls === nothing ? fill(nothing, length(cfg.boundary_faces)) :
        collect(boundary_face_impls)
    filler_impl_vec = filler_impls === nothing ? fill(nothing, length(cfg.filler_faces)) :
        collect(filler_impls)
    length(boundary_impl_vec) == length(cfg.boundary_faces) ||
        error("boundary_face_impls must match boundary_faces length")
    length(filler_impl_vec) == length(cfg.filler_faces) ||
        error("filler_impls must match filler_faces length")

    D = Diagram(cfg.name)
    for object_name in cfg.object_names
        add_object!(D, object_name; kind=:state)
    end

    for (idx, face_name) in pairs(cfg.boundary_faces)
        add_morphism!(D, face_name, cfg.object_names[idx], cfg.object_names[idx + 1];
                      implementation=boundary_impl_vec[idx],
                      metadata=Dict{Symbol, Any}(:macro => :HigherHorn,
                                                 :role => :horn_face,
                                                 :position => idx))
    end

    source_object = first(cfg.object_names)
    target_object = last(cfg.object_names)
    for (idx, filler_name) in pairs(cfg.filler_faces)
        add_morphism!(D, filler_name, source_object, target_object;
                      implementation=filler_impl_vec[idx],
                      metadata=Dict{Symbol, Any}(:macro => :HigherHorn,
                                                 :role => :horn_filler,
                                                 :position => idx))
    end

    compose!(D, cfg.boundary_faces...; name=cfg.boundary_path)
    add_obstruction_loss!(D, cfg.loss_name;
                          paths=[(cfg.boundary_path, filler_name) for filler_name in cfg.filler_faces],
                          comparator=cfg.comparator,
                          metadata=Dict{Symbol, Any}(:macro => :HigherHorn))
    expose_port!(D, :input, source_object; direction=INPUT, port_type=:state)
    expose_port!(D, :boundary, cfg.boundary_path; direction=OUTPUT, port_type=:state)
    for filler_name in cfg.filler_faces
        expose_port!(D, filler_name, filler_name; direction=OUTPUT, port_type=:state)
    end
    expose_port!(D, :loss, cfg.loss_name; kind=:loss, direction=OUTPUT, port_type=:loss)
    D
end

"""
    bisimulation_quotient_block(; config=BisimulationQuotientConfig(), kwargs...) -> Diagram

Build a behavioral quotient block by composing a bisimulation relation with two
observation maps and then taking their coequalizer. This turns behavioral
equivalence witnesses into an explicit quotient object.
"""
function bisimulation_quotient_block(; config::BisimulationQuotientConfig=BisimulationQuotientConfig(),
                                     left_projection_impl=nothing, right_projection_impl=nothing,
                                     observe_a_impl=nothing, observe_b_impl=nothing, kwargs...)
    cfg = _apply_overrides(config, kwargs)
    base = Diagram(cfg.name)
    left_path_impl = (left_projection_impl === nothing || observe_a_impl === nothing) ? nothing :
        x -> observe_a_impl(left_projection_impl(x))
    right_path_impl = (right_projection_impl === nothing || observe_b_impl === nothing) ? nothing :
        x -> observe_b_impl(right_projection_impl(x))
    add_object!(base, cfg.relation_object; kind=:relation)
    add_object!(base, cfg.state_a_object; kind=:state)
    add_object!(base, cfg.state_b_object; kind=:state)
    add_object!(base, cfg.behavior_object; kind=:behavior)
    add_morphism!(base, cfg.left_projection, cfg.relation_object, cfg.state_a_object;
                  implementation=left_projection_impl,
                  metadata=Dict{Symbol, Any}(:macro => :BisimulationQuotient, :role => :left_projection))
    add_morphism!(base, cfg.right_projection, cfg.relation_object, cfg.state_b_object;
                  implementation=right_projection_impl,
                  metadata=Dict{Symbol, Any}(:macro => :BisimulationQuotient, :role => :right_projection))
    add_morphism!(base, cfg.observe_a, cfg.state_a_object, cfg.behavior_object;
                  implementation=observe_a_impl,
                  metadata=Dict{Symbol, Any}(:macro => :BisimulationQuotient, :role => :observe_a))
    add_morphism!(base, cfg.observe_b, cfg.state_b_object, cfg.behavior_object;
                  implementation=observe_b_impl,
                  metadata=Dict{Symbol, Any}(:macro => :BisimulationQuotient, :role => :observe_b))
    add_morphism!(base, cfg.left_path, cfg.relation_object, cfg.behavior_object;
                  implementation=left_path_impl,
                  metadata=Dict{Symbol, Any}(:macro => :BisimulationQuotient, :role => :left_behavior))
    add_morphism!(base, cfg.right_path, cfg.relation_object, cfg.behavior_object;
                  implementation=right_path_impl,
                  metadata=Dict{Symbol, Any}(:macro => :BisimulationQuotient, :role => :right_behavior))

    coeq = coequalizer(base, cfg.left_path, cfg.right_path;
                       name=cfg.quotient_name,
                       metadata=Dict{Symbol, Any}(:macro => :BisimulationQuotient))
    D = coeq.coequalizer_diagram

    expose_port!(D, :relation, Symbol(:base__, cfg.relation_object);
                 direction=INPUT, port_type=:relation)
    expose_port!(D, :left_behavior, Symbol(:base__, cfg.left_path);
                 direction=OUTPUT, port_type=:behavior)
    expose_port!(D, :right_behavior, Symbol(:base__, cfg.right_path);
                 direction=OUTPUT, port_type=:behavior)
    expose_port!(D, :output, coeq.coequalizer_map;
                 direction=OUTPUT, port_type=:quotient)
    D
end

# ---------------------------------------------------------------------------
# Macro registry
# ---------------------------------------------------------------------------

"""Registry of named block builders."""
const MACRO_LIBRARY = Dict{Symbol, Any}(
    :ket => ket_block,
    :completion => completion_block,
    :structured_lm_duality => structured_lm_duality,
    :db_square => db_square,
    :gt_neighborhood => gt_neighborhood_block,
    :basket_workflow => basket_workflow_block,
    :rocket_repair => rocket_repair_block,
    :democritus_gluing => democritus_gluing_block,
    :democritus_assembly => democritus_assembly_pipeline,
    :basket_rocket_pipeline => basket_rocket_pipeline,
    :topocoend => topocoend_block,
    :horn_fill => horn_fill_block,
    :higher_horn => higher_horn_block,
    :bisimulation_quotient => bisimulation_quotient_block,
)

"""
    build_macro(name; kwargs...) -> Diagram

Build a diagram from the macro library by name.
"""
function build_macro(name::Union{Symbol, AbstractString}; kwargs...)
    factory = get(MACRO_LIBRARY, Symbol(name)) do
        error("Macro :$(name) not found in library. Available: $(join(sort(collect(keys(MACRO_LIBRARY))), ", "))")
    end
    factory(; kwargs...)
end

# ---------------------------------------------------------------------------
# Config override helper
# ---------------------------------------------------------------------------

function _apply_overrides(config::T, kwargs) where T
    isempty(kwargs) && return config
    fields = fieldnames(T)
    vals = Dict{Symbol, Any}()
    for f in fields
        vals[f] = get(kwargs, f, getfield(config, f))
    end
    T(; vals...)
end
