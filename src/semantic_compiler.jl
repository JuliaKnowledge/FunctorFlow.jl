# ============================================================================
# semantic_compiler.jl — Semantic-first compilation artifacts
# ============================================================================

"""
    CompilationNode(name, node_kind; inputs=Symbol[], outputs=Symbol[], metadata=Dict())

A lowered semantic node in the parity-oriented compiler plan.
"""
struct CompilationNode
    name::Symbol
    node_kind::Symbol
    inputs::Vector{Symbol}
    outputs::Vector{Symbol}
    metadata::Dict{Symbol, Any}
end

function CompilationNode(name, node_kind;
                         inputs::Vector{Symbol}=Symbol[],
                         outputs::Vector{Symbol}=Symbol[],
                         metadata::Dict=Dict{Symbol, Any}())
    CompilationNode(Symbol(name), Symbol(node_kind), copy(inputs), copy(outputs),
                    Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    CompiledArtifact(subject_name, subject_kind, nodes; proof_shapes=ProofShape[], metadata=Dict())

A compiled semantic subject together with any attached proof-shape claims.
"""
struct CompiledArtifact
    subject_name::Symbol
    subject_kind::Symbol
    nodes::Vector{CompilationNode}
    proof_shapes::Vector{ProofShape}
    metadata::Dict{Symbol, Any}
end

function CompiledArtifact(subject_name, subject_kind, nodes::Vector{CompilationNode};
                          proof_shapes::Vector{ProofShape}=ProofShape[],
                          metadata::Dict=Dict{Symbol, Any}())
    CompiledArtifact(Symbol(subject_name), Symbol(subject_kind), copy(nodes), copy(proof_shapes),
                     Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    CompilationPlan(name, artifacts; metadata=Dict())

A collection of compiled artifacts lowered from several semantic subjects.
"""
struct CompilationPlan
    name::Symbol
    artifacts::Vector{CompiledArtifact}
    metadata::Dict{Symbol, Any}
end

function CompilationPlan(name, artifacts::Vector{CompiledArtifact};
                         metadata::Dict=Dict{Symbol, Any}())
    CompilationPlan(Symbol(name), copy(artifacts),
                    Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    IRType(name; components=IRTypeComponent[], metadata=Dict())

A lightweight compositional type descriptor for placeholder executable IR.
"""
struct IRType
    name::Symbol
    components::Vector{Any}
    metadata::Dict{Symbol, Any}
end

function IRType(name;
                components::Vector=Any[],
                metadata::Dict=Dict{Symbol, Any}())
    IRType(Symbol(name), collect(components),
           Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    IRTypeComponent(name, ir_type; metadata=Dict())

A named component nested inside a compositional `IRType`.
"""
struct IRTypeComponent
    name::Symbol
    ir_type::IRType
    metadata::Dict{Symbol, Any}
end

function IRTypeComponent(name, ir_type::IRType;
                         metadata::Dict=Dict{Symbol, Any}())
    IRTypeComponent(Symbol(name), ir_type,
                    Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    TypedIRValue(name, ir_type, expression; metadata=Dict())

A typed symbolic value produced by placeholder IR execution.
"""
struct TypedIRValue
    name::Symbol
    ir_type::IRType
    expression::String
    metadata::Dict{Symbol, Any}
end

function TypedIRValue(name, ir_type::IRType, expression;
                      metadata::Dict=Dict{Symbol, Any}())
    TypedIRValue(Symbol(name), ir_type, String(expression),
                 Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    IRInstruction(name, opcode; inputs=Symbol[], outputs=Symbol[], output_types=IRType[], proof_shapes=Symbol[], metadata=Dict())

A linear placeholder IR instruction.
"""
struct IRInstruction
    name::Symbol
    opcode::Symbol
    inputs::Vector{Symbol}
    outputs::Vector{Symbol}
    output_types::Vector{IRType}
    proof_shapes::Vector{Symbol}
    metadata::Dict{Symbol, Any}
end

function IRInstruction(name, opcode;
                       inputs::Vector{Symbol}=Symbol[],
                       outputs::Vector{Symbol}=Symbol[],
                       output_types::Vector{IRType}=IRType[],
                       proof_shapes::Vector{Symbol}=Symbol[],
                       metadata::Dict=Dict{Symbol, Any}())
    IRInstruction(Symbol(name), Symbol(opcode), copy(inputs), copy(outputs), copy(output_types),
                  copy(proof_shapes), Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    ExecutableIR(name, instructions; metadata=Dict())

A linearized placeholder IR program ready for symbolic execution.
"""
struct ExecutableIR
    name::Symbol
    instructions::Vector{IRInstruction}
    metadata::Dict{Symbol, Any}
end

function ExecutableIR(name, instructions::Vector{IRInstruction};
                      metadata::Dict=Dict{Symbol, Any}())
    ExecutableIR(Symbol(name), copy(instructions),
                 Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    PlaceholderExecutionResult(ir_name, environment, trace; metadata=Dict())

Symbolic execution result for `ExecutableIR`.
"""
struct PlaceholderExecutionResult
    ir_name::Symbol
    environment::Dict{Symbol, TypedIRValue}
    trace::Vector{String}
    metadata::Dict{Symbol, Any}
end

function PlaceholderExecutionResult(ir_name, environment::Dict{Symbol, TypedIRValue},
                                    trace::Vector{String};
                                    metadata::Dict=Dict{Symbol, Any}())
    PlaceholderExecutionResult(Symbol(ir_name), copy(environment), copy(trace),
                               Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

_reference_component(name, type_name; metadata...) =
    IRTypeComponent(name, IRType(type_name; metadata=Dict{Symbol, Any}(metadata)))

function _compile_ffobject(obj::FFObject)
    CompiledArtifact(obj.name, :ff_object, [
        CompilationNode(Symbol(obj.name, :__object), :object;
            outputs=[obj.name],
            metadata=Dict(
                :kind => obj.kind,
                :shape => obj.shape,
                :description => obj.description,
                :metadata => copy(obj.metadata),
            )),
    ]; metadata=copy(obj.metadata))
end

function _compile_morphism(m::Morphism)
    CompiledArtifact(m.name, :morphism, [
        CompilationNode(Symbol(m.name, :__morphism), :morphism;
            inputs=[m.source],
            outputs=[m.name],
            metadata=Dict(
                :source => m.source,
                :target => m.target,
                :implementation_key => m.implementation_key,
                :description => m.description,
                :metadata => copy(m.metadata),
            )),
    ]; metadata=copy(m.metadata))
end

function _compile_composition(comp::Composition)
    CompiledArtifact(comp.name, :composition, [
        CompilationNode(Symbol(comp.name, :__composition), :composition;
            inputs=[comp.source],
            outputs=[comp.name],
            metadata=Dict(
                :chain => copy(comp.chain),
                :source => comp.source,
                :target => comp.target,
                :description => comp.description,
                :metadata => copy(comp.metadata),
            )),
    ]; metadata=copy(comp.metadata))
end

function _compile_kan(kan::KanExtension)
    node_kind = kan.direction == LEFT ? :left_kan : :right_kan
    proof_shape = kan.direction == LEFT ? prove_left_kan_shape(kan).claim : prove_right_kan_shape(kan).claim
    inputs = kan.target === nothing ? [kan.source] : [kan.source, kan.target]
    CompiledArtifact(kan.name, node_kind, [
        CompilationNode(Symbol(kan.name, :__, node_kind), node_kind;
            inputs=inputs,
            outputs=[kan.name],
            metadata=Dict(
                :source => kan.source,
                :along => kan.along,
                :target => kan.target,
                :reducer => kan.reducer,
                :description => kan.description,
                :metadata => copy(kan.metadata),
            )),
    ]; proof_shapes=[proof_shape], metadata=copy(kan.metadata))
end

function _compile_loss(loss::ObstructionLoss)
    inputs = unique(vcat([Symbol[a, b] for (a, b) in loss.paths]...))
    CompiledArtifact(loss.name, :obstruction_loss, [
        CompilationNode(Symbol(loss.name, :__loss), :obstruction_loss;
            inputs=inputs,
            outputs=[loss.name],
            metadata=Dict(
                :paths => copy(loss.paths),
                :comparator => loss.comparator,
                :weight => loss.weight,
                :description => loss.description,
                :metadata => copy(loss.metadata),
            )),
    ]; metadata=copy(loss.metadata))
end

function _compile_diagram(D::Diagram)
    nodes = CompilationNode[]
    for obj in values(D.objects)
        append!(nodes, _compile_ffobject(obj).nodes)
    end
    for op in values(D.operations)
        artifact = if op isa Morphism
            _compile_morphism(op)
        elseif op isa Composition
            _compile_composition(op)
        elseif op isa KanExtension
            _compile_kan(op)
        else
            continue
        end
        append!(nodes, artifact.nodes)
    end
    for loss in values(D.losses)
        append!(nodes, _compile_loss(loss).nodes)
    end
    CompiledArtifact(D.name, :diagram, nodes;
                     metadata=Dict(
                         :n_objects => length(D.objects),
                         :n_operations => length(D.operations),
                         :n_losses => length(D.losses),
                         :n_ports => length(D.ports),
                     ))
end

function _compile_categorical_model_object(obj::CategoricalModelObject)
    interface_specs = [(p.name, p.port_type, p.direction) for p in obj.interface_ports]
    boundary_specs = [(m.name, m.source, m.target) for m in obj.boundary_maps]
    CompiledArtifact(obj.name, :categorical_model_object, [
        CompilationNode(Symbol(obj.name, :__categorical_model_object), :categorical_model_object;
            outputs=[obj.name],
            metadata=Dict(
                :ambient_category => obj.ambient_category,
                :interface_specs => interface_specs,
                :boundary_specs => boundary_specs,
                :has_diagram => obj.diagram !== nothing,
                :metadata => copy(obj.metadata),
            )),
    ]; metadata=copy(obj.metadata))
end

function _compile_model_morphism(f::ModelMorphism)
    CompiledArtifact(f.name, :model_morphism, [
        CompilationNode(Symbol(f.name, :__model_morphism), :model_morphism;
            inputs=[f.source, f.target],
            outputs=[f.name],
            metadata=Dict(
                :source => f.source,
                :target => f.target,
                :has_functor_data => f.functor_data !== nothing,
                :metadata => copy(f.metadata),
            )),
    ]; metadata=copy(f.metadata))
end

function _compile_natural_transformation(α::NaturalTransformation)
    CompiledArtifact(α.name, :natural_transformation, [
        CompilationNode(Symbol(α.name, :__natural_transformation), :natural_transformation;
            inputs=[α.source_functor, α.target_functor],
            outputs=[α.name],
            metadata=Dict(
                :source_functor => α.source_functor,
                :target_functor => α.target_functor,
                :components => collect(keys(α.components)),
                :metadata => copy(α.metadata),
            )),
    ]; metadata=copy(α.metadata))
end

function _compile_scm_model_object(obj::SCMModelObject)
    local_function_specs = [
        Dict(
            :name => fn.name,
            :target_variable => fn.target_variable,
            :exogenous_parents => copy(fn.exogenous_parents),
            :endogenous_parents => copy(fn.endogenous_parents),
            :expression => fn.expression,
        )
        for fn in obj.local_functions
    ]
    CompiledArtifact(obj.name, :scm_model_object, [
        CompilationNode(Symbol(obj.name, :__scm_object), :scm_object;
            outputs=[obj.name],
            metadata=Dict(
                :ambient_category => obj.category,
                :exogenous_variables => copy(obj.exogenous_variables),
                :endogenous_variables => copy(obj.endogenous_variables),
                :local_function_specs => local_function_specs,
                :metadata => copy(obj.object.metadata),
            )),
    ]; metadata=copy(obj.object.metadata))
end

function _compile_scm_morphism(morphism::SCMMorphism)
    CompiledArtifact(morphism.name, :scm_morphism, [
        CompilationNode(Symbol(morphism.name, :__scm_morphism), :scm_morphism;
            inputs=[morphism.source_scm.name, morphism.target_scm.name],
            outputs=[morphism.name],
            metadata=Dict(
                :source => morphism.source_scm.name,
                :target => morphism.target_scm.name,
                :exogenous_variable_map => copy(morphism.exogenous_variable_map),
                :endogenous_variable_map => copy(morphism.endogenous_variable_map),
                :local_function_map => copy(morphism.local_function_map),
                :metadata => copy(morphism.metadata),
            )),
    ]; metadata=copy(morphism.metadata))
end

function _compile_scm_monomorphism(mono::SCMMonomorphism)
    proof_shape = prove_scm_monomorphism_shape(mono).claim
    CompiledArtifact(mono.name, :scm_monomorphism, [
        CompilationNode(Symbol(mono.name, :__scm_monomorphism), :scm_monomorphism;
            inputs=[mono.source_scm.name, mono.target_scm.name],
            outputs=[mono.name],
            metadata=Dict(
                :source => mono.source_scm.name,
                :target => mono.target_scm.name,
                :predicate_clauses => get(mono.metadata, :predicate_clauses, Any[]),
                :metadata => copy(mono.metadata),
            )),
    ]; proof_shapes=[proof_shape], metadata=copy(mono.metadata))
end

function _compile_scm_subobject(sub::SCMSubobject)
    clause_specs = [Dict(:name => clause.name, :clause_kind => clause.clause_kind, :statement => clause.statement)
                    for clause in sub.clauses]
    CompiledArtifact(sub.name, :scm_subobject, [
        CompilationNode(Symbol(sub.name, :__scm_subobject), :scm_subobject;
            inputs=[sub.object_scm.name, sub.ambient_scm.name, sub.inclusion.name],
            outputs=[sub.name],
            metadata=Dict(
                :object_name => sub.object_scm.name,
                :ambient_name => sub.ambient_scm.name,
                :inclusion_name => sub.inclusion.name,
                :clause_specs => clause_specs,
                :metadata => copy(sub.metadata),
            )),
    ]; metadata=copy(sub.metadata))
end

function _compile_scm_predicate(pred::SCMPredicate)
    clause_specs = [Dict(:name => clause.name, :clause_kind => clause.clause_kind, :statement => clause.statement)
                    for clause in pred.clauses]
    CompiledArtifact(pred.name, :scm_predicate, [
        CompilationNode(Symbol(pred.name, :__scm_predicate), :scm_predicate;
            inputs=[pred.ambient_scm.name, pred.subobject.name],
            outputs=[pred.name],
            metadata=Dict(
                :ambient_name => pred.ambient_scm.name,
                :subobject_name => pred.subobject.name,
                :clause_specs => clause_specs,
                :metadata => copy(pred.metadata),
            )),
    ]; metadata=copy(pred.metadata))
end

function _compile_omega_scm(omega::OmegaSCM)
    truth_values = [Dict(:name => value.name, :meaning => value.meaning) for value in omega.truth_values]
    CompiledArtifact(omega.name, :omega_scm, [
        CompilationNode(Symbol(omega.name, :__omega_scm), :omega_scm;
            outputs=[omega.name],
            metadata=Dict(
                :truth_values => truth_values,
                :ambient_category => omega.category,
                :metadata => copy(omega.metadata),
            )),
    ]; metadata=copy(omega.metadata))
end

function _compile_scm_characteristic_map(map::SCMCharacteristicMap)
    proof_shape = prove_scm_characteristic_map_shape(map).claim
    CompiledArtifact(map.name, :scm_characteristic_map, [
        CompilationNode(Symbol(map.name, :__scm_characteristic_map), :scm_characteristic_map;
            inputs=[map.ambient_scm.name, map.predicate.name, map.omega.name],
            outputs=[map.name],
            metadata=Dict(
                :ambient_name => map.ambient_scm.name,
                :predicate_name => map.predicate.name,
                :omega_name => map.omega.name,
                :classifying_truth_value => map.classifying_truth_value.name,
                :false_truth_value => map.false_truth_value.name,
                :metadata => copy(map.metadata),
            )),
    ]; proof_shapes=[proof_shape], metadata=copy(map.metadata))
end

function _compile_predictive_context(context::PredictiveContext)
    interface_specs = [(p.name, p.port_type) for p in context.object.interface_ports]
    CompiledArtifact(context.name, :predictive_context, [
        CompilationNode(Symbol(context.name, :__predictive_context), :predictive_context;
            outputs=[context.name],
            metadata=Dict(
                :category => context.category,
                :interfaces => [p.name for p in context.object.interface_ports],
                :interface_specs => interface_specs,
                :context_id => context.context_id,
                :context_label => context.label,
                :action_alphabet => copy(context.action_alphabet),
                :observation_alphabet => copy(context.observation_alphabet),
                :metadata => copy(context.metadata),
            )),
    ]; metadata=copy(context.metadata))
end

function _compile_predictive_state_model_object(state::PredictiveStateModelObject)
    interface_specs = [(p.name, p.port_type) for p in state.object.interface_ports]
    CompiledArtifact(state.name, :predictive_state_object, [
        CompilationNode(Symbol(state.name, :__predictive_state), :predictive_state_object;
            outputs=[state.name],
            metadata=Dict(
                :category => state.category,
                :interfaces => [p.name for p in state.object.interface_ports],
                :interface_specs => interface_specs,
                :company => state.company,
                :year => state.year,
                :context_id => state.context_id,
                :context_label => state.context_label,
                :sector => state.sector,
                :predictive_tests => copy(state.predictive_tests),
                :observation_heads => copy(state.observation_heads),
                :evidence_channels => copy(state.evidence_channels),
                :metadata => copy(state.metadata),
            )),
    ]; metadata=copy(state.metadata))
end

function _compile_predictive_state_trajectory(traj::PredictiveStateTrajectory)
    CompiledArtifact(traj.name, :predictive_state_trajectory, [
        CompilationNode(Symbol(traj.name, :__predictive_state_trajectory), :predictive_state_trajectory;
            inputs=[state.name for state in traj.states],
            outputs=[traj.name],
            metadata=Dict(
                :source_category => traj.source_category,
                :target_category => traj.target_category,
                :company => traj.company,
                :context_id => traj.context_id,
                :years => copy(traj.years),
                :state_names => [state.name for state in traj.states],
                :transition_maps => [morphism.name for morphism in traj.transition_maps],
                :metadata => copy(traj.metadata),
            )),
    ]; metadata=copy(traj.metadata))
end

function _compile_predictive_global_section(section::PredictiveGlobalSection)
    interface_specs = [(p.name, p.port_type) for p in section.object.interface_ports]
    CompiledArtifact(section.name, :predictive_global_section, [
        CompilationNode(Symbol(section.name, :__predictive_global_section), :predictive_global_section;
            outputs=[section.name],
            metadata=Dict(
                :category => section.category,
                :interfaces => [p.name for p in section.object.interface_ports],
                :interface_specs => interface_specs,
                :company => section.company,
                :year => section.year,
                :context_ids => copy(section.context_ids),
                :predictive_tests => copy(section.predictive_tests),
                :metadata => copy(section.metadata),
            )),
    ]; metadata=copy(section.metadata))
end

function _compile_persistent_state_model_object(state::PersistentStateModelObject)
    interface_specs = [(p.name, p.port_type) for p in state.object.interface_ports]
    CompiledArtifact(state.name, :persistent_state_object, [
        CompilationNode(Symbol(state.name, :__persistent_state), :persistent_state_object;
            outputs=[state.name],
            metadata=Dict(
                :category => state.category,
                :interfaces => [p.name for p in state.object.interface_ports],
                :interface_specs => interface_specs,
                :company => state.company,
                :year => state.year,
                :action_ontology => copy(state.action_ontology),
                :relation_types => copy(state.relation_types),
                :motif_features => copy(state.motif_features),
                :evidence_channels => copy(state.evidence_channels),
                :metadata => copy(state.metadata),
            )),
    ]; metadata=copy(state.metadata))
end

function _compile_temporal_block(block::TemporalBlockModel)
    interface_specs = [(p.name, p.port_type) for p in block.object.interface_ports]
    CompiledArtifact(block.name, :temporal_block, [
        CompilationNode(Symbol(block.name, :__temporal_block), :temporal_block;
            outputs=[block.name],
            metadata=Dict(
                :category => block.category,
                :interfaces => [p.name for p in block.object.interface_ports],
                :interface_specs => interface_specs,
                :company => block.company,
                :years => copy(block.years),
                :block_length => block.block_length,
                :corruption_modes => copy(block.corruption_modes),
                :consistency_channels => copy(block.consistency_channels),
                :metadata => copy(block.metadata),
            )),
    ]; metadata=copy(block.metadata))
end

function _compile_persistent_trajectory(traj::PersistentTrajectory)
    CompiledArtifact(traj.name, :trajectory_functor, [
        CompilationNode(Symbol(traj.name, :__trajectory), :trajectory_functor;
            inputs=[state.name for state in traj.states],
            outputs=[traj.name],
            metadata=Dict(
                :source_category => traj.source_category,
                :target_category => traj.target_category,
                :company => traj.company,
                :years => copy(traj.years),
                :state_names => [state.name for state in traj.states],
                :transition_maps => [morphism.name for morphism in traj.transition_maps],
                :metadata => copy(traj.metadata),
            )),
    ]; metadata=copy(traj.metadata))
end

function _compile_temporal_repair(repair::TemporalRepair)
    component_specs = [(string(key), morphism.source, morphism.target) for (key, morphism) in sort!(collect(repair.repair_map.components); by=first)]
    CompiledArtifact(repair.name, :temporal_repair, [
        CompilationNode(Symbol(repair.name, :__temporal_repair), :temporal_repair;
            inputs=[repair.raw_trajectory.name, repair.temporal_block.name],
            outputs=[repair.repaired_trajectory.name],
            metadata=Dict(
                :repair_map => repair.repair_map.name,
                :company => repair.raw_trajectory.company,
                :years => copy(repair.raw_trajectory.years),
                :block_length => repair.block_length,
                :corruption_modes => copy(repair.corruption_modes),
                :consistency_penalties => copy(repair.consistency_penalties),
                :repair_objective => repair.repair_objective,
                :component_specs => component_specs,
                :metadata => copy(repair.metadata),
            )),
    ]; metadata=copy(repair.metadata))
end

function _compile_rocket_refinement(refinement::ROCKETRefinement)
    CompiledArtifact(refinement.name, :rocket_refinement, [
        CompilationNode(Symbol(refinement.name, :__rocket_refinement), :rocket_refinement;
            inputs=[refinement.base_state.name, refinement.refined_state.name],
            outputs=[refinement.name],
            metadata=Dict(
                :refinement_map => refinement.refinement.name,
                :neighborhood_ops => copy(refinement.neighborhood_ops),
                :reward_targets => copy(refinement.reward_targets),
                :retrieval_sources => copy(refinement.retrieval_sources),
                :edit_budget => refinement.edit_budget,
                :metadata => copy(refinement.metadata),
            )),
    ]; metadata=copy(refinement.metadata))
end

function _compile_agentic_workflow(workflow::AgenticWorkflow)
    step_specs = [(state.name, get(state.metadata, :step_index, idx - 1), copy(get(state.metadata, :action_prefix, Any[])))
                  for (idx, state) in enumerate(workflow.step_states)]
    transition_specs = [(transition.name, transition.source, transition.target, get(transition.metadata, :action, ""))
                        for transition in workflow.action_transitions]
    CompiledArtifact(workflow.name, :agentic_workflow, [
        CompilationNode(Symbol(workflow.name, :__agentic_workflow), :agentic_workflow;
            outputs=[workflow.name],
            metadata=Dict(
                :workflow_object => workflow.object.name,
                :workflow_functor => workflow.functor.name,
                :source_category => workflow.source_category,
                :target_category => workflow.target_category,
                :company => workflow.company,
                :year => workflow.year,
                :statement_id => workflow.statement_id,
                :actions => copy(workflow.actions),
                :edges => copy(workflow.edges),
                :action_types => copy(workflow.action_types),
                :evidence_channels => copy(workflow.evidence_channels),
                :stage => workflow.stage,
                :step_specs => step_specs,
                :transition_specs => transition_specs,
                :metadata => copy(workflow.metadata),
            )),
    ]; metadata=copy(workflow.metadata))
end

function _compile_rocket_workflow_refinement(refinement::ROCKETWorkflowRefinement)
    CompiledArtifact(refinement.name, :rocket_workflow_refinement, [
        CompilationNode(Symbol(refinement.name, :__rocket_workflow_refinement), :rocket_workflow_refinement;
            inputs=[refinement.base_workflow.name, refinement.refined_workflow.name],
            outputs=[refinement.name],
            metadata=Dict(
                :refinement_map => refinement.refinement.name,
                :reward_mode => refinement.reward_mode,
                :reward_targets => copy(refinement.reward_targets),
                :neighborhood_sources => copy(refinement.neighborhood_sources),
                :candidate_budget => refinement.candidate_budget,
                :company => refinement.base_workflow.company,
                :year => refinement.base_workflow.year,
                :statement_id => refinement.base_workflow.statement_id,
                :metadata => copy(refinement.metadata),
            )),
    ]; metadata=copy(refinement.metadata))
end

function _compile_temporal_schrodinger_bridge(bridge::TemporalSchrodingerBridge)
    endpoint_specs = [(constraint.company, constraint.year_from, constraint.year_to, constraint.split)
                      for constraint in bridge.endpoint_constraints]
    metric_specs = [(String(method), [(String(metric_name), Float64(metric_value)) for (metric_name, metric_value) in sort!(collect(metrics); by=first)])
                    for (method, metrics) in sort!(collect(bridge.summary_metrics); by=first)]
    CompiledArtifact(bridge.name, :temporal_schrodinger_bridge, [
        CompilationNode(Symbol(bridge.name, :__temporal_schrodinger_bridge), :temporal_schrodinger_bridge;
            inputs=[trajectory.name for trajectory in bridge.linked_trajectories],
            outputs=[bridge.name],
            metadata=Dict(
                :dataset_label => bridge.dataset_label,
                :reference_process => bridge.reference_process,
                :solver_family => bridge.solver_family,
                :bridge_method => bridge.bridge_method,
                :state_family => bridge.state_family,
                :conditioning_scope => bridge.conditioning_scope,
                :endpoint_specs => endpoint_specs,
                :metric_specs => metric_specs,
                :linked_trajectory_names => [trajectory.name for trajectory in bridge.linked_trajectories],
                :metadata => merge(copy(bridge.object.metadata), copy(bridge.metadata)),
            )),
    ]; metadata=merge(copy(bridge.object.metadata), copy(bridge.metadata)))
end

function _compile_csql_object(obj::CSQLObject)
    table_specs = [Dict(:name => table.name, :source => table.source, :columns => copy(table.columns), :metadata => copy(table.metadata))
                   for table in obj.tables]
    CompiledArtifact(obj.name, :csql_object, [
        CompilationNode(Symbol(obj.name, :__csql_object), :csql_object;
            outputs=[obj.name],
            metadata=Dict(
                :table_specs => table_specs,
                :metadata => copy(obj.metadata),
            )),
    ]; metadata=copy(obj.metadata))
end

function _compile_csql_morphism(morphism::CSQLMorphism)
    CompiledArtifact(morphism.name, :csql_morphism, [
        CompilationNode(Symbol(morphism.name, :__csql_morphism), :csql_morphism;
            inputs=[morphism.source.name, morphism.target.name],
            outputs=[morphism.name],
            metadata=Dict(
                :source => morphism.source.name,
                :target => morphism.target.name,
                :key_fields => copy(morphism.key_fields),
                :relation_maps => copy(morphism.relation_maps),
                :sql_reference => morphism.sql_reference,
                :metadata => copy(morphism.metadata),
            )),
    ]; metadata=copy(morphism.metadata))
end

function _compile_csql_pullback(pb::CSQLPullbackConstruction)
    CompiledArtifact(pb.name, :csql_pullback, [
        CompilationNode(Symbol(pb.name, :__csql_pullback), :csql_pullback;
            inputs=[pb.left.name, pb.right.name, pb.base.name],
            outputs=[pb.output.name],
            metadata=Dict(
                :left_to_base => pb.left_to_base.name,
                :right_to_base => pb.right_to_base.name,
                :match_fields => copy(pb.match_fields),
                :sql_script => pb.sql_script,
                :output_table => pb.output_table,
                :construction_kind => pb.construction_kind,
                :metadata => copy(pb.metadata),
            )),
    ]; metadata=copy(pb.metadata))
end

function _compile_csql_pushout(po::CSQLPushoutConstruction)
    CompiledArtifact(po.name, :csql_pushout, [
        CompilationNode(Symbol(po.name, :__csql_pushout), :csql_pushout;
            inputs=[po.left.name, po.right.name, po.glue.name],
            outputs=[po.output.name],
            metadata=Dict(
                :sql_script => po.sql_script,
                :output_table => po.output_table,
                :metadata => copy(po.metadata),
            )),
    ]; metadata=copy(po.metadata))
end

function _compile_categorical_db_bridge(bridge::CategoricalDBBridge)
    CompiledArtifact(Symbol(bridge.study.name, :__categorical_db_bridge), :categorical_db_bridge, [
        CompilationNode(Symbol(bridge.study.name, :__categorical_db_bridge), :categorical_db_bridge;
            inputs=[bridge.base_object.name, bridge.atlas_a_object.name, bridge.atlas_b_object.name, bridge.exact_pullback.name, bridge.soft_pullback.name, bridge.pushout.name],
            outputs=[Symbol(bridge.study.name, :__categorical_db_bridge)],
            metadata=Dict(
                :study_name => bridge.study.name,
                :bridge_prefix => get(bridge.study.metadata, :bridge_prefix, bridge.study.name),
                :metadata => copy(bridge.metadata),
            )),
    ]; metadata=copy(bridge.metadata))
end

function _compile_intuitionistic_db_bridge(bridge::IntuitionisticDBBridge)
    truth_counts = Dict(name => count for (name, count) in bridge.materialization.truth_value_counts)
    CompiledArtifact(Symbol(bridge.study.name, :__intuitionistic_db_bridge), :intuitionistic_db_bridge, [
        CompilationNode(Symbol(bridge.study.name, :__intuitionistic_db_bridge), :intuitionistic_db_bridge;
            inputs=[bridge.categorical_db_bridge.base_object.name, bridge.bridge_scm.name, bridge.omega.name],
            outputs=[Symbol(bridge.study.name, :__intuitionistic_db_bridge)],
            metadata=Dict(
                :study_name => bridge.study.name,
                :truth_value_counts => truth_counts,
                :witness_count => length(bridge.materialization.witnesses),
                :metadata => copy(bridge.metadata),
            )),
    ]; metadata=copy(bridge.metadata))
end

function _compile_tcc_atlas_profile(profile::TCCAtlasProfile)
    CompiledArtifact(Symbol(profile.spec.name, :__tcc_atlas_profile), :tcc_atlas_profile, [
        CompilationNode(Symbol(profile.spec.name, :__tcc_atlas_profile), :tcc_atlas_profile;
            inputs=[profile.csql_object.name],
            outputs=[Symbol(profile.spec.name, :__profile)],
            metadata=Dict(
                :atlas_name => profile.spec.name,
                :study_label => profile.spec.study_label,
                :node_count => profile.node_count,
                :edge_count => profile.edge_count,
                :top_edge_count => length(profile.top_edges),
                :metadata => copy(profile.metadata),
            )),
    ]; metadata=copy(profile.metadata))
end

function _compile_tcc_method_pullback(summary::TCCMethodPullbackSummary)
    CompiledArtifact(:TCCMethodPullbackSummary, :tcc_method_pullback, [
        CompilationNode(:TCCMethodPullbackSummary__node, :tcc_method_pullback;
            outputs=[:TCCMethodPullbackSummary],
            metadata=Dict(
                :workspace_root => summary.workspace_root,
                :data_root => summary.data_root,
                :compiled_counts => copy(summary.compiled_counts),
                :omega_counts => copy(summary.omega_counts),
                :pullback_rows => length(summary.did_iv_pullback),
                :method_conflicts => length(summary.method_conflicts),
                :metadata => copy(summary.metadata),
            )),
    ]; metadata=copy(summary.metadata))
end

function _compile_pullback(pb::PullbackResult)
    claim = prove_pullback_shape(pb).claim
    CompiledArtifact(pb.name, :pullback, [
        CompilationNode(Symbol(pb.name, :__pullback), :pullback;
            inputs=[pb.projection1, pb.projection2, pb.shared_object],
            outputs=[pb.name],
            metadata=Dict(
                :projection_left => pb.projection1,
                :projection_right => pb.projection2,
                :shared_object => pb.shared_object,
                :interface_morphisms => copy(pb.interface_morphisms),
                :cone_name => pb.cone.name,
                :metadata => copy(pb.metadata),
            )),
    ]; proof_shapes=[claim], metadata=copy(pb.metadata))
end

function _compile_pushout(po::PushoutResult)
    claim = prove_pushout_shape(po).claim
    CompiledArtifact(po.name, :pushout, [
        CompilationNode(Symbol(po.name, :__pushout), :pushout;
            inputs=[po.injection1, po.injection2, po.shared_object],
            outputs=[po.name],
            metadata=Dict(
                :injection_left => po.injection1,
                :injection_right => po.injection2,
                :shared_object => po.shared_object,
                :interface_morphisms => copy(po.interface_morphisms),
                :cocone_name => po.cocone.name,
                :metadata => copy(po.metadata),
            )),
    ]; proof_shapes=[claim], metadata=copy(po.metadata))
end

function _compile_product(prod::ProductResult)
    CompiledArtifact(prod.name, :product, [
        CompilationNode(Symbol(prod.name, :__product), :product;
            inputs=copy(prod.projections),
            outputs=[prod.name],
            metadata=Dict(
                :projections => copy(prod.projections),
                :diagram_name => prod.product_diagram.name,
                :metadata => copy(prod.metadata),
            )),
    ]; metadata=copy(prod.metadata))
end

function _compile_coproduct(coprod::CoproductResult)
    CompiledArtifact(coprod.name, :coproduct, [
        CompilationNode(Symbol(coprod.name, :__coproduct), :coproduct;
            inputs=copy(coprod.injections),
            outputs=[coprod.name],
            metadata=Dict(
                :injections => copy(coprod.injections),
                :diagram_name => coprod.coproduct_diagram.name,
                :metadata => copy(coprod.metadata),
            )),
    ]; metadata=copy(coprod.metadata))
end

function _compile_equalizer(eq::EqualizerResult)
    CompiledArtifact(eq.name, :equalizer, [
        CompilationNode(Symbol(eq.name, :__equalizer), :equalizer;
            inputs=[eq.equalizer_map],
            outputs=[eq.name],
            metadata=Dict(
                :equalizer_map => eq.equalizer_map,
                :diagram_name => eq.equalizer_diagram.name,
                :metadata => copy(eq.metadata),
            )),
    ]; metadata=copy(eq.metadata))
end

function _compile_coequalizer(coeq::CoequalizerResult)
    CompiledArtifact(coeq.name, :coequalizer, [
        CompilationNode(Symbol(coeq.name, :__coequalizer), :coequalizer;
            inputs=[coeq.coequalizer_map, coeq.quotient_object],
            outputs=[coeq.name],
            metadata=Dict(
                :coequalizer_map => coeq.coequalizer_map,
                :quotient_object => coeq.quotient_object,
                :diagram_name => coeq.coequalizer_diagram.name,
                :metadata => copy(coeq.metadata),
            )),
    ]; metadata=copy(coeq.metadata))
end

"""
    compile_v1(subject) -> CompiledArtifact

Compile one semantic subject into a parity-oriented artifact.
"""
function compile_v1(subject)
    if subject isa Diagram
        _compile_diagram(subject)
    elseif subject isa FFObject
        _compile_ffobject(subject)
    elseif subject isa Morphism
        _compile_morphism(subject)
    elseif subject isa Composition
        _compile_composition(subject)
    elseif subject isa KanExtension
        _compile_kan(subject)
    elseif subject isa ObstructionLoss
        _compile_loss(subject)
    elseif subject isa CategoricalModelObject
        _compile_categorical_model_object(subject)
    elseif subject isa ModelMorphism
        _compile_model_morphism(subject)
    elseif subject isa NaturalTransformation
        _compile_natural_transformation(subject)
    elseif subject isa SCMModelObject
        _compile_scm_model_object(subject)
    elseif subject isa SCMMorphism
        _compile_scm_morphism(subject)
    elseif subject isa SCMMonomorphism
        _compile_scm_monomorphism(subject)
    elseif subject isa SCMSubobject
        _compile_scm_subobject(subject)
    elseif subject isa SCMPredicate
        _compile_scm_predicate(subject)
    elseif subject isa OmegaSCM
        _compile_omega_scm(subject)
    elseif subject isa SCMCharacteristicMap
        _compile_scm_characteristic_map(subject)
    elseif subject isa PredictiveContext
        _compile_predictive_context(subject)
    elseif subject isa PredictiveStateModelObject
        _compile_predictive_state_model_object(subject)
    elseif subject isa PredictiveStateTrajectory
        _compile_predictive_state_trajectory(subject)
    elseif subject isa PredictiveGlobalSection
        _compile_predictive_global_section(subject)
    elseif subject isa PersistentStateModelObject
        _compile_persistent_state_model_object(subject)
    elseif subject isa TemporalBlockModel
        _compile_temporal_block(subject)
    elseif subject isa PersistentTrajectory
        _compile_persistent_trajectory(subject)
    elseif subject isa TemporalRepair
        _compile_temporal_repair(subject)
    elseif subject isa ROCKETRefinement
        _compile_rocket_refinement(subject)
    elseif subject isa AgenticWorkflow
        _compile_agentic_workflow(subject)
    elseif subject isa ROCKETWorkflowRefinement
        _compile_rocket_workflow_refinement(subject)
    elseif subject isa TemporalSchrodingerBridge
        _compile_temporal_schrodinger_bridge(subject)
    elseif subject isa CSQLObject
        _compile_csql_object(subject)
    elseif subject isa CSQLMorphism
        _compile_csql_morphism(subject)
    elseif subject isa CSQLPullbackConstruction
        _compile_csql_pullback(subject)
    elseif subject isa CSQLPushoutConstruction
        _compile_csql_pushout(subject)
    elseif subject isa CategoricalDBBridge
        _compile_categorical_db_bridge(subject)
    elseif subject isa IntuitionisticDBBridge
        _compile_intuitionistic_db_bridge(subject)
    elseif subject isa TCCAtlasProfile
        _compile_tcc_atlas_profile(subject)
    elseif subject isa TCCMethodPullbackSummary
        _compile_tcc_method_pullback(subject)
    elseif subject isa PullbackResult
        _compile_pullback(subject)
    elseif subject isa PushoutResult
        _compile_pushout(subject)
    elseif subject isa ProductResult
        _compile_product(subject)
    elseif subject isa CoproductResult
        _compile_coproduct(subject)
    elseif subject isa EqualizerResult
        _compile_equalizer(subject)
    elseif subject isa CoequalizerResult
        _compile_coequalizer(subject)
    else
        throw(ArgumentError("Unsupported v1 compilation subject: $(typeof(subject))"))
    end
end

"""
    compile_plan(name, subjects...; metadata=Dict()) -> CompilationPlan

Compile several semantic subjects into one compilation plan.
"""
function compile_plan(name, subjects...; metadata::Dict=Dict{Symbol, Any}())
    artifacts = [compile_v1(subject) for subject in subjects]
    CompilationPlan(name, artifacts; metadata=metadata)
end

function _opcode_for_node(node::CompilationNode)
    get(Dict(
        :object => :declare_object,
        :morphism => :declare_morphism,
        :composition => :compose_morphism_chain,
        :left_kan => :extend_left_kan,
        :right_kan => :extend_right_kan,
        :obstruction_loss => :measure_obstruction,
        :categorical_model_object => :declare_model_object,
        :model_morphism => :declare_functor,
        :natural_transformation => :declare_natural_transformation,
        :scm_object => :instantiate_scm,
        :scm_morphism => :declare_scm_morphism,
        :scm_monomorphism => :declare_scm_monomorphism,
        :scm_subobject => :classify_scm_subobject,
        :scm_predicate => :declare_scm_predicate,
        :omega_scm => :instantiate_omega_scm,
        :scm_characteristic_map => :classify_to_omega,
        :predictive_context => :declare_predictive_context,
        :predictive_state_object => :declare_predictive_state,
        :predictive_state_trajectory => :declare_predictive_trajectory,
        :predictive_global_section => :glue_predictive_section,
        :persistent_state_object => :declare_persistent_state,
        :temporal_block => :declare_temporal_block,
        :trajectory_functor => :declare_trajectory_functor,
        :temporal_repair => :repair_temporal_block,
        :rocket_refinement => :refine_structured_state,
        :agentic_workflow => :declare_agentic_workflow,
        :rocket_workflow_refinement => :refine_agentic_workflow,
        :temporal_schrodinger_bridge => :instantiate_temporal_bridge,
        :csql_object => :declare_csql_object,
        :csql_morphism => :declare_csql_morphism,
        :csql_pullback => :compose_csql_pullback,
        :csql_pushout => :compose_csql_pushout,
        :categorical_db_bridge => :declare_categorical_db_bridge,
        :intuitionistic_db_bridge => :declare_intuitionistic_db_bridge,
        :tcc_atlas_profile => :profile_tcc_atlas,
        :tcc_method_pullback => :materialize_tcc_pullback,
        :pullback => :compose_pullback,
        :pushout => :compose_pushout,
        :product => :compose_product,
        :coproduct => :compose_coproduct,
        :equalizer => :compose_equalizer,
        :coequalizer => :compose_coequalizer,
    ), node.node_kind, :emit_symbolic_value)
end

function _type_for_node(node::CompilationNode)
    base_metadata = Dict{Symbol, Any}(:node_kind => node.node_kind, :node_name => node.name)
    if node.node_kind == :object
        return IRType(:ff_object;
            components=[_reference_component(:kind, :object_kind; kind=node.metadata[:kind])],
            metadata=merge(base_metadata, Dict(:description => get(node.metadata, :description, ""))))
    elseif node.node_kind == :morphism
        return IRType(:morphism_map;
            components=[
                _reference_component(:source, :source_ref; object_name=node.metadata[:source]),
                _reference_component(:target, :target_ref; object_name=node.metadata[:target]),
            ],
            metadata=base_metadata)
    elseif node.node_kind == :composition
        chain_components = [
            _reference_component(Symbol(:step_, i), :morphism_ref; morphism_name=morphism)
            for (i, morphism) in enumerate(get(node.metadata, :chain, Symbol[]))
        ]
        return IRType(:composition_map; components=chain_components, metadata=base_metadata)
    elseif node.node_kind == :left_kan || node.node_kind == :right_kan
        components = Any[
            _reference_component(:source, :source_ref; object_name=node.metadata[:source]),
            _reference_component(:along, :morphism_ref; morphism_name=node.metadata[:along]),
        ]
        target = get(node.metadata, :target, nothing)
        if target !== nothing
            push!(components, _reference_component(:target, :target_ref; object_name=target))
        end
        type_name = node.node_kind == :left_kan ? :left_kan_object : :right_kan_object
        return IRType(type_name; components=components, metadata=base_metadata)
    elseif node.node_kind == :obstruction_loss
        path_components = [
            _reference_component(Symbol(:path_, i), :path_pair; left=left, right=right)
            for (i, (left, right)) in enumerate(get(node.metadata, :paths, Tuple{Symbol, Symbol}[]))
        ]
        return IRType(:obstruction_loss; components=path_components, metadata=base_metadata)
    elseif node.node_kind == :pullback
        return IRType(:pullback_object;
            components=[
                _reference_component(:left, :operand_ref; object_name=node.inputs[1]),
                _reference_component(:right, :operand_ref; object_name=node.inputs[2]),
                _reference_component(:over, :shared_context_ref; object_name=node.inputs[3]),
                _reference_component(:projection_left, :projection_map; morphism_name=node.metadata[:projection_left]),
                _reference_component(:projection_right, :projection_map; morphism_name=node.metadata[:projection_right]),
            ],
            metadata=base_metadata)
    elseif node.node_kind == :pushout
        return IRType(:pushout_object;
            components=[
                _reference_component(:left, :operand_ref; object_name=node.inputs[1]),
                _reference_component(:right, :operand_ref; object_name=node.inputs[2]),
                _reference_component(:along, :shared_subobject_ref; object_name=node.inputs[3]),
                _reference_component(:injection_left, :injection_map; morphism_name=node.metadata[:injection_left]),
                _reference_component(:injection_right, :injection_map; morphism_name=node.metadata[:injection_right]),
            ],
            metadata=base_metadata)
    elseif node.node_kind == :product
        return IRType(:product_object; metadata=merge(base_metadata, Dict(:arity => length(node.inputs))))
    elseif node.node_kind == :coproduct
        return IRType(:coproduct_object; metadata=merge(base_metadata, Dict(:arity => length(node.inputs))))
    elseif node.node_kind == :equalizer
        return IRType(:equalizer_object; metadata=base_metadata)
    elseif node.node_kind == :coequalizer
        return IRType(:coequalizer_object; metadata=base_metadata)
    elseif node.node_kind == :categorical_model_object
        return IRType(:categorical_model_object; metadata=base_metadata)
    elseif node.node_kind == :model_morphism
        return IRType(:functor_morphism; metadata=base_metadata)
    elseif node.node_kind == :natural_transformation
        return IRType(:natural_transformation; metadata=base_metadata)
    elseif node.node_kind == :scm_object
        return IRType(:scm_model_object;
            components=[
                _reference_component(:exogenous_variables, :exogenous_variable_bundle; values=get(node.metadata, :exogenous_variables, Symbol[])),
                _reference_component(:endogenous_variables, :endogenous_variable_bundle; values=get(node.metadata, :endogenous_variables, Symbol[])),
            ],
            metadata=base_metadata)
    elseif node.node_kind == :scm_morphism
        return IRType(:scm_morphism;
            components=[
                _reference_component(:source, :source_ref; object_name=node.metadata[:source]),
                _reference_component(:target, :target_ref; object_name=node.metadata[:target]),
            ],
            metadata=base_metadata)
    elseif node.node_kind == :scm_monomorphism
        return IRType(:scm_monomorphism; metadata=base_metadata)
    elseif node.node_kind == :scm_subobject
        return IRType(:scm_subobject; metadata=base_metadata)
    elseif node.node_kind == :scm_predicate
        return IRType(:scm_predicate; metadata=base_metadata)
    elseif node.node_kind == :omega_scm
        return IRType(:omega_scm; metadata=base_metadata)
    elseif node.node_kind == :scm_characteristic_map
        return IRType(:scm_characteristic_map; metadata=base_metadata)
    elseif node.node_kind == :predictive_context
        return IRType(:predictive_context; metadata=base_metadata)
    elseif node.node_kind == :predictive_state_object
        return IRType(:predictive_state_object; metadata=base_metadata)
    elseif node.node_kind == :predictive_state_trajectory
        return IRType(:predictive_state_trajectory; metadata=base_metadata)
    elseif node.node_kind == :predictive_global_section
        return IRType(:predictive_global_section; metadata=base_metadata)
    elseif node.node_kind == :persistent_state_object
        return IRType(:persistent_state_object; metadata=base_metadata)
    elseif node.node_kind == :temporal_block
        return IRType(:temporal_block; metadata=base_metadata)
    elseif node.node_kind == :trajectory_functor
        return IRType(:trajectory_functor; metadata=base_metadata)
    elseif node.node_kind == :temporal_repair
        return IRType(:temporal_repair; metadata=base_metadata)
    elseif node.node_kind == :rocket_refinement
        return IRType(:rocket_refinement; metadata=base_metadata)
    elseif node.node_kind == :agentic_workflow
        return IRType(:agentic_workflow; metadata=base_metadata)
    elseif node.node_kind == :rocket_workflow_refinement
        return IRType(:rocket_workflow_refinement; metadata=base_metadata)
    elseif node.node_kind == :temporal_schrodinger_bridge
        return IRType(:temporal_schrodinger_bridge; metadata=base_metadata)
    elseif node.node_kind == :csql_object
        return IRType(:csql_object; metadata=base_metadata)
    elseif node.node_kind == :csql_morphism
        return IRType(:csql_morphism; metadata=base_metadata)
    elseif node.node_kind == :csql_pullback
        return IRType(:csql_pullback; metadata=base_metadata)
    elseif node.node_kind == :csql_pushout
        return IRType(:csql_pushout; metadata=base_metadata)
    elseif node.node_kind == :categorical_db_bridge
        return IRType(:categorical_db_bridge; metadata=base_metadata)
    elseif node.node_kind == :intuitionistic_db_bridge
        return IRType(:intuitionistic_db_bridge; metadata=base_metadata)
    elseif node.node_kind == :tcc_atlas_profile
        return IRType(:tcc_atlas_profile; metadata=base_metadata)
    elseif node.node_kind == :tcc_method_pullback
        return IRType(:tcc_method_pullback; metadata=base_metadata)
    end
    IRType(:symbolic_value; metadata=base_metadata)
end

"""
    lower_artifact_to_ir(artifact) -> Vector{IRInstruction}

Lower a compiled artifact into placeholder IR instructions.
"""
function lower_artifact_to_ir(artifact::CompiledArtifact)
    proof_shape_names = [claim.name for claim in artifact.proof_shapes]
    [IRInstruction(node.name, _opcode_for_node(node);
        inputs=node.inputs,
        outputs=node.outputs,
        output_types=[_type_for_node(node) for _ in node.outputs],
        proof_shapes=proof_shape_names,
        metadata=merge(Dict{Symbol, Any}(
            :subject_name => artifact.subject_name,
            :subject_kind => artifact.subject_kind,
        ), node.metadata),
    ) for node in artifact.nodes]
end

"""
    lower_plan_to_executable_ir(plan) -> ExecutableIR

Lower a semantic compilation plan into an executable placeholder IR.
"""
function lower_plan_to_executable_ir(plan::CompilationPlan)
    instructions = IRInstruction[]
    for artifact in plan.artifacts
        append!(instructions, lower_artifact_to_ir(artifact))
    end
    ExecutableIR(Symbol(plan.name, :__ir), instructions;
                 metadata=merge(Dict{Symbol, Any}(:source_plan => plan.name), plan.metadata))
end

"""
    compile_to_executable_ir(name, subjects...; metadata=Dict()) -> ExecutableIR

Compile semantic subjects all the way down to placeholder executable IR.
"""
function compile_to_executable_ir(name, subjects...; metadata::Dict=Dict{Symbol, Any}())
    plan = compile_plan(name, subjects...; metadata=metadata)
    lower_plan_to_executable_ir(plan)
end

function _render_placeholder_output(instruction::IRInstruction,
                                    environment::Dict{Symbol, TypedIRValue})
    rendered_inputs = [haskey(environment, input) ? environment[input].expression : String(input)
                       for input in instruction.inputs]
    joined_inputs = isempty(rendered_inputs) ? "unit" : join(rendered_inputs, ", ")
    types = isempty(instruction.output_types) ?
        [IRType(:symbolic_value; metadata=Dict(:opcode => instruction.opcode)) for _ in instruction.outputs] :
        instruction.output_types
    [TypedIRValue(output_name, output_type, "$(instruction.opcode)($joined_inputs)";
                  metadata=merge(Dict{Symbol, Any}(
                      :instruction => instruction.name,
                      :proof_shapes => copy(instruction.proof_shapes),
                  ), instruction.metadata))
     for (output_name, output_type) in zip(instruction.outputs, types)]
end

"""
    execute_placeholder_ir(ir::ExecutableIR) -> PlaceholderExecutionResult

Symbolically execute placeholder IR so shared Julia/Python tests can compare
normalized structured outputs without requiring a numerical backend.
"""
function execute_placeholder_ir(ir::ExecutableIR)
    environment = Dict{Symbol, TypedIRValue}()
    trace = String[]
    for instruction in ir.instructions
        rendered_outputs = _render_placeholder_output(instruction, environment)
        for output_value in rendered_outputs
            environment[output_value.name] = output_value
        end
        proof_suffix = isempty(instruction.proof_shapes) ? "" :
            " [proofs: $(join(String.(instruction.proof_shapes), ", "))]"
        push!(trace,
              "$(instruction.name): $(instruction.opcode) inputs=$(Tuple(instruction.inputs)) " *
              "outputs=$(Tuple(instruction.outputs)) " *
              "types=$(Tuple(String(t.name) for t in instruction.output_types))$proof_suffix")
    end
    PlaceholderExecutionResult(ir.name, environment, trace; metadata=copy(ir.metadata))
end

function as_dict(shape::ProofShape)
    Dict{String, Any}(
        "name" => String(shape.name),
        "claim_kind" => String(shape.claim_kind),
        "subject_name" => String(shape.subject_name),
        "assumptions" => copy(shape.assumptions),
        "obligations" => copy(shape.obligations),
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in shape.metadata),
    )
end

function as_dict(bundle::ProofBundle)
    Dict{String, Any}(
        "name" => String(bundle.name),
        "claims" => [as_dict(claim) for claim in bundle.claims],
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in bundle.metadata),
    )
end

function as_dict(node::CompilationNode)
    Dict{String, Any}(
        "name" => String(node.name),
        "node_kind" => String(node.node_kind),
        "inputs" => String.(node.inputs),
        "outputs" => String.(node.outputs),
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in node.metadata),
    )
end

function as_dict(artifact::CompiledArtifact)
    Dict{String, Any}(
        "subject_name" => String(artifact.subject_name),
        "subject_kind" => String(artifact.subject_kind),
        "nodes" => [as_dict(node) for node in artifact.nodes],
        "proof_shapes" => [as_dict(shape) for shape in artifact.proof_shapes],
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in artifact.metadata),
    )
end

function as_dict(plan::CompilationPlan)
    Dict{String, Any}(
        "name" => String(plan.name),
        "artifacts" => [as_dict(artifact) for artifact in plan.artifacts],
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in plan.metadata),
    )
end

function as_dict(ir_type::IRType)
    Dict{String, Any}(
        "name" => String(ir_type.name),
        "components" => [as_dict(component) for component in ir_type.components],
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in ir_type.metadata),
    )
end

function as_dict(component::IRTypeComponent)
    Dict{String, Any}(
        "name" => String(component.name),
        "ir_type" => as_dict(component.ir_type),
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in component.metadata),
    )
end

function as_dict(instr::IRInstruction)
    Dict{String, Any}(
        "name" => String(instr.name),
        "opcode" => String(instr.opcode),
        "inputs" => String.(instr.inputs),
        "outputs" => String.(instr.outputs),
        "output_types" => [as_dict(t) for t in instr.output_types],
        "proof_shapes" => String.(instr.proof_shapes),
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in instr.metadata),
    )
end

function as_dict(ir::ExecutableIR)
    Dict{String, Any}(
        "name" => String(ir.name),
        "instructions" => [as_dict(instr) for instr in ir.instructions],
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in ir.metadata),
    )
end

function as_dict(value::TypedIRValue)
    Dict{String, Any}(
        "name" => String(value.name),
        "ir_type" => as_dict(value.ir_type),
        "expression" => value.expression,
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in value.metadata),
    )
end

function as_dict(result::PlaceholderExecutionResult)
    ordered_env = sort!(collect(result.environment); by=first)
    Dict{String, Any}(
        "ir_name" => String(result.ir_name),
        "environment" => Dict(String(name) => as_dict(value) for (name, value) in ordered_env),
        "trace" => copy(result.trace),
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in result.metadata),
    )
end

to_json(shape::ProofShape) = JSON3.write(as_dict(shape))
to_json(bundle::ProofBundle) = JSON3.write(as_dict(bundle))
to_json(artifact::CompiledArtifact) = JSON3.write(as_dict(artifact))
to_json(plan::CompilationPlan) = JSON3.write(as_dict(plan))
to_json(ir::ExecutableIR) = JSON3.write(as_dict(ir))
to_json(result::PlaceholderExecutionResult) = JSON3.write(as_dict(result))
