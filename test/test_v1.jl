using FunctorFlow
using Test
using JSON3

const PY_V1_ROOT = normpath(joinpath(@__DIR__, "..", "..", "FunctorFlow_v1"))
const PY_V1_SAMPLE_ROOT = normpath(joinpath(PY_V1_ROOT, "sample_data", "demo_repo_root"))
const PYTHON3 = Sys.which("python3")

function _materialize_json(value)
    if value isa JSON3.Object
        return Dict(String(k) => _materialize_json(v) for (k, v) in pairs(value))
    elseif value isa JSON3.Array
        return [_materialize_json(v) for v in value]
    else
        return value
    end
end

function python_v1_summary(script::AbstractString)
    cmd = addenv(`$(PYTHON3) -c $script`,
        "PYTHONPATH" => PY_V1_ROOT,
        "PATH" => get(ENV, "PATH", "") * ":/opt/homebrew/bin")
    _materialize_json(JSON3.read(read(cmd, String)))
end

@testset "V1 Features" begin
    # ==================================================================
    # A. Semantic Kernel (catlab_interop.jl)
    # ==================================================================
    @testset "Semantic Kernel" begin
        @testset "CategoricalModelObject construction" begin
            obj = CategoricalModelObject(:TestModel;
                                          ambient_category=:FinSet,
                                          semantic_laws=[(_obj -> true)])
            @test obj.name == :TestModel
            @test obj.ambient_category == :FinSet
            @test length(obj.semantic_laws) == 1
            @test obj.diagram === nothing

            # From Diagram
            D = ket_block()
            cat_obj = CategoricalModelObject(D)
            @test cat_obj.name == D.name
            @test cat_obj.diagram === D
            @test length(cat_obj.interface_ports) == length(D.ports)
        end

        @testset "to_diagram" begin
            D = ket_block()
            cat_obj = CategoricalModelObject(D)
            D2 = to_diagram(cat_obj)
            @test D2 isa FunctorFlow.Diagram
            @test D2.name == D.name
        end

        @testset "ModelMorphism" begin
            f = ModelMorphism(:f, :A, :B; functor_data=x -> x * 2)
            @test f.name == :f
            @test f.source == :A
            @test f.target == :B
            @test f.functor_data !== nothing
        end

        @testset "compose ModelMorphism" begin
            f = ModelMorphism(:f, :A, :B; functor_data=x -> x * 2)
            g = ModelMorphism(:g, :B, :C; functor_data=x -> x + 1)
            fg = FunctorFlow.compose(f, g)
            @test fg.source == :A
            @test fg.target == :C
            @test occursin("f", string(fg.name))

            # Composed functor_data preserves both as NamedTuple
            @test fg.functor_data isa NamedTuple
            @test fg.functor_data.second(fg.functor_data.first(3)) == 3 * 2 + 1

            # Incompatible morphisms
            h = ModelMorphism(:h, :X, :Y)
            @test_throws ArgumentError FunctorFlow.compose(f, h)
        end

        @testset "apply ModelMorphism" begin
            obj_A = CategoricalModelObject(:A; ambient_category=:FinSet)
            f = ModelMorphism(:f, :A, :B; functor_data=identity)
            obj_B = FunctorFlow.apply(f, obj_A)
            @test obj_B.name == :B
            @test obj_B.ambient_category == :FinSet

            # Wrong source
            obj_C = CategoricalModelObject(:C)
            @test_throws ArgumentError FunctorFlow.apply(f, obj_C)
        end

        @testset "NaturalTransformation" begin
            α = NaturalTransformation(:alpha, :F, :G;
                                       components=Dict(:X => identity, :Y => x -> x + 1))
            @test α.name == :alpha
            @test α.source_functor == :F
            @test α.target_functor == :G
            @test length(α.components) == 2
        end

        @testset "is_natural" begin
            F = ModelMorphism(:F, :A, :B; functor_data=x -> x * 2)
            G = ModelMorphism(:G, :A, :B; functor_data=x -> x + 10)
            obj_A = CategoricalModelObject(:A)

            α = NaturalTransformation(:alpha, :F, :G;
                                       components=Dict(:A => x -> x + 10 - x * 2))
            @test FunctorFlow.is_natural(α, F, G, [obj_A])

            # Missing component
            α2 = NaturalTransformation(:alpha2, :F, :G;
                                        components=Dict(:Z => identity))
            @test !FunctorFlow.is_natural(α2, F, G, [obj_A])
        end

        @testset "check_laws" begin
            # Law that accepts obj argument
            obj = CategoricalModelObject(:Test;
                                          semantic_laws=[(_obj -> true)])
            results = check_laws(obj)
            @test length(results) == 1
            @test results[1][2] == true

            obj2 = CategoricalModelObject(:Empty)
            @test isempty(check_laws(obj2))
        end

        @testset "MODEL_REGISTRY" begin
            obj = CategoricalModelObject(:RegTest)
            register_model!(obj)
            @test get_model(:RegTest) === obj
            @test haskey(MODEL_REGISTRY, :RegTest)
            delete!(MODEL_REGISTRY, :RegTest)
        end
    end

    # ==================================================================
    # B0. Proof Shapes & Semantic Compiler
    # ==================================================================
    @testset "Proof Shapes & Semantic Compiler" begin
        @testset "proof-shape records" begin
            pb = pullback(ket_block(; name=:Left), ket_block(; name=:Right); over=:Shared)
            pb_shape = prove_pullback_shape(pb)
            @test pb_shape.claim.claim_kind == :pullback
            @test pb_shape.claim.subject_name == pb.name
            @test occursin("commuting square", join(pb_shape.claim.obligations, " "))

            D = Diagram(:ParityKan)
            add_object!(D, :Values; kind=:messages)
            add_object!(D, :Incidence; kind=:relation)
            lkan = add_left_kan!(D, :aggregate; source=:Values, along=:Incidence, target=:Ctx, reducer=:sum)
            rkan = add_right_kan!(D, :complete; source=:Values, along=:Incidence, target=:Ctx, reducer=:first_non_null)

            left_shape = prove_left_kan_shape(lkan)
            right_shape = prove_right_kan_shape(rkan)
            bundle = bundle_proof_shapes(:ParityBundle, pb_shape.claim, left_shape.claim, right_shape.claim)
            @test length(bundle.claims) == 3
            @test as_dict(bundle)["name"] == "ParityBundle"
        end

        @testset "semantic compilation artifacts" begin
            D = ket_block(; name=:ParityKET)
            artifact = compile_v1(D)
            @test artifact.subject_kind == :diagram
            @test !isempty(artifact.nodes)

            pb = pullback(ket_block(; name=:Left), ket_block(; name=:Right); over=:Shared)
            pb_artifact = compile_v1(pb)
            @test pb_artifact.subject_kind == :pullback
            @test length(pb_artifact.proof_shapes) == 1

            plan = compile_plan(:ParityPlan, D, pb; metadata=Dict(:example => "parity"))
            @test length(plan.artifacts) == 2
            @test as_dict(plan)["metadata"]["example"] == "parity"
        end

        @testset "placeholder executable IR" begin
            D = Diagram(:ExecParity)
            add_object!(D, :Tokens; kind=:messages)
            add_object!(D, :Nbrs; kind=:relation)
            add_left_kan!(D, :aggregate; source=:Tokens, along=:Nbrs, target=:Ctx, reducer=:sum)

            ir = compile_to_executable_ir(:ExecParityPlan, D)
            @test ir.name == :ExecParityPlan__ir
            @test !isempty(ir.instructions)
            @test any(instr -> instr.opcode == :extend_left_kan, ir.instructions)

            executed = execute_placeholder_ir(ir)
            @test haskey(executed.environment, :aggregate)
            @test occursin("extend_left_kan", executed.environment[:aggregate].expression)
            @test !isempty(executed.trace)

            ir_dict = as_dict(ir)
            @test ir_dict["name"] == "ExecParityPlan__ir"
            @test ir_dict["instructions"][1]["name"] isa String
        end
    end

    # ==================================================================
    # B1. SCM-specialized semantics
    # ==================================================================
    @testset "SCM Semantics" begin
        @testset "SCM object and morphism builders" begin
            spec = SCMObjectSpec(:WeatherSCM,
                [:u_weather],
                [:traffic],
                [SCMLocalFunctionSpec(:f_traffic, :traffic;
                    exogenous_parents=[:u_weather],
                    expression="traffic := weather_to_traffic(u_weather)")])
            @test validate_scm_spec(spec)

            scm = build_scm_model_object(spec; category=:StructuralCausalModels)
            @test scm.name == :WeatherSCM
            @test scm.exogenous_variables == [:u_weather]
            @test local_function_for_target(scm, :traffic).name == :f_traffic

            ambient = build_scm_model_object(spec; category=:StructuralCausalModels)
            morph = build_scm_morphism(
                name=:WeatherMap,
                source_scm=scm,
                target_scm=ambient,
                exogenous_variable_map=[(:u_weather, :u_weather)],
                endogenous_variable_map=[(:traffic, :traffic)],
                local_function_map=[(:f_traffic, :f_traffic)],
            )
            @test morph.name == :WeatherMap
            @test morph.source == :WeatherSCM
            @test morph.target == :WeatherSCM
        end

        @testset "SCM pullback and predicate logic" begin
            example = build_transport_scm_pullback_example()
            @test example[:pullback] isa PullbackResult
            @test example[:left_to_shared] isa SCMMorphism
            @test example[:right_to_shared] isa SCMMorphism

            predicate_example = build_transport_scm_predicate_example(example)
            monotonicity = predicate_example[:monotonicity_predicate]
            conjunction = predicate_example[:conjunction_predicate]
            @test monotonicity isa SCMPredicate
            @test length(conjunction.clauses) == 2

            omega_example = build_transport_scm_omega_example(predicate_example)
            omega = omega_example[:omega]
            classifier = scm_subobject_classifier(omega)
            internal = as_internal_predicate(omega_example[:monotonicity_classifier])
            @test omega isa OmegaSCM
            @test :top in Set(value.name for value in omega.truth_values)
            @test internal.classifier.truth_values == classifier.truth_values
            @test evaluate_predicate(internal, true) == :top
            @test evaluate_predicate(internal, false) == :bottom
        end

        @testset "SCM compilation parity surfaces" begin
            example = build_transport_scm_pullback_example()
            proof_bundle = build_transport_scm_pullback_proof_bundle(example)
            @test proof_bundle.name == :IntegratedTransportSCMProofBundle

            plan = build_transport_scm_pullback_compilation_plan(example)
            executed = execute_transport_scm_pullback_example(example)
            @test plan.name == :IntegratedTransportSCMCompilationPlan
            @test haskey(executed.environment, :IntegratedTransportSCM)
            @test occursin("compose_pullback", executed.environment[:IntegratedTransportSCM].expression)

            omega_plan = build_transport_scm_omega_compilation_plan()
            omega_exec = execute_transport_scm_omega_example()
            @test length(omega_plan.artifacts) >= 5
            @test any(artifact -> artifact.subject_kind == :omega_scm, omega_plan.artifacts)
            @test any(occursin("classify_to_omega", line) for line in omega_exec.trace)

            plan_dict = as_dict(omega_plan)
            @test plan_dict["name"] == "TransportSCMOmegaCompilationPlan"
            @test length(plan_dict["artifacts"]) == length(omega_plan.artifacts)
        end
    end

    # ==================================================================
    # B2. Predictive-state / PSR semantics
    # ==================================================================
    @testset "Predictive-State Semantics" begin
        example = build_predictive_state_example()
        plan = build_predictive_state_compilation_plan(example; include_gluing_witnesses=true)
        executed = execute_predictive_state_example(example; include_gluing_witnesses=true)
        summary = summarize_predictive_state_example(example)

        @test length(example[:contexts]) == 2
        @test length(example[:local_predictive_states]) == 4
        @test length(example[:predictive_state_trajectories]) == 2
        @test summary["counts"]["global_sections"] == 2
        @test any(artifact -> artifact.subject_kind == :predictive_context, plan.artifacts)
        @test any(artifact -> artifact.subject_kind == :predictive_state_trajectory, plan.artifacts)
        @test any(occursin("declare_predictive_context", line) for line in executed.trace)
        @test any(occursin("glue_predictive_section", line) for line in executed.trace)

        py_summary = python_v1_summary("""
import json
from functorflow_v1.semantic_kernel import Category, Morphism
from functorflow_v1.predictive_states import (
    PredictiveContextSpec, PredictiveStateSpec, PredictiveGlobalSectionSpec,
    build_predictive_context, build_predictive_state_model_object,
    build_predictive_state_trajectory, build_predictive_global_section,
)
context_category = Category("PredictiveContexts")
state_category = Category("CompanyLocalPredictiveStates")
year_category = Category("Years")
section_category = Category("CompanyPredictiveGlobalSections")
market = build_predictive_context(PredictiveContextSpec("MarketContext", "market", "Market conditions", ("launch","price_cut"), ("growth","retention"), {"test_count": 2}), category=context_category)
product = build_predictive_context(PredictiveContextSpec("ProductContext", "product", "Product operations", ("pilot","expand"), ("quality","satisfaction"), {"test_count": 2}), category=context_category)
s1 = build_predictive_state_model_object(PredictiveStateSpec("Acme2023MarketPSR", "acme", 2023, "market", "Market conditions", "software", ("launch_growth","pricing_retention"), ("growth","retention")), category=state_category)
s2 = build_predictive_state_model_object(PredictiveStateSpec("Acme2024MarketPSR", "acme", 2024, "market", "Market conditions", "software", ("launch_growth","pricing_retention"), ("growth","retention")), category=state_category)
s3 = build_predictive_state_model_object(PredictiveStateSpec("Acme2023ProductPSR", "acme", 2023, "product", "Product operations", "software", ("pilot_quality","expand_satisfaction"), ("quality","satisfaction")), category=state_category)
s4 = build_predictive_state_model_object(PredictiveStateSpec("Acme2024ProductPSR", "acme", 2024, "product", "Product operations", "software", ("pilot_quality","expand_satisfaction"), ("quality","satisfaction")), category=state_category)
t1 = Morphism("AcmeMarketTransition", s1.object, s2.object, {"semantic_role": "predictive_state_transition", "company": "acme", "context_id": "market"})
t2 = Morphism("AcmeProductTransition", s3.object, s4.object, {"semantic_role": "predictive_state_transition", "company": "acme", "context_id": "product"})
traj1 = build_predictive_state_trajectory("AcmeMarketTrajectory", company="acme", context_id="market", years=(2023, 2024), states=(s1, s2), year_category=year_category, state_category=state_category, transition_maps=(t1,))
traj2 = build_predictive_state_trajectory("AcmeProductTrajectory", company="acme", context_id="product", years=(2023, 2024), states=(s3, s4), year_category=year_category, state_category=state_category, transition_maps=(t2,))
g1 = build_predictive_global_section(PredictiveGlobalSectionSpec("Acme2023GlobalPSR", "acme", 2023, ("market","product"), ("launch_growth","pilot_quality"), {"pairwise_glueable": True}), category=section_category)
g2 = build_predictive_global_section(PredictiveGlobalSectionSpec("Acme2024GlobalPSR", "acme", 2024, ("market","product"), ("pricing_retention","expand_satisfaction"), {"pairwise_glueable": True}), category=section_category)
summary = {
  "companies": ["acme"],
  "context_ids": ["market", "product"],
  "counts": {"contexts": 2, "local_predictive_states": 4, "predictive_state_trajectories": 2, "restriction_maps": 1, "global_sections": 2, "gluing_witnesses": 1},
  "context_summaries": [
    {"context_id": market.context_id, "label": market.label, "action_alphabet": list(market.action_alphabet), "observation_alphabet": list(market.observation_alphabet), "test_count": market.metadata.get("test_count", 0)},
    {"context_id": product.context_id, "label": product.label, "action_alphabet": list(product.action_alphabet), "observation_alphabet": list(product.observation_alphabet), "test_count": product.metadata.get("test_count", 0)},
  ],
  "company_summaries": [{"company": "acme", "years": [2023, 2024], "n_local_states": 4, "n_trajectories": 2, "n_global_sections": 2, "all_global_sections_glueable": bool(g1.metadata.get("pairwise_glueable", False) and g2.metadata.get("pairwise_glueable", False))}],
}
print(json.dumps(summary, sort_keys=True))
""")
        @test summary == py_summary
    end

    # ==================================================================
    # B3. Persistent-world / temporal semantics
    # ==================================================================
    @testset "Temporal Semantics" begin
        example = build_temporal_repair_example()
        plan = build_temporal_repair_compilation_plan(example)
        executed = execute_temporal_repair_example(example)
        summary = summarize_temporal_repair_example(example)

        @test summary["counts"]["raw_states"] == 3
        @test summary["bridge"]["endpoint_constraints"] == 2
        @test any(artifact -> artifact.subject_kind == :temporal_repair, plan.artifacts)
        @test any(artifact -> artifact.subject_kind == :temporal_schrodinger_bridge, plan.artifacts)
        @test any(occursin("repair_temporal_block", line) for line in executed.trace)
        @test any(occursin("instantiate_temporal_bridge", line) for line in executed.trace)

        py_summary = python_v1_summary("""
import json
from functorflow_v1.semantic_kernel import Category, Morphism
from functorflow_v1.persistent_world_models import (
    PersistentStateSpec, TemporalBlockSpec, TemporalRepairSpec,
    build_persistent_state_model_object, build_temporal_block_model,
    build_persistent_trajectory, build_temporal_repair,
)
from functorflow_v1.schrodinger_bridges import EndpointConstraint, SchrodingerBridgeSpec, build_temporal_schrodinger_bridge
state_category = Category("PersistentWorldStates")
year_category = Category("Years")
block_category = Category("TemporalBlocks")
bridge_category = Category("TemporalSchrodingerBridges")
r23 = build_persistent_state_model_object(PersistentStateSpec("AcmeRaw2023","acme",2023,("detect","pilot"),("supports","depends_on")), category=state_category)
r24 = build_persistent_state_model_object(PersistentStateSpec("AcmeRaw2024","acme",2024,("pilot","launch"),("supports","conflicts_with")), category=state_category)
r25 = build_persistent_state_model_object(PersistentStateSpec("AcmeRaw2025","acme",2025,("launch","expand"),("supports","amplifies")), category=state_category)
p23 = build_persistent_state_model_object(PersistentStateSpec("AcmeRepaired2023","acme",2023,("detect","pilot"),("supports","depends_on")), category=state_category)
p24 = build_persistent_state_model_object(PersistentStateSpec("AcmeRepaired2024","acme",2024,("pilot","stabilize","launch"),("supports","depends_on")), category=state_category)
p25 = build_persistent_state_model_object(PersistentStateSpec("AcmeRepaired2025","acme",2025,("launch","expand"),("supports","amplifies")), category=state_category)
rt1 = Morphism("AcmeRaw2023to2024", r23.object, r24.object, {"year_from": 2023, "year_to": 2024})
rt2 = Morphism("AcmeRaw2024to2025", r24.object, r25.object, {"year_from": 2024, "year_to": 2025})
pt1 = Morphism("AcmeRepaired2023to2024", p23.object, p24.object, {"year_from": 2023, "year_to": 2024})
pt2 = Morphism("AcmeRepaired2024to2025", p24.object, p25.object, {"year_from": 2024, "year_to": 2025})
raw = build_persistent_trajectory(name="AcmeRawTrajectory", company="acme", years=(2023,2024,2025), states=(r23,r24,r25), year_category=year_category, state_category=state_category, transition_maps=(rt1,rt2))
repaired = build_persistent_trajectory(name="AcmeRepairedTrajectory", company="acme", years=(2023,2024,2025), states=(p23,p24,p25), year_category=year_category, state_category=state_category, transition_maps=(pt1,pt2))
block = build_temporal_block_model(TemporalBlockSpec("AcmeTemporalBlock","acme",(2023,2024,2025)), category=block_category)
repair = build_temporal_repair(TemporalRepairSpec("AcmeTemporalRepair"), raw_trajectory=raw, repaired_trajectory=repaired, temporal_block=block)
bridge = build_temporal_schrodinger_bridge(SchrodingerBridgeSpec("AcmeTemporalBridge","synthetic_temporal_panel"), category=bridge_category, endpoint_constraints=(EndpointConstraint("acme",2023,2024,"train"), EndpointConstraint("acme",2024,2025,"eval")), linked_trajectories=(raw,repaired), summary_metrics={"csb_sde_mean":{"mse":0.12,"mae":0.08}, "conditional_flow":{"mse":0.09,"mae":0.06}})
summary = {
  "counts": {"raw_states": 3, "repaired_states": 3, "raw_trajectories": 1, "repaired_trajectories": 1, "temporal_blocks": 1, "temporal_repairs": 1},
  "bridge": {"name": bridge.name, "dataset_label": bridge.dataset_label, "reference_process": bridge.reference_process, "solver_family": bridge.solver_family, "bridge_method": bridge.bridge_method, "conditioning_scope": bridge.conditioning_scope, "endpoint_constraints": len(bridge.endpoint_constraints), "linked_trajectories": [trajectory.name for trajectory in bridge.linked_trajectories], "summary_metrics": bridge.summary_metrics},
  "company_summaries": [{"company": "acme", "years": [2023,2024,2025], "raw_trajectory": raw.name, "repaired_trajectory": repaired.name, "temporal_block": block.name, "temporal_repair": {"name": repair.name, "repair_map": repair.repair_map.name, "component_names": sorted(component.name for component in repair.repair_map.components), "repair_objective": repair.repair_objective}}],
}
print(json.dumps(summary, sort_keys=True))
""")
        @test summary == py_summary
    end

    # ==================================================================
    # B4. Workflow semantics
    # ==================================================================
    @testset "Workflow Semantics" begin
        example = build_agentic_workflow_example()
        plan = build_agentic_workflow_compilation_plan(example)
        executed = execute_agentic_workflow_example(example)
        summary = summarize_agentic_workflow_example(example)

        @test summary["counts"]["raw_workflows"] == 1
        @test summary["workflow_examples"][1]["changed"]
        @test any(artifact -> artifact.subject_kind == :agentic_workflow, plan.artifacts)
        @test any(artifact -> artifact.subject_kind == :rocket_workflow_refinement, plan.artifacts)
        @test any(occursin("declare_agentic_workflow", line) for line in executed.trace)
        @test any(occursin("refine_agentic_workflow", line) for line in executed.trace)

        py_summary = python_v1_summary("""
import json
from functorflow_v1.semantic_kernel import Category
from functorflow_v1.agentic_workflows import AgenticWorkflowSpec, ROCKETWorkflowRefinementSpec, build_agentic_workflow, build_rocket_workflow_refinement
workflow_category = Category("AgenticWorkflows")
step_index_category = Category("WorkflowStepIndex")
plan_state_category = Category("WorkflowPlanStates")
raw = build_agentic_workflow(AgenticWorkflowSpec("AcmeWorkflow","acme",2024,"stmt-001",("detect","prioritize","launch"),(("detect","prioritize"),("prioritize","launch")), (("detect","analysis"),("prioritize","ranking"),("launch","execution"))), workflow_category=workflow_category, step_index_category=step_index_category, plan_state_category=plan_state_category)
refined = build_agentic_workflow(AgenticWorkflowSpec("AcmeWorkflowRefined","acme",2024,"stmt-001",("detect","prioritize","pilot","launch"),(("detect","prioritize"),("prioritize","pilot"),("pilot","launch")), (("detect","analysis"),("prioritize","ranking"),("pilot","validation"),("launch","execution")), stage="rocket_refined", metadata={"changed": True, "selected_source": "local_merge", "selected_label": "pilot insertion", "score_gain": 0.27, "score_components": (("financial_alignment", 0.81), ("workflow_coherence", 0.89))}), workflow_category=workflow_category, step_index_category=step_index_category, plan_state_category=plan_state_category)
refinement = build_rocket_workflow_refinement(ROCKETWorkflowRefinementSpec("AcmeWorkflowRefinement", metadata={"changed": True, "selected_source": "local_merge", "selected_label": "pilot insertion", "score_gain": 0.27}), base_workflow=raw, refined_workflow=refined)
summary = {
  "counts": {"raw_workflows": 1, "refined_workflows": 1, "rocket_workflow_refinements": 1},
  "workflow_examples": [{
    "statement_id": refinement.base_workflow.statement_id,
    "company": refinement.base_workflow.company,
    "year": refinement.base_workflow.year,
    "changed": bool(refinement.metadata.get("changed", False)),
    "selected_source": refinement.metadata.get("selected_source", ""),
    "selected_label": refinement.metadata.get("selected_label", ""),
    "score_gain": float(refinement.metadata.get("score_gain", 0.0) or 0.0),
    "base_actions": list(refinement.base_workflow.actions),
    "selected_actions": list(refinement.refined_workflow.actions),
    "selected_score_components": {str(name): float(value) for name, value in refinement.refined_workflow.metadata.get("score_components", ())},
  }]
}
print(json.dumps(summary, sort_keys=True))
""")
        @test summary == py_summary
    end

    # ==================================================================
    # B5. Data-bridge semantics
    # ==================================================================
    @testset "Data Bridge Semantics" begin
        plan = build_data_bridge_compilation_plan()
        executed = execute_data_bridge_example()
        summary = summarize_data_bridge_example()

        @test summary["study_name"] == "red_wine_cardio_resveratrol"
        @test summary["truth_value_counts"]["CONSENSUS"] == 24
        @test any(artifact -> artifact.subject_kind == :categorical_db_bridge, plan.artifacts)
        @test any(artifact -> artifact.subject_kind == :intuitionistic_db_bridge, plan.artifacts)
        @test any(artifact -> artifact.subject_kind == :tcc_atlas_profile, plan.artifacts)
        @test any(occursin("declare_categorical_db_bridge", line) for line in executed.trace)
        @test any(occursin("materialize_tcc_pullback", line) for line in executed.trace)

        py_summary = python_v1_summary("""
import json
from pathlib import Path
from functorflow_v1.csql_objects import AtlasFileSet, AtlasSummary, SQLScriptSet, CSQLAtlasStudy
from functorflow_v1.csql_category import CSQLTableRef, CSQLObject, CSQLMorphism, CSQLPullbackConstruction, CSQLPushoutConstruction, RedWineCategoricalDBBridge
from functorflow_v1.categorical_db_bridge import CSQLTruthWitness, RedWineCSQLMaterialization
from functorflow_v1.tcc_atlas import TCCAtlasSpec, TCCEdgeWitness, TCCAtlasProfile
from functorflow_v1.tcc_method_pullback import TCCMethodPullbackWitness, TCCMethodConflictWitness, TCCMethodPullbackSummary
atlas_a = AtlasFileSet(Path("red_wine_cardio/atlas_cardio"), Path("atlas_cardio/nodes.parquet"), Path("atlas_cardio/edges.parquet"), Path("atlas_cardio/edge_support.parquet"))
atlas_b = AtlasFileSet(Path("red_wine_cardio/atlas_resveratrol"), Path("atlas_resveratrol/nodes.parquet"), Path("atlas_resveratrol/edges.parquet"), Path("atlas_resveratrol/edge_support.parquet"))
study = CSQLAtlasStudy("red_wine_cardio_resveratrol", Path("red_wine_cardio"), atlas_a, atlas_b, SQLScriptSet(Path("pullback_reconcile.sql"), Path("soft_atlas_pullback.sql"), Path("pushout_merge.sql")), AtlasSummary(nodes=120, edges=340, edge_support_rows=910, top_hub="resveratrol"), AtlasSummary(nodes=98, edges=275, edge_support_rows=822, top_hub="cardio"), {"bridge_prefix": "RedWine", "study_label": "red_wine"})
base = CSQLObject("claim_key_base", (CSQLTableRef("claim_key_base", "shared canonical claim interface", ("src","rel","dst"), {"semantic_role": "shared_claim_interface"}),), {"semantic_role": "shared_base"})
a_obj = CSQLObject("RedWineCardioAtlas", (CSQLTableRef("nodes_A","atlas_cardio/nodes.parquet",("node_id","label_canon")), CSQLTableRef("edges_A","atlas_cardio/edges.parquet",("edge_id","src_label_canon","rel_type","dst_label_canon"))), {"atlas_role": "cardio"})
b_obj = CSQLObject("RedWineResveratrolAtlas", (CSQLTableRef("nodes_B","atlas_resveratrol/nodes.parquet",("node_id","label_canon")), CSQLTableRef("edges_B","atlas_resveratrol/edges.parquet",("edge_id","src_label_canon","rel_type","dst_label_canon"))), {"atlas_role": "resveratrol"})
a_to_base = CSQLMorphism("RedWineCardioToBase", a_obj, base, ("src","rel","dst"), (("edges_A","claim_key_base"),), Path("pullback_reconcile.sql"))
b_to_base = CSQLMorphism("RedWineResveratrolToBase", b_obj, base, ("src","rel","dst"), (("edges_B","claim_key_base"),), Path("pullback_reconcile.sql"))
exact_output = CSQLObject("RedWineExactPullback", (CSQLTableRef("pullback_edges","pullback_edges",("src","rel","dst","score_sum_joint")),), {"construction_kind": "exact"})
soft_output = CSQLObject("RedWineSoftPullback", (CSQLTableRef("pullback_resv_soft","pullback_resv_soft",("srcA","rel","dstB","sim_dst")),), {"construction_kind": "soft"})
pushout_output = CSQLObject("RedWinePushout", (CSQLTableRef("pushout_edges","pushout_edges",("src","rel","dst","truth_value")),), {"construction_kind": "pushout"})
exact = CSQLPullbackConstruction("RedWineExactPullbackConstruction", a_obj, b_obj, base, a_to_base, b_to_base, exact_output, ("src","rel","dst"), Path("pullback_reconcile.sql"), "pullback_edges", "exact")
soft = CSQLPullbackConstruction("RedWineSoftPullbackConstruction", a_obj, b_obj, base, a_to_base, b_to_base, soft_output, ("rel","dst"), Path("soft_atlas_pullback.sql"), "pullback_resv_soft", "soft")
pushout = CSQLPushoutConstruction("RedWinePushoutConstruction", a_obj, b_obj, exact, pushout_output, Path("pushout_merge.sql"), "pushout_edges")
bridge = RedWineCategoricalDBBridge(study, base, a_obj, b_obj, a_to_base, b_to_base, exact, soft, pushout)
materialization = RedWineCSQLMaterialization(study, (("pullback_edges", 42), ("A_only_edges", 11), ("B_only_edges", 9)), (("CONSENSUS", 24), ("WEAK_CONSENSUS", 18), ("A_ONLY", 11), ("B_ONLY", 9)), (CSQLTruthWitness("CONSENSUS", "supports", "resveratrol", "heart_health", 1.73, 0.99, 5, 6), CSQLTruthWitness("A_ONLY", "contraindicates", "red_wine", "insomnia", 0.81, None, 3, None)))
profile = TCCAtlasProfile(TCCAtlasSpec("atlas_TCC","atlas_TCC","tcc","TCC",{"corpus_scale":"~45k papers"}), AtlasFileSet(Path("democritus_atlas/atlas_TCC"), Path("atlas_TCC/nodes.parquet"), Path("atlas_TCC/edges.parquet"), Path("atlas_TCC/edge_support.parquet")), CSQLObject("TCCAtlasObject",(CSQLTableRef("edges_tcc","atlas_TCC/edges.parquet",("src","rel","dst")),), {"study_label":"tcc"}), 1200, 5400, 12800, 3.4, 29, (("causes", 1700), ("improves", 820), ("reduces", 610)), ((2019, 1200), (2020, 1800), (2021, 2100)), (TCCEdgeWitness("minimum_wage","affects","employment",29,17.2), TCCEdgeWitness("education","improves","earnings",24,14.8)))
pullback = TCCMethodPullbackSummary(Path("tcc_workspace"), Path("tcc_data"), (("claims", 6400), ("did_claims", 1200), ("iv_claims", 950)), (TCCMethodPullbackWitness("minimum_wage","positive","employment",14,9,6.2,5.1),), (("CONSENSUS", 122), ("CONFLICT", 37)), (TCCMethodConflictWitness("minimum_wage","employment","did_vs_iv","conflict",11,2018,2023,8.4),))
summary = {
  "study_name": bridge.study.name,
  "base_object": bridge.base_object.name,
  "exact_pullback_table": bridge.exact_pullback.output_table,
  "soft_pullback_table": bridge.soft_pullback.output_table,
  "pushout_table": bridge.pushout.output_table,
  "truth_value_counts": {name: count for name, count in materialization.truth_value_counts},
  "omega_truth_values": ["CONSENSUS", "WEAK_CONSENSUS", "A_ONLY", "B_ONLY"],
  "tcc_profile": {"atlas_name": profile.spec.name, "node_count": profile.node_count, "edge_count": profile.edge_count, "top_edge_count": len(profile.top_edges)},
  "tcc_method_pullback": {"compiled_counts": {name: count for name, count in pullback.compiled_counts}, "pullback_rows": len(pullback.did_iv_pullback), "omega_counts": {name: count for name, count in pullback.omega_counts}, "method_conflicts": len(pullback.method_conflicts)},
}
print(json.dumps(summary, sort_keys=True))
        """)
        @test summary == py_summary
    end

    @testset "Concrete Data Bridge Materialization" begin
        study_summary = describe_named_csql_study(PY_V1_SAMPLE_ROOT, "red_wine_cardio_resveratrol")
        materialization_summary = describe_named_csql_materialization(PY_V1_SAMPLE_ROOT, "red_wine_cardio_resveratrol"; witness_limit=4)
        tcc_pullback_summary = describe_tcc_method_pullback(PY_V1_SAMPLE_ROOT; top_k=6, workspace_root=PY_V1_ROOT)

        @test study_summary["name"] == "red_wine_cardio_resveratrol"
        @test materialization_summary["truth_value_counts"]["CONSENSUS"] >= 1
        @test tcc_pullback_summary["compiled_counts"]["csql_nodes"] >= 1

        py_study_summary = python_v1_summary("""
import json
from functorflow_v1.csql_objects import describe_named_csql_study
print(json.dumps(describe_named_csql_study(r"$PY_V1_SAMPLE_ROOT", "red_wine_cardio_resveratrol"), sort_keys=True))
""")
        @test study_summary == py_study_summary

        py_materialization_summary = python_v1_summary("""
import json
from functorflow_v1.categorical_db_bridge import materialize_named_csql_results
from functorflow_v1.csql_category import build_named_csql_categorical_bridge
bridge = build_named_csql_categorical_bridge(r"$PY_V1_SAMPLE_ROOT", "red_wine_cardio_resveratrol")
materialization = materialize_named_csql_results(r"$PY_V1_SAMPLE_ROOT", "red_wine_cardio_resveratrol")
summary = {
  "study_name": bridge.study.name,
  "base_object": bridge.base_object.name,
  "atlas_a_object": bridge.atlas_a_object.name,
  "atlas_b_object": bridge.atlas_b_object.name,
  "exact_pullback_table": bridge.exact_pullback.output_table,
  "soft_pullback_table": bridge.soft_pullback.output_table,
  "pushout_table": bridge.pushout.output_table,
  "table_counts": {name: count for name, count in materialization.table_counts},
  "truth_value_counts": {name: count for name, count in materialization.truth_value_counts},
  "witnesses": [{
    "truth_value": witness.truth_value,
    "relation": witness.relation,
    "source": witness.source,
    "target": witness.target,
    "score_joint": witness.score_joint,
    "similarity": witness.similarity,
    "support_lcms_a": witness.support_lcms_a,
    "support_lcms_b": witness.support_lcms_b,
  } for witness in materialization.witnesses[:4]],
}
print(json.dumps(summary, sort_keys=True))
""")
        @test materialization_summary == py_materialization_summary

        py_tcc_pullback_summary = python_v1_summary("""
import json
from functorflow_v1.tcc_method_pullback import describe_tcc_method_pullback
summary = describe_tcc_method_pullback(r"$PY_V1_SAMPLE_ROOT", top_k=6, workspace_root=r"$PY_V1_ROOT")
print(json.dumps(summary, sort_keys=True))
""")
        @test tcc_pullback_summary == py_tcc_pullback_summary
    end

    # ==================================================================
    # B. Universal Constructions (universal.jl)
    # ==================================================================
    @testset "Universal Constructions" begin
        @testset "pullback" begin
            ket1 = ket_block(; name=:KET1)
            ket2 = ket_block(; name=:KET2)
            pb = pullback(ket1, ket2; over=:SharedContext)
            @test pb isa PullbackResult
            @test pb.cone isa FunctorFlow.Diagram
            @test pb.shared_object == :SharedContext
            @test pb.projection1 == :left
            @test pb.projection2 == :right
            @test length(pb.interface_morphisms) > 0
        end

        @testset "pushout" begin
            ket1 = ket_block(; name=:KET1)
            ket2 = ket_block(; name=:KET2)
            po = pushout(ket1, ket2; along=:SharedBase)
            @test po isa PushoutResult
            @test po.cocone isa FunctorFlow.Diagram
            @test po.shared_object == :SharedBase
        end

        @testset "product" begin
            D1 = ket_block(; name=:A)
            D2 = ket_block(; name=:B)
            D3 = ket_block(; name=:C)

            prod2 = product(D1, D2)
            @test prod2 isa ProductResult
            @test length(prod2.projections) == 2

            prod3 = product(D1, D2, D3)
            @test length(prod3.projections) == 3
        end

        @testset "coproduct" begin
            D1 = ket_block(; name=:A)
            D2 = ket_block(; name=:B)
            coprod = coproduct(D1, D2)
            @test coprod isa CoproductResult
            @test length(coprod.injections) == 2
        end

        @testset "equalizer" begin
            D = db_square(; first_impl=x -> x * 2, second_impl=x -> x + 1)
            eq = equalizer(D, :f, :g)
            @test eq isa EqualizerResult
            @test eq.equalizer_map isa Symbol
        end

        @testset "coequalizer" begin
            D = Diagram(:ParallelPair)
            add_object!(D, :A; kind=:state)
            add_object!(D, :B; kind=:state)
            add_morphism!(D, :f, :A, :B)
            add_morphism!(D, :g, :A, :B)

            coeq = coequalizer(D, :f, :g)
            @test coeq isa CoequalizerResult
            @test coeq.coequalizer_map isa Symbol
            @test coeq.quotient_object isa Symbol
            @test haskey(coeq.coequalizer_diagram.objects, coeq.quotient_object)
            @test haskey(coeq.coequalizer_diagram.operations, coeq.coequalizer_map)
            @test !isempty(coeq.coequalizer_diagram.losses)
        end

        @testset "verify pullback" begin
            pb = pullback(ket_block(; name=:A), ket_block(; name=:B); over=:Shared)
            v = verify(pb)
            @test v.construction == :pullback
            @test v.passed isa Bool
            @test haskey(v.checks, :has_left_factor)
            @test haskey(v.checks, :has_right_factor)
            @test haskey(v.checks, :has_shared_object)
        end

        @testset "verify pushout" begin
            po = pushout(ket_block(; name=:A), ket_block(; name=:B); along=:Shared)
            v = verify(po)
            @test v.construction == :pushout
            @test v.passed isa Bool
        end

        @testset "verify product" begin
            prod = product(ket_block(; name=:A), ket_block(; name=:B))
            v = verify(prod)
            @test v.construction == :product
            @test v.passed isa Bool
        end

        @testset "verify coproduct" begin
            coprod = coproduct(ket_block(; name=:A), ket_block(; name=:B))
            v = verify(coprod)
            @test v.construction == :coproduct
            @test v.passed isa Bool
        end

        @testset "verify equalizer" begin
            D = db_square(; first_impl=x -> x * 2, second_impl=x -> x + 1)
            eq = equalizer(D, :f, :g)
            v = verify(eq)
            @test v.construction == :equalizer
            @test haskey(v.checks, :has_equalizer_map)
        end

        @testset "verify coequalizer" begin
            D = Diagram(:PP)
            add_object!(D, :A; kind=:state)
            add_object!(D, :B; kind=:state)
            add_morphism!(D, :f, :A, :B)
            add_morphism!(D, :g, :A, :B)
            coeq = coequalizer(D, :f, :g)
            v = verify(coeq)
            @test v.construction == :coequalizer
            @test v.passed
            @test haskey(v.checks, :has_quotient_object)
            @test haskey(v.checks, :has_coequalizer_map)
            @test haskey(v.checks, :map_targets_quotient)
            @test haskey(v.checks, :has_coeq_loss)
        end

        @testset "compile_construction" begin
            pb = pullback(ket_block(; name=:A), ket_block(; name=:B); over=:Shared)
            compiled = compile_construction(pb)
            @test compiled isa FunctorFlow.CompiledDiagram
        end

        @testset "universal_morphism" begin
            pb = pullback(ket_block(; name=:A), ket_block(; name=:B); over=:Shared)
            D_cone = ket_block(; name=:Cone)
            um = universal_morphism(pb, D_cone)
            @test um isa FunctorFlow.Diagram
        end
    end

    # ==================================================================
    # C. Causal Semantics (causal.jl)
    # ==================================================================
    @testset "Causal Semantics" begin
        @testset "CausalContext" begin
            ctx = CausalContext(:test;
                                observational_regime=:obs,
                                interventional_regime=:do)
            @test ctx.name == :test
            @test ctx.observational_regime == :obs
            @test ctx.interventional_regime == :do
        end

        @testset "build_causal_diagram" begin
            ctx = CausalContext(:test; observational_regime=:obs, interventional_regime=:do)
            cd = build_causal_diagram(:CausalTest; context=ctx)
            @test cd isa CausalDiagram
            @test cd.base_diagram isa FunctorFlow.Diagram
            @test haskey(cd.base_diagram.operations, :intervene)
            @test haskey(cd.base_diagram.operations, :condition)
            @test cd.base_diagram.operations[:intervene].direction == FunctorFlow.LEFT
            @test cd.base_diagram.operations[:condition].direction == FunctorFlow.RIGHT
        end

        @testset "causal_transport" begin
            ctx = CausalContext(:test; observational_regime=:obs, interventional_regime=:do)
            cd_src = build_causal_diagram(:Src; context=ctx)
            cd_tgt = build_causal_diagram(:Tgt; context=ctx)

            # Without density ratio
            D = causal_transport(cd_src, cd_tgt)
            @test D isa FunctorFlow.Diagram

            # With density ratio function
            density = x -> x * 1.5
            D2 = causal_transport(cd_src, cd_tgt; density_ratio=density)
            @test D2 isa FunctorFlow.Diagram
            @test haskey(D2.objects, :DensityRatio)
            @test haskey(D2.operations, :rn_reweight)
        end

        @testset "interventional_expectation" begin
            ctx = CausalContext(:test; observational_regime=:obs, interventional_regime=:do)
            cd = build_causal_diagram(:IntExp; context=ctx)
            # Provide data matching diagram objects
            obs_data = Dict(
                :Observations => Dict(:a => 1.0, :b => 2.0),
                :CausalStructure => Dict((:a, :b) => true)
            )
            result = interventional_expectation(cd, obs_data)
            @test result isa Dict
            @test haskey(result, :all_values)
        end

        @testset "is_identifiable" begin
            ctx = CausalContext(:test; observational_regime=:obs, interventional_regime=:do)
            cd = build_causal_diagram(:Ident; context=ctx)
            result = is_identifiable(cd, :Y)
            @test result isa NamedTuple
            @test haskey(result, :identifiable)
            @test haskey(result, :rule)
            @test haskey(result, :reasoning)
        end
    end

    # ==================================================================
    # D. Topos Foundations (topos.jl)
    # ==================================================================
    @testset "Topos Foundations" begin
        @testset "SubobjectClassifier" begin
            omega = SubobjectClassifier(:Omega)
            @test omega.truth_object == :Omega
            @test omega.true_map == :true_map
            @test Symbol("true") in omega.truth_values
            @test Symbol("false") in omega.truth_values
        end

        @testset "SheafSection" begin
            sec = SheafSection(:s1; base_space=:X, section_data=[1, 2, 3])
            @test sec.name == :s1
            @test sec.base_space == :X
            @test sec.section_data == [1, 2, 3]
        end

        @testset "InternalPredicate" begin
            omega = SubobjectClassifier(:Omega)
            pred = InternalPredicate(:positive, omega;
                                      characteristic_map=x -> x > 0)
            @test pred.name == :positive
            @test pred.classifier === omega
            @test pred.characteristic_map !== nothing
        end

        @testset "build_sheaf_diagram" begin
            sec1 = SheafSection(:s1; base_space=:X, section_data=[1, 2, 3])
            sec2 = SheafSection(:s2; base_space=:X, section_data=[2, 3, 4])
            D = build_sheaf_diagram([sec1, sec2]; name=:GlueTest)
            @test haskey(D.operations, :glue)
            @test D.operations[:glue].reducer == :set_union
        end

        @testset "evaluate_predicate" begin
            omega = SubobjectClassifier(:Omega)
            pred = InternalPredicate(:positive, omega;
                                      characteristic_map=x -> x > 0)

            @test evaluate_predicate(pred, 5) == Symbol("true")
            @test evaluate_predicate(pred, -1) == Symbol("false")
            @test evaluate_predicate(pred, 0) == Symbol("false")

            # No characteristic map
            pred2 = InternalPredicate(:empty, omega)
            @test_throws ArgumentError evaluate_predicate(pred2, 1)
        end

        @testset "classify_subobject" begin
            omega = SubobjectClassifier(:Omega)

            # Dict data
            data = Dict(:a => 5, :b => -3, :c => 10)
            cl = classify_subobject(omega, x -> x > 0, data)
            @test cl[:a] == Symbol("true")
            @test cl[:b] == Symbol("false")
            @test cl[:c] == Symbol("true")

            # Vector data
            vec_data = [1, -2, 3, 0, -5]
            cl2 = classify_subobject(omega, x -> x > 0, vec_data)
            @test cl2[1] == Symbol("true")
            @test cl2[2] == Symbol("false")
            @test cl2[3] == Symbol("true")
            @test cl2[4] == Symbol("false")
            @test cl2[5] == Symbol("false")
        end

        @testset "check_coherence" begin
            sec1 = SheafSection(:s1; base_space=:X, section_data=[1, 2, 3],
                                domain=Set([:a, :b, :c]))
            sec2 = SheafSection(:s2; base_space=:X, section_data=[2, 3, 4],
                                domain=Set([:b, :c, :d]))
            check = SheafCoherenceCheck([sec1, sec2])
            result = check_coherence(check)
            @test result isa NamedTuple
            @test haskey(result, :passed)
            @test haskey(result, :locality)
            @test haskey(result, :gluing)
            @test haskey(result, :stability)
        end

        @testset "internal logic operators" begin
            omega = SubobjectClassifier(:Omega)
            p = InternalPredicate(:pos, omega; characteristic_map=x -> x > 0)
            q = InternalPredicate(:even, omega; characteristic_map=x -> x % 2 == 0)

            # AND
            pq = internal_and(p, q)
            @test pq.characteristic_map(4) == true
            @test pq.characteristic_map(-2) == false
            @test pq.characteristic_map(3) == false

            # OR
            p_or_q = internal_or(p, q)
            @test p_or_q.characteristic_map(4) == true
            @test p_or_q.characteristic_map(-2) == true
            @test p_or_q.characteristic_map(3) == true
            @test p_or_q.characteristic_map(-3) == false

            # NOT
            not_p = internal_not(p)
            @test not_p.characteristic_map(5) == false
            @test not_p.characteristic_map(-1) == true
        end
    end

    # ==================================================================
    # E. Proof Interface (proof_interface.jl)
    # ==================================================================
    @testset "Proof Interface — Construction Certificates" begin
        @testset "pullback certificate" begin
            pb = pullback(ket_block(; name=:A), ket_block(; name=:B); over=:Shared)
            cert = render_construction_certificate(pb)
            @test cert isa String
            @test occursin("theorem", cert)
            @test occursin("pullback", lowercase(cert))
        end

        @testset "pushout certificate" begin
            po = pushout(ket_block(; name=:A), ket_block(; name=:B); along=:Shared)
            cert = render_construction_certificate(po)
            @test cert isa String
            @test occursin("pushout", lowercase(cert))
        end

        @testset "product certificate" begin
            prod = product(ket_block(; name=:A), ket_block(; name=:B))
            cert = render_construction_certificate(prod)
            @test cert isa String
            @test occursin("product", lowercase(cert))
        end

        @testset "coproduct certificate" begin
            coprod = coproduct(ket_block(; name=:A), ket_block(; name=:B))
            cert = render_construction_certificate(coprod)
            @test cert isa String
            @test occursin("coproduct", lowercase(cert))
        end

        @testset "equalizer certificate" begin
            D = db_square(; first_impl=x -> x * 2, second_impl=x -> x + 1)
            eq = equalizer(D, :f, :g)
            cert = render_construction_certificate(eq)
            @test cert isa String
            @test occursin("equalizer", lowercase(cert))
        end

        @testset "coequalizer certificate" begin
            D = Diagram(:PP)
            add_object!(D, :A; kind=:state)
            add_object!(D, :B; kind=:state)
            add_morphism!(D, :f, :A, :B)
            add_morphism!(D, :g, :A, :B)
            coeq = coequalizer(D, :f, :g)
            cert = render_construction_certificate(coeq)
            @test cert isa String
            @test occursin("coequalizer", lowercase(cert))
        end

        @testset "lean certificate roundtrip" begin
            D = ket_block()
            cert = render_lean_certificate(D)
            @test cert isa String
            @test occursin("structure", cert) || occursin("def", cert)
        end
    end
end
