# ============================================================================
# data_bridges/examples.jl — In-memory example builders + cross-cutting compilation plan / executable IR / summarization helpers.
# ============================================================================
#
# Sourced from the original src/data_bridges.jl as part of the v0.3.4 split.
# All public symbols are unchanged; this file only contains existing code.
# ============================================================================


function practical_csql_truth_values()
    [
        SCMTruthValue(:CONSENSUS, "exact pullback support in both atlases"),
        SCMTruthValue(:WEAK_CONSENSUS, "shared relation with soft target agreement"),
        SCMTruthValue(:A_ONLY, "supported only by atlas A"),
        SCMTruthValue(:B_ONLY, "supported only by atlas B"),
    ]
end

function build_categorical_db_bridge_example()
    atlas_a = AtlasFileSet("red_wine_cardio/atlas_cardio", "atlas_cardio/nodes.parquet", "atlas_cardio/edges.parquet", "atlas_cardio/edge_support.parquet")
    atlas_b = AtlasFileSet("red_wine_cardio/atlas_resveratrol", "atlas_resveratrol/nodes.parquet", "atlas_resveratrol/edges.parquet", "atlas_resveratrol/edge_support.parquet")
    study = CSQLAtlasStudy("red_wine_cardio_resveratrol", "red_wine_cardio", atlas_a, atlas_b,
        SQLScriptSet("pullback_reconcile.sql", "soft_atlas_pullback.sql", "pushout_merge.sql"),
        AtlasSummary(nodes=120, edges=340, edge_support_rows=910, top_hub="resveratrol"),
        AtlasSummary(nodes=98, edges=275, edge_support_rows=822, top_hub="cardio");
        metadata=Dict(:bridge_prefix => "RedWine", :study_label => "red_wine"))

    base = CSQLObject(:claim_key_base, [CSQLTableRef(:claim_key_base, "shared canonical claim interface", [:src, :rel, :dst];
                                                     metadata=Dict(:semantic_role => :shared_claim_interface))];
                      metadata=Dict(:semantic_role => :shared_base))
    atlas_a_object = CSQLObject(:RedWineCardioAtlas, [
        CSQLTableRef(:nodes_A, "atlas_cardio/nodes.parquet", [:node_id, :label_canon]),
        CSQLTableRef(:edges_A, "atlas_cardio/edges.parquet", [:edge_id, :src_label_canon, :rel_type, :dst_label_canon]),
    ]; metadata=Dict(:atlas_role => :cardio))
    atlas_b_object = CSQLObject(:RedWineResveratrolAtlas, [
        CSQLTableRef(:nodes_B, "atlas_resveratrol/nodes.parquet", [:node_id, :label_canon]),
        CSQLTableRef(:edges_B, "atlas_resveratrol/edges.parquet", [:edge_id, :src_label_canon, :rel_type, :dst_label_canon]),
    ]; metadata=Dict(:atlas_role => :resveratrol))

    a_to_base = CSQLMorphism(:RedWineCardioToBase, atlas_a_object, base, [:src, :rel, :dst], [(:edges_A, :claim_key_base)];
                             sql_reference="pullback_reconcile.sql")
    b_to_base = CSQLMorphism(:RedWineResveratrolToBase, atlas_b_object, base, [:src, :rel, :dst], [(:edges_B, :claim_key_base)];
                             sql_reference="pullback_reconcile.sql")
    exact_output = CSQLObject(:RedWineExactPullback, [CSQLTableRef(:pullback_edges, "pullback_edges", [:src, :rel, :dst, :score_sum_joint])];
                              metadata=Dict(:construction_kind => :exact))
    soft_output = CSQLObject(:RedWineSoftPullback, [CSQLTableRef(:pullback_resv_soft, "pullback_resv_soft", [:srcA, :rel, :dstB, :sim_dst])];
                             metadata=Dict(:construction_kind => :soft))
    pushout_output = CSQLObject(:RedWinePushout, [CSQLTableRef(:pushout_edges, "pushout_edges", [:src, :rel, :dst, :truth_value])];
                                metadata=Dict(:construction_kind => :pushout))

    exact_pullback = CSQLPullbackConstruction(:RedWineExactPullbackConstruction, atlas_a_object, atlas_b_object, base,
        a_to_base, b_to_base, exact_output, [:src, :rel, :dst], "pullback_reconcile.sql", :pullback_edges;
        construction_kind="exact")
    soft_pullback = CSQLPullbackConstruction(:RedWineSoftPullbackConstruction, atlas_a_object, atlas_b_object, base,
        a_to_base, b_to_base, soft_output, [:rel, :dst], "soft_atlas_pullback.sql", :pullback_resv_soft;
        construction_kind="soft")
    pushout = CSQLPushoutConstruction(:RedWinePushoutConstruction, atlas_a_object, atlas_b_object, exact_pullback,
        pushout_output, "pushout_merge.sql", :pushout_edges)

    CategoricalDBBridge(study, base, atlas_a_object, atlas_b_object, a_to_base, b_to_base, exact_pullback, soft_pullback, pushout,
                        Dict(:study_label => "red_wine"))
end

function build_intuitionistic_db_bridge_example(cbridge::Union{Nothing, CategoricalDBBridge}=nothing)
    cbridge = cbridge === nothing ? build_categorical_db_bridge_example() : cbridge
    materialization = CSQLMaterialization(
        cbridge.study,
        [("pullback_edges", 42), ("A_only_edges", 11), ("B_only_edges", 9)],
        [("CONSENSUS", 24), ("WEAK_CONSENSUS", 18), ("A_ONLY", 11), ("B_ONLY", 9)],
        [
            CSQLTruthWitness("CONSENSUS", "supports", "resveratrol", "heart_health", 1.73; similarity=0.99, support_lcms_a=5, support_lcms_b=6),
            CSQLTruthWitness("A_ONLY", "contraindicates", "red_wine", "insomnia", 0.81; support_lcms_a=3),
        ],
        Dict(:materialized_from => "synthetic_example"),
    )

    scm = build_scm_model_object(
        SCMObjectSpec(:RedWineBridgeSCM, [:atlas_a_support, :atlas_b_support], [:claim_alignment], [
            SCMLocalFunctionSpec(:f_claim_alignment, :claim_alignment;
                exogenous_parents=[:atlas_a_support, :atlas_b_support],
                expression="claim_alignment := reconcile(atlas_a_support, atlas_b_support)")
        ]);
        category=:CSQLSCM)
    omega = build_omega_scm(category=:CSQLSCM, truth_values=practical_csql_truth_values())
    consensus = build_scm_predicate(name=:ConsensusClaimPredicate, ambient_scm=scm,
        clauses=[SCMPredicateClause(:consensus_claim, "claim has exact support in both atlases"; clause_kind=:consensus)])
    weak = build_scm_predicate(name=:WeakConsensusPredicate, ambient_scm=scm,
        clauses=[SCMPredicateClause(:weak_consensus_claim, "claim has soft support across atlases"; clause_kind=:weak_consensus)])
    a_only = build_scm_predicate(name=:AtlasAOnlyPredicate, ambient_scm=scm,
        clauses=[SCMPredicateClause(:atlas_a_only, "claim is supported only in atlas A"; clause_kind=:exclusive_support)])
    b_only = build_scm_predicate(name=:AtlasBOnlyPredicate, ambient_scm=scm,
        clauses=[SCMPredicateClause(:atlas_b_only, "claim is supported only in atlas B"; clause_kind=:exclusive_support)])

    consensus_classifier = build_scm_characteristic_map(name=:chi_ConsensusClaim, ambient_scm=scm, predicate=consensus, omega=omega,
        classifying_truth_value=:CONSENSUS, false_truth_value=:B_ONLY)
    weak_classifier = build_scm_characteristic_map(name=:chi_WeakConsensusClaim, ambient_scm=scm, predicate=weak, omega=omega,
        classifying_truth_value=:WEAK_CONSENSUS, false_truth_value=:B_ONLY)
    a_only_classifier = build_scm_characteristic_map(name=:chi_AtlasAOnlyClaim, ambient_scm=scm, predicate=a_only, omega=omega,
        classifying_truth_value=:A_ONLY, false_truth_value=:CONSENSUS)
    b_only_classifier = build_scm_characteristic_map(name=:chi_AtlasBOnlyClaim, ambient_scm=scm, predicate=b_only, omega=omega,
        classifying_truth_value=:B_ONLY, false_truth_value=:CONSENSUS)

    IntuitionisticDBBridge(cbridge.study, cbridge, materialization, scm, omega,
                           consensus, weak, a_only, b_only,
                           consensus_classifier, weak_classifier, a_only_classifier, b_only_classifier,
                           Dict(:bridge_label => "red_wine_intuitionistic"))
end

function build_tcc_examples()
    atlas = AtlasFileSet("democritus_atlas/atlas_TCC", "atlas_TCC/nodes.parquet", "atlas_TCC/edges.parquet", "atlas_TCC/edge_support.parquet")
    object = CSQLObject(:TCCAtlasObject, [CSQLTableRef(:edges_tcc, "atlas_TCC/edges.parquet", [:src, :rel, :dst])];
                        metadata=Dict(:study_label => "tcc"))
    profile = TCCAtlasProfile(
        TCCAtlasSpec("atlas_TCC", "atlas_TCC", "tcc", "TCC"; metadata=Dict(:corpus_scale => "~45k papers")),
        atlas,
        object,
        1200,
        5400,
        12800,
        3.4,
        29,
        [("causes", 1700), ("improves", 820), ("reduces", 610)],
        [(2019, 1200), (2020, 1800), (2021, 2100)],
        [
            TCCEdgeWitness("minimum_wage", "affects", "employment", 29, 17.2),
            TCCEdgeWitness("education", "improves", "earnings", 24, 14.8),
        ],
        Dict(:profile_label => "tcc_single_atlas"),
    )
    pullback = TCCMethodPullbackSummary(
        "tcc_workspace",
        "tcc_data",
        [("claims", 6400), ("did_claims", 1200), ("iv_claims", 950)],
        [TCCMethodPullbackWitness("minimum_wage", "positive", "employment", 14, 9, 6.2, 5.1)],
        [("CONSENSUS", 122), ("CONFLICT", 37)],
        [TCCMethodConflictWitness("minimum_wage", "employment", "did_vs_iv", "conflict", 11, 2018, 2023, 8.4)],
        Dict(:summary_label => "tcc_method_pullback"),
    )
    Dict(:atlas_profile => profile, :method_pullback => pullback)
end

function build_data_bridge_compilation_plan()
    cbridge = build_categorical_db_bridge_example()
    ibridge = build_intuitionistic_db_bridge_example(cbridge)
    tcc = build_tcc_examples()
    compile_plan(:DataBridgeExamplePlan,
        cbridge.base_object,
        cbridge.atlas_a_object,
        cbridge.atlas_b_object,
        cbridge.atlas_a_to_base,
        cbridge.atlas_b_to_base,
        cbridge.exact_pullback,
        cbridge.soft_pullback,
        cbridge.pushout,
        cbridge,
        ibridge,
        ibridge.bridge_scm,
        ibridge.omega,
        ibridge.consensus_predicate,
        ibridge.consensus_classifier,
        tcc[:atlas_profile],
        tcc[:method_pullback];
        metadata=Dict(:example => "data_bridges"))
end

build_data_bridge_executable_ir() = lower_plan_to_executable_ir(build_data_bridge_compilation_plan())
execute_data_bridge_example() = execute_placeholder_ir(build_data_bridge_executable_ir())

function summarize_data_bridge_example()
    cbridge = build_categorical_db_bridge_example()
    ibridge = build_intuitionistic_db_bridge_example(cbridge)
    tcc = build_tcc_examples()
    Dict(
        "study_name" => cbridge.study.name,
        "base_object" => String(cbridge.base_object.name),
        "exact_pullback_table" => String(cbridge.exact_pullback.output_table),
        "soft_pullback_table" => String(cbridge.soft_pullback.output_table),
        "pushout_table" => String(cbridge.pushout.output_table),
        "truth_value_counts" => Dict(name => count for (name, count) in ibridge.materialization.truth_value_counts),
        "omega_truth_values" => [String(value.name) for value in ibridge.omega.truth_values],
        "tcc_profile" => Dict(
            "atlas_name" => tcc[:atlas_profile].spec.name,
            "node_count" => tcc[:atlas_profile].node_count,
            "edge_count" => tcc[:atlas_profile].edge_count,
            "top_edge_count" => length(tcc[:atlas_profile].top_edges),
        ),
        "tcc_method_pullback" => Dict(
            "compiled_counts" => Dict(name => count for (name, count) in tcc[:method_pullback].compiled_counts),
            "pullback_rows" => length(tcc[:method_pullback].did_iv_pullback),
            "omega_counts" => Dict(name => count for (name, count) in tcc[:method_pullback].omega_counts),
            "method_conflicts" => length(tcc[:method_pullback].method_conflicts),
        ),
    )
end
