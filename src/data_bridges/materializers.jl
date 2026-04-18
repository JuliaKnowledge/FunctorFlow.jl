# ============================================================================
# data_bridges/materializers.jl — Bridge constructors and SQL materializers: locate_/describe_/build_named_csql_*, _render_*_pullback_sql, locate_/materialize_/describe_tcc_*, plus DuckDB helpers.
# ============================================================================
#
# Sourced from the original src/data_bridges.jl as part of the v0.3.4 split.
# All public symbols are unchanged; this file only contains existing code.
# ============================================================================

function locate_named_csql_study(root::AbstractString, name::AbstractString)
    spec = _atlas_pair_study_spec_named(name)
    atlas_root = _require_existing_path(joinpath(abspath(root), "democritus_atlas"))
    atlas_a = _build_pair_atlas_fileset(joinpath(atlas_root, spec.atlas_a_dir))
    atlas_b = _build_pair_atlas_fileset(joinpath(atlas_root, spec.atlas_b_dir))
    scripts = SQLScriptSet(
        _require_existing_path(joinpath(atlas_root, spec.exact_pullback_sql)),
        _require_existing_path(joinpath(atlas_root, spec.soft_pullback_sql)),
        _require_existing_path(joinpath(atlas_root, spec.pushout_sql)),
    )
    CSQLAtlasStudy(
        spec.name,
        atlas_root,
        atlas_a,
        atlas_b,
        scripts,
        parse_atlas_summary(atlas_a.summary_markdown),
        parse_atlas_summary(atlas_b.summary_markdown);
        metadata=merge(
            Dict{Symbol, Any}(
                :atlas_a_role => spec.atlas_a_role,
                :atlas_b_role => spec.atlas_b_role,
                :focus_terms => copy(spec.focus_terms),
                :bridge_prefix => spec.bridge_prefix,
                :study_label => spec.study_label,
            ),
            spec.metadata,
        ),
    )
end

locate_red_wine_csql_study(root::AbstractString) = locate_named_csql_study(root, "red_wine_cardio_resveratrol")
locate_tylenol_csql_study(root::AbstractString) = locate_named_csql_study(root, "tylenol_lancet_paracetamol")

function describe_named_csql_study(root::AbstractString, name::AbstractString)
    study = locate_named_csql_study(root, name)
    Dict(
        "name" => study.name,
        "atlas_a" => Dict(
            "name" => String(_atlas_dir_name(study.atlas_a)),
            "role" => String(study.metadata[:atlas_a_role]),
            "nodes" => study.summary_a.nodes,
            "edges" => study.summary_a.edges,
            "edge_support_rows" => study.summary_a.edge_support_rows,
            "top_hub" => study.summary_a.top_hub,
        ),
        "atlas_b" => Dict(
            "name" => String(_atlas_dir_name(study.atlas_b)),
            "role" => String(study.metadata[:atlas_b_role]),
            "nodes" => study.summary_b.nodes,
            "edges" => study.summary_b.edges,
            "edge_support_rows" => study.summary_b.edge_support_rows,
            "top_hub" => study.summary_b.top_hub,
        ),
        "scripts" => Dict(
            "exact_pullback_sql" => study.scripts.exact_pullback_sql,
            "soft_pullback_sql" => study.scripts.soft_pullback_sql,
            "pushout_sql" => study.scripts.pushout_sql,
        ),
        "tables" => Dict(
            "exact_pullback" => String(get(study.metadata, :exact_pullback_table, "pullback_edges")),
            "soft_pullback" => String(get(study.metadata, :soft_pullback_table, "pullback_resv_soft")),
            "pushout" => String(get(study.metadata, :pushout_table, "pushout_edges")),
        ),
        "focus_terms" => copy(String.(get(study.metadata, :focus_terms, String[]))),
    )
end

describe_red_wine_csql_study(root::AbstractString) = describe_named_csql_study(root, "red_wine_cardio_resveratrol")
describe_tylenol_csql_study(root::AbstractString) = describe_named_csql_study(root, "tylenol_lancet_paracetamol")

_camelize(text::AbstractString) = join(uppercasefirst.(split(String(text), "_")))

function _bridge_prefix(study::CSQLAtlasStudy)
    String(get(study.metadata, :bridge_prefix, _camelize(study.name)))
end

function _atlas_object_name(study::CSQLAtlasStudy, role_key::Symbol, default_label::AbstractString)
    string(_bridge_prefix(study), _camelize(String(study.metadata[role_key])), default_label)
end

function _base_tables()
    [
        CSQLTableRef("claim_key_base", "shared canonical claim interface", [:src, :rel, :dst];
            metadata=Dict(:semantic_role => "shared_claim_interface"))
    ]
end

function _atlas_tables(role::AbstractString, study_root_name::AbstractString, atlas_dir_name::AbstractString)
    prefix = role in ("cardio", "nyt_tylenol") ? "A" : "B"
    [
        CSQLTableRef("nodes_$prefix", "$study_root_name/$atlas_dir_name/atlas_nodes.parquet", [:node_id, :label_canon];
            metadata=Dict(:semantic_role => "atlas_nodes", :atlas_role => role)),
        CSQLTableRef("edges_$prefix", "$study_root_name/$atlas_dir_name/atlas_edges.parquet",
            [:edge_id, :src_label_canon, :rel_type, :dst_label_canon, :support_docs, :support_lcms, :score_sum, :controversy];
            metadata=Dict(:semantic_role => "atlas_edges", :atlas_role => role)),
        CSQLTableRef("supp_$prefix", "$study_root_name/$atlas_dir_name/atlas_edge_support.parquet",
            [:edge_id, :doc_id, :lcm_instance_id, :focus, :radius, :score, :score_raw, :coupling];
            metadata=Dict(:semantic_role => "atlas_edge_support", :atlas_role => role)),
        CSQLTableRef("key_$prefix", "VIEW key_$prefix",
            [Symbol("edge_id_$prefix"), :src, :rel, :dst, Symbol("support_docs_$prefix"), Symbol("support_lcms_$prefix"), Symbol("score_sum_$prefix"), Symbol("controversy_$prefix")];
            metadata=Dict(:semantic_role => "canonical_claim_key", :atlas_role => role)),
    ]
end

function build_named_csql_categorical_bridge(root::AbstractString, study_name::AbstractString)
    study = locate_named_csql_study(root, study_name)
    prefix = _bridge_prefix(study)
    role_a = String(study.metadata[:atlas_a_role])
    role_b = String(study.metadata[:atlas_b_role])
    study_root_name = splitdir(study.root)[2]
    atlas_a_name = _atlas_dir_name(study.atlas_a)
    atlas_b_name = _atlas_dir_name(study.atlas_b)

    base_object = CSQLObject("$(prefix)ClaimKeyBase", _base_tables();
        metadata=Dict(:shared_key_fields => [:src, :rel, :dst]))
    atlas_a_object = CSQLObject(_atlas_object_name(study, :atlas_a_role, "DBObject"),
        _atlas_tables(role_a, study_root_name, atlas_a_name);
        metadata=Dict(:atlas_role => role_a, :top_hub => study.summary_a.top_hub))
    atlas_b_object = CSQLObject(_atlas_object_name(study, :atlas_b_role, "DBObject"),
        _atlas_tables(role_b, study_root_name, atlas_b_name);
        metadata=Dict(:atlas_role => role_b, :top_hub => study.summary_b.top_hub))

    atlas_a_to_base = CSQLMorphism("$(String(atlas_a_object.name))ToClaimBase", atlas_a_object, base_object,
        [:src, :rel, :dst], [(:key_A, :claim_key_base)];
        sql_reference=study.scripts.exact_pullback_sql,
        metadata=Dict(:semantic_role => "canonical_claim_projection", :atlas_role => role_a))
    atlas_b_to_base = CSQLMorphism("$(String(atlas_b_object.name))ToClaimBase", atlas_b_object, base_object,
        [:src, :rel, :dst], [(:key_B, :claim_key_base)];
        sql_reference=study.scripts.exact_pullback_sql,
        metadata=Dict(:semantic_role => "canonical_claim_projection", :atlas_role => role_b))

    exact_pullback_output = CSQLObject("$(prefix)ExactPullbackDB", [
        CSQLTableRef("pullback_edges", "TABLE pullback_edges",
            [:src, :rel, :dst, :edge_id_A, :edge_id_B, :support_docs_A, :support_docs_B, :support_lcms_A, :support_lcms_B, :score_sum_A, :score_sum_B, :score_sum_joint, :controversy_joint];
            metadata=Dict(:semantic_role => "exact_pullback_output")),
        CSQLTableRef("A_only_edges", "TABLE A_only_edges", [:edge_id_A, :src, :rel, :dst];
            metadata=Dict(:semantic_role => "pullback_left_difference")),
        CSQLTableRef("B_only_edges", "TABLE B_only_edges", [:edge_id_B, :src, :rel, :dst];
            metadata=Dict(:semantic_role => "pullback_right_difference")),
        CSQLTableRef("pullback_evidence", "VIEW pullback_evidence", [:src, :rel, :dst, :which, :doc_id, :lcm_instance_id];
            metadata=Dict(:semantic_role => "pullback_evidence_panel")),
    ]; metadata=Dict(:construction_kind => "exact_pullback"))

    exact_pullback = CSQLPullbackConstruction("$(prefix)ExactPullback", atlas_a_object, atlas_b_object, base_object,
        atlas_a_to_base, atlas_b_to_base, exact_pullback_output, [:src, :rel, :dst], study.scripts.exact_pullback_sql, "pullback_edges";
        construction_kind="exact", metadata=Dict(:difference_tables => ["A_only_edges", "B_only_edges"]))

    soft_pullback_table = String(get(study.metadata, :soft_pullback_table, "pullback_resv_soft"))
    soft_pullback_output = CSQLObject("$(prefix)SoftPullbackDB", [
        CSQLTableRef(soft_pullback_table, "TABLE $soft_pullback_table",
            [:edge_id_A, :edge_id_B, :rel, :srcA, :dstA, :srcB, :dstB, :sim_dst, :score_sum_joint, :support_lcms_A, :support_lcms_B];
            metadata=Dict(:semantic_role => "soft_pullback_output")),
    ]; metadata=Dict(:construction_kind => "soft_pullback"))

    soft_pullback = CSQLPullbackConstruction("$(prefix)SoftPullback", atlas_a_object, atlas_b_object, base_object,
        atlas_a_to_base, atlas_b_to_base, soft_pullback_output,
        Symbol.(String.(get(study.metadata, :soft_pullback_match_fields, ["rel", "rapidfuzz(dst)"]))),
        study.scripts.soft_pullback_sql, soft_pullback_table;
        construction_kind="soft",
        metadata=Dict(
            :similarity_threshold => Int(get(study.metadata, :soft_pullback_similarity_threshold, 85)),
            :matching_extension => "rapidfuzz_token_set_ratio",
            :focus_terms => copy(String.(get(study.metadata, :focus_terms, String[]))),
        ))

    pushout_output = CSQLObject("$(prefix)PushoutDB", [
        CSQLTableRef("pushout_edges", "TABLE pushout_edges",
            [:edge_key, :rel_type, :src, :dst, :has_A, :has_B, :score_sum_joint, :origin];
            metadata=Dict(:semantic_role => "pushout_output")),
        CSQLTableRef("pushout_edge_support", "TABLE pushout_edge_support",
            [:edge_key, :atlas, :doc_id, :lcm_instance_id, :score];
            metadata=Dict(:semantic_role => "pushout_support")),
    ]; metadata=Dict(:construction_kind => "pushout"))

    pushout = CSQLPushoutConstruction("$(prefix)Pushout", atlas_a_object, atlas_b_object, soft_pullback, pushout_output,
        study.scripts.pushout_sql, String(get(study.metadata, :pushout_table, "pushout_edges"));
        metadata=Dict(:glue_table => String(get(study.metadata, :soft_pullback_alias, soft_pullback.output_table))))

    CategoricalDBBridge(
        study,
        base_object,
        atlas_a_object,
        atlas_b_object,
        atlas_a_to_base,
        atlas_b_to_base,
        exact_pullback,
        soft_pullback,
        pushout,
        Dict(
            :shared_key_fields => [:src, :rel, :dst],
            :semantic_role => "categorical_db_bridge",
            :study_label => String(get(study.metadata, :study_label, study.name)),
        ),
    )
end

build_red_wine_csql_categorical_bridge(root::AbstractString) = build_named_csql_categorical_bridge(root, "red_wine_cardio_resveratrol")
build_tylenol_csql_categorical_bridge(root::AbstractString) = build_named_csql_categorical_bridge(root, "tylenol_lancet_paracetamol")

_sql_quote(text::AbstractString) = replace(String(text), "'" => "''")

function _run_duckdb(; cwd::AbstractString, sql::AbstractString, db_path::Union{Nothing, AbstractString}=nothing, json_output::Bool=false)
    args = String[]
    json_output && push!(args, "-json")
    db_path === nothing || push!(args, String(db_path))
    append!(args, ["-c", sql])
    read(Cmd(Cmd(["duckdb"; args]); dir=String(cwd)), String)
end

function _query_duckdb_json(; cwd::AbstractString, sql::AbstractString, db_path::Union{Nothing, AbstractString}=nothing)
    stdout = strip(_run_duckdb(; cwd, sql, db_path, json_output=true))
    isempty(stdout) && return Any[]
    collect(JSON3.read(stdout))
end

_rowget(row, key::AbstractString) = hasproperty(row, Symbol(key)) ? getproperty(row, Symbol(key)) : row[key]

function _focus_terms_condition(alias::AbstractString, focus_terms::Vector{String})
    isempty(focus_terms) && return "TRUE"
    join(["lower($alias.src_label_canon) LIKE '%$(lowercase(term))%'" for term in focus_terms], " OR ")
end

function _truth_case(study::CSQLAtlasStudy)
    consensus_threshold = Int(get(study.metadata, :consensus_similarity_threshold, 90))
    """
    CASE
      WHEN origin = 'AB' AND COALESCE(sim_dst, 0) >= $consensus_threshold THEN 'CONSENSUS'
      WHEN origin = 'AB' THEN 'WEAK_CONSENSUS'
      WHEN origin = 'A' THEN 'A_ONLY'
      ELSE 'B_ONLY'
    END
    """
end

function _render_exact_pullback_sql(study::CSQLAtlasStudy)
    atlas_a_name = _atlas_dir_name(study.atlas_a)
    atlas_b_name = _atlas_dir_name(study.atlas_b)
    """
    CREATE OR REPLACE VIEW nodes_A AS SELECT * FROM read_parquet('$(_sql_quote("$atlas_a_name/atlas_nodes.parquet"))');
    CREATE OR REPLACE VIEW edges_A AS SELECT * FROM read_parquet('$(_sql_quote("$atlas_a_name/atlas_edges.parquet"))');
    CREATE OR REPLACE VIEW supp_A AS SELECT * FROM read_parquet('$(_sql_quote("$atlas_a_name/atlas_edge_support.parquet"))');

    CREATE OR REPLACE VIEW nodes_B AS SELECT * FROM read_parquet('$(_sql_quote("$atlas_b_name/atlas_nodes.parquet"))');
    CREATE OR REPLACE VIEW edges_B AS SELECT * FROM read_parquet('$(_sql_quote("$atlas_b_name/atlas_edges.parquet"))');
    CREATE OR REPLACE VIEW supp_B AS SELECT * FROM read_parquet('$(_sql_quote("$atlas_b_name/atlas_edge_support.parquet"))');

    CREATE OR REPLACE VIEW key_A AS
    SELECT
      edge_id AS edge_id_A,
      src_label_canon AS src,
      rel_type AS rel,
      dst_label_canon AS dst,
      support_docs AS support_docs_A,
      support_lcms AS support_lcms_A,
      score_sum AS score_sum_A,
      controversy AS controversy_A
    FROM edges_A;

    CREATE OR REPLACE VIEW key_B AS
    SELECT
      edge_id AS edge_id_B,
      src_label_canon AS src,
      rel_type AS rel,
      dst_label_canon AS dst,
      support_docs AS support_docs_B,
      support_lcms AS support_lcms_B,
      score_sum AS score_sum_B,
      controversy AS controversy_B
    FROM edges_B;

    CREATE OR REPLACE TABLE pullback_edges AS
    SELECT
      A.src, A.rel, A.dst,
      A.edge_id_A, B.edge_id_B,
      A.support_docs_A, B.support_docs_B,
      A.support_lcms_A, B.support_lcms_B,
      A.score_sum_A, B.score_sum_B,
      (A.score_sum_A + B.score_sum_B) AS score_sum_joint,
      GREATEST(A.controversy_A, B.controversy_B) AS controversy_joint
    FROM key_A A
    JOIN key_B B USING (src, rel, dst);

    CREATE OR REPLACE TABLE A_only_edges AS
    SELECT A.*
    FROM key_A A
    LEFT JOIN pullback_edges P USING (src, rel, dst)
    WHERE P.src IS NULL;

    CREATE OR REPLACE TABLE B_only_edges AS
    SELECT B.*
    FROM key_B B
    LEFT JOIN pullback_edges P USING (src, rel, dst)
    WHERE P.src IS NULL;

    CREATE OR REPLACE VIEW pullback_evidence AS
    SELECT
      P.src, P.rel, P.dst,
      'A' AS which,
      S.doc_id, S.lcm_instance_id, S.focus, S.radius,
      S.score, S.score_raw, S.coupling
    FROM pullback_edges P
    JOIN supp_A S ON S.edge_id = P.edge_id_A

    UNION ALL

    SELECT
      P.src, P.rel, P.dst,
      'B' AS which,
      S.doc_id, S.lcm_instance_id, S.focus, S.radius,
      S.score, S.score_raw, S.coupling
    FROM pullback_edges P
    JOIN supp_B S ON S.edge_id = P.edge_id_B;
    """
end

function _render_focus_term_soft_pullback_sql(study::CSQLAtlasStudy)
    focus_terms = String.(get(study.metadata, :focus_terms, String[]))
    threshold = Int(get(study.metadata, :soft_pullback_similarity_threshold, 85))
    soft_pullback_table = String(get(study.metadata, :soft_pullback_table, "pullback_resv_soft"))
    """
    INSTALL rapidfuzz FROM community;
    LOAD rapidfuzz;

    CREATE OR REPLACE VIEW edgesA_focus AS
    SELECT * FROM edges_A
    WHERE $(_focus_terms_condition("edges_A", focus_terms));

    CREATE OR REPLACE VIEW edgesB_focus AS
    SELECT * FROM edges_B
    WHERE $(_focus_terms_condition("edges_B", focus_terms));

    CREATE OR REPLACE TABLE $soft_pullback_table AS
    WITH CANDS AS (
      SELECT
        A.edge_id AS edge_id_A,
        B.edge_id AS edge_id_B,
        A.rel_type AS rel,
        A.src_label_canon AS srcA,
        A.dst_label_canon AS dstA,
        B.src_label_canon AS srcB,
        B.dst_label_canon AS dstB,
        rapidfuzz_token_set_ratio(A.dst_label_canon, B.dst_label_canon) AS sim_dst,
        (A.score_sum + B.score_sum) AS score_sum_joint,
        A.support_lcms AS support_lcms_A,
        B.support_lcms AS support_lcms_B
      FROM edgesA_focus A
      JOIN edgesB_focus B
        ON A.rel_type = B.rel_type
      WHERE rapidfuzz_token_set_ratio(A.dst_label_canon, B.dst_label_canon) >= $threshold
    )
    SELECT *
    FROM CANDS
    QUALIFY row_number() OVER (
      PARTITION BY edge_id_A
      ORDER BY sim_dst DESC, score_sum_joint DESC
    ) = 1;
    """
end

function _render_apap_soft_pullback_sql(study::CSQLAtlasStudy)
    outcome_terms = String.(get(study.metadata, :outcome_terms, ["aut", "adhd", "neuro"]))
    threshold = Int(get(study.metadata, :soft_pullback_similarity_threshold, 80))
    soft_pullback_table = String(get(study.metadata, :soft_pullback_table, "pullback_apap_soft"))
    outcome_predicate = join(["lower(dst_label_canon) LIKE '%$(lowercase(term))%'" for term in outcome_terms], " OR ")
    """
    INSTALL rapidfuzz FROM community;
    LOAD rapidfuzz;

    CREATE OR REPLACE VIEW edgesA_norm AS
    SELECT *,
      regexp_replace(lower(src_label_canon), '(paracetamol|tylenol)', 'acetaminophen') AS srcN,
      regexp_replace(lower(dst_label_canon), '(paracetamol|tylenol)', 'acetaminophen') AS dstN
    FROM edges_A;

    CREATE OR REPLACE VIEW edgesB_norm AS
    SELECT *,
      regexp_replace(lower(src_label_canon), '(paracetamol|tylenol)', 'acetaminophen') AS srcN,
      regexp_replace(lower(dst_label_canon), '(paracetamol|tylenol)', 'acetaminophen') AS dstN
    FROM edges_B;

    CREATE OR REPLACE VIEW edgesA_apap AS
    SELECT * FROM edgesA_norm
    WHERE srcN LIKE '%acetaminophen%' OR dstN LIKE '%acetaminophen%';

    CREATE OR REPLACE VIEW edgesB_apap AS
    SELECT * FROM edgesB_norm
    WHERE srcN LIKE '%acetaminophen%' OR dstN LIKE '%acetaminophen%';

    CREATE OR REPLACE VIEW edgesA_apap_outcomes AS
    SELECT * FROM edgesA_apap
    WHERE $outcome_predicate;

    CREATE OR REPLACE VIEW edgesB_apap_outcomes AS
    SELECT * FROM edgesB_apap
    WHERE $outcome_predicate;

    CREATE OR REPLACE TABLE $soft_pullback_table AS
    WITH CANDS AS (
      SELECT
        A.edge_id AS edge_id_A,
        B.edge_id AS edge_id_B,
        A.rel_type AS rel,
        A.src_label_canon AS srcA,
        A.dst_label_canon AS dstA,
        B.src_label_canon AS srcB,
        B.dst_label_canon AS dstB,
        rapidfuzz_token_set_ratio(A.dst_label_canon, B.dst_label_canon) AS sim_dst,
        (A.score_sum + B.score_sum) AS score_sum_joint,
        A.support_lcms AS support_lcms_A,
        B.support_lcms AS support_lcms_B
      FROM edgesA_apap_outcomes A
      JOIN edgesB_apap_outcomes B
        ON A.rel_type = B.rel_type
      WHERE rapidfuzz_token_set_ratio(A.dst_label_canon, B.dst_label_canon) >= $threshold
    )
    SELECT *
    FROM CANDS
    QUALIFY row_number() OVER (
      PARTITION BY edge_id_A
      ORDER BY sim_dst DESC, score_sum_joint DESC
    ) = 1;
    """
end

function _render_soft_pullback_sql(study::CSQLAtlasStudy)
    String(get(study.metadata, :soft_pullback_mode, "focus_source_terms")) == "apap_outcomes" ?
        _render_apap_soft_pullback_sql(study) :
        _render_focus_term_soft_pullback_sql(study)
end

function _render_soft_pullback_alias_sql(study::CSQLAtlasStudy)
    soft_pullback_table = String(get(study.metadata, :soft_pullback_table, "pullback_resv_soft"))
    soft_pullback_alias = String(get(study.metadata, :soft_pullback_alias, soft_pullback_table))
    soft_pullback_alias == soft_pullback_table ? "" : "CREATE OR REPLACE VIEW $soft_pullback_alias AS SELECT * FROM $soft_pullback_table;"
end

function _bootstrap_named_csql_database(study::CSQLAtlasStudy; db_path::AbstractString)
    sql = join((
        _render_exact_pullback_sql(study),
        _render_soft_pullback_sql(study),
        _render_soft_pullback_alias_sql(study),
        read(study.scripts.pushout_sql, String),
    ), "\n\n")
    _run_duckdb(; cwd=study.root, sql, db_path)
end

function materialize_named_csql_results(root::AbstractString, study_name::AbstractString)
    study = locate_named_csql_study(root, study_name)
    soft_pullback_table = String(get(study.metadata, :soft_pullback_table, "pullback_resv_soft"))
    truth_case = _truth_case(study)
    mktempdir(prefix="functorflow_csql_") do temp_dir
        db_path = joinpath(temp_dir, "csql_materialization.duckdb")
        _bootstrap_named_csql_database(study; db_path)

        table_count_rows = _query_duckdb_json(
            cwd=study.root,
            db_path=db_path,
            sql="""
            SELECT 'exact_pullback' AS table_name, COUNT(*) AS edge_count FROM pullback_edges
            UNION ALL
            SELECT 'soft_pullback' AS table_name, COUNT(*) AS edge_count FROM $soft_pullback_table
            UNION ALL
            SELECT 'pushout' AS table_name, COUNT(*) AS edge_count FROM pushout_edges
            ORDER BY table_name
            """,
        )
        truth_rows = _query_duckdb_json(
            cwd=study.root,
            db_path=db_path,
            sql="""
            SELECT
              $truth_case AS truth_value,
              COUNT(*) AS edge_count
            FROM pushout_edges
            GROUP BY 1
            ORDER BY
              CASE truth_value
                WHEN 'CONSENSUS' THEN 1
                WHEN 'WEAK_CONSENSUS' THEN 2
                WHEN 'A_ONLY' THEN 3
                ELSE 4
              END
            """,
        )
        witness_rows = _query_duckdb_json(
            cwd=study.root,
            db_path=db_path,
            sql="""
            SELECT
              $truth_case AS truth_value,
              rel_type AS relation,
              src AS source,
              dst AS target,
              score_sum_joint AS score_joint,
              sim_dst AS similarity,
              support_lcms_A AS support_lcms_a,
              support_lcms_B AS support_lcms_b
            FROM pushout_edges
            WHERE origin = 'AB'
            ORDER BY score_sum_joint DESC, similarity DESC
            """,
        )

        CSQLMaterialization(
            study,
            [(String(_rowget(row, "table_name")), Int(_rowget(row, "edge_count"))) for row in table_count_rows],
            [(String(_rowget(row, "truth_value")), Int(_rowget(row, "edge_count"))) for row in truth_rows],
            [
                CSQLTruthWitness(
                    String(_rowget(row, "truth_value")),
                    String(_rowget(row, "relation")),
                    String(_rowget(row, "source")),
                    String(_rowget(row, "target")),
                    Float64(_rowget(row, "score_joint"));
                    similarity=_rowget(row, "similarity") === nothing ? nothing : Float64(_rowget(row, "similarity")),
                    support_lcms_a=_rowget(row, "support_lcms_a") === nothing ? nothing : Int(_rowget(row, "support_lcms_a")),
                    support_lcms_b=_rowget(row, "support_lcms_b") === nothing ? nothing : Int(_rowget(row, "support_lcms_b")),
                    metadata=Dict(:source_table => "pushout_edges"),
                ) for row in witness_rows
            ],
            Dict(
                :exact_pullback_sql => study.scripts.exact_pullback_sql,
                :soft_pullback_sql => study.scripts.soft_pullback_sql,
                :pushout_sql => study.scripts.pushout_sql,
            ),
        )
    end
end

materialize_red_wine_csql_results(root::AbstractString) = materialize_named_csql_results(root, "red_wine_cardio_resveratrol")
materialize_tylenol_csql_results(root::AbstractString) = materialize_named_csql_results(root, "tylenol_lancet_paracetamol")

function describe_named_csql_materialization(root::AbstractString, study_name::AbstractString; witness_limit::Integer=5)
    bridge = build_named_csql_categorical_bridge(root, study_name)
    materialization = materialize_named_csql_results(root, study_name)
    Dict(
        "study_name" => bridge.study.name,
        "base_object" => String(bridge.base_object.name),
        "atlas_a_object" => String(bridge.atlas_a_object.name),
        "atlas_b_object" => String(bridge.atlas_b_object.name),
        "exact_pullback_table" => String(bridge.exact_pullback.output_table),
        "soft_pullback_table" => String(bridge.soft_pullback.output_table),
        "pushout_table" => String(bridge.pushout.output_table),
        "table_counts" => Dict(name => count for (name, count) in materialization.table_counts),
        "truth_value_counts" => Dict(name => count for (name, count) in materialization.truth_value_counts),
        "witnesses" => [
            Dict(
                "truth_value" => witness.truth_value,
                "relation" => witness.relation,
                "source" => witness.source,
                "target" => witness.target,
                "score_joint" => witness.score_joint,
                "similarity" => witness.similarity,
                "support_lcms_a" => witness.support_lcms_a,
                "support_lcms_b" => witness.support_lcms_b,
            ) for witness in materialization.witnesses[1:min(length(materialization.witnesses), Int(witness_limit))]
        ],
    )
end

describe_red_wine_csql_materialization(root::AbstractString; witness_limit::Integer=5) =
    describe_named_csql_materialization(root, "red_wine_cardio_resveratrol"; witness_limit)
describe_tylenol_csql_materialization(root::AbstractString; witness_limit::Integer=5) =
    describe_named_csql_materialization(root, "tylenol_lancet_paracetamol"; witness_limit)

function locate_tcc_atlas(root::AbstractString, name::AbstractString="atlas_TCC")
    spec = _tcc_atlas_spec_named(name)
    atlas_root = _require_existing_path(joinpath(abspath(root), "democritus_atlas", spec.atlas_dir))
    AtlasFileSet(
        atlas_root,
        _require_existing_path(joinpath(atlas_root, "nodes.parquet")),
        _require_existing_path(joinpath(atlas_root, "edges.parquet")),
        _require_existing_path(joinpath(atlas_root, "edge_support.parquet"));
        summary_markdown=isfile(joinpath(atlas_root, "summary.md")) ? joinpath(atlas_root, "summary.md") : nothing,
    )
end

function materialize_tcc_atlas_profile(root::AbstractString, name::AbstractString="atlas_TCC")
    spec = _tcc_atlas_spec_named(name)
    atlas = locate_tcc_atlas(root, name)
    rows = _query_duckdb_json(
        cwd=atlas.root,
        sql="""
        SELECT 'nodes' AS metric, COUNT(*)::BIGINT AS value FROM read_parquet('$(_sql_quote(atlas.nodes_parquet))')
        UNION ALL
        SELECT 'edges' AS metric, COUNT(*)::BIGINT AS value FROM read_parquet('$(_sql_quote(atlas.edges_parquet))')
        UNION ALL
        SELECT 'edge_support' AS metric, COUNT(*)::BIGINT AS value FROM read_parquet('$(_sql_quote(atlas.edge_support_parquet))')
        UNION ALL
        SELECT 'avg_support_docs' AS metric, AVG(support_docs) AS value FROM read_parquet('$(_sql_quote(atlas.edges_parquet))')
        UNION ALL
        SELECT 'max_support_docs' AS metric, MAX(support_docs)::BIGINT AS value FROM read_parquet('$(_sql_quote(atlas.edges_parquet))')
        """,
    )
    metric_map = Dict(String(_rowget(row, "metric")) => _rowget(row, "value") for row in rows)

    relation_rows = _query_duckdb_json(
        cwd=atlas.root,
        sql="""
        SELECT rel_type, COUNT(*)::BIGINT AS edge_count
        FROM read_parquet('$(_sql_quote(atlas.edges_parquet))')
        GROUP BY 1
        ORDER BY edge_count DESC, rel_type
        """,
    )
    yearly_rows = _query_duckdb_json(
        cwd=atlas.root,
        sql="""
        SELECT year, COUNT(*)::BIGINT AS support_count
        FROM read_parquet('$(_sql_quote(atlas.edge_support_parquet))')
        WHERE year IS NOT NULL
        GROUP BY 1
        ORDER BY support_count DESC, year DESC
        LIMIT 12
        """,
    )
    edge_rows = _query_duckdb_json(
        cwd=atlas.root,
        sql="""
        SELECT
          src_label_canon AS source,
          rel_type AS relation,
          dst_label_canon AS target,
          support_docs,
          score_sum
        FROM read_parquet('$(_sql_quote(atlas.edges_parquet))')
        WHERE lower(coalesce(src_label_canon, '')) NOT IN ('', 'none', 'nan', 'null')
          AND lower(coalesce(dst_label_canon, '')) NOT IN ('', 'none', 'nan', 'null')
        ORDER BY support_docs DESC, score_sum DESC, source, target
        LIMIT 12
        """,
    )

    object = CSQLObject("$(spec.bridge_prefix)CorpusDBObject", [
        CSQLTableRef("nodes", atlas.nodes_parquet, [:node_id, :label_canon];
            metadata=Dict(:semantic_role => "atlas_nodes", :atlas_name => spec.name)),
        CSQLTableRef("edges", atlas.edges_parquet,
            [:edge_id, :src_id, :dst_id, :src_label_canon, :dst_label_canon, :rel_type, :support_docs, :score_sum];
            metadata=Dict(:semantic_role => "atlas_edges", :atlas_name => spec.name)),
        CSQLTableRef("edge_support", atlas.edge_support_parquet,
            [:edge_id, :src_label_canon, :dst_label_canon, :rel_type, :doc_id, :method, :year, :journal, :p_value];
            metadata=Dict(:semantic_role => "atlas_edge_support", :atlas_name => spec.name)),
    ]; metadata=Dict(:atlas_name => spec.name, :corpus_scale => get(spec.metadata, :corpus_scale, nothing)))

    TCCAtlasProfile(
        spec,
        atlas,
        object,
        Int(metric_map["nodes"]),
        Int(metric_map["edges"]),
        Int(metric_map["edge_support"]),
        Float64(metric_map["avg_support_docs"]),
        Int(metric_map["max_support_docs"]),
        [(String(_rowget(row, "rel_type")), Int(_rowget(row, "edge_count"))) for row in relation_rows],
        [(Int(_rowget(row, "year")), Int(_rowget(row, "support_count"))) for row in yearly_rows],
        [
            TCCEdgeWitness(
                String(_rowget(row, "source")),
                String(_rowget(row, "relation")),
                String(_rowget(row, "target")),
                Int(_rowget(row, "support_docs")),
                Float64(_rowget(row, "score_sum")),
            ) for row in edge_rows
        ],
        Dict(:summary_markdown => atlas.summary_markdown, :corpus_scale => get(spec.metadata, :corpus_scale, nothing)),
    )
end

function describe_tcc_atlas_profile(root::AbstractString, name::AbstractString="atlas_TCC")
    profile = materialize_tcc_atlas_profile(root, name)
    Dict(
        "name" => profile.spec.name,
        "study_label" => profile.spec.study_label,
        "node_count" => profile.node_count,
        "edge_count" => profile.edge_count,
        "edge_support_count" => profile.edge_support_count,
        "average_support_docs" => profile.average_support_docs,
        "max_support_docs" => profile.max_support_docs,
        "relation_counts" => Dict(name => count for (name, count) in profile.relation_counts),
        "yearly_support_counts" => [Dict("year" => year, "support_count" => count) for (year, count) in profile.yearly_support_counts],
        "top_edges" => [
            Dict(
                "source" => edge.source,
                "relation" => edge.relation,
                "target" => edge.target,
                "support_docs" => edge.support_docs,
                "score_sum" => edge.score_sum,
            ) for edge in profile.top_edges
        ],
    )
end

function _bootstrap_tcc_method_database(data_root::AbstractString; db_path::AbstractString)
    claims_path = joinpath(data_root, "causal_claims_beta.parquet")
    sql = """
    CREATE OR REPLACE VIEW tcc AS
    SELECT * FROM read_parquet('$(_sql_quote(claims_path))');

    CREATE OR REPLACE VIEW tcc_norm AS
    SELECT *,
      lower(trim(cause)) AS causeN,
      lower(trim(effect)) AS effectN
    FROM tcc;

    CREATE OR REPLACE VIEW tcc_mass AS
    SELECT *,
      CASE
        WHEN level_of_tentativeness = 'certain' THEN 1.0
        WHEN level_of_tentativeness = 'tentative' THEN 0.6
        ELSE 0.8
      END AS w_tent,
      CASE statistical_significance
        WHEN 'p<0.01' THEN 1.0
        WHEN '0.01<=p<0.05' THEN 0.9
        WHEN '0.05<=p<0.1' THEN 0.7
        WHEN 'p>0.1' THEN 0.4
        ELSE 0.5
      END AS w_sig,
      CASE
        WHEN is_evidence_provided_in_paper THEN 1.0
        ELSE 0.7
      END AS w_ev
    FROM tcc_norm;

    CREATE OR REPLACE TABLE csql_edge_support AS
    SELECT
      paper_edge_id,
      paper_id,
      paper_repo,
      year,
      causeN AS src,
      effectN AS dst,
      sign_of_impact AS sign,
      causal_inference_method,
      statistical_significance,
      level_of_tentativeness,
      is_evidence_provided_in_paper,
      (w_tent * w_sig * w_ev) AS mass
    FROM tcc_mass;

    CREATE OR REPLACE TABLE csql_edges AS
    SELECT
      src,
      dst,
      sign,
      COUNT(*) AS support_rows,
      COUNT(DISTINCT paper_id) AS support_docs,
      SUM(mass) AS score_sum,
      AVG(mass) AS score_mean,
      MAX(mass) AS score_max
    FROM csql_edge_support
    GROUP BY src, dst, sign;

    CREATE OR REPLACE TABLE csql_nodes AS
    SELECT src AS label_canon FROM csql_edge_support
    UNION
    SELECT dst AS label_canon FROM csql_edge_support;

    CREATE OR REPLACE VIEW support AS
    SELECT * FROM csql_edge_support;

    CREATE OR REPLACE VIEW support_method AS
    SELECT *,
      CASE
        WHEN causal_inference_method IN ('DID', 'TWFE', 'Event Study') THEN 'DID_FAMILY'
        WHEN causal_inference_method IN ('IV', '2SLS') THEN 'IV_FAMILY'
        WHEN causal_inference_method = 'RDD' THEN 'RDD'
        WHEN causal_inference_method = 'RCT' THEN 'RCT'
        WHEN causal_inference_method IN ('Theoretical/Non-Empirical', 'Simulations') THEN 'NONEMPIRICAL'
        WHEN causal_inference_method IN ('Do not know', 'NULL') OR causal_inference_method IS NULL THEN 'UNKNOWN'
        ELSE 'OTHER'
      END AS method_class
    FROM support;

    CREATE OR REPLACE VIEW omega_did_iv AS
    WITH did AS (
      SELECT
        src,
        dst,
        SUM(CASE WHEN sign = 'increase' THEN mass ELSE 0 END) AS did_inc,
        SUM(CASE WHEN sign = 'decrease' THEN mass ELSE 0 END) AS did_dec,
        SUM(CASE WHEN sign = 'null result' THEN mass ELSE 0 END) AS did_null,
        COUNT(DISTINCT paper_id) AS docs_did
      FROM support_method
      WHERE method_class = 'DID_FAMILY'
      GROUP BY src, dst
    ),
    iv AS (
      SELECT
        src,
        dst,
        SUM(CASE WHEN sign = 'increase' THEN mass ELSE 0 END) AS iv_inc,
        SUM(CASE WHEN sign = 'decrease' THEN mass ELSE 0 END) AS iv_dec,
        SUM(CASE WHEN sign = 'null result' THEN mass ELSE 0 END) AS iv_null,
        COUNT(DISTINCT paper_id) AS docs_iv
      FROM support_method
      WHERE method_class = 'IV_FAMILY'
      GROUP BY src, dst
    )
    SELECT
      did.src,
      did.dst,
      did.docs_did,
      iv.docs_iv,
      did.did_inc,
      did.did_dec,
      did.did_null,
      iv.iv_inc,
      iv.iv_dec,
      iv.iv_null,
      CASE
        WHEN (did.did_inc >= 2 * (did.did_dec + did.did_null))
          AND (iv.iv_inc >= 2 * (iv.iv_dec + iv.iv_null)) THEN 'CONSENSUS_INC'
        WHEN (did.did_dec >= 2 * (did.did_inc + did.did_null))
          AND (iv.iv_dec >= 2 * (iv.iv_inc + iv.iv_null)) THEN 'CONSENSUS_DEC'
        WHEN (did.did_null >= 2 * (did.did_inc + did.did_dec))
          AND (iv.iv_null >= 2 * (iv.iv_inc + iv.iv_dec)) THEN 'CONSENSUS_NULL'
        ELSE 'METHOD_CONFLICT'
      END AS omega
    FROM did
    JOIN iv USING (src, dst);
    """
    _run_duckdb(; cwd=data_root, sql, db_path)
end

_deck_sign_label(sign::AbstractString) = get(Dict("increase" => "inc", "decrease" => "dec", "null result" => "null"), String(sign), String(sign))

function materialize_tcc_method_pullback(root::AbstractString; top_k::Integer=8, workspace_root::Union{Nothing, AbstractString}=nothing)
    resolved_root = abspath(root)
    data_root = _require_existing_path(joinpath(resolved_root, "democritus_atlas", "causal_claims"))
    resolved_workspace_root = workspace_root === nothing ? abspath(joinpath(@__DIR__, "..")) : abspath(workspace_root)
    mktempdir(prefix="functorflow_tcc_methods_") do temp_dir
        db_path = joinpath(temp_dir, "tcc_methods.duckdb")
        _bootstrap_tcc_method_database(data_root; db_path)

        counts = _query_duckdb_json(
            cwd=data_root,
            db_path=db_path,
            sql="""
            SELECT 'csql_nodes' AS metric, COUNT(*)::BIGINT AS value FROM csql_nodes
            UNION ALL
            SELECT 'csql_edges' AS metric, COUNT(*)::BIGINT AS value FROM csql_edges
            UNION ALL
            SELECT 'csql_edge_support' AS metric, COUNT(*)::BIGINT AS value FROM csql_edge_support
            ORDER BY metric
            """,
        )
        pullback_rows = _query_duckdb_json(
            cwd=data_root,
            db_path=db_path,
            sql="""
            WITH A AS (
              SELECT src, dst, sign, paper_id, mass
              FROM support_method
              WHERE method_class = 'DID_FAMILY'
            ),
            B AS (
              SELECT src, dst, sign, paper_id, mass
              FROM support_method
              WHERE method_class = 'IV_FAMILY'
            )
            SELECT
              A.src,
              A.sign,
              A.dst,
              COUNT(DISTINCT A.paper_id) AS docs_did,
              COUNT(DISTINCT B.paper_id) AS docs_iv,
              ROUND(SUM(A.mass), 3) AS mass_did,
              ROUND(SUM(B.mass), 3) AS mass_iv
            FROM A
            JOIN B USING (src, sign, dst)
            GROUP BY A.src, A.sign, A.dst
            ORDER BY (mass_did + mass_iv) DESC, A.src, A.dst
            LIMIT $(Int(top_k))
            """,
        )
        omega_counts = _query_duckdb_json(
            cwd=data_root,
            db_path=db_path,
            sql="""
            SELECT omega, COUNT(*)::BIGINT AS pair_count
            FROM omega_did_iv
            GROUP BY omega
            ORDER BY pair_count DESC, omega
            """,
        )
        conflict_rows = _query_duckdb_json(
            cwd=data_root,
            db_path=db_path,
            sql="""
            WITH atlas_method_conflict AS (
              SELECT * FROM omega_did_iv WHERE omega = 'METHOD_CONFLICT'
            )
            SELECT
              c.src,
              c.dst,
              s.method_class,
              s.sign,
              COUNT(DISTINCT s.paper_id) AS n_papers,
              MIN(s.year) AS min_year,
              MAX(s.year) AS max_year,
              ROUND(SUM(s.mass), 3) AS mass_sum
            FROM atlas_method_conflict c
            JOIN support_method s
              ON s.src = c.src AND s.dst = c.dst
            WHERE s.method_class IN ('DID_FAMILY', 'IV_FAMILY')
            GROUP BY c.src, c.dst, s.method_class, s.sign
            ORDER BY c.src, c.dst, s.method_class, s.sign
            LIMIT $(Int(top_k))
            """,
        )

        TCCMethodPullbackSummary(
            resolved_workspace_root,
            data_root,
            [(String(_rowget(row, "metric")), Int(_rowget(row, "value"))) for row in counts],
            [
                TCCMethodPullbackWitness(
                    String(_rowget(row, "src")),
                    _deck_sign_label(String(_rowget(row, "sign"))),
                    String(_rowget(row, "dst")),
                    Int(_rowget(row, "docs_did")),
                    Int(_rowget(row, "docs_iv")),
                    Float64(_rowget(row, "mass_did")),
                    Float64(_rowget(row, "mass_iv")),
                ) for row in pullback_rows
            ],
            [(String(_rowget(row, "omega")), Int(_rowget(row, "pair_count"))) for row in omega_counts],
            [
                TCCMethodConflictWitness(
                    String(_rowget(row, "src")),
                    String(_rowget(row, "dst")),
                    String(_rowget(row, "method_class")),
                    String(_rowget(row, "sign")),
                    Int(_rowget(row, "n_papers")),
                    _rowget(row, "min_year") === nothing ? nothing : Int(_rowget(row, "min_year")),
                    _rowget(row, "max_year") === nothing ? nothing : Int(_rowget(row, "max_year")),
                    Float64(_rowget(row, "mass_sum")),
                ) for row in conflict_rows
            ],
            Dict(:top_k => Int(top_k)),
        )
    end
end

function describe_tcc_method_pullback(root::AbstractString; top_k::Integer=8, workspace_root::Union{Nothing, AbstractString}=nothing)
    summary = materialize_tcc_method_pullback(root; top_k, workspace_root)
    Dict(
        "workspace_root" => summary.workspace_root,
        "data_root" => summary.data_root,
        "compiled_counts" => Dict(name => count for (name, count) in summary.compiled_counts),
        "did_iv_pullback" => [
            Dict(
                "source" => row.source,
                "sign" => row.sign,
                "target" => row.target,
                "docs_did" => row.docs_did,
                "docs_iv" => row.docs_iv,
                "mass_did" => row.mass_did,
                "mass_iv" => row.mass_iv,
            ) for row in summary.did_iv_pullback
        ],
        "omega_counts" => Dict(name => count for (name, count) in summary.omega_counts),
        "method_conflicts" => [
            Dict(
                "source" => row.source,
                "target" => row.target,
                "method_class" => row.method_class,
                "sign" => row.sign,
                "n_papers" => row.n_papers,
                "min_year" => row.min_year,
                "max_year" => row.max_year,
                "mass_sum" => row.mass_sum,
            ) for row in summary.method_conflicts
        ],
    )
end
