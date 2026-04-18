# ============================================================================
# data_bridges/types.jl — Struct definitions for atlas / CSQL / TCC bridges and TCC truth-witness records.
# ============================================================================
#
# Sourced from the original src/data_bridges.jl as part of the v0.3.4 split.
# All public symbols are unchanged; this file only contains existing code.
# ============================================================================

# ============================================================================
# data_bridges.jl — Categorical DB / atlas / TCC bridge semantics
# ============================================================================

struct AtlasFileSet
    root::String
    nodes_parquet::String
    edges_parquet::String
    edge_support_parquet::String
    scc_parquet::Union{Nothing, String}
    summary_markdown::Union{Nothing, String}
end

AtlasFileSet(root, nodes_parquet, edges_parquet, edge_support_parquet; scc_parquet=nothing, summary_markdown=nothing) =
    AtlasFileSet(String(root), String(nodes_parquet), String(edges_parquet), String(edge_support_parquet),
                 scc_parquet === nothing ? nothing : String(scc_parquet),
                 summary_markdown === nothing ? nothing : String(summary_markdown))

Base.getproperty(atlas::AtlasFileSet, sym::Symbol) =
    sym === :name ? Symbol(splitdir(getfield(atlas, :root))[2]) : getfield(atlas, sym)

struct AtlasSummary
    nodes::Union{Nothing, Int}
    edges::Union{Nothing, Int}
    edge_support_rows::Union{Nothing, Int}
    scc_modules::Union{Nothing, Int}
    top_hub::Union{Nothing, String}
    metadata::Dict{Symbol, Any}
end

function AtlasSummary(; nodes=nothing, edges=nothing, edge_support_rows=nothing, scc_modules=nothing, top_hub=nothing, metadata::Dict=Dict{Symbol, Any}())
    AtlasSummary(nodes, edges, edge_support_rows, scc_modules, top_hub === nothing ? nothing : String(top_hub),
                 Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

struct SQLScriptSet
    exact_pullback_sql::String
    soft_pullback_sql::String
    pushout_sql::String
end

SQLScriptSet(exact_pullback_sql::AbstractString, soft_pullback_sql::AbstractString, pushout_sql::AbstractString) =
    SQLScriptSet(String(exact_pullback_sql), String(soft_pullback_sql), String(pushout_sql))

struct CSQLAtlasStudy
    name::String
    root::String
    atlas_a::AtlasFileSet
    atlas_b::AtlasFileSet
    scripts::SQLScriptSet
    summary_a::AtlasSummary
    summary_b::AtlasSummary
    metadata::Dict{Symbol, Any}
end

function CSQLAtlasStudy(name, root, atlas_a::AtlasFileSet, atlas_b::AtlasFileSet, scripts::SQLScriptSet,
                        summary_a::AtlasSummary, summary_b::AtlasSummary; metadata::Dict=Dict{Symbol, Any}())
    CSQLAtlasStudy(String(name), String(root), atlas_a, atlas_b, scripts, summary_a, summary_b,
                   Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

struct CSQLTableRef
    name::Symbol
    source::String
    columns::Vector{Symbol}
    metadata::Dict{Symbol, Any}
end

function CSQLTableRef(name, source, columns::Vector{Symbol}; metadata::Dict=Dict{Symbol, Any}())
    CSQLTableRef(Symbol(name), String(source), copy(columns), Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

struct CSQLObject
    name::Symbol
    tables::Vector{CSQLTableRef}
    metadata::Dict{Symbol, Any}
end

function CSQLObject(name, tables::Vector{CSQLTableRef}; metadata::Dict=Dict{Symbol, Any}())
    CSQLObject(Symbol(name), copy(tables), Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

struct CSQLMorphism
    name::Symbol
    source::CSQLObject
    target::CSQLObject
    key_fields::Vector{Symbol}
    relation_maps::Vector{Tuple{Symbol, Symbol}}
    sql_reference::Union{Nothing, String}
    metadata::Dict{Symbol, Any}
end

function CSQLMorphism(name, source::CSQLObject, target::CSQLObject, key_fields::Vector{Symbol}, relation_maps::Vector{Tuple{Symbol, Symbol}};
                      sql_reference=nothing, metadata::Dict=Dict{Symbol, Any}())
    CSQLMorphism(Symbol(name), source, target, copy(key_fields), copy(relation_maps),
                 sql_reference === nothing ? nothing : String(sql_reference),
                 Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

struct CSQLPullbackConstruction
    name::Symbol
    left::CSQLObject
    right::CSQLObject
    base::CSQLObject
    left_to_base::CSQLMorphism
    right_to_base::CSQLMorphism
    output::CSQLObject
    match_fields::Vector{Symbol}
    sql_script::String
    output_table::Symbol
    construction_kind::String
    metadata::Dict{Symbol, Any}
end

function CSQLPullbackConstruction(name, left::CSQLObject, right::CSQLObject, base::CSQLObject,
                                  left_to_base::CSQLMorphism, right_to_base::CSQLMorphism, output::CSQLObject,
                                  match_fields::Vector{Symbol}, sql_script, output_table;
                                  construction_kind="exact", metadata::Dict=Dict{Symbol, Any}())
    CSQLPullbackConstruction(Symbol(name), left, right, base, left_to_base, right_to_base, output,
                             copy(match_fields), String(sql_script), Symbol(output_table), String(construction_kind),
                             Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

struct CSQLPushoutConstruction
    name::Symbol
    left::CSQLObject
    right::CSQLObject
    glue::CSQLPullbackConstruction
    output::CSQLObject
    sql_script::String
    output_table::Symbol
    metadata::Dict{Symbol, Any}
end

function CSQLPushoutConstruction(name, left::CSQLObject, right::CSQLObject, glue::CSQLPullbackConstruction, output::CSQLObject,
                                 sql_script, output_table; metadata::Dict=Dict{Symbol, Any}())
    CSQLPushoutConstruction(Symbol(name), left, right, glue, output, String(sql_script), Symbol(output_table),
                            Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

struct CategoricalDBBridge
    study::CSQLAtlasStudy
    base_object::CSQLObject
    atlas_a_object::CSQLObject
    atlas_b_object::CSQLObject
    atlas_a_to_base::CSQLMorphism
    atlas_b_to_base::CSQLMorphism
    exact_pullback::CSQLPullbackConstruction
    soft_pullback::CSQLPullbackConstruction
    pushout::CSQLPushoutConstruction
    metadata::Dict{Symbol, Any}
end

struct CSQLTruthWitness
    truth_value::String
    relation::String
    source::String
    target::String
    score_joint::Float64
    similarity::Union{Nothing, Float64}
    support_lcms_a::Union{Nothing, Int}
    support_lcms_b::Union{Nothing, Int}
    metadata::Dict{Symbol, Any}
end

function CSQLTruthWitness(truth_value, relation, source, target, score_joint;
                          similarity=nothing, support_lcms_a=nothing, support_lcms_b=nothing, metadata::Dict=Dict{Symbol, Any}())
    CSQLTruthWitness(String(truth_value), String(relation), String(source), String(target), Float64(score_joint),
                     similarity === nothing ? nothing : Float64(similarity),
                     support_lcms_a === nothing ? nothing : Int(support_lcms_a),
                     support_lcms_b === nothing ? nothing : Int(support_lcms_b),
                     Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

struct CSQLMaterialization
    study::CSQLAtlasStudy
    table_counts::Vector{Tuple{String, Int}}
    truth_value_counts::Vector{Tuple{String, Int}}
    witnesses::Vector{CSQLTruthWitness}
    metadata::Dict{Symbol, Any}
end

struct IntuitionisticDBBridge
    study::CSQLAtlasStudy
    categorical_db_bridge::CategoricalDBBridge
    materialization::CSQLMaterialization
    bridge_scm::SCMModelObject
    omega::OmegaSCM
    consensus_predicate::SCMPredicate
    weak_consensus_predicate::SCMPredicate
    a_only_predicate::SCMPredicate
    b_only_predicate::SCMPredicate
    consensus_classifier::SCMCharacteristicMap
    weak_consensus_classifier::SCMCharacteristicMap
    a_only_classifier::SCMCharacteristicMap
    b_only_classifier::SCMCharacteristicMap
    metadata::Dict{Symbol, Any}
end

struct TCCAtlasSpec
    name::String
    atlas_dir::String
    study_label::String
    bridge_prefix::String
    metadata::Dict{Symbol, Any}
end

function TCCAtlasSpec(name, atlas_dir, study_label, bridge_prefix; metadata::Dict=Dict{Symbol, Any}())
    TCCAtlasSpec(String(name), String(atlas_dir), String(study_label), String(bridge_prefix),
                 Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

struct TCCEdgeWitness
    source::String
    relation::String
    target::String
    support_docs::Int
    score_sum::Float64
    metadata::Dict{Symbol, Any}
end

function TCCEdgeWitness(source, relation, target, support_docs, score_sum; metadata::Dict=Dict{Symbol, Any}())
    TCCEdgeWitness(String(source), String(relation), String(target), Int(support_docs), Float64(score_sum),
                   Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

struct TCCAtlasProfile
    spec::TCCAtlasSpec
    atlas::AtlasFileSet
    csql_object::CSQLObject
    node_count::Int
    edge_count::Int
    edge_support_count::Int
    average_support_docs::Float64
    max_support_docs::Int
    relation_counts::Vector{Tuple{String, Int}}
    yearly_support_counts::Vector{Tuple{Int, Int}}
    top_edges::Vector{TCCEdgeWitness}
    metadata::Dict{Symbol, Any}
end

struct TCCMethodPullbackWitness
    source::String
    sign::String
    target::String
    docs_did::Int
    docs_iv::Int
    mass_did::Float64
    mass_iv::Float64
end

function TCCMethodPullbackWitness(source::AbstractString, sign::AbstractString, target::AbstractString,
                                  docs_did::Integer, docs_iv::Integer, mass_did::Real, mass_iv::Real)
    TCCMethodPullbackWitness(String(source), String(sign), String(target), Int(docs_did), Int(docs_iv),
                             Float64(mass_did), Float64(mass_iv))
end

struct TCCMethodConflictWitness
    source::String
    target::String
    method_class::String
    sign::String
    n_papers::Int
    min_year::Union{Nothing, Int}
    max_year::Union{Nothing, Int}
    mass_sum::Float64
end

function TCCMethodConflictWitness(source::AbstractString, target::AbstractString, method_class::AbstractString,
                                  sign::AbstractString, n_papers::Integer,
                                  min_year::Union{Nothing, Integer}, max_year::Union{Nothing, Integer}, mass_sum::Real)
    TCCMethodConflictWitness(String(source), String(target), String(method_class), String(sign), Int(n_papers),
                             min_year === nothing ? nothing : Int(min_year),
                             max_year === nothing ? nothing : Int(max_year),
                             Float64(mass_sum))
end

struct TCCMethodPullbackSummary
    workspace_root::String
    data_root::String
    compiled_counts::Vector{Tuple{String, Int}}
    did_iv_pullback::Vector{TCCMethodPullbackWitness}
    omega_counts::Vector{Tuple{String, Int}}
    method_conflicts::Vector{TCCMethodConflictWitness}
    metadata::Dict{Symbol, Any}
end
