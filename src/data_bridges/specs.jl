# ============================================================================
# data_bridges/specs.jl — AtlasPairStudySpec, atlas/TCC spec catalogues, sample-data path helpers (parse_atlas_summary, _build_pair_atlas_fileset).
# ============================================================================
#
# Sourced from the original src/data_bridges.jl as part of the v0.3.4 split.
# All public symbols are unchanged; this file only contains existing code.
# ============================================================================


# ============================================================================
# Concrete sample-data-backed bridge loaders/materializers
# ============================================================================

struct AtlasPairStudySpec
    name::String
    atlas_a_dir::String
    atlas_b_dir::String
    atlas_a_role::String
    atlas_b_role::String
    exact_pullback_sql::String
    soft_pullback_sql::String
    pushout_sql::String
    focus_terms::Vector{String}
    bridge_prefix::String
    study_label::String
    metadata::Dict{Symbol, Any}
end

function AtlasPairStudySpec(name, atlas_a_dir, atlas_b_dir, atlas_a_role, atlas_b_role,
                            exact_pullback_sql, soft_pullback_sql, pushout_sql, focus_terms,
                            bridge_prefix, study_label; metadata::Dict=Dict{Symbol, Any}())
    AtlasPairStudySpec(
        String(name),
        String(atlas_a_dir),
        String(atlas_b_dir),
        String(atlas_a_role),
        String(atlas_b_role),
        String(exact_pullback_sql),
        String(soft_pullback_sql),
        String(pushout_sql),
        String.(collect(focus_terms)),
        String(bridge_prefix),
        String(study_label),
        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata),
    )
end

function atlas_pair_study_specs()
    [
        AtlasPairStudySpec(
            "red_wine_cardio_resveratrol",
            "atlas_cardio",
            "atlas_resveratrol",
            "cardio",
            "resveratrol",
            "pullback_reconcile.sql",
            "soft_atlas_pullback.sql",
            "pushout_merge.sql",
            ("resveratrol",),
            "RedWine",
            "red_wine";
            metadata=Dict(
                :exact_pullback_table => "pullback_edges",
                :soft_pullback_table => "pullback_resv_soft",
                :soft_pullback_alias => "pullback_resv_soft",
                :soft_pullback_mode => "focus_source_terms",
                :soft_pullback_similarity_threshold => 85,
                :consensus_similarity_threshold => 90,
                :soft_pullback_match_fields => ["rel", "resveratrol", "rapidfuzz(dst)"],
                :pushout_table => "pushout_edges",
            ),
        ),
        AtlasPairStudySpec(
            "tylenol_lancet_paracetamol",
            "atlas_NYT_Tylenol",
            "atlas_Lancet_Paracetomol",
            "nyt_tylenol",
            "lancet_paracetamol",
            "pullback_reconcile_tylenol.sql",
            "soft_atlas_pullback_tylenol.sql",
            "pushout_merge.sql",
            ("acetaminophen", "paracetamol", "tylenol"),
            "TylenolLancet",
            "tylenol";
            metadata=Dict(
                :exact_pullback_table => "pullback_edges",
                :soft_pullback_table => "pullback_apap_soft",
                :soft_pullback_alias => "pullback_resv_soft",
                :soft_pullback_mode => "apap_outcomes",
                :soft_pullback_similarity_threshold => 80,
                :consensus_similarity_threshold => 90,
                :soft_pullback_match_fields => ["rel", "acetaminophen_synonym_normalized", "rapidfuzz(neurodevelopment_dst)"],
                :outcome_terms => ["aut", "adhd", "neuro"],
                :pushout_table => "pushout_edges",
            ),
        ),
    ]
end

function tcc_atlas_specs()
    [
        TCCAtlasSpec("atlas_TCC", "atlas_TCC", "tcc", "TCC"; metadata=Dict(:corpus_scale => "~45k papers")),
        TCCAtlasSpec("atlas_TCC_v2", "atlas_TCC_v2", "tcc_v2", "TCCv2"; metadata=Dict(:corpus_scale => "~45k papers")),
    ]
end

_atlas_dir_name(atlas::AtlasFileSet) = splitdir(atlas.root)[2]

function _require_existing_path(path::AbstractString)
    ispath(path) || throw(ArgumentError("Required path does not exist: $(abspath(path))"))
    abspath(path)
end

function _extract_int(text::AbstractString, pattern::AbstractString)
    match_obj = match(Regex(pattern, "m"), text)
    match_obj === nothing && return nothing
    parse(Int, replace(only(match_obj.captures), "," => ""))
end

function _extract_top_hub(text::AbstractString)
    match_obj = match(r"\|\s*1\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*`([^`]+)`\s*\|", text)
    match_obj === nothing ? nothing : only(match_obj.captures)
end

function parse_atlas_summary(summary_path::Union{Nothing, AbstractString})
    summary_path === nothing && return AtlasSummary()
    isfile(summary_path) || return AtlasSummary()
    text = read(summary_path, String)
    AtlasSummary(
        nodes=_extract_int(text, raw"- Nodes:\s*([0-9,]+)"),
        edges=_extract_int(text, raw"- Edges \(unique\):\s*([0-9,]+)"),
        edge_support_rows=_extract_int(text, raw"- Edge-support rows:\s*([0-9,]+)"),
        scc_modules=_extract_int(text, raw"- SCC modules \(size>1\):\s*([0-9,]+)"),
        top_hub=_extract_top_hub(text),
    )
end

function _atlas_pair_study_spec_named(name::AbstractString)
    for spec in atlas_pair_study_specs()
        spec.name == name && return spec
    end
    known = join(getfield.(atlas_pair_study_specs(), :name), ", ")
    throw(KeyError("Unknown cSQL atlas study $(repr(name)). Known studies: $known."))
end

function _tcc_atlas_spec_named(name::AbstractString)
    for spec in tcc_atlas_specs()
        spec.name == name && return spec
    end
    known = join(getfield.(tcc_atlas_specs(), :name), ", ")
    throw(KeyError("Unknown TCC atlas $(repr(name)). Known atlases: $known."))
end

function _build_pair_atlas_fileset(root::AbstractString)
    AtlasFileSet(
        _require_existing_path(root),
        _require_existing_path(joinpath(root, "atlas_nodes.parquet")),
        _require_existing_path(joinpath(root, "atlas_edges.parquet")),
        _require_existing_path(joinpath(root, "atlas_edge_support.parquet"));
        scc_parquet=isfile(joinpath(root, "atlas_scc.parquet")) ? joinpath(root, "atlas_scc.parquet") : nothing,
        summary_markdown=isfile(joinpath(root, "atlas_summary.md")) ? joinpath(root, "atlas_summary.md") : nothing,
    )
end

