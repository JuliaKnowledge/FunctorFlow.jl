# ============================================================================
# cliff_router.jl — CLIFF-style routed query orchestration
# ============================================================================

const _CLIFF_SEC_MARKERS = [
    (r"\b10-k\b", "10-k"),
    (r"\b10-q\b", "10-q"),
    (r"\b8-k\b", "8-k"),
    (r"\bedgar\b", "edgar"),
    (r"\bsec\b", "sec"),
    (r"\bfiling\b", "filing"),
    (r"\bfilings\b", "filings"),
    (r"\bannual report\b", "annual report"),
    (r"\bquarterly report\b", "quarterly report"),
]

const _CLIFF_PRODUCT_MARKERS = [
    (r"\breview\b", "review"),
    (r"\breviews\b", "reviews"),
    (r"\beasy to drive\b", "easy to drive"),
    (r"\beasy to run\b", "easy to run"),
    (r"\brunning shoe\b", "running shoe"),
    (r"\brunning shoes\b", "running shoes"),
    (r"\bshoe\b", "shoe"),
    (r"\bshoes\b", "shoes"),
    (r"\bdrive\b", "drive"),
    (r"\bdriving\b", "driving"),
    (r"\bcar\b", "car"),
    (r"\bvehicle\b", "vehicle"),
    (r"\bsteering\b", "steering"),
    (r"\bhandling\b", "handling"),
    (r"\bseat comfort\b", "seat comfort"),
    (r"\bcomfortable\b", "comfortable"),
    (r"\bcomfort\b", "comfort"),
    (r"\bsofa\b", "sofa"),
    (r"\bcouch\b", "couch"),
    (r"\bsectional\b", "sectional"),
    (r"\bmattress\b", "mattress"),
    (r"\bchair\b", "chair"),
    (r"\breturn risk\b", "return risk"),
    (r"\breturns\b", "returns"),
    (r"\bdurability\b", "durability"),
    (r"\bowners say\b", "owners say"),
    (r"\bfeedback\b", "feedback"),
]

const _CLIFF_CULINARY_MARKERS = [
    (r"\bculinary\b", "culinary"),
    (r"\bfood tour\b", "food tour"),
    (r"\brestaurant\b", "restaurant"),
    (r"\brestaurants\b", "restaurants"),
    (r"\bmeal\b", "meal"),
    (r"\bmeals\b", "meals"),
    (r"\bdining\b", "dining"),
    (r"\bdine\b", "dine"),
    (r"\bitinerary\b", "itinerary"),
    (r"\btravel\b", "travel"),
    (r"\bfood\b", "food"),
]

struct CLIFFRouteSpec
    name::Symbol
    module_name::String
    description::String
    default_rationale::String
    required_capabilities::Vector{Symbol}
    supported_execution_modes::Vector{Symbol}
    metadata::Dict{Symbol, Any}
end

function CLIFFRouteSpec(name;
                        module_name="",
                        description="",
                        default_rationale="",
                        required_capabilities::Vector{Symbol}=Symbol[],
                        supported_execution_modes::Vector{Symbol}=[:quick, :interactive, :deep],
                        metadata::Dict=Dict{Symbol, Any}())
    CLIFFRouteSpec(
        Symbol(name),
        String(module_name),
        String(description),
        String(default_rationale),
        copy(required_capabilities),
        copy(supported_execution_modes),
        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata),
    )
end

struct CLIFFRouteDecision
    route_name::Symbol
    module_name::String
    rationale::String
    execution_mode::Symbol
    matched_markers::Vector{String}
    required_capabilities::Vector{Symbol}
    metadata::Dict{Symbol, Any}
end

function CLIFFRouteDecision(route_name, module_name, rationale;
                            execution_mode=:quick,
                            matched_markers::Vector{String}=String[],
                            required_capabilities::Vector{Symbol}=Symbol[],
                            metadata::Dict=Dict{Symbol, Any}())
    CLIFFRouteDecision(
        Symbol(route_name),
        String(module_name),
        String(rationale),
        _normalize_cliff_execution_mode(execution_mode),
        copy(matched_markers),
        copy(required_capabilities),
        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata),
    )
end

struct CLIFFQueryRouter
    routes::OrderedDict{Symbol, CLIFFRouteSpec}
    default_route::Symbol
    metadata::Dict{Symbol, Any}
end

function CLIFFQueryRouter(routes::OrderedDict{Symbol, CLIFFRouteSpec};
                          default_route=:democritus,
                          metadata::Dict=Dict{Symbol, Any}())
    haskey(routes, Symbol(default_route)) || throw(ArgumentError("default_route must exist in the router"))
    CLIFFQueryRouter(
        deepcopy(routes),
        Symbol(default_route),
        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata),
    )
end

function _normalize_cliff_execution_mode(mode)
    normalized = mode isa Symbol ? mode : Symbol(lowercase(strip(String(mode))))
    normalized in (:quick, :interactive, :deep) || throw(ArgumentError("Unsupported CLIFF execution mode: $(mode)"))
    normalized
end

function _normalize_cliff_query(query)
    join(split(lowercase(String(query))), " ")
end

function _normalize_route_override(route_override)
    route_override === nothing && return :auto
    route_override isa Symbol && return route_override
    Symbol(lowercase(strip(String(route_override))))
end

function _marker_hits(normalized::AbstractString, marker_specs)
    hits = String[]
    for (pattern, label) in marker_specs
        occursin(pattern, normalized) && push!(hits, label)
    end
    unique(hits)
end

function _company_similarity_markers(normalized::AbstractString)
    similarity_markers = String[marker for marker in ("similar", "similarity", "compare", "comparison", "versus", " vs ") if occursin(marker, normalized)]
    isempty(similarity_markers) && return String[]
    connector_markers = String[String(strip(marker)) for marker in (" to ", " and ", " vs ", " versus ") if occursin(marker, normalized)]
    isempty(connector_markers) && return String[]
    unique(vcat(similarity_markers, connector_markers))
end

function _culinary_tour_markers(normalized::AbstractString)
    direct_hits = _marker_hits(normalized, _CLIFF_CULINARY_MARKERS)
    !isempty(direct_hits) && return direct_hits
    occursin(r"\btour\b", normalized) || return String[]
    supporting_hits = String[]
    for marker in ("budget", "dinner", "lunch", "breakfast", "meal", "meals", "weekend", "evening", "day")
        occursin(marker, normalized) && push!(supporting_hits, marker)
    end
    isempty(supporting_hits) ? String[] : unique(vcat(["tour"], supporting_hits))
end

function _course_demo_markers(normalized::AbstractString)
    markers = String[]
    occursin("category theory for agi", normalized) && push!(markers, "category theory for agi")
    occursin("textbook", normalized) && push!(markers, "textbook")
    occursin("course demo", normalized) && push!(markers, "course demo")
    occursin("julia demo", normalized) && push!(markers, "julia demo")
    if occursin("demo", normalized) && any(term -> occursin(term, normalized), ("kan extension", "sheaf", "democritus", "subobject classifier", "category theory", "agi"))
        push!(markers, "demo")
    end
    if occursin("course project", normalized) || occursin("project idea", normalized) || occursin("learning resource", normalized) || occursin("book section", normalized)
        push!(markers, "course support")
    end
    unique(markers)
end

looks_like_company_similarity_query(query) = !isempty(_company_similarity_markers(_normalize_cliff_query(query)))
looks_like_culinary_tour_query(query) = !isempty(_culinary_tour_markers(_normalize_cliff_query(query)))
looks_like_course_demo_query(query) = !isempty(_course_demo_markers(_normalize_cliff_query(query)))
looks_like_product_feedback_query(query) = !isempty(_marker_hits(_normalize_cliff_query(query), _CLIFF_PRODUCT_MARKERS))
looks_like_sec_query(query) = !isempty(_marker_hits(_normalize_cliff_query(query), _CLIFF_SEC_MARKERS))

function _default_cliff_route_specs()
    OrderedDict{Symbol, CLIFFRouteSpec}(
        :company_similarity => CLIFFRouteSpec(:company_similarity;
            module_name="functorflow_v3.company_similarity_agentic",
            description="Cross-company temporal diffusion comparison over routed corpora.",
            default_rationale="Query asks for cross-company similarity, so route to temporal diffusion construction and cross-company functor comparison.",
            required_capabilities=[:llm_inference, :retrieval]),
        :basket_rocket_sec => CLIFFRouteSpec(:basket_rocket_sec;
            module_name="functorflow_v3.basket_rocket_sec_agentic",
            description="SEC-backed BASKET/ROCKET ingress for filing workflows.",
            default_rationale="Query mentions SEC or filing-specific language, so route to the SEC-backed BASKET/ROCKET ingress.",
            required_capabilities=[:llm_inference, :retrieval]),
        :culinary_tour => CLIFFRouteSpec(:culinary_tour;
            module_name="functorflow_v3.culinary_tour_agentic",
            description="Interactive culinary-tour orchestration.",
            default_rationale="Query looks like a food, travel, or itinerary planning request, so route to the CLIFF culinary tour orchestrator.",
            required_capabilities=[:llm_inference]),
        :course_demo => CLIFFRouteSpec(:course_demo;
            module_name="functorflow_v3.course_demo_agentic",
            description="Category Theory for AGI textbook and demo launcher.",
            default_rationale="Query matches a registered Category Theory for AGI course demo, so route to the course notebook launcher.",
            required_capabilities=Symbol[]),
        :product_feedback => CLIFFRouteSpec(:product_feedback;
            module_name="functorflow_v3.product_feedback_query_agentic",
            description="Consumer product feedback retrieval and synthesis.",
            default_rationale="Query looks like a consumer product or review question, so route to product-feedback retrieval and analysis.",
            required_capabilities=[:llm_inference, :retrieval]),
        :democritus => CLIFFRouteSpec(:democritus;
            module_name="functorflow_v3.democritus_query_agentic",
            description="Open-ended evidence acquisition and synthesis.",
            default_rationale="Default route to Democritus for study, paper, corpus, and open-ended evidence acquisition queries.",
            required_capabilities=[:llm_inference, :retrieval]),
    )
end

build_cliff_query_router(; metadata::Dict=Dict{Symbol, Any}()) =
    CLIFFQueryRouter(_default_cliff_route_specs(); default_route=:democritus, metadata=metadata)

function _decision_for_override(router::CLIFFQueryRouter, route_override::Symbol; execution_mode)
    haskey(router.routes, route_override) || throw(ArgumentError("Unsupported route override: $(route_override)"))
    spec = router.routes[route_override]
    CLIFFRouteDecision(
        spec.name,
        spec.module_name,
        "Route override selected the $(String(spec.name)) route.";
        execution_mode=execution_mode,
        matched_markers=[String(spec.name)],
        required_capabilities=spec.required_capabilities,
        metadata=merge(copy(spec.metadata), Dict{Symbol, Any}(:override => true)),
    )
end

function route_cliff_query(router::CLIFFQueryRouter, query;
                           route_override=:auto,
                           execution_mode=:quick)
    normalized_query = _normalize_cliff_query(query)
    isempty(normalized_query) && throw(ArgumentError("A non-empty CLIFF query is required"))
    route_override = _normalize_route_override(route_override)
    execution_mode = _normalize_cliff_execution_mode(execution_mode)

    route_override == :auto || return _decision_for_override(router, route_override; execution_mode=execution_mode)

    company_markers = _company_similarity_markers(normalized_query)
    !isempty(company_markers) && begin
        spec = router.routes[:company_similarity]
        return CLIFFRouteDecision(spec.name, spec.module_name, spec.default_rationale;
            execution_mode=execution_mode,
            matched_markers=company_markers,
            required_capabilities=spec.required_capabilities,
            metadata=copy(spec.metadata))
    end

    sec_markers = _marker_hits(normalized_query, _CLIFF_SEC_MARKERS)
    !isempty(sec_markers) && begin
        spec = router.routes[:basket_rocket_sec]
        return CLIFFRouteDecision(spec.name, spec.module_name, spec.default_rationale;
            execution_mode=execution_mode,
            matched_markers=sec_markers,
            required_capabilities=spec.required_capabilities,
            metadata=copy(spec.metadata))
    end

    culinary_markers = _culinary_tour_markers(normalized_query)
    !isempty(culinary_markers) && begin
        spec = router.routes[:culinary_tour]
        return CLIFFRouteDecision(spec.name, spec.module_name, spec.default_rationale;
            execution_mode=execution_mode,
            matched_markers=culinary_markers,
            required_capabilities=spec.required_capabilities,
            metadata=copy(spec.metadata))
    end

    course_markers = _course_demo_markers(normalized_query)
    !isempty(course_markers) && begin
        spec = router.routes[:course_demo]
        return CLIFFRouteDecision(spec.name, spec.module_name, spec.default_rationale;
            execution_mode=execution_mode,
            matched_markers=course_markers,
            required_capabilities=spec.required_capabilities,
            metadata=copy(spec.metadata))
    end

    product_markers = _marker_hits(normalized_query, _CLIFF_PRODUCT_MARKERS)
    !isempty(product_markers) && begin
        spec = router.routes[:product_feedback]
        return CLIFFRouteDecision(spec.name, spec.module_name, spec.default_rationale;
            execution_mode=execution_mode,
            matched_markers=product_markers,
            required_capabilities=spec.required_capabilities,
            metadata=copy(spec.metadata))
    end

    spec = router.routes[router.default_route]
    CLIFFRouteDecision(spec.name, spec.module_name, spec.default_rationale;
        execution_mode=execution_mode,
        matched_markers=String[],
        required_capabilities=spec.required_capabilities,
        metadata=copy(spec.metadata))
end

route_cliff_query(query; route_override=:auto, execution_mode=:quick) =
    route_cliff_query(build_cliff_query_router(), query; route_override=route_override, execution_mode=execution_mode)

function as_dict(spec::CLIFFRouteSpec)
    Dict(
        "name" => String(spec.name),
        "module_name" => spec.module_name,
        "description" => spec.description,
        "default_rationale" => spec.default_rationale,
        "required_capabilities" => String.(spec.required_capabilities),
        "supported_execution_modes" => String.(spec.supported_execution_modes),
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in spec.metadata),
    )
end

function as_dict(decision::CLIFFRouteDecision)
    Dict(
        "route_name" => String(decision.route_name),
        "module_name" => decision.module_name,
        "rationale" => decision.rationale,
        "execution_mode" => String(decision.execution_mode),
        "matched_markers" => copy(decision.matched_markers),
        "required_capabilities" => String.(decision.required_capabilities),
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in decision.metadata),
    )
end

function as_dict(router::CLIFFQueryRouter)
    Dict(
        "routes" => [as_dict(spec) for spec in values(router.routes)],
        "default_route" => String(router.default_route),
        "metadata" => Dict(String(k) => _serialize_value(v) for (k, v) in router.metadata),
    )
end

to_json(spec::CLIFFRouteSpec) = JSON3.write(as_dict(spec))
to_json(decision::CLIFFRouteDecision) = JSON3.write(as_dict(decision))
to_json(router::CLIFFQueryRouter) = JSON3.write(as_dict(router))
