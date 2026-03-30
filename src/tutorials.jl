# ============================================================================
# tutorials.jl — Tutorial library packaging
# ============================================================================

"""
    TutorialLibrary(name, macro_names; adapter_library_names=Symbol[], description="")

A curated bundle of macro builders and adapter libraries for a specific
tutorial path.
"""
struct TutorialLibrary
    name::Symbol
    macro_names::Vector{Symbol}
    adapter_library_names::Vector{Symbol}
    description::String
end

function TutorialLibrary(name, macro_names;
                         adapter_library_names::Vector{Symbol}=Symbol[],
                         description::AbstractString="")
    TutorialLibrary(Symbol(name), Symbol.(macro_names),
                    adapter_library_names, String(description))
end

"""Build a diagram from a tutorial library's macro subset."""
function build_tutorial_macro(lib::TutorialLibrary, name::Union{Symbol, AbstractString}; kwargs...)
    n = Symbol(name)
    n in lib.macro_names || error("Macro :$n not in tutorial library :$(lib.name). Available: $(join(lib.macro_names, ", "))")
    build_macro(n; kwargs...)
end

"""Get the macro builders available in a tutorial library."""
function macro_builders(lib::TutorialLibrary)
    Dict(n => MACRO_LIBRARY[n] for n in lib.macro_names if haskey(MACRO_LIBRARY, n))
end

# Pre-built tutorial libraries

"""Core aggregation, commutativity, and geometric neighborhood blocks."""
const FOUNDATIONS_TUTORIAL_LIBRARY = TutorialLibrary(
    :foundations,
    [:ket, :completion, :structured_lm_duality, :db_square, :gt_neighborhood];
    description="Core aggregation, commutativity, and geometric neighborhood blocks."
)

"""Workflow drafting and repair blocks."""
const PLANNING_TUTORIAL_LIBRARY = TutorialLibrary(
    :planning,
    [:basket_workflow, :rocket_repair, :basket_rocket_pipeline, :democritus_assembly];
    adapter_library_names=[:standard],
    description="Workflow drafting and repair blocks."
)

"""Complete FunctorFlow macro set."""
const UNIFIED_TUTORIAL_LIBRARY = TutorialLibrary(
    :unified,
    sort(collect(keys(MACRO_LIBRARY)));
    adapter_library_names=[:standard],
    description="Complete FunctorFlow macro set."
)

const TUTORIAL_LIBRARIES = Dict{Symbol, TutorialLibrary}(
    :foundations => FOUNDATIONS_TUTORIAL_LIBRARY,
    :planning => PLANNING_TUTORIAL_LIBRARY,
    :unified => UNIFIED_TUTORIAL_LIBRARY,
)

"""
    get_tutorial_library(name) -> TutorialLibrary

Look up a registered tutorial library by name.
"""
function get_tutorial_library(name::Union{Symbol, AbstractString})
    get(TUTORIAL_LIBRARIES, Symbol(name)) do
        error("Tutorial library :$(name) not found")
    end
end

"""
    install_tutorial_library!(D, name_or_library)

Install a tutorial library's adapter libraries into a diagram.
"""
function install_tutorial_library!(D::Diagram, name::Union{Symbol, AbstractString})
    lib = get_tutorial_library(name)
    install_tutorial_library!(D, lib)
end

function install_tutorial_library!(D::Diagram, lib::TutorialLibrary)
    for adapter_name in lib.adapter_library_names
        use_adapter_library!(D, adapter_name)
    end
    lib
end
