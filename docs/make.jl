using Pkg

Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
Pkg.instantiate()

using Documenter
using FunctorFlow

DocMeta.setdocmeta!(FunctorFlow, :DocTestSetup, :(using FunctorFlow); recursive=true)

const CI_BUILD = get(ENV, "CI", "false") == "true"

makedocs(
    sitename = "FunctorFlow.jl",
    modules = [FunctorFlow],
    authors = "Simon Frost",
    clean = true,
    checkdocs = :none,
    format = Documenter.HTML(
        prettyurls=CI_BUILD,
        edit_link="main",
        repolink="https://github.com/JuliaKnowledge/FunctorFlow.jl",
    ),
    remotes = CI_BUILD ? Dict(joinpath(@__DIR__, "..") => Documenter.Remotes.GitHub("JuliaKnowledge", "FunctorFlow.jl")) : nothing,
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting-started.md",
        "Core Concepts" => "core-concepts.md",
        "Block Library" => "block-library.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/JuliaKnowledge/FunctorFlow.jl.git",
    devbranch = "main",
)
