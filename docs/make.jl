using Pkg

Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
Pkg.instantiate()

using Documenter
using FunctorFlow

DocMeta.setdocmeta!(FunctorFlow, :DocTestSetup, :(using FunctorFlow); recursive=true)

const CI_BUILD = get(ENV, "CI", "false") == "true"
const DOCS_ROOT = @__DIR__
const REPO_ROOT = normpath(joinpath(DOCS_ROOT, ".."))
const VIGNETTES_ROOT = joinpath(REPO_ROOT, "vignettes")

function publish_vignettes!(build_root::AbstractString)
    target_root = joinpath(build_root, "vignettes")
    mkpath(target_root)

    for entry in sort(readdir(VIGNETTES_ROOT))
        src = joinpath(VIGNETTES_ROOT, entry)
        if isdir(src) && occursin(r"^\d{2}-", entry)
            dst = joinpath(target_root, entry)
            rm(dst; recursive=true, force=true)
            cp(src, dst; force=true)
        end
    end
end

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
        "Vignettes" => "vignettes.md",
        "API Reference" => "api.md",
    ],
)

publish_vignettes!(joinpath(DOCS_ROOT, "build"))

deploydocs(
    repo = "github.com/JuliaKnowledge/FunctorFlow.jl.git",
    devbranch = "main",
)
