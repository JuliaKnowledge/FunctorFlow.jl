#!/usr/bin/env julia
# setup_local_dev.jl — develop sibling repos into the FunctorFlow test env
#
# `Pkg.test()` for FunctorFlow runs in a sandbox built from the [extras]
# in Project.toml. TinyGrad.jl and CategoricalDiagramSchema.jl are not
# included there because both are sibling private packages that aren't in
# the General registry. This script develops them into the FunctorFlow
# project so that:
#
#   * the FunctorFlowTinyGradExt extension is precompiled and exercised
#     by `test/test_tinygrad_ext.jl`
#   * the FunctorFlowSchemaExt-gated tests run
#
# Usage:
#   julia --project=. test/setup_local_dev.jl
#   julia --project=. -e 'using Pkg; Pkg.test()'
#
# Assumes TinyGrad.jl and CategoricalDiagramSchema.jl are sibling
# directories of FunctorFlow.jl in juliaknowledge/.

using Pkg

repo_root = abspath(joinpath(@__DIR__, "..", ".."))

specs = Pkg.PackageSpec[]
for name in ("TinyGrad.jl", "CategoricalDiagramSchema.jl")
    p = joinpath(repo_root, name)
    if isdir(p)
        push!(specs, Pkg.PackageSpec(path=p))
        @info "Will develop $name from $p"
    else
        @warn "$name not found at $p; corresponding tests will be skipped"
    end
end

if !isempty(specs)
    Pkg.develop(specs)
end

Pkg.resolve()
Pkg.instantiate()
@info "Local dev setup complete. Run: julia --project=. -e 'using Pkg; Pkg.test()'"
