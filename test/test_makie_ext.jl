# ============================================================================
# test_makie_ext.jl — FunctorFlowMakieExt smoke tests
#
# Skipped unless a Makie backend is installable in the active environment.
# To exercise locally:  julia --project=. -e 'using Pkg; Pkg.add("CairoMakie")'
# then run the suite. CI installs a backend via the test target.
# ============================================================================

using Test
using FunctorFlow

const MAKIE_AVAILABLE = try
    @eval using CairoMakie
    true
catch err
    @info "test_makie_ext.jl: CairoMakie not available — skipping" error=err
    false
end

@testset "FunctorFlowMakieExt" begin
    if !MAKIE_AVAILABLE
        @test_skip "CairoMakie not installed"
    else
        # Without a backend loaded the shim should error; here one IS loaded,
        # so plot_diagram returns a Figure for every node kind.
        for builder in (ket_block, () -> db_square(; first_impl=x->x*2, second_impl=x->x+1),
                        jepa_block, completion_block)
            D = builder()
            fig = plot_diagram(D)
            @test fig isa CairoMakie.Makie.Figure
        end

        # In-place variant draws into an existing axis and returns it.
        fig = CairoMakie.Makie.Figure()
        ax = CairoMakie.Makie.Axis(fig[1, 1])
        out = plot_diagram!(ax, ket_block())
        @test out === ax

        # Saving produces a non-empty file.
        path = tempname() * ".png"
        CairoMakie.Makie.save(path, plot_diagram(ket_block()))
        @test isfile(path) && filesize(path) > 0
        rm(path; force=true)
    end
end
