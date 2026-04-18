using Test
using FunctorFlow

# Self-contained, opt-in Lean certificate verification test.
#
# Skipped by default. Set `FF_LEAN_CI=true` and ensure `lake` is on PATH
# to run the full pipeline:
#
#   1. emit a small diagram + construction certificate via FunctorFlow.jl
#   2. write them under proofs/FunctorFlowProofs/Generated/
#   3. invoke `lake build` in proofs/ and assert exit code 0
#   4. clean up the generated files
#
# Local recipe:
#   FF_LEAN_CI=true julia --project=. test/test_lean_certificates.jl

@testset "Lean certificate verification" begin
    if get(ENV, "FF_LEAN_CI", "false") != "true"
        @info "Skipping Lean certificate verification (set FF_LEAN_CI=true to enable)"
        return
    end

    lake = Sys.which("lake")
    if lake === nothing
        @warn "FF_LEAN_CI=true but `lake` is not on PATH; skipping"
        return
    end

    proofs_dir = abspath(joinpath(@__DIR__, "..", "proofs"))
    gen_dir = joinpath(proofs_dir, "FunctorFlowProofs", "Generated")
    @assert isdir(proofs_dir) "proofs/ directory missing"
    mkpath(gen_dir)

    written = String[]
    try
        D = ket_block()
        diag_cert = render_lean_certificate(D; module_name="CITestDiagram")
        diag_path = joinpath(gen_dir, "CITestDiagram.lean")
        write(diag_path, "import FunctorFlowProofs\n\n" * diag_cert)
        push!(written, diag_path)

        ket1 = ket_block(; name=:CITest1)
        ket2 = ket_block(; name=:CITest2)
        pb = pullback(ket1, ket2; over=:CITestShared)
        pb_cert = render_construction_certificate(pb; module_name="CITestPullback")
        pb_path = joinpath(gen_dir, "CITestPullback.lean")
        write(pb_path, "import FunctorFlowProofs\n\n" * pb_cert)
        push!(written, pb_path)

        cmd = setenv(`$lake build`, ENV; dir=proofs_dir)
        @info "Running `lake build` in $(proofs_dir) — this may download the Lean toolchain on first run"
        result = run(pipeline(cmd; stdout=stdout, stderr=stderr); wait=false)
        wait(result)
        @test result.exitcode == 0
    finally
        for f in written
            try; rm(f; force=true); catch; end
        end
    end
end
