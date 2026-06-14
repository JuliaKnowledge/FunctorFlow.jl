using Test
using FunctorFlow

# Self-contained, opt-in Lean certificate verification test.
#
# Skipped by default. Set `FF_LEAN_CI=true` and ensure `lake` is on PATH
# to run the full pipeline:
#
#   1. emit diagram / construction / JEPA certificates via FunctorFlow.jl
#   2. write them under proofs/FunctorFlowProofs/Generated/
#   3. `lake build` and assert the genuine certificates type-check
#   4. emit a *nonzero-obstruction* certificate and assert it is REJECTED
#      (the exactness theorems are falsifiable, not vacuous)
#   5. clean up the generated files
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
    rm(gen_dir; force=true, recursive=true)
    mkpath(gen_dir)

    emit(name, body) = write(joinpath(gen_dir, "$name.lean"),
                             "import FunctorFlowProofs\n\n" * body)
    lakebuild(target=nothing) = begin
        cmd = target === nothing ? `$lake build` : `$lake build $target`
        run(pipeline(ignorestatus(setenv(cmd, ENV; dir=proofs_dir));
                     stdout=stdout, stderr=stderr)).exitcode
    end

    try
        # ---- Genuine certificates (must all type-check) ----
        emit("CITestDiagram", render_lean_certificate(ket_block(); module_name="CITestDiagram"))
        emit("CITestDb",      render_lean_certificate(db_square(; first_impl=x->x, second_impl=x->x);
                                                      module_name="CITestDb"))

        ket1 = ket_block(; name=:CITest1); ket2 = ket_block(; name=:CITest2)
        emit("CITestPullback",  render_construction_certificate(pullback(ket1, ket2; over=:CIShared);  module_name="CITestPullback"))
        emit("CITestProduct",   render_construction_certificate(product(ket1, ket2);                    module_name="CITestProduct"))
        emit("CITestCoproduct", render_construction_certificate(coproduct(ket1, ket2);                  module_name="CITestCoproduct"))

        P = Diagram(:CIPar)
        add_object!(P, :A; kind=:state); add_object!(P, :B; kind=:state)
        add_morphism!(P, :f, :A, :B; implementation=x->x)
        add_morphism!(P, :g, :A, :B; implementation=x->x)
        emit("CITestEqualizer",   render_construction_certificate(equalizer(P, :f, :g);   module_name="CITestEqualizer"))
        emit("CITestCoequalizer", render_construction_certificate(coequalizer(P, :f, :g); module_name="CITestCoequalizer"))

        J = jepa_block()
        add_bisimulation!(J, :b1; coalgebra_a=:jepa_dynamics, coalgebra_b=:jepa_dynamics, relation=:behavioral_eq)
        add_energy_function!(J, :e1; domain=[:ContextRepr, :TargetRepr], energy_type=:vicreg)
        emit("CITestJepa", render_jepa_certificate(J; module_name="CITestJepa"))

        # Cat-kernel certificates: a presented category + a functor, machine-checked
        kcat = FunctorFlow.Cat
        emit("CITestCat", render_cat_certificate(kcat.commutative_square(); module_name="CITestCat"))
        arrow = FreeCat([:a, :b], [(:f, :a, :b)])
        chain = FreeCat([:a, :b, :cc], [(:f, :a, :b), (:g, :b, :cc)])
        Fun = FinFunctor(arrow, chain; ob_map=Dict(:a => :a, :b => :cc),
                         edge_map=Dict(:f => kcat.homset(chain, :a, :cc)[1]))
        emit("CITestFunctor", render_functor_certificate(Fun; module_name="CITestFunctor"))
        # adjunction (initial object) and monad (closure operator) certificates
        emit("CITestAdj", render_adjunction_certificate(
            kcat.initial_object_adjunction(chain, :a); module_name="CITestAdj"))
        emit("CITestMonad", render_monad_certificate(
            kcat.closure_monad(chain, Dict(:a => :b, :b => :b, :cc => :cc)); module_name="CITestMonad"))
        # colimit (pushout) and limit (pullback) certificates — the Kan/colimit laws
        span = FreeCat([:s, :l, :r], [(:il, :s, :l), (:ir, :s, :r)])
        Xsp = kcat.SetFunctor(span;
            ob_map=Dict(:s => kcat.FinSet([1]), :l => kcat.FinSet([:x, :y]), :r => kcat.FinSet([:u, :v])),
            edge_map=Dict(:il => kcat.FinFunction(kcat.FinSet([1]), kcat.FinSet([:x, :y]), [1 => :x]),
                          :ir => kcat.FinFunction(kcat.FinSet([1]), kcat.FinSet([:u, :v]), [1 => :u])))
        emit("CITestColimit", render_colimit_certificate(kcat.colimit(Xsp); module_name="CITestColimit"))
        cospan = FreeCat([:l, :r, :s], [(:pl, :l, :s), (:pr, :r, :s)])
        Xcs = kcat.SetFunctor(cospan;
            ob_map=Dict(:l => kcat.FinSet([:x, :y]), :r => kcat.FinSet([:u, :v]), :s => kcat.FinSet([1, 2])),
            edge_map=Dict(:pl => kcat.FinFunction(kcat.FinSet([:x, :y]), kcat.FinSet([1, 2]), [:x => 1, :y => 2]),
                          :pr => kcat.FinFunction(kcat.FinSet([:u, :v]), kcat.FinSet([1, 2]), [:u => 1, :v => 2])))
        emit("CITestLimit", render_limit_certificate(kcat.limit(Xcs); module_name="CITestLimit"))
        # backprop-as-functor: the chain rule for a 2-layer ℤ₇ network
        W1 = kcat.LinMap(7, 3, 2, [1 0 2; 0 1 1])
        W2 = kcat.LinMap(7, 2, 1, reshape([1, 1], 1, 2))
        emit("CITestBackprop", render_backprop_certificate(W1, W2; module_name="CITestBackprop"))
        # bisimulation (behavioural equivalence) of a Moore machine
        MM = kcat.MooreMachine([:s0, :s1, :s2], [:a], [:x, :y],
            Dict((:s0, :a) => :s1, (:s1, :a) => :s2, (:s2, :a) => :s1),
            Dict(:s0 => :x, :s1 => :y, :s2 => :y))
        emit("CITestBisim", render_bisimulation_certificate(MM,
            [(:s1, :s2), (:s2, :s1), (:s1, :s1), (:s2, :s2), (:s0, :s0)]; module_name="CITestBisim"))
        # enriched/metric and lens-law certificates
        emit("CITestMetric", render_metric_certificate(
            kcat.embedding_metric(Dict(:a => [0, 0], :b => [3, 0], :c => [3, 4]); metric=:l1);
            module_name="CITestMetric"))
        emit("CITestLens", render_lens_certificate(
            kcat.record_lens([:a1, :a2], [:b1, :b2]); module_name="CITestLens"))
        # Heyting (internal logic) and Galois-connection certificates
        emit("CITestHeyting", render_heyting_certificate(
            kcat.cosieve_heyting(arrow, :a); module_name="CITestHeyting"))
        Pp = kcat.Poset([0, 1, 2], Dict((x, y) => (x <= y) for x in 0:2 for y in 0:2))
        emit("CITestGalois", render_galois_certificate(Pp, Pp,
            Dict(x => x for x in 0:2), Dict(x => x for x in 0:2); module_name="CITestGalois"))

        @info "Running `lake build` on genuine certificates (may download the toolchain on first run)"
        @test lakebuild() == 0

        # ---- Negative control: a nonzero obstruction loss must be REJECTED ----
        D = db_square(; first_impl=x->x, second_impl=x->x)
        lossname = first(keys(D.losses))
        emit("CITestNonzero", render_lean_certificate(D; module_name="CITestNonzero",
                                                      loss_values=Dict(lossname => 7)))
        @info "Building the nonzero-obstruction certificate (expected to FAIL — proving the exactness proof is falsifiable)"
        @test lakebuild("FunctorFlowProofs.Generated.CITestNonzero") != 0
        rm(joinpath(gen_dir, "CITestNonzero.lean"); force=true)

        # Negative control 2: a corrupted category table (empty composition) must
        # fail isCategory — the Cat-kernel certification has teeth too.
        broken = replace(render_cat_certificate(chain; module_name="CITestCatBroken"),
                         r"comp := \[[^\]]*\]" => "comp := []")
        emit("CITestCatBroken", broken)
        @info "Building the corrupted category certificate (expected to FAIL)"
        @test lakebuild("FunctorFlowProofs.Generated.CITestCatBroken") != 0
        rm(joinpath(gen_dir, "CITestCatBroken.lean"); force=true)

        # Negative control 3: a corrupted colimit apex must fail isColimit.
        brokenColim = replace(render_colimit_certificate(kcat.colimit(Xsp); module_name="CITestColimBroken"),
                              r"apex := \[[^\]]*\]" => "apex := [\"merged\"]")
        emit("CITestColimBroken", brokenColim)
        @info "Building the corrupted colimit certificate (expected to FAIL)"
        @test lakebuild("FunctorFlowProofs.Generated.CITestColimBroken") != 0
        rm(joinpath(gen_dir, "CITestColimBroken.lean"); force=true)
    finally
        rm(gen_dir; force=true, recursive=true)
    end
end
