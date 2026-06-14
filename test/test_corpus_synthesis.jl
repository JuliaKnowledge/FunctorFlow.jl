# ============================================================================
# test_corpus_synthesis.jl — multi-document causal-claim synthesis
# ============================================================================

using Test
using FunctorFlow

@testset "Truth tiers" begin
    @test corpus_truth_value(3, 3) == :entailed
    @test corpus_truth_value(2, 3) == :strong_support      # 0.667 ≥ 0.5
    @test corpus_truth_value(2, 5) == :provisional_support # ≥2 docs
    @test corpus_truth_value(2, 6) == :provisional_support # 0.333 but ≥2 docs
    @test corpus_truth_value(1, 5) == :weak_support
    @test corpus_truth_value(1, 1) == :entailed
end

@testset "Gluing variants across documents" begin
    # identical extracted entities, different documents → one glued claim, support 3
    claims = [CorpusClaim("a", "increases", "b"; document="d$i") for i in 1:3]
    glued = glue_corpus_claims(claims)
    @test length(glued) == 1
    @test glued[1].support == 3
    @test glued[1].truth_value == :entailed
    @test length(glued[1].variants) == 3

    # token-Jaccard merge of wording variants (≥0.65)
    vc = [CorpusClaim("ocean_warming", "reduces", "fish_population"; document="d1"),
          CorpusClaim("ocean_temperature_warming", "reduces", "fish_population"; document="d2")]
    @test length(glue_corpus_claims(vc)) == 1               # {ocean,warming} vs {ocean,temperature,warming}=0.667

    # below-threshold variants stay separate
    far = [CorpusClaim("cats", "increase", "joy"; document="d1"),
           CorpusClaim("dogs", "increase", "joy"; document="d2")]
    @test length(glue_corpus_claims(far)) == 2

    # polarity conflict flagged, not split
    conf = [CorpusClaim("x", "increases", "y"; document="d1"),
            CorpusClaim("x", "reduces", "y"; document="d2")]
    g = glue_corpus_claims(conf)
    @test length(g) == 1
    @test g[1].conflicted
end

@testset "Simplicial horn-fill coherence" begin
    # filled triangle a→b, b→c, a→c  ⇒ ratio 1.0
    full = glue_corpus_claims([CorpusClaim("a","→","b";document="d"),
                               CorpusClaim("b","→","c";document="d"),
                               CorpusClaim("a","→","c";document="d")])
    m = homotopy_coherence(full)
    @test m.triangles == 1
    @test m.open_horns == 0
    @test m.horn_fill_ratio == 1.0
    @test m.state == :coherent

    # open 2-horn a→b, b→c with a→c missing ⇒ ratio 0.0
    open2 = glue_corpus_claims([CorpusClaim("a","→","b";document="d"),
                                CorpusClaim("b","→","c";document="d")])
    mo = homotopy_coherence(open2)
    @test mo.triangles == 0
    @test mo.open_horns == 1
    @test mo.horn_fill_ratio == 0.0

    # two disconnected edges ⇒ two components
    disc = glue_corpus_claims([CorpusClaim("a","→","b";document="d"),
                               CorpusClaim("c","→","d";document="d")])
    @test homotopy_coherence(disc).components == 2
end

@testset "Query alignment" begin
    c = glue_corpus_claims([CorpusClaim("minimum_wage","raises","employment"; document="d")])[1]
    s_hi, lbl_hi = query_alignment(c, "how does minimum wage affect employment")
    s_lo, lbl_lo = query_alignment(c, "penguin migration patterns")
    @test s_hi > s_lo
    @test lbl_hi in (:high, :moderate)
    @test lbl_lo == :low
end

@testset "End-to-end synthesis example" begin
    ex = build_corpus_synthesis_example()
    res = synthesize_corpus(ex.claims; query=ex.query)

    @test res.n_documents == 3
    # corroborated claim entailed across all 3 docs
    @test any(c -> c.canonical_obj == :earnings && c.truth_value == :entailed && c.support == 3, res.claims)
    # polarity disagreement surfaced
    @test length(res.disagreements) >= 1
    @test any(c -> c.canonical_obj == :employment && c.conflicted, res.disagreements)
    # transitive chain fills a triangle
    @test res.coherence.triangles >= 1
    @test res.coherence.horn_fill_ratio > 0.0

    # ranking puts the query-relevant entailed claim near the top
    @test res.claims[1].relevance >= res.claims[end].relevance

    # summary / JSON
    summ = summarize_corpus_synthesis(res)
    @test summ["n_documents"] == 3
    @test haskey(summ["counts"]["by_truth_value"], "entailed")
    @test occursin("minimum", to_json(res))

    # limit caps returned claims
    @test length(synthesize_corpus(ex.claims; query=ex.query, limit=2).claims) == 2
end

@testset "Claim gluing is a colimit (categorical re-founding)" begin
    C = FunctorFlow.Cat
    ex = build_corpus_synthesis_example()
    # the gluing diagram is a genuine functor; its colimit is the glued corpus
    @test C.is_functorial(corpus_gluing_diagram(ex.claims))
    col = corpus_colimit(ex.claims)
    @test C.verify_colimit(col)
    # the categorical colimit agrees with the engine's equivalence classes
    @test length(col.apex) == length(glue_corpus_claims(ex.claims))
    # no variants ⇒ identity colimit (one class per claim)
    distinct = [CorpusClaim("p$i", "increases", "q$i"; document="d") for i in 1:3]
    @test length(corpus_colimit(distinct).apex) == 3
end
