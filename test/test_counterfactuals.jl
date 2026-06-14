# ============================================================================
# test_counterfactuals.jl — counterfactuals grounded in identify_effect
# ============================================================================

using Test
using FunctorFlow

@testset "Relation polarity" begin
    @test relation_polarity("increases") == 1
    @test relation_polarity("reduces") == -1
    @test relation_polarity("leads to") == 1
    @test relation_polarity("inhibits") == -1
    @test relation_polarity("is correlated with") == 0
end

@testset "CausalTriple + DAG assembly" begin
    t = CausalTriple(:smoking, "increases", :cancer; domain="health")
    @test t.polarity == 1
    @test t.subj == :smoking && t.obj == :cancer

    # claim-string parser
    t2 = causal_triple("minimum wage -> employment"; rel="reduces")
    @test t2.subj == :minimum_wage && t2.obj == :employment && t2.polarity == -1

    # cycle breaking keeps the DAG acyclic
    cyc = [CausalTriple(:a, "increases", :b), CausalTriple(:b, "increases", :a)]
    G, dropped = build_causal_dag_from_triples(cyc)
    @test length(dropped) == 1
    @test length(G.directed) == 1
    @test topological_order(G) isa Vector{Symbol}   # acyclic ⇒ no throw

    # latent confounder becomes a bidirected edge
    G2, _ = build_causal_dag_from_triples([CausalTriple(:x, "increases", :y)];
                                          latent_pairs=[(:x, :y)])
    @test (:x, :y) in G2.bidirected
end

@testset "counterfactual_effect via identify_effect" begin
    # Front-door: smoking → tar → cancer with latent smoking ↔ cancer.
    triples = [CausalTriple(:smoking, "increases", :tar),
               CausalTriple(:tar, "increases", :cancer)]
    G, _ = build_causal_dag_from_triples(triples; latent_pairs=[(:smoking, :cancer)])
    cf = counterfactual_effect(G, triples, :smoking, :cancer)
    @test cf.identifiable                          # front-door criterion
    @test cf.estimand !== nothing
    @test cf.path == [:smoking, :tar, :cancer]
    @test cf.expected_direction == 1               # (+)·(+)
    @test cf.support == 2
    @test occursin("identifiable", cf.text)

    # decrease intervention flips the predicted sign
    cfd = counterfactual_effect(G, triples, :smoking, :cancer; intervention_level=:decrease)
    @test cfd.expected_direction == -1

    # mixed-polarity path: (+)·(−) = (−)
    t2 = [CausalTriple(:smoking, "increases", :tar),
          CausalTriple(:tar, "reduces", :fitness)]
    G2, _ = build_causal_dag_from_triples(t2)
    @test counterfactual_effect(G2, t2, :smoking, :fitness).expected_direction == -1

    # Bow arc X → Y with latent X ↔ Y is NOT identifiable (hedge witness).
    bow = [CausalTriple(:x, "increases", :y)]
    Gb, _ = build_causal_dag_from_triples(bow; latent_pairs=[(:x, :y)])
    cfb = counterfactual_effect(Gb, bow, :x, :y)
    @test !cfb.identifiable
    @test cfb.failure_reason == :hedge
    @test cfb.witness !== nothing

    # no causal path ⇒ direction 0
    np = [CausalTriple(:a, "increases", :b), CausalTriple(:c, "increases", :d)]
    Gn, _ = build_causal_dag_from_triples(np)
    @test counterfactual_effect(Gn, np, :a, :d).expected_direction == 0
end

@testset "build_counterfactuals_from_triples (batch)" begin
    triples = [CausalTriple(:smoking, "increases", :tar; domain="health"),
               CausalTriple(:tar, "increases", :cancer; domain="health")]
    payload = build_counterfactuals_from_triples(triples; domain="health")
    @test payload["counts"]["triples"] == 2
    @test payload["counts"]["counterfactuals"] == 2
    @test payload["counts"]["identifiable"] >= 1
    @test payload["domain"] == "health"
    @test length(payload["dag"]["edges"]) == 2

    # limit caps output
    @test build_counterfactuals_from_triples(triples; limit=1)["counts"]["counterfactuals"] == 1

    # JSON serialization round-trips
    cf = counterfactual_effect(first(build_causal_dag_from_triples(triples)), triples, :smoking, :cancer)
    @test occursin("intervention", to_json(cf))
end
