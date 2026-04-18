using Test
using FunctorFlow

# ---------------------------------------------------------------------------
# Helpers for building SCMs over a common toy domain.
#
# Ambient: U = {u1, u2, u3}, V = {a, b, c}
#   f_a uses {u1}; f_b uses {u2, a}; f_c uses {u3, a, b}
# ---------------------------------------------------------------------------
function _make_ambient()
    spec = SCMObjectSpec(:Ambient,
        [:u1, :u2, :u3], [:a, :b, :c],
        [
            SCMLocalFunctionSpec(:f_a, :a; exogenous_parents=[:u1], expression="a := u1"),
            SCMLocalFunctionSpec(:f_b, :b; exogenous_parents=[:u2], endogenous_parents=[:a],
                                  expression="b := u2 + a"),
            SCMLocalFunctionSpec(:f_c, :c; exogenous_parents=[:u3], endogenous_parents=[:a, :b],
                                  expression="c := u3 + a + b"),
        ])
    build_scm_model_object(spec; category=:StructuralCausalModels)
end

# Sub-SCM with V' = {a, b}, U' = {u1, u2}, identical mechanisms restricted.
function _make_strict_sub()
    spec = SCMObjectSpec(:StrictSub,
        [:u1, :u2], [:a, :b],
        [
            SCMLocalFunctionSpec(:f_a, :a; exogenous_parents=[:u1], expression="a := u1"),
            SCMLocalFunctionSpec(:f_b, :b; exogenous_parents=[:u2], endogenous_parents=[:a],
                                  expression="b := u2 + a"),
        ])
    build_scm_model_object(spec; category=:StructuralCausalModels)
end

# Renamed sub-SCM with V' = {x, y} ↪ {a, b}, U' = {p, q} ↪ {u1, u2}.
function _make_renamed_sub()
    spec = SCMObjectSpec(:RenamedSub,
        [:p, :q], [:x, :y],
        [
            SCMLocalFunctionSpec(:f_x, :x; exogenous_parents=[:p], expression="a := u1"),
            SCMLocalFunctionSpec(:f_y, :y; exogenous_parents=[:q], endogenous_parents=[:x],
                                  expression="b := u2 + a"),
        ])
    build_scm_model_object(spec; category=:StructuralCausalModels)
end

@testset "SCM monomorphism (sub-SCM inclusion)" begin
    ambient = _make_ambient()

    @testset "Identity inclusion succeeds" begin
        mono = build_scm_monomorphism(name=:id_inclusion,
                                      constrained_scm=ambient,
                                      ambient_scm=ambient)
        @test mono isa SCMMonomorphism
        @test mono.source_scm === ambient
        @test mono.target_scm === ambient
        @test !haskey(mono.metadata, :soft_intervention)
        @test mono.morphism.exogenous_variable_map == [(:u1, :u1), (:u2, :u2), (:u3, :u3)]
        @test mono.morphism.endogenous_variable_map == [(:a, :a), (:b, :b), (:c, :c)]
    end

    @testset "Strict subset inclusion succeeds" begin
        sub = _make_strict_sub()
        mono = build_scm_monomorphism(name=:strict_sub,
                                      constrained_scm=sub,
                                      ambient_scm=ambient)
        @test mono.source_scm === sub
        @test mono.target_scm === ambient
        @test mono.morphism.exogenous_variable_map == [(:u1, :u1), (:u2, :u2)]
        @test mono.morphism.endogenous_variable_map == [(:a, :a), (:b, :b)]
        @test mono.morphism.local_function_map == [(:f_a, :f_a), (:f_b, :f_b)]
        # Mechanisms agree on the inherited parent sets, so no soft intervention.
        @test !haskey(mono.metadata, :soft_intervention)
    end

    @testset "Renamed subset inclusion succeeds" begin
        sub = _make_renamed_sub()
        mono = build_scm_monomorphism(name=:renamed_sub,
                                      constrained_scm=sub,
                                      ambient_scm=ambient,
                                      variable_map=Dict(:x => :a, :y => :b,
                                                        :p => :u1, :q => :u2))
        @test mono.morphism.exogenous_variable_map == [(:p, :u1), (:q, :u2)]
        @test mono.morphism.endogenous_variable_map == [(:x, :a), (:y, :b)]
        @test mono.morphism.local_function_map == [(:f_x, :f_a), (:f_y, :f_b)]
    end

    @testset "Missing variable raises" begin
        bad = build_scm_model_object(
            SCMObjectSpec(:BadSub, [:u1], [:e],
                          [SCMLocalFunctionSpec(:f_e, :e; exogenous_parents=[:u1])]);
            category=:StructuralCausalModels)
        @test_throws ArgumentError build_scm_monomorphism(name=:bad,
                                                          constrained_scm=bad,
                                                          ambient_scm=ambient)
    end

    @testset "Parent-set violation raises" begin
        # Sub-SCM declares a parent (:u3) for :a that the ambient mechanism
        # for :a does not have.
        bad = build_scm_model_object(
            SCMObjectSpec(:OverParented, [:u1, :u3], [:a],
                          [SCMLocalFunctionSpec(:f_a, :a; exogenous_parents=[:u1, :u3])]);
            category=:StructuralCausalModels)
        @test_throws ArgumentError build_scm_monomorphism(name=:bad,
                                                          constrained_scm=bad,
                                                          ambient_scm=ambient)
    end

    @testset "Non-injective rename raises" begin
        sub = _make_renamed_sub()
        @test_throws ArgumentError build_scm_monomorphism(name=:collide,
            constrained_scm=sub, ambient_scm=ambient,
            variable_map=Dict(:x => :a, :y => :a, :p => :u1, :q => :u2))
    end

    @testset "Soft intervention via strict=false" begin
        # Same support as ambient :a but the expression differs (e.g. an
        # interventional override of :a with a different mechanism).
        soft_spec = SCMObjectSpec(:SoftSub,
            [:u1], [:a],
            [SCMLocalFunctionSpec(:f_a, :a; exogenous_parents=[:u1],
                                  expression="a := constant_intervention(u1)")])
        soft = build_scm_model_object(soft_spec; category=:StructuralCausalModels)

        @test_throws ArgumentError build_scm_monomorphism(name=:soft_strict,
            constrained_scm=soft, ambient_scm=ambient)

        mono = build_scm_monomorphism(name=:soft_loose,
                                      constrained_scm=soft,
                                      ambient_scm=ambient,
                                      strict=false)
        @test mono isa SCMMonomorphism
        @test get(mono.metadata, :soft_intervention, nothing) == Set([:a])
        @test get(mono.morphism.metadata, :soft_intervention, nothing) == Set([:a])
    end

    @testset "build_scm_subobject still works (backward compat)" begin
        clauses = [SCMPredicateClause(:c1, "true")]
        sub = build_scm_subobject(name=:Sub, ambient_scm=ambient, clauses=clauses)
        @test sub isa SCMSubobject
        @test sub.inclusion isa SCMMonomorphism
    end
end
