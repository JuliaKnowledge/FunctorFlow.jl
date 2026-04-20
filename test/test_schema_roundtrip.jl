# test_schema_roundtrip.jl — round-trip via CategoricalDiagramSchema
#
# Exercises the FunctorFlowSchemaExt: Diagram → CategoricalDiagramACSet → Diagram

using FunctorFlow
using CategoricalDiagramSchema
using Catlab.CategoricalAlgebra: nparts, subpart, incident
using Test

@testset "Schema roundtrip via CategoricalDiagramSchema" begin

    @testset "Trivial 2-node diagram" begin
        D = FunctorFlow.Diagram(:Triv)
        FunctorFlow.add_object!(D, :A; kind=:value)
        FunctorFlow.add_object!(D, :B; kind=:value)
        FunctorFlow.add_morphism!(D, :f, :A, :B)
        acs = to_acset(D)
        @test acs isa CategoricalDiagramACSet
        @test nparts(acs, :Node) == 2
        @test nparts(acs, :Edge) == 1
        @test nparts(acs, :Kan) == 0
        @test nparts(acs, :ObsLoss) == 0
        D2 = from_acset(acs; name=:Triv2)
        @test D2.name == :Triv2
        @test Set(keys(D2.objects)) == Set([:A, :B])
        @test haskey(D2.operations, :f)
        @test D2.operations[:f].source == :A
        @test D2.operations[:f].target == :B
    end

    @testset "DB square" begin
        D = db_square(; first_impl=x->x, second_impl=x->x)
        acs = to_acset(D)
        @test nparts(acs, :Node) >= 1
        @test nparts(acs, :Edge) >= 4   # 2 morphisms + 2 compositions
        @test nparts(acs, :ObsLoss) == 1
        @test nparts(acs, :ObsPath) == 1
        D2 = from_acset(acs; name=:DBR)
        @test length(D2.losses) == 1
        loss = first(values(D2.losses))
        @test length(loss.paths) == 1
    end

    @testset "Linear chain via Composition preserves chain metadata" begin
        D = FunctorFlow.Diagram(:Chain)
        FunctorFlow.add_object!(D, :A); FunctorFlow.add_object!(D, :B); FunctorFlow.add_object!(D, :C)
        FunctorFlow.add_morphism!(D, :f, :A, :B)
        FunctorFlow.add_morphism!(D, :g, :B, :C)
        FunctorFlow.compose!(D, :f, :g; name=:fg)
        acs = to_acset(D)
        comp_eids = incident(acs, :composition, :edge_kind)
        @test length(comp_eids) == 1
        meta = subpart(acs, comp_eids[1], :edge_metadata)
        @test haskey(meta, :chain)
        @test Symbol.(meta[:chain]) == [:f, :g]
        D2 = from_acset(acs; name=:ChainR)
        @test haskey(D2.operations, :fg)
        @test D2.operations[:fg] isa FunctorFlow.Composition
        @test D2.operations[:fg].chain == [:f, :g]
    end

    @testset "Kan extension target synthesis (auto target)" begin
        D = FunctorFlow.Diagram(:Kanned)
        FunctorFlow.add_object!(D, :Tokens; kind=:messages)
        FunctorFlow.add_object!(D, :Nbrs; kind=:relation)
        FunctorFlow.add_left_kan!(D, :agg; source=:Tokens, along=:Nbrs, reducer=:sum)
        acs = to_acset(D)
        @test nparts(acs, :Kan) == 1
        # Auto-synthesised :agg_target node should exist
        ids = incident(acs, :agg_target, :node_name)
        @test length(ids) == 1
        meta = subpart(acs, ids[1], :node_metadata)
        @test get(meta, :auto_kan_target, false) === true
        D2 = from_acset(acs; name=:KannedR)
        # Auto target should NOT be re-introduced as a user object
        @test !haskey(D2.objects, :agg_target)
        @test haskey(D2.operations, :agg)
        kan = D2.operations[:agg]
        @test kan isa FunctorFlow.KanExtension
        @test kan.target === nothing
        @test kan.direction == FunctorFlow.LEFT
    end

    @testset "Kan extension explicit target preserved" begin
        D = FunctorFlow.Diagram(:Kanned2)
        FunctorFlow.add_object!(D, :S); FunctorFlow.add_object!(D, :R); FunctorFlow.add_object!(D, :T)
        FunctorFlow.add_right_kan!(D, :rk; source=:S, along=:R, target=:T, reducer=:first_non_null)
        acs = to_acset(D)
        @test nparts(acs, :Kan) == 1
        kid = 1
        tid = subpart(acs, kid, :kan_tgt)
        @test subpart(acs, tid, :node_name) == :T
        D2 = from_acset(acs; name=:Kanned2R)
        @test D2.operations[:rk].target == :T
        @test D2.operations[:rk].direction == FunctorFlow.RIGHT
    end

    @testset "ObstructionLoss roundtrips ObsLoss + ObsPath" begin
        D = FunctorFlow.Diagram(:Obs)
        FunctorFlow.add_object!(D, :X)
        FunctorFlow.add_morphism!(D, :p, :X, :X)
        FunctorFlow.add_morphism!(D, :q, :X, :X)
        FunctorFlow.add_obstruction_loss!(D, :L; paths=[(:p, :q)],
                                           comparator=:cosine, weight=2.5)
        acs = to_acset(D)
        @test nparts(acs, :ObsLoss) == 1
        @test nparts(acs, :ObsPath) == 1
        @test subpart(acs, 1, :obs_comparator) == :cosine
        @test subpart(acs, 1, :obs_weight) == 2.5
        D2 = from_acset(acs; name=:ObsR)
        @test length(D2.losses) == 1
        L = D2.losses[:L]
        @test L.comparator == :cosine
        @test L.weight == 2.5
        @test L.paths == [(:p, :q)]
    end

    @testset "Node shape/dtype propagate through ACSet" begin
        D = FunctorFlow.Diagram(:Shaped)
        FunctorFlow.add_object!(D, :T; kind=:tensor, shape="(3, 4)",
                                metadata=Dict{Symbol,Any}(:dtype=>Float32))
        acs = to_acset(D)
        @test subpart(acs, 1, :node_shape) == (3, 4)
        @test subpart(acs, 1, :node_dtype) === Float32
        D2 = from_acset(acs; name=:ShapedR)
        @test D2.objects[:T].shape == "(3, 4)"
        @test D2.objects[:T].metadata[:dtype] === Float32
    end

    @testset "Metadata roundtrips" begin
        D = FunctorFlow.Diagram(:Meta)
        FunctorFlow.add_object!(D, :A; description="alpha",
                                metadata=Dict{Symbol,Any}(:tag=>:hello))
        FunctorFlow.add_morphism!(D, :f, :A, :A;
                                  metadata=Dict{Symbol,Any}(:provenance=>:mine))
        acs = to_acset(D)
        D2 = from_acset(acs; name=:MetaR)
        @test D2.objects[:A].description == "alpha"
        @test D2.objects[:A].metadata[:tag] == :hello
        @test D2.operations[:f].metadata[:provenance] == :mine
    end

    @testset "ket_block roundtrip" begin
        D = ket_block()
        acs = to_acset(D)
        @test acs isa CategoricalDiagramACSet
        @test nparts(acs, :Kan) == 1
        D2 = from_acset(acs; name=:KETR)
        @test haskey(D2.objects, :Values)
        @test haskey(D2.objects, :Incidence)
    end

    @testset "diagram_to_acset / acset_to_diagram aliases" begin
        D = ket_block()
        acs = diagram_to_acset(D)
        @test acs isa CategoricalDiagramACSet
        D2 = acset_to_diagram(acs; name=:Aliased)
        @test D2.name == :Aliased
    end

    @testset "json_portable=true round-trip via cds_to_json" begin
        D = FunctorFlow.Diagram(:JP)
        FunctorFlow.add_object!(D, :x; shape="(10,)", kind=:input,
                                metadata=Dict{Symbol,Any}(:dtype=>Float32))
        FunctorFlow.add_object!(D, :y; shape="(4,)", kind=:output)
        FunctorFlow.add_morphism!(D, :f, :x, :y)

        acs = to_acset(D; json_portable=true)
        # JP-mode column types: shape→Vector{Int}, dtype→Symbol
        @test acs[1, :node_shape] isa AbstractVector
        @test acs[:, :node_dtype] isa AbstractVector{Symbol}
        # dtype was promoted out of metadata, so it must NOT also live there
        for md in acs[:, :node_metadata]
            @test !haskey(md, :dtype)
        end

        j = cds_to_json(acs)
        acs_back = cds_from_json(typeof(acs), j)
        @test cds_isequal(acs, acs_back)

        D2 = from_acset(acs_back; name=:JP2)
        @test D2.objects[:x].shape == "(10,)"
        @test D2.objects[:x].metadata[:dtype] === :Float32
        @test D2.objects[:y].shape == "(4,)"
        @test haskey(D2.operations, :f)
    end
end
