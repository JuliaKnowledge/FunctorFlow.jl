using FunctorFlow
using Catlab
using Test

@testset "FunctorFlowCatlabExt" begin
    @testset "diagram_to_acset — KET block" begin
        D = ket_block()
        acs = diagram_to_acset(D)
        @test nparts(acs, :Node) == length(D.objects)
        @test nparts(acs, :Edge) >= 1
        names = subpart(acs, :node_name)
        @test :Values in names
        @test :Incidence in names
    end

    @testset "diagram_to_acset — DB square" begin
        D = db_square(; first_impl=x -> x * 2, second_impl=x -> x + 1)
        acs = diagram_to_acset(D)
        @test nparts(acs, :Node) == length(D.objects)
        edge_names = subpart(acs, :edge_name)
        @test :f in edge_names
        @test :g in edge_names
        optypes = subpart(acs, :edge_optype)
        @test :morphism in optypes || :composition in optypes
    end

    @testset "diagram_to_acset — morphism edges" begin
        D = FunctorFlow.Diagram(:MorphTest)
        FunctorFlow.add_object!(D, :A; kind=:value)
        FunctorFlow.add_object!(D, :B; kind=:value)
        FunctorFlow.add_morphism!(D, :f, :A, :B)
        acs = diagram_to_acset(D)
        @test nparts(acs, :Node) == 2
        @test nparts(acs, :Edge) == 1
        @test subpart(acs, 1, :edge_optype) == :morphism
    end

    @testset "acset_to_diagram roundtrip" begin
        D = ket_block()
        acs = diagram_to_acset(D)
        D2 = acset_to_diagram(acs; name=:Roundtrip)
        @test D2.name == :Roundtrip
        @test Set(keys(D2.objects)) == Set(keys(D.objects))
        @test length(D2.operations) >= 1
    end

    @testset "acset_to_diagram with morphisms" begin
        D = FunctorFlow.Diagram(:MorphTest)
        FunctorFlow.add_object!(D, :A; kind=:value)
        FunctorFlow.add_object!(D, :B; kind=:value)
        FunctorFlow.add_morphism!(D, :f, :A, :B)
        acs = diagram_to_acset(D)
        D2 = acset_to_diagram(acs)
        @test haskey(D2.objects, :A)
        @test haskey(D2.objects, :B)
        @test haskey(D2.operations, :f)
    end

    @testset "define_theory" begin
        D1 = ket_block(; name=:KET)
        D2 = db_square(; first_impl=x -> x, second_impl=x -> x)
        obj1 = CategoricalModelObject(D1)
        obj2 = CategoricalModelObject(D2)
        theory = define_theory([obj1, obj2])
        @test theory isa Catlab.Theories.Presentation
        gens = generators(theory)
        @test length(gens) >= 2  # at least the two model objects
    end

    @testset "diagram_to_free — symbolic morphisms" begin
        D = FunctorFlow.Diagram(:SymTest)
        FunctorFlow.add_object!(D, :A; kind=:value)
        FunctorFlow.add_object!(D, :B; kind=:value)
        FunctorFlow.add_object!(D, :C; kind=:value)
        FunctorFlow.add_morphism!(D, :f, :A, :B)
        FunctorFlow.add_morphism!(D, :g, :B, :C)

        ext = Base.get_extension(FunctorFlow, :FunctorFlowCatlabExt)
        sym = ext.diagram_to_free(D)
        @test haskey(sym.objects, :A)
        @test haskey(sym.objects, :B)
        @test haskey(sym.objects, :C)
        @test haskey(sym.morphisms, :f)
        @test haskey(sym.morphisms, :g)
        @test dom(sym.morphisms[:f]) == sym.objects[:A]
        @test codom(sym.morphisms[:f]) == sym.objects[:B]
    end

    @testset "verify_naturality" begin
        obj = CategoricalModelObject(:X)
        α = NaturalTransformation(:alpha, :F, :G;
                                   components=Dict(:X => identity))
        ext = Base.get_extension(FunctorFlow, :FunctorFlowCatlabExt)
        result = ext.verify_naturality(α, [obj])
        @test result.passed == true

        # Missing component
        α2 = NaturalTransformation(:alpha2, :F, :G;
                                    components=Dict(:Z => identity))
        result2 = ext.verify_naturality(α2, [obj])
        @test result2.passed == false
    end
end
