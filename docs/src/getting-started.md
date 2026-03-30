# Getting Started

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaKnowledge/FunctorFlow.jl")
```

## Your first diagram

The core workflow is:

1. build a `Diagram`
2. add objects and operations
3. compile it
4. run it on structured inputs

```julia
using FunctorFlow

D = Diagram(:MyKET)
add_object!(D, :Values; kind=:messages)
add_object!(D, :Incidence; kind=:relation)
add_left_kan!(D, :aggregate; source=:Values, along=:Incidence, reducer=:sum)

compiled = compile_to_callable(D)
result = run(compiled, Dict(
    :Values => Dict(:obj1 => 1.0, :obj2 => 2.0, :obj3 => 4.0),
    :Incidence => Dict(:left => [:obj1, :obj2], :right => [:obj2, :obj3]),
))
```

`result.values[:aggregate]` contains the aggregated context.

## Macro DSL

FunctorFlow also supports a more compact surface syntax:

```julia
using FunctorFlow

D = @functorflow MyPlanner begin
    Tokens::messages
    Nbrs::relation
    Ctx::contextualized_messages
    embed = Tokens → Ctx
    aggregate = Σ(:Tokens; along=:Nbrs, reducer=:sum)
end
```

This macro form is useful when the diagram structure itself is the main object of design.

## Named block builders

For common patterns, use prebuilt block constructors:

```julia
using FunctorFlow

ket = ket_block(; reducer=:mean)
db = db_square(; first_impl=x -> x + 1, second_impl=x -> 2x)
gt = gt_neighborhood_block()
planner = basket_rocket_pipeline()
```

These blocks are a convenient starting point for experimentation before dropping down to lower-level primitives.
