# FunctorFlow.jl

[![Build Status](https://github.com/JuliaKnowledge/FunctorFlow.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaKnowledge/FunctorFlow.jl/actions)
[![Documentation](https://github.com/JuliaKnowledge/FunctorFlow.jl/actions/workflows/Documentation.yml/badge.svg)](https://JuliaKnowledge.github.io/FunctorFlow.jl/dev/)

A categorical DSL and executable IR for building diagrammatic AI systems in
Julia, grounded in [*Categories for AGI*](https://people.cs.umass.edu/~mahadeva/papers/catagi.pdf).

```
Diagram / Spec → Categorical IR → Neural Architecture
```

## Acknowledgement

FunctorFlow.jl is a Julia port of the Python
[FunctorFlow](https://github.com/sridharmahadevan/catagi) package by
**Sridhar Mahadevan**, which is the first executable software implementation
of the categorical systems developed in:

- Sridhar Mahadevan, [*Categories for AGI*](https://people.cs.umass.edu/~mahadeva/papers/catagi.pdf)
- Sridhar Mahadevan, [*Large Causal Models from Large Language Models*](https://arxiv.org/abs/2512.07796) (arXiv:2512.07796)
- The Lean 4 formalization: [catagi](https://github.com/sridharmahadevan/catagi)

This Julia implementation extends the original with Julia macro-based DSL
support, integration with the [AlgebraicJulia](https://www.algebraicjulia.org/)
ecosystem, [Lux.jl](https://github.com/LuxDL/Lux.jl) as the neural backend,
and v1 features including universal constructions and causal semantics.

## Installation

Until registration, install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaKnowledge/FunctorFlow.jl")
```

Once the package is registered, installation will simplify to:

```julia
using Pkg
Pkg.add("FunctorFlow")
```

## Quick start

### Programmatic API

```julia
using FunctorFlow

# Build a KET (Kan Extension Template) block
D = Diagram(:MyKET)
add_object!(D, :Values; kind=:messages)
add_object!(D, :Incidence; kind=:relation)
add_left_kan!(D, :aggregate; source=:Values, along=:Incidence, reducer=:sum)

# Compile and run
compiled = compile_to_callable(D)
result = run(compiled, Dict(
    :Values => Dict(1 => 1.0, 2 => 2.0, 3 => 4.0),
    :Incidence => Dict("left" => [1, 2], "right" => [2, 3])
))
result.values[:aggregate]
# Dict("left" => 3.0, "right" => 6.0)
```

### Macro DSL

```julia
using FunctorFlow

D = @diagram KET begin
    @object Tokens kind=:messages
    @object Nbrs kind=:relation
    @object Ctx kind=:contextualized_messages
    @left_kan aggregate source=Tokens along=Nbrs target=Ctx reducer=:sum
    @port input Tokens direction=:input type=:messages
    @port output aggregate direction=:output type=:contextualized_messages
end
```

### Named blocks

```julia
using FunctorFlow

# Pre-built block patterns
ket = ket_block(; name=:EdgeAggregator, reducer=:mean)
db  = db_square(; first_impl=x -> x*2, second_impl=x -> x+1)
gt  = gt_neighborhood_block()

# From the registry
diagram = build_macro(:ket; name=:TutorialKET)
```

## Core concepts

FunctorFlow operationalizes the categorical design language from *Categories for AGI*:

| Concept | FunctorFlow primitive | Covers |
|---------|----------------------|--------|
| **KET** (Kan Extension Transformer) | `left_kan` | Attention, pooling, message passing, context fusion |
| **DB** (Diagrammatic Backpropagation) | `obstruction_loss` | Commutativity control, consistency-aware learning |
| **GT** (Graph Transformer) | `gt_neighborhood_block` | Geometric message passing over simplicial structure |
| **BASKET** | `basket_workflow_block` | Plan fragment composition via left-Kan aggregation |
| **ROCKET** | `rocket_repair_block` | Plan repair via right-Kan completion |
| **Democritus** | `democritus_gluing_block` | Sheaf-theoretic local-to-global gluing |

## Compilation pipeline

```
Surface DSL (Diagram)
    ↓  to_ir()
Normalized IR (DiagramIR)
    ↓  compile_to_callable()
Backend-neutral executor (CompiledDiagram)
    ↓  [with Lux.jl extension]
Neural architecture (Lux layer)
```

## Diagram composition

Diagrams compose via namespaced inclusion:

```julia
parent = Diagram(:Parent)
child = ket_block()
inc = include!(parent, child; namespace=:encoder)

# Access namespaced elements
operation_ref(inc, :aggregate)  # :encoder__aggregate
port_spec(inc, :output)         # Port with namespaced ref
```

## v1 features

### Universal constructions

Build models through categorical universal properties:

```julia
ket1 = ket_block(; name=:KET1)
ket2 = ket_block(; name=:KET2)

# Pullback: joint constraint-compatible model
pb = pullback(ket1, ket2; over=:SharedContext)

# Product: independent combination
prod = product(ket1, ket2)

# Pushout: merge along shared interface
po = pushout(ket1, ket2; along=:SharedBase)
```

### Causal semantics (RN-Kan-Do-Calculus)

Explicit causal interpretation of Kan primitives:

```julia
ctx = CausalContext(:experiment;
    observational_regime=:obs,
    interventional_regime=:do)

cd = build_causal_diagram(:CausalModel; context=ctx)
# cd.base_diagram has:
#   :intervene (left-Kan → intervention/do-calculus)
#   :condition (right-Kan → conditioning/observational)
```

### Lean proof certificates

```julia
D = ket_block()
lean_code = render_lean_certificate(D)
write_lean_certificate(D; output_dir="proofs/generated")
```

## Dependencies

- **Required**: OrderedCollections.jl, JSON3.jl
- **Optional**: Lux.jl (neural backend), Catlab.jl (categorical algebra), Makie.jl (visualization)

## Related packages

- [Catlab.jl](https://github.com/AlgebraicJulia/Catlab.jl) — Applied category theory
- [ACSets.jl](https://github.com/AlgebraicJulia/ACSets.jl) — Attributed C-Sets
- [GATlab.jl](https://github.com/AlgebraicJulia/GATlab.jl) — Generalized Algebraic Theories
- [CSQL.jl](https://github.com/JuliaKnowledge/CSQL.jl) — Causal SQL databases
- [CQL.jl](https://github.com/JuliaKnowledge/CQL.jl) — Categorical Query Language

## License

MIT
