# FunctorFlow.jl

FunctorFlow.jl is a Julia package for building AI systems as **categorical diagrams**.
It provides:

- a typed diagram DSL for objects, morphisms, Kan extensions, and obstruction losses
- a backend-neutral execution pipeline
- Lux-backed neural compilation for trainable architectures
- categorical extensions for universal constructions, causal semantics, sheaves, coalgebra, and JEPA-style world models

FunctorFlow.jl is a Julia port of the original Python
[FunctorFlow](https://github.com/sridharmahadevan/catagi) package by
Sridhar Mahadevan, and extends it with a Julia-native macro DSL, Lux integration,
and tighter interoperability with the AlgebraicJulia ecosystem.

## Documentation map

- [`Getting Started`](getting-started.md): installation, a first diagram, and macro syntax
- [`Core Concepts`](core-concepts.md): the package model of diagrams, Kan extensions, losses, and execution
- [`Block Library`](block-library.md): prebuilt categorical building blocks and training-oriented Lux helpers
- [`Vignettes`](vignettes.md): rendered tutorial notebooks and longer worked examples
- [`API Reference`](api.md): reference pages for the exported API

## Package highlights

FunctorFlow centers on a small set of primitives:

- `Diagram` for architecture specification
- `FFObject`, `Morphism`, `KanExtension`, and `ObstructionLoss` for categorical structure
- `compile_to_callable` for backend-neutral execution
- `compile_to_lux` for differentiable execution with Lux
- `@functorflow` and Unicode operators like `Σ`, `Δ`, and `→` for concise authoring

For longer worked examples, see the published [`Vignettes`](vignettes.md) page.
