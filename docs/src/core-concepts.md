# Core Concepts

## Diagrams

A `Diagram` is the package's central representation of a structured AI system.
Objects define interfaces, morphisms define transformations, Kan extensions define universal
aggregation or completion, and obstruction losses encode consistency constraints.

## Kan extensions

FunctorFlow uses categorical language directly:

- `Σ` / left Kan extension for aggregation, pooling, attention, and message passing
- `Δ` / right Kan extension for completion, repair, conditioning, and reconciliation

These are available both as functions and as Unicode operators.

## Obstruction losses

An `ObstructionLoss` measures how far a diagram is from commuting.
This is the package's basic mechanism for diagrammatic consistency:

- symbolic workflows can compare strings, sets, or structured values
- neural workflows can optimize differentiable comparators such as `:l2`, `:l1`, or cosine-style losses

## Compilation pipeline

FunctorFlow separates specification from execution:

```text
Diagram
  -> DiagramIR
  -> CompiledDiagram
  -> callable or Lux-backed executable model
```

The same categorical structure can therefore be used for both symbolic and differentiable programs.

## Beyond the core

The package also includes higher-level categorical layers:

- universal constructions such as `pullback`, `pushout`, `product`, and `coequalizer`
- causal semantics via `CausalContext` and `build_causal_diagram`
- topos-style reasoning tools such as `SubobjectClassifier`
- coalgebra and JEPA-inspired world-model structures

These features extend FunctorFlow from a DSL for blocks into a broader neurosymbolic research toolkit.
