# Macros and build_macro (Python)


- [Introduction](#introduction)
- [Setup](#setup)
- [The MACRO_LIBRARY](#the-macro_library)
- [Pre-Built Patterns with
  build_macro()](#pre-built-patterns-with-build_macro)
  - [KET Block (Left Kan Aggregation)](#ket-block-left-kan-aggregation)
  - [DB Square (Diagrammatic
    Backpropagation)](#db-square-diagrammatic-backpropagation)
  - [GT Neighborhood Block](#gt-neighborhood-block)
- [Customizing Macros with Keyword
  Arguments](#customizing-macros-with-keyword-arguments)
- [Manual Diagram Construction](#manual-diagram-construction)
  - [Basic Diagram (Julia @diagram
    equivalent)](#basic-diagram-julia-diagram-equivalent)
  - [Composition](#composition)
  - [Kan Extensions](#kan-extensions)
  - [Obstruction Loss](#obstruction-loss)
  - [Ports](#ports)
- [Comparing: build_macro vs Manual
  Construction](#comparing-build_macro-vs-manual-construction)
- [Julia vs Python: A Comparison](#julia-vs-python-a-comparison)

## Introduction

In Julia, FunctorFlow provides the `@diagram` macro — a concise,
declarative DSL for building categorical diagrams. Python does not have
a macro system, but FunctorFlow’s Python API provides an equivalent
capability through two mechanisms:

1.  **`build_macro()`** — a factory function that instantiates pre-built
    architectural patterns from the `MACRO_LIBRARY`.
2.  **Direct builder API** — `D.object()`, `D.morphism()`,
    `D.left_kan()`, etc. — the imperative interface that the Julia
    `@diagram` macro desugars into.

This vignette demonstrates both approaches and compares them to the
Julia DSL.

## Setup

``` python
from FunctorFlow import (
    Diagram,
    compile_to_callable,
    build_macro,
    MACRO_LIBRARY,
)
import json
```

## The MACRO_LIBRARY

The `MACRO_LIBRARY` is a dictionary mapping pattern names to factory
functions. Each factory returns a fully constructed `Diagram`.

``` python
print("Available macros:")
for name in sorted(MACRO_LIBRARY.keys()):
    print(f"  {name}")
```

    Available macros:
      basket_rocket_pipeline
      basket_workflow
      completion
      db_square
      democritus_gluing
      gt_neighborhood
      ket
      rocket_repair
      structured_lm_duality

## Pre-Built Patterns with build_macro()

### KET Block (Left Kan Aggregation)

The KET (Kan Extension Transformer) block is the canonical aggregation
pattern — a left Kan extension over an incidence relation.

``` python
D_ket = build_macro("ket")
print(D_ket.summary())
```

    Diagram(KETBlock)
      Objects: Values, Incidence, ContextualizedValues
      Operations: aggregate
      Losses: <none>
      Ports: input, relation, output

``` python
# The KET block is ready to compile and run
compiled_ket = compile_to_callable(D_ket)
result_ket = compiled_ket.run({
    "Values": {"a": 10, "b": 20, "c": 30},
    "Incidence": {"x": ["a", "b"], "y": ["b", "c"]}
})
print("KET aggregation (sum):", result_ket.values["aggregate"])
```

    KET aggregation (sum): {'x': 30, 'y': 50}

### DB Square (Diagrammatic Backpropagation)

The DB Square measures non-commutativity between two composed paths —
the foundation of obstruction-based learning.

``` python
D_db = build_macro("db_square")
print(D_db.summary())
```

    Diagram(DBSquare)
      Objects: State
      Operations: f, g, p1, p2
      Losses: obstruction
      Ports: input, left_path, right_path, loss

``` python
# Bind implementations to the morphisms
D_db.bind_morphism("f", lambda x: x + 1)
D_db.bind_morphism("g", lambda x: x * 2)

compiled_db = compile_to_callable(D_db)
result_db = compiled_db.run({"State": 3.0})
print(f"f∘g(3) = {result_db.values['p1']}")
print(f"g∘f(3) = {result_db.values['p2']}")
print(f"Obstruction loss: {result_db.losses['obstruction']}")
```

    f∘g(3) = 8.0
    g∘f(3) = 7.0
    Obstruction loss: 1.0

### GT Neighborhood Block

The GT (Graph Transformer) Neighborhood block lifts token states into
messages and aggregates them via a left Kan extension over a simplicial
relation.

``` python
D_gt = build_macro("gt_neighborhood")
print(D_gt.summary())
```

    Diagram(GTNeighborhoodBlock)
      Objects: Tokens, Messages, NeighborhoodIncidence, UpdatedTokens
      Operations: lift_messages, kan_aggregate
      Losses: <none>
      Ports: input, relation, messages, output

``` python
# The GT block has a lift morphism and a Kan aggregation
ir = D_gt.to_ir().as_dict()
print("Operations:")
for op in ir["operations"]:
    print(f"  {op['name']} ({op['kind']})")
print("\nPorts:")
for port in ir["ports"]:
    print(f"  {port['name']} -> {port['ref']} ({port['direction']})")
```

    Operations:
      lift_messages (morphism)
      kan_aggregate (kanextension)

    Ports:
      input -> Tokens (input)
      relation -> NeighborhoodIncidence (input)
      messages -> Messages (internal)
      output -> kan_aggregate (output)

## Customizing Macros with Keyword Arguments

Each `build_macro()` call accepts keyword arguments that override the
default configuration. This lets you customize object names, reducers,
and other parameters.

``` python
# KET block with mean reducer instead of sum
D_mean = build_macro("ket", reducer="mean")

compiled_mean = compile_to_callable(D_mean)
result_mean = compiled_mean.run({
    "Values": {"a": 10.0, "b": 20.0, "c": 30.0},
    "Incidence": {"x": ["a", "b"], "y": ["b", "c"]}
})
print("KET aggregation (mean):", result_mean.values["aggregate"])
```

    KET aggregation (mean): {'x': 15.0, 'y': 25.0}

``` python
# DB Square with custom morphism names
D_custom = build_macro("db_square",
                       first_morphism="encode",
                       second_morphism="decode")
print(D_custom.summary())
```

    Diagram(DBSquare)
      Objects: State
      Operations: encode, decode, p1, p2
      Losses: obstruction
      Ports: input, left_path, right_path, loss

## Manual Diagram Construction

For full control, you can construct diagrams imperatively. This is the
Python equivalent of what the Julia `@diagram` macro desugars into.

### Basic Diagram (Julia @diagram equivalent)

In Julia, you would write:

``` julia
D = @diagram SimpleTransform begin
    @object X kind=:input description="Input vector"
    @object Y kind=:output description="Output vector"
    @morphism transform X Y
end
```

In Python, the equivalent is:

``` python
D = Diagram("SimpleTransform")
D.object("X", kind="input", description="Input vector")
D.object("Y", kind="output", description="Output vector")
D.morphism("transform", "X", "Y")
print(D.summary())
```

    Diagram(SimpleTransform)
      Objects: X, Y
      Operations: transform
      Losses: <none>
      Ports: <none>

### Composition

Composition uses self-morphisms for natural chaining:

``` python
D3 = Diagram("Pipeline")
D3.object("S", kind="state")

D3.morphism("step_a", "S", "S")
D3.morphism("step_b", "S", "S")
D3.compose("step_a", "step_b", name="full_pipeline")

d3 = D3.to_ir().as_dict()
print("Operations:", [op["name"] for op in d3["operations"]])
```

    Operations: ['step_a', 'step_b', 'full_pipeline']

We can bind implementations and run:

``` python
D3.bind_morphism("step_a", lambda x: x + 10)
D3.bind_morphism("step_b", lambda x: x * 2)

compiled = compile_to_callable(D3)
result = compiled.run({"S": 5})
print("step_a(5) =", result.values["step_a"])
print("step_b(5) =", result.values["step_b"])
print("Pipeline (step_b ∘ step_a)(5) =", result.values["full_pipeline"])
```

    step_a(5) = 15
    step_b(5) = 10
    Pipeline (step_b ∘ step_a)(5) = 30

### Kan Extensions

``` python
D4 = Diagram("Aggregator")
D4.object("Messages", kind="messages")
D4.object("Neighbors", kind="relation")
D4.object("Pooled", kind="output")

D4.left_kan(source="Messages", along="Neighbors",
            target="Pooled", name="pool", reducer="mean")

compiled4 = compile_to_callable(D4)
result4 = compiled4.run({
    "Messages": {"a": 10.0, "b": 20.0, "c": 30.0},
    "Neighbors": {"x": ["a", "b"], "y": ["b", "c"]}
})
print("Pooled (mean):", result4.values["pool"])
```

    Pooled (mean): {'x': 15.0, 'y': 25.0}

Right Kan extensions use `D.right_kan()` and default to the
`"first_non_null"` reducer:

``` python
D5 = Diagram("Completer")
D5.object("Partial", kind="partial")
D5.object("Compat", kind="relation")

D5.right_kan(source="Partial", along="Compat", name="repair",
             reducer="first_non_null")

compiled5 = compile_to_callable(D5)
result5 = compiled5.run({
    "Partial": {"a": None, "b": 42, "c": None},
    "Compat": {"a": ["b", "c"], "c": ["b"]}
})
print("Repaired:", result5.values["repair"])
```

    Repaired: {'a': 42, 'c': 42}

### Obstruction Loss

``` python
D6 = Diagram("DBSquareDemo")
D6.object("S", kind="state")

D6.morphism("f", "S", "S")
D6.morphism("g", "S", "S")

D6.compose("f", "g", name="fg")
D6.compose("g", "f", name="gf")

D6.obstruction_loss(paths=[("fg", "gf")], name="consistency",
                    comparator="l2", weight=1.0)

D6.bind_morphism("f", lambda x: x + 1)
D6.bind_morphism("g", lambda x: x * 2)

compiled6 = compile_to_callable(D6)
result6 = compiled6.run({"S": 3.0})
print(f"f∘g(3) = {result6.values['fg']}")
print(f"g∘f(3) = {result6.values['gf']}")
print(f"Obstruction loss: {result6.losses['consistency']}")
```

    f∘g(3) = 8.0
    g∘f(3) = 7.0
    Obstruction loss: 1.0

The loss measures $\|f(g(x)) - g(f(x))\|_2$ — the degree to which $f$
and $g$ fail to commute.

### Ports

``` python
D7 = Diagram("PortedModel")
D7.object("Tokens", kind="messages")
D7.object("Neighbors", kind="relation")
D7.object("Output", kind="contextualized_messages")

D7.left_kan(source="Tokens", along="Neighbors",
            target="Output", name="aggregate", reducer="sum")

D7.expose_port("input", "Tokens", direction="input", port_type="messages")
D7.expose_port("relation", "Neighbors", direction="input", port_type="relation")
D7.expose_port("output", "aggregate", direction="output",
               port_type="contextualized_messages")

d7 = D7.to_ir().as_dict()
print("Ports:", [p["name"] for p in d7["ports"]])
```

    Ports: ['input', 'relation', 'output']

## Comparing: build_macro vs Manual Construction

Here is the same KET diagram built with `build_macro()` and with the
manual builder API.

**build_macro:**

``` python
D_macro = build_macro("ket")
print(f"Macro diagram: {D_macro.name} — "
      f"{len(D_macro.objects)} objects, "
      f"{len(D_macro.operations)} operations, "
      f"{len(D_macro.ports)} ports")
```

    Macro diagram: KETBlock — 3 objects, 1 operations, 3 ports

**Manual builder:**

``` python
D_manual = Diagram("KETManual")
D_manual.object("Values", kind="messages")
D_manual.object("Incidence", kind="relation")
D_manual.object("ContextualizedValues", kind="contextualized_messages")
D_manual.left_kan(source="Values", along="Incidence",
                  target="ContextualizedValues", name="aggregate",
                  reducer="sum")
D_manual.expose_port("input", "Values", direction="input", port_type="messages")
D_manual.expose_port("relation", "Incidence", direction="input", port_type="relation")
D_manual.expose_port("output", "aggregate", direction="output",
                     port_type="contextualized_messages")
print(f"Manual diagram: {D_manual.name} — "
      f"{len(D_manual.objects)} objects, "
      f"{len(D_manual.operations)} operations, "
      f"{len(D_manual.ports)} ports")
```

    Manual diagram: KETManual — 3 objects, 1 operations, 3 ports

Both produce structurally equivalent diagrams.

## Julia vs Python: A Comparison

| Feature | Julia | Python |
|----|----|----|
| Declarative DSL | `@diagram` macro | Not available |
| Pre-built patterns | `build_block(:ket)` | `build_macro("ket")` |
| Manual construction | `add_object!(D, :X; kind=:input)` | `D.object("X", kind="input")` |
| Morphism binding | `bind_morphism!(D, :f, fn)` | `D.bind_morphism("f", fn)` |
| Kan extensions | `add_left_kan!(D, :k; ...)` | `D.left_kan(source=..., name="k")` |
| Obstruction loss | `add_obstruction_loss!(D, :loss; ...)` | `D.obstruction_loss(paths=..., name="loss")` |
| Names | Symbols (`:X`) | Strings (`"X"`) |

In Julia, the `@diagram` macro provides a concise DSL that reduces
boilerplate. In Python, use `build_macro()` for pre-built patterns or
construct diagrams manually with the builder API. Both approaches
produce the same underlying `Diagram` objects and can be compiled and
executed identically.
