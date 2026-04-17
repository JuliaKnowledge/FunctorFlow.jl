# Obstruction Loss — Diagrammatic Backpropagation (Python)


- [Introduction](#introduction)
- [Setup](#setup)
- [Non-Commuting Squares](#non-commuting-squares)
- [The `paths` Parameter](#the-paths-parameter)
- [Comparators](#comparators)
  - [Built-in: L2 (default)](#built-in-l2-default)
  - [Custom Comparators](#custom-comparators)
- [Multi-Path Losses](#multi-path-losses)
- [Training Interpretation](#training-interpretation)
- [Weighted Losses](#weighted-losses)
- [Full DB Square](#full-db-square)
- [Proof Certificate](#proof-certificate)
- [Summary](#summary)

## Introduction

In category theory, a diagram **commutes** when all directed paths
between the same pair of objects yield the same result. In practice,
neural or learned morphisms rarely commute exactly. **Obstruction loss**
quantifies this failure:

$$\mathcal{L}_{\text{obstruct}} = \| f \circ g - g \circ f \|^2$$

This is the core idea behind **Diagrammatic Backpropagation (DB)**:
instead of a single scalar loss, the training signal comes from how
badly the diagram fails to commute. Minimising obstruction loss pushes
the system toward structural consistency — the morphisms learn to
respect the diagrammatic contracts they are embedded in.

FunctorFlow provides first-class support for obstruction losses via
`obstruction_loss` and the `db_square` block builder.

## Setup

``` python
from FunctorFlow import (
    Diagram,
    compile_to_callable,
    diagram_certificate_payload,
    render_lean_certificate,
)
```

## Non-Commuting Squares

The simplest DB pattern is a square with two morphisms `f` and `g`
acting on the same state space. The two paths around the square are
`f∘g` and `g∘f`. If the morphisms don’t commute, the obstruction loss
will be nonzero.

We create a single object `S` (the state space) and two morphisms
`f: S → S` and `g: S → S`, then compose them in both orders.

``` python
D = Diagram("ManualDB")

D.object("S", kind="value")

D.morphism("f", "S", "S")
D.morphism("g", "S", "S")

D.compose("f", "g", name="fg")
D.compose("g", "f", name="gf")
```

    Composition(name='gf', chain=('g', 'f'), source='S', target='S', description='', metadata={})

Now we add an obstruction loss that compares the two composed paths.

``` python
D.obstruction_loss(
    paths=[("fg", "gf")],
    name="commutativity",
    comparator="l2",
    weight=1.0,
)
```

    ObstructionLoss(name='commutativity', paths=(('fg', 'gf'),), comparator='l2', weight=1.0, description='', metadata={})

Bind concrete implementations to the morphisms — simple functions where
`f(x) = 2x` and `g(x) = x + 1`.

``` python
D.bind_morphism("f", lambda x: x * 2)
D.bind_morphism("g", lambda x: x + 1)
```

Compile and run the diagram.

``` python
compiled = compile_to_callable(D)
result = compiled.run({"S": 10.0})
```

Inspect the output values and the obstruction loss. Since
`f∘g(x) = 2(x+1) = 2x+2` while `g∘f(x) = 2x+1`, the paths do not agree —
the loss should be nonzero.

``` python
print("f∘g path:", result.values["fg"])
print("g∘f path:", result.values["gf"])
print("Obstruction losses:", result.losses)
```

    f∘g path: 21.0
    g∘f path: 22.0
    Obstruction losses: {'commutativity': 1.0}

With `x = 10`, `f∘g = 2(10+1) = 22` and `g∘f = 2·10+1 = 21`, giving an
L2 loss of `|22 − 21| = 1.0`.

## The `paths` Parameter

The `paths` argument to `obstruction_loss` is a list of
`(morph1, morph2)` tuples. Each tuple names two operations whose outputs
are compared. In the square above we wrote:

``` python
paths=[("fg", "gf")]
```

This tells FunctorFlow: “compare the result of the composition `fg`
against `gf`.” You can list as many pairs as you like in a single loss.

## Comparators

The `comparator` keyword controls how two path outputs are compared.

### Built-in: L2 (default)

The `'l2'` comparator computes the absolute difference (for scalars) or
squared L2 norm (for arrays).

``` python
D_l2 = Diagram("L2Demo")
D_l2.object("S", kind="value")
D_l2.morphism("f", "S", "S")
D_l2.morphism("g", "S", "S")
D_l2.obstruction_loss(paths=[("f", "g")], name="l2_loss", comparator="l2")
D_l2.bind_morphism("f", lambda x: x * 2)
D_l2.bind_morphism("g", lambda x: x * 3)

compiled_l2 = compile_to_callable(D_l2)
result_l2 = compiled_l2.run({"S": 10.0})
print("L2 loss (|20 − 30|):", result_l2.losses["l2_loss"])
```

    L2 loss (|20 − 30|): 10.0

### Custom Comparators

You can bind an arbitrary comparator function. A comparator takes two
values and returns a scalar loss. Use a custom comparator name in
`obstruction_loss` and bind it with `bind_comparator`.

``` python
D_custom = Diagram("CustomComp")
D_custom.object("S", kind="value")
D_custom.morphism("f", "S", "S")
D_custom.morphism("g", "S", "S")
D_custom.compose("f", "g", name="fg")
D_custom.compose("g", "f", name="gf")

D_custom.obstruction_loss(
    paths=[("fg", "gf")],
    name="cosine_loss",
    comparator="custom_cosine",
)

D_custom.bind_morphism("f", lambda x: x * 2)
D_custom.bind_morphism("g", lambda x: x + 1)
```

Bind an L1 comparator for a simple example:

``` python
D_custom.bind_comparator("custom_cosine", lambda a, b: abs(a - b))

compiled_custom = compile_to_callable(D_custom)
result_custom = compiled_custom.run({"S": 10.0})
print("Custom (L1) obstruction loss:", result_custom.losses["cosine_loss"])
```

    Custom (L1) obstruction loss: 1.0

For array-valued paths a cosine-distance comparator might look like:

``` python
import numpy as np

def cosine_comparator(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    dot = np.sum(a * b)
    norm_a = np.sqrt(np.sum(a ** 2))
    norm_b = np.sqrt(np.sum(b ** 2))
    return 1.0 - dot / (norm_a * norm_b + 1e-8)

D_cos = Diagram("CosineDemo")
D_cos.object("S", kind="value")
D_cos.morphism("f", "S", "S")
D_cos.morphism("g", "S", "S")
D_cos.compose("f", "g", name="fg")
D_cos.compose("g", "f", name="gf")
D_cos.obstruction_loss(
    paths=[("fg", "gf")],
    name="cos_loss",
    comparator="cosine",
)
D_cos.bind_morphism("f", lambda x: np.asarray(x) * 2)
D_cos.bind_morphism("g", lambda x: np.asarray(x) + 1)
D_cos.bind_comparator("cosine", cosine_comparator)

compiled_cos = compile_to_callable(D_cos)
result_cos = compiled_cos.run({"S": np.array([1.0, 2.0, 3.0, 4.0])})
print("f∘g path:", result_cos.values["fg"])
print("g∘f path:", result_cos.values["gf"])
print("Cosine obstruction loss:", result_cos.losses["cos_loss"])
```

    f∘g path: [3. 5. 7. 9.]
    g∘f path: [ 4.  6.  8. 10.]
    Cosine obstruction loss: 0.0011298162538414536

## Multi-Path Losses

A single obstruction loss can monitor multiple path pairs
simultaneously. This is useful when a diagram has several faces that
should commute.

``` python
D_multi = Diagram("MultiPath")
D_multi.object("A", kind="value")

D_multi.morphism("f", "A", "A")
D_multi.morphism("g", "A", "A")
D_multi.morphism("h", "A", "A")

D_multi.obstruction_loss(
    paths=[("f", "g"), ("g", "h")],
    name="multi_loss",
    comparator="l2",
    weight=1.0,
)
```

    ObstructionLoss(name='multi_loss', paths=(('f', 'g'), ('g', 'h')), comparator='l2', weight=1.0, description='', metadata={})

Here the loss is the sum of `|f(x) − g(x)|` and `|g(x) − h(x)|`. All
three morphisms are encouraged to agree — enforcing consistency across
redundant paths.

``` python
D_multi.bind_morphism("f", lambda x: x * 2)
D_multi.bind_morphism("g", lambda x: x * 2.1)
D_multi.bind_morphism("h", lambda x: x * 1.9)

compiled_multi = compile_to_callable(D_multi)
result_multi = compiled_multi.run({"A": 10.0})
print("Values:", result_multi.values)
print("Multi-path losses:", result_multi.losses)
```

    Values: {'A': 10.0, 'f': 20.0, 'g': 21.0, 'h': 19.0}
    Multi-path losses: {'multi_loss': 3.0}

## Training Interpretation

Obstruction loss is not just a diagnostic — it is a **training signal**.
In Diagrammatic Backpropagation:

1.  Each face of the diagram contributes an obstruction loss.
2.  The total loss is the (weighted) sum of all obstruction losses.
3.  Gradient descent on this total loss pushes morphisms toward
    commutativity.

This is structurally richer than a single end-to-end loss because it
enforces **local consistency** at every face of the diagram, not just
global input→output accuracy. The result is more modular, interpretable,
and composable learning.

## Weighted Losses

The `weight` parameter scales the contribution of each obstruction loss
to the total. This lets you prioritise certain commutativity constraints
over others.

``` python
D_w = Diagram("Weighted")
D_w.object("S", kind="value")
D_w.morphism("f", "S", "S")
D_w.morphism("g", "S", "S")
D_w.compose("f", "g", name="fg")
D_w.compose("g", "f", name="gf")

# High weight: strongly enforce this face
D_w.obstruction_loss(
    paths=[("fg", "gf")],
    name="critical_face",
    comparator="l2",
    weight=10.0,
)
```

    ObstructionLoss(name='critical_face', paths=(('fg', 'gf'),), comparator='l2', weight=10.0, description='', metadata={})

``` python
D_w.bind_morphism("f", lambda x: x * 2)
D_w.bind_morphism("g", lambda x: x + 1)

compiled_w = compile_to_callable(D_w)
result_w = compiled_w.run({"S": 10.0})
print("Weighted loss:", result_w.losses)
```

    Weighted loss: {'critical_face': 10.0}

A weight of `10.0` means this face’s contribution to the total loss is
scaled by 10×, making the optimiser prioritise its commutativity above
other, lower-weighted faces.

## Full DB Square

A full DB square has four morphisms forming two paths through a square.
We compare the composed top path `f → p1` against the composed bottom
path `g → p2`.

``` python
D_full = Diagram("FullDBSquare")
D_full.object("S", kind="value")

D_full.morphism("f",  "S", "S")
D_full.morphism("g",  "S", "S")
D_full.morphism("p1", "S", "S")
D_full.morphism("p2", "S", "S")

D_full.compose("f", "p1", name="path_top")
D_full.compose("g", "p2", name="path_bot")

D_full.obstruction_loss(
    paths=[("path_top", "path_bot")],
    name="db_loss",
    comparator="l2",
    weight=1.0,
)
```

    ObstructionLoss(name='db_loss', paths=(('path_top', 'path_bot'),), comparator='l2', weight=1.0, description='', metadata={})

Bind implementations and run:

``` python
D_full.bind_morphism("f",  lambda x: x * 2)
D_full.bind_morphism("g",  lambda x: x + 1)
D_full.bind_morphism("p1", lambda x: x + 3)
D_full.bind_morphism("p2", lambda x: x * 4)

compiled_full = compile_to_callable(D_full)
result_full = compiled_full.run({"S": 5.0})
```

``` python
print("All values:")
for name, val in result_full.values.items():
    print(f"  {name}: {val}")

print("\nAll losses:")
for name, val in result_full.losses.items():
    print(f"  {name}: {val}")
```

    All values:
      S: 5.0
      f: 10.0
      g: 6.0
      p1: 8.0
      p2: 20.0
      path_top: 13.0
      path_bot: 24.0

    All losses:
      db_loss: 11.0

The top path computes `p1(f(5)) = (5·2)+3 = 13` and the bottom path
computes `p2(g(5)) = (5+1)·4 = 24`, giving an L2 loss of
`|13 − 24| = 11.0`.

## Proof Certificate

FunctorFlow can generate a formal proof certificate for any diagram.
This produces a Lean 4 declaration that captures the diagram’s structure
— objects, operations, and their wiring — for downstream verification.

``` python
cert = diagram_certificate_payload(D_full)
print("Diagram name:", cert["diagram_name"])
print("Objects:", cert["objects"])
print("Operations:")
for op in cert["operations"]:
    print(f"  {op['name']} ({op['kind']}): refs={op['refs']}")
```

    Diagram name: FullDBSquare
    Objects: ['S']
    Operations:
      f (OperationKind.morphism): refs=['S', 'S']
      g (OperationKind.morphism): refs=['S', 'S']
      p1 (OperationKind.morphism): refs=['S', 'S']
      p2 (OperationKind.morphism): refs=['S', 'S']
      path_top (OperationKind.composition): refs=['f', 'p1', 'S', 'S']
      path_bot (OperationKind.composition): refs=['g', 'p2', 'S', 'S']

The `render_lean_certificate` function emits Lean 4 source code:

``` python
lean_code = render_lean_certificate(D_full)
print(lean_code)
```

    import FunctorFlowProofs.Compiler

    open FunctorFlowProofs

    namespace FunctorFlowProofs.Generated.FullDBSquare

    def exportedDiagram : DiagramDecl := {
      name := "FullDBSquare"
      objects := ["S"]
      operations := [
        {
          name := "f"
          kind := OperationKind.morphism
          refs := ["S", "S"]
        },
        {
          name := "g"
          kind := OperationKind.morphism
          refs := ["S", "S"]
        },
        {
          name := "p1"
          kind := OperationKind.morphism
          refs := ["S", "S"]
        },
        {
          name := "p2"
          kind := OperationKind.morphism
          refs := ["S", "S"]
        },
        {
          name := "path_top"
          kind := OperationKind.composition
          refs := ["f", "p1", "S", "S"]
        },
        {
          name := "path_bot"
          kind := OperationKind.composition
          refs := ["g", "p2", "S", "S"]
        },
      ]
      ports := [
      ]
    }

    def exportedArtifact : LoweringArtifact := {
      diagram := exportedDiagram
      loweredOps := ["f", "g", "p1", "p2", "path_top", "path_bot"]
    }

    theorem exportedArtifact_checks : exportedArtifact.check = true := rfl

    theorem exportedArtifact_sound : exportedArtifact.Sound :=
      LoweringArtifact.sound_of_check_eq_true exportedArtifact_checks

    end FunctorFlowProofs.Generated.FullDBSquare

## Summary

| Concept | Python API |
|----|----|
| Add obstruction loss | `D.obstruction_loss(paths=[...], name=..., comparator=..., weight=...)` |
| Built-in comparator | `'l2'` (default) |
| Custom comparator | `D.bind_comparator('name', fn)` where `fn(a, b) → float` |
| Multi-path monitoring | `paths=[('f','g'), ('g','h')]` |
| Weighted loss | `weight=10.0` scales the loss contribution |
| Proof certificate | `diagram_certificate_payload(D)`, `render_lean_certificate(D)` |
