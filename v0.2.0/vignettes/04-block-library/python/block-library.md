# Block Library (Python)


- [Introduction](#introduction)
- [Setup](#setup)
- [The Macro Registry](#the-macro-registry)
- [KET Block](#ket-block)
  - [Inspecting Structure](#inspecting-structure)
  - [Binding and Running](#binding-and-running)
  - [Building with Options](#building-with-options)
- [DB Square](#db-square)
  - [Inspecting Structure](#inspecting-structure-1)
  - [Binding and Running](#binding-and-running-1)
- [GT Neighborhood](#gt-neighborhood)
  - [Inspecting Structure](#inspecting-structure-2)
- [Completion Block](#completion-block)
- [BASKET Workflow](#basket-workflow)
- [ROCKET Repair](#rocket-repair)
- [Democritus Gluing](#democritus-gluing)
- [Structured LM Duality](#structured-lm-duality)
- [BASKET-ROCKET Pipeline](#basket-rocket-pipeline)
- [Full IR Example](#full-ir-example)
- [Summary](#summary)

## Introduction

FunctorFlow provides a library of pre-built block macros — parameterized
diagram constructors that encode common architectural patterns from
categorical AI. Each block returns a `Diagram` with named objects,
operations, and ports. You can inspect their structure, bind concrete
implementations, and compose them into larger pipelines.

This vignette mirrors the Julia block library vignette using the Python
API.

## Setup

``` python
from FunctorFlow import Diagram, build_macro, compile_to_callable, MACRO_LIBRARY
import json
```

## The Macro Registry

All blocks are registered in `MACRO_LIBRARY` and can be built by name
using `build_macro`:

``` python
print("Available macros:", list(MACRO_LIBRARY.keys()))
```

    Available macros: ['ket', 'completion', 'structured_lm_duality', 'db_square', 'gt_neighborhood', 'basket_workflow', 'rocket_repair', 'democritus_gluing', 'basket_rocket_pipeline']

## KET Block

The **KET** (Kan Extension Template) block is the most fundamental
pattern. It performs left-Kan aggregation over an incidence relation —
the universal building block for attention, pooling, and message
passing.

Categorically, the KET computes a left Kan extension $\mathrm{Lan}_J F$
along an incidence functor $J$, producing contextualized values by
aggregating source values over the fibers of $J$.

``` python
D_ket = build_macro('ket')
print(D_ket.summary())
```

    Diagram(KETBlock)
      Objects: Values, Incidence, ContextualizedValues
      Operations: aggregate
      Losses: <none>
      Ports: input, relation, output

### Inspecting Structure

``` python
ir = D_ket.to_ir().as_dict()
print("Objects:", [obj['name'] for obj in ir['objects']])
print("Operations:", [op['name'] for op in ir['operations']])
```

    Objects: ['Values', 'Incidence', 'ContextualizedValues']
    Operations: ['aggregate']

### Binding and Running

The default KET block uses the built-in `sum` reducer. Compile and run
with sample data:

``` python
D_ket2 = build_macro('ket')

compiled = compile_to_callable(D_ket2)
result = compiled.run({
    'Values': {'a': 1.0, 'b': 2.0, 'c': 3.0},
    'Incidence': {'x': ['a', 'b'], 'y': ['b', 'c']}
})
print("KET output:", result.values)
```

    KET output: {'Values': {'a': 1.0, 'b': 2.0, 'c': 3.0}, 'Incidence': {'x': ['a', 'b'], 'y': ['b', 'c']}, 'aggregate': {'x': 3.0, 'y': 5.0}}

### Building with Options

Build a KET with a specific reducer by passing `reducer=` to
`build_macro`:

``` python
D_mean = build_macro('ket', reducer='mean')

compiled_mean = compile_to_callable(D_mean)
result_mean = compiled_mean.run({
    'Values': {'a': 10.0, 'b': 20.0},
    'Incidence': {'x': ['a', 'b']}
})
print("KET with mean:", result_mean.values)
```

    KET with mean: {'Values': {'a': 10.0, 'b': 20.0}, 'Incidence': {'x': ['a', 'b']}, 'aggregate': {'x': 15.0}}

## DB Square

The **DB Square** (Diagrammatic Backpropagation Square) measures the
obstruction to commutativity of two morphisms. It computes paths
$f \circ g$ and $g \circ f$ and reports the distance as a loss.

Categorically, a DB square tests whether a diagram commutes: given
morphisms $f$ and $g$ on a state object, the obstruction loss measures
$\|p_1 - p_2\|$ where $p_1 = f \circ g$ and $p_2 = g \circ f$. The loss
vanishes iff $f$ and $g$ commute.

``` python
D_db = build_macro('db_square')
print(D_db.summary())
```

    Diagram(DBSquare)
      Objects: State
      Operations: f, g, p1, p2
      Losses: obstruction
      Ports: input, left_path, right_path, loss

### Inspecting Structure

The DB square has four morphisms (`f`, `g`, `p1`, `p2`) and an
obstruction loss:

``` python
ir_db = D_db.to_ir().as_dict()
print("Objects:", [obj['name'] for obj in ir_db['objects']])
print("Operations:", [op['name'] for op in ir_db['operations']])
print("Losses:", ir_db.get('losses', []))
```

    Objects: ['State']
    Operations: ['f', 'g', 'p1', 'p2']
    Losses: [{'name': 'obstruction', 'paths': (('p1', 'p2'),), 'comparator': 'l2', 'weight': 1.0, 'description': 'Measure how far the square is from commuting.', 'metadata': {'macro': 'DBSquare'}}]

### Binding and Running

``` python
D_db2 = build_macro('db_square')
D_db2.bind_morphism('f', lambda x: x + 1.0)
D_db2.bind_morphism('g', lambda x: x * 2.0)
D_db2.bind_morphism('p1', lambda x: (x + 1.0) * 2.0)
D_db2.bind_morphism('p2', lambda x: (x * 2.0) + 1.0)
D_db2.bind_comparator('obstruction', lambda a, b: abs(a - b))

compiled_db = compile_to_callable(D_db2)
result_db = compiled_db.run({'State': 5.0})
print("f∘g path (p1):", result_db.values.get('p1'))
print("g∘f path (p2):", result_db.values.get('p2'))
print("Obstruction loss:", result_db.losses.get('obstruction'))
```

    f∘g path (p1): 12.0
    g∘f path (p2): 11.0
    Obstruction loss: 1.0

The loss is zero only when $f$ and $g$ commute. Here,
$(5+1) \times 2 = 12$ and $(5 \times 2) + 1 = 11$, so the obstruction is
$|12 - 11| = 1$.

## GT Neighborhood

The **GT Neighborhood** (Graph Transformer Neighborhood) block first
lifts tokens to edge messages via a morphism, then aggregates them with
a left Kan extension. This is the standard two-step pattern in graph
neural networks and graph transformers.

Categorically, this is a composite: first a morphism
$\mathrm{lift}: \mathrm{Tokens} \to \mathrm{Messages}$, then a left Kan
extension $\mathrm{Lan}_J(\mathrm{Messages})$ along the neighborhood
incidence $J$.

``` python
D_gt = build_macro('gt_neighborhood')
print(D_gt.summary())
```

    Diagram(GTNeighborhoodBlock)
      Objects: Tokens, Messages, NeighborhoodIncidence, UpdatedTokens
      Operations: lift_messages, kan_aggregate
      Losses: <none>
      Ports: input, relation, messages, output

### Inspecting Structure

``` python
ir_gt = D_gt.to_ir().as_dict()
print("Objects:", [obj['name'] for obj in ir_gt['objects']])
print("Operations:", [op['name'] for op in ir_gt['operations']])
```

    Objects: ['Tokens', 'Messages', 'NeighborhoodIncidence', 'UpdatedTokens']
    Operations: ['lift_messages', 'kan_aggregate']

The `lift_messages` morphism transforms tokens to messages, and
`kan_aggregate` performs the Kan extension aggregation.

## Completion Block

The **Completion Block** uses a right Kan extension for universal
completion — filling in partial or missing data from compatible
neighbors.

Categorically, this computes a right Kan extension $\mathrm{Ran}_J F$,
the universal construction dual to the left Kan. Where left Kan
aggregates (colimit-like), right Kan completes (limit-like), selecting
consistent values from the fibers of a compatibility relation.

``` python
D_comp = build_macro('completion')
print(D_comp.summary())
```

    Diagram(CompletionBlock)
      Objects: PartialState, CompatibilityRelation, CompletedState
      Operations: complete
      Losses: <none>
      Ports: input, relation, output

``` python
ir_comp = D_comp.to_ir().as_dict()
print("Objects:", [obj['name'] for obj in ir_comp['objects']])
print("Operations:", [op['name'] for op in ir_comp['operations']])
```

    Objects: ['PartialState', 'CompatibilityRelation', 'CompletedState']
    Operations: ['complete']

## BASKET Workflow

The **BASKET** (Bounded Aggregation via Sheaf-theoretic Kan Extension
Templates) workflow block composes local plan fragments into a composed
plan using left Kan with a `:concat` reducer.

Categorically, BASKET performs a left Kan extension of plan fragments
along an observation-context incidence, with a monoidal concatenation
reducer — assembling a global plan from local contributions.

``` python
D_basket = build_macro('basket_workflow')
print(D_basket.summary())
```

    Diagram(BASKETWorkflowBlock)
      Objects: PlanFragments, ObservationContexts, PlanState
      Operations: draft_plan
      Losses: <none>
      Ports: fragments, context, output

``` python
ir_basket = D_basket.to_ir().as_dict()
print("Objects:", [obj['name'] for obj in ir_basket['objects']])
print("Operations:", [op['name'] for op in ir_basket['operations']])
```

    Objects: ['PlanFragments', 'ObservationContexts', 'PlanState']
    Operations: ['draft_plan']

## ROCKET Repair

The **ROCKET** (Robust Obstruction-Corrected Kan Extension Transform)
repair block uses a right Kan extension to repair candidates using edit
neighborhoods.

Categorically, ROCKET applies a right Kan extension to candidate
fragments along an edit-neighborhood relation, using a `:first_non_null`
reducer — selecting the best available repair from local neighborhoods.

``` python
D_rocket = build_macro('rocket_repair')
print(D_rocket.summary())
```

    Diagram(ROCKETRepairBlock)
      Objects: CandidateFragments, EditNeighborhood, RepairedPlan
      Operations: repair
      Losses: <none>
      Ports: candidates, relation, output

``` python
ir_rocket = D_rocket.to_ir().as_dict()
print("Objects:", [obj['name'] for obj in ir_rocket['objects']])
print("Operations:", [op['name'] for op in ir_rocket['operations']])
```

    Objects: ['CandidateFragments', 'EditNeighborhood', 'RepairedPlan']
    Operations: ['repair']

## Democritus Gluing

The **Democritus Gluing** block implements sheaf-theoretic
local-to-global assembly. It uses a right Kan extension to glue local
causal claims over overlap regions into a global relational state.

Categorically, this is a sheaf-theoretic construction: local sections
(claims on patches) are glued into a global section on the base space,
mediated by overlap regions that enforce consistency.

``` python
D_demo = build_macro('democritus_gluing')
print(D_demo.summary())
```

    Diagram(DemocritusGluingBlock)
      Objects: LocalClaims, OverlapRegions, GlobalManifold
      Operations: glue
      Losses: <none>
      Ports: locals, relation, output

``` python
ir_demo = D_demo.to_ir().as_dict()
print("Objects:", [obj['name'] for obj in ir_demo['objects']])
print("Operations:", [op['name'] for op in ir_demo['operations']])
```

    Objects: ['LocalClaims', 'OverlapRegions', 'GlobalManifold']
    Operations: ['glue']

## Structured LM Duality

The **Structured LM Duality** block runs parallel left-Kan (prediction)
and right-Kan (completion/repair) branches from a shared input. This
captures the predict-then-repair duality central to structured language
modeling.

Categorically, this encodes the duality between left and right Kan
extensions as parallel functors from a shared domain — the prediction
branch computes a colimit (aggregation), while the repair branch
computes a limit (completion).

``` python
D_lm = build_macro('structured_lm_duality')
print(D_lm.summary())
```

    Diagram(StructuredLMDuality)
      Objects: HiddenStates, CausalRelation, ContextualizedStates, NoisyBlock, DenoiseCondition, CompletedBlock
      Operations: predict__aggregate_context, repair__complete_block
      Losses: <none>
      Ports: predict__input, predict__relation, predict__output, repair__input, repair__relation, repair__output, hidden, relation, context, noisy_block, condition, completed

``` python
ir_lm = D_lm.to_ir().as_dict()
print("Objects:", [obj['name'] for obj in ir_lm['objects']])
print("Operations:", [op['name'] for op in ir_lm['operations']])
```

    Objects: ['HiddenStates', 'CausalRelation', 'ContextualizedStates', 'NoisyBlock', 'DenoiseCondition', 'CompletedBlock']
    Operations: ['predict__aggregate_context', 'repair__complete_block']

## BASKET-ROCKET Pipeline

The **BASKET-ROCKET Pipeline** composes two stages: a BASKET draft phase
(left Kan with `:concat`) followed by a ROCKET repair phase (right Kan
with `:first_non_null`). This is a complete draft-then-repair workflow
built via diagram inclusion.

Categorically, this is a sequential composite of two Kan extensions: the
left Kan assembles a draft from local fragments, then the right Kan
repairs defects by completing from edit neighborhoods. The composition
is mediated by the port system and namespacing.

``` python
D_br = build_macro('basket_rocket_pipeline')
print(D_br.summary())
```

    Diagram(BASKETROCKETPipeline)
      Objects: PlanFragments, ObservationContexts, draft__PlanState, EditNeighborhood, RepairedPlan
      Operations: draft__draft_plan, repair__repair
      Losses: <none>
      Ports: draft__fragments, draft__context, draft__output, repair__candidates, repair__relation, repair__output, fragments, context, repair_relation, draft_output, output

``` python
ir_br = D_br.to_ir().as_dict()
print("Objects:", [obj['name'] for obj in ir_br['objects']])
print("Operations:", [op['name'] for op in ir_br['operations']])
```

    Objects: ['PlanFragments', 'ObservationContexts', 'draft__PlanState', 'EditNeighborhood', 'RepairedPlan']
    Operations: ['draft__draft_plan', 'repair__repair']

Note the namespaced operations: `draft__draft_plan` and `repair__repair`
reflect the included sub-diagrams.

## Full IR Example

Here is the full intermediate representation of the KET block as JSON:

``` python
D_ir_example = build_macro('ket')
ir_full = D_ir_example.to_ir().as_dict()
print(json.dumps(ir_full, indent=2, default=str))
```

    {
      "name": "KETBlock",
      "objects": [
        {
          "name": "Values",
          "kind": "messages",
          "shape": null,
          "description": "",
          "metadata": {}
        },
        {
          "name": "Incidence",
          "kind": "relation",
          "shape": null,
          "description": "",
          "metadata": {}
        },
        {
          "name": "ContextualizedValues",
          "kind": "contextualized_messages",
          "shape": null,
          "description": "",
          "metadata": {}
        }
      ],
      "operations": [
        {
          "name": "aggregate",
          "direction": "left",
          "source": "Values",
          "along": "Incidence",
          "target": "ContextualizedValues",
          "reducer": "sum",
          "description": "Universal aggregation over an incidence structure.",
          "metadata": {
            "macro": "KETBlock"
          },
          "kind": "kanextension"
        }
      ],
      "losses": [],
      "ports": [
        {
          "name": "input",
          "ref": "Values",
          "kind": "object",
          "port_type": "messages",
          "direction": "input",
          "description": "",
          "metadata": {}
        },
        {
          "name": "relation",
          "ref": "Incidence",
          "kind": "object",
          "port_type": "relation",
          "direction": "input",
          "description": "",
          "metadata": {}
        },
        {
          "name": "output",
          "ref": "aggregate",
          "kind": "operation",
          "port_type": "contextualized_messages",
          "direction": "output",
          "description": "",
          "metadata": {}
        }
      ]
    }

## Summary

| Block | Pattern | Kan Type | Key Idea |
|----|----|----|----|
| KET | Aggregation | Left Kan | Universal attention/pooling |
| DB Square | Commutativity test | Loss | Obstruction to diagram commutativity |
| GT Neighborhood | Lift + Aggregate | Left Kan | Graph transformer message passing |
| Completion | Gap filling | Right Kan | Universal completion from neighbors |
| BASKET Workflow | Plan assembly | Left Kan | Sheaf-theoretic plan composition |
| ROCKET Repair | Defect correction | Right Kan | Obstruction-corrected repair |
| Democritus Gluing | Local→Global | Right Kan | Sheaf gluing of local claims |
| Structured LM Duality | Predict + Repair | Both | Left/right Kan duality |
| BASKET-ROCKET | Draft + Repair | Both | Composed pipeline via inclusion |
