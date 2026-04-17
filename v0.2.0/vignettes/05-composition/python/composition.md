# Diagram Composition (Python)


- [Introduction](#introduction)
- [Setup](#setup)
- [Ports](#ports)
- [Including Sub-Diagrams](#including-sub-diagrams)
- [Namespacing](#namespacing)
- [Composing with Ports](#composing-with-ports)
- [Two-Stage Pipeline](#two-stage-pipeline)
- [Adapters](#adapters)
  - [Registering Adapters](#registering-adapters)
  - [Using Adapter Libraries](#using-adapter-libraries)
  - [Coercion](#coercion)
- [Full IR of a Composed Diagram](#full-ir-of-a-composed-diagram)
- [Summary](#summary)

## Introduction

FunctorFlow diagrams can be composed hierarchically. A parent diagram
can include child sub-diagrams using `D.include()`, which embeds all
objects and operations under a namespace. Ports provide stable typed
interfaces that survive composition, and adapters handle principled type
coercion when port types don’t match. This enables modular, reusable
architectural patterns.

This vignette mirrors the Julia composition vignette using the Python
API.

## Setup

``` python
from FunctorFlow import Diagram, build_macro, compile_to_callable
import json
```

## Ports

Ports expose semantic interfaces on a diagram. They declare which
objects are externally accessible, along with their direction and type.
Think of ports as the typed “sockets” of a diagram component.

``` python
D = Diagram('Encoder')
D.object('Tokens', kind='messages')
D.object('Neighbors', kind='relation')
D.object('Output', kind='contextualized_messages')
D.left_kan(source='Tokens', along='Neighbors', reducer='sum', name='aggregate')

# Expose ports
D.expose_port('input', 'Tokens', direction='input')
D.expose_port('relation', 'Neighbors', direction='input')
D.expose_port('output', 'Output', direction='output')

ir = D.to_ir().as_dict()
print("Ports:", list(p['name'] for p in ir['ports']))
```

    Ports: ['input', 'relation', 'output']

You can also inspect the port specification:

``` python
print("Port spec:", ir['ports'])
```

    Port spec: [{'name': 'input', 'ref': 'Tokens', 'kind': 'object', 'port_type': 'messages', 'direction': 'input', 'description': '', 'metadata': {}}, {'name': 'relation', 'ref': 'Neighbors', 'kind': 'object', 'port_type': 'relation', 'direction': 'input', 'description': '', 'metadata': {}}, {'name': 'output', 'ref': 'Output', 'kind': 'object', 'port_type': 'contextualized_messages', 'direction': 'output', 'description': '', 'metadata': {}}]

Pre-built blocks from the block library already have ports defined:

``` python
D_ket = build_macro('ket')
ir_ket = D_ket.to_ir().as_dict()
print("KET block ports:", list(p['name'] for p in ir_ket['ports']))
```

    KET block ports: ['input', 'relation', 'output']

## Including Sub-Diagrams

Use `D.include()` to embed one diagram inside another. All objects and
operations from the child diagram are copied into the parent under a
namespace prefix.

``` python
# Create a parent diagram
parent = Diagram('Pipeline')
parent.object('RawInput', kind='input')

# Create a child encoder diagram
encoder = Diagram('Encoder')
encoder.object('Values', kind='messages')
encoder.object('Incidence', kind='relation')
encoder.left_kan(source='Values', along='Incidence', reducer='sum', name='aggregate')

# Show parent before inclusion
print("=== Parent BEFORE inclusion ===")
print(parent.summary())
```

    === Parent BEFORE inclusion ===
    Diagram(Pipeline)
      Objects: RawInput
      Operations: <none>
      Losses: <none>
      Ports: <none>

``` python
# Include the encoder in the parent under namespace 'enc'
parent.include(encoder, namespace='enc')

print("=== Parent AFTER inclusion ===")
print(parent.summary())
```

    === Parent AFTER inclusion ===
    Diagram(Pipeline)
      Objects: RawInput, enc__Values, enc__Incidence
      Operations: enc__aggregate
      Losses: <none>
      Ports: <none>

``` python
ir_parent = parent.to_ir().as_dict()
print("Parent objects:", [obj['name'] for obj in ir_parent['objects']])
print("Parent operations:", [op['name'] for op in ir_parent['operations']])
```

    Parent objects: ['RawInput', 'enc__Values', 'enc__Incidence']
    Parent operations: ['enc__aggregate']

All child elements are prefixed with the namespace: `enc__Values`,
`enc__Incidence`, `enc__aggregate`.

## Namespacing

When a sub-diagram is included under a namespace, all its elements are
prefixed with `namespace__` (double underscore). This prevents name
collisions when the same block type is included multiple times.

``` python
parent_multi = Diagram('MultiEncoder')

encoder1 = Diagram('Enc1')
encoder1.object('Values', kind='messages')
encoder1.object('Incidence', kind='relation')
encoder1.left_kan(source='Values', along='Incidence', reducer='sum', name='aggregate')

encoder2 = Diagram('Enc2')
encoder2.object('Values', kind='messages')
encoder2.object('Incidence', kind='relation')
encoder2.left_kan(source='Values', along='Incidence', reducer='mean', name='aggregate')

parent_multi.include(encoder1, namespace='layer1')
parent_multi.include(encoder2, namespace='layer2')

ir_multi = parent_multi.to_ir().as_dict()
print("All objects:")
for obj in sorted(ir_multi['objects'], key=lambda o: o['name']):
    print(f"  {obj['name']}")

print("\nAll operations:")
for op in sorted(ir_multi['operations'], key=lambda o: o['name']):
    print(f"  {op['name']}")
```

    All objects:
      layer1__Incidence
      layer1__Values
      layer2__Incidence
      layer2__Values

    All operations:
      layer1__aggregate
      layer2__aggregate

Each layer gets its own prefixed copies of `Values`, `Incidence`, and
`aggregate`, so they operate independently.

## Composing with Ports

Ports make composition principled. A child diagram declares its
interface via ports, and the parent can connect those interfaces:

``` python
# Child with well-defined ports
child = Diagram('Aggregator')
child.object('Input', kind='messages')
child.object('Relation', kind='relation')
child.object('Output', kind='contextualized_messages')
child.left_kan(source='Input', along='Relation', reducer='sum', name='agg')
child.expose_port('data_in', 'Input', direction='input')
child.expose_port('rel_in', 'Relation', direction='input')
child.expose_port('data_out', 'Output', direction='output')

print("Child ports:")
for p in child.to_ir().as_dict()['ports']:
    print(f"  {p['name']}: direction={p['direction']}, ref={p.get('ref', 'N/A')}")
```

    Child ports:
      data_in: direction=input, ref=Input
      rel_in: direction=input, ref=Relation
      data_out: direction=output, ref=Output

``` python
# Parent includes the child
pipeline = Diagram('PortPipeline')
pipeline.object('GlobalInput', kind='messages')
pipeline.object('GlobalRelation', kind='relation')
pipeline.include(child, namespace='step1')

ir_pipe = pipeline.to_ir().as_dict()
print("Pipeline objects:", [obj['name'] for obj in ir_pipe['objects']])
print("Pipeline ports:", [p['name'] for p in ir_pipe['ports']])
```

    Pipeline objects: ['GlobalInput', 'GlobalRelation', 'step1__Input', 'step1__Relation', 'step1__Output']
    Pipeline ports: ['step1__data_in', 'step1__rel_in', 'step1__data_out']

The child’s ports survive inclusion and are accessible under the
namespace.

## Two-Stage Pipeline

Let’s compose two blocks into a predict-then-repair pipeline — a KET
block for prediction (left Kan) and a completion block for repair (right
Kan):

``` python
predictor = Diagram('Predictor')
predictor.object('Values', kind='messages')
predictor.object('Incidence', kind='relation')
predictor.object('Predicted', kind='contextualized_messages')
predictor.left_kan(source='Values', along='Incidence', reducer='sum', name='predict')
predictor.expose_port('values_in', 'Values', direction='input')
predictor.expose_port('incidence_in', 'Incidence', direction='input')
predictor.expose_port('predicted_out', 'Predicted', direction='output')

repairer = Diagram('Repairer')
repairer.object('Partial', kind='partial_state')
repairer.object('Compatibility', kind='relation')
repairer.object('Completed', kind='completed_state')
repairer.right_kan(source='Partial', along='Compatibility', name='repair')
repairer.expose_port('partial_in', 'Partial', direction='input')
repairer.expose_port('compat_in', 'Compatibility', direction='input')
repairer.expose_port('completed_out', 'Completed', direction='output')

pipeline = Diagram('PredictRepairPipeline')
pipeline.object('InputValues', kind='messages')
pipeline.object('PredictRelation', kind='relation')
pipeline.object('RepairRelation', kind='relation')

pipeline.include(predictor, namespace='predict')
pipeline.include(repairer, namespace='repair')

print(pipeline.summary())
```

    Diagram(PredictRepairPipeline)
      Objects: InputValues, PredictRelation, RepairRelation, predict__Values, predict__Incidence, predict__Predicted, repair__Partial, repair__Compatibility, repair__Completed
      Operations: predict__predict, repair__repair
      Losses: <none>
      Ports: predict__values_in, predict__incidence_in, predict__predicted_out, repair__partial_in, repair__compat_in, repair__completed_out

``` python
ir_pipeline = pipeline.to_ir().as_dict()
print("Pipeline objects:", [obj['name'] for obj in ir_pipeline['objects']])
print("\nPipeline operations:", [op['name'] for op in ir_pipeline['operations']])
print("\nPipeline ports:", [p['name'] for p in ir_pipeline['ports']])
```

    Pipeline objects: ['InputValues', 'PredictRelation', 'RepairRelation', 'predict__Values', 'predict__Incidence', 'predict__Predicted', 'repair__Partial', 'repair__Compatibility', 'repair__Completed']

    Pipeline operations: ['predict__predict', 'repair__repair']

    Pipeline ports: ['predict__values_in', 'predict__incidence_in', 'predict__predicted_out', 'repair__partial_in', 'repair__compat_in', 'repair__completed_out']

## Adapters

When composing diagrams, port types may not align. For example, one
block’s output type might be `contextualized_messages` while the next
block expects `plan_candidates`. Adapters provide principled type
bridges.

### Registering Adapters

``` python
D_adapt = Diagram('AdapterDemo')
D_adapt.object('Input', kind='messages')
D_adapt.object('Output', kind='plan_candidates')
D_adapt.morphism('transform', 'Input', 'Output')
D_adapt.bind_morphism('transform', lambda x: x)

# Register an adapter for type coercion
D_adapt.register_adapter(
    'msg_to_candidates',
    source_type='messages',
    target_type='plan_candidates',
    implementation=lambda x: x
)

print(D_adapt.summary())
```

    Diagram(AdapterDemo)
      Objects: Input, Output
      Operations: transform
      Losses: <none>
      Ports: <none>

### Using Adapter Libraries

FunctorFlow provides adapter libraries for common type coercions.
Install an entire library into a diagram:

``` python
from FunctorFlow import STANDARD_ADAPTER_LIBRARY

D_lib = Diagram('WithLibrary')
D_lib.object('X', kind='contextualized_messages')
D_lib.object('Y', kind='plan_candidates')

D_lib.use_adapter_library(STANDARD_ADAPTER_LIBRARY)
print(D_lib.summary())
```

    Diagram(WithLibrary)
      Objects: X, Y
      Operations: <none>
      Losses: <none>
      Ports: <none>

### Coercion

Once adapters are registered, use `coerce()` to insert a coercion
morphism that adapts an object’s type:

``` python
D_coerce = Diagram('CoerceDemo')
D_coerce.object('Input', kind='contextualized_messages')

# Register the adapter
D_coerce.register_adapter(
    'ctx_to_candidates',
    source_type='contextualized_messages',
    target_type='plan_candidates',
    implementation=lambda x: x
)

# Insert a coercion morphism
coercion_name = D_coerce.coerce('Input', to_type='plan_candidates')
print("Generated coercion morphism:", coercion_name)

ir_coerce = D_coerce.to_ir().as_dict()
print("Operations after coerce:", [op['name'] for op in ir_coerce['operations']])
```

    Generated coercion morphism: adapt_0
    Operations after coerce: ['adapt_0']

## Full IR of a Composed Diagram

Here is the complete intermediate representation of the two-stage
pipeline as JSON:

``` python
ir_full = pipeline.to_ir().as_dict()
print(json.dumps(ir_full, indent=2, default=str))
```

    {
      "name": "PredictRepairPipeline",
      "objects": [
        {
          "name": "InputValues",
          "kind": "messages",
          "shape": null,
          "description": "",
          "metadata": {}
        },
        {
          "name": "PredictRelation",
          "kind": "relation",
          "shape": null,
          "description": "",
          "metadata": {}
        },
        {
          "name": "RepairRelation",
          "kind": "relation",
          "shape": null,
          "description": "",
          "metadata": {}
        },
        {
          "name": "predict__Values",
          "kind": "messages",
          "shape": null,
          "description": "",
          "metadata": {
            "namespace": "predict",
            "included_from": "Predictor"
          }
        },
        {
          "name": "predict__Incidence",
          "kind": "relation",
          "shape": null,
          "description": "",
          "metadata": {
            "namespace": "predict",
            "included_from": "Predictor"
          }
        },
        {
          "name": "predict__Predicted",
          "kind": "contextualized_messages",
          "shape": null,
          "description": "",
          "metadata": {
            "namespace": "predict",
            "included_from": "Predictor"
          }
        },
        {
          "name": "repair__Partial",
          "kind": "partial_state",
          "shape": null,
          "description": "",
          "metadata": {
            "namespace": "repair",
            "included_from": "Repairer"
          }
        },
        {
          "name": "repair__Compatibility",
          "kind": "relation",
          "shape": null,
          "description": "",
          "metadata": {
            "namespace": "repair",
            "included_from": "Repairer"
          }
        },
        {
          "name": "repair__Completed",
          "kind": "completed_state",
          "shape": null,
          "description": "",
          "metadata": {
            "namespace": "repair",
            "included_from": "Repairer"
          }
        }
      ],
      "operations": [
        {
          "name": "predict__predict",
          "direction": "left",
          "source": "predict__Values",
          "along": "predict__Incidence",
          "target": null,
          "reducer": "sum",
          "description": "",
          "metadata": {
            "namespace": "predict",
            "included_from": "Predictor"
          },
          "kind": "kanextension"
        },
        {
          "name": "repair__repair",
          "direction": "right",
          "source": "repair__Partial",
          "along": "repair__Compatibility",
          "target": null,
          "reducer": "first_non_null",
          "description": "",
          "metadata": {
            "namespace": "repair",
            "included_from": "Repairer"
          },
          "kind": "kanextension"
        }
      ],
      "losses": [],
      "ports": [
        {
          "name": "predict__values_in",
          "ref": "predict__Values",
          "kind": "object",
          "port_type": "messages",
          "direction": "input",
          "description": "",
          "metadata": {
            "namespace": "predict",
            "included_from": "Predictor"
          }
        },
        {
          "name": "predict__incidence_in",
          "ref": "predict__Incidence",
          "kind": "object",
          "port_type": "relation",
          "direction": "input",
          "description": "",
          "metadata": {
            "namespace": "predict",
            "included_from": "Predictor"
          }
        },
        {
          "name": "predict__predicted_out",
          "ref": "predict__Predicted",
          "kind": "object",
          "port_type": "contextualized_messages",
          "direction": "output",
          "description": "",
          "metadata": {
            "namespace": "predict",
            "included_from": "Predictor"
          }
        },
        {
          "name": "repair__partial_in",
          "ref": "repair__Partial",
          "kind": "object",
          "port_type": "partial_state",
          "direction": "input",
          "description": "",
          "metadata": {
            "namespace": "repair",
            "included_from": "Repairer"
          }
        },
        {
          "name": "repair__compat_in",
          "ref": "repair__Compatibility",
          "kind": "object",
          "port_type": "relation",
          "direction": "input",
          "description": "",
          "metadata": {
            "namespace": "repair",
            "included_from": "Repairer"
          }
        },
        {
          "name": "repair__completed_out",
          "ref": "repair__Completed",
          "kind": "object",
          "port_type": "completed_state",
          "direction": "output",
          "description": "",
          "metadata": {
            "namespace": "repair",
            "included_from": "Repairer"
          }
        }
      ]
    }

## Summary

| Concept | API | Purpose |
|----|----|----|
| Ports | `D.expose_port(name, ref, direction=)` | Typed interface sockets |
| Include | `D.include(child, namespace=)` | Embed sub-diagram with namespace |
| Namespacing | `namespace__element` | Prevent name collisions |
| Adapters | `D.register_adapter(name, source_type=, target_type=, implementation=)` | Type bridges |
| Adapter Library | `D.use_adapter_library(lib)` | Bulk adapter installation |
| Coercion | `D.coerce(obj, to_type=)` | Insert coercion morphism |

The composition system makes FunctorFlow diagrams modular and reusable.
Ports define stable interfaces, namespacing prevents collisions, and
adapters handle type mismatches — all grounded in the categorical
semantics of functorial composition.
