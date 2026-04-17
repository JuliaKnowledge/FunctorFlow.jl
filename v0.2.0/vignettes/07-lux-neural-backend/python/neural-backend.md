# Neural Network Backend (Python)


- [Introduction](#introduction)
- [Setup](#setup)
- [NumPy Forward Pass](#numpy-forward-pass)
  - [Building a KET-style diagram](#building-a-ket-style-diagram)
  - [Binding NumPy morphisms](#binding-numpy-morphisms)
  - [Compile and run](#compile-and-run)
  - [Diagram summary](#diagram-summary)
- [PyTorch Neural Backend](#pytorch-neural-backend)
  - [Dense morphisms with `nn.Linear`](#dense-morphisms-with-nnlinear)
  - [Forward pass](#forward-pass)
  - [Inspecting intermediate values](#inspecting-intermediate-values)
  - [Gradient flow](#gradient-flow)
- [Composed Neural Morphisms](#composed-neural-morphisms)
- [Mixed Neural / Symbolic Morphisms](#mixed-neural--symbolic-morphisms)
- [Julia ↔ Python Comparison](#julia--python-comparison)
- [Summary](#summary)

## Introduction

In Julia, FunctorFlow.jl uses **Lux.jl** as its differentiable neural
backend. Diagrams are compiled into Lux models via `compile_to_lux`,
producing standard Lux layers with extractable parameters and automatic
differentiation.

In Python, FunctorFlow uses **PyTorch** via `compile_to_torch`. The
compiled diagram becomes a `torch.nn.Module` whose morphisms can be
arbitrary `nn.Module` layers. If PyTorch is not installed, you can still
bind **NumPy** implementations to morphisms and use
`compile_to_callable` for non-differentiable forward passes.

## Setup

``` python
import numpy as np
from FunctorFlow import Diagram, compile_to_callable

try:
    import torch
    import torch.nn as nn
    from FunctorFlow import compile_to_torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

print("PyTorch available:", HAS_TORCH)
if HAS_TORCH:
    print("PyTorch version:", torch.__version__)
```

    PyTorch available: True
    PyTorch version: 2.11.0

## NumPy Forward Pass

Even without PyTorch, every diagram can be executed by binding plain
Python or NumPy functions and compiling with `compile_to_callable`.

### Building a KET-style diagram

We create a diagram with a linear projection, a softmax activation, and
a second linear layer, then compose them into a pipeline.

``` python
D_np = Diagram("NumpyNet")
D_np.object("X", kind="value")

D_np.morphism("linear1",  "X", "X")
D_np.morphism("activate", "X", "X")
D_np.morphism("linear2",  "X", "X")

D_np.compose("linear1", "activate", "linear2", name="pipeline")
```

    Composition(name='pipeline', chain=('linear1', 'activate', 'linear2'), source='X', target='X', description='', metadata={})

### Binding NumPy morphisms

Weight matrices and biases are ordinary NumPy arrays. Each morphism is a
callable `x → y`.

``` python
np.random.seed(42)

W1 = np.random.randn(4, 4) * 0.1
b1 = np.zeros(4)
W2 = np.random.randn(4, 4) * 0.1
b2 = np.zeros(4)

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

D_np.bind_morphism("linear1",  lambda x: x @ W1.T + b1)
D_np.bind_morphism("activate", softmax)
D_np.bind_morphism("linear2",  lambda x: x @ W2.T + b2)
```

### Compile and run

``` python
compiled_np = compile_to_callable(D_np)

x = np.random.randn(2, 4)
result_np = compiled_np.run({"X": x})

print("Input shape:", x.shape)
print("\nAll intermediate values:")
for name, val in result_np.values.items():
    if isinstance(val, np.ndarray):
        print(f"  {name}: shape={val.shape}")
    else:
        print(f"  {name}: {type(val).__name__}")
```

    Input shape: (2, 4)

    All intermediate values:
      X: shape=(2, 4)
      linear1: shape=(2, 4)
      activate: shape=(2, 4)
      linear2: shape=(2, 4)
      pipeline: shape=(2, 4)

``` python
print("Pipeline output:")
print(result_np.values["pipeline"])
```

    Pipeline output:
     [[-0.07518218 -0.01219451 -0.02582367  0.01688567]
     [-0.09018743 -0.02790556 -0.01818857  0.04273625]]

Each morphism executes independently on the source object, while
`pipeline` chains them: `linear1 → activate → linear2`.

### Diagram summary

``` python
print(D_np.summary())
```

    Diagram(NumpyNet)
      Objects: X
      Operations: linear1, activate, linear2, pipeline
      Losses: <none>
      Ports: <none>

## PyTorch Neural Backend

When PyTorch is available, `compile_to_torch` turns a diagram into a
`torch.nn.Module`. Morphisms can be `nn.Module` layers with learnable
parameters, and gradients flow through the entire diagram.

### Dense morphisms with `nn.Linear`

``` python
if HAS_TORCH:
    torch.manual_seed(42)

    D_torch = Diagram("TorchPipeline")
    D_torch.object("S", kind="value")

    D_torch.morphism("encode",   "S", "S")
    D_torch.morphism("activate", "S", "S")
    D_torch.morphism("decode",   "S", "S")

    D_torch.compose("encode", "activate", "decode", name="pipeline")

    D_torch.bind_morphism("encode",   nn.Linear(4, 4))
    D_torch.bind_morphism("activate", nn.ReLU())
    D_torch.bind_morphism("decode",   nn.Linear(4, 4))

    model = compile_to_torch(D_torch)
    print("Model type:", type(model).__name__)
    print("Learnable parameters:", sum(p.numel() for p in model.parameters()))
else:
    print("Skipping — PyTorch not available")
```

    Model type: TorchCompiledDiagram
    Learnable parameters: 40

### Forward pass

The compiled model accepts a dict mapping object names to tensors and
returns a dict of all computed values.

``` python
if HAS_TORCH:
    x = torch.randn(3, 4)
    out = model({"S": x})

    print("Output keys:", list(out.keys()))
    for name, val in out.items():
        if isinstance(val, torch.Tensor):
            print(f"  {name}: shape={tuple(val.shape)}, requires_grad={val.requires_grad}")
else:
    print("Skipping — PyTorch not available")
```

    Output keys: ['S', 'encode', 'activate', 'decode', 'pipeline']
      S: shape=(3, 4), requires_grad=False
      encode: shape=(3, 4), requires_grad=True
      activate: shape=(3, 4), requires_grad=False
      decode: shape=(3, 4), requires_grad=True
      pipeline: shape=(3, 4), requires_grad=True

Gradients propagate through all neural morphisms, so the pipeline output
carries a `grad_fn` for back-propagation.

### Inspecting intermediate values

Every morphism’s output is available in the result dict — not just the
final composed pipeline. This makes it easy to inspect or visualise
intermediate representations.

``` python
if HAS_TORCH:
    print("Encoded (first row):", out["encode"][0].detach().numpy())
    print("Activated (first row):", out["activate"][0].detach().numpy())
    print("Decoded / pipeline (first row):", out["pipeline"][0].detach().numpy())
else:
    print("Skipping — PyTorch not available")
```

    Encoded (first row): [1.5908327  0.30819827 0.0429416  0.5384189 ]
    Activated (first row): [1.3525478 0.6863219 0.        0.7949687]
    Decoded / pipeline (first row): [-0.13675855 -0.46286857 -0.23176119  0.6414279 ]

### Gradient flow

Since the model is a standard `torch.nn.Module`, you can compute
gradients with the usual PyTorch API.

``` python
if HAS_TORCH:
    x = torch.randn(3, 4)
    out = model({"S": x})
    loss = out["pipeline"].sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"  {name}: grad norm = {param.grad.norm().item():.4f}")
else:
    print("Skipping — PyTorch not available")
```

      lowered_morphisms.encode.weight: grad norm = 1.7252
      lowered_morphisms.encode.bias: grad norm = 2.1916
      lowered_morphisms.decode.weight: grad norm = 4.5244
      lowered_morphisms.decode.bias: grad norm = 6.0000

## Composed Neural Morphisms

FunctorFlow’s `compose` wires morphisms together so the output of one
feeds into the next. This mirrors Julia’s `compose!` call.

``` python
if HAS_TORCH:
    torch.manual_seed(0)

    D_deep = Diagram("DeepNet")
    D_deep.object("S", kind="value")

    D_deep.morphism("layer1", "S", "S")
    D_deep.morphism("relu1",  "S", "S")
    D_deep.morphism("layer2", "S", "S")
    D_deep.morphism("relu2",  "S", "S")
    D_deep.morphism("head",   "S", "S")

    D_deep.compose("layer1", "relu1", "layer2", "relu2", "head", name="forward")

    D_deep.bind_morphism("layer1", nn.Linear(8, 8))
    D_deep.bind_morphism("relu1",  nn.ReLU())
    D_deep.bind_morphism("layer2", nn.Linear(8, 8))
    D_deep.bind_morphism("relu2",  nn.ReLU())
    D_deep.bind_morphism("head",   nn.Linear(8, 2))

    deep_model = compile_to_torch(D_deep)
    x = torch.randn(5, 8)
    out = deep_model({"S": x})

    print("Forward output shape:", tuple(out["forward"].shape))
    print("All intermediate keys:", list(out.keys()))
else:
    print("Skipping — PyTorch not available")
```

    Forward output shape: (5, 2)
    All intermediate keys: ['S', 'layer1', 'relu1', 'layer2', 'relu2', 'head', 'forward']

## Mixed Neural / Symbolic Morphisms

A key strength of the FunctorFlow backend is mixing **neural** morphisms
(learnable `nn.Module` layers) and **symbolic** morphisms (plain Python
functions) in the same diagram. The compiler handles the routing
automatically.

In Julia, `compile_to_lux` treats unbound morphisms as neural layers and
bound morphisms as symbolic; the same pattern applies in Python with
`compile_to_torch`.

``` python
if HAS_TORCH:
    torch.manual_seed(42)

    D_mix = Diagram("MixedModel")
    D_mix.object("S", kind="value")

    # Neural: learned encoder
    D_mix.morphism("encode", "S", "S")

    # Symbolic: deterministic L2 normalisation
    D_mix.morphism("normalize", "S", "S")
    D_mix.bind_morphism(
        "normalize",
        lambda x: x / (torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True)) + 1e-8),
    )

    # Neural: learned decoder
    D_mix.morphism("decode", "S", "S")

    D_mix.compose("encode", "normalize", "decode", name="pipeline")

    D_mix.bind_morphism("encode", nn.Linear(4, 4))
    D_mix.bind_morphism("decode", nn.Linear(4, 4))

    mixed_model = compile_to_torch(D_mix)
    x = torch.randn(3, 4)
    out = mixed_model({"S": x})

    print("Pipeline output shape:", tuple(out["pipeline"].shape))
    print("Normalised (first row, should be unit norm):")
    normed = out["normalize"][0].detach()
    print(f"  values: {normed.numpy()}")
    print(f"  L2 norm: {torch.norm(normed).item():.6f}")
else:
    print("Skipping — PyTorch not available")
```

    Pipeline output shape: (3, 4)
    Normalised (first row, should be unit norm):
      values: [ 0.77576184  0.39364403 -0.18798792  0.45595905]
      L2 norm: 1.000000

The `normalize` morphism uses a plain lambda (no parameters), while
`encode` and `decode` use learnable `nn.Linear` instances. Gradients
flow through the symbolic normalisation via standard PyTorch autograd.

## Julia ↔ Python Comparison

| Concept | Julia (Lux) | Python (PyTorch) |
|----|----|----|
| Compile | `compile_to_lux(D)` → `LuxDiagramModel` | `compile_to_torch(D)` → `nn.Module` |
| Dense layer | `DiagramDenseLayer(in, out)` | `nn.Linear(in, out)` |
| Activation | Bound Julia function | `nn.ReLU()` or lambda |
| Parameters | `Lux.setup(rng, model)` | `model.parameters()` |
| Forward pass | `model(inputs, ps, st)` | `model({"obj": tensor})` |
| Gradient | Zygote / Enzyme AD | `loss.backward()` |
| Fallback | `compile_to_callable` | `compile_to_callable` (NumPy) |

## Summary

- **`compile_to_callable`** works with any Python callable (NumPy, plain
  math, etc.) — no framework dependency.
- **`compile_to_torch`** wraps the diagram as a `torch.nn.Module` with
  learnable parameters and full autograd support.
- Morphisms can be `nn.Module` layers, plain functions, or a mix of
  both.
- `compose` chains morphisms so that intermediate values flow correctly
  through the pipeline.
- All intermediate values are accessible in the output dict, making
  inspection and debugging straightforward.
