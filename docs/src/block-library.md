# Block Library

FunctorFlow ships with a library of reusable diagram builders.

## Foundational blocks

These are the basic categorical patterns used throughout the package:

- `ket_block` for left-Kan aggregation
- `completion_block` for right-Kan completion
- `db_square` for commutativity and obstruction losses
- `gt_neighborhood_block` for graph-transformer style neighborhood aggregation

## Planning and repair

FunctorFlow includes block patterns inspired by categorical planning:

- `basket_workflow_block`
- `rocket_repair_block`
- `basket_rocket_pipeline`
- `democritus_assembly_pipeline`
- `topocoend_block`
- `horn_fill_block`
- `bisimulation_quotient_block`

These builders support symbolic examples directly and can also be compiled to Lux-backed models where appropriate.

## Tutorial libraries

To expose curated subsets of the block registry, use tutorial libraries:

```julia
using FunctorFlow

lib = get_tutorial_library(:planning)
diagram = build_tutorial_macro(lib, :basket_rocket_pipeline)
```

Key exported libraries include:

- `FOUNDATIONS_TUTORIAL_LIBRARY`
- `PLANNING_TUTORIAL_LIBRARY`
- `UNIFIED_TUTORIAL_LIBRARY`

## Lux-backed helpers

For neural execution, FunctorFlow provides:

- `compile_to_lux`
- `KETAttentionLayer`
- `DiagramDenseLayer`
- `DiagramChainLayer`
- `build_ket_lux_model`
- `build_db_lux_model`
- `build_gt_lux_model`
- `build_basket_rocket_lux_model`

These helper constructors are especially useful for moving from symbolic diagram design to trainable models.
