# API Reference

## Core authoring and execution

```@docs
FunctorFlow.Diagram
FunctorFlow.add_object!
FunctorFlow.add_morphism!
FunctorFlow.add_left_kan!
FunctorFlow.add_right_kan!
FunctorFlow.add_obstruction_loss!
FunctorFlow.compile_to_callable
FunctorFlow.run
FunctorFlow.to_ir
```

## DSL and composition

```@docs
FunctorFlow.@functorflow
FunctorFlow.@diagram
FunctorFlow.include!
FunctorFlow.object_ref
FunctorFlow.operation_ref
FunctorFlow.port_spec
```

## Block builders

```@docs
FunctorFlow.ket_block
FunctorFlow.db_square
FunctorFlow.gt_neighborhood_block
FunctorFlow.completion_block
FunctorFlow.basket_workflow_block
FunctorFlow.rocket_repair_block
FunctorFlow.basket_rocket_pipeline
FunctorFlow.democritus_assembly_pipeline
FunctorFlow.topocoend_block
FunctorFlow.horn_fill_block
FunctorFlow.bisimulation_quotient_block
```

## Lux backend

```@docs
FunctorFlow.compile_to_lux
FunctorFlow.KETAttentionLayer
FunctorFlow.DiagramDenseLayer
FunctorFlow.DiagramChainLayer
FunctorFlow.LuxDiagramModel
FunctorFlow.build_ket_lux_model
FunctorFlow.build_db_lux_model
FunctorFlow.build_gt_lux_model
FunctorFlow.build_basket_rocket_lux_model
FunctorFlow.predict_detach_source
```

## Categorical extensions

```@docs
FunctorFlow.pullback
FunctorFlow.pushout
FunctorFlow.product
FunctorFlow.coproduct
FunctorFlow.equalizer
FunctorFlow.coequalizer
FunctorFlow.build_causal_diagram
FunctorFlow.interventional_expectation
FunctorFlow.check_coherence
FunctorFlow.classify_subobject
```
