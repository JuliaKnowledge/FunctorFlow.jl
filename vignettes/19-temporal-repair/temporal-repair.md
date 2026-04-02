# Temporal Repair and Schrodinger Bridges
Simon Frost

- [Introduction](#introduction)
- [Setup](#setup)
- [Build the bundled temporal repair
  example](#build-the-bundled-temporal-repair-example)
- [Compile the temporal stack](#compile-the-temporal-stack)
- [Execute the placeholder trace](#execute-the-placeholder-trace)
- [Why this matters](#why-this-matters)

## Introduction

`FunctorFlow.jl` now includes the full v1 temporal surface: persistent
states, trajectories, temporal repair objects, and Schrodinger-bridge
summaries. This vignette shows the highest-level Julia-native path
through that stack:

1.  build the bundled temporal repair example;
2.  inspect the repaired trajectory and bridge summary;
3.  compile the example into parity-oriented semantic artifacts; and
4.  execute the lowered placeholder IR to expose the temporal trace.

## Setup

``` julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using FunctorFlow
```

## Build the bundled temporal repair example

`build_temporal_repair_example()` returns a self-contained example with
raw states, repaired states, trajectories, a temporal block, a repair
object, and a bridge object.

``` julia
example = build_temporal_repair_example()
summary = summarize_temporal_repair_example(example)

println("Counts: ", summary["counts"])
println("Bridge summary: ", summary["bridge"])
println("Company summary: ", summary["company_summaries"][1])
```

    Counts: Dict("temporal_repairs" => 1, "repaired_states" => 3, "repaired_trajectories" => 1, "raw_trajectories" => 1, "raw_states" => 3, "temporal_blocks" => 1)
    Bridge summary: Dict{String, Any}("summary_metrics" => Dict("csb_sde_mean" => Dict("mse" => 0.12, "mae" => 0.08), "conditional_flow" => Dict("mse" => 0.09, "mae" => 0.06)), "name" => "AcmeTemporalBridge", "conditioning_scope" => "adjacent_year_endpoint_matching", "endpoint_constraints" => 2, "dataset_label" => "synthetic_temporal_panel", "solver_family" => "conditional_sde_flow_matching", "reference_process" => "brownian_reference", "bridge_method" => "csb_sde_mean", "linked_trajectories" => ["AcmeRawTrajectory", "AcmeRepairedTrajectory"])
    Company summary: Dict{String, Any}("company" => "acme", "years" => [2023, 2024, 2025], "temporal_repair" => Dict{String, Any}("name" => "AcmeTemporalRepair", "component_names" => ["AcmeTemporalRepair__repair_2023", "AcmeTemporalRepair__repair_2024", "AcmeTemporalRepair__repair_2025"], "repair_objective" => "temporal_block_denoising", "repair_map" => "AcmeTemporalRepair__repair_map"), "raw_trajectory" => "AcmeRawTrajectory", "temporal_block" => "AcmeTemporalBlock", "repaired_trajectory" => "AcmeRepairedTrajectory")

The repaired 2024 state inserts a stabilizing action, so the temporal
repair is not just a relabeling; it changes the intermediate state while
preserving the endpoint-compatible trajectory.

``` julia
raw = example[:raw_trajectory]
repaired = example[:repaired_trajectory]
repair = example[:temporal_repair]
bridge = example[:bridge]

println("Raw trajectory years: ", raw.years)
println("Raw state names: ", [state.name for state in raw.states])
println("Repaired state names: ", [state.name for state in repaired.states])
println("Repair objective: ", repair.repair_objective)
println("Repair map components: ", sort([component.name for component in values(repair.repair_map.components)]))
println("Bridge metrics: ", bridge.summary_metrics)
```

    Raw trajectory years: [2023, 2024, 2025]
    Raw state names: [:AcmeRaw2023, :AcmeRaw2024, :AcmeRaw2025]
    Repaired state names: [:AcmeRepaired2023, :AcmeRepaired2024, :AcmeRepaired2025]
    Repair objective: temporal_block_denoising
    Repair map components: [:AcmeTemporalRepair__repair_2023, :AcmeTemporalRepair__repair_2024, :AcmeTemporalRepair__repair_2025]
    Bridge metrics: Dict(:csb_sde_mean => Dict(:mse => 0.12, :mae => 0.08), :conditional_flow => Dict(:mse => 0.09, :mae => 0.06))

## Compile the temporal stack

The semantic compiler treats the temporal repair and the Schrodinger
bridge as first-class compiled artifacts.

``` julia
plan = build_temporal_repair_compilation_plan(example)

compiled_subjects = [
    (artifact.subject_name, artifact.subject_kind)
    for artifact in plan.artifacts
    if artifact.subject_kind in (
        :trajectory_functor,
        :temporal_block,
        :temporal_repair,
        :temporal_schrodinger_bridge,
    )
]

println("Compiled temporal subjects:")
for subject in compiled_subjects
    println("  ", subject)
end
```

    Compiled temporal subjects:
      (:AcmeRawTrajectory, :trajectory_functor)
      (:AcmeRepairedTrajectory, :trajectory_functor)
      (:AcmeTemporalBlock, :temporal_block)
      (:AcmeTemporalRepair, :temporal_repair)
      (:AcmeTemporalBridge, :temporal_schrodinger_bridge)

Lowering the plan to placeholder IR makes the temporal semantics
explicit as operations such as `repair_temporal_block` and
`instantiate_temporal_bridge`.

``` julia
ir = build_temporal_repair_executable_ir(example)

println("IR instructions:")
for instruction in ir.instructions
    if instruction.opcode in (:repair_temporal_block, :instantiate_temporal_bridge)
        println("  ", instruction.name, " => ", instruction.opcode)
    end
end
```

    IR instructions:
      AcmeTemporalRepair__temporal_repair => repair_temporal_block
      AcmeTemporalBridge__temporal_schrodinger_bridge => instantiate_temporal_bridge

## Execute the placeholder trace

The execution trace is symbolic rather than numeric, but it exposes the
exact categorical steps the temporal example lowers to.

``` julia
executed = execute_temporal_repair_example(example)

println("Execution trace:")
for line in executed.trace
    println("  ", line)
end
```

    Execution trace:
      AcmeRaw2024__persistent_state: declare_persistent_state inputs=() outputs=(:AcmeRaw2024,) types=("persistent_state_object",)
      AcmeRaw2025__persistent_state: declare_persistent_state inputs=() outputs=(:AcmeRaw2025,) types=("persistent_state_object",)
      AcmeRaw2023__persistent_state: declare_persistent_state inputs=() outputs=(:AcmeRaw2023,) types=("persistent_state_object",)
      AcmeRepaired2024__persistent_state: declare_persistent_state inputs=() outputs=(:AcmeRepaired2024,) types=("persistent_state_object",)
      AcmeRepaired2025__persistent_state: declare_persistent_state inputs=() outputs=(:AcmeRepaired2025,) types=("persistent_state_object",)
      AcmeRepaired2023__persistent_state: declare_persistent_state inputs=() outputs=(:AcmeRepaired2023,) types=("persistent_state_object",)
      AcmeRawTrajectory__trajectory: declare_trajectory_functor inputs=(:AcmeRaw2023, :AcmeRaw2024, :AcmeRaw2025) outputs=(:AcmeRawTrajectory,) types=("trajectory_functor",)
      AcmeRepairedTrajectory__trajectory: declare_trajectory_functor inputs=(:AcmeRepaired2023, :AcmeRepaired2024, :AcmeRepaired2025) outputs=(:AcmeRepairedTrajectory,) types=("trajectory_functor",)
      AcmeTemporalBlock__temporal_block: declare_temporal_block inputs=() outputs=(:AcmeTemporalBlock,) types=("temporal_block",)
      AcmeTemporalRepair__temporal_repair: repair_temporal_block inputs=(:AcmeRawTrajectory, :AcmeTemporalBlock) outputs=(:AcmeRepairedTrajectory,) types=("temporal_repair",)
      AcmeTemporalBridge__temporal_schrodinger_bridge: instantiate_temporal_bridge inputs=(:AcmeRawTrajectory, :AcmeRepairedTrajectory) outputs=(:AcmeTemporalBridge,) types=("temporal_schrodinger_bridge",)

## Why this matters

The v1 temporal layer is now documented as an actual Julia workflow, not
just a test surface:

- **persistent states** give year-indexed categorical state objects;
- **temporal repair** records how a corrupted trajectory is repaired;
  and
- **Schrodinger bridges** summarize the endpoint-conditioning geometry
  that ties raw and repaired trajectories together.

That makes the Julia port usable for both symbolic parity checks and
future Lux-backed temporal modeling work.
