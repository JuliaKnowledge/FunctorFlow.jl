# ============================================================================
# compiler.jl — Backend-neutral compilation and execution
# ============================================================================

"""
    ExecutionResult(values, losses)

Result of executing a compiled diagram. `values` contains all computed values
(inputs and operations), `losses` contains scalar loss values.
"""
struct ExecutionResult
    values::Dict{Symbol, Any}
    losses::Dict{Symbol, Float64}
end

"""
    CompiledDiagram(diagram; morphisms=nothing, reducers=nothing, comparators=nothing)

A compiled diagram ready for execution. Merges built-in registries with
diagram-bound and user-supplied implementations.
"""
struct CompiledDiagram
    diagram::Diagram
    morphisms::Dict{Symbol, Any}
    reducers::Dict{Symbol, Any}
    comparators::Dict{Symbol, Any}
end

function CompiledDiagram(D::Diagram;
                         morphisms::Union{Nothing, Dict}=nothing,
                         reducers::Union{Nothing, Dict}=nothing,
                         comparators::Union{Nothing, Dict}=nothing)
    m = merge(D.implementations,
              morphisms === nothing ? Dict{Symbol, Any}() : Dict{Symbol, Any}(Symbol(k) => v for (k, v) in morphisms))
    r = merge(BUILTIN_REDUCERS, D.reducers,
              reducers === nothing ? Dict{Symbol, Any}() : Dict{Symbol, Any}(Symbol(k) => v for (k, v) in reducers))
    c = merge(BUILTIN_COMPARATORS, D.comparators,
              comparators === nothing ? Dict{Symbol, Any}() : Dict{Symbol, Any}(Symbol(k) => v for (k, v) in comparators))
    CompiledDiagram(D, m, r, c)
end

"""
    compile_to_callable(D; morphisms=nothing, reducers=nothing, comparators=nothing)

Compile a diagram to a callable executor. This is the backend-neutral
execution target.
"""
function compile_to_callable(D::Diagram; kwargs...)
    CompiledDiagram(D; kwargs...)
end

# ---------------------------------------------------------------------------
# Execution primitives
# ---------------------------------------------------------------------------

function _execute_morphism(compiled::CompiledDiagram, m::Morphism, env::Dict{Symbol, Any})
    haskey(env, m.source) || error("Missing source value :$(m.source) for morphism :$(m.name)")
    fn = get(compiled.morphisms, m.name, nothing)
    if fn === nothing && m.implementation_key !== nothing
        fn = get(compiled.morphisms, m.implementation_key, nothing)
    end
    fn === nothing && error("No implementation bound for morphism :$(m.name)")
    fn(env[m.source])
end

function _execute_composition(compiled::CompiledDiagram, comp::Composition, env::Dict{Symbol, Any})
    haskey(env, comp.source) || error("Missing source value :$(comp.source) for composition :$(comp.name)")
    current = env[comp.source]
    for morph_name in comp.chain
        fn = get(compiled.morphisms, morph_name, nothing)
        fn === nothing && error("No implementation bound for morphism :$(morph_name) in composition :$(comp.name)")
        current = fn(current)
    end
    current
end

function _execute_kan(compiled::CompiledDiagram, kan::KanExtension, env::Dict{Symbol, Any})
    haskey(env, kan.source) || error("Missing source value :$(kan.source) for Kan extension :$(kan.name)")
    haskey(env, kan.along) || error("Missing relation value :$(kan.along) for Kan extension :$(kan.name)")
    reducer = get(compiled.reducers, kan.reducer, nothing)
    reducer === nothing && error("No reducer :$(kan.reducer) bound for Kan extension :$(kan.name)")
    metadata = Dict{String, Any}("direction" => string(kan.direction))
    merge!(metadata, Dict{String, Any}(string(k) => v for (k, v) in kan.metadata))
    reducer(env[kan.source], env[kan.along], metadata)
end

function _execute_loss(compiled::CompiledDiagram, loss::ObstructionLoss, env::Dict{Symbol, Any})
    comparator = get(compiled.comparators, loss.comparator, nothing)
    comparator === nothing && error("No comparator :$(loss.comparator) bound for loss :$(loss.name)")
    total = 0.0
    for (left_name, right_name) in loss.paths
        haskey(env, left_name) || error("Missing value :$(left_name) for loss :$(loss.name)")
        haskey(env, right_name) || error("Missing value :$(right_name) for loss :$(loss.name)")
        total += Float64(comparator(env[left_name], env[right_name]))
    end
    loss.weight * total
end

# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

"""
    run(compiled, inputs; morphisms=nothing, reducers=nothing, comparators=nothing)

Execute a compiled diagram with the given inputs. Returns an `ExecutionResult`
with all computed values and losses.

Operations are executed in insertion order. Each operation's result is stored
in the environment under its name, available to subsequent operations.
"""
function run(compiled::CompiledDiagram, inputs::AbstractDict;
             morphisms::Union{Nothing, Dict}=nothing,
             reducers::Union{Nothing, Dict}=nothing,
             comparators::Union{Nothing, Dict}=nothing)
    # Build environment from inputs
    env = Dict{Symbol, Any}(Symbol(k) => v for (k, v) in inputs)

    # Merge runtime overrides
    morph_reg = morphisms === nothing ? compiled.morphisms :
        merge(compiled.morphisms, Dict{Symbol, Any}(Symbol(k) => v for (k, v) in morphisms))
    red_reg = reducers === nothing ? compiled.reducers :
        merge(compiled.reducers, Dict{Symbol, Any}(Symbol(k) => v for (k, v) in reducers))
    comp_reg = comparators === nothing ? compiled.comparators :
        merge(compiled.comparators, Dict{Symbol, Any}(Symbol(k) => v for (k, v) in comparators))

    temp_compiled = CompiledDiagram(compiled.diagram, morph_reg, red_reg, comp_reg)

    # Execute operations in order
    for op in values(compiled.diagram.operations)
        if op isa Morphism
            val = _execute_morphism(temp_compiled, op, env)
            env[op.name] = val
            # Also populate target object (enables downstream morphisms to read it)
            # Skip if target == source to avoid overwriting endomorphism inputs
            if op.target != op.source
                env[op.target] = val
            end
        elseif op isa Composition
            env[op.name] = _execute_composition(temp_compiled, op, env)
        elseif op isa KanExtension
            result = _execute_kan(temp_compiled, op, env)
            env[op.name] = result
            # Also populate target object if specified (enables downstream morphisms)
            if op.target !== nothing && op.target != op.name
                env[op.target] = result
            end
        end
    end

    # Compute losses
    loss_values = Dict{Symbol, Float64}()
    for loss in values(compiled.diagram.losses)
        loss_values[loss.name] = _execute_loss(temp_compiled, loss, env)
    end

    ExecutionResult(env, loss_values)
end

# Convenience: run directly on a Diagram
function run(D::Diagram, inputs::AbstractDict; kwargs...)
    compiled = compile_to_callable(D)
    run(compiled, inputs; kwargs...)
end
