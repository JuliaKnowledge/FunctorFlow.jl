# ============================================================================
# dsl.jl — Julia macro-based DSL for FunctorFlow diagrams
# ============================================================================

# ---------------------------------------------------------------------------
# @functorflow — primary macro DSL
# ---------------------------------------------------------------------------

"""
    @functorflow name body

Construct a FunctorFlow diagram using mathematical notation.

# Syntax
- `Name::kind` — declare an object with a semantic kind
- `name = Source → Target` — declare a morphism
- `name = Σ(source; along=rel, reducer=:sum)` — left Kan extension
- `name = Δ(source; along=rel)` — right Kan extension
- `name = compose(f, g)` — named composition
- `obstruction(name; paths=[(a,b)], ...)` — obstruction loss
- `port(name, ref; direction=:input, type=:kind)` — expose port

# Example
```julia
D = @functorflow MyKET begin
    Tokens::messages
    Nbrs::relation
    Ctx::contextualized_messages
    embed = Tokens → Ctx
    aggregate = Σ(:Tokens; along=:Nbrs, reducer=:sum)
end
```
"""
macro functorflow(name, body)
    _functorflow_impl(name, body)
end

function _functorflow_impl(name, body)
    stmts = Any[]
    dsym = gensym(:diagram)
    push!(stmts, :($dsym = $Diagram($(QuoteNode(_to_symbol(name))))))

    if body.head == :block
        for expr in body.args
            expr isa LineNumberNode && continue
            s = _parse_ff_statement(dsym, expr)
            s !== nothing && push!(stmts, s)
        end
    end

    push!(stmts, dsym)
    Expr(:block, stmts...)
end

function _to_symbol(x::Symbol)
    x
end
function _to_symbol(x::QuoteNode)
    x.value
end
function _to_symbol(x::String)
    Symbol(x)
end
function _to_symbol(x::Expr)
    x.head == :quote ? x.args[1] : Symbol(x)
end

# ---------------------------------------------------------------------------
# Statement parsers for @functorflow
# ---------------------------------------------------------------------------

function _parse_ff_statement(dsym, expr::Expr)
    # Type annotation: Name::kind → object declaration
    if expr.head == :(::)
        return _parse_ff_object(dsym, expr)
    end

    # Assignment: name = ... → morphism, Kan, or composition
    if expr.head == :(=) && length(expr.args) == 2
        return _parse_ff_assignment(dsym, expr)
    end

    # Function call: obstruction(...), port(...)
    if expr.head == :call
        fname = expr.args[1]
        if fname == :obstruction
            return _parse_ff_obstruction(dsym, expr)
        elseif fname == :port
            return _parse_ff_port(dsym, expr)
        end
    end

    # Old-style @macrocall syntax (backward compat within the block)
    if expr.head == :macrocall
        return _parse_legacy_statement(dsym, expr)
    end

    # Pass through other expressions (let users put arbitrary code)
    return expr
end

function _parse_ff_statement(dsym, expr)
    nothing
end

"""Parse `Name::kind` as an object declaration."""
function _parse_ff_object(dsym, expr)
    name = expr.args[1]
    kind = expr.args[2]
    :(add_object!($dsym, $(QuoteNode(_to_symbol(name)));
                  kind=$(QuoteNode(_to_symbol(kind)))))
end

"""Parse `name = rhs` as morphism, Kan extension, or composition."""
function _parse_ff_assignment(dsym, expr)
    name = expr.args[1]
    rhs = expr.args[2]

    if rhs isa Expr
        # name = Source → Target  (morphism)
        if rhs.head == :call && rhs.args[1] == :→ && length(rhs.args) == 3
            source = rhs.args[2]
            target = rhs.args[3]
            return :(add_morphism!($dsym,
                        $(QuoteNode(_to_symbol(name))),
                        $(QuoteNode(_to_symbol(source))),
                        $(QuoteNode(_to_symbol(target)))))
        end

        # name = Σ(...) or name = Δ(...)
        if rhs.head == :call && rhs.args[1] in (:Σ, :Δ, :left_kan, :right_kan)
            return _parse_ff_kan_call(dsym, name, rhs)
        end

        # name = compose(f, g, ...) or name = f ⋅ g
        if rhs.head == :call && rhs.args[1] == :compose
            chain = [QuoteNode(_to_symbol(a)) for a in rhs.args[2:end] if !(a isa Expr && a.head == :kw)]
            return :(compose!($dsym, $(chain...); name=$(QuoteNode(_to_symbol(name)))))
        end
    end

    # Fallback: pass through as regular assignment
    return expr
end

"""Parse a Σ(...) or Δ(...) call inside an assignment."""
function _parse_ff_kan_call(dsym, name, call_expr)
    is_left = call_expr.args[1] in (:Σ, :left_kan)

    # Extract positional and keyword args
    # Note: Julia puts ; kwargs in a :parameters Expr as args[2]
    source = nothing
    kwargs = Dict{Symbol, Any}()
    for arg in call_expr.args[2:end]
        if arg isa Expr && arg.head == :parameters
            # Keyword arguments after semicolon
            for kw in arg.args
                if kw isa Expr && kw.head == :kw
                    kwargs[kw.args[1]] = kw.args[2]
                end
            end
        elseif arg isa Expr && arg.head == :kw
            kwargs[arg.args[1]] = arg.args[2]
        elseif arg isa Expr && arg.head == :(=)
            kwargs[arg.args[1]] = arg.args[2]
        elseif source === nothing
            source = arg
        end
    end

    source !== nothing || error("Σ/Δ requires a source argument")
    haskey(kwargs, :along) || error("Σ/Δ requires along=...")

    along = kwargs[:along]
    target = get(kwargs, :target, nothing)
    reducer = get(kwargs, :reducer, is_left ? :(:sum) : :(:first_non_null))

    fn = is_left ? :add_left_kan! : :add_right_kan!
    target_expr = target !== nothing ? (target isa QuoteNode ? target : QuoteNode(_to_symbol(target))) : :nothing

    # Handle source — could be a QuoteNode (:X) or bare symbol (X)
    src_expr = source isa QuoteNode ? source : QuoteNode(_to_symbol(source))
    along_expr = along isa QuoteNode ? along : QuoteNode(_to_symbol(along))

    :($fn($dsym, $(QuoteNode(_to_symbol(name)));
          source=$src_expr,
          along=$along_expr,
          target=$target_expr,
          reducer=$reducer))
end

"""Parse `obstruction(name; paths=[(a,b)], comparator=:l2, weight=1.0)`."""
function _parse_ff_obstruction(dsym, expr)
    args = expr.args[2:end]
    positional = Any[]
    kwargs = Dict{Symbol, Any}()
    for arg in args
        if arg isa Expr && (arg.head == :kw || arg.head == :(=))
            kwargs[arg.args[1]] = arg.args[2]
        elseif !(arg isa LineNumberNode)
            push!(positional, arg)
        end
    end
    isempty(positional) && error("obstruction() requires a name")
    name = positional[1]
    haskey(kwargs, :paths) || error("obstruction() requires paths=...")
    paths = kwargs[:paths]
    comparator = get(kwargs, :comparator, :(:l2))
    weight = get(kwargs, :weight, 1.0)
    :(add_obstruction_loss!($dsym, $(QuoteNode(_to_symbol(name)));
                            paths=$paths, comparator=$comparator, weight=$weight))
end

"""Parse `port(name, ref; direction=:input, type=:kind)`."""
function _parse_ff_port(dsym, expr)
    args = expr.args[2:end]
    positional = Any[]
    kwargs = Dict{Symbol, Any}()
    for arg in args
        if arg isa Expr && (arg.head == :kw || arg.head == :(=))
            kwargs[arg.args[1]] = arg.args[2]
        elseif !(arg isa LineNumberNode)
            push!(positional, arg)
        end
    end
    length(positional) >= 2 || error("port() requires name and ref")
    name, ref = positional[1], positional[2]
    direction_val = get(kwargs, :direction, :(:internal))
    port_type = get(kwargs, :type, :nothing)
    quote
        _dir = let d = $direction_val
            d == :input ? $INPUT : d == :output ? $OUTPUT : $INTERNAL
        end
        expose_port!($dsym, $(QuoteNode(_to_symbol(name))),
                     $(QuoteNode(_to_symbol(ref)));
                     direction=_dir,
                     port_type=$port_type === nothing ? nothing : $port_type)
    end
end

# ---------------------------------------------------------------------------
# Legacy @diagram macro (preserved internally for block builders)
# ---------------------------------------------------------------------------

"""
    @diagram name body

Legacy macro — use `@functorflow` instead. Preserved for backward
compatibility with block builders.
"""
macro diagram(name, body)
    _diagram_impl(name, body)
end

function _diagram_impl(name, body)
    stmts = Any[]
    dsym = gensym(:diagram)
    push!(stmts, :($dsym = $Diagram($(QuoteNode(_to_symbol(name))))))

    if body.head == :block
        for expr in body.args
            expr isa LineNumberNode && continue
            s = _parse_legacy_statement(dsym, expr)
            s !== nothing && push!(stmts, s)
        end
    end

    push!(stmts, dsym)
    Expr(:block, stmts...)
end

function _parse_legacy_statement(dsym, expr::Expr)
    if expr.head == :macrocall
        macro_name = expr.args[1]
        macro_str = string(macro_name)
        if endswith(macro_str, "object")
            return _parse_legacy_object(dsym, expr)
        elseif endswith(macro_str, "morphism")
            return _parse_legacy_morphism(dsym, expr)
        elseif endswith(macro_str, "compose")
            return _parse_legacy_compose(dsym, expr)
        elseif endswith(macro_str, "left_kan")
            return _parse_legacy_kan(dsym, expr, :LEFT)
        elseif endswith(macro_str, "right_kan")
            return _parse_legacy_kan(dsym, expr, :RIGHT)
        elseif endswith(macro_str, "obstruction_loss")
            return _parse_legacy_obstruction(dsym, expr)
        elseif endswith(macro_str, "port")
            return _parse_legacy_port(dsym, expr)
        end
    end
    return expr
end

function _parse_legacy_statement(dsym, expr)
    nothing
end

function _extract_kwargs(args)
    positional = Any[]
    kwargs = Dict{Symbol, Any}()
    for arg in args
        arg isa LineNumberNode && continue
        if arg isa Expr && arg.head == :(=) && arg.args[1] isa Symbol
            kwargs[arg.args[1]] = arg.args[2]
        else
            push!(positional, arg)
        end
    end
    (positional, kwargs)
end

function _parse_legacy_object(dsym, expr)
    args = filter(a -> !(a isa LineNumberNode), expr.args[2:end])
    positional, kwargs = _extract_kwargs(args)
    isempty(positional) && error("@object requires a name")
    name = positional[1]
    kind = get(kwargs, :kind, :(:object))
    shape = get(kwargs, :shape, :nothing)
    desc = get(kwargs, :description, "")
    :(add_object!($dsym, $(QuoteNode(_to_symbol(name)));
                  kind=$kind, shape=$shape, description=$desc))
end

function _parse_legacy_morphism(dsym, expr)
    args = filter(a -> !(a isa LineNumberNode), expr.args[2:end])
    positional, kwargs = _extract_kwargs(args)
    length(positional) >= 3 || error("@morphism requires name source target")
    name, source, target = positional[1], positional[2], positional[3]
    impl = get(kwargs, :implementation, :nothing)
    :(add_morphism!($dsym, $(QuoteNode(_to_symbol(name))),
                    $(QuoteNode(_to_symbol(source))),
                    $(QuoteNode(_to_symbol(target)));
                    implementation=$impl))
end

function _parse_legacy_compose(dsym, expr)
    args = filter(a -> !(a isa LineNumberNode), expr.args[2:end])
    positional, kwargs = _extract_kwargs(args)
    haskey(kwargs, :name) || error("@compose requires name=...")
    name = kwargs[:name]
    chain = [QuoteNode(_to_symbol(p)) for p in positional]
    :(compose!($dsym, $(chain...); name=$name))
end

function _parse_legacy_kan(dsym, expr, direction)
    args = filter(a -> !(a isa LineNumberNode), expr.args[2:end])
    positional, kwargs = _extract_kwargs(args)
    isempty(positional) && error("@left_kan/@right_kan requires a name")
    name = positional[1]
    haskey(kwargs, :source) || error("Kan extension requires source=")
    haskey(kwargs, :along) || error("Kan extension requires along=")
    source = kwargs[:source]
    along = kwargs[:along]
    has_target = haskey(kwargs, :target)
    target = get(kwargs, :target, nothing)
    reducer = get(kwargs, :reducer, direction == :LEFT ? :(:sum) : :(:first_non_null))
    fn = direction == :LEFT ? :add_left_kan! : :add_right_kan!
    target_expr = has_target ? QuoteNode(_to_symbol(target)) : :nothing
    :($fn($dsym, $(QuoteNode(_to_symbol(name)));
          source=$(QuoteNode(_to_symbol(source))),
          along=$(QuoteNode(_to_symbol(along))),
          target=$target_expr,
          reducer=$reducer))
end

function _parse_legacy_obstruction(dsym, expr)
    args = filter(a -> !(a isa LineNumberNode), expr.args[2:end])
    positional, kwargs = _extract_kwargs(args)
    isempty(positional) && error("@obstruction_loss requires a name")
    name = positional[1]
    haskey(kwargs, :paths) || error("@obstruction_loss requires paths=")
    paths = kwargs[:paths]
    comparator = get(kwargs, :comparator, :(:l2))
    weight = get(kwargs, :weight, 1.0)
    :(add_obstruction_loss!($dsym, $(QuoteNode(_to_symbol(name)));
                            paths=$paths, comparator=$comparator, weight=$weight))
end

function _parse_legacy_port(dsym, expr)
    args = filter(a -> !(a isa LineNumberNode), expr.args[2:end])
    positional, kwargs = _extract_kwargs(args)
    length(positional) >= 2 || error("@port requires name ref")
    name, ref = positional[1], positional[2]
    direction_val = get(kwargs, :direction, :(:internal))
    port_type = get(kwargs, :type, :nothing)
    quote
        _dir = let d = $direction_val
            d == :input ? $INPUT : d == :output ? $OUTPUT : $INTERNAL
        end
        expose_port!($dsym, $(QuoteNode(_to_symbol(name))),
                     $(QuoteNode(_to_symbol(ref)));
                     direction=_dir,
                     port_type=$port_type === nothing ? nothing : $port_type)
    end
end
