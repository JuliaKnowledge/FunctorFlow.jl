# ============================================================================
# reducers.jl — Built-in reducers and comparators
# ============================================================================

# ---------------------------------------------------------------------------
# Relation normalization
# ---------------------------------------------------------------------------

"""
    normalize_relation(relation) -> Dict{Any, Vector}

Normalize a relation into a dict mapping target keys to lists of source keys.
Accepts: Dict{target => [source_keys]}, Vector of (source, target) pairs,
or Dict{target => source_key}.
"""
function normalize_relation(relation::AbstractDict)
    result = Dict{Any, Vector{Any}}()
    for (k, v) in relation
        if v isa AbstractVector
            result[k] = collect(v)
        else
            result[k] = [v]
        end
    end
    result
end

function normalize_relation(relation::AbstractVector)
    result = Dict{Any, Vector{Any}}()
    for (src, tgt) in relation
        push!(get!(Vector{Any}, result, tgt), src)
    end
    result
end

"""
    group_values(source_value, relation) -> Dict{target_key, Vector{values}}

Group source values by target key according to the relation.
"""
function group_values(source_value, relation)
    normed = normalize_relation(relation)
    result = Dict{Any, Vector{Any}}()
    for (tgt_key, src_keys) in normed
        vals = Any[]
        for sk in src_keys
            v = _lookup_source(source_value, sk)
            v !== nothing && push!(vals, v)
        end
        if !isempty(vals)
            result[tgt_key] = vals
        end
    end
    result
end

function _lookup_source(source::AbstractDict, key)
    get(source, key, nothing)
end

function _lookup_source(source::AbstractVector, key::Integer)
    checkbounds(Bool, source, key) ? source[key] : nothing
end

function _lookup_source(source, key)
    nothing
end

# ---------------------------------------------------------------------------
# Built-in reducers
# ---------------------------------------------------------------------------

function _reduce_sum(source, relation, metadata)
    grouped = group_values(source, relation)
    Dict(k => sum(vs) for (k, vs) in grouped)
end

function _reduce_mean(source, relation, metadata)
    grouped = group_values(source, relation)
    Dict(k => sum(vs) / length(vs) for (k, vs) in grouped)
end

function _reduce_first_non_null(source, relation, metadata)
    grouped = group_values(source, relation)
    result = Dict{Any, Any}()
    for (k, vs) in grouped
        for v in vs
            if v !== nothing
                result[k] = v
                break
            end
        end
    end
    result
end

function _reduce_concat(source, relation, metadata)
    grouped = group_values(source, relation)
    result = Dict{Any, Any}()
    for (k, vs) in grouped
        if all(v -> v isa AbstractString, vs)
            result[k] = join(vs)
        elseif all(v -> v isa AbstractVector, vs)
            result[k] = reduce(vcat, vs)
        else
            result[k] = vs
        end
    end
    result
end

function _reduce_majority(source, relation, metadata)
    grouped = group_values(source, relation)
    result = Dict{Any, Any}()
    for (k, vs) in grouped
        counts = Dict{Any, Int}()
        for v in vs
            counts[v] = get(counts, v, 0) + 1
        end
        result[k] = first(sort(collect(counts), by=last, rev=true)).first
    end
    result
end

function _reduce_set_union(source, relation, metadata)
    grouped = group_values(source, relation)
    result = Dict{Any, Any}()
    for (k, vs) in grouped
        s = Set{Any}()
        for v in vs
            if v isa Union{AbstractSet, AbstractVector}
                union!(s, v)
            else
                push!(s, v)
            end
        end
        result[k] = s
    end
    result
end

function _reduce_tuple(source, relation, metadata)
    grouped = group_values(source, relation)
    Dict(k => Tuple(vs) for (k, vs) in grouped)
end

"""Registry of built-in reducers."""
const BUILTIN_REDUCERS = Dict{Symbol, Any}(
    :sum => _reduce_sum,
    :mean => _reduce_mean,
    :first_non_null => _reduce_first_non_null,
    :concat => _reduce_concat,
    :majority => _reduce_majority,
    :set_union => _reduce_set_union,
    :tuple => _reduce_tuple,
)

# ---------------------------------------------------------------------------
# Built-in comparators (distance metrics)
# ---------------------------------------------------------------------------

function _flatten_numeric(value)::Vector{Float64}
    result = Float64[]
    _flatten_numeric!(result, value)
    result
end

function _flatten_numeric!(result::Vector{Float64}, value::Nothing)
    push!(result, 0.0)
end

function _flatten_numeric!(result::Vector{Float64}, value::Bool)
    push!(result, value ? 1.0 : 0.0)
end

function _flatten_numeric!(result::Vector{Float64}, value::Real)
    push!(result, Float64(value))
end

function _flatten_numeric!(result::Vector{Float64}, value::AbstractDict)
    for v in values(value)
        _flatten_numeric!(result, v)
    end
end

function _flatten_numeric!(result::Vector{Float64}, value::Union{AbstractVector, Tuple})
    for v in value
        _flatten_numeric!(result, v)
    end
end

function _l2_distance(a, b)
    fa = _flatten_numeric(a)
    fb = _flatten_numeric(b)
    n = max(length(fa), length(fb))
    total = 0.0
    for i in 1:n
        va = i <= length(fa) ? fa[i] : 0.0
        vb = i <= length(fb) ? fb[i] : 0.0
        total += (va - vb)^2
    end
    sqrt(total)
end

function _l1_distance(a, b)
    fa = _flatten_numeric(a)
    fb = _flatten_numeric(b)
    n = max(length(fa), length(fb))
    total = 0.0
    for i in 1:n
        va = i <= length(fa) ? fa[i] : 0.0
        vb = i <= length(fb) ? fb[i] : 0.0
        total += abs(va - vb)
    end
    total
end

function _tokenize_for_overlap(value)::Set{String}
    result = Set{String}()
    _tokenize_for_overlap!(result, value)
    result
end

function _tokenize_for_overlap!(result::Set{String}, value::Nothing)
    result
end

function _tokenize_for_overlap!(result::Set{String}, value::AbstractString)
    for token in split(lowercase(value))
        isempty(token) || push!(result, token)
    end
    result
end

function _tokenize_for_overlap!(result::Set{String}, value::Symbol)
    push!(result, lowercase(String(value)))
    result
end

function _tokenize_for_overlap!(result::Set{String}, value::Real)
    push!(result, string(value))
    result
end

function _tokenize_for_overlap!(result::Set{String}, value::Bool)
    push!(result, value ? "true" : "false")
    result
end

function _tokenize_for_overlap!(result::Set{String}, value::AbstractDict)
    for (k, v) in value
        _tokenize_for_overlap!(result, k)
        _tokenize_for_overlap!(result, v)
    end
    result
end

function _tokenize_for_overlap!(result::Set{String}, value::Union{AbstractVector, Tuple, AbstractSet})
    for item in value
        _tokenize_for_overlap!(result, item)
    end
    result
end

function _tokenize_for_overlap!(result::Set{String}, value)
    push!(result, lowercase(string(value)))
    result
end

function _jaccard_distance(a, b)
    ta = _tokenize_for_overlap(a)
    tb = _tokenize_for_overlap(b)
    isempty(ta) && isempty(tb) && return 0.0
    1.0 - length(intersect(ta, tb)) / length(union(ta, tb))
end

"""Registry of built-in comparators."""
const BUILTIN_COMPARATORS = Dict{Symbol, Any}(
    :l2 => _l2_distance,
    :l1 => _l1_distance,
    :jaccard => _jaccard_distance,
)
