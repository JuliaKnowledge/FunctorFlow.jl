# ============================================================================
# acset_adapter.jl — Diagram ⇌ CategoricalDiagramACSet
# ============================================================================
#
# Provides methods for `FunctorFlow.to_acset` and `FunctorFlow.from_acset`
# that target the shared schema `SchCategoricalDiagram` from
# CategoricalDiagramSchema.jl.

# ---- helpers ---------------------------------------------------------------

"""
Parse an FFObject.shape::String into a Tuple of Ints on a best-effort basis.
Falls back to `DEFAULT_SHAPE = ()` when the string is empty or unparseable.
"""
function _parse_shape(s::Union{Nothing, AbstractString})
    s === nothing && return DEFAULT_SHAPE
    str = strip(String(s))
    isempty(str) && return DEFAULT_SHAPE
    inner = strip(str, ['(', ')', '[', ']', ' '])
    isempty(inner) && return DEFAULT_SHAPE
    try
        parts = split(inner, ',')
        dims = Int[]
        for p in parts
            q = strip(p)
            isempty(q) && continue
            push!(dims, parse(Int, q))
        end
        return Tuple(dims)
    catch
        return DEFAULT_SHAPE
    end
end

"Format a shape Tuple back into an FFObject.shape::String representation."
function _format_shape(shape::Tuple)
    isempty(shape) && return nothing
    return "(" * join(shape, ", ") * ")"
end

_edge_metadata(op) = copy(op.metadata)

# ---- to_acset --------------------------------------------------------------

function to_acset(D::Diagram)
    acs = make_diagram()
    node_idx = Dict{Symbol, Int}()
    edge_idx = Dict{Symbol, Int}()

    # Nodes
    for (name, obj) in D.objects
        shape = _parse_shape(obj.shape)
        dtype = get(obj.metadata, :dtype, DEFAULT_DTYPE)
        md = copy(obj.metadata)
        if obj.shape !== nothing
            md[:ff_string_shape] = obj.shape
        end
        md[:description] = obj.description
        nid = add_part!(acs, :Node;
            node_name = name,
            node_kind = obj.kind,
            node_shape = shape,
            node_dtype = dtype,
            node_metadata = md,
        )
        node_idx[name] = nid
    end

    # Operations
    for (name, op) in D.operations
        if op isa Morphism
            sid = get(node_idx, op.source, nothing)
            tid = get(node_idx, op.target, nothing)
            (sid === nothing || tid === nothing) && continue
            eid = add_part!(acs, :Edge;
                src = sid, tgt = tid,
                edge_name = name,
                edge_kind = :morphism,
                edge_metadata = _edge_metadata(op),
            )
            edge_idx[name] = eid
        elseif op isa Composition
            sid = get(node_idx, op.source, nothing)
            tid = get(node_idx, op.target, nothing)
            (sid === nothing || tid === nothing) && continue
            md = merge(_edge_metadata(op), Dict{Symbol, Any}(:chain => op.chain))
            eid = add_part!(acs, :Edge;
                src = sid, tgt = tid,
                edge_name = name,
                edge_kind = :composition,
                edge_metadata = md,
            )
            edge_idx[name] = eid
        elseif op isa KanExtension
            sid = get(node_idx, op.source, nothing)
            aid = get(node_idx, op.along, nothing)
            (sid === nothing || aid === nothing) && continue
            # Synthesise a target node if absent
            tgt_name = op.target === nothing ? Symbol(name, :_target) : op.target
            if !haskey(node_idx, tgt_name)
                auto_flag = op.target === nothing
                tmd = Dict{Symbol, Any}(:auto_kan_target => auto_flag)
                tid = add_part!(acs, :Node;
                    node_name = tgt_name,
                    node_kind = :kan_target,
                    node_shape = DEFAULT_SHAPE,
                    node_dtype = DEFAULT_DTYPE,
                    node_metadata = tmd,
                )
                node_idx[tgt_name] = tid
            end
            add_part!(acs, :Kan;
                kan_src = sid,
                kan_along = aid,
                kan_tgt = node_idx[tgt_name],
                kan_name = name,
                kan_dir = op.direction == LEFT ? :left : :right,
                kan_reducer = op.reducer,
                kan_metadata = copy(op.metadata),
            )
        end
    end

    # Obstruction losses
    for (name, loss) in D.losses
        lid = add_part!(acs, :ObsLoss;
            obs_name = name,
            obs_comparator = loss.comparator,
            obs_weight = loss.weight,
            obs_metadata = copy(loss.metadata),
        )
        for (left_name, right_name) in loss.paths
            l_eid = get(edge_idx, left_name, nothing)
            r_eid = get(edge_idx, right_name, nothing)
            (l_eid === nothing || r_eid === nothing) &&
                error("Obstruction loss :$name references unknown edge(s): " *
                      "($left_name, $right_name)")
            add_part!(acs, :ObsPath;
                obs = lid,
                path_left = l_eid,
                path_right = r_eid,
            )
        end
    end

    acs
end

# ---- from_acset ------------------------------------------------------------

function from_acset(acs; name::Union{Symbol, AbstractString} = :Imported)
    D = Diagram(Symbol(name))

    node_names = subpart(acs, :node_name)
    node_kinds = subpart(acs, :node_kind)
    node_shapes = subpart(acs, :node_shape)
    node_metas = subpart(acs, :node_metadata)

    # Track which nodes are auto-synthesised Kan targets so we can skip
    # adding them as user-visible objects (they will be re-synthesised
    # by the appropriate add_*_kan! call).
    auto_targets = Set{Symbol}()

    for i in 1:nparts(acs, :Node)
        nname = node_names[i]
        meta = node_metas[i]
        if meta isa AbstractDict && get(meta, :auto_kan_target, false) === true
            push!(auto_targets, nname)
            continue
        end
        shape_str = _format_shape(node_shapes[i])
        # Prefer a previously-recorded ff_string_shape for exact round-trip
        if meta isa AbstractDict && haskey(meta, :ff_string_shape)
            ff_s = meta[:ff_string_shape]
            if ff_s isa AbstractString || ff_s === nothing
                shape_str = ff_s
            end
        end
        desc = meta isa AbstractDict ? String(get(meta, :description, "")) : ""
        # Rebuild clean metadata (strip internal round-trip keys)
        clean_meta = Dict{Symbol, Any}()
        if meta isa AbstractDict
            for (k, v) in meta
                k in (:ff_string_shape, :description, :auto_kan_target) && continue
                clean_meta[Symbol(k)] = v
            end
        end
        add_object!(D, nname;
            kind = node_kinds[i],
            shape = shape_str,
            description = desc,
            metadata = clean_meta,
        )
    end

    # Edges
    edge_names = subpart(acs, :edge_name)
    edge_kinds = subpart(acs, :edge_kind)
    edge_metas = subpart(acs, :edge_metadata)
    edge_srcs = subpart(acs, :src)
    edge_tgts = subpart(acs, :tgt)

    for i in 1:nparts(acs, :Edge)
        ename = edge_names[i]
        ekind = edge_kinds[i]
        src_name = node_names[edge_srcs[i]]
        tgt_name = node_names[edge_tgts[i]]
        emeta = edge_metas[i]
        clean_meta = Dict{Symbol, Any}()
        chain = nothing
        if emeta isa AbstractDict
            for (k, v) in emeta
                if k == :chain
                    chain = v
                else
                    clean_meta[Symbol(k)] = v
                end
            end
        end
        if ekind == :composition && chain !== nothing &&
           chain isa AbstractVector && all(c -> haskey(D.operations, Symbol(c)), chain)
            try
                compose!(D, Symbol.(chain)...; name = ename,
                         metadata = clean_meta)
                continue
            catch
                # fall through to morphism fallback
            end
        end
        add_morphism!(D, ename, src_name, tgt_name; metadata = clean_meta)
    end

    # Kans
    kan_names = subpart(acs, :kan_name)
    kan_srcs = subpart(acs, :kan_src)
    kan_alongs = subpart(acs, :kan_along)
    kan_tgts = subpart(acs, :kan_tgt)
    kan_dirs = subpart(acs, :kan_dir)
    kan_reducers = subpart(acs, :kan_reducer)
    kan_metas = subpart(acs, :kan_metadata)

    for i in 1:nparts(acs, :Kan)
        kname = kan_names[i]
        src_name = node_names[kan_srcs[i]]
        along_name = node_names[kan_alongs[i]]
        tgt_name = node_names[kan_tgts[i]]
        kmeta = kan_metas[i] isa AbstractDict ? copy(kan_metas[i]) : Dict{Symbol, Any}()
        # If the target was auto-synthesised, pass target=nothing
        target = tgt_name in auto_targets ? nothing : tgt_name
        if kan_dirs[i] == :left
            add_left_kan!(D, kname;
                source = src_name, along = along_name,
                target = target, reducer = kan_reducers[i],
                metadata = kmeta,
            )
        else
            add_right_kan!(D, kname;
                source = src_name, along = along_name,
                target = target, reducer = kan_reducers[i],
                metadata = kmeta,
            )
        end
    end

    # Obstruction losses
    obs_names = subpart(acs, :obs_name)
    obs_comparators = subpart(acs, :obs_comparator)
    obs_weights = subpart(acs, :obs_weight)
    obs_metas = subpart(acs, :obs_metadata)

    for lid in 1:nparts(acs, :ObsLoss)
        pids = incident(acs, lid, :obs)
        paths = Tuple{Symbol, Symbol}[]
        for p in pids
            l_eid = subpart(acs, p, :path_left)
            r_eid = subpart(acs, p, :path_right)
            push!(paths, (edge_names[l_eid], edge_names[r_eid]))
        end
        lmeta = obs_metas[lid] isa AbstractDict ? copy(obs_metas[lid]) : Dict{Symbol, Any}()
        add_obstruction_loss!(D, obs_names[lid];
            paths = paths,
            comparator = obs_comparators[lid],
            weight = obs_weights[lid],
            metadata = lmeta,
        )
    end

    D
end
