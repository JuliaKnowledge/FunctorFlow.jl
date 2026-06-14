module FunctorFlowMakieExt

# ============================================================================
# FunctorFlowMakieExt — Makie visualization of FunctorFlow diagrams.
#
# Provides methods for `FunctorFlow.plot_diagram` / `plot_diagram!`. The
# layout is a self-contained layered (Sugiyama-style) DAG placement computed
# in pure Julia — no Graphs.jl / GraphMakie dependency — so it works with any
# Makie backend (CairoMakie, GLMakie, WGLMakie).
#
# Node kinds:
#   • objects            → circles
#   • morphisms          → squares          (source → op → target)
#   • compositions       → diamonds         (source → op → target, chain label)
#   • Kan extensions     → up/down triangle (Σ left / Δ right; source,along → op → target)
#   • obstruction losses → x-cross          (linked to each path endpoint)
# ============================================================================

using FunctorFlow
using FunctorFlow: Diagram, FFObject, Morphism, Composition, KanExtension,
                   ObstructionLoss, KanDirection, LEFT
import Makie

# ----------------------------------------------------------------------------
# Graph extraction
# ----------------------------------------------------------------------------

struct _Node
    name::Symbol
    kind::Symbol          # :object | :morphism | :composition | :kan_left | :kan_right | :loss
    label::String
end

# Directed edges between node names; `style` distinguishes relation inputs.
struct _Edge
    from::Symbol
    to::Symbol
    style::Symbol         # :flow | :relation | :loss
end

function _build_graph(D::Diagram)
    nodes = FunctorFlowMakieExt._Node[]
    seen = Set{Symbol}()
    edges = FunctorFlowMakieExt._Edge[]

    add_node!(name, kind, label) = begin
        if !(name in seen)
            push!(nodes, _Node(name, kind, label))
            push!(seen, name)
        end
    end

    # Objects first so they get the earliest (left-most) layers.
    for (name, obj) in D.objects
        add_node!(name, :object, string(name))
    end

    for (name, op) in D.operations
        if op isa Morphism
            add_node!(op.source, :object, string(op.source))
            add_node!(op.target, :object, string(op.target))
            add_node!(name, :morphism, string(name))
            push!(edges, _Edge(op.source, name, :flow))
            push!(edges, _Edge(name, op.target, :flow))
        elseif op isa Composition
            add_node!(op.source, :object, string(op.source))
            add_node!(op.target, :object, string(op.target))
            chain = join(string.(op.chain), "∘")
            add_node!(name, :composition, "$(name)\n[$(chain)]")
            push!(edges, _Edge(op.source, name, :flow))
            push!(edges, _Edge(name, op.target, :flow))
        elseif op isa KanExtension
            add_node!(op.source, :object, string(op.source))
            add_node!(op.along, :object, string(op.along))
            kind = op.direction == LEFT ? :kan_left : :kan_right
            sym = op.direction == LEFT ? "Σ" : "Δ"
            add_node!(name, kind, "$(sym) $(name)")
            push!(edges, _Edge(op.source, name, :flow))
            push!(edges, _Edge(op.along, name, :relation))
            if op.target !== nothing
                add_node!(op.target, :object, string(op.target))
                push!(edges, _Edge(name, op.target, :flow))
            end
        end
    end

    for (name, loss) in D.losses
        add_node!(name, :loss, "⊿ $(name)")
        for (left, right) in loss.paths
            (left in seen) && push!(edges, _Edge(left, name, :loss))
            (right in seen) && push!(edges, _Edge(right, name, :loss))
        end
    end

    nodes, edges
end

# ----------------------------------------------------------------------------
# Layered layout (longest-path layering with cycle guard)
# ----------------------------------------------------------------------------

function _layout(nodes::Vector{_Node}, edges::Vector{_Edge})
    names = [n.name for n in nodes]
    idx = Dict(n => i for (i, n) in enumerate(names))
    # Forward adjacency for flow/relation edges only (losses don't drive layers).
    succ = Dict(n => Symbol[] for n in names)
    indeg = Dict(n => 0 for n in names)
    for e in edges
        e.style == :loss && continue
        (haskey(idx, e.from) && haskey(idx, e.to)) || continue
        push!(succ[e.from], e.to)
        indeg[e.to] += 1
    end

    # Longest-path layer via relaxation, capped to guard against cycles.
    layer = Dict(n => 0 for n in names)
    ncap = length(names) + 1
    for _ in 1:ncap
        changed = false
        for e in edges
            e.style == :loss && continue
            (haskey(layer, e.from) && haskey(layer, e.to)) || continue
            if layer[e.to] < layer[e.from] + 1
                layer[e.to] = layer[e.from] + 1
                changed = true
            end
        end
        changed || break
    end
    # Losses sit one layer right of the deepest endpoint they touch.
    for e in edges
        e.style == :loss || continue
        (haskey(layer, e.from) && haskey(layer, e.to)) || continue
        layer[e.to] = max(layer[e.to], layer[e.from] + 1)
    end

    maxlayer = isempty(layer) ? 0 : maximum(values(layer))
    # Group by layer, assign y positions evenly within each column.
    bylayer = Dict{Int, Vector{Symbol}}()
    for n in names
        push!(get!(bylayer, layer[n], Symbol[]), n)
    end
    pos = Dict{Symbol, Tuple{Float64, Float64}}()
    for l in 0:maxlayer
        col = get(bylayer, l, Symbol[])
        m = length(col)
        for (k, n) in enumerate(col)
            y = m == 1 ? 0.0 : (k - (m + 1) / 2) * 1.6
            pos[n] = (Float64(2.2 * l), y)
        end
    end
    pos
end

const _NODE_STYLE = Dict(
    :object      => (marker = :circle,    color = (:steelblue, 0.85), size = 34),
    :morphism    => (marker = :rect,      color = (:seagreen, 0.85),  size = 30),
    :composition => (marker = :diamond,   color = (:darkorange, 0.85), size = 32),
    :kan_left    => (marker = :utriangle, color = (:mediumpurple, 0.9), size = 34),
    :kan_right   => (marker = :dtriangle, color = (:indianred, 0.9),   size = 34),
    :loss        => (marker = :xcross,    color = (:crimson, 0.9),     size = 28),
)

const _EDGE_COLOR = Dict(
    :flow     => (:gray25, 0.8),
    :relation => (:mediumpurple, 0.7),
    :loss     => (:crimson, 0.5),
)

# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------

function FunctorFlow.plot_diagram!(ax::Makie.Axis, D::Diagram; show_labels::Bool=true)
    nodes, edges = _build_graph(D)
    pos = _layout(nodes, edges)

    # Edges (arrows shortened so heads land near node boundaries).
    for e in edges
        (haskey(pos, e.from) && haskey(pos, e.to)) || continue
        (x0, y0) = pos[e.from]
        (x1, y1) = pos[e.to]
        u, v = x1 - x0, y1 - y0
        L = sqrt(u^2 + v^2)
        L == 0 && continue
        shrink = min(0.32, 0.28 * L) / L
        sx, sy = x0 + u * shrink, y0 + v * shrink
        du, dv = u * (1 - 2 * shrink), v * (1 - 2 * shrink)
        ls = e.style == :loss ? :dash : :solid
        Makie.arrows!(ax, [sx], [sy], [du], [dv];
                      color = _EDGE_COLOR[e.style], linewidth = 1.6,
                      arrowsize = 11, linestyle = ls)
    end

    # Nodes grouped by kind so each gets its marker/colour.
    for kind in keys(_NODE_STYLE)
        group = [n for n in nodes if n.kind == kind]
        isempty(group) && continue
        xs = [pos[n.name][1] for n in group]
        ys = [pos[n.name][2] for n in group]
        st = _NODE_STYLE[kind]
        Makie.scatter!(ax, xs, ys; marker = st.marker, color = st.color,
                       markersize = st.size, strokewidth = 1, strokecolor = :black)
    end

    if show_labels
        xs = [pos[n.name][1] for n in nodes]
        ys = [pos[n.name][2] + 0.42 for n in nodes]
        Makie.text!(ax, xs, ys; text = [n.label for n in nodes],
                    align = (:center, :bottom), fontsize = 11)
    end

    ax
end

function FunctorFlow.plot_diagram(D::Diagram; show_labels::Bool=true,
                                  size=(900, 600), title=nothing)
    fig = Makie.Figure(; size = size)
    ax = Makie.Axis(fig[1, 1];
                    title = title === nothing ? "Diagram: $(D.name)" : String(title),
                    aspect = Makie.DataAspect())
    Makie.hidedecorations!(ax)
    Makie.hidespines!(ax)
    FunctorFlow.plot_diagram!(ax, D; show_labels = show_labels)
    # Legend describing the node glyphs.
    elems = [Makie.MarkerElement(marker = _NODE_STYLE[k].marker,
                                 color = _NODE_STYLE[k].color, markersize = 14)
             for k in (:object, :morphism, :composition, :kan_left, :kan_right, :loss)]
    labels = ["object", "morphism", "composition", "Σ left-Kan", "Δ right-Kan", "loss"]
    Makie.Legend(fig[1, 2], elems, labels, "Legend"; framevisible = true)
    fig
end

end # module FunctorFlowMakieExt
