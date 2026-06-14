# ============================================================================
# enriched.jl — enriched categories: metric / embedding spaces (Lawvere)
# (included into module Cat)
#
# Lawvere's insight: a (generalized) metric space *is* a category enriched over
# the cost quantale `([0,∞], ≥, +, 0)`. Objects are points, the "hom-object"
# `d(x,y)` is a distance, the identity is `d(x,x) = 0`, and composition is the
# **triangle inequality** `d(x,z) ≤ d(x,y) + d(y,z)`. A non-expansive (1-Lipschitz)
# map is an **enriched functor**. This is the categorical home of metric /
# embedding representation learning: an embedding induces such a category, and
# contrastive/metric objectives are statements about it. Distances are `Nat`
# here, so the axioms are decidable and Lean-certifiable.
# ============================================================================

"""
    MetricCat(points, dist)

A generalized (Lawvere) metric space, i.e. a category enriched over the cost
quantale: `dist[(x,y)]` is the distance `x → y` (a `Nat`; need not be symmetric).
"""
struct MetricCat
    points::Vector{Any}
    dist::Dict{Tuple{Any,Any}, Int}
    function MetricCat(points, dist)
        pts = collect(points)
        d = Dict{Tuple{Any,Any}, Int}((a, b) => Int(v) for ((a, b), v) in dist)
        for x in pts, y in pts
            haskey(d, (x, y)) || throw(ArgumentError("missing distance ($x, $y)"))
            d[(x, y)] >= 0 || throw(ArgumentError("negative distance ($x, $y)"))
        end
        new(pts, d)
    end
end

metric_dist(M::MetricCat, x, y) = M.dist[(x, y)]

"""
    is_lawvere_metric(M) -> Bool

Check the enriched-category axioms: `d(x,x) = 0` (identity) and
`d(x,z) ≤ d(x,y) + d(y,z)` for all `x,y,z` (composition = triangle inequality).
"""
function is_lawvere_metric(M::MetricCat)
    for x in M.points
        M.dist[(x, x)] == 0 || return false
    end
    for x in M.points, y in M.points, z in M.points
        M.dist[(x, z)] <= M.dist[(x, y)] + M.dist[(y, z)] || return false
    end
    true
end

"""
    is_enriched_functor(M, N, f) -> Bool

Is `f : points(M) → points(N)` an enriched functor — a **non-expansive**
(1-Lipschitz) map: `d_N(f x, f y) ≤ d_M(x, y)` for all `x, y`.
"""
function is_enriched_functor(M::MetricCat, N::MetricCat, f::AbstractDict)
    fd = Dict(k => v for (k, v) in f)
    for x in M.points, y in M.points
        haskey(fd, x) || return false
        N.dist[(fd[x], fd[y])] <= M.dist[(x, y)] || return false
    end
    true
end

"""
    embedding_metric(embedding; metric=:l1) -> MetricCat

The enriched category induced by an **embedding** (a representation
`point ↦ integer vector`): distances are the `:l1` or `:linf` distances between
embedded points. The triangle inequality holds, so the embedding *is* a Lawvere
metric space — the categorical content of a learned representation.
"""
function embedding_metric(embedding::AbstractDict; metric::Symbol=:l1)
    pts = collect(keys(embedding))
    d = Dict{Tuple{Any,Any}, Int}()
    for x in pts, y in pts
        vx = embedding[x]; vy = embedding[y]
        d[(x, y)] = metric === :l1 ? sum(abs.(vx .- vy)) :
                    metric === :linf ? maximum(abs.(vx .- vy); init=0) :
                    throw(ArgumentError("metric must be :l1 or :linf"))
    end
    MetricCat(pts, d)
end
