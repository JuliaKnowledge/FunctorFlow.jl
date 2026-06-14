# ============================================================================
# kan_general.jl — Kan extensions Lan_F / Ran_F along an arbitrary functor
# (included into module Cat)
#
# For F : C → D and X : C → Set the (pointwise) Kan extensions are
#   Lan_F X (d) = colim_{(c, h:F(c)→d) ∈ F↓d} X(c)      (left, a colimit)
#   Ran_F X (d) = lim_{(c, h:d→F(c)) ∈ d↓F} X(c)        (right, a limit)
# computed directly over the comma categories. Each is returned as a full
# `SetFunctor` on `D` (with its action on D-morphisms). These are the genuine
# adjoints of the restriction functor `F*` (`Lan_F ⊣ F* ⊣ Ran_F`); the
# `colimit`/`limit` in `kan.jl` are the special case `F = ! : C → 1`.
# ============================================================================

"""
    left_kan(F::FinFunctor, X::SetFunctor) -> SetFunctor

The left Kan extension `Lan_F X : D → Set` of `X : C → Set` along `F : C → D`,
computed pointwise as the colimit over each comma category `F ↓ d`.
"""
function left_kan(F::FinFunctor, X::SetFunctor)
    C, D = F.dom, F.cod
    X.cat == C || throw(ArgumentError("left_kan: X must be a functor on dom(F)"))
    repcache = Dict{Symbol, Dict{Any,Any}}()
    ob_map = Dict{Symbol, FinSet}()
    for d in objects(D)
        tagged = Any[]
        for c in objects(C), h in homset(D, F.ob_map[c], d), x in ob(X, c).elements
            push!(tagged, (c, h, x))
        end
        idx = Dict{Any,Int}(t => i for (i, t) in enumerate(tagged))
        parent = collect(1:length(tagged))
        find(i) = (while parent[i] != i; parent[i] = parent[parent[i]]; i = parent[i]; end; i)
        for c in objects(C), c2 in objects(C), u in homset(C, c, c2)
            Fu = F(u); Xu = hommap(X, u)
            for h2 in homset(D, F.ob_map[c2], d)
                h = compose(D, Fu, h2)
                for x in ob(X, c).elements
                    i, j = find(idx[(c, h, x)]), find(idx[(c2, h2, Xu(x))])
                    i == j || (parent[i] = j)
                end
            end
        end
        repof = Dict{Any,Any}(t => tagged[find(idx[t])] for t in tagged)
        repcache[d] = repof
        ob_map[d] = FinSet(unique(values(repof)))
    end
    edge_map = Dict{Symbol, FinFunction}()
    for (n, d, d2) in D.edges
        g = PathMor(d, d2, Symbol[n])
        m = Dict{Any,Any}()
        for t in ob_map[d].elements
            (c, h, x) = t
            m[t] = repcache[d2][(c, compose(D, h, g), x)]
        end
        edge_map[n] = FinFunction(ob_map[d], ob_map[d2], m)
    end
    SetFunctor(D; ob_map=ob_map, edge_map=edge_map)
end

"""
    right_kan(F::FinFunctor, X::SetFunctor) -> SetFunctor

The right Kan extension `Ran_F X : D → Set` of `X : C → Set` along `F : C → D`,
computed pointwise as the limit (compatible families) over each comma category
`d ↓ F`.
"""
function right_kan(F::FinFunctor, X::SetFunctor)
    C, D = F.dom, F.cod
    X.cat == C || throw(ArgumentError("right_kan: X must be a functor on dom(F)"))
    comma = Dict{Symbol, Vector{Any}}()
    ob_map = Dict{Symbol, FinSet}()
    for d in objects(D)
        cobjs = Any[]
        for c in objects(C), h in homset(D, d, F.ob_map[c])
            push!(cobjs, (c, h))
        end
        comma[d] = cobjs
        families = Any[]
        if isempty(cobjs)
            push!(families, ())
        else
            for combo in Iterators.product((ob(X, co[1]).elements for co in cobjs)...)
                fam = Dict{Any,Any}(cobjs[i] => combo[i] for i in eachindex(cobjs))
                ok = true
                for (c, h) in cobjs
                    for c2 in objects(C), u in homset(C, c, c2)
                        h2 = compose(D, h, F(u))               # d → F(c2)
                        if hommap(X, u)(fam[(c, h)]) != fam[(c2, h2)]
                            ok = false; break
                        end
                    end
                    ok || break
                end
                ok && push!(families, Tuple(combo))
            end
        end
        ob_map[d] = FinSet(families)
    end
    edge_map = Dict{Symbol, FinFunction}()
    for (n, d, d2) in D.edges
        g = PathMor(d, d2, Symbol[n])
        cobjs_d = comma[d]
        pos = Dict{Any,Int}(cobjs_d[i] => i for i in eachindex(cobjs_d))
        m = Dict{Any,Any}()
        for fam in ob_map[d].elements
            newfam = Tuple(fam[pos[(c, compose(D, g, h2))]] for (c, h2) in comma[d2])
            m[fam] = newfam
        end
        edge_map[n] = FinFunction(ob_map[d], ob_map[d2], m)
    end
    SetFunctor(D; ob_map=ob_map, edge_map=edge_map)
end
