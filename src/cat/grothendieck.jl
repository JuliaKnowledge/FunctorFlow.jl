# ============================================================================
# grothendieck.jl — the Grothendieck construction (category of elements)
# (included into module Cat)
#
# For a copresheaf `F : C → Set`, the category of elements `∫F` has objects the
# pairs `(c, x∈F(c))` and a morphism `(c,x) → (c',x')` for each `f : c→c'` with
# `F(f)(x) = x'`. This is the categorical "database of rows" of a C-Set: it
# fibres over `C` and its objects are the actual data items. Being a free
# category on the element graph, it is law-checked and Lean-certifiable via the
# existing `render_cat_certificate`.
# ============================================================================

_elem_obj(c, x) = Symbol(string(c, "#", x))

"""
    category_of_elements(F::SetFunctor) -> FreeCat

The Grothendieck construction `∫F`: objects are `(c, x∈F(c))` pairs, generating
morphisms are the actions of `C`'s generators on elements.
"""
function category_of_elements(F::SetFunctor)
    C = F.cat
    objs = Symbol[]
    for c in objects(C), x in ob(F, c).elements
        push!(objs, _elem_obj(c, x))
    end
    edges = Tuple{Symbol,Symbol,Symbol}[]
    for (n, c, d) in C.edges
        f = hommap(F, PathMor(c, d, Symbol[n]))
        for x in ob(F, c).elements
            push!(edges, (Symbol(string(n, "@", c, "#", x)), _elem_obj(c, x), _elem_obj(d, f(x))))
        end
    end
    FreeCat(objs, edges)
end

"""
    elements_projection(F::SetFunctor) -> FinFunctor

The projection functor `∫F → C` sending each `(c, x)` to `c` (the fibration of
the category of elements over its base). Witnesses `∫F` as living *over* `C`.
"""
function elements_projection(F::SetFunctor)
    C = F.cat
    E = category_of_elements(F)
    ob_map = Dict{Symbol, Symbol}()
    for c in objects(C), x in ob(F, c).elements
        ob_map[_elem_obj(c, x)] = c
    end
    edge_map = Dict{Symbol, PathMor}()
    for (n, c, d) in C.edges, x in ob(F, c).elements
        en = Symbol(string(n, "@", c, "#", x))
        edge_map[en] = PathMor(c, d, Symbol[n])   # the underlying base morphism
    end
    FinFunctor(E, C; ob_map=ob_map, edge_map=edge_map)
end
