# ============================================================================
# twocat.jl — strict 2-categories & bicategories (included into module Cat)
#
# A **strict 2-category** has 0-cells (objects), 1-cells (morphisms) between
# 0-cells, and 2-cells (morphisms between *parallel* 1-cells). 2-cells compose
# in two ways:
#   • **vertical** (∘): α : f ⇒ g and β : g ⇒ h give β∘α : f ⇒ h
#   • **horizontal** (∗): α : f ⇒ g (a→b) and α' : f' ⇒ g' (b→c) give
#       α'∗α : f'∘f ⇒ g'∘g (a→c)
# The two compositions are tied together by the **interchange law**
#       (β'∘β) ∗ (α'∘α) = (β'∗α') ∘ (β∗α),
# which says the two ways of evaluating a "pasting square" of 2-cells agree.
# A strict 2-category is exactly a category enriched in Cat.
#
# This module represents a *finite* strict 2-category purely by its data: tables
# for the two compositions and the two families of identities, all keyed by
# identifiers (any hashable values). Because everything is finite and tabulated,
# every law — vertical/horizontal associativity & unit, and interchange — is
# *checkable by enumeration*, in the same "category theory with teeth" spirit as
# the rest of the kernel.
#
# Layering on top of the kernel: a strict 2-category built from small categories
# (1-cells = `FinFunctor`, 2-cells = `FunctorNatTrans`) is provided by
# `cat_two_category`, which derives all the tables by actually computing the
# vertical/horizontal composites of natural transformations. The delooping of a
# commutative monoid (`deloop_monoid`) gives the smallest nontrivial example.
#
# Para (reparametrisations as 2-cells) forms a *bi*category, not a strict one;
# see the note `para_is_bicategory_note` at the foot of this file.
# ============================================================================

# ----------------------------------------------------------------------------
# Core data structure
# ----------------------------------------------------------------------------

"""
    TwoCell(name, dom, cod)

A 2-cell `name : dom ⇒ cod` between two **parallel** 1-cells `dom, cod` (the
1-cells must share their source and target 0-cell). `name` is any hashable
identifier; `dom` / `cod` are 1-cell identifiers.
"""
struct TwoCell
    name::Any
    dom::Any     # source 1-cell id
    cod::Any     # target 1-cell id
end

Base.:(==)(a::TwoCell, b::TwoCell) = a.name == b.name && a.dom == b.dom && a.cod == b.cod
Base.hash(a::TwoCell, h::UInt) = hash((a.name, a.dom, a.cod), h)
Base.show(io::IO, a::TwoCell) = print(io, "2cell ", a.name, " : ", a.dom, " ⇒ ", a.cod)

"""
    OneCell(name, dom, cod)

A 1-cell `name : dom → cod` between 0-cells `dom, cod`. `name` is any hashable
identifier; `dom` / `cod` are 0-cell identifiers.
"""
struct OneCell
    name::Any
    dom::Any     # source 0-cell id
    cod::Any     # target 0-cell id
end

Base.:(==)(a::OneCell, b::OneCell) = a.name == b.name && a.dom == b.dom && a.cod == b.cod
Base.hash(a::OneCell, h::UInt) = hash((a.name, a.dom, a.cod), h)
Base.show(io::IO, a::OneCell) = print(io, "1cell ", a.name, " : ", a.dom, " → ", a.cod)

"""
    TwoCategory(; zerocells, onecells, twocells,
                  id1, id2, vcomp, hcomp)

A finite **strict 2-category**, given entirely by data:

* `zerocells :: Vector` — the 0-cell identifiers.
* `onecells  :: Vector{OneCell}` — the 1-cells (each with dom/cod 0-cells).
* `twocells  :: Vector{TwoCell}` — the 2-cells (each between parallel 1-cells).
* `id1 :: Dict` — `0cell ↦ 1cell-id`, the identity 1-cell on each 0-cell.
* `id2 :: Dict` — `1cell-id ↦ 2cell-id`, the identity 2-cell on each 1-cell.
* `vcomp :: Dict{Tuple,Any}` — vertical composition table:
      `(βname, αname) ↦ (β∘α)name` for `α:f⇒g`, `β:g⇒h` (returns a 2-cell id).
* `hcomp :: Dict{Tuple,Any}` — horizontal composition table:
      `(α'name, αname) ↦ (α'∗α)name` for composable α (a→b), α' (b→c).

The constructor validates that all references are well-typed and that the
identity / composition tables have the correct (co)domains; it does **not**
silently accept ill-typed tables. Use [`check_two_category_laws`](@ref) to
verify the algebraic axioms (associativity, units, interchange) by enumeration.
"""
struct TwoCategory
    zerocells::Vector{Any}
    onecells::Vector{OneCell}
    twocells::Vector{TwoCell}
    id1::Dict{Any,Any}
    id2::Dict{Any,Any}
    vcomp::Dict{Any,Any}    # (β,α) ↦ result   (keys are 2-cell names)
    hcomp::Dict{Any,Any}    # (α',α) ↦ result  (keys are 2-cell names)
    # name → cell lookups (derived)
    _one::Dict{Any,OneCell}
    _two::Dict{Any,TwoCell}
end

function TwoCategory(; zerocells, onecells, twocells,
                       id1, id2, vcomp, hcomp)
    zc = collect(Any, zerocells)
    oc = collect(OneCell, onecells)
    tc = collect(TwoCell, twocells)
    zset = Set(zc)
    _one = Dict{Any,OneCell}()
    for f in oc
        f.dom in zset || throw(ArgumentError("1-cell $(f.name): source 0-cell $(f.dom) not declared"))
        f.cod in zset || throw(ArgumentError("1-cell $(f.name): target 0-cell $(f.cod) not declared"))
        haskey(_one, f.name) && throw(ArgumentError("duplicate 1-cell name $(f.name)"))
        _one[f.name] = f
    end
    _two = Dict{Any,TwoCell}()
    for α in tc
        haskey(_one, α.dom) || throw(ArgumentError("2-cell $(α.name): source 1-cell $(α.dom) unknown"))
        haskey(_one, α.cod) || throw(ArgumentError("2-cell $(α.name): target 1-cell $(α.cod) unknown"))
        d, c = _one[α.dom], _one[α.cod]
        (d.dom == c.dom && d.cod == c.cod) ||
            throw(ArgumentError("2-cell $(α.name): its 1-cells $(α.dom),$(α.cod) are not parallel"))
        haskey(_two, α.name) && throw(ArgumentError("duplicate 2-cell name $(α.name)"))
        _two[α.name] = α
    end
    id1d = Dict{Any,Any}(k => v for (k, v) in id1)
    id2d = Dict{Any,Any}(k => v for (k, v) in id2)
    vc = Dict{Any,Any}(k => v for (k, v) in vcomp)
    hc = Dict{Any,Any}(k => v for (k, v) in hcomp)

    # validate id1: id1[a] is a 1-cell a→a
    for a in zc
        haskey(id1d, a) || throw(ArgumentError("missing identity 1-cell on 0-cell $a"))
        f = get(_one, id1d[a], nothing)
        f === nothing && throw(ArgumentError("id1[$a] = $(id1d[a]) is not a declared 1-cell"))
        (f.dom == a && f.cod == a) || throw(ArgumentError("id1[$a] is not an endo-1-cell on $a"))
    end
    # validate id2: id2[f] is a 2-cell f⇒f
    for f in oc
        haskey(id2d, f.name) || throw(ArgumentError("missing identity 2-cell on 1-cell $(f.name)"))
        α = get(_two, id2d[f.name], nothing)
        α === nothing && throw(ArgumentError("id2[$(f.name)] = $(id2d[f.name]) is not a declared 2-cell"))
        (α.dom == f.name && α.cod == f.name) ||
            throw(ArgumentError("id2[$(f.name)] is not an endo-2-cell on $(f.name)"))
    end
    # validate vcomp typing: defined exactly for vertically composable pairs
    for α in tc, β in tc
        if α.cod == β.dom    # α:f⇒g , β:g⇒h  composable vertically
            haskey(vc, (β.name, α.name)) ||
                throw(ArgumentError("vcomp missing for composable ($(β.name) ∘ $(α.name))"))
            r = get(_two, vc[(β.name, α.name)], nothing)
            r === nothing && throw(ArgumentError("vcomp($(β.name),$(α.name)) is not a 2-cell"))
            (r.dom == α.dom && r.cod == β.cod) ||
                throw(ArgumentError("vcomp($(β.name),$(α.name)) has wrong (co)domain: " *
                                    "got $(r.dom)⇒$(r.cod), want $(α.dom)⇒$(β.cod)"))
        end
    end
    # validate hcomp typing: defined exactly for horizontally composable pairs
    for α in tc, β in tc
        # α : a→b , β : b→c  (horizontally composable: cod 0-cell of α = dom 0-cell of β)
        fα = _one[α.dom]; fβ = _one[β.dom]
        if fα.cod == fβ.dom
            haskey(hc, (β.name, α.name)) ||
                throw(ArgumentError("hcomp missing for composable ($(β.name) ∗ $(α.name))"))
            r = get(_two, hc[(β.name, α.name)], nothing)
            r === nothing && throw(ArgumentError("hcomp($(β.name),$(α.name)) is not a 2-cell"))
            # result : (β.dom ∘ α.dom) ⇒ (β.cod ∘ α.cod) — we only check the 0-cell endpoints
            rd, rc = _one[r.dom], _one[r.cod]
            (rd.dom == fα.dom && rd.cod == fβ.cod && rc.dom == fα.dom && rc.cod == fβ.cod) ||
                throw(ArgumentError("hcomp($(β.name),$(α.name)) has wrong 0-cell endpoints"))
        end
    end
    TwoCategory(zc, oc, tc, id1d, id2d, vc, hc, _one, _two)
end

# convenience accessors --------------------------------------------------------

"""`zerocells(K)` — the 0-cell identifiers of a 2-category."""
zerocells(K::TwoCategory) = copy(K.zerocells)
"""`onecells(K)` — all 1-cells; `onecells(K, a, b)` — those `a → b`."""
onecells(K::TwoCategory) = copy(K.onecells)
onecells(K::TwoCategory, a, b) = [f for f in K.onecells if f.dom == a && f.cod == b]
"""`twocells(K)` — all 2-cells; `twocells(K, f, g)` — those `f ⇒ g` (by 1-cell name)."""
twocells(K::TwoCategory) = copy(K.twocells)
twocells(K::TwoCategory, f, g) = [α for α in K.twocells if α.dom == f && α.cod == g]

"""`id1(K, a)` — identity 1-cell on 0-cell `a` (returns its 1-cell id)."""
id1(K::TwoCategory, a) = K.id1[a]
"""`id2(K, f)` — identity 2-cell on 1-cell `f` (returns its 2-cell id)."""
id2(K::TwoCategory, f) = K.id2[f]

"""`vcomp(K, β, α)` — vertical composite `β ∘ α` (2-cell ids in, id out)."""
function vcomp(K::TwoCategory, β, α)
    a = K._two[α]; b = K._two[β]
    a.cod == b.dom || throw(ArgumentError("not vertically composable: $α : ⇒$(a.cod), $β : $(b.dom)⇒"))
    K.vcomp[(β, α)]
end
"""`hcomp(K, β, α)` — horizontal composite `β ∗ α` (2-cell ids in, id out)."""
function hcomp(K::TwoCategory, β, α)
    a = K._two[α]; b = K._two[β]
    fα = K._one[a.dom]; fβ = K._one[b.dom]
    fα.cod == fβ.dom || throw(ArgumentError("not horizontally composable: $α then $β at 0-cells"))
    K.hcomp[(β, α)]
end

# ----------------------------------------------------------------------------
# Law checks — all by enumeration
# ----------------------------------------------------------------------------

"""
    check_vertical_category_laws(K) -> Bool

For each fixed pair of parallel 1-cells `f, g`, the 2-cells `f ⇒ g` together
with `∘` and the `id2`'s must form a category (the "hom-category"). Checks
left/right unit and associativity of vertical composition by enumeration.
"""
function check_vertical_category_laws(K::TwoCategory)
    tc = K.twocells
    # unit laws: id2[cod α] ∘ α = α = α ∘ id2[dom α]
    for α in tc
        vcomp(K, id2(K, α.cod), α.name) == α.name || return false
        vcomp(K, α.name, id2(K, α.dom)) == α.name || return false
    end
    # associativity: (γ∘β)∘α = γ∘(β∘α) whenever vertically composable
    for α in tc, β in tc
        α.cod == β.dom || continue
        for γ in tc
            β.cod == γ.dom || continue
            lhs = vcomp(K, vcomp(K, γ.name, β.name), α.name)
            rhs = vcomp(K, γ.name, vcomp(K, β.name, α.name))
            lhs == rhs || return false
        end
    end
    true
end

"""
    check_horizontal_category_laws(K) -> Bool

The 0-cells, 1-cells and *horizontal* composition of 2-cells (with identity
2-cells on the identity 1-cells as units) must form a category up to the
2-cell level. Checks horizontal unit and associativity by enumeration.

Unit: `id2[id1[cod]] ∗ α = α = α ∗ id2[id1[dom]]` (using that strictness makes
`id1∘f = f = f∘id1` on the nose at the 1-cell level).
"""
function check_horizontal_category_laws(K::TwoCategory)
    tc = K.twocells
    # horizontal unit laws via identity 2-cells on identity 1-cells
    for α in tc
        f = K._one[α.dom]              # 1-cell a→b
        idA = id2(K, id1(K, f.dom))    # 2-cell on id1 at source 0-cell a
        idB = id2(K, id1(K, f.cod))    # 2-cell on id1 at target 0-cell b
        # idB ∗ α : (id1_b ∘ f) ⇒ (id1_b ∘ g);  strictness ⇒ equals α
        hcomp(K, idB, α.name) == α.name || return false
        hcomp(K, α.name, idA) == α.name || return false
    end
    # horizontal associativity: (γ∗β)∗α = γ∗(β∗α)
    for α in tc, β in tc, γ in tc
        fα = K._one[α.dom]; fβ = K._one[β.dom]; fγ = K._one[γ.dom]
        (fα.cod == fβ.dom && fβ.cod == fγ.dom) || continue
        lhs = hcomp(K, hcomp(K, γ.name, β.name), α.name)
        rhs = hcomp(K, γ.name, hcomp(K, β.name, α.name))
        lhs == rhs || return false
    end
    true
end

"""
    check_interchange_law(K) -> Bool

The heart of a strict 2-category. For every "pasting square" of 2-cells

```
      a --f-->--g-->--h--> b   (left column, on 0-cells a→b)
            α      β
      b --f'-->-g'-->-h'-> c   (right column, on 0-cells b→c)
            α'     β'
```

with vertically composable left column `α:f⇒g, β:g⇒h` and right column
`α':f'⇒g', β':g'⇒h'`, horizontally composable (`a→b` then `b→c`), the
interchange law requires

    (β' ∘ α') ∗ (β ∘ α) = (β' ∗ β) ∘ (α' ∗ α),

i.e. composing each *column* vertically then horizontally equals composing each
*row* horizontally then vertically. (This is the same statement as the textbook
`(β'∘β) ∗ (α'∘α) = (β'∗α') ∘ (β∗α)` under a relabelling of the four 2-cells.)
Checked over every such quadruple by enumeration.
"""
function check_interchange_law(K::TwoCategory)
    tc = K.twocells
    for α in tc, β in tc
        α.cod == β.dom || continue                       # α:f⇒g, β:g⇒h
        for α′ in tc, β′ in tc
            α′.cod == β′.dom || continue                 # α′:f′⇒g′, β′:g′⇒h′
            fα = K._one[α.dom]; fα′ = K._one[α′.dom]
            fα.cod == fα′.dom || continue                # horizontally composable
            # Configuration (left column on a→b, right column on b→c):
            #   bottom row: α : f⇒g  (left),  α′ : f′⇒g′ (right)
            #   top row:    β : g⇒h  (left),  β′ : g′⇒h′ (right)
            # LHS — compose each column vertically, then the two results horizontally:
            #   (β′∘α′) ∗ (β∘α)
            lhs = hcomp(K, vcomp(K, β′.name, α′.name), vcomp(K, β.name, α.name))
            # RHS — compose each row horizontally, then the two results vertically:
            #   (β′∗β) ∘ (α′∗α)
            rhs = vcomp(K, hcomp(K, β′.name, β.name), hcomp(K, α′.name, α.name))
            lhs == rhs || return false
        end
    end
    true
end

"""
    check_two_category_laws(K) -> Bool

All strict-2-category axioms by enumeration: vertical category laws, horizontal
category laws, and the interchange law.
"""
check_two_category_laws(K::TwoCategory) =
    check_vertical_category_laws(K) &&
    check_horizontal_category_laws(K) &&
    check_interchange_law(K)

# ----------------------------------------------------------------------------
# Worked example 1: delooping of a commutative monoid
# ----------------------------------------------------------------------------

"""
    deloop_monoid(elements, mul, unit) -> TwoCategory

The **delooping** `B²M` of a commutative monoid `(M, ·, e)` as a one-0-cell,
one-1-cell strict 2-category: the unique 0-cell `★`, the unique 1-cell `id`
(`★→★`), and the elements of `M` as 2-cells `id ⇒ id`. *Both* vertical and
horizontal composition are the monoid product `·`, and both identities are the
unit `e`. The interchange law for this 2-category is precisely the
**Eckmann–Hilton** statement that the two products agree and are commutative —
so a commutative monoid is exactly the data that makes this a strict
2-category. (`mul(x,y)` is the product; `unit` is `e`.)
"""
function deloop_monoid(elements, mul, unit)
    M = collect(elements)
    star = :★
    one1 = OneCell(:id, star, star)
    twos = TwoCell[TwoCell(m, :id, :id) for m in M]
    vc = Dict{Any,Any}()
    hc = Dict{Any,Any}()
    for x in M, y in M
        # vertical β∘α and horizontal β∗α both = monoid product
        vc[(x, y)] = mul(x, y)
        hc[(x, y)] = mul(x, y)
    end
    TwoCategory(;
        zerocells = [star],
        onecells  = [one1],
        twocells  = twos,
        id1 = Dict(star => :id),
        id2 = Dict(:id => unit),
        vcomp = vc,
        hcomp = hc)
end

# ----------------------------------------------------------------------------
# Worked example 2: 2-category from small categories
#   0-cells = small categories, 1-cells = FinFunctors, 2-cells = FunctorNatTrans
# ----------------------------------------------------------------------------

"""
    vcompose(β::FunctorNatTrans, α::FunctorNatTrans) -> FunctorNatTrans

Vertical composite `β ∘ α : F ⇒ H` of `α : F ⇒ G` and `β : G ⇒ H` (parallel
functors `C → D`). Componentwise: `(β∘α)_c = α_c · β_c` in `D` (diagrammatic).
"""
function vcompose(β::FunctorNatTrans, α::FunctorNatTrans)
    α.cod == β.dom || throw(ArgumentError("vertical: cod(α) ≠ dom(β)"))
    D = α.dom.cod
    comps = Dict{Symbol,PathMor}(
        c => compose(D, α.components[c], β.components[c]) for c in α.dom.dom.objects)
    FunctorNatTrans(α.dom, β.cod; components = comps)
end

"""
    hcompose(β::FunctorNatTrans, α::FunctorNatTrans) -> FunctorNatTrans

Horizontal composite `β ∗ α` of `α : F ⇒ G` (functors `C → D`) and
`β : F′ ⇒ G′` (functors `D → E`), giving `(F′∘F) ⇒ (G′∘G) : C → E`.
Standard formula at object `c` (one of the two equal diagonals):
`(β ∗ α)_c = β_{F c} · G′(α_c)`  (diagrammatic order in `E`).
"""
function hcompose(β::FunctorNatTrans, α::FunctorNatTrans)
    F, G = α.dom, α.cod          # C → D
    F′, G′ = β.dom, β.cod        # D → E
    F.cod == F′.dom || throw(ArgumentError("horizontal: cod-category of α ≠ dom-category of β"))
    C = F.dom; E = F′.cod
    comps = Dict{Symbol,PathMor}()
    for c in C.objects
        # β_{F c} : F′(F c) → G′(F c)  ;  G′(α_c) : G′(F c) → G′(G c)
        left  = β.components[F.ob_map[c]]
        right = G′(α.components[c])
        comps[c] = compose(E, left, right)
    end
    FunctorNatTrans(compose(F, F′), compose(G, G′); components = comps)
end

"""
    identity_nat(F::FinFunctor) -> FunctorNatTrans

The identity natural transformation `id_F : F ⇒ F` (each component is the
identity morphism on `F(c)`).
"""
identity_nat(F::FinFunctor) =
    FunctorNatTrans(F, F; components = Dict(c => id(F.cod, F.ob_map[c]) for c in F.dom.objects))

"""
    cat_two_category(cats, functors, nats; names...) -> TwoCategory

Assemble an explicit finite strict `TwoCategory` whose 0-cells are the supplied
small categories, 1-cells the supplied `FinFunctor`s, and 2-cells the supplied
`FunctorNatTrans`es. The composition tables are *computed* from the actual
vertical/horizontal composites of natural transformations and matched back to
the supplied 2-cells, so the resulting `TwoCategory` is a faithful tabulation of
(a finite full sub-2-category of) **Cat** — and `check_two_category_laws` on it
re-derives the interchange law from honest computation.

Arguments (all keyword, as named collections):
* `cats     :: Dict{Any,FreeCat}` — 0-cell id ↦ category.
* `functors :: Dict{Any,FinFunctor}` — 1-cell id ↦ functor. Must include an
  identity functor on each category, referenced by `id1`.
* `nats     :: Dict{Any,FunctorNatTrans}` — 2-cell id ↦ nat. transformation.
  Must be closed under the vertical/horizontal composites that arise, and
  include the identity nat-trans on each functor, referenced by `id2`.
* `id1 :: Dict` — 0-cell id ↦ 1-cell id (identity functor).
* `id2 :: Dict` — 1-cell id ↦ 2-cell id (identity nat-trans).

Throws if a required composite is not present among `nats` (so you cannot build
a 2-category that isn't closed under composition).
"""
function cat_two_category(; cats::AbstractDict, functors::AbstractDict,
                            nats::AbstractDict, id1::AbstractDict, id2::AbstractDict)
    # 0-cell ids
    zc = collect(Any, keys(cats))
    # map a category value back to its 0-cell id
    cat_id = Dict{Any,Any}()
    for (k, C) in cats
        cat_id[objects(C) => [(n, s, t) for (n, s, t) in C.edges]] = k
    end
    _whichcat(C) = cat_id[objects(C) => [(n, s, t) for (n, s, t) in C.edges]]

    onecells = OneCell[]
    for (fid, F) in functors
        push!(onecells, OneCell(fid, _whichcat(F.dom), _whichcat(F.cod)))
    end
    # map a FinFunctor value back to its 1-cell id (by structural equality)
    func_id = Dict{Any,Any}()
    for (fid, F) in functors
        func_id[_functor_key(F)] = fid
    end
    _whichfunctor(F) = func_id[_functor_key(F)]

    twocells = TwoCell[]
    for (aid, α) in nats
        push!(twocells, TwoCell(aid, _whichfunctor(α.dom), _whichfunctor(α.cod)))
    end
    nat_id = Dict{Any,Any}()
    for (aid, α) in nats
        nat_id[_nat_key(α)] = aid
    end
    _whichnat(α) = get(nat_id, _nat_key(α), nothing)

    # vertical composition table
    vc = Dict{Any,Any}()
    for (aid, α) in nats, (bid, β) in nats
        α.cod == β.dom || continue
        γ = vcompose(β, α)
        gid = _whichnat(γ)
        gid === nothing &&
            throw(ArgumentError("nats not closed under vertical composition: " *
                                "$bid ∘ $aid produced a nat-trans not in `nats`"))
        vc[(bid, aid)] = gid
    end
    # horizontal composition table
    hc = Dict{Any,Any}()
    for (aid, α) in nats, (bid, β) in nats
        α.cod.cod == β.dom.dom || continue   # cod-category of α = dom-category of β
        γ = hcompose(β, α)
        gid = _whichnat(γ)
        gid === nothing &&
            throw(ArgumentError("nats not closed under horizontal composition: " *
                                "$bid ∗ $aid produced a nat-trans not in `nats`"))
        hc[(bid, aid)] = gid
    end

    TwoCategory(;
        zerocells = zc,
        onecells  = onecells,
        twocells  = twocells,
        id1 = Dict(k => v for (k, v) in id1),
        id2 = Dict(k => v for (k, v) in id2),
        vcomp = vc,
        hcomp = hc)
end

# structural keys for matching functor / nat-trans values back to ids
_functor_key(F::FinFunctor) =
    (objects(F.dom), [(n, s, t) for (n, s, t) in F.dom.edges],
     objects(F.cod), [(n, s, t) for (n, s, t) in F.cod.edges],
     sort(collect(F.ob_map)), sort(collect(F.edge_map); by = first))

_nat_key(α::FunctorNatTrans) =
    (_functor_key(α.dom), _functor_key(α.cod),
     sort(collect(α.components); by = first))

# ----------------------------------------------------------------------------
# Para is a bicategory (note + realization)
# ----------------------------------------------------------------------------

"""
    para_reparam_two_cell(f::ParaMap, g::ParaMap, r::FinFunction) -> Bool

A **2-cell** of the bicategory **Para** from `f : A→B` (params `P`) to
`g : A→B` (params `Q`) is a *reparametrisation*: a map `r : Q → P` on parameter
spaces such that running `g` at `q` equals running `f` at `r(q)`:

    g.impl(q, a) = f.impl(r(q), a)   for all q ∈ Q, a ∈ A.

This predicate checks that `r` is such a 2-cell. Note Para is only a *bi*category
(not strict): composition of 1-cells pairs parameter spaces, and `P×(Q×R)` is
merely isomorphic — not equal — to `(P×Q)×R`, so associativity holds only up to
(invertible) 2-cell. The present `TwoCategory` struct models *strict* 2-categories;
Para's weak associator/unitors live one level up and are recorded here as a
documented realization rather than enforced by the struct.
"""
function para_reparam_two_cell(f::ParaMap, g::ParaMap, r::FinFunction)
    (f.A == g.A && f.B == g.B) || return false
    Pset = Set(f.P); Qset = Set(g.P)
    r.dom == FinSet(collect(Qset)) || return false
    all(r(q) in Pset for q in g.P) || return false
    all(g.impl((q, a)) == f.impl((r(q), a)) for q in g.P, a in f.A)
end

"""
    para_is_bicategory_note() -> String

A short statement of why **Para** is a bicategory: 1-cells `A→B` are parametric
maps `P×A→B`; horizontal composition tensors the parameter spaces; the
associator/unitors are the canonical isomorphisms reassociating those products,
which are invertible 2-cells (reparametrisations) but not identities. Hence Para
is weak (a bicategory), in contrast to the strict 2-categories tabulated by
[`TwoCategory`].
"""
para_is_bicategory_note() = """
Para is a BICATEGORY (weak 2-category):
  • 0-cells: finite sets (objects).
  • 1-cells A→B: parametric maps `impl : P×A → B` (ParaMap), composed by
    pairing parameter spaces — (g∘f) has parameters Q×P.
  • 2-cells f⇒g: reparametrisations `r : Q → P` with g(q,a) = f(r(q),a)
    (see `para_reparam_two_cell`).
  • Associativity of 1-cell composition holds only up to the canonical
    isomorphism P×(Q×R) ≅ (P×Q)×R — an invertible 2-cell, not an identity —
    so Para is weak. The strict `TwoCategory` here cannot encode Para on the
    nose; that requires bicategory coherence (pentagon/triangle).
"""
