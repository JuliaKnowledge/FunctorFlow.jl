# ============================================================================
# sheaf.jl — Grothendieck (co)topologies, the sheaf condition, sheafification
# (included into module Cat)
#
# VARIANCE.  The whole `Cat` kernel works with **copresheaves** `F : C → Set`
# (`SetFunctor`), and the subobject classifier in `topos.jl` is built from
# **cosieves** — sets of morphisms *out of* an object, closed under
# post-composition.  To stay consistent with that machinery we develop the
# sheaf theory in the *exactly dual* ("co") setting:
#
#   * a covering family on `c` is a **cosieve** `R` on `c`
#     (morphisms with domain `c`, closed under post-composition);
#   * a **coverage / Grothendieck (co)topology** `J` assigns to each object `c`
#     a set `J(c)` of covering cosieves;
#   * for a copresheaf `F : C → Set`, a **matching family** for a cover `R` on
#     `c` is a choice `x_f ∈ F(cod f)` for each `f ∈ R` that is compatible with
#     post-composition: `F(g)(x_f) = x_{f·g}` for every `g : cod f → ·`;
#   * an **amalgamation** is an `x ∈ F(c)` with `F(f)(x) = x_f` for all `f ∈ R`;
#   * `F` is **separated** for `J` if every matching family has *at most one*
#     amalgamation, and a **sheaf** if every matching family has *exactly one*.
#
# This is the literal dual of the textbook presheaf/sieve story (just replace
# "sieve / morphisms into c / pre-composition" by "cosieve / morphisms out of
# c / post-composition"), and it reuses `_cosieves` / `_morphisms_out` from
# `topos.jl` verbatim.  Everything is finite, so each clause is checked by
# honest enumeration.
# ============================================================================

# ----------------------------------------------------------------------------
# Coverage / Grothendieck (co)topology
# ----------------------------------------------------------------------------

"""
    Coverage(C; covers)

A finite **coverage** (Grothendieck co-topology candidate) on a [`FreeCat`](@ref)
`C`: `covers` maps each object `c` to a `Vector` of covering cosieves on `c`
(each cosieve is a sorted `Vector{PathMor}` of morphisms out of `c`, closed
under post-composition — i.e. an element of `Ω(c)` from the subobject
classifier).  The maximal cosieve on every object is added automatically, so
the maximal-sieve axiom always holds; the remaining axioms are checked by
[`is_grothendieck_topology`](@ref).
"""
struct Coverage
    cat::AbstractCategory
    covers::Dict{Symbol, Vector{Vector{PathMor}}}
end

function Coverage(C::AbstractCategory; covers::AbstractDict=Dict())
    cov = Dict{Symbol, Vector{Vector{PathMor}}}()
    for c in objects(C)
        cov[c] = Vector{PathMor}[]
    end
    for (c, Rs) in covers
        c = Symbol(c)
        haskey(cov, c) || throw(ArgumentError("Coverage: unknown object $c"))
        for R in Rs
            Rs_sorted = sort(collect(R); by=_mor_key)
            _is_cosieve(C, c, Rs_sorted) ||
                throw(ArgumentError("Coverage: family on $c is not a (closed) cosieve"))
            push!(cov[c], Rs_sorted)
        end
    end
    # always include the maximal cosieve (the maximal-sieve / "isomorphism" axiom)
    for c in objects(C)
        mx = _maximal_cosieve(C, c)
        any(_same_cosieve(R, mx) for R in cov[c]) || push!(cov[c], mx)
    end
    Coverage(C, cov)
end

# is `R` (morphisms out of c) a cosieve: every f∈R, every g post-composable ⇒ f·g∈R
function _is_cosieve(C::AbstractCategory, c, R)
    Rset = Set(R)
    for f in R
        f.dom == Symbol(c) || return false
        for d in objects(C), g in homset(C, f.cod, d)
            compose(C, f, g) in Rset || return false
        end
    end
    true
end

_same_cosieve(R, S) = Set(R) == Set(S)

"""`covering_sieves(J, c)` — the covering cosieves on object `c`."""
covering_sieves(J::Coverage, c) = J.covers[Symbol(c)]

"""
    is_grothendieck_topology(J::Coverage) -> Bool

Check the Grothendieck-topology axioms for the coverage `J`, dualised to
cosieves on the covariant kernel:

  * **(maximal)** the maximal cosieve covers every object — true by construction;
  * **(stability)** if `R` covers `c` and `h : c → d` is *any* morphism, then the
    "pushforward" `h^*R = {g : d → · | h·g ∈ R}` covers `d`;
  * **(transitivity)** if `R` covers `c` and `S` is a cosieve on `c` such that
    for every `f : c → e` in `R` the pulled-back family `f^*S = {g | f·g ∈ S}`
    covers `e`, then `S` covers `c`.

(These are the exact duals of the sieve axioms; over a finite site they are
fully decidable, so this is a real certificate, not vocabulary.)
"""
function is_grothendieck_topology(J::Coverage)
    C = J.cat
    covers_c(c) = J.covers[Symbol(c)]
    is_cover(c, R) = any(_same_cosieve(R, S) for S in covers_c(c))
    # (maximal) — guaranteed by the constructor, re-asserted here
    for c in objects(C)
        is_cover(c, _maximal_cosieve(C, c)) || return false
    end
    # (stability)
    for c in objects(C), R in covers_c(c)
        Rset = Set(R)
        for d in objects(C), h in homset(C, c, d)
            push = PathMor[g for g in _morphisms_out(C, d) if compose(C, h, g) in Rset]
            push = sort(push; by=_mor_key)
            is_cover(d, push) || return false
        end
    end
    # (transitivity)
    for c in objects(C), R in covers_c(c)
        # every cosieve S on c that is "locally covering along R" must be a cover
        for S in _cosieves(C, c)
            Sset = Set(S)
            ok = true
            for f in R
                pull = PathMor[g for g in _morphisms_out(C, f.cod) if compose(C, f, g) in Sset]
                pull = sort(pull; by=_mor_key)
                is_cover(f.cod, pull) || (ok = false; break)
            end
            ok || continue
            is_cover(c, S) || return false
        end
    end
    true
end

# ----------------------------------------------------------------------------
# Matching families and the sheaf condition
# ----------------------------------------------------------------------------

"""
    matching_families(F::SetFunctor, R) -> Vector{Dict{PathMor,Any}}

All **matching families** for the covering cosieve `R` (on `c = dom` of `R`'s
morphisms) and copresheaf `F`: every assignment `f ↦ x_f ∈ F(cod f)` that is
compatible with post-composition, `F(g)(x_f) = x_{f·g}` for all `g`. Enumerated
exhaustively (finite).
"""
function matching_families(F::SetFunctor, R)
    C = F.cat
    isempty(R) && return [Dict{PathMor,Any}()]
    fs = collect(R)
    choices = [ob(F, f.cod).elements for f in fs]
    out = Dict{PathMor,Any}[]
    # cartesian product of choices
    idx = ones(Int, length(fs))
    function emit()
        fam = Dict{PathMor,Any}(fs[i] => choices[i][idx[i]] for i in eachindex(fs))
        _is_matching(F, R, fam) && push!(out, fam)
    end
    while true
        emit()
        # increment mixed-radix counter
        k = length(fs)
        while k >= 1
            idx[k] += 1
            idx[k] <= length(choices[k]) && break
            idx[k] = 1; k -= 1
        end
        k == 0 && break
    end
    out
end

# is `fam` (f ↦ x_f) compatible: F(g)(x_f) = x_{f·g} for every post-composable g
function _is_matching(F::SetFunctor, R, fam)
    C = F.cat
    Rset = Set(R)
    for f in R
        xf = fam[f]
        for d in objects(C), g in homset(C, f.cod, d)
            fg = compose(C, f, g)
            fg in Rset || continue                # R is a cosieve ⇒ this holds
            haskey(fam, fg) || return false
            hommap(F, g)(xf) == fam[fg] || return false
        end
    end
    true
end

"""
    amalgamations(F::SetFunctor, R, fam) -> Vector{Any}

All amalgamations of the matching family `fam` for cover `R` on `c`: elements
`x ∈ F(c)` with `F(f)(x) = x_f` for every `f ∈ R`. (A sheaf has exactly one for
every matching family; a separated presheaf at most one.)
"""
function amalgamations(F::SetFunctor, R, fam)
    isempty(R) && throw(ArgumentError("amalgamations: empty cover has no domain object"))
    c = first(R).dom
    out = Any[]
    for x in ob(F, c).elements
        all(hommap(F, f)(x) == fam[f] for f in R) && push!(out, x)
    end
    out
end

"""
    is_separated(F::SetFunctor, J::Coverage) -> Bool

`F` is **separated** for the coverage `J`: every matching family (over every
cover at every object) has *at most one* amalgamation. Equivalently, the
restriction map `F(c) → ∏_{f∈R} F(cod f)` is injective for every cover `R`.
"""
function is_separated(F::SetFunctor, J::Coverage)
    F.cat === J.cat || F.cat == J.cat ||
        throw(ArgumentError("is_separated: F and J must be on the same category"))
    for c in objects(F.cat), R in covering_sieves(J, c)
        isempty(R) && continue
        for fam in matching_families(F, R)
            length(amalgamations(F, R, fam)) <= 1 || return false
        end
    end
    true
end

"""
    is_sheaf(F::SetFunctor, J::Coverage) -> Bool

`F` is a **sheaf** for the coverage `J`: every matching family (over every cover
at every object) has *exactly one* amalgamation. Equivalently, every restriction
map `F(c) → ∏_{f∈R} F(cod f)` restricts to a bijection onto the matching
families. Checked by exhaustive enumeration over the finite site.
"""
function is_sheaf(F::SetFunctor, J::Coverage)
    F.cat === J.cat || F.cat == J.cat ||
        throw(ArgumentError("is_sheaf: F and J must be on the same category"))
    for c in objects(F.cat), R in covering_sieves(J, c)
        isempty(R) && continue
        for fam in matching_families(F, R)
            length(amalgamations(F, R, fam)) == 1 || return false
        end
    end
    true
end

# ----------------------------------------------------------------------------
# Separated reflection (the first half of sheafification)
# ----------------------------------------------------------------------------

"""
    separated_reflection(F::SetFunctor, J::Coverage) -> (SetFunctor, CatNatTrans)

The **separated reflection** `F → F_s`: quotient each `F(c)` by the smallest
equivalence relation that identifies `x ~ y` whenever they agree after
restriction along *some* covering cosieve (`F(f)(x) = F(f)(y)` for all `f ∈ R`,
`R ∈ J(c)`). The resulting copresheaf is separated, and the quotient map is the
universal map to a separated presheaf. Returns `(F_s, η)` with `η : F ⇒ F_s` the
natural quotient.

This is the first plus-construction; applying it is enough to make `is_separated`
hold. (Full sheafification — the *second* plus-construction adding amalgamations
for matching families that lacked one — is left as future work; see the module
docstring.)
"""
function separated_reflection(F::SetFunctor, J::Coverage)
    C = F.cat
    # build the equivalence on each F(c)
    reps = Dict{Symbol, Dict{Any,Any}}()    # element ↦ canonical representative
    classes = Dict{Symbol, FinSet}()
    for c in objects(C)
        els = ob(F, c).elements
        parent = Dict{Any,Any}(x => x for x in els)
        find(x) = (parent[x] == x ? x : (parent[x] = find(parent[x])))
        union!(x, y) = (parent[find(x)] = find(y))
        for R in covering_sieves(J, c)
            isempty(R) && continue
            for i in eachindex(els), j in eachindex(els)
                i < j || continue
                x, y = els[i], els[j]
                if all(hommap(F, f)(x) == hommap(F, f)(y) for f in R)
                    union!(x, y)
                end
            end
        end
        rep = Dict{Any,Any}(x => find(x) for x in els)
        reps[c] = rep
        classes[c] = FinSet(unique(values(rep)))
    end
    # the quotient copresheaf: action on representatives
    ob_map = Dict{Symbol, FinSet}(c => classes[c] for c in objects(C))
    edge_map = Dict{Symbol, FinFunction}()
    for (n, c, d) in C.edges
        f = F.edge_map[n]
        m = Dict{Any,Any}()
        for x in classes[c].elements                       # x is a representative
            m[x] = reps[d][f(x)]
        end
        edge_map[n] = FinFunction(classes[c], classes[d], m)
    end
    Fs = SetFunctor(C; ob_map=ob_map, edge_map=edge_map)
    # the natural quotient η : F ⇒ Fs
    components = Dict{Symbol, FinFunction}()
    for c in objects(C)
        components[c] = FinFunction(ob(F, c), classes[c],
                                    Dict{Any,Any}(x => reps[c][x] for x in ob(F, c).elements))
    end
    η = CatNatTrans(F, Fs; components=components)
    (Fs, η)
end

# ----------------------------------------------------------------------------
# A concrete worked site: the span s ⇉ {a, b}
# ----------------------------------------------------------------------------

"""
    span_site() -> (FreeCat, Coverage)

A small worked **site**: the span category `a ← s → b` (objects `s, a, b`;
generators `p : s → a`, `q : s → b`), with the coverage whose only nontrivial
cover is the cosieve `R = {p, q}` on `s`. A copresheaf `F` is a sheaf for this
site iff `(F(p), F(q)) : F(s) → F(a) × F(b)` is a bijection — i.e. `F(s)` really
*is* the product glued from the two legs. See [`span_sheaf`](@ref) and
[`span_non_sheaf`](@ref).
"""
function span_site()
    C = FreeCat([:s, :a, :b], [(:p, :s, :a), (:q, :s, :b)])
    p = PathMor(:s, :a, [:p]); q = PathMor(:s, :b, [:q])
    R = sort([p, q]; by=_mor_key)               # cosieve {p, q} on s (closed: a,b are sinks)
    J = Coverage(C; covers=Dict(:s => [R]))
    (C, J)
end

"""
    span_sheaf() -> SetFunctor

A copresheaf on [`span_site`](@ref) that **is** a sheaf: `F(a)=F(b)={0,1}` and
`F(s) = F(a)×F(b)` with `p, q` the two projections — so the gluing map is a
bijection.
"""
function span_sheaf()
    C, _ = span_site()
    AB = [(0,0), (0,1), (1,0), (1,1)]
    Sset = FinSet(AB); Aset = FinSet([0,1]); Bset = FinSet([0,1])
    p = FinFunction(Sset, Aset, Dict{Any,Any}(t => t[1] for t in AB))
    q = FinFunction(Sset, Bset, Dict{Any,Any}(t => t[2] for t in AB))
    SetFunctor(C; ob_map=Dict(:s=>Sset, :a=>Aset, :b=>Bset),
                  edge_map=Dict(:p=>p, :q=>q))
end

"""
    span_non_sheaf() -> SetFunctor

A copresheaf on [`span_site`](@ref) that is **not** a sheaf: same legs
`F(a)=F(b)={0,1}`, but `F(s)` is only the diagonal `{(0,0),(1,1)}`. The gluing
map is injective (so `F` is separated) but not surjective — the matching family
`(x_p, x_q) = (0, 1)` has no amalgamation — hence `is_sheaf` fails.
"""
function span_non_sheaf()
    C, _ = span_site()
    diag = [(0,0), (1,1)]
    Sset = FinSet(diag); Aset = FinSet([0,1]); Bset = FinSet([0,1])
    p = FinFunction(Sset, Aset, Dict{Any,Any}(t => t[1] for t in diag))
    q = FinFunction(Sset, Bset, Dict{Any,Any}(t => t[2] for t in diag))
    SetFunctor(C; ob_map=Dict(:s=>Sset, :a=>Aset, :b=>Bset),
                  edge_map=Dict(:p=>p, :q=>q))
end
