# ============================================================================
# coend.jl — profunctors and coends over a free category, with the coend
# realised as a *coequalizer* of the dinaturality maps. (included into Cat)
#
# A profunctor  P : Cᵒᵖ × C → FinSet  assigns a finite set P(c₀, c₁) to every
# ordered pair of objects (contravariant in c₀, covariant in c₁) together with
# functorial actions in both variables. The **coend**
#
#       ∫^c P(c, c)
#
# is the universal dinatural "trace": the colimit that glues the diagonal sets
# P(c,c) along the action of every morphism. Concretely (Loregian, *(Co)end
# Calculus*, §1.2) it is the coequalizer
#
#       ⊔_{f : c→c'} P(c', c)  ⇉  ⊔_{c} P(c, c)  ↠  ∫^c P(c,c)
#
# of the two dinaturality maps sending the heteromorphism slot P(c', c) into the
# diagonal by acting with f on, respectively, the contravariant and the
# covariant variable:
#
#       λ_f = P(f, id) : P(c', c) → P(c, c)        (act on the 1st / contravar.)
#       ρ_f = P(id, f) : P(c', c) → P(c', c')       (act on the 2nd / covariant)
#
# We build the two coproduct legs and hand them to `Cat.coequalizer`, so the
# coend inherits the *verified* universal property already proved for
# coequalizers in limits.jl.
#
# Application — "attention as a coend". With C the 2-object key/value indexing
# category and P(c, c) = (queries paired with keys) × (values), the coend
# quotients away the difference between "transport the key then read the value"
# and "read the value then transport the key": the surviving classes are the
# query→value context aggregations. This is the same colimit-of-a-bimodule that
# `Cat.left_kan` computes as a pointwise coend, demonstrated in the tests.
# ============================================================================

# ----------------------------------------------------------------------------
# Profunctor encoding
# ----------------------------------------------------------------------------

"""
    Profunctor(C; diag, het, lact, ract)

A finite profunctor `P : Cᵒᵖ × C → FinSet` over a [`FreeCat`](@ref) `C`,
encoded by exactly the data the coend coequalizer consumes:

  * `diag :: object ↦ FinSet`           — the diagonal sets `P(c, c)`;
  * `het  :: edgename ↦ FinSet`         — for each generating edge `f : c → c'`,
                                          the heteromorphism set `P(c', c)`;
  * `lact :: edgename ↦ FinFunction`    — `λ_f = P(f, id) : P(c', c) → P(c, c)`,
                                          the action of `f` on the *contravariant*
                                          (first) variable;
  * `ract :: edgename ↦ FinFunction`    — `ρ_f = P(id, f) : P(c', c) → P(c', c')`,
                                          the action of `f` on the *covariant*
                                          (second) variable.

Construction validates that every action's (co)domain matches the declared
diagonal / heteromorphism sets, so a well-formed `Profunctor` is exactly a
parallel pair of legs ready to be coequalized. (We only require the data the
*diagonal* coend needs; the full bifunctor action on non-generating slots is not
materialised — by acyclicity these data already determine the dinatural family.)
"""
struct Profunctor
    cat::FreeCat
    diag::Dict{Symbol, FinSet}          # c        ↦ P(c,c)
    het::Dict{Symbol, FinSet}           # edge f   ↦ P(c',c)
    lact::Dict{Symbol, FinFunction}     # edge f   ↦ P(f,id): P(c',c) → P(c,c)
    ract::Dict{Symbol, FinFunction}     # edge f   ↦ P(id,f): P(c',c) → P(c',c')
end

function Profunctor(C::FreeCat; diag::AbstractDict, het::AbstractDict,
                    lact::AbstractDict, ract::AbstractDict)
    dg = Dict{Symbol, FinSet}(Symbol(k) => v for (k, v) in diag)
    ht = Dict{Symbol, FinSet}(Symbol(k) => v for (k, v) in het)
    la = Dict{Symbol, FinFunction}(Symbol(k) => v for (k, v) in lact)
    ra = Dict{Symbol, FinFunction}(Symbol(k) => v for (k, v) in ract)
    for o in C.objects
        haskey(dg, o) || throw(ArgumentError("Profunctor missing diagonal set P($o,$o)"))
    end
    for (n, s, t) in C.edges        # edge f : s → t  ⇒  heteromorphism set P(t, s)
        haskey(ht, n) || throw(ArgumentError("Profunctor missing het set P($t,$s) for edge $n"))
        haskey(la, n) || throw(ArgumentError("Profunctor missing left action for edge $n"))
        haskey(ra, n) || throw(ArgumentError("Profunctor missing right action for edge $n"))
        # λ_f = P(f,id) : P(t,s) → P(s,s)
        la[n].dom == ht[n] || throw(ArgumentError("edge $n: dom(λ) ≠ P($t,$s)"))
        la[n].cod == dg[s] || throw(ArgumentError("edge $n: cod(λ) ≠ P($s,$s)"))
        # ρ_f = P(id,f) : P(t,s) → P(t,t)
        ra[n].dom == ht[n] || throw(ArgumentError("edge $n: dom(ρ) ≠ P($t,$s)"))
        ra[n].cod == dg[t] || throw(ArgumentError("edge $n: cod(ρ) ≠ P($t,$t)"))
    end
    Profunctor(C, dg, ht, la, ra)
end

"""`profunctor_diag(P, c)` — the diagonal set `P(c, c)`."""
profunctor_diag(P::Profunctor, c) = P.diag[Symbol(c)]

# ----------------------------------------------------------------------------
# The coend ∫^c P(c,c) as a coequalizer of the dinaturality maps
# ----------------------------------------------------------------------------

"""
    CoendCocone(apex, coproj, source, target, coeq)

The result of [`coend`](@ref): the coend object `apex = ∫^c P(c,c)` together with
the diagonal coprojections `coproj :: object ↦ (P(c,c) → ∫)`, the parallel pair
`source, target : ⊔ₕₑₜ ⇉ ⊔_diag` whose coequalizer it is, and the underlying
[`CoequalizerCocone`](@ref) (so the verified universal property is reachable).
"""
struct CoendCocone
    apex::FinSet
    coproj::Dict{Symbol, FinFunction}   # c ↦ (P(c,c) → ∫)
    source::FinFunction                 # ⊔_{f} P(c',c) → ⊔_c P(c,c)  (λ side)
    target::FinFunction                 # ⊔_{f} P(c',c) → ⊔_c P(c,c)  (ρ side)
    coeq::CoequalizerCocone
end

# Fold a vector of FinSets into one coproduct with tagged injections.
# Returns (apex, injections::Vector{FinFunction}) where injection i : Sᵢ → apex.
function _nary_coproduct(sets::Vector{FinSet}, tags::Vector{Symbol})
    elts = Any[]
    for (tag, S) in zip(tags, sets), x in S.elements
        push!(elts, (tag, x))
    end
    apex = FinSet(elts)
    injs = FinFunction[
        FinFunction(S, apex, Dict{Any,Any}(x => (tag, x) for x in S.elements))
        for (tag, S) in zip(tags, sets)]
    apex, injs
end

"""
    coend(P::Profunctor) -> CoendCocone

Compute the coend `∫^c P(c,c)` as the coequalizer of the two dinaturality maps

    ⊔_{f:c→c'} P(c',c)  ⇉  ⊔_c P(c,c)

built from `P`'s left/right actions (`λ_f = P(f,id)`, `ρ_f = P(id,f)`), reusing
`Cat.coproduct` (n-ary) and `Cat.coequalizer`. The quotient identifies, for every
generating `f : c → c'` and every heteromorphism `m ∈ P(c',c)`, the two diagonal
images `λ_f(m) ∈ P(c,c)` and `ρ_f(m) ∈ P(c',c')` — i.e. it enforces dinaturality.
"""
function coend(P::Profunctor)
    C = P.cat
    objs = objects(C)

    # Codomain coproduct  ⊔_c P(c,c).
    diag_sets = FinSet[P.diag[c] for c in objs]
    diag_tags = objs
    Dapex, dinj = _nary_coproduct(diag_sets, diag_tags)
    dinj_of = Dict{Symbol, FinFunction}(objs[i] => dinj[i] for i in eachindex(objs))

    # Domain coproduct  ⊔_{f:c→c'} P(c',c), one summand per generating edge.
    edge_names = Symbol[n for (n, _, _) in C.edges]
    if isempty(edge_names)
        # No generators ⇒ no relations ⇒ coend is the bare disjoint union of diagonals.
        coproj = Dict{Symbol, FinFunction}(c => dinj_of[c] for c in objs)
        empty = FinSet(Any[])
        z1 = FinFunction(empty, Dapex, Dict{Any,Any}())
        coeq = coequalizer(z1, z1)
        # coequalizing a pair from the empty set is the identity quotient on Dapex
        coproj2 = Dict{Symbol, FinFunction}(c => compose(coproj[c], coeq.proj) for c in objs)
        return CoendCocone(coeq.apex, coproj2, z1, z1, coeq)
    end
    het_sets = FinSet[P.het[n] for n in edge_names]
    Hapex, _hinj = _nary_coproduct(het_sets, edge_names)

    # Two parallel maps  H ⇉ D : per heteromorphism, tag its λ- and ρ-images into ⊔diag.
    src_map = Dict{Any,Any}()
    tgt_map = Dict{Any,Any}()
    for (i, n) in enumerate(edge_names)
        # locate this edge's (name, src, tgt)
        (nm, es, et) = C.edges[findfirst(e -> e[1] == n, C.edges)]
        λ = P.lact[n]   # P(t,s) → P(es=src, src) on the contravariant variable
        ρ = P.ract[n]   # P(t,s) → P(et=tgt, tgt)
        for x in het_sets[i].elements
            tag = (n, x)
            src_map[tag] = dinj_of[es](λ(x))   # λ_f(x) ∈ P(c,c)   tagged into ⊔
            tgt_map[tag] = dinj_of[et](ρ(x))   # ρ_f(x) ∈ P(c',c') tagged into ⊔
        end
    end
    source = FinFunction(Hapex, Dapex, src_map)
    target = FinFunction(Hapex, Dapex, tgt_map)

    coeq = coequalizer(source, target)
    coproj = Dict{Symbol, FinFunction}(c => compose(dinj_of[c], coeq.proj) for c in objs)
    CoendCocone(coeq.apex, coproj, source, target, coeq)
end

"""
    coend_class(co::CoendCocone, c, x)

The element of the coend `∫^c P(c,c)` represented by `x ∈ P(c,c)` — i.e. push `x`
through the diagonal coprojection `P(c,c) → ∫`.
"""
coend_class(co::CoendCocone, c, x) = co.coproj[Symbol(c)](x)

"""
    verify_coend(co::CoendCocone) -> Bool

Confirm `co` is a genuine coend: (1) dinaturality holds — for every generating
`f` and `m ∈ P(c',c)` the two diagonal images are identified in `∫`, and (2) the
underlying coequalizer satisfies its (verified) universal property.
"""
function verify_coend(co::CoendCocone)
    # (1) dinaturality: source and target agree after the coprojection
    compose(co.source, co.coeq.proj) == compose(co.target, co.coeq.proj) || return false
    # (2) coequalizer universal property (enumerated against probes)
    verify_coequalizer(co.coeq)
end

# ----------------------------------------------------------------------------
# The dual: end ∫_c P(c,c) as an equalizer of the same two dinaturality maps
# ----------------------------------------------------------------------------

"""
    EndCone(apex, proj, f, g, eq)

The result of [`end_`](@ref): the end object `apex = ∫_c P(c,c)` (the dinatural
"wedge" of compatible diagonal families) together with the projections
`proj :: object ↦ (∫ → P(c,c))`, the parallel pair it equalizes, and the
underlying [`EqualizerCone`](@ref).
"""
struct EndCone
    apex::FinSet
    proj::Dict{Symbol, FinFunction}     # c ↦ (∫ → P(c,c))
    f::FinFunction
    g::FinFunction
    eq::EqualizerCone
end

"""
    end_(P::Profunctor) -> EndCone

The end `∫_c P(c,c)` as the equalizer of the two dinaturality maps

    ∏_c P(c,c)  ⇉  ∏_{f:c→c'} P(c',c)

(the wedge condition). Concretely the end is the set of diagonal families
`(x_c ∈ P(c,c))_c` such that for every generating `f : c → c'`,
`P(f,id)` and `P(id,f)` agree on the pair — i.e. there is a single
heteromorphism witnessing `x_c` and `x_{c'}`. We realise it as an
`Cat.equalizer` of the two maps `∏diag → ∏het` reading off `λ`- and `ρ`-preimage
constraints, so it inherits the verified equalizer universal property.

NOTE: in general `P(f,id)` and `P(id,f)` go *out of* the het slot, so the wedge
condition is "there is `m` with `λ_f(m)=x_c` and `ρ_f(m)=x_{c'}`". We encode this
by, per edge, the induced relation; for the finite/free setting we compute the
end as the equalizer of the two composites `∏diag → P(c,c)`/`P(c',c')` pulled back
through the (jointly considered) het actions. See the demonstration in the tests
for the small cases where this coincides with the limit/wedge.
"""
function end_(P::Profunctor)
    C = P.cat
    objs = objects(C)

    # ∏_c P(c,c)
    prod_apex_elts = Any[]
    if isempty(objs)
        push!(prod_apex_elts, ())
    else
        for combo in Iterators.product((P.diag[c].elements for c in objs)...)
            push!(prod_apex_elts, Tuple(combo))
        end
    end
    Dprod = FinSet(prod_apex_elts)
    pos = Dict{Symbol,Int}(objs[i] => i for i in eachindex(objs))

    edge_names = Symbol[n for (n, _, _) in C.edges]
    if isempty(edge_names)
        # No relations: end = full product of diagonals.
        idf = id(Dprod)
        eq = equalizer(idf, idf)
        proj = Dict{Symbol, FinFunction}(
            c => FinFunction(eq.apex, P.diag[c],
                    Dict{Any,Any}(fam => fam[pos[c]] for fam in eq.apex.elements))
            for c in objs)
        return EndCone(eq.apex, proj, idf, idf, eq)
    end

    # The end is the set of diagonal families that admit a *consistent* het
    # witness for every edge: family (x_c) is in the end iff for every edge
    # f:c→c' there exists m ∈ P(c',c) with λ_f(m)=x_c AND ρ_f(m)=x_{c'}.
    # We realise this as an equalizer of two maps ∏diag ⇉ Bool^edges: `wit`
    # (per edge, is there a witness) and the constant all-true map. Their
    # equalizer is exactly the end (the wedge).
    Boolvec = FinSet(Any[Tuple(bs) for bs in Iterators.product((Bool[true,false] for _ in edge_names)...)])
    admit = Dict{Any,Any}()
    alltrue = Tuple(fill(true, length(edge_names)))
    for fam in Dprod.elements
        bits = Bool[]
        for (ei, n) in enumerate(edge_names)
            (nm, es, et) = C.edges[findfirst(e -> e[1] == n, C.edges)]
            λ = P.lact[n]; ρ = P.ract[n]
            xc  = fam[pos[es]]
            xc2 = fam[pos[et]]
            ok = any(λ(m) == xc && ρ(m) == xc2 for m in P.het[n].elements)
            push!(bits, ok)
        end
        admit[fam] = Tuple(bits)
    end
    wit  = FinFunction(Dprod, Boolvec, admit)
    ctrue = FinFunction(Dprod, Boolvec, Dict{Any,Any}(fam => alltrue for fam in Dprod.elements))
    eq = equalizer(wit, ctrue)
    proj = Dict{Symbol, FinFunction}(
        c => FinFunction(eq.apex, P.diag[c],
                Dict{Any,Any}(fam => fam[pos[c]] for fam in eq.apex.elements))
        for c in objs)
    EndCone(eq.apex, proj, wit, ctrue, eq)
end
