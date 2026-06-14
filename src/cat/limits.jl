# ============================================================================
# limits.jl — limits and colimits in FinSet, with *verified* universal
# properties (included into module Cat)
#
# Each construction returns its (co)cone plus a mediating-morphism builder, and
# a `verify_*` that confirms the universal property by enumeration: for every
# probe test (co)cone there exists a unique mediating map making the relevant
# triangles commute. This is the genuine universal-property check the
# diagram-level constructions in `universal.jl` only assert structurally.
# ============================================================================

const _DEFAULT_PROBES = FinSet[FinSet(Int[]), FinSet([:pt]), FinSet([:pt1, :pt2])]

"""Count the FinFunctions `X → L` satisfying predicate `pred`."""
_count_mediators(X::FinSet, L::FinSet, pred) = count(pred, _all_functions(X, L))

# ----------------------------------------------------------------------------
# Product (limit of a discrete 2-object diagram)
# ----------------------------------------------------------------------------

struct ProductCone
    apex::FinSet
    proj1::FinFunction
    proj2::FinFunction
end

"""`product(A, B)` — the binary product `A × B` in FinSet with its projections."""
function product(A::FinSet, B::FinSet)
    elts = Any[(a, b) for a in A.elements for b in B.elements]
    P = FinSet(elts)
    π1 = FinFunction(P, A, Dict{Any,Any}(p => p[1] for p in elts))
    π2 = FinFunction(P, B, Dict{Any,Any}(p => p[2] for p in elts))
    ProductCone(P, π1, π2)
end

"""Mediating `⟨q1, q2⟩ : X → A×B` for a cone `(X, q1:X→A, q2:X→B)`."""
function mediate(pc::ProductCone, q1::FinFunction, q2::FinFunction)
    X = q1.dom
    FinFunction(X, pc.apex, Dict{Any,Any}(x => (q1(x), q2(x)) for x in X.elements))
end

"""Verify the product universal property against probe objects."""
function verify_product(pc::ProductCone, A::FinSet, B::FinSet; probes=_DEFAULT_PROBES)
    for X in probes
        for q1 in _all_functions(X, A), q2 in _all_functions(X, B)
            u = mediate(pc, q1, q2)
            (compose(u, pc.proj1) == q1 && compose(u, pc.proj2) == q2) || return false
            _count_mediators(X, pc.apex,
                v -> compose(v, pc.proj1) == q1 && compose(v, pc.proj2) == q2) == 1 || return false
        end
    end
    true
end

# ----------------------------------------------------------------------------
# Coproduct (colimit of a discrete 2-object diagram)
# ----------------------------------------------------------------------------

struct CoproductCocone
    apex::FinSet
    inj1::FinFunction
    inj2::FinFunction
end

"""`coproduct(A, B)` — the binary coproduct `A ⊔ B` with its injections (tagged)."""
function coproduct(A::FinSet, B::FinSet)
    elts = vcat(Any[(:inl, a) for a in A.elements], Any[(:inr, b) for b in B.elements])
    S = FinSet(elts)
    ι1 = FinFunction(A, S, Dict{Any,Any}(a => (:inl, a) for a in A.elements))
    ι2 = FinFunction(B, S, Dict{Any,Any}(b => (:inr, b) for b in B.elements))
    CoproductCocone(S, ι1, ι2)
end

"""Mediating `[q1, q2] : A⊔B → X` for a cocone `(X, q1:A→X, q2:B→X)`."""
function comediate(cc::CoproductCocone, q1::FinFunction, q2::FinFunction)
    X = q1.cod
    m = Dict{Any,Any}()
    for s in cc.apex.elements
        m[s] = s[1] === :inl ? q1(s[2]) : q2(s[2])
    end
    FinFunction(cc.apex, X, m)
end

"""Verify the coproduct universal property against probe objects."""
function verify_coproduct(cc::CoproductCocone, A::FinSet, B::FinSet; probes=_DEFAULT_PROBES)
    for X in probes
        for q1 in _all_functions(A, X), q2 in _all_functions(B, X)
            u = comediate(cc, q1, q2)
            (compose(cc.inj1, u) == q1 && compose(cc.inj2, u) == q2) || return false
            _count_mediators(cc.apex, X,
                v -> compose(cc.inj1, v) == q1 && compose(cc.inj2, v) == q2) == 1 || return false
        end
    end
    true
end

# ----------------------------------------------------------------------------
# Equalizer (limit of a parallel pair f, g : A ⇉ B)
# ----------------------------------------------------------------------------

struct EqualizerCone
    apex::FinSet
    incl::FinFunction
    f::FinFunction
    g::FinFunction
end

"""`equalizer(f, g)` — the equalizer `{a : f(a)=g(a)} ↪ A` of a parallel pair."""
function equalizer(f::FinFunction, g::FinFunction)
    (f.dom == g.dom && f.cod == g.cod) || throw(ArgumentError("equalizer needs a parallel pair"))
    elts = Any[a for a in f.dom.elements if f(a) == g(a)]
    E = FinSet(elts)
    incl = FinFunction(E, f.dom, Dict{Any,Any}(a => a for a in elts))
    EqualizerCone(E, incl, f, g)
end

"""Mediating `X → E` for `h : X → A` with `h·f = h·g`."""
function mediate(ec::EqualizerCone, h::FinFunction)
    compose(h, ec.f) == compose(h, ec.g) ||
        throw(ArgumentError("h does not equalize f and g"))
    FinFunction(h.dom, ec.apex, Dict{Any,Any}(x => h(x) for x in h.dom.elements))
end

"""Verify the equalizer universal property against probe objects."""
function verify_equalizer(ec::EqualizerCone; probes=_DEFAULT_PROBES)
    A = ec.f.dom
    for X in probes
        for h in _all_functions(X, A)
            compose(h, ec.f) == compose(h, ec.g) || continue   # only cones that equalize
            u = mediate(ec, h)
            compose(u, ec.incl) == h || return false
            _count_mediators(X, ec.apex, v -> compose(v, ec.incl) == h) == 1 || return false
        end
    end
    true
end

# ----------------------------------------------------------------------------
# Coequalizer (colimit of a parallel pair f, g : A ⇉ B)
# ----------------------------------------------------------------------------

struct CoequalizerCocone
    apex::FinSet
    proj::FinFunction
    f::FinFunction
    g::FinFunction
end

"""`coequalizer(f, g)` — the quotient `B ↠ B/∼` identifying `f(a) ∼ g(a)`."""
function coequalizer(f::FinFunction, g::FinFunction)
    (f.dom == g.dom && f.cod == g.cod) || throw(ArgumentError("coequalizer needs a parallel pair"))
    B = f.cod
    idx = Dict(b => i for (i, b) in enumerate(B.elements))
    parent = collect(1:length(B.elements))
    find(i) = (while parent[i] != i; parent[i] = parent[parent[i]]; i = parent[i]; end; i)
    for a in f.dom.elements
        i, j = find(idx[f(a)]), find(idx[g(a)])
        i == j || (parent[i] = j)
    end
    # canonical class representative = smallest-index element
    rep = Dict{Any,Any}()
    for b in B.elements
        r = find(idx[b])
        rep[b] = B.elements[r]
    end
    classes = unique(rep[b] for b in B.elements)
    Q = FinSet(classes)
    proj = FinFunction(B, Q, Dict{Any,Any}(b => rep[b] for b in B.elements))
    CoequalizerCocone(Q, proj, f, g)
end

"""Mediating `Q → X` for `h : B → X` with `f·h = g·h`."""
function comediate(cc::CoequalizerCocone, h::FinFunction)
    compose(cc.f, h) == compose(cc.g, h) ||
        throw(ArgumentError("h does not coequalize f and g"))
    # every class element maps consistently; use the representative's image
    m = Dict{Any,Any}()
    for b in cc.proj.dom.elements
        q = cc.proj(b)
        haskey(m, q) || (m[q] = h(b))
    end
    FinFunction(cc.apex, h.cod, m)
end

"""Verify the coequalizer universal property against probe objects."""
function verify_coequalizer(cc::CoequalizerCocone; probes=_DEFAULT_PROBES)
    B = cc.f.cod
    for X in probes
        for h in _all_functions(B, X)
            compose(cc.f, h) == compose(cc.g, h) || continue
            u = comediate(cc, h)
            compose(cc.proj, u) == h || return false
            _count_mediators(cc.apex, X, v -> compose(cc.proj, v) == h) == 1 || return false
        end
    end
    true
end

# ----------------------------------------------------------------------------
# Pullback (limit of a cospan A →f→ C ←g← B)
# ----------------------------------------------------------------------------

struct PullbackCone
    apex::FinSet
    p1::FinFunction
    p2::FinFunction
    f::FinFunction
    g::FinFunction
end

"""`pullback(f, g)` — the fibre product `{(a,b) : f(a)=g(b)}` of a cospan."""
function pullback(f::FinFunction, g::FinFunction)
    f.cod == g.cod || throw(ArgumentError("pullback needs a cospan (shared codomain)"))
    elts = Any[(a, b) for a in f.dom.elements for b in g.dom.elements if f(a) == g(b)]
    P = FinSet(elts)
    p1 = FinFunction(P, f.dom, Dict{Any,Any}(p => p[1] for p in elts))
    p2 = FinFunction(P, g.dom, Dict{Any,Any}(p => p[2] for p in elts))
    PullbackCone(P, p1, p2, f, g)
end

"""Mediating `X → P` for `(q1:X→A, q2:X→B)` with `q1·f = q2·g`."""
function mediate(pb::PullbackCone, q1::FinFunction, q2::FinFunction)
    compose(q1, pb.f) == compose(q2, pb.g) ||
        throw(ArgumentError("cone does not commute over the cospan"))
    FinFunction(q1.dom, pb.apex, Dict{Any,Any}(x => (q1(x), q2(x)) for x in q1.dom.elements))
end

"""Verify the pullback universal property against probe objects."""
function verify_pullback(pb::PullbackCone; probes=_DEFAULT_PROBES)
    A, B = pb.f.dom, pb.g.dom
    for X in probes
        for q1 in _all_functions(X, A), q2 in _all_functions(X, B)
            compose(q1, pb.f) == compose(q2, pb.g) || continue
            u = mediate(pb, q1, q2)
            (compose(u, pb.p1) == q1 && compose(u, pb.p2) == q2) || return false
            _count_mediators(X, pb.apex,
                v -> compose(v, pb.p1) == q1 && compose(v, pb.p2) == q2) == 1 || return false
        end
    end
    true
end

# ----------------------------------------------------------------------------
# Pushout (colimit of a span A ←f← C →g→ B)
# ----------------------------------------------------------------------------

struct PushoutCocone
    apex::FinSet
    i1::FinFunction
    i2::FinFunction
    f::FinFunction
    g::FinFunction
end

"""`pushout(f, g)` — the amalgam `A ⊔_C B` of a span (coproduct quotiented by `f(c) ∼ g(c)`)."""
function pushout(f::FinFunction, g::FinFunction)
    f.dom == g.dom || throw(ArgumentError("pushout needs a span (shared domain)"))
    A, B, C = f.cod, g.cod, f.dom
    cop = coproduct(A, B)
    # coequalize the two legs C → A⊔B
    cf = compose(f, cop.inj1)   # C → A⊔B via A
    cg = compose(g, cop.inj2)   # C → A⊔B via B
    coeq = coequalizer(cf, cg)
    apex = coeq.apex
    i1 = compose(cop.inj1, coeq.proj)
    i2 = compose(cop.inj2, coeq.proj)
    PushoutCocone(apex, i1, i2, f, g)
end

"""Mediating `P → X` for `(q1:A→X, q2:B→X)` with `f·q1 = g·q2`."""
function comediate(po::PushoutCocone, q1::FinFunction, q2::FinFunction)
    compose(po.f, q1) == compose(po.g, q2) ||
        throw(ArgumentError("cocone does not commute over the span"))
    m = Dict{Any,Any}()
    # each apex class has a preimage in A⊔B; assign via the injections
    for a in q1.dom.elements
        m[po.i1(a)] = q1(a)
    end
    for b in q2.dom.elements
        m[po.i2(b)] = q2(b)
    end
    FinFunction(po.apex, q1.cod, m)
end

"""Verify the pushout universal property against probe objects."""
function verify_pushout(po::PushoutCocone; probes=_DEFAULT_PROBES)
    A, B = po.f.cod, po.g.cod
    for X in probes
        for q1 in _all_functions(A, X), q2 in _all_functions(B, X)
            compose(po.f, q1) == compose(po.g, q2) || continue
            u = comediate(po, q1, q2)
            (compose(po.i1, u) == q1 && compose(po.i2, u) == q2) || return false
            _count_mediators(po.apex, X,
                v -> compose(po.i1, v) == q1 && compose(po.i2, v) == q2) == 1 || return false
        end
    end
    true
end
