# ============================================================================
# poly.jl — polynomial functors (Spivak's Poly): interfaces & dynamics
# (included into module Cat)
#
# A polynomial functor `p = Σ_{i ∈ positions} y^{directions(i)}` models an
# interactive interface: positions are outputs/observations, directions are the
# admissible inputs at each position. Poly morphisms are **dependent lenses**
# (forward on positions, backward on directions) — so a wiring/behaviour is a
# poly map, and a **dynamical system (Moore machine) is a lens `Sy^S → p`**
# (Spivak). This unifies the lens (`optics.jl`) and coalgebra (`coalg.jl`) stories.
# ============================================================================

"""
    Poly(positions, directions)

A polynomial functor `Σ_{i} y^{directions(i)}`: a finite set of `positions`,
each with a finite set of `directions`.
"""
struct Poly
    positions::Vector{Any}
    directions::Dict{Any, Vector{Any}}
    function Poly(positions, directions)
        pos = collect(positions)
        dir = Dict{Any, Vector{Any}}(p => collect(directions[p]) for p in pos)
        new(pos, dir)
    end
end

"""`monomial(S)` — the linear/monomial polynomial `S·y^S` (the state interface of `S`)."""
monomial(S) = Poly(collect(S), Dict{Any, Vector{Any}}(s => collect(S) for s in S))

"""
    PolyMap(dom, cod, on_pos, on_dir)

A morphism of polynomials = a dependent lens: `on_pos : positions(dom) →
positions(cod)` (forward) and, for each `i`, `on_dir[i] : directions(cod)(on_pos i)
→ directions(dom)(i)` (backward).
"""
struct PolyMap
    dom::Poly
    cod::Poly
    on_pos::Dict{Any, Any}
    on_dir::Dict{Any, Dict{Any, Any}}
end

"""`is_poly_morphism(φ)` — the forward/backward maps are total and well-typed."""
function is_poly_morphism(φ::PolyMap)
    for i in φ.dom.positions
        haskey(φ.on_pos, i) || return false
        j = φ.on_pos[i]
        j in φ.cod.positions || return false
        m = get(φ.on_dir, i, Dict{Any,Any}())
        for d in φ.cod.directions[j]
            haskey(m, d) || return false
            m[d] in φ.dom.directions[i] || return false
        end
    end
    true
end

"""`poly_id(p)` — the identity poly morphism."""
poly_id(p::Poly) = PolyMap(p, p,
    Dict{Any,Any}(i => i for i in p.positions),
    Dict{Any,Dict{Any,Any}}(i => Dict{Any,Any}(d => d for d in p.directions[i]) for i in p.positions))

"""`poly_compose(φ, ψ)` — composition of dependent lenses `φ : p→q`, `ψ : q→r`."""
function poly_compose(φ::PolyMap, ψ::PolyMap)
    on_pos = Dict{Any,Any}(i => ψ.on_pos[φ.on_pos[i]] for i in φ.dom.positions)
    on_dir = Dict{Any,Dict{Any,Any}}()
    for i in φ.dom.positions
        j = φ.on_pos[i]; k = ψ.on_pos[j]
        # r-direction at k → (ψ backward) q-direction at j → (φ backward) p-direction at i
        on_dir[i] = Dict{Any,Any}(d => φ.on_dir[i][ψ.on_dir[j][d]] for d in ψ.cod.directions[k])
    end
    PolyMap(φ.dom, ψ.cod, on_pos, on_dir)
end

"""
    moore_to_poly(M::MooreMachine) -> PolyMap

A Moore machine *is* a polynomial coalgebra / lens `S·y^S → O·y^I`: the readout
`state ↦ output` (forward on positions) and the dynamics `(state, input) ↦
next state` (backward on directions). Realises the coalgebra of `coalg.jl` as a
`Poly` morphism, unifying automata, lenses, and dynamical systems.
"""
function moore_to_poly(M::MooreMachine)
    S = monomial(M.states)                      # state interface  S·y^S
    interface = Poly(M.outputs, Dict{Any, Vector{Any}}(o => collect(M.inputs) for o in M.outputs))  # O·y^I
    on_pos = Dict{Any,Any}(s => M.output[s] for s in M.states)          # readout
    on_dir = Dict{Any,Dict{Any,Any}}(
        s => Dict{Any,Any}(i => moore_step(M, s, i) for i in M.inputs)   # dynamics: (s,i) ↦ next state
        for s in M.states)
    PolyMap(S, interface, on_pos, on_dir)
end
