# ============================================================================
# coalg.jl — coalgebras: state machines (and RNNs) as F-coalgebras
# (included into module Cat)
#
# A Moore machine is a coalgebra for the behaviour functor `F(X) = O × X^I`
# (an output plus, for each input, a next state). This is the categorical home
# of state machines and recurrent models: behaviour is the unique map into the
# final coalgebra, and bisimilarity is behavioural equivalence. Everything is
# finite, so bisimulation, minimisation, and homomorphism are computable — and
# Lean-certifiable (see `Automata.lean`).
# ============================================================================

"""
    MooreMachine(states, inputs, outputs, transition, output)

A coalgebra `s ↦ (output(s), i ↦ transition(s,i))` for `F(X) = O × X^I`:
`transition[(s,i)]` is the next state and `output[s]` the observed output.
"""
struct MooreMachine
    states::Vector{Symbol}
    inputs::Vector{Symbol}
    outputs::Vector{Symbol}
    transition::Dict{Tuple{Symbol,Symbol}, Symbol}
    output::Dict{Symbol, Symbol}
    function MooreMachine(states, inputs, outputs, transition, output)
        ss = Symbol.(states); is = Symbol.(inputs); os = Symbol.(outputs)
        sset = Set(ss)
        δ = Dict{Tuple{Symbol,Symbol}, Symbol}((Symbol(s), Symbol(i)) => Symbol(t) for ((s, i), t) in transition)
        o = Dict{Symbol, Symbol}(Symbol(s) => Symbol(v) for (s, v) in output)
        for s in ss
            haskey(o, s) || throw(ArgumentError("no output for state $s"))
            o[s] in Set(os) || throw(ArgumentError("output $(o[s]) of $s not in output alphabet"))
            for i in is
                haskey(δ, (s, i)) || throw(ArgumentError("no transition for ($s, $i)"))
                δ[(s, i)] in sset || throw(ArgumentError("transition ($s,$i) leaves the state set"))
            end
        end
        new(ss, is, os, δ, o)
    end
end

"""`moore_step(M, s, i)` — the next state from `s` on input `i`."""
moore_step(M::MooreMachine, s, i) = M.transition[(Symbol(s), Symbol(i))]

"""`moore_run(M, s, word)` — outputs observed while reading `word` from `s` (incl. the start output)."""
function moore_run(M::MooreMachine, s, word)
    cur = Symbol(s); out = Symbol[M.output[cur]]
    for i in word
        cur = moore_step(M, cur, i)
        push!(out, M.output[cur])
    end
    out
end

"""
    is_bisimulation(M, R) -> Bool

Is `R` (a set/collection of state pairs) a bisimulation: related states share
an output and their successors stay related for every input.
"""
function is_bisimulation(M::MooreMachine, R)
    Rset = Set((Symbol(a), Symbol(b)) for (a, b) in R)
    for (s, t) in Rset
        M.output[s] == M.output[t] || return false
        for i in M.inputs
            (moore_step(M, s, i), moore_step(M, t, i)) in Rset || return false
        end
    end
    true
end

"""
    bisimilar(M) -> Dict{Symbol, Int}

The largest bisimulation (coarsest stable partition), as a map state ↦ class id.
Two states are behaviourally equivalent iff they share a class. (Partition
refinement = Moore's minimisation algorithm.)
"""
function bisimilar(M::MooreMachine)
    block = Dict{Symbol, Int}()
    labels = Dict{Any, Int}()
    for s in M.states
        block[s] = get!(labels, M.output[s], length(labels) + 1)
    end
    while true
        labels = Dict{Any, Int}()
        newblock = Dict{Symbol, Int}()
        for s in M.states
            sig = (block[s], Tuple(block[moore_step(M, s, i)] for i in M.inputs))
            newblock[s] = get!(labels, sig, length(labels) + 1)
        end
        length(unique(values(newblock))) == length(unique(values(block))) && return newblock
        block = newblock
    end
end

"""
    minimize(M) -> MooreMachine

The minimal machine: the quotient by bisimilarity — i.e. the image of `M` in
the final coalgebra. Behaviourally equivalent to `M` with no redundant states.
"""
function minimize(M::MooreMachine)
    cls = bisimilar(M)
    rep = Dict{Int, Symbol}()
    for s in M.states
        haskey(rep, cls[s]) || (rep[cls[s]] = s)   # first state in each class is its rep
    end
    newstates = [Symbol("q", c) for c in sort(collect(keys(rep)))]
    name(c) = Symbol("q", c)
    transition = Dict{Tuple{Symbol,Symbol}, Symbol}()
    output = Dict{Symbol, Symbol}()
    for (c, r) in rep
        output[name(c)] = M.output[r]
        for i in M.inputs
            transition[(name(c), i)] = name(cls[moore_step(M, r, i)])
        end
    end
    MooreMachine(newstates, M.inputs, M.outputs, transition, output)
end

"""
    coalgebra_morphism(M, N, h) -> Bool

Is `h : states(M) → states(N)` a coalgebra homomorphism: it preserves outputs
(`output_N(h s) = output_M(s)`) and commutes with transitions
(`h(δ_M(s,i)) = δ_N(h s, i)`).
"""
function coalgebra_morphism(M::MooreMachine, N::MooreMachine, h::AbstractDict)
    M.inputs == N.inputs || return false
    hd = Dict{Symbol, Symbol}(Symbol(k) => Symbol(v) for (k, v) in h)
    for s in M.states
        haskey(hd, s) || return false
        N.output[hd[s]] == M.output[s] || return false
        for i in M.inputs
            hd[moore_step(M, s, i)] == moore_step(N, hd[s], i) || return false
        end
    end
    true
end
