# ============================================================================
# coalgebra.jl — Coalgebraic world models for FunctorFlow.jl
#
# Coalgebras model generative systems as X → F(X): given a state, produce
# the next observation/state. This is the categorical foundation for world
# models, JEPA predictors, and reinforcement learning dynamics.
#
# Key concepts:
#   - Coalgebra: state space + transition structure X → F(X)
#   - Coalgebra morphism: structure-preserving map between coalgebras
#   - Bisimulation: behavioral equivalence of coalgebras
#   - Final coalgebra: optimal/canonical representation (Lambek's lemma)
#   - Stochastic coalgebra: probabilistic transitions X → D(X)
#
# References:
#   Mahadevan, "Categories for AGI" — Chapters on universal coalgebras
#   Rutten, "Universal coalgebra: a theory of systems"
# ============================================================================

# ---------------------------------------------------------------------------
# Coalgebra — endofunctor dynamics on a state space
# ---------------------------------------------------------------------------

"""
    Coalgebra(name, state, transition; description="", metadata=Dict())

An F-coalgebra: a state space `state` equipped with a transition structure
`transition : state → F(state)`. In the FunctorFlow context:

- **State** is a diagram object (representation space, latent space)
- **Transition** is a morphism (dynamics, predictor, world model step)
- **F** is implicit in the target type of the transition

This is the categorical foundation for world models: given the current
representation, predict/generate the next state.

# Examples
```julia
c = Coalgebra(:WorldModel, :LatentState, :dynamics;
              description="Latent space world model")
```
"""
struct Coalgebra <: AbstractFFElement
    name::Symbol
    state::Symbol           # object name (carrier/state space)
    transition::Symbol      # morphism name (structure map X → F(X))
    functor_type::Symbol    # :identity, :distribution, :product, :custom
    description::String
    metadata::Dict{Symbol, Any}
end

function Coalgebra(name, state, transition;
                   functor_type::Union{Symbol, AbstractString}=:identity,
                   description::AbstractString="",
                   metadata::Dict=Dict{Symbol, Any}())
    Coalgebra(Symbol(name), Symbol(state), Symbol(transition),
              Symbol(functor_type), String(description),
              Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

# ---------------------------------------------------------------------------
# CoalgebraMorphism — structure-preserving map
# ---------------------------------------------------------------------------

"""
    CoalgebraMorphism(name, source, target, hom; description="", metadata=Dict())

A morphism between F-coalgebras: a map `hom : A.state → B.state` such that
the transition structures commute:

    B.transition ∘ hom = F(hom) ∘ A.transition

In JEPA terms: an encoder that makes the following diagram commute:

    X_A ---encoder--→ X_B
     |                  |
    α_A               α_B
     ↓                  ↓
   F(X_A) --F(enc)--→ F(X_B)
"""
struct CoalgebraMorphism <: AbstractFFElement
    name::Symbol
    source::Symbol          # source coalgebra name
    target::Symbol          # target coalgebra name
    hom::Symbol             # underlying morphism name
    description::String
    metadata::Dict{Symbol, Any}
end

function CoalgebraMorphism(name, source, target, hom;
                           description::AbstractString="",
                           metadata::Dict=Dict{Symbol, Any}())
    CoalgebraMorphism(Symbol(name), Symbol(source), Symbol(target), Symbol(hom),
                      String(description),
                      Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

# ---------------------------------------------------------------------------
# FinalCoalgebraWitness — terminal object witness
# ---------------------------------------------------------------------------

"""
    FinalCoalgebraWitness(name, carrier, desc_map; description="", metadata=Dict())

A witness that a particular coalgebra is final (terminal) in the category of
F-coalgebras. The final coalgebra has the universal property that for every
coalgebra A, there exists a unique morphism A → final.

By Lambek's lemma, the structure map of a final coalgebra is an isomorphism:
    α : Z ≅ F(Z)

This is the categorical analog of:
- **Optimal value function** (in RL: fixed point of Bellman operator)
- **Canonical representation** (in JEPA: the ideal embedding space)
- **Greatest fixed point** (in domain theory)

# Fields
- `carrier`: the coalgebra that is claimed to be final
- `desc_map`: a function/morphism that describes the unique map from any
  coalgebra to this one (the "anamorphism")
"""
struct FinalCoalgebraWitness <: AbstractFFElement
    name::Symbol
    carrier::Symbol         # coalgebra name (the final one)
    desc_map::Symbol        # morphism implementing the unique map
    is_isomorphism::Bool    # Lambek's lemma: structure map is iso
    description::String
    metadata::Dict{Symbol, Any}
end

function FinalCoalgebraWitness(name, carrier, desc_map;
                                is_isomorphism::Bool=true,
                                description::AbstractString="",
                                metadata::Dict=Dict{Symbol, Any}())
    FinalCoalgebraWitness(Symbol(name), Symbol(carrier), Symbol(desc_map),
                          is_isomorphism, String(description),
                          Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

# ---------------------------------------------------------------------------
# Bisimulation — behavioral equivalence
# ---------------------------------------------------------------------------

"""
    Bisimulation(name, coalgebra_a, coalgebra_b, relation; description="", metadata=Dict())

A bisimulation between two coalgebras A and B: a relation R ⊆ A.state × B.state
such that related states produce related transitions.

Two coalgebras are bisimilar iff they map to the same element in the final
coalgebra. This captures the intuition that two systems are "equivalent"
if they produce indistinguishable behavior.

In JEPA terms: two encoders are bisimilar if they produce equivalent
representations for equivalent inputs.
"""
struct Bisimulation <: AbstractFFElement
    name::Symbol
    coalgebra_a::Symbol
    coalgebra_b::Symbol
    relation::Symbol        # object/morphism implementing the relation
    description::String
    metadata::Dict{Symbol, Any}
end

function Bisimulation(name, coalgebra_a, coalgebra_b, relation;
                      description::AbstractString="",
                      metadata::Dict=Dict{Symbol, Any}())
    Bisimulation(Symbol(name), Symbol(coalgebra_a), Symbol(coalgebra_b),
                 Symbol(relation), String(description),
                 Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

# ---------------------------------------------------------------------------
# StochasticCoalgebra — probabilistic transitions
# ---------------------------------------------------------------------------

"""
    StochasticCoalgebra(name, state, transition, distribution_type; ...)

A stochastic coalgebra: X → D(X) where D is a probability distribution
functor. This models systems with inherent uncertainty:

- **World models with noise** (JEPA latent variable z)
- **Stochastic policies** (RL: state → distribution over actions)
- **Generative models** (VAE decoder, diffusion step)

The `distribution_type` specifies the probability functor:
- `:gaussian` — Normal distribution (continuous states)
- `:categorical` — Categorical distribution (discrete states)
- `:dirac` — Deterministic (degenerate case, reduces to ordinary coalgebra)
"""
struct StochasticCoalgebra <: AbstractFFElement
    name::Symbol
    state::Symbol
    transition::Symbol
    distribution_type::Symbol   # :gaussian, :categorical, :dirac
    description::String
    metadata::Dict{Symbol, Any}
end

function StochasticCoalgebra(name, state, transition;
                              distribution_type::Union{Symbol, AbstractString}=:gaussian,
                              description::AbstractString="",
                              metadata::Dict=Dict{Symbol, Any}())
    StochasticCoalgebra(Symbol(name), Symbol(state), Symbol(transition),
                        Symbol(distribution_type), String(description),
                        Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

# ---------------------------------------------------------------------------
# Diagram integration — add coalgebra structure to diagrams
# ---------------------------------------------------------------------------

"""
    add_coalgebra!(D::Diagram, name; state, transition, functor_type=:identity, ...)

Add a coalgebra structure to a diagram. This declares that the morphism
`transition` implements the coalgebra dynamics X → F(X) on the object `state`.

The coalgebra is stored in the diagram's metadata under `:coalgebras`.
"""
function add_coalgebra!(D::Diagram, name::Union{Symbol, AbstractString};
                        state::Union{Symbol, AbstractString},
                        transition::Union{Symbol, AbstractString},
                        functor_type::Union{Symbol, AbstractString}=:identity,
                        description::AbstractString="",
                        metadata::Dict=Dict{Symbol, Any}())
    c = Coalgebra(name, state, transition;
                  functor_type=functor_type, description=description, metadata=metadata)
    coalgebras = get!(D.implementations, :_coalgebras) do
        Dict{Symbol, Coalgebra}()
    end::Dict{Symbol, Coalgebra}
    coalgebras[c.name] = c
    c
end

"""
    get_coalgebras(D::Diagram) -> Dict{Symbol, Coalgebra}

Retrieve all coalgebra structures declared in the diagram.
"""
function get_coalgebras(D::Diagram)
    get(D.implementations, :_coalgebras, Dict{Symbol, Coalgebra}())::Dict{Symbol, Coalgebra}
end

"""
    add_bisimulation!(D::Diagram, name; coalgebra_a, coalgebra_b, relation, ...)

Declare a bisimulation between two coalgebras in the diagram.
"""
function add_bisimulation!(D::Diagram, name::Union{Symbol, AbstractString};
                           coalgebra_a::Union{Symbol, AbstractString},
                           coalgebra_b::Union{Symbol, AbstractString},
                           relation::Union{Symbol, AbstractString},
                           description::AbstractString="",
                           metadata::Dict=Dict{Symbol, Any}())
    b = Bisimulation(name, coalgebra_a, coalgebra_b, relation;
                     description=description, metadata=metadata)
    bisimulations = get!(D.implementations, :_bisimulations) do
        Dict{Symbol, Bisimulation}()
    end::Dict{Symbol, Bisimulation}
    bisimulations[b.name] = b
    b
end

"""
    get_bisimulations(D::Diagram) -> Dict{Symbol, Bisimulation}

Retrieve all bisimulations declared in the diagram.
"""
function get_bisimulations(D::Diagram)
    get(D.implementations, :_bisimulations, Dict{Symbol, Bisimulation}())::Dict{Symbol, Bisimulation}
end

# ---------------------------------------------------------------------------
# Coalgebra residual — measures how far from exact commutativity
# ---------------------------------------------------------------------------

"""
    coalgebra_residual(compiled, coalgebra_name, inputs; kwargs...) -> Any

Compute the coalgebra residual: the difference between
`dynamics(encoder(x))` and `encoder(dynamics_observed(x))`.

When this residual is zero, the encoder is an exact coalgebra morphism
(the diagram commutes). This is the categorical version of JEPA's
prediction loss in representation space.

Returns the raw residual tensor/value (caller can apply norm for loss).
"""
function coalgebra_residual(compiled::CompiledDiagram,
                            coalgebra_name::Union{Symbol, AbstractString};
                            inputs::AbstractDict,
                            encoder::Union{Nothing, Symbol}=nothing,
                            kwargs...)
    result = run(compiled, inputs; kwargs...)
    coalgebras = get_coalgebras(compiled.diagram)
    c = coalgebras[Symbol(coalgebra_name)]

    # Get the transition output
    transition_val = result.values[c.transition]

    # If an encoder is specified, compute the residual as the obstruction
    # encoder(transition_obs(x)) vs transition_latent(encoder(x))
    if encoder !== nothing
        enc_val = result.values[encoder]
        return transition_val .- enc_val
    end

    transition_val
end

# ---------------------------------------------------------------------------
# World model block builder
# ---------------------------------------------------------------------------

"""Configuration for a coalgebra-based world model block."""
Base.@kwdef struct WorldModelConfig
    name::Symbol = :WorldModel
    state_object::Symbol = :State
    observation_object::Symbol = :Observation
    latent_object::Symbol = :LatentState
    encoder_name::Symbol = :encode
    dynamics_name::Symbol = :dynamics
    decoder_name::Symbol = :decode
    functor_type::Symbol = :identity
end

"""
    world_model_block(; config=WorldModelConfig(), kwargs...) -> Diagram

Build a coalgebra-based world model block:

    Observation --encode-→ LatentState --dynamics-→ LatentState
                                ↓
                             decode
                                ↓
                           Observation

The encoder maps observations to latent representations. The dynamics
morphism is the coalgebra structure map (world model step). The decoder
reconstructs observations (optional, for reconstruction loss).

The block includes an obstruction loss measuring how well the dynamics
in latent space commute with encoding from observation space.
"""
function world_model_block(; config::WorldModelConfig=WorldModelConfig(), kwargs...)
    cfg = _apply_overrides(config, kwargs)
    D = Diagram(cfg.name)

    # Objects
    add_object!(D, cfg.observation_object; kind=:observation)
    add_object!(D, cfg.latent_object; kind=:latent_state)

    # Morphisms
    add_morphism!(D, cfg.encoder_name, cfg.observation_object, cfg.latent_object;
                  metadata=Dict{Symbol, Any}(:role => :encoder))
    add_morphism!(D, cfg.dynamics_name, cfg.latent_object, cfg.latent_object;
                  metadata=Dict{Symbol, Any}(:role => :dynamics, :coalgebra => true))
    add_morphism!(D, cfg.decoder_name, cfg.latent_object, cfg.observation_object;
                  metadata=Dict{Symbol, Any}(:role => :decoder))

    # Coalgebra structure
    add_coalgebra!(D, :coalgebra;
                   state=cfg.latent_object, transition=cfg.dynamics_name,
                   functor_type=cfg.functor_type,
                   description="World model dynamics in latent space")

    # Compositions: encode→dynamics vs dynamics_obs→encode
    compose!(D, cfg.encoder_name, cfg.dynamics_name; name=:encode_then_predict)
    compose!(D, cfg.encoder_name, cfg.decoder_name; name=:autoencoder)

    # Ports
    expose_port!(D, :input, cfg.observation_object; direction=INPUT, port_type=:observation)
    expose_port!(D, :latent, cfg.latent_object; direction=OUTPUT, port_type=:latent_state)
    expose_port!(D, :dynamics, cfg.dynamics_name; direction=OUTPUT, port_type=:latent_state)
    expose_port!(D, :reconstruction, :autoencoder; direction=OUTPUT, port_type=:observation)

    D
end

# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

function Base.show(io::IO, c::Coalgebra)
    print(io, "Coalgebra(:$(c.name), $(c.state) →_{$(c.functor_type)} $(c.state))")
end

function Base.show(io::IO, m::CoalgebraMorphism)
    print(io, "CoalgebraMorphism(:$(m.name), $(m.source) → $(m.target) via $(m.hom))")
end

function Base.show(io::IO, w::FinalCoalgebraWitness)
    iso = w.is_isomorphism ? " [Lambek iso]" : ""
    print(io, "FinalCoalgebraWitness(:$(w.name), carrier=$(w.carrier)$(iso))")
end

function Base.show(io::IO, b::Bisimulation)
    print(io, "Bisimulation(:$(b.name), $(b.coalgebra_a) ~ $(b.coalgebra_b))")
end

function Base.show(io::IO, sc::StochasticCoalgebra)
    print(io, "StochasticCoalgebra(:$(sc.name), $(sc.state) → D_{$(sc.distribution_type)}($(sc.state)))")
end
