# ============================================================================
# energy.jl — Energy-based cost module for FunctorFlow.jl
#
# Energy-Based Models (EBMs) learn an energy function F(x,y) that takes low
# values for compatible (x,y) pairs and high values otherwise. In the JEPA
# framework, energy measures prediction quality in representation space.
#
# The cost module decomposes into:
#   C(s) = IC(s) + TC(s)
# where IC is the immutable intrinsic cost and TC is the trainable critic.
#
# In categorical terms, energy is a functor from the diagram's state category
# to ℝ (the real numbers as a poset category). The configurator adjusts
# the weighting of sub-costs via natural transformations.
#
# References:
#   LeCun, "A Path Towards Autonomous Machine Intelligence" — Cost module
#   Mahadevan, "Categories for AGI" — Energy as categorical functor
# ============================================================================

# ---------------------------------------------------------------------------
# Energy function types
# ---------------------------------------------------------------------------

"""
    EnergyFunction(name, domain, codomain, energy_type; ...)

An energy function F : X × Y → ℝ that measures compatibility between
observations and predictions. Low energy = compatible, high energy = incompatible.

In the FunctorFlow context, the energy function operates on diagram states
(values flowing through objects and operations).

# Energy types
- `:l2` — Squared L2 distance (standard for continuous representations)
- `:cosine` — Cosine similarity energy (1 - cos(x, y))
- `:contrastive` — InfoNCE / contrastive loss
- `:vicreg` — Variance-Invariance-Covariance regularization
- `:barlow_twins` — Cross-correlation redundancy reduction
- `:custom` — User-provided energy function
"""
struct EnergyFunction <: AbstractFFElement
    name::Symbol
    domain::Vector{Symbol}      # input object names (e.g., [:ContextRepr, :TargetRepr])
    codomain::Symbol            # typically :energy_scalar
    energy_type::Symbol         # :l2, :cosine, :contrastive, :vicreg, :barlow_twins, :custom
    temperature::Float64        # scaling parameter (for contrastive)
    description::String
    metadata::Dict{Symbol, Any}
end

function EnergyFunction(name, domain;
                        codomain::Union{Symbol, AbstractString}=:energy,
                        energy_type::Union{Symbol, AbstractString}=:l2,
                        temperature::Real=0.07,
                        description::AbstractString="",
                        metadata::Dict=Dict{Symbol, Any}())
    EnergyFunction(Symbol(name), Symbol.(domain), Symbol(codomain),
                   Symbol(energy_type), Float64(temperature),
                   String(description),
                   Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

# ---------------------------------------------------------------------------
# Cost module — intrinsic + trainable decomposition
# ---------------------------------------------------------------------------

"""
    IntrinsicCost(name, cost_type, weight; ...)

An immutable cost component that constrains agent behavior. Intrinsic costs
are hardwired and cannot be learned, preventing the system from drifting
toward degenerate solutions.

In JEPA/world model terms:
- **Reconstruction cost**: decoded output should match input
- **Prediction cost**: predicted embedding should match target embedding
- **Regularization cost**: representations should be informative (VICReg)
- **Collapse prevention**: variance term ensuring non-degenerate representations

# Cost types
- `:prediction` — ‖predicted_repr - target_repr‖²
- `:reconstruction` — ‖decoded - input‖²
- `:variance` — -Var(representations) (collapse prevention)
- `:covariance` — off-diagonal covariance penalty
- `:information` — negative mutual information estimate
"""
struct IntrinsicCost <: AbstractFFElement
    name::Symbol
    cost_type::Symbol
    weight::Float64
    source_refs::Vector{Symbol}     # objects/operations this cost operates on
    description::String
    metadata::Dict{Symbol, Any}
end

function IntrinsicCost(name;
                       cost_type::Union{Symbol, AbstractString}=:prediction,
                       weight::Real=1.0,
                       source_refs::Vector=Symbol[],
                       description::AbstractString="",
                       metadata::Dict=Dict{Symbol, Any}())
    IntrinsicCost(Symbol(name), Symbol(cost_type), Float64(weight),
                  Symbol.(source_refs), String(description),
                  Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    TrainableCost(name, critic_morphism, weight; ...)

A trainable critic that predicts future intrinsic costs. The critic is
trained from past states and subsequent intrinsic cost values.

    TC(sₜ) ≈ IC(sₜ₊δ)

The critic enables the agent to minimize future cost, not just current cost.

# Categorical interpretation
The critic is a coalgebra morphism from the state coalgebra to the
cost category (ℝ as a poset), trained to approximate the composed map:
    state →^dynamics next_state →^IC ℝ
"""
struct TrainableCost <: AbstractFFElement
    name::Symbol
    critic_morphism::Symbol     # morphism that computes the trainable cost
    weight::Float64
    lookahead::Int              # how many steps ahead to predict (δ)
    discount::Float64           # temporal discount factor γ
    description::String
    metadata::Dict{Symbol, Any}
end

function TrainableCost(name;
                       critic_morphism::Union{Symbol, AbstractString}=:critic,
                       weight::Real=1.0,
                       lookahead::Int=1,
                       discount::Real=0.99,
                       description::AbstractString="",
                       metadata::Dict=Dict{Symbol, Any}())
    TrainableCost(Symbol(name), Symbol(critic_morphism), Float64(weight),
                  lookahead, Float64(discount), String(description),
                  Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

"""
    CostModule(name, intrinsic_costs, trainable_costs; ...)

The full cost module: C(s) = Σᵢ uᵢ·ICᵢ(s) + Σⱼ vⱼ·TCⱼ(s)

The weights u and v are controlled by the configurator, allowing the system
to focus on different objectives at different times.

# JEPA training criteria (as cost decomposition)
1. IC₁: Maximize information content of sₓ about x (-I(sₓ))
2. IC₂: Maximize information content of sᵧ about y (-I(sᵧ))
3. IC₃: Make sᵧ predictable from sₓ (D(sᵧ, s̃ᵧ))
4. IC₄: Minimize information in latent variable (R(z))
"""
struct CostModule <: AbstractFFElement
    name::Symbol
    intrinsic_costs::Vector{IntrinsicCost}
    trainable_costs::Vector{TrainableCost}
    description::String
    metadata::Dict{Symbol, Any}
end

function CostModule(name;
                    intrinsic_costs::Vector{IntrinsicCost}=IntrinsicCost[],
                    trainable_costs::Vector{TrainableCost}=TrainableCost[],
                    description::AbstractString="",
                    metadata::Dict=Dict{Symbol, Any}())
    CostModule(Symbol(name), intrinsic_costs, trainable_costs,
               String(description),
               Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

# ---------------------------------------------------------------------------
# Configurator — controls cost weighting and module behavior
# ---------------------------------------------------------------------------

"""
    Configurator(name, cost_weights, module_configs; ...)

The configurator controls the behavior of other system components by
adjusting their parameters. In the JEPA framework:

- Sets cost weights (u, v) for intrinsic and trainable costs
- Injects configuration tokens into transformer-based modules
- Decomposes complex tasks into subgoals

# Categorical interpretation
The configurator is a natural transformation between the "unweighted cost"
functor and the "weighted cost" functor, parameterized by the current context.
"""
struct Configurator <: AbstractFFElement
    name::Symbol
    cost_weights::Dict{Symbol, Float64}     # cost_name → weight
    module_configs::Dict{Symbol, Any}       # module_name → config
    description::String
    metadata::Dict{Symbol, Any}
end

function Configurator(name;
                      cost_weights::Dict=Dict{Symbol, Float64}(),
                      module_configs::Dict=Dict{Symbol, Any}(),
                      description::AbstractString="",
                      metadata::Dict=Dict{Symbol, Any}())
    Configurator(Symbol(name),
                 Dict{Symbol, Float64}(Symbol(k) => Float64(v) for (k, v) in cost_weights),
                 Dict{Symbol, Any}(Symbol(k) => v for (k, v) in module_configs),
                 String(description),
                 Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

# ---------------------------------------------------------------------------
# Collapse prevention strategies
# ---------------------------------------------------------------------------

"""
    CollapsePreventionStrategy

Strategy for preventing representation collapse in self-supervised learning.
All strategies can be expressed as regularization terms in the energy function.

- `:ema_target` — Exponential Moving Average target network (BYOL, JEPA)
- `:contrastive` — Negative examples increase energy (SimCLR, MoCo)
- `:vicreg` — Variance-Invariance-Covariance regularization
- `:barlow_twins` — Cross-correlation identity target
- `:whitening` — Whitening of representations
"""
@enum CollapsePreventionStrategy begin
    EMA_TARGET          # BYOL/JEPA style
    CONTRASTIVE         # SimCLR/MoCo style
    VICREG              # Variance-Invariance-Covariance
    BARLOW_TWINS        # Cross-correlation
    WHITENING           # Representation whitening
end

# ---------------------------------------------------------------------------
# Energy-aware diagram integration
# ---------------------------------------------------------------------------

"""
    add_energy_function!(D::Diagram, name; domain, energy_type=:l2, ...)

Add an energy function to a diagram. The energy function measures
compatibility between specified objects/operations in the diagram.
"""
function add_energy_function!(D::Diagram, name::Union{Symbol, AbstractString};
                              domain::Vector,
                              codomain::Union{Symbol, AbstractString}=:energy,
                              energy_type::Union{Symbol, AbstractString}=:l2,
                              temperature::Real=0.07,
                              description::AbstractString="",
                              metadata::Dict=Dict{Symbol, Any}())
    ef = EnergyFunction(name, domain;
                        codomain=codomain, energy_type=energy_type,
                        temperature=temperature,
                        description=description, metadata=metadata)
    energy_fns = get!(D.implementations, :_energy_functions) do
        Dict{Symbol, EnergyFunction}()
    end::Dict{Symbol, EnergyFunction}
    energy_fns[ef.name] = ef
    ef
end

"""
    get_energy_functions(D::Diagram) -> Dict{Symbol, EnergyFunction}

Retrieve all energy functions declared in the diagram.
"""
function get_energy_functions(D::Diagram)
    get(D.implementations, :_energy_functions, Dict{Symbol, EnergyFunction}())::Dict{Symbol, EnergyFunction}
end

"""
    add_cost_module!(D::Diagram, name; intrinsic_costs, trainable_costs, ...)

Add a full cost module (IC + TC decomposition) to the diagram.
"""
function add_cost_module!(D::Diagram, name::Union{Symbol, AbstractString};
                          intrinsic_costs::Vector{IntrinsicCost}=IntrinsicCost[],
                          trainable_costs::Vector{TrainableCost}=TrainableCost[],
                          description::AbstractString="",
                          metadata::Dict=Dict{Symbol, Any}())
    cm = CostModule(name;
                    intrinsic_costs=intrinsic_costs,
                    trainable_costs=trainable_costs,
                    description=description, metadata=metadata)
    cost_modules = get!(D.implementations, :_cost_modules) do
        Dict{Symbol, CostModule}()
    end::Dict{Symbol, CostModule}
    cost_modules[cm.name] = cm
    cm
end

"""
    get_cost_modules(D::Diagram) -> Dict{Symbol, CostModule}

Retrieve all cost modules declared in the diagram.
"""
function get_cost_modules(D::Diagram)
    get(D.implementations, :_cost_modules, Dict{Symbol, CostModule}())::Dict{Symbol, CostModule}
end

# ---------------------------------------------------------------------------
# Built-in energy function implementations
# ---------------------------------------------------------------------------

"""Squared L2 energy: ‖x - y‖²"""
function energy_l2(x, y)
    diff = x .- y
    sum(diff .^ 2)
end

"""Cosine similarity energy: 1 - cos(x, y)"""
function energy_cosine(x, y)
    nx = sqrt(sum(x .^ 2) + 1e-8)
    ny = sqrt(sum(y .^ 2) + 1e-8)
    1.0 - sum(x .* y) / (nx * ny)
end

"""Smooth L1 (Huber) energy"""
function energy_smooth_l1(x, y; beta::Real=1.0)
    diff = abs.(x .- y)
    sum(ifelse.(diff .< beta, 0.5 .* diff .^ 2 ./ beta, diff .- 0.5 * beta))
end

"""
    variance_regularization(representations; eps=1e-4)

VICReg variance term: penalizes collapse by requiring each representation
dimension to have variance above a threshold.

Returns the hinge loss: Σ max(0, γ - sqrt(Var(z_d) + ε))
"""
function variance_regularization(representations; eps::Real=1e-4, gamma::Real=1.0)
    # representations: matrix where each column is a sample
    if ndims(representations) == 1
        return zero(eltype(representations))
    end
    μ = sum(representations; dims=2) ./ size(representations, 2)
    centered = representations .- μ
    vars = sum(centered .^ 2; dims=2) ./ max(1, size(representations, 2) - 1)
    stds = sqrt.(vars .+ eps)
    sum(max.(0, gamma .- stds))
end

"""
    covariance_regularization(representations; eps=1e-4)

VICReg covariance term: penalizes redundancy by decorrelating representation
dimensions. Minimizes off-diagonal elements of the covariance matrix.
"""
function covariance_regularization(representations; eps::Real=1e-4)
    if ndims(representations) == 1
        return zero(eltype(representations))
    end
    n = size(representations, 2)
    μ = sum(representations; dims=2) ./ n
    centered = representations .- μ
    cov = (centered * centered') ./ max(1, n - 1)
    d = size(cov, 1)
    # Zero out diagonal, sum off-diagonal squared elements
    off_diag_sum = sum(cov .^ 2) - sum([cov[i, i]^2 for i in 1:d])
    off_diag_sum / d
end

"""Registry of built-in energy function implementations."""
const BUILTIN_ENERGY_FUNCTIONS = Dict{Symbol, Any}(
    :l2 => energy_l2,
    :cosine => energy_cosine,
    :smooth_l1 => energy_smooth_l1,
)

"""Registry of built-in regularization functions."""
const BUILTIN_REGULARIZERS = Dict{Symbol, Any}(
    :variance => variance_regularization,
    :covariance => covariance_regularization,
)

# ---------------------------------------------------------------------------
# Energy-based cost block builder
# ---------------------------------------------------------------------------

"""Configuration for an energy-based cost block."""
Base.@kwdef struct EnergyBlockConfig
    name::Symbol = :EnergyCost
    prediction_object::Symbol = :Prediction
    target_object::Symbol = :Target
    energy_type::Symbol = :l2
    temperature::Float64 = 0.07
    # Intrinsic costs
    prediction_weight::Float64 = 1.0
    variance_weight::Float64 = 0.0
    covariance_weight::Float64 = 0.0
    # Collapse prevention
    collapse_strategy::CollapsePreventionStrategy = EMA_TARGET
end

"""
    energy_block(; config=EnergyBlockConfig(), kwargs...) -> Diagram

Build an energy-based cost block that measures compatibility between
prediction and target in representation space.

Includes optional VICReg-style regularization for collapse prevention.
"""
function energy_block(; config::EnergyBlockConfig=EnergyBlockConfig(), kwargs...)
    cfg = _apply_overrides(config, kwargs)
    D = Diagram(cfg.name)

    # Objects
    add_object!(D, cfg.prediction_object; kind=:representation)
    add_object!(D, cfg.target_object; kind=:representation)

    # Energy function
    add_energy_function!(D, :energy;
                         domain=[cfg.prediction_object, cfg.target_object],
                         energy_type=cfg.energy_type,
                         temperature=cfg.temperature,
                         description="Prediction-target compatibility energy")

    # Intrinsic costs
    ics = IntrinsicCost[]
    push!(ics, IntrinsicCost(:prediction_cost;
                             cost_type=:prediction,
                             weight=cfg.prediction_weight,
                             source_refs=[cfg.prediction_object, cfg.target_object]))

    if cfg.variance_weight > 0
        push!(ics, IntrinsicCost(:variance_cost;
                                 cost_type=:variance,
                                 weight=cfg.variance_weight,
                                 source_refs=[cfg.prediction_object]))
    end

    if cfg.covariance_weight > 0
        push!(ics, IntrinsicCost(:covariance_cost;
                                 cost_type=:covariance,
                                 weight=cfg.covariance_weight,
                                 source_refs=[cfg.prediction_object]))
    end

    add_cost_module!(D, :cost;
                     intrinsic_costs=ics,
                     description="Energy-based cost with $(cfg.collapse_strategy) collapse prevention")

    # Ports
    expose_port!(D, :prediction, cfg.prediction_object;
                 direction=INPUT, port_type=:representation)
    expose_port!(D, :target, cfg.target_object;
                 direction=INPUT, port_type=:representation)

    D
end

# ---------------------------------------------------------------------------
# Register in macro library
# ---------------------------------------------------------------------------

MACRO_LIBRARY[:energy] = energy_block

# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

function Base.show(io::IO, ef::EnergyFunction)
    print(io, "EnergyFunction(:$(ef.name), $(join(ef.domain, " × ")) → $(ef.codomain), type=$(ef.energy_type))")
end

function Base.show(io::IO, ic::IntrinsicCost)
    print(io, "IntrinsicCost(:$(ic.name), type=$(ic.cost_type), weight=$(ic.weight))")
end

function Base.show(io::IO, tc::TrainableCost)
    print(io, "TrainableCost(:$(tc.name), critic=$(tc.critic_morphism), δ=$(tc.lookahead), γ=$(tc.discount))")
end

function Base.show(io::IO, cm::CostModule)
    print(io, "CostModule(:$(cm.name), $(length(cm.intrinsic_costs)) IC + $(length(cm.trainable_costs)) TC)")
end

function Base.show(io::IO, cfg::Configurator)
    print(io, "Configurator(:$(cfg.name), $(length(cfg.cost_weights)) weights, $(length(cfg.module_configs)) configs)")
end
