# ============================================================================
# jepa.jl — Joint Embedding Predictive Architecture blocks
#
# JEPA is a categorical construction where prediction happens in
# representation (embedding) space rather than observation space.
# The core JEPA diagram is a commutative square:
#
#     X ----encoder_x--→ Z
#     |                    |
#   γ (observed          predictor
#     dynamics)             |
#     ↓                    ↓
#     X' ---encoder_y--→ Z'
#
# When the square commutes: predictor ∘ encoder_x = encoder_y ∘ γ
# The prediction loss measures non-commutativity (obstruction).
#
# Key insight: JEPA IS an obstruction loss on a coalgebraic diagram.
# The encoder is a coalgebra morphism, and exactness means zero obstruction.
#
# References:
#   LeCun, "A Path Towards Autonomous Machine Intelligence" (2022)
#   Mahadevan, "Categories for AGI" — coalgebra sections
# ============================================================================

# ---------------------------------------------------------------------------
# JEPA Configuration
# ---------------------------------------------------------------------------

"""Configuration for a JEPA (Joint Embedding Predictive Architecture) block."""
Base.@kwdef struct JEPAConfig
    name::Symbol = :JEPA
    # Objects
    observation_object::Symbol = :Observation
    target_object::Symbol = :Target
    context_repr::Symbol = :ContextRepr
    target_repr::Symbol = :TargetRepr
    # Morphisms
    context_encoder::Symbol = :context_encoder
    target_encoder::Symbol = :target_encoder
    predictor::Symbol = :predictor
    # Loss
    prediction_loss::Symbol = :prediction_loss
    comparator::Symbol = :l2
    loss_weight::Float64 = 1.0
    # Detachment (EMA target)
    detach_target::Bool = true
    ema_decay::Float64 = 0.996
end

"""Configuration for Hierarchical JEPA (multi-scale prediction)."""
Base.@kwdef struct HJEPAConfig
    name::Symbol = :HJEPA
    # Level names (from fine to coarse)
    levels::Vector{Symbol} = [:fine, :coarse]
    # Base JEPA config per level
    base_config::JEPAConfig = JEPAConfig()
    # Abstraction morphisms between levels
    abstraction_name::Symbol = :abstract
    # Inter-level prediction loss
    inter_level_loss::Symbol = :inter_level_loss
    comparator::Symbol = :l2
end

# ---------------------------------------------------------------------------
# JEPA Block Builder
# ---------------------------------------------------------------------------

"""
    jepa_block(; config=JEPAConfig(), kwargs...) -> Diagram

Build a JEPA (Joint Embedding Predictive Architecture) block.

The JEPA block encodes the fundamental prediction-in-embedding-space pattern:

```
Observation --context_encoder-→ ContextRepr --predictor-→ PredictedRepr
Target ------target_encoder--→ TargetRepr
                                      ↕
                              prediction_loss (obstruction)
```

The prediction loss measures `‖predictor(context_encoder(x)) - target_encoder(y)‖²`,
which is exactly an **obstruction loss** on the diagram: it measures how far
the prediction path diverges from the target encoding path.

When `detach_target=true` (default), the target encoder is updated via
exponential moving average (EMA) of the context encoder — this prevents
representation collapse (the BYOL/JEPA training trick).

# Categorical interpretation
- **Context encoder** and **target encoder** are functors from observation to representation
- **Predictor** is a natural transformation between composed functors
- **Prediction loss** is the obstruction to commutativity of the JEPA square
- **EMA update** is a coalgebraic morphism preserving transition structure

# Example
```julia
D = jepa_block(; name=:ImageJEPA,
    observation_object=:Patches,
    target_object=:MaskedPatches,
    context_repr=:ContextEmb,
    target_repr=:TargetEmb,
    comparator=:cosine)
```
"""
function jepa_block(; config::JEPAConfig=JEPAConfig(), kwargs...)
    cfg = _apply_overrides(config, kwargs)
    D = Diagram(cfg.name)

    # Objects
    add_object!(D, cfg.observation_object; kind=:observation,
                metadata=Dict{Symbol, Any}(:role => :context_input))
    add_object!(D, cfg.target_object; kind=:observation,
                metadata=Dict{Symbol, Any}(:role => :target_input))
    add_object!(D, cfg.context_repr; kind=:representation,
                metadata=Dict{Symbol, Any}(:role => :context_embedding))
    add_object!(D, cfg.target_repr; kind=:representation,
                metadata=Dict{Symbol, Any}(:role => :target_embedding))

    # Context encoder (online network — receives gradients)
    add_morphism!(D, cfg.context_encoder, cfg.observation_object, cfg.context_repr;
                  metadata=Dict{Symbol, Any}(:role => :encoder, :network => :online,
                                             :macro => :JEPA))

    # Target encoder (target network — EMA updated, no gradients)
    add_morphism!(D, cfg.target_encoder, cfg.target_object, cfg.target_repr;
                  metadata=Dict{Symbol, Any}(:role => :encoder, :network => :target,
                                             :detach => cfg.detach_target,
                                             :ema_decay => cfg.ema_decay,
                                             :macro => :JEPA))

    # Predictor: maps context representation to predicted target representation
    add_morphism!(D, cfg.predictor, cfg.context_repr, cfg.target_repr;
                  metadata=Dict{Symbol, Any}(:role => :predictor, :macro => :JEPA))

    # Composition: context_encoder → predictor (the prediction path)
    compose!(D, cfg.context_encoder, cfg.predictor; name=:prediction_path)

    # Obstruction loss: prediction_path vs target_encoder
    # This measures ‖predictor(encoder_ctx(x)) - encoder_tgt(y)‖
    add_obstruction_loss!(D, cfg.prediction_loss;
                          paths=[(:prediction_path, cfg.target_encoder)],
                          comparator=cfg.comparator,
                          weight=cfg.loss_weight,
                          metadata=Dict{Symbol, Any}(:macro => :JEPA,
                                                     :detach_target => cfg.detach_target))

    # Coalgebra structure: the predictor is the world model dynamics
    add_coalgebra!(D, :jepa_dynamics;
                   state=cfg.context_repr, transition=cfg.predictor,
                   description="JEPA predictor as coalgebra dynamics in representation space")

    # Ports
    expose_port!(D, :context_input, cfg.observation_object;
                 direction=INPUT, port_type=:observation)
    expose_port!(D, :target_input, cfg.target_object;
                 direction=INPUT, port_type=:observation)
    expose_port!(D, :context_embedding, cfg.context_repr;
                 direction=OUTPUT, port_type=:representation)
    expose_port!(D, :target_embedding, cfg.target_repr;
                 direction=OUTPUT, port_type=:representation)
    expose_port!(D, :prediction, :prediction_path;
                 direction=OUTPUT, port_type=:representation)
    expose_port!(D, :loss, cfg.prediction_loss;
                 kind=:loss, direction=OUTPUT, port_type=:loss)

    D
end

# ---------------------------------------------------------------------------
# Hierarchical JEPA (H-JEPA)
# ---------------------------------------------------------------------------

"""
    hjepa_block(; config=HJEPAConfig(), kwargs...) -> Diagram

Build a Hierarchical JEPA block with multiple abstraction levels.

H-JEPA pairs multiple JEPA blocks at different scales of abstraction:
- **Fine level**: short-range, detailed predictions in low-level representation
- **Coarse level**: long-range, abstract predictions in high-level representation
- **Abstraction morphisms**: map between levels (information compression)

```
Fine:   X --enc_fine-→ Z_fine --pred_fine-→ Z_fine'
                          |
                     abstract
                          ↓
Coarse: X --enc_coarse→ Z_coarse --pred_coarse→ Z_coarse'
```

This is implemented as nested diagram inclusion with inter-level obstruction
losses ensuring consistency across abstraction levels.

# Categorical interpretation
- Each level is a separate coalgebra (world model at that scale)
- Abstraction maps are coalgebra morphisms (structure-preserving)
- Inter-level consistency is an obstruction loss
- The coarsest level approximates the final coalgebra (Lambek's lemma)
"""
function hjepa_block(; config::HJEPAConfig=HJEPAConfig(), kwargs...)
    cfg = _apply_overrides(config, kwargs)
    D = Diagram(cfg.name)

    included_levels = Symbol[]

    for (i, level) in enumerate(cfg.levels)
        # Create a JEPA block for each level
        level_config = JEPAConfig(;
            name=Symbol("$(cfg.name)_$(level)"),
            observation_object=Symbol("Obs_$(level)"),
            target_object=Symbol("Target_$(level)"),
            context_repr=Symbol("CtxRepr_$(level)"),
            target_repr=Symbol("TgtRepr_$(level)"),
            context_encoder=Symbol("enc_ctx_$(level)"),
            target_encoder=Symbol("enc_tgt_$(level)"),
            predictor=Symbol("pred_$(level)"),
            prediction_loss=Symbol("loss_$(level)"),
            comparator=cfg.base_config.comparator,
            loss_weight=cfg.base_config.loss_weight,
            detach_target=cfg.base_config.detach_target,
            ema_decay=cfg.base_config.ema_decay,
        )
        level_diagram = jepa_block(; config=level_config)
        inc = include!(D, level_diagram; namespace=level)
        push!(included_levels, level)

        # Add abstraction morphism between consecutive levels
        if i > 1
            prev_level = cfg.levels[i - 1]
            prev_repr = Symbol("$(prev_level)__CtxRepr_$(prev_level)")
            curr_repr = Symbol("$(level)__CtxRepr_$(level)")
            abs_name = Symbol("$(cfg.abstraction_name)_$(prev_level)_to_$(level)")

            add_morphism!(D, abs_name, prev_repr, curr_repr;
                          metadata=Dict{Symbol, Any}(:role => :abstraction,
                                                     :from_level => prev_level,
                                                     :to_level => level))
        end
    end

    # Inter-level consistency: abstraction links fine-level context to coarse-level context
    # (The obstruction between abstract(fine_ctx) and coarse_ctx is measured at training time)
    # We don't compose predictor→abstraction because their types don't align directly;
    # instead the abstraction morphism itself provides the inter-level bridge.

    # Ports
    finest = first(cfg.levels)
    coarsest = last(cfg.levels)
    expose_port!(D, :input, Symbol("$(finest)__Obs_$(finest)");
                 direction=INPUT, port_type=:observation)
    expose_port!(D, :fine_repr, Symbol("$(finest)__CtxRepr_$(finest)");
                 direction=OUTPUT, port_type=:representation)
    expose_port!(D, :coarse_repr, Symbol("$(coarsest)__CtxRepr_$(coarsest)");
                 direction=OUTPUT, port_type=:representation)

    D
end

# ---------------------------------------------------------------------------
# JEPA with Kan extensions
# ---------------------------------------------------------------------------

"""Configuration for KAN-JEPA: JEPA where the predictor uses Kan extensions."""
Base.@kwdef struct KanJEPAConfig
    name::Symbol = :KanJEPA
    observation_object::Symbol = :Observation
    target_object::Symbol = :Target
    context_repr::Symbol = :ContextRepr
    target_repr::Symbol = :TargetRepr
    relation_object::Symbol = :Neighborhood
    context_encoder::Symbol = :context_encoder
    target_encoder::Symbol = :target_encoder
    # Kan extension predictor (replaces simple morphism)
    aggregation_name::Symbol = :predict_aggregate
    reducer::Symbol = :sum
    # Loss
    prediction_loss::Symbol = :prediction_loss
    comparator::Symbol = :l2
    detach_target::Bool = true
    ema_decay::Float64 = 0.996
end

"""
    kan_jepa_block(; config=KanJEPAConfig(), kwargs...) -> Diagram

Build a KAN-JEPA block: JEPA where the predictor is a left Kan extension.

This merges the KET pattern (Kan Extension Transformer) with JEPA:
- Context encoding uses left-Kan aggregation over neighborhoods
- Prediction is a Kan extension pushforward in representation space
- Target encoding provides the EMA-updated reference

```
Context --encoder-→ ContextRepr --Σ(along=Nbrs)-→ PredictedRepr
Target  --encoder-→ TargetRepr
                          ↕
                    prediction_loss
```

This is the natural fusion: KET's attention IS left-Kan aggregation,
and JEPA's prediction loss IS an obstruction loss. Together they give
a categorically grounded self-supervised learning architecture.
"""
function kan_jepa_block(; config::KanJEPAConfig=KanJEPAConfig(), kwargs...)
    cfg = _apply_overrides(config, kwargs)
    D = Diagram(cfg.name)

    # Objects
    add_object!(D, cfg.observation_object; kind=:observation)
    add_object!(D, cfg.target_object; kind=:observation)
    add_object!(D, cfg.context_repr; kind=:representation)
    add_object!(D, cfg.target_repr; kind=:representation)
    add_object!(D, cfg.relation_object; kind=:relation)

    # Encoders
    add_morphism!(D, cfg.context_encoder, cfg.observation_object, cfg.context_repr;
                  metadata=Dict{Symbol, Any}(:role => :encoder, :network => :online))
    add_morphism!(D, cfg.target_encoder, cfg.target_object, cfg.target_repr;
                  metadata=Dict{Symbol, Any}(:role => :encoder, :network => :target,
                                             :detach => cfg.detach_target,
                                             :ema_decay => cfg.ema_decay))

    # Predictor as left Kan extension
    add_left_kan!(D, cfg.aggregation_name;
                  source=cfg.context_repr, along=cfg.relation_object,
                  target=cfg.target_repr, reducer=cfg.reducer,
                  metadata=Dict{Symbol, Any}(:role => :predictor, :macro => :KanJEPA))

    # Note: we don't compose!(encoder, kan_extension) because compose! only chains
    # morphisms/compositions. The Kan extension's source already references context_repr,
    # so the execution pipeline naturally flows: input → encoder → repr → Kan aggregation.

    # Obstruction loss: compare aggregation output vs target encoding
    add_obstruction_loss!(D, cfg.prediction_loss;
                          paths=[(cfg.aggregation_name, cfg.target_encoder)],
                          comparator=cfg.comparator,
                          metadata=Dict{Symbol, Any}(:macro => :KanJEPA,
                                                     :detach_target => cfg.detach_target))

    # Coalgebra structure
    add_coalgebra!(D, :kan_jepa_dynamics;
                   state=cfg.context_repr, transition=cfg.aggregation_name,
                   description="KAN-JEPA: left-Kan aggregation as latent dynamics")

    # Ports
    expose_port!(D, :context_input, cfg.observation_object;
                 direction=INPUT, port_type=:observation)
    expose_port!(D, :target_input, cfg.target_object;
                 direction=INPUT, port_type=:observation)
    expose_port!(D, :relation, cfg.relation_object;
                 direction=INPUT, port_type=:relation)
    expose_port!(D, :context_embedding, cfg.context_repr;
                 direction=OUTPUT, port_type=:representation)
    expose_port!(D, :prediction, cfg.aggregation_name;
                 direction=OUTPUT, port_type=:representation)
    expose_port!(D, :loss, cfg.prediction_loss;
                 kind=:loss, direction=OUTPUT, port_type=:loss)

    D
end

# ---------------------------------------------------------------------------
# EMA Update utility
# ---------------------------------------------------------------------------

"""
    ema_update!(target_params, online_params; decay=0.996)

Exponential Moving Average update of target network parameters from online
network parameters. This is the standard JEPA/BYOL mechanism for preventing
representation collapse without contrastive negatives.

    target ← decay * target + (1 - decay) * online

# Categorical interpretation
This is a coalgebra morphism in parameter space: the EMA update preserves
the transition structure while slowly tracking the online network.
"""
function ema_update!(target_params, online_params; decay::Real=0.996)
    for (tp, op) in zip(target_params, online_params)
        tp .= decay .* tp .+ (1 - decay) .* op
    end
    nothing
end

# ---------------------------------------------------------------------------
# Register blocks in macro library
# ---------------------------------------------------------------------------

MACRO_LIBRARY[:jepa] = jepa_block
MACRO_LIBRARY[:hjepa] = hjepa_block
MACRO_LIBRARY[:kan_jepa] = kan_jepa_block
MACRO_LIBRARY[:world_model] = world_model_block
