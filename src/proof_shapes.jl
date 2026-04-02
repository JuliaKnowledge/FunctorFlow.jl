# ============================================================================
# proof_shapes.jl — First-class proof-shape records
# ============================================================================

"""
    ProofShape(name, claim_kind, subject_name; assumptions=String[], obligations=String[], metadata=Dict())

Record the structural obligations associated with a categorical claim. These
records do not prove anything by themselves; they package the assumptions and
obligations that downstream proof tooling or cross-language parity tests should
check.
"""
struct ProofShape
    name::Symbol
    claim_kind::Symbol
    subject_name::Symbol
    assumptions::Vector{String}
    obligations::Vector{String}
    metadata::Dict{Symbol, Any}
end

function ProofShape(name, claim_kind, subject_name;
                    assumptions::Vector{String}=String[],
                    obligations::Vector{String}=String[],
                    metadata::Dict=Dict{Symbol, Any}())
    ProofShape(Symbol(name), Symbol(claim_kind), Symbol(subject_name),
               copy(assumptions), copy(obligations),
               Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

struct PullbackProofShape
    construction::PullbackResult
    claim::ProofShape
end

struct PushoutProofShape
    construction::PushoutResult
    claim::ProofShape
end

struct LeftKanProofShape
    construction::KanExtension
    claim::ProofShape
end

struct RightKanProofShape
    construction::KanExtension
    claim::ProofShape
end

struct SCMMonomorphismProofShape
    construction::Any
    claim::ProofShape
end

struct SCMCharacteristicMapProofShape
    construction::Any
    claim::ProofShape
end

"""
    ProofBundle(name, claims; metadata=Dict())

Collect a small family of proof-shape claims for one semantic example or
compiler pipeline.
"""
struct ProofBundle
    name::Symbol
    claims::Vector{ProofShape}
    metadata::Dict{Symbol, Any}
end

function ProofBundle(name, claims::Vector{ProofShape};
                     metadata::Dict=Dict{Symbol, Any}())
    ProofBundle(Symbol(name), copy(claims),
                Dict{Symbol, Any}(Symbol(k) => v for (k, v) in metadata))
end

function bundle_proof_shapes(name, claims::ProofShape...; metadata::Dict=Dict{Symbol, Any}())
    ProofBundle(name, collect(claims); metadata=metadata)
end

"""
    prove_pullback_shape(pb::PullbackResult) -> PullbackProofShape

Record the proof obligations associated with a pullback claim.
"""
function prove_pullback_shape(pb::PullbackResult)
    claim = ProofShape(Symbol(pb.name, :__pullback_claim), :pullback, pb.name;
        assumptions=[
            "projection $(pb.projection1) factors through the left leg",
            "projection $(pb.projection2) factors through the right leg",
            "shared object $(pb.shared_object) mediates compatibility",
        ],
        obligations=[
            "commuting square condition",
            "universal factorization property",
        ],
        metadata=Dict{Symbol, Any}(
            :construction => pb.name,
            :shared_object => pb.shared_object,
            :interface_morphisms => copy(pb.interface_morphisms),
        ),
    )
    PullbackProofShape(pb, claim)
end

"""
    prove_pushout_shape(po::PushoutResult) -> PushoutProofShape

Record the proof obligations associated with a pushout claim.
"""
function prove_pushout_shape(po::PushoutResult)
    claim = ProofShape(Symbol(po.name, :__pushout_claim), :pushout, po.name;
        assumptions=[
            "injection $(po.injection1) includes the left leg",
            "injection $(po.injection2) includes the right leg",
            "shared object $(po.shared_object) is glued consistently",
        ],
        obligations=[
            "commuting square condition",
            "universal cofactorization property",
        ],
        metadata=Dict{Symbol, Any}(
            :construction => po.name,
            :shared_object => po.shared_object,
            :interface_morphisms => copy(po.interface_morphisms),
        ),
    )
    PushoutProofShape(po, claim)
end

"""
    prove_left_kan_shape(kan::KanExtension) -> LeftKanProofShape

Record the proof obligations associated with a left-Kan / aggregation claim.
"""
function prove_left_kan_shape(kan::KanExtension)
    kan.direction == LEFT || throw(ArgumentError("Expected a LEFT Kan extension, got $(kan.direction)"))
    claim = ProofShape(Symbol(kan.name, :__left_kan_claim), :left_kan, kan.name;
        assumptions=[
            "source $(kan.source) is aggregated along $(kan.along)",
            "reducer $(kan.reducer) witnesses the colimit-style aggregation",
        ],
        obligations=[
            "left universal property",
            "intervention-compatible pushforward property",
        ],
        metadata=Dict{Symbol, Any}(
            :construction => kan.name,
            :source => kan.source,
            :along => kan.along,
            :target => kan.target,
            :reducer => kan.reducer,
        ),
    )
    LeftKanProofShape(kan, claim)
end

"""
    prove_right_kan_shape(kan::KanExtension) -> RightKanProofShape

Record the proof obligations associated with a right-Kan / completion claim.
"""
function prove_right_kan_shape(kan::KanExtension)
    kan.direction == RIGHT || throw(ArgumentError("Expected a RIGHT Kan extension, got $(kan.direction)"))
    claim = ProofShape(Symbol(kan.name, :__right_kan_claim), :right_kan, kan.name;
        assumptions=[
            "source $(kan.source) is completed along $(kan.along)",
            "reducer $(kan.reducer) witnesses the limit-style completion",
        ],
        obligations=[
            "right universal property",
            "conditioning-compatible extension property",
        ],
        metadata=Dict{Symbol, Any}(
            :construction => kan.name,
            :source => kan.source,
            :along => kan.along,
            :target => kan.target,
            :reducer => kan.reducer,
        ),
    )
    RightKanProofShape(kan, claim)
end

"""
    prove_scm_monomorphism_shape(mono::SCMMonomorphism) -> SCMMonomorphismProofShape

Record the obligations associated with an SCM monomorphism claim.
"""
function prove_scm_monomorphism_shape(mono)
    claim = ProofShape(Symbol(mono.name, :__scm_monomorphism_claim), :scm_monomorphism, mono.name;
        assumptions=[
            "$(mono.name): $(mono.source_scm.name) -> $(mono.target_scm.name)",
        ],
        obligations=[
            "left-cancellability / monic behavior",
            "predicate subobject inclusion preserves causal signature",
        ],
        metadata=Dict(
            :construction => mono.name,
            :ambient_scm => mono.ambient_scm.name,
        ),
    )
    SCMMonomorphismProofShape(mono, claim)
end

"""
    prove_scm_characteristic_map_shape(map::SCMCharacteristicMap) -> SCMCharacteristicMapProofShape

Record the obligations associated with an SCM characteristic-map claim.
"""
function prove_scm_characteristic_map_shape(map)
    claim = ProofShape(Symbol(map.name, :__scm_characteristic_map_claim), :scm_characteristic_map, map.name;
        assumptions=[
            "$(map.name): $(map.ambient_scm.name) -> $(map.omega.name)",
            "$(map.predicate.name) is classified by $(map.classifying_truth_value.name)",
        ],
        obligations=[
            "characteristic-map classifies the intended subobject",
            "pullback along truth identifies the predicate subobject",
        ],
        metadata=Dict(
            :construction => map.name,
            :predicate => map.predicate.name,
            :omega => map.omega.name,
            :classifying_truth_value => map.classifying_truth_value.name,
        ),
    )
    SCMCharacteristicMapProofShape(map, claim)
end
