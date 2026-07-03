/-!
# FunctorFlowProofs.Core

Core schema for FunctorFlow.jl Lean certificates: `OperationKind`,
`OperationDecl`, `PortDecl`, `DiagramDecl`, `LoweringArtifact`, plus the
`check`/`Sound` decision/property pair.

This file is consumed by certificates that `FunctorFlow.jl`'s
`render_lean_certificate` (in `src/proof_interface.jl`) emits into
`FunctorFlowProofs.Generated.*`. It mirrors the schema used by the sister
Python project's `FunctorFlow/proofs/` Lake project, extended with the
extra fields the Julia emitter writes (`PortDecl.kind`, `.portType`,
`.direction`; `LoweringArtifact.resolvedRefs`, `.portsClosed`, `.losses`).
-/

namespace FunctorFlowProofs

inductive OperationKind where
  | morphism
  | composition
  | leftKan
  | rightKan
  | unknown
deriving Repr, DecidableEq

structure OperationDecl where
  name : String
  kind : OperationKind
  refs : List String
deriving Repr, DecidableEq

structure PortDecl where
  name : String
  ref : String
  kind : String := ""
  portType : String := ""
  direction : String := ""
deriving Repr, DecidableEq

structure DiagramDecl where
  name : String
  objects : List String
  operations : List OperationDecl
  ports : List PortDecl
  /-- Names of the diagram's obstruction losses. These are legitimate
  reference targets for loss-kind ports, so they count as declared refs. -/
  lossNames : List String := []
deriving Repr, DecidableEq

/-- Loss declaration carried inside a `LoweringArtifact`. `valueText` preserves
the Julia-side numeric observation for auditability; `zeroValue` is the
Boolean fact the Julia emitter exposes to Lean when deciding exactness. -/
structure LossDecl where
  name : String
  valueText : String := "unknown"
  zeroValue : Bool := false
deriving Repr, DecidableEq

/-! ## Reference well-formedness checks (Bool decisions). -/

def listAllMembers (refs declared : List String) : Bool :=
  refs.all fun ref => ref ∈ declared

def operationListRefsDeclared (ops : List OperationDecl) (declared : List String) : Bool :=
  ops.all fun op => listAllMembers op.refs declared

def portListRefsDeclared (ports : List PortDecl) (declared : List String) : Bool :=
  ports.all fun port => port.ref ∈ declared

def DiagramDecl.declaredRefs (diagram : DiagramDecl) : List String :=
  diagram.objects ++ diagram.operations.map (·.name) ++ diagram.lossNames

def DiagramDecl.operationRefsDeclared (diagram : DiagramDecl) : Bool :=
  operationListRefsDeclared diagram.operations diagram.declaredRefs

def DiagramDecl.portRefsDeclared (diagram : DiagramDecl) : Bool :=
  portListRefsDeclared diagram.ports diagram.declaredRefs

def DiagramDecl.WellFormed (diagram : DiagramDecl) : Prop :=
  (∀ op, op ∈ diagram.operations → ∀ ref, ref ∈ op.refs → ref ∈ diagram.declaredRefs) ∧
  (∀ port, port ∈ diagram.ports → port.ref ∈ diagram.declaredRefs)

theorem listAllMembers_sound {refs declared : List String}
    (h : listAllMembers refs declared = true) :
    ∀ ref, ref ∈ refs → ref ∈ declared := by
  rw [listAllMembers, List.all_eq_true] at h
  intro ref href
  exact of_decide_eq_true (h ref href)

theorem operationListRefsDeclared_sound {ops : List OperationDecl} {declared : List String}
    (h : operationListRefsDeclared ops declared = true) :
    ∀ op, op ∈ ops → ∀ ref, ref ∈ op.refs → ref ∈ declared := by
  rw [operationListRefsDeclared, List.all_eq_true] at h
  intro op hop ref href
  exact listAllMembers_sound (h op hop) ref href

theorem portListRefsDeclared_sound {ports : List PortDecl} {declared : List String}
    (h : portListRefsDeclared ports declared = true) :
    ∀ port, port ∈ ports → port.ref ∈ declared := by
  rw [portListRefsDeclared, List.all_eq_true] at h
  intro port hport
  exact of_decide_eq_true (h port hport)

theorem DiagramDecl.wellFormed_of_checks {diagram : DiagramDecl}
    (hOps : diagram.operationRefsDeclared = true)
    (hPorts : diagram.portRefsDeclared = true) :
    diagram.WellFormed := by
  refine ⟨?_, ?_⟩
  · exact operationListRefsDeclared_sound hOps
  · exact portListRefsDeclared_sound hPorts

/-! ## LoweringArtifact -/

structure LoweringArtifact where
  diagram : DiagramDecl
  resolvedRefs : Bool := true
  portsClosed : Bool := true
  losses : List LossDecl := []
deriving Repr

/-- Decidable check for an artifact: structural well-formedness plus the
two emitter-asserted Bool flags. -/
def LoweringArtifact.check (a : LoweringArtifact) : Bool :=
  a.diagram.operationRefsDeclared &&
  a.diagram.portRefsDeclared &&
  a.resolvedRefs &&
  a.portsClosed

/-- An artifact is sound iff its diagram is well-formed and the emitter's
ref-resolution / port-closure invariants hold. -/
def LoweringArtifact.Sound (a : LoweringArtifact) : Prop :=
  a.diagram.WellFormed ∧ a.resolvedRefs = true ∧ a.portsClosed = true

theorem LoweringArtifact.sound_of_check {a : LoweringArtifact}
    (h : a.check = true) : a.Sound := by
  have hAll :
      ((a.diagram.operationRefsDeclared = true ∧
        a.diagram.portRefsDeclared = true) ∧
        a.resolvedRefs = true) ∧ a.portsClosed = true := by
    simpa [LoweringArtifact.check, Bool.and_eq_true] using h
  refine ⟨?_, hAll.1.2, hAll.2⟩
  exact a.diagram.wellFormed_of_checks hAll.1.1.1 hAll.1.1.2

/-! ## Loss / obstruction semantics.

These give the JEPA / coalgebra / construction certificates genuine,
*falsifiable* content. A recorded obstruction loss is "zero" iff its carried
`zeroValue` field is `true`, and an artifact is *exact* iff every recorded loss
is zero. The emitter records the observed Julia-side numeric text together with
that Boolean zero/nonzero fact; the theorems below derive the categorical
consequences **from** those recorded zero facts via `native_decide`, so a
certificate that records a nonzero obstruction loss will fail to type-check its
exactness/commutativity claims. This is the content the earlier schema-level
`True` placeholders lacked. -/

/-- Every recorded loss component has value `0` — the artifact is exact. -/
def LoweringArtifact.AllLossesZero (a : LoweringArtifact) : Prop :=
  ∀ l ∈ a.losses, l.zeroValue = true

/-- Decidable Boolean check for `AllLossesZero`. -/
def LoweringArtifact.allLossesZeroCheck (a : LoweringArtifact) : Bool :=
  a.losses.all (fun l => l.zeroValue)

theorem LoweringArtifact.allLossesZero_of_check {a : LoweringArtifact}
    (h : a.allLossesZeroCheck = true) : a.AllLossesZero := by
  rw [LoweringArtifact.allLossesZeroCheck, List.all_eq_true] at h
  intro l hl
  exact h l hl

/-- `s` names an obstruction loss component actually tracked by the artifact.
Falsifiable: false when no recorded loss is named `s`. -/
def LoweringArtifact.lossIsObstruction (a : LoweringArtifact) (s : String) : Prop :=
  ∃ l ∈ a.losses, l.name = s

/-- Decidable check that `s` names a recorded loss. -/
def LoweringArtifact.hasLoss (a : LoweringArtifact) (s : String) : Bool :=
  a.losses.any (fun l => l.name == s)

theorem LoweringArtifact.loss_obstruction_of_hasLoss {a : LoweringArtifact} {s : String}
    (h : a.hasLoss s = true) : a.lossIsObstruction s := by
  rw [LoweringArtifact.hasLoss, List.any_eq_true] at h
  obtain ⟨l, hl, hname⟩ := h
  exact ⟨l, hl, eq_of_beq hname⟩

/-- A coalgebra certificate is *exact* iff every recorded prediction/obstruction
loss is zero (the JEPA encoder/predictor square commutes). -/
def LoweringArtifact.CoalgebraExact (a : LoweringArtifact) : Prop :=
  a.AllLossesZero

theorem LoweringArtifact.coalgebra_exact_of_zero_loss {a : LoweringArtifact}
    (h : a.AllLossesZero) : a.CoalgebraExact := h

end FunctorFlowProofs
