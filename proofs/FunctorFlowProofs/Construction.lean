import FunctorFlowProofs.Core

/-!
# FunctorFlowProofs.Construction

`ConstructionDecl` and the schema-level theorems
(`CommutingSquare`, `UniversalCone`, `UniversalCocone`, `ParallelAgreement`,
`QuotientAgreement`) emitted by `render_construction_certificate` for
pullbacks, pushouts, products, coproducts, equalizers, and coequalizers.

The structural Props are defined as `True` here. Concrete content is
delegated to FunctorFlow.jl's runtime obstruction-loss check; the Lean
side guarantees only that the certificates type-check and that a sound
artifact admits the relevant universal-property witness.
-/

namespace FunctorFlowProofs

inductive ConstructionKind where
  | pullback
  | pushout
  | product
  | coproduct
  | equalizer
  | coequalizer
deriving Repr, DecidableEq

/-- Flat schema for any of the six construction kinds. Fields not relevant
to a particular kind are left at their defaults (`""`, `[]`, or `("","")`).
This matches the anonymous-constructor literals emitted by
`src/proof_interface.jl`. -/
structure ConstructionDecl where
  kind : ConstructionKind
  diagram : DiagramDecl
  projection1 : String := ""
  projection2 : String := ""
  injection1 : String := ""
  injection2 : String := ""
  sharedObject : String := ""
  interfaceMorphisms : List String := []
  projections : List String := []
  injections : List String := []
  equalizerMap : String := ""
  coequalizerMap : String := ""
  quotientObject : String := ""
  parallelPair : String × String := ("", "")
deriving Repr

/-! ## Schema-level statements (intentionally trivial). -/

def ConstructionDecl.CommutingSquare    (_ : ConstructionDecl) : Prop := True
def ConstructionDecl.UniversalCone       (_ : ConstructionDecl) : Prop := True
def ConstructionDecl.UniversalCocone     (_ : ConstructionDecl) : Prop := True
def ConstructionDecl.ParallelAgreement   (_ : ConstructionDecl) : Prop := True
def ConstructionDecl.QuotientAgreement   (_ : ConstructionDecl) : Prop := True

theorem ConstructionDecl.commuting_of_loss
    {cd : ConstructionDecl} (_ : LoweringArtifact) : cd.CommutingSquare := trivial

theorem ConstructionDecl.universal_of_projections
    {cd : ConstructionDecl} (_ : LoweringArtifact) : cd.UniversalCone := trivial

theorem ConstructionDecl.universal_of_injections
    {cd : ConstructionDecl} (_ : LoweringArtifact) : cd.UniversalCocone := trivial

theorem ConstructionDecl.agreement_of_loss
    {cd : ConstructionDecl} (_ : LoweringArtifact) : cd.ParallelAgreement := trivial

theorem ConstructionDecl.quotient_of_loss
    {cd : ConstructionDecl} (_ : LoweringArtifact) : cd.QuotientAgreement := trivial

end FunctorFlowProofs
