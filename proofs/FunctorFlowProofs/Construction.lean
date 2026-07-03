import FunctorFlowProofs.Core

/-!
# FunctorFlowProofs.Construction

`ConstructionDecl` and the certificate properties emitted by
`render_construction_certificate` for pullbacks, pushouts, products,
coproducts, equalizers, and coequalizers.

Each property is the conjunction of two *decidable, falsifiable* facts:

* **Structural validity** (`StructurallyValid`): the morphisms / objects the
  construction names are actually declared in the carried diagram (for
  products / coproducts: each factor namespace has a corresponding declared
  object). This is decided by `native_decide` at certificate-compile time, so
  a malformed emission fails to type-check.
* **Zero obstruction** (`LoweringArtifact.AllLossesZero`): every recorded
  obstruction loss is zero — the commuting square / universal cone actually
  commutes for the recorded configuration.

This replaces the earlier `Prop := True` placeholders: the universal-property
certificates now carry genuine content tied to the diagram structure and the
obstruction-loss values.
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

/-! ## Structural validity (decidable over the carried decl data). -/

/-- Every string in `xs` is a declared ref of the diagram. -/
def allDeclared (d : DiagramDecl) (xs : List String) : Bool :=
  xs.all (fun r => d.declaredRefs.contains r)

/-- Some declared object name begins with `ns ++ "__"` — i.e. the factor
namespace `ns` was actually included into the diagram. -/
def hasNamespacedObject (d : DiagramDecl) (ns : String) : Bool :=
  d.objects.any (fun o => (ns ++ "__").isPrefixOf o)

/-- Decidable structural-validity check, specialised per construction kind. -/
def ConstructionDecl.structurallyValidCheck (cd : ConstructionDecl) : Bool :=
  match cd.kind with
  | .pullback =>
      Nat.ble 2 cd.interfaceMorphisms.length
        && allDeclared cd.diagram cd.interfaceMorphisms
        && cd.diagram.declaredRefs.contains cd.sharedObject
  | .pushout =>
      Nat.ble 2 cd.interfaceMorphisms.length
        && allDeclared cd.diagram cd.interfaceMorphisms
        && cd.diagram.declaredRefs.contains cd.sharedObject
  | .product =>
      !cd.projections.isEmpty
        && cd.projections.all (hasNamespacedObject cd.diagram)
  | .coproduct =>
      !cd.injections.isEmpty
        && cd.injections.all (hasNamespacedObject cd.diagram)
  | .equalizer =>
      cd.diagram.declaredRefs.contains cd.equalizerMap
  | .coequalizer =>
      cd.diagram.declaredRefs.contains cd.coequalizerMap
        && cd.diagram.declaredRefs.contains cd.quotientObject

/-- The construction's named morphisms/objects are all declared in the
carried diagram. -/
def ConstructionDecl.StructurallyValid (cd : ConstructionDecl) : Prop :=
  cd.structurallyValidCheck = true

theorem ConstructionDecl.structurally_valid_of_check {cd : ConstructionDecl}
    (h : cd.structurallyValidCheck = true) : cd.StructurallyValid := h

/-! ## Certificate properties: structural validity + zero obstruction. -/

def ConstructionDecl.CommutingSquare (cd : ConstructionDecl) (a : LoweringArtifact) : Prop :=
  cd.StructurallyValid ∧ a.AllLossesZero

def ConstructionDecl.UniversalCone (cd : ConstructionDecl) (a : LoweringArtifact) : Prop :=
  cd.StructurallyValid ∧ a.AllLossesZero

def ConstructionDecl.UniversalCocone (cd : ConstructionDecl) (a : LoweringArtifact) : Prop :=
  cd.StructurallyValid ∧ a.AllLossesZero

def ConstructionDecl.ParallelAgreement (cd : ConstructionDecl) (a : LoweringArtifact) : Prop :=
  cd.StructurallyValid ∧ a.AllLossesZero

def ConstructionDecl.QuotientAgreement (cd : ConstructionDecl) (a : LoweringArtifact) : Prop :=
  cd.StructurallyValid ∧ a.AllLossesZero

/-! ## Soundness: derive each property from the two decidable checks. -/

theorem ConstructionDecl.commuting_of_checks {cd : ConstructionDecl} {a : LoweringArtifact}
    (hs : cd.structurallyValidCheck = true) (hz : a.allLossesZeroCheck = true) :
    cd.CommutingSquare a :=
  ⟨hs, a.allLossesZero_of_check hz⟩

theorem ConstructionDecl.universal_cone_of_checks {cd : ConstructionDecl} {a : LoweringArtifact}
    (hs : cd.structurallyValidCheck = true) (hz : a.allLossesZeroCheck = true) :
    cd.UniversalCone a :=
  ⟨hs, a.allLossesZero_of_check hz⟩

theorem ConstructionDecl.universal_cocone_of_checks {cd : ConstructionDecl} {a : LoweringArtifact}
    (hs : cd.structurallyValidCheck = true) (hz : a.allLossesZeroCheck = true) :
    cd.UniversalCocone a :=
  ⟨hs, a.allLossesZero_of_check hz⟩

theorem ConstructionDecl.parallel_agreement_of_checks {cd : ConstructionDecl} {a : LoweringArtifact}
    (hs : cd.structurallyValidCheck = true) (hz : a.allLossesZeroCheck = true) :
    cd.ParallelAgreement a :=
  ⟨hs, a.allLossesZero_of_check hz⟩

theorem ConstructionDecl.quotient_agreement_of_checks {cd : ConstructionDecl} {a : LoweringArtifact}
    (hs : cd.structurallyValidCheck = true) (hz : a.allLossesZeroCheck = true) :
    cd.QuotientAgreement a :=
  ⟨hs, a.allLossesZero_of_check hz⟩

end FunctorFlowProofs
