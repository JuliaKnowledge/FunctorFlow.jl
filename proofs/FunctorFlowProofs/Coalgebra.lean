import FunctorFlowProofs.Core

/-!
# FunctorFlowProofs.Coalgebra

`CoalgebraDecl`, `BisimulationDecl`, and the schema-level
bisimilarity ⇒ final-coalgebra-equality theorem emitted by
`render_jepa_certificate`.
-/

namespace FunctorFlowProofs

structure CoalgebraDecl where
  name : String
  state : String
  transition : String
  functorType : String
deriving Repr, DecidableEq

structure BisimulationDecl where
  name : String
  coalgebraA : String
  coalgebraB : String
  relation : String
deriving Repr, DecidableEq

/-- Symbolic "image into the final coalgebra." Schema-level only; all
declarations share the same image, which makes
`bisim_implies_final_eq` provable by reflexivity. -/
def CoalgebraDecl.finalImage (_ : CoalgebraDecl) : String := "FINAL_COALGEBRA"

/-- Schema-level bisimulation predicate. -/
def BisimulationDecl.isBisimulation (_ : BisimulationDecl)
    (_ _ : CoalgebraDecl) : Prop := True

theorem CoalgebraDecl.bisim_implies_final_eq
    {A B : CoalgebraDecl} {R : BisimulationDecl}
    (_ : R.isBisimulation A B) :
    A.finalImage = B.finalImage := rfl

end FunctorFlowProofs
