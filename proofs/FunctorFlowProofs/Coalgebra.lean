import FunctorFlowProofs.Core

/-!
# FunctorFlowProofs.Coalgebra

`CoalgebraDecl`, `BisimulationDecl`, and the well-formedness fact emitted by
`render_jepa_certificate`.

The earlier `bisim_implies_final_eq` "theorem" proved
`A.finalImage = B.finalImage` by `rfl` over a *constant* `finalImage`, so it
held for any two declarations whatsoever — no content. That is removed.

In its place, `BisimulationDecl.WellFormed` is a genuine, falsifiable
predicate: a recorded bisimulation must name both of its coalgebras and carry
a non-empty relation witness. (A Mathlib-free Lake project cannot host the
full final-coalgebra/behavioural-equivalence statement; the honest
schema-level claim is structural well-formedness, with the numeric exactness
content carried by `LoweringArtifact.CoalgebraExact` in `Core`.)
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

/-- A recorded bisimulation is well-formed when it names both related
coalgebras and carries a non-empty relation witness. Falsifiable: empty
endpoints or relation make it false. -/
def BisimulationDecl.WellFormed (R : BisimulationDecl) : Prop :=
  R.coalgebraA ≠ "" ∧ R.coalgebraB ≠ "" ∧ R.relation ≠ ""

/-- Decidable check for `WellFormed`. -/
def BisimulationDecl.wellFormedCheck (R : BisimulationDecl) : Bool :=
  (R.coalgebraA != "") && (R.coalgebraB != "") && (R.relation != "")

theorem BisimulationDecl.wellFormed_of_check {R : BisimulationDecl}
    (h : R.wellFormedCheck = true) : R.WellFormed := by
  rw [BisimulationDecl.wellFormedCheck] at h
  -- decompose the conjunction of Boolean inequalities
  have h' := h
  rw [Bool.and_eq_true, Bool.and_eq_true] at h'
  obtain ⟨⟨ha, hb⟩, hr⟩ := h'
  refine ⟨?_, ?_, ?_⟩
  · exact fun he => by simp [he] at ha
  · exact fun he => by simp [he] at hb
  · exact fun he => by simp [he] at hr

end FunctorFlowProofs
