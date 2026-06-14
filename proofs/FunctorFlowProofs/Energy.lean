import FunctorFlowProofs.Core

/-!
# FunctorFlowProofs.Energy

`EnergyDecl` and the schema-level facts emitted by `render_jepa_certificate`
for energy-based cost components.

`evaluate` now returns the certificate's *carried* discretised energy `value`
(not a hard-coded `0`). Energy values are modelled as `Nat` (this Lake
project is Mathlib-free, so genuine ℝ is unavailable); non-negativity is
therefore a real fact about the carried datum (`Nat.zero_le`), and
`Compatible` (zero energy = compatible pair) is a falsifiable predicate.
-/

namespace FunctorFlowProofs

structure EnergyDecl where
  name : String
  domain : List String
  energyType : String
  /-- Discretised energy magnitude carried by the certificate (0 = a
  perfectly compatible / predicted pair). -/
  value : Nat := 0
deriving Repr, DecidableEq

/-- The carried energy magnitude. -/
def EnergyDecl.evaluate (e : EnergyDecl) : Nat := e.value

/-- Energy is non-negative — a genuine fact about the carried `Nat` value
(for any energy type). -/
theorem EnergyDecl.nonneg (e : EnergyDecl) : 0 ≤ e.evaluate := Nat.zero_le _

/-- Non-negativity for the standard energy families. Kept for the emitter's
`energy_nonneg` theorem; broadened to the self-supervised energies. -/
theorem EnergyDecl.nonneg_of_standard {e : EnergyDecl}
    (_ : e.energyType ∈ ["l2", "cosine", "smooth_l1", "vicreg", "barlow_twins", "contrastive"]) :
    0 ≤ e.evaluate := Nat.zero_le _

/-- A pair is *compatible* (low-energy) iff the carried energy is zero. -/
def EnergyDecl.Compatible (e : EnergyDecl) : Prop := e.value = 0

/-- Decidable check for `Compatible`. -/
def EnergyDecl.compatibleCheck (e : EnergyDecl) : Bool := e.value == 0

theorem EnergyDecl.compatible_of_check {e : EnergyDecl}
    (h : e.compatibleCheck = true) : e.Compatible :=
  Nat.eq_of_beq_eq_true h

end FunctorFlowProofs
