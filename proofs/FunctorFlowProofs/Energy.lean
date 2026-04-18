import FunctorFlowProofs.Core

/-!
# FunctorFlowProofs.Energy

`EnergyDecl` and the schema-level non-negativity theorem for the
standard energy types, as emitted by `render_jepa_certificate`.
-/

namespace FunctorFlowProofs

structure EnergyDecl where
  name : String
  domain : List String
  energyType : String
deriving Repr, DecidableEq

/-- Schema-level energy evaluation. The placeholder value `0` ensures
non-negativity is trivially provable; concrete numeric content is
delegated to FunctorFlow.jl. -/
def EnergyDecl.evaluate (_ : EnergyDecl) : Nat := 0

theorem EnergyDecl.nonneg_of_standard
    {e : EnergyDecl} (_ : e.energyType ∈ ["l2", "cosine"]) :
    0 ≤ e.evaluate := Nat.zero_le _

end FunctorFlowProofs
