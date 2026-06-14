/-!
# FunctorFlowProofs.Enriched

Machine-checked enriched category theory: a generalized (Lawvere) metric space
is a category enriched over the cost quantale. `isLawvereMetric` decides the
enriched-category axioms — identity `d(x,x) = 0` and composition (the triangle
inequality `d(x,z) ≤ d(x,y) + d(y,z)`) — and `isNonExpansive` decides whether a
map is an enriched functor (1-Lipschitz). Distances are `Nat`, so both are
decidable and a certificate proves `… = true` by `native_decide`.
-/

namespace FunctorFlowProofs

structure MetricDecl where
  points : List String
  dist : List ((String × String) × Nat)
deriving Repr

def MetricDecl.d (M : MetricDecl) (x y : String) : Nat :=
  ((M.dist.find? (·.1 == (x, y))).map (·.2)).getD 0

/-- The enriched-category axioms of a Lawvere metric space. -/
def MetricDecl.isLawvereMetric (M : MetricDecl) : Bool :=
  (M.points.all (fun x => M.d x x == 0)) &&
  (M.points.all (fun x => M.points.all (fun y => M.points.all (fun z =>
    Nat.ble (M.d x z) (M.d x y + M.d y z)))))

def MetricDecl.IsLawvereMetric (M : MetricDecl) : Prop := M.isLawvereMetric = true
theorem MetricDecl.isLawvereMetric_sound {M : MetricDecl}
    (h : M.isLawvereMetric = true) : M.IsLawvereMetric := h

/-- An enriched functor `M → N`: a non-expansive (1-Lipschitz) map `f`. -/
def isNonExpansive (M N : MetricDecl) (f : List (String × String)) : Bool :=
  M.points.all (fun x => M.points.all (fun y =>
    match (f.find? (·.1 == x)).map (·.2), (f.find? (·.1 == y)).map (·.2) with
    | some fx, some fy => Nat.ble (N.d fx fy) (M.d x y)
    | _, _ => false))

end FunctorFlowProofs
