/-!
# FunctorFlowProofs.Heyting

Machine-checked intuitionistic internal logic: a finite Heyting algebra carries
its `leq`/`meet`/`imply` tables and `isHeyting` decides the defining Heyting
adjunction `z∧x ≤ y ⇔ z ≤ (x⇒y)`. Decidable, so a certificate proves it by
`native_decide`.
-/

namespace FunctorFlowProofs

structure HeytingDecl where
  elements : List String
  leq : List ((String × String) × Bool)
  meet : List ((String × String) × String)
  imply : List ((String × String) × String)
deriving Repr

def HeytingDecl.le (H : HeytingDecl) (x y : String) : Bool :=
  ((H.leq.find? (·.1 == (x, y))).map (·.2)).getD false
def HeytingDecl.mt (H : HeytingDecl) (x y : String) : String :=
  ((H.meet.find? (·.1 == (x, y))).map (·.2)).getD ""
def HeytingDecl.imp (H : HeytingDecl) (x y : String) : String :=
  ((H.imply.find? (·.1 == (x, y))).map (·.2)).getD ""

/-- The Heyting adjunction `z∧x ≤ y ⇔ z ≤ (x⇒y)` for all `x, y, z`. -/
def HeytingDecl.isHeyting (H : HeytingDecl) : Bool :=
  H.elements.all (fun x => H.elements.all (fun y => H.elements.all (fun z =>
    H.le (H.mt z x) y == H.le z (H.imp x y))))

def HeytingDecl.IsHeyting (H : HeytingDecl) : Prop := H.isHeyting = true
theorem HeytingDecl.isHeyting_sound {H : HeytingDecl} (h : H.isHeyting = true) : H.IsHeyting := h

end FunctorFlowProofs
