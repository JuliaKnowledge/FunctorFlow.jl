/-!
# FunctorFlowProofs.Optics

Machine-checked lens laws — the bidirectional structure behind backpropagation.
A `LensDecl` carries `get`/`put` tables; `veryWellBehaved` decides the three
very-well-behaved lens laws (GetPut, PutGet, PutPut). All decidable, so a
certificate proves `… = true` by `native_decide`.
-/

namespace FunctorFlowProofs

structure LensDecl where
  S : List String
  A : List String
  get : List (String × String)              -- S → A
  put : List ((String × String) × String)   -- (s, a) → S
deriving Repr

def LensDecl.getOf (L : LensDecl) (s : String) : String :=
  ((L.get.find? (·.1 == s)).map (·.2)).getD ""

def LensDecl.putOf (L : LensDecl) (s a : String) : String :=
  ((L.put.find? (·.1 == (s, a))).map (·.2)).getD ""

/-- GetPut: `put(s, get s) = s`. -/
def LensDecl.getPut (L : LensDecl) : Bool :=
  L.S.all (fun s => L.putOf s (L.getOf s) == s)

/-- PutGet: `get(put s a) = a`. -/
def LensDecl.putGet (L : LensDecl) : Bool :=
  L.S.all (fun s => L.A.all (fun a => L.getOf (L.putOf s a) == a))

/-- PutPut: `put(put s a) a' = put s a'`. -/
def LensDecl.putPut (L : LensDecl) : Bool :=
  L.S.all (fun s => L.A.all (fun a => L.A.all (fun a2 =>
    L.putOf (L.putOf s a) a2 == L.putOf s a2)))

def LensDecl.veryWellBehaved (L : LensDecl) : Bool :=
  L.getPut && L.putGet && L.putPut

def LensDecl.VeryWellBehaved (L : LensDecl) : Prop := L.veryWellBehaved = true
theorem LensDecl.veryWellBehaved_sound {L : LensDecl}
    (h : L.veryWellBehaved = true) : L.VeryWellBehaved := h

end FunctorFlowProofs
