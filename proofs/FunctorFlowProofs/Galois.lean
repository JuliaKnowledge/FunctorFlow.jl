/-!
# FunctorFlowProofs.Galois

Machine-checked Galois connections: `isGalois` decides the adjunction
`f(p) ≤ q ⇔ p ≤ g(q)` between two finite posets. Decidable, so a certificate
proves it by `native_decide`.
-/

namespace FunctorFlowProofs

structure GaloisDecl where
  P : List String
  Q : List String
  pleq : List ((String × String) × Bool)
  qleq : List ((String × String) × Bool)
  f : List (String × String)        -- P → Q
  g : List (String × String)        -- Q → P
deriving Repr

def GaloisDecl.ple (G : GaloisDecl) (x y : String) : Bool :=
  ((G.pleq.find? (·.1 == (x, y))).map (·.2)).getD false
def GaloisDecl.qle (G : GaloisDecl) (x y : String) : Bool :=
  ((G.qleq.find? (·.1 == (x, y))).map (·.2)).getD false
def GaloisDecl.fOf (G : GaloisDecl) (p : String) : String :=
  ((G.f.find? (·.1 == p)).map (·.2)).getD ""
def GaloisDecl.gOf (G : GaloisDecl) (q : String) : String :=
  ((G.g.find? (·.1 == q)).map (·.2)).getD ""

/-- The Galois adjunction `f(p) ≤ q ⇔ p ≤ g(q)` for all `p ∈ P, q ∈ Q`. -/
def GaloisDecl.isGalois (G : GaloisDecl) : Bool :=
  G.P.all (fun p => G.Q.all (fun q =>
    G.qle (G.fOf p) q == G.ple p (G.gOf q)))

def GaloisDecl.IsGalois (G : GaloisDecl) : Prop := G.isGalois = true
theorem GaloisDecl.isGalois_sound {G : GaloisDecl} (h : G.isGalois = true) : G.IsGalois := h

end FunctorFlowProofs
