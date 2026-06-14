/-!
# FunctorFlowProofs.Automata

Machine-checked coalgebra: Moore machines (state machines / recurrent models)
as coalgebras for `F(X) = O × X^I`. A `MooreDecl` carries the transition and
output tables; `isBisimulation` decides whether a relation is a bisimulation
(behavioural equivalence) and `isCoalgMorphism` whether a state map is a
coalgebra homomorphism. Both are decidable, so a certificate proves `… = true`
by `native_decide`.
-/

namespace FunctorFlowProofs

structure MooreDecl where
  states : List String
  inputs : List String
  transition : List ((String × String) × String)   -- (state, input) ↦ next state
  output : List (String × String)                   -- state ↦ output
deriving Repr

def MooreDecl.step (M : MooreDecl) (s i : String) : Option String :=
  (M.transition.find? (·.1 == (s, i))).map (·.2)

def MooreDecl.out (M : MooreDecl) (s : String) : Option String :=
  (M.output.find? (·.1 == s)).map (·.2)

/-- `R` is a bisimulation: related states share an output and their successors
stay related for every input. -/
def MooreDecl.isBisimulation (M : MooreDecl) (R : List (String × String)) : Bool :=
  R.all (fun pr =>
    (M.out pr.1 == M.out pr.2) &&
    M.inputs.all (fun i =>
      match M.step pr.1 i, M.step pr.2 i with
      | some a, some b => R.contains (a, b)
      | _, _ => false))

def MooreDecl.IsBisimulation (M : MooreDecl) (R : List (String × String)) : Prop :=
  M.isBisimulation R = true
theorem MooreDecl.isBisimulation_sound {M : MooreDecl} {R}
    (h : M.isBisimulation R = true) : M.IsBisimulation R := h

/-- `h` (state ↦ state) is a coalgebra homomorphism `M → N`: it preserves outputs
and commutes with transitions. -/
def isCoalgMorphism (M N : MooreDecl) (h : List (String × String)) : Bool :=
  (M.inputs == N.inputs) &&
  M.states.all (fun s =>
    match (h.find? (·.1 == s)).map (·.2) with
    | some hs =>
        (M.out s == N.out hs) &&
        M.inputs.all (fun i =>
          match M.step s i, N.step hs i with
          | some ms, some ns => ((h.find? (·.1 == ms)).map (·.2)) == some ns
          | _, _ => false)
    | none => false)

end FunctorFlowProofs
