/-!
# FunctorFlowProofs.Cat

Machine-checked certification of the `FunctorFlow.Cat` kernel. A finite
category is emitted as a `CatTable` (objects, morphisms, a composition table
and identities) and a finite functor as a `FunctorDecl`; the decidable
`isCategory` / `isFunctor` predicates encode the category and functor laws, so a
certificate proves `… = true` by `native_decide` — a kernel-checked proof of the
very laws the Julia kernel verifies at runtime by enumeration.

Composition is diagrammatic: an entry `(f, g, h)` in `comp` means `f` then `g`
equals `h` (i.e. `g ∘ f = h`).
-/

namespace FunctorFlowProofs

structure MorDecl where
  name : String
  dom : String
  cod : String
deriving Repr, DecidableEq

structure CatTable where
  objects : List String
  morphisms : List MorDecl
  comp : List (String × String × String)   -- (f, g, h):  f then g = h
  ids : List (String × String)             -- (object, identity-morphism name)
deriving Repr

def CatTable.find? (C : CatTable) (m : String) : Option MorDecl :=
  C.morphisms.find? (·.name == m)

def CatTable.idOf? (C : CatTable) (o : String) : Option String :=
  (C.ids.find? (·.1 == o)).map (·.2)

def CatTable.compOf? (C : CatTable) (f g : String) : Option String :=
  (C.comp.find? (fun t => t.1 == f && t.2.1 == g)).map (·.2.2)

/-- The category axioms as a decidable Boolean check over the finite table. -/
def CatTable.isCategory (C : CatTable) : Bool :=
  -- endpoints of every morphism are declared objects
  C.morphisms.all (fun m => C.objects.contains m.dom && C.objects.contains m.cod) &&
  -- every object has a correctly-typed identity
  C.objects.all (fun o =>
    match C.idOf? o with
    | some i => match C.find? i with
                | some im => im.dom == o && im.cod == o
                | none => false
    | none => false) &&
  -- composition is total and correctly typed on composable pairs
  C.morphisms.all (fun f => C.morphisms.all (fun g =>
    if f.cod == g.dom then
      match C.compOf? f.name g.name with
      | some h => match C.find? h with
                  | some hm => hm.dom == f.dom && hm.cod == g.cod
                  | none => false
      | none => false
    else true)) &&
  -- left / right identity laws
  C.morphisms.all (fun f =>
    (match C.idOf? f.dom with | some i => C.compOf? i f.name == some f.name | none => false) &&
    (match C.idOf? f.cod with | some i => C.compOf? f.name i == some f.name | none => false)) &&
  -- associativity on composable triples
  C.morphisms.all (fun f => C.morphisms.all (fun g => C.morphisms.all (fun h =>
    if f.cod == g.dom && g.cod == h.dom then
      match C.compOf? f.name g.name, C.compOf? g.name h.name with
      | some fg, some gh => C.compOf? fg h.name == C.compOf? f.name gh
      | _, _ => false
    else true)))

/-- A `CatTable` satisfying the axioms (the certificate-level statement). -/
def CatTable.IsCategory (C : CatTable) : Prop := C.isCategory = true

theorem CatTable.isCategory_sound {C : CatTable} (h : C.isCategory = true) : C.IsCategory := h

structure FunctorDecl where
  dom : CatTable
  cod : CatTable
  obMap : List (String × String)
  morMap : List (String × String)
deriving Repr

def FunctorDecl.obOf? (F : FunctorDecl) (o : String) : Option String :=
  (F.obMap.find? (·.1 == o)).map (·.2)

def FunctorDecl.morOf? (F : FunctorDecl) (m : String) : Option String :=
  (F.morMap.find? (·.1 == m)).map (·.2)

/-- The functor laws as a decidable Boolean check. -/
def FunctorDecl.isFunctor (F : FunctorDecl) : Bool :=
  -- object map total, landing in cod objects
  F.dom.objects.all (fun o =>
    match F.obOf? o with | some o2 => F.cod.objects.contains o2 | none => false) &&
  -- morphism map total and typed: F f : F (dom f) → F (cod f)
  F.dom.morphisms.all (fun f =>
    match F.morOf? f.name, F.obOf? f.dom, F.obOf? f.cod with
    | some f2, some a, some b =>
        match F.cod.find? f2 with
        | some fm => fm.dom == a && fm.cod == b
        | none => false
    | _, _, _ => false) &&
  -- preserves identities
  F.dom.objects.all (fun o =>
    match F.dom.idOf? o, F.obOf? o with
    | some i, some o2 =>
        match F.morOf? i, F.cod.idOf? o2 with
        | some fi, some j => fi == j
        | _, _ => false
    | _, _ => false) &&
  -- preserves composition: F (f then g) = (F f) then (F g)
  F.dom.morphisms.all (fun f => F.dom.morphisms.all (fun g =>
    if f.cod == g.dom then
      match F.dom.compOf? f.name g.name with
      | some fg =>
          match F.morOf? f.name, F.morOf? g.name, F.morOf? fg with
          | some Ff, some Fg, some Ffg => F.cod.compOf? Ff Fg == some Ffg
          | _, _, _ => false
      | none => false
    else true))

def FunctorDecl.IsFunctor (F : FunctorDecl) : Prop := F.isFunctor = true

theorem FunctorDecl.isFunctor_sound {F : FunctorDecl} (h : F.isFunctor = true) : F.IsFunctor := h

/-! ## Adjunctions and monads. -/

def alist? (l : List (String × String)) (k : String) : Option String :=
  (l.find? (·.1 == k)).map (·.2)

/-- An adjunction `F ⊣ G` with unit `η` (object `c` of `C` ↦ `η_c : c → GF c`)
and counit `ε` (object `d` of `D` ↦ `ε_d : FG d → d`). -/
structure AdjunctionDecl where
  F : FunctorDecl                       -- C → D
  G : FunctorDecl                       -- D → C
  unit : List (String × String)         -- c ↦ η_c   (morphism of C)
  counit : List (String × String)       -- d ↦ ε_d   (morphism of D)
deriving Repr

/-- Decidable adjunction check: `F`, `G` are functors and both triangle
identities hold. (Composition is diagrammatic: `compOf? f g = g ∘ f`.) -/
def AdjunctionDecl.isAdjunction (A : AdjunctionDecl) : Bool :=
  let C := A.F.dom; let D := A.F.cod
  A.F.isFunctor && A.G.isFunctor &&
  -- triangle 1:  ε_{F c} ∘ F(η_c) = id_{F c}
  C.objects.all (fun c =>
    match alist? A.unit c, A.F.obOf? c with
    | some ηc, some Fc =>
        match A.F.morOf? ηc, alist? A.counit Fc, D.idOf? Fc with
        | some Fηc, some εFc, some idFc => D.compOf? Fηc εFc == some idFc
        | _, _, _ => false
    | _, _ => false) &&
  -- triangle 2:  G(ε_d) ∘ η_{G d} = id_{G d}
  D.objects.all (fun d =>
    match A.G.obOf? d with
    | some Gd =>
        match alist? A.unit Gd, alist? A.counit d with
        | some ηGd, some εd =>
            match A.G.morOf? εd, C.idOf? Gd with
            | some Gεd, some idGd => C.compOf? ηGd Gεd == some idGd
            | _, _ => false
        | _, _ => false
    | none => false)

def AdjunctionDecl.IsAdjunction (A : AdjunctionDecl) : Prop := A.isAdjunction = true

theorem AdjunctionDecl.isAdjunction_sound {A : AdjunctionDecl}
    (h : A.isAdjunction = true) : A.IsAdjunction := h

/-- A monad `(T, η, μ)` on a category. `unit` is `c ↦ η_c : c → T c`,
`mult` is `c ↦ μ_c : T(T c) → T c`. -/
structure MonadDecl where
  T : FunctorDecl                       -- endofunctor C → C
  unit : List (String × String)
  mult : List (String × String)
deriving Repr

/-- Decidable monad check: `T` is a functor and the unit and associativity
laws hold (`μ_c ∘ η_{Tc} = id`, `μ_c ∘ T(η_c) = id`, `μ_c ∘ μ_{Tc} = μ_c ∘ T(μ_c)`). -/
def MonadDecl.isMonad (M : MonadDecl) : Bool :=
  let C := M.T.dom
  M.T.isFunctor &&
  C.objects.all (fun c =>
    match M.T.obOf? c with
    | some Tc =>
        match alist? M.unit c, alist? M.unit Tc, alist? M.mult c, C.idOf? Tc with
        | some ηc, some ηTc, some μc, some idTc =>
            (C.compOf? ηTc μc == some idTc) &&
            (match M.T.morOf? ηc with
             | some Tηc => C.compOf? Tηc μc == some idTc
             | none => false) &&
            (match alist? M.mult Tc, M.T.morOf? μc with
             | some μTc, some Tμc => C.compOf? μTc μc == C.compOf? Tμc μc
             | _, _ => false)
        | _, _, _, _ => false
    | none => false)

def MonadDecl.IsMonad (M : MonadDecl) : Prop := M.isMonad = true

theorem MonadDecl.isMonad_sound {M : MonadDecl} (h : M.isMonad = true) : M.IsMonad := h

end FunctorFlowProofs
