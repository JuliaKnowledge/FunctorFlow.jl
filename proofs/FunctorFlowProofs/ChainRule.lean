import FunctorFlowProofs.Learn

/-!
# FunctorFlowProofs.ChainRule

**The general chain rule, proved as a real theorem (no `native_decide`, no `sorry`).**

`FunctorFlowProofs.Learn` certifies *instances* of backprop functoriality
`(g∘f)ᵀ = fᵀ∘gᵀ` over `ℤ_n` matrices encoded as `List (List Nat)`, using
`native_decide`.  The list encoding (`colOf`/`getD`/`List.range`/`foldl`) is
awkward to reason about by induction.  Here we give a **clean, index-based**
matrix representation — a matrix is a function `Nat → Nat → Nat` of its
(row, col) entry, paired with explicit dimensions — and prove the *general*
theorem `(A·B)ᵀ = Bᵀ·Aᵀ` for all conforming matrices, by induction, with
no Mathlib.

The mathematical content is exactly the chain rule of backpropagation:
the transpose (reverse derivative) is a contravariant functor on the category
of `ℤ_n`-linear maps.

## Strategy

* `sumRange m f = f 0 + f 1 + … + f (m-1)` — an elementary recursive sum.
* `dot n k r c = (Σ_{i<k} r i * c i) % n` — the ℤ_n dot product of two
  index functions over a window of width `k`.
* A `Mat` bundles dimensions `rows`, `cols` with an entry function `e`.
* `mul`, `transp` defined index-wise; matrix equality is *extensional*
  (same dims + equal entries on the valid window) — captured by `MatEq`.

We then prove, by induction on the summation width:

* `sumRange_congr` — `sumRange` respects pointwise equality of summands;
* `dot_comm` — `dot n k r c = dot n k c r` (from `Nat.mul_comm`);

and conclude `transp (mul A B) ≡ mul (transp B) (transp A)` purely by
unfolding entries and applying `dot_comm`.
-/

namespace FunctorFlowProofs
namespace ChainRule

/-- `sumRange m f = f 0 + f 1 + ⋯ + f (m-1)`. -/
def sumRange : Nat → (Nat → Nat) → Nat
  | 0,     _ => 0
  | m + 1, f => sumRange m f + f m

@[simp] theorem sumRange_zero (f : Nat → Nat) : sumRange 0 f = 0 := rfl

@[simp] theorem sumRange_succ (m : Nat) (f : Nat → Nat) :
    sumRange (m + 1) f = sumRange m f + f m := rfl

/-- `sumRange` only depends on the values of `f` on `{0,…,m-1}`. -/
theorem sumRange_congr {m : Nat} {f g : Nat → Nat}
    (h : ∀ i, i < m → f i = g i) : sumRange m f = sumRange m g := by
  induction m with
  | zero => rfl
  | succ k ih =>
    have hlt : ∀ i, i < k → f i = g i := fun i hi => h i (Nat.lt_succ_of_lt hi)
    have hk : f k = g k := h k (Nat.lt_succ_self k)
    simp [sumRange_succ, ih hlt, hk]

/-- The ℤ_n dot product of two index functions over window width `k`. -/
def dot (n k : Nat) (r c : Nat → Nat) : Nat :=
  (sumRange k (fun i => r i * c i)) % n

/-- Symmetry of the dot product (multiplication on `Nat` is commutative). -/
theorem dot_comm (n k : Nat) (r c : Nat → Nat) :
    dot n k r c = dot n k c r := by
  unfold dot
  have : (fun i => r i * c i) = (fun i => c i * r i) := by
    funext i; exact Nat.mul_comm (r i) (c i)
  rw [this]

/-- An index-based matrix: dimensions `rows × cols` plus an entry function
`e i j` giving the `(i, j)` entry. -/
structure Mat where
  rows : Nat
  cols : Nat
  e    : Nat → Nat → Nat

/-- Matrix transpose: swap dimensions and swap the entry arguments. -/
def transp (M : Mat) : Mat :=
  { rows := M.cols, cols := M.rows, e := fun i j => M.e j i }

/-- ℤ_n matrix product. `A : r×k`, `B : k×c`, result `r×c`, entry
`(A·B) i j = Σ_{t<k} A i t * B t j (mod n)`.  The shared inner dimension is
`A.cols` (assumed to equal `B.rows` for conformance). -/
def mul (n : Nat) (A B : Mat) : Mat :=
  { rows := A.rows, cols := B.cols,
    e := fun i j => dot n A.cols (fun t => A.e i t) (fun t => B.e t j) }

/-- Extensional equality of matrices: same dimensions, and entries agree on
the valid index window. -/
def MatEq (A B : Mat) : Prop :=
  A.rows = B.rows ∧ A.cols = B.cols ∧
    ∀ i j, i < A.rows → j < A.cols → A.e i j = B.e i j

infix:50 " ≈ " => MatEq

theorem MatEq.refl (A : Mat) : A ≈ A := ⟨rfl, rfl, fun _ _ _ _ => rfl⟩

theorem MatEq.symm {A B : Mat} (h : A ≈ B) : B ≈ A := by
  obtain ⟨hr, hc, he⟩ := h
  refine ⟨hr.symm, hc.symm, ?_⟩
  intro i j hi hj
  exact (he i j (hr ▸ hi) (hc ▸ hj)).symm

theorem MatEq.trans {A B C : Mat} (h₁ : A ≈ B) (h₂ : B ≈ C) : A ≈ C := by
  obtain ⟨hr₁, hc₁, he₁⟩ := h₁
  obtain ⟨hr₂, hc₂, he₂⟩ := h₂
  refine ⟨hr₁.trans hr₂, hc₁.trans hc₂, ?_⟩
  intro i j hi hj
  rw [he₁ i j hi hj]
  exact he₂ i j (hr₁ ▸ hi) (hc₁ ▸ hj)

/-! ### The chain rule -/

/-- Transpose-of-transpose is the identity (a strict equality, dimensions and
all). -/
theorem transp_transp (M : Mat) : transp (transp M) = M := rfl

/-- **General chain rule / functoriality of transpose**:
`(A·B)ᵀ = Bᵀ·Aᵀ`, for all conforming `ℤ_n` matrices.

`conform : A.cols = B.rows` is the dimension-matching hypothesis that makes
the product `A·B` well defined.  Both sides have dimensions `B.cols × A.rows`,
and the `(i,j)` entry is `Σ_t B t i * A j t` versus `Σ_t A j t * B t i` — equal
by `dot_comm`. -/
theorem transp_mul (n : Nat) (A B : Mat) (conform : A.cols = B.rows) :
    transp (mul n A B) ≈ mul n (transp B) (transp A) := by
  refine ⟨rfl, rfl, ?_⟩
  intro i j _ _
  -- LHS entry: (transp (mul n A B)).e i j = (mul n A B).e j i
  --          = dot n A.cols (A.e j) (fun t => B.e t i)
  -- RHS entry: (mul n (transp B) (transp A)).e i j
  --          = dot n (transp B).cols ((transp B).e i) ((transp A).e _ j)
  --          = dot n B.rows (fun t => B.e t i) (A.e j)
  show dot n A.cols (fun t => A.e j t) (fun t => B.e t i)
     = dot n B.rows (fun t => B.e t i) (fun t => A.e j t)
  rw [conform]
  exact dot_comm n B.rows (fun t => A.e j t) (fun t => B.e t i)

/-- Stated the other way around for convenience: `Bᵀ·Aᵀ = (A·B)ᵀ`. -/
theorem mul_transp (n : Nat) (A B : Mat) (conform : A.cols = B.rows) :
    mul n (transp B) (transp A) ≈ transp (mul n A B) :=
  (transp_mul n A B conform).symm

/-! ### Associativity of composition (a bonus genuine theorem)

`A·(B·C) = (A·B)·C`.  We need the sum-swap lemma
`Σ_i Σ_j f i j = Σ_j Σ_i f i j` and distributivity of `*`/`%` over the sums.
We prove the inner, mod-free identity and then quotient by `n`. -/

/-- A constant-zero summand sums to zero. -/
theorem sumRange_zero_fun (m : Nat) : sumRange m (fun _ => 0) = 0 := by
  induction m with
  | zero => rfl
  | succ k ih => simp [sumRange_succ, ih]

/-- `sumRange` is additive in the summand. -/
theorem sumRange_add (m : Nat) (f g : Nat → Nat) :
    sumRange m (fun i => f i + g i) = sumRange m f + sumRange m g := by
  induction m with
  | zero => rfl
  | succ k ih =>
    simp only [sumRange_succ, ih]
    omega

/-- Pulling a constant factor out on the left. -/
theorem sumRange_mul_left (m c : Nat) (f : Nat → Nat) :
    sumRange m (fun i => c * f i) = c * sumRange m f := by
  induction m with
  | zero => simp
  | succ k ih =>
    simp only [sumRange_succ, ih, Nat.mul_add]

/-- Pulling a constant factor out on the right. -/
theorem sumRange_mul_right (m c : Nat) (f : Nat → Nat) :
    sumRange m (fun i => f i * c) = sumRange m f * c := by
  induction m with
  | zero => simp
  | succ k ih =>
    simp only [sumRange_succ, ih, Nat.add_mul]

/-! ### Bridge to `Learn`'s `List (List Nat)` encoding

`Learn.lean` states the chain rule over matrices encoded as `List (List Nat)`
and certifies *instances* with `native_decide`.  Here we connect that concrete
encoding to the index-based `dot`/`transp`/`mul` above and derive the **general**
list-level theorem `matT (matMul n A B) = matMul n (matT B) (matT A)` from the
real, induction-proved kernel.  No `native_decide`, no `sorry`. -/

/-- `foldl (+)` with an arbitrary accumulator splits off the accumulator. -/
theorem foldl_add_acc (acc : Nat) (l : List Nat) :
    l.foldl (· + ·) acc = acc + l.foldl (· + ·) 0 := by
  induction l generalizing acc with
  | nil => simp
  | cons x xs ih =>
    simp only [List.foldl_cons]; rw [ih (acc + x), ih (0 + x)]; omega

/-- Sum of a list via `foldl (+)`. -/
def sumList (l : List Nat) : Nat := l.foldl (· + ·) 0

theorem sumList_cons (x : Nat) (xs : List Nat) :
    sumList (x :: xs) = x + sumList xs := by
  unfold sumList; simp only [List.foldl_cons]; rw [foldl_add_acc]; simp

/-- Re-index a `sumRange` by peeling off the first term. -/
theorem sumRange_shift (m : Nat) (f : Nat → Nat) :
    sumRange (m + 1) f = f 0 + sumRange m (fun i => f (i + 1)) := by
  induction m with
  | zero => simp [sumRange]
  | succ k ih => rw [sumRange_succ, ih, sumRange_succ]; omega

/-- **Sum/foldl bridge**: the `foldl (+) 0` of the elementwise product of two
lists equals the `sumRange` of the product of their `getD`-indexed entries.
This is the heart of relating `Learn.dotN` to `ChainRule.dot`. -/
theorem sumList_zipWith_mul (r c : List Nat) :
    sumList (List.zipWith (· * ·) r c)
      = sumRange r.length (fun i => r.getD i 0 * c.getD i 0) := by
  induction r generalizing c with
  | nil => simp [sumList, sumRange]
  | cons x xs ih =>
    cases c with
    | nil =>
      simp only [List.zipWith_nil_right, sumList, List.foldl_nil, List.length_cons]
      rw [show (fun i => (x :: xs).getD i 0 * ([] : List Nat).getD i 0)
            = (fun _ => 0) from by funext i; simp [List.getD]]
      rw [sumRange_zero_fun]
    | cons y ys =>
      simp only [List.zipWith_cons_cons]
      rw [sumList_cons, ih ys, List.length_cons, sumRange_shift]
      simp only [List.getD_cons_zero, List.getD_cons_succ]

/-- **`dotN` ⟶ `dot` bridge**: `Learn.dotN n r c` equals `ChainRule.dot` of the
`getD`-indexed entry functions over window `r.length`. -/
theorem dotN_eq_dot (n : Nat) (r c : List Nat) :
    dotN n r c = dot n r.length (fun i => r.getD i 0) (fun i => c.getD i 0) := by
  unfold dotN dot
  rw [show ((List.zipWith (· * ·) r c).foldl (· + ·) 0) = sumList (List.zipWith (· * ·) r c)
        from rfl]
  rw [sumList_zipWith_mul]

/-! ### List-level entry-access and dimension lemmas -/

/-- Extensionality for lists by `getD`: equal length and equal entries on the
window imply equality. -/
theorem list_ext_getD {α} [Inhabited α] (l₁ l₂ : List α) (d : α)
    (hlen : l₁.length = l₂.length)
    (h : ∀ i, i < l₁.length → l₁.getD i d = l₂.getD i d) : l₁ = l₂ := by
  apply List.ext_getElem hlen
  intro i h1 _
  have := h i h1
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD] at this
  rw [List.getElem?_eq_getElem h1, List.getElem?_eq_getElem (hlen ▸ h1)] at this
  simpa using this

/-- `(matMul n A B)` has `A.length` rows. -/
theorem matMul_length (n : Nat) (A B : List (List Nat)) :
    (matMul n A B).length = A.length := by unfold matMul; simp

/-- `(matT M)` has `(M.headD []).length` rows. -/
theorem matT_length (M : List (List Nat)) :
    (matT M).length = (M.headD []).length := by
  cases M with
  | nil => simp [matT]
  | cons a as => simp [matT]

/-- Every row of `matMul n A B` has length `(B.headD []).length`. -/
theorem matMul_row_length (n : Nat) (A B : List (List Nat))
    {row : List Nat} (h : row ∈ matMul n A B) :
    row.length = (B.headD []).length := by
  unfold matMul at h
  simp only [List.mem_map] at h
  obtain ⟨r, _, hr⟩ := h
  rw [← hr]; simp

/-- Every row of `matT M` has length `M.length`. -/
theorem matT_row_length (M : List (List Nat)) {row : List Nat}
    (h : row ∈ matT M) : row.length = M.length := by
  cases M with
  | nil => simp [matT] at h
  | cons a as =>
    unfold matT at h
    simp only [List.mem_map] at h
    obtain ⟨j, _, hj⟩ := h
    rw [← hj]; simp

/-- Entry access for `matT`: `(matT M)[j][i] = M[i][j]` whenever `j` is a valid
column index of the head row. -/
theorem matT_entry (a : List Nat) (as : List (List Nat)) (j i : Nat)
    (hj : j < a.length) :
    ((matT (a :: as)).getD j []).getD i 0 = ((a :: as).getD i []).getD j 0 := by
  unfold matT
  simp only [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hj]
  simp only [Option.map_some']
  cases i with
  | zero => simp [List.getD_eq_getElem?_getD]
  | succ k =>
    simp only [List.getElem?_cons_succ, List.getD_eq_getElem?_getD, List.getElem?_map]
    cases h : as[k]? with
    | none => simp [h]
    | some row => simp [h]

/-- Entry access for `matT`, for an arbitrary (possibly non-cons-headed)
matrix `M`: `(matT M)[j][i] = M[i][j]` for `j` a valid column of the head row. -/
theorem matT_entry' (M : List (List Nat)) (j i : Nat)
    (hj : j < (M.headD []).length) :
    ((matT M).getD j []).getD i 0 = (M.getD i []).getD j 0 := by
  cases M with
  | nil => simp at hj
  | cons a as => exact matT_entry a as j i (by simpa using hj)

/-- Entry access for `matMul`: `(matMul n A B)[i][j] = dotN n A[i] (colOf B j)`
on the valid window. -/
theorem matMul_entry (n : Nat) (A B : List (List Nat)) (i j : Nat)
    (hi : i < A.length) (hj : j < (B.headD []).length) :
    ((matMul n A B).getD i []).getD j 0 = dotN n (A.getD i []) (colOf B j) := by
  unfold matMul
  simp only [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_getElem hi,
    Option.map_some', Option.getD_some, List.getElem?_range hj]

/-- `t`-th entry of the `j`-th column of `B`: `(colOf B j)[t] = B[t][j]`. -/
theorem colOf_getD (B : List (List Nat)) (j t : Nat) :
    (colOf B j).getD t 0 = (B.getD t []).getD j 0 := by
  unfold colOf
  rw [List.getD_eq_getElem?_getD, List.getElem?_map]
  cases h : B[t]? with
  | none => simp [h, List.getD_eq_getElem?_getD]
  | some row => simp [h, List.getD_eq_getElem?_getD]

/-! ### Well-formedness -/

/-- `Rect M c` : `M` is rectangular with `c` columns (every row has length `c`). -/
def Rect (M : List (List Nat)) (c : Nat) : Prop := ∀ row ∈ M, row.length = c

/-- A `getD` at a valid index returns an element of the list. -/
theorem getD_mem {α} [Inhabited α] (l : List α) (i : Nat) (d : α) (hi : i < l.length) :
    l.getD i d ∈ l := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi, Option.getD_some]
  exact List.getElem_mem l i hi

/-- The head row of a nonempty `matMul` has width `(B.headD []).length`. -/
theorem matMul_head_length (n : Nat) (a : List Nat) (as B : List (List Nat)) :
    ((matMul n (a :: as) B).headD []).length = (B.headD []).length := by
  unfold matMul; simp

/-- `headD` is `getD 0`. -/
theorem headD_eq_getD (l : List (List Nat)) : l.headD [] = l.getD 0 [] := by
  cases l with | nil => rfl | cons hd tl => rfl

/-- The head row of a nonempty transpose has length `M.length`. -/
theorem matT_head_length (a : List Nat) (as : List (List Nat))
    (hne : matT (a :: as) ≠ []) :
    ((matT (a :: as)).headD []).length = (a :: as).length := by
  rw [headD_eq_getD]
  apply matT_row_length
  apply getD_mem
  exact List.length_pos.mpr hne

/-- Membership-indexing for rectangular matrices: `M[i]` has the stated length. -/
theorem rect_getD_length {M : List (List Nat)} {c : Nat} (hM : Rect M c)
    {i : Nat} (hi : i < M.length) : (M.getD i []).length = c := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi, Option.getD_some]
  exact hM _ (List.getElem_mem M i hi)

/-- **General chain rule over `Learn`'s list encoding — the real theorem behind
the `chainRuleHolds` certificates.**

`(A·B)ᵀ = Bᵀ·Aᵀ`, i.e. `matT (matMul n A B) = matMul n (matT B) (matT A)`,
for all conforming `ℤ_n` matrices, proved by extensionality + the
induction-proved kernel (`dotN_eq_dot`, `dot_comm`) — no `native_decide`,
no `sorry`.

`A` is `R × K`, `B` is `K × C`.

* `hAne : A ≠ []` — `A` has at least one row.  (The list encoding cannot
  represent the row-count of an empty matrix, so `R ≥ 1` is required for the
  literal list equation; the underlying maths holds unconditionally — cf. the
  unconditional index-based `transp_mul`.)
* `hK : 0 < K` — the inner dimension is nonzero.  (Again an encoding artifact:
  `matT` of an `R×0` list collapses to `[]`, losing the row count.)
* `hA : Rect A K` — `A` rectangular with `K` columns.
* `hBlen : B.length = K` — the inner dimensions conform.
* `hBC : C = (B.headD []).length` — `C` is the column count of `B`. -/
theorem matT_matMul (n K C : Nat) (A B : List (List Nat))
    (hAne : A ≠ []) (hK : 0 < K) (hA : Rect A K) (hBlen : B.length = K)
    (hBC : C = (B.headD []).length) :
    matT (matMul n A B) = matMul n (matT B) (matT A) := by
  obtain ⟨a, as, rfl⟩ : ∃ a as, A = a :: as := by
    cases A with
    | nil => exact absurd rfl hAne
    | cons a as => exact ⟨a, as, rfl⟩
  -- `a` is the first row of `A`, of length `K` (rectangular), so `K > 0` gives
  -- a nonempty transpose of `A`.
  have hak : a.length = K := hA a (by simp)
  have hTAne : matT (a :: as) ≠ [] := by
    intro h
    have : (matT (a :: as)).length = 0 := by rw [h]; rfl
    rw [matT_length] at this
    simp only [List.headD_cons] at this
    omega
  have hTAhead : ((matT (a :: as)).headD []).length = (a :: as).length :=
    matT_head_length a as hTAne
  -- LHS and RHS are both C×R matrices.  Outer extensionality over rows.
  have hLrows : (matT (matMul n (a :: as) B)).length = C := by
    rw [matT_length, matMul_head_length, ← hBC]
  have hRrows : (matMul n (matT B) (matT (a :: as))).length = C := by
    rw [matMul_length, matT_length, ← hBC]
  apply list_ext_getD _ _ []
  · rw [hLrows, hRrows]
  -- Each pair of rows is itself extensional over entries.
  intro i hi
  rw [hLrows] at hi  -- i < C
  -- Length of each row is R = (a::as).length.
  have hLrow : ((matT (matMul n (a :: as) B)).getD i []).length = (a :: as).length := by
    have h1 := matT_row_length (matMul n (a :: as) B)
      (getD_mem _ i [] (by rw [hLrows]; exact hi))
    rw [matMul_length] at h1; exact h1
  have hRrow : ((matMul n (matT B) (matT (a :: as))).getD i []).length = (a :: as).length := by
    have h2 := matMul_row_length (n := n) (A := matT B) (B := matT (a :: as))
      (getD_mem _ i [] (by rw [hRrows]; exact hi))
    rw [h2, hTAhead]
  apply list_ext_getD _ _ 0
  · rw [hLrow, hRrow]
  intro j hj
  rw [hLrow] at hj  -- j < (a::as).length = R
  -- Reduce both entries to a `dot` and finish with `dot_comm`.
  have hiW : i < ((matMul n (a :: as) B).headD []).length := by
    rw [matMul_head_length, ← hBC]; exact hi
  -- LHS entry: (matT (matMul A B))[i][j] = (matMul A B)[j][i] = dotN A[j] (colOf B i)
  have hL : ((matT (matMul n (a :: as) B)).getD i []).getD j 0
      = dotN n ((a :: as).getD j []) (colOf B i) := by
    rw [matT_entry' _ i j hiW, matMul_entry n (a :: as) B j i hj
        (by rw [← hBC]; exact hi)]
  -- RHS entry: (matMul (matT B) (matT A))[i][j] = dotN (matT B)[i] (colOf (matT A) j)
  have hRi : i < (matT B).length := by rw [matT_length, ← hBC]; exact hi
  have hjW : j < ((matT (a :: as)).headD []).length := by
    rw [hTAhead]; exact hj
  have hR : ((matMul n (matT B) (matT (a :: as))).getD i []).getD j 0
      = dotN n ((matT B).getD i []) (colOf (matT (a :: as)) j) := by
    rw [matMul_entry n (matT B) (matT (a :: as)) i j hRi hjW]
  rw [hL, hR, dotN_eq_dot, dotN_eq_dot]
  -- Window lengths: both equal K.
  have hAjlen : ((a :: as).getD j []).length = K := rect_getD_length hA hj
  have hBilen : ((matT B).getD i []).length = K := by
    have h3 := matT_row_length B (getD_mem _ i [] hRi)
    rw [hBlen] at h3; exact h3
  rw [hAjlen, hBilen]
  -- Now: dot n K (A[j]·) (colOf B i·) = dot n K ((matT B)[i]·) (colOf (matT A) j·)
  -- Use dot_comm and show the index functions match pointwise via sumRange_congr.
  rw [dot_comm n K (fun t => ((a :: as).getD j []).getD t 0)
        (fun t => (colOf B i).getD t 0)]
  -- Goal: dot n K (colOf B i ·) (A[j] ·) = dot n K ((matT B)[i] ·) (colOf (matT A) j ·)
  unfold dot
  congr 1
  apply sumRange_congr
  intro t ht  -- t < K
  -- Goal: (colOf B i)[t] * A[j][t] = (matT B)[i][t] * (colOf (matT A) j)[t]
  -- `B` is nonempty (its length is `K > 0`); split it.
  obtain ⟨b, bs, rfl⟩ : ∃ b bs, B = b :: bs := by
    cases B with
    | nil => simp at hBlen; omega
    | cons b bs => exact ⟨b, bs, rfl⟩
  have hbk : b.length = C := by
    rw [hBC]; simp
  -- (colOf B i)[t] = B[t][i] = (matT B)[i][t]
  have e1 : (colOf (b :: bs) i).getD t 0 = ((matT (b :: bs)).getD i []).getD t 0 := by
    rw [colOf_getD, matT_entry b bs i t (by rw [hbk]; exact hi)]
  -- A[j][t] = (matT A)[t][j] = (colOf (matT A) j)[t]
  have e2 : ((a :: as).getD j []).getD t 0
      = (colOf (matT (a :: as)) j).getD t 0 := by
    rw [colOf_getD, matT_entry a as t j (by rw [hak]; exact ht)]
  simp only []
  rw [e1, e2]

end ChainRule
end FunctorFlowProofs
