import DeletionFloor.Ridge
open Matrix
open scoped BigOperators
noncomputable section
namespace DeletionFloor
variable {p ι : Type*} [Fintype p] [DecidableEq p] [DecidableEq ι]

 def quadratic (M : Matrix p p ℝ) (b w : p → ℝ) : ℝ :=
   w ⬝ᵥ (M *ᵥ w) - 2 * (b ⬝ᵥ w)

 theorem quadratic_difference (M : Matrix p p ℝ) (b w z : p → ℝ)
     (hM : M.PosDef) (hz : M *ᵥ z = b) :
     quadratic M b w - quadratic M b z = (w-z) ⬝ᵥ (M *ᵥ (w-z)) := by
   have hsym : Mᵀ = M := by exact (Matrix.isHermitian_iff_isSymm.mp hM.isHermitian).eq
   have hcross : z ⬝ᵥ (M *ᵥ w) = w ⬝ᵥ b := by
     rw [← hsym, dotProduct_transpose_mulVec, hz, dotProduct_comm]
   simp only [quadratic, mulVec_sub, sub_dotProduct, dotProduct_sub, hz, hcross]
   rw [dotProduct_comm b w, dotProduct_comm b z]
   ring

 theorem minimizer_normal_equation (M : Matrix p p ℝ) (b w : p → ℝ)
     (hM : M.PosDef) (hmin : ∀ z, quadratic M b w ≤ quadratic M b z) :
     M *ᵥ w = b := by
   let z := M⁻¹ *ᵥ b
   have hz : M *ᵥ z = b := solve_eq M hM b
   have hd := quadratic_difference M b w z hM hz
   have hm := hmin z
   by_cases hw : w-z = 0
   · have : w = z := sub_eq_zero.mp hw
     simpa [this] using hz
   · have hp := hM.dotProduct_mulVec_pos hw
     simp only [star_trivial] at hp
     linarith

 def normalMatrix (s : Finset ι) (a : ι → p → ℝ) (α : ℝ) : Matrix p p ℝ :=
   (∑ i ∈ s, vecMulVec (a i) (a i)) + α • 1
 def normalRhs (s : Finset ι) (a : ι → p → ℝ) (y : ι → ℝ) : p → ℝ :=
   ∑ i ∈ s, (y i) • a i
 def squareObjective (s : Finset ι) (a : ι → p → ℝ) (y : ι → ℝ)
     (α : ℝ) (w : p → ℝ) : ℝ :=
   (∑ i ∈ s, (a i ⬝ᵥ w - y i)^2) + α * (w ⬝ᵥ w)
 def meanObjective (s : Finset ι) (a : ι → p → ℝ) (y : ι → ℝ)
     (lam : ℝ) (w : p → ℝ) : ℝ :=
   (s.card : ℝ)⁻¹ * (∑ i ∈ s, (a i ⬝ᵥ w - y i)^2) + lam * (w ⬝ᵥ w)

 theorem normalMatrix_posDef (s : Finset ι) (a : ι → p → ℝ)
     {α : ℝ} (hα : 0 < α) : (normalMatrix s a α).PosDef := by
   apply Matrix.PosDef.posSemidef_add
   · apply Matrix.posSemidef_sum
     intro i hi
     simpa using Matrix.posSemidef_vecMulVec_self_star (a i)
   · exact (Matrix.PosDef.one : (1 : Matrix p p ℝ).PosDef).smul hα

 theorem squareObjective_eq (s : Finset ι) (a : ι → p → ℝ) (y : ι → ℝ)
     (α : ℝ) (w : p → ℝ) :
     squareObjective s a y α w = quadratic (normalMatrix s a α) (normalRhs s a y) w + ∑ i ∈ s, (y i)^2 := by
   simp only [squareObjective, quadratic, normalMatrix, normalRhs,
     add_mulVec, sum_mulVec, smul_mulVec, one_mulVec, dotProduct_add,
     dotProduct_sum, sum_dotProduct, vecMulVec_mulVec, op_smul_eq_smul,
     dotProduct_smul, smul_dotProduct, smul_eq_mul]
   rw [Finset.mul_sum]
   have h : (∑ i ∈ s, (a i ⬝ᵥ w - y i)^2) =
       ∑ i ∈ s, ((a i ⬝ᵥ w) * (w ⬝ᵥ a i) - 2 * (y i * (a i ⬝ᵥ w)) + (y i)^2) := by
     apply Finset.sum_congr rfl
     intro i hi
     rw [dotProduct_comm w (a i)]
     ring
   simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib] at h
   linarith

 theorem meanObjective_scale (s : Finset ι) (a : ι → p → ℝ) (y : ι → ℝ)
     (lam : ℝ) (w : p → ℝ) (hs : 0 < s.card) :
     (s.card : ℝ) * meanObjective s a y lam w = squareObjective s a y ((s.card : ℝ)*lam) w := by
   have hc : (s.card : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hs
   simp [meanObjective, squareObjective, mul_add, ← mul_assoc, mul_inv_cancel₀ hc]

 theorem mean_minimizer_normal_equation (s : Finset ι) (a : ι → p → ℝ) (y : ι → ℝ)
     (lam : ℝ) (w : p → ℝ) (hs : 0 < s.card) (hl : 0 < lam)
     (hw : ∀ z, meanObjective s a y lam w ≤ meanObjective s a y lam z) :
     normalMatrix s a ((s.card : ℝ)*lam) *ᵥ w = normalRhs s a y := by
   apply minimizer_normal_equation _ _ _ (normalMatrix_posDef s a (by positivity))
   intro z
   have h := mul_le_mul_of_nonneg_left (hw z) (show (0:ℝ) ≤ s.card by positivity)
   rw [meanObjective_scale s a y lam w hs, meanObjective_scale s a y lam z hs,
     squareObjective_eq, squareObjective_eq] at h
   linarith

 theorem erase_normalMatrix (s : Finset ι) (a : ι → p → ℝ) (α : ℝ)
     (i : ι) (hi : i ∈ s) :
     normalMatrix (s.erase i) a α = normalMatrix s a α - vecMulVec (a i) (a i) := by
   have h := Finset.sum_erase_add s (fun j ↦ vecMulVec (a j) (a j)) hi
   unfold normalMatrix
   rw [← h]
   abel

 theorem erase_normalRhs (s : Finset ι) (a : ι → p → ℝ) (y : ι → ℝ)
     (i : ι) (hi : i ∈ s) :
     normalRhs (s.erase i) a y = normalRhs s a y - (y i) • a i := by
   exact (eq_sub_iff_add_eq).mpr (Finset.sum_erase_add s (fun j ↦ (y j) • a j) hi)

 theorem indexed_ridge_residual (s : Finset ι) (a : ι → p → ℝ) (y : ι → ℝ)
     (lam : ℝ) (i : ι) (hi : i ∈ s) (hs : 2 ≤ s.card) (hl : 0 < lam)
     (w : p → ℝ) (hw : ∀ z, meanObjective (s.erase i) a y lam w ≤ meanObjective (s.erase i) a y lam z) :
     let M := normalMatrix s a ((s.card-1 : ℝ)*lam)
     a i ⬝ᵥ w - y i = (a i ⬝ᵥ (M⁻¹ *ᵥ normalRhs s a y) - y i) /
       (1 - a i ⬝ᵥ (M⁻¹ *ᵥ a i)) := by
   have he : 0 < (s.erase i).card := by rw [Finset.card_erase_of_mem hi]; omega
   have hc : ((s.erase i).card : ℝ) = (s.card : ℝ)-1 := by
     rw [Finset.card_erase_of_mem hi, Nat.cast_sub (by omega)]; simp
   have hp : 0 < ((s.card : ℝ)-1)*lam := by
     have hcard : (2:ℝ) ≤ s.card := by exact_mod_cast hs
     exact mul_pos (by linarith) hl
   have hM := normalMatrix_posDef s a hp
   have hH := normalMatrix_posDef (s.erase i) a hp
   rw [erase_normalMatrix s a _ i hi] at hH
   have hnormal := mean_minimizer_normal_equation (s.erase i) a y lam w he hl hw
   rw [hc, erase_normalMatrix s a _ i hi, erase_normalRhs s a y i hi] at hnormal
   exact ridge_residual _ _ _ _ _ hM hH hnormal

 theorem indexed_prediction_change (s : Finset ι) (a : ι → p → ℝ) (y : ι → ℝ)
     (lam : ℝ) (i : ι) (hi : i ∈ s) (hs : 2 ≤ s.card) (hl : 0 < lam)
     (w wd : p → ℝ)
     (hw : ∀ z, meanObjective (s.erase i) a y lam w ≤ meanObjective (s.erase i) a y lam z)
     (hwd : ∀ z, meanObjective s a y lam wd ≤ meanObjective s a y lam z) :
     let M := normalMatrix s a ((s.card-1 : ℝ)*lam)
     a i ⬝ᵥ (w-wd) = lam * (a i ⬝ᵥ (M⁻¹ *ᵥ wd)) +
       (a i ⬝ᵥ (M⁻¹ *ᵥ a i)) * (a i ⬝ᵥ (M⁻¹ *ᵥ normalRhs s a y) - y i) /
         (1 - a i ⬝ᵥ (M⁻¹ *ᵥ a i)) := by
   have he : 0 < (s.erase i).card := by rw [Finset.card_erase_of_mem hi]; omega
   have hc : ((s.erase i).card : ℝ) = (s.card : ℝ)-1 := by
     rw [Finset.card_erase_of_mem hi, Nat.cast_sub (by omega)]; simp
   have hp : 0 < ((s.card : ℝ)-1)*lam := by
     have hcard : (2:ℝ) ≤ s.card := by exact_mod_cast hs
     exact mul_pos (by linarith) hl
   have hM := normalMatrix_posDef s a hp
   have hH := normalMatrix_posDef (s.erase i) a hp
   rw [erase_normalMatrix s a _ i hi] at hH
   have hnormal := mean_minimizer_normal_equation (s.erase i) a y lam w he hl hw
   rw [hc, erase_normalMatrix s a _ i hi, erase_normalRhs s a y i hi] at hnormal
   have hfull := mean_minimizer_normal_equation s a y lam wd (by omega) hl hwd
   have hmat : normalMatrix s a (((s.card : ℝ)-1)*lam) + lam • 1 =
       normalMatrix s a ((s.card : ℝ)*lam) := by
     unfold normalMatrix
     rw [add_assoc, ← add_smul]
     congr 2
     ring
   rw [← hmat] at hfull
   exact ridge_prediction_change _ _ _ _ _ _ _ hM hH hnormal hfull

 theorem square_minimizer_normal_equation (s : Finset ι) (a : ι → p → ℝ) (y : ι → ℝ)
     (α : ℝ) (w : p → ℝ) (hα : 0 < α)
     (hw : ∀ z, squareObjective s a y α w ≤ squareObjective s a y α z) :
     normalMatrix s a α *ᵥ w = normalRhs s a y := by
   apply minimizer_normal_equation _ _ _ (normalMatrix_posDef s a hα)
   intro z
   have h := hw z
   rw [squareObjective_eq, squareObjective_eq] at h
   linarith

 theorem fixed_penalty_residual (s : Finset ι) (a : ι → p → ℝ) (y : ι → ℝ)
     (α : ℝ) (i : ι) (hi : i ∈ s) (hα : 0 < α) (w wd : p → ℝ)
     (hw : ∀ z, squareObjective (s.erase i) a y α w ≤ squareObjective (s.erase i) a y α z)
     (hwd : ∀ z, squareObjective s a y α wd ≤ squareObjective s a y α z) :
     a i ⬝ᵥ w - y i = (a i ⬝ᵥ wd - y i) /
       (1 - a i ⬝ᵥ ((normalMatrix s a α)⁻¹ *ᵥ a i)) := by
   have hM := normalMatrix_posDef s a hα
   have hH := normalMatrix_posDef (s.erase i) a hα
   rw [erase_normalMatrix s a α i hi] at hH
   have hn := square_minimizer_normal_equation (s.erase i) a y α w hα hw
   rw [erase_normalMatrix s a α i hi, erase_normalRhs s a y i hi] at hn
   have hfull := square_minimizer_normal_equation s a y α wd hα hwd
   have hwsolve : (normalMatrix s a α)⁻¹ *ᵥ normalRhs s a y = wd := by
     letI := hM.isUnit.invertible
     rw [← hfull, mulVec_mulVec, Matrix.inv_mul_of_invertible, one_mulVec]
   have h := ridge_residual _ _ _ _ _ hM hH hn
   rw [hwsolve] at h
   exact h

end DeletionFloor
