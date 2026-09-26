import Mathlib
open Matrix
open scoped BigOperators
noncomputable section
namespace DeletionFloor

variable {n p : Type*} [Fintype n] [Fintype p] [DecidableEq p]

 def gram (A : Matrix n p ℝ) (α : ℝ) : Matrix p p ℝ := Aᵀ * A + α • 1

 theorem gram_posDef (A : Matrix n p ℝ) {α : ℝ} (hα : 0 < α) :
     (gram A α).PosDef := by
   have hA : (Aᵀ * A).PosSemidef := by
     simpa using Matrix.posSemidef_conjTranspose_mul_self A
   exact Matrix.PosDef.posSemidef_add hA ((Matrix.PosDef.one : (1 : Matrix p p ℝ).PosDef).smul hα)

 theorem solve_eq (M : Matrix p p ℝ) (hM : M.PosDef) (b : p → ℝ) :
     M *ᵥ (M⁻¹ *ᵥ b) = b := by
   letI := hM.isUnit.invertible
   rw [mulVec_mulVec, Matrix.mul_inv_of_invertible, one_mulVec]

 theorem leverage_lt_one (M : Matrix p p ℝ) (a : p → ℝ)
     (hM : M.PosDef) (hH : (M - vecMulVec a a).PosDef) :
     a ⬝ᵥ (M⁻¹ *ᵥ a) < 1 := by
   let v := M⁻¹ *ᵥ a
   have hv : M *ᵥ v = a := solve_eq M hM a
   by_cases hz : v = 0
   · have : a = 0 := by rw [← hv, hz]; simp
     simp [this]
   · have hpos := hH.dotProduct_mulVec_pos hz
     have hbase := hM.dotProduct_mulVec_pos hz
     simp only [star_trivial] at hpos hbase
     rw [hv, dotProduct_comm v a] at hbase
     rw [sub_mulVec, hv, vecMulVec_mulVec, op_smul_eq_smul, dotProduct_sub, dotProduct_smul,
       smul_eq_mul, dotProduct_comm v a] at hpos
     dsimp [v] at hpos hbase
     nlinarith

 theorem ridge_residual (M : Matrix p p ℝ) (a b w : p → ℝ) (y : ℝ)
     (hM : M.PosDef) (hH : (M - vecMulVec a a).PosDef)
     (hw : (M - vecMulVec a a) *ᵥ w = b - y • a) :
     a ⬝ᵥ w - y = (a ⬝ᵥ (M⁻¹ *ᵥ b) - y) / (1 - a ⬝ᵥ (M⁻¹ *ᵥ a)) := by
   letI := hM.isUnit.invertible
   have ht := congrArg (fun z ↦ M⁻¹ *ᵥ z) hw
   simp only [sub_mulVec, vecMulVec_mulVec, op_smul_eq_smul, mulVec_sub, mulVec_smul,
     mulVec_mulVec, Matrix.inv_mul_of_invertible, one_mulVec] at ht
   have hr := congrArg (fun z ↦ a ⬝ᵥ z) ht
   simp only [dotProduct_sub, dotProduct_smul, smul_eq_mul] at hr
   apply (eq_div_iff (ne_of_gt (sub_pos.mpr (leverage_lt_one M a hM hH)))).mpr
   nlinarith

 theorem ridge_prediction_change (M : Matrix p p ℝ) (a b w wd : p → ℝ)
     (y lam : ℝ) (hM : M.PosDef) (hH : (M - vecMulVec a a).PosDef)
     (hw : (M - vecMulVec a a) *ᵥ w = b - y • a)
     (hwd : (M + lam • 1) *ᵥ wd = b) :
     a ⬝ᵥ (w - wd) = lam * (a ⬝ᵥ (M⁻¹ *ᵥ wd)) +
       (a ⬝ᵥ (M⁻¹ *ᵥ a)) * (a ⬝ᵥ (M⁻¹ *ᵥ b) - y) /
         (1 - a ⬝ᵥ (M⁻¹ *ᵥ a)) := by
   letI := hM.isUnit.invertible
   have ht := congrArg (fun z ↦ M⁻¹ *ᵥ z) hwd
   simp only [add_mulVec, smul_mulVec, one_mulVec, mulVec_add, mulVec_smul,
     mulVec_mulVec, Matrix.inv_mul_of_invertible, one_mulVec] at ht
   have hadj := congrArg (fun z ↦ a ⬝ᵥ z) ht
   simp only [dotProduct_add, dotProduct_smul, smul_eq_mul] at hadj
   have hr := ridge_residual M a b w y hM hH hw
   rw [dotProduct_sub]
   have hd : 1 - a ⬝ᵥ (M⁻¹ *ᵥ a) ≠ 0 := ne_of_gt (sub_pos.mpr (leverage_lt_one M a hM hH))
   apply (mul_right_inj' hd).mp
   have hr' := (eq_div_iff hd).mp hr
   have hadj' := congrArg (fun t : ℝ ↦ t * (1 - a ⬝ᵥ (M⁻¹ *ᵥ a))) hadj
   field_simp
   nlinarith

 theorem fixed_penalty_ratio {r rr h : ℝ} (hr : rr = r / (1-h))
     (hh : h < 1) (hrr : rr^2 > 0) : r^2 / rr^2 = (1-h)^2 := by
   have hd : 1-h ≠ 0 := ne_of_gt (sub_pos.mpr hh)
   have he := (eq_div_iff hd).mp hr
   apply (div_eq_iff (ne_of_gt hrr)).mpr
   rw [← he]
   ring
end DeletionFloor
