import Mathlib
open Set Filter
open scoped RealInnerProductSpace BigOperators
namespace DeletionFloor
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

 theorem convex_support {f : E → ℝ} {x y : E} {f' : E →L[ℝ] ℝ}
     (hc : ConvexOn ℝ univ f) (hf : HasFDerivAt f f' x) :
     f' (y - x) ≤ f y - f x := by
   have hline : ConvexOn ℝ univ (f ∘ AffineMap.lineMap (k := ℝ) x y) := by
     simpa using hc.comp_affineMap (AffineMap.lineMap (k := ℝ) x y)
   have hd : HasDerivAt (f ∘ AffineMap.lineMap (k := ℝ) x y) (f' (y-x)) 0 := by
     apply HasFDerivAt.comp_hasDerivAt
     · simpa using hf
     · exact AffineMap.hasDerivAt_lineMap
   have h := hline.le_slope_of_hasDerivAt (mem_univ 0) (mem_univ 1) zero_lt_one hd
   simpa [slope] using h

 theorem strong_support {f : E → ℝ} {g : E} {x y : E} {μ : ℝ}
     (hc : StrongConvexOn univ μ f) (hf : HasGradientAt f g x) :
     f x + ⟪g, y-x⟫ + μ/2 * ‖y-x‖^2 ≤ f y := by
   have hc' := (strongConvexOn_iff_convex).mp hc
   have hg := (hasGradientAt_iff_hasFDerivAt).mp hf
   have hnorm := (hasStrictFDerivAt_norm_sq x).hasFDerivAt
   have hd := hg.sub (hnorm.const_mul (μ/2))
   have h := convex_support (y := y) hc' hd
   simp only [ContinuousLinearMap.sub_apply, ContinuousLinearMap.smul_apply,
     ContinuousLinearMap.comp_apply, smul_eq_mul, InnerProductSpace.toDual_apply_apply,
     innerSL_apply_apply] at h
   rw [norm_sub_sq_real] 
   simp only [inner_sub_right, real_inner_self_eq_norm_sq] at h
   simp only [inner_sub_right, two_smul] at h ⊢
   rw [real_inner_comm x y]
   nlinarith

 theorem gradient_stationary {f : E → ℝ} {g x : E}
     (hmin : ∀ y, f x ≤ f y) (hg : HasGradientAt f g x) : g = 0 := by
   have hm : IsLocalMin f x := Filter.Eventually.of_forall hmin
   have hz := hm.hasFDerivAt_eq_zero ((hasGradientAt_iff_hasFDerivAt).mp hg)
   have h := congrArg (fun l : E →L[ℝ] ℝ ↦ l g) hz
   simpa using h

 theorem strong_gradient_distance {f : E → ℝ} {g x y : E} {μ : ℝ}
     (hμ : 0 < μ) (hc : StrongConvexOn univ μ f)
     (hx : HasGradientAt f 0 x) (hy : HasGradientAt f g y) :
     μ * ‖y-x‖ ≤ ‖g‖ := by
   have h1 := strong_support (y := y) hc hx
   have h2 := strong_support (y := x) hc hy
   simp only [inner_zero_left, zero_add] at h1
   rw [norm_sub_rev x y, ← neg_sub y x, inner_neg_right] at h2
   have hb := real_inner_le_norm g (y-x)
   by_cases hz : ‖y-x‖ = 0
   · simp [hz]
   · have hp : 0 < ‖y-x‖ := lt_of_le_of_ne (norm_nonneg _) (Ne.symm hz)
     nlinarith

 theorem average_gradient_bound {ι : Type*} (s : Finset ι) (hs : 0 < s.card)
     (v : ι → E) {G : ℝ} (hv : ∀ i ∈ s, ‖v i‖ ≤ G) :
     ‖((s.card : ℝ)⁻¹) • ∑ i ∈ s, v i‖ ≤ G := by
   rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (by positivity)]
   have hn := norm_sum_le s v
   have hb : (∑ i ∈ s, ‖v i‖) ≤ (s.card : ℝ) * G := by
     simpa using Finset.sum_le_sum hv
   have hc : (0 : ℝ) < s.card := by exact_mod_cast hs
   have hm := mul_le_mul_of_nonneg_left (hn.trans hb) (inv_nonneg.mpr hc.le)
   simpa [← mul_assoc, inv_mul_cancel₀ hc.ne'] using hm

 theorem gradient_cancellation {n : ℝ} (hn : n ≠ 0) (hn1 : n-1 ≠ 0)
     (v s r : E) (hret : (n-1)⁻¹ • s + r = 0) :
     n⁻¹ • (s + v) + r = n⁻¹ • (v - (n-1)⁻¹ • s) := by
   have hr : r = -((n-1)⁻¹ • s) := eq_neg_of_add_eq_zero_right hret
   rw [hr]
   have hscalar : n⁻¹ - (n-1)⁻¹ = -(n⁻¹ * (n-1)⁻¹) := by
     field_simp
     ring
   calc
     n⁻¹ • (s + v) + -((n-1)⁻¹ • s) = n⁻¹ • v + (n⁻¹ - (n-1)⁻¹) • s := by module
     _ = n⁻¹ • v + (-(n⁻¹ * (n-1)⁻¹)) • s := by rw [hscalar]
     _ = _ := by module

 theorem leave_one_out_stability {ι : Type*} (s : Finset ι) (hs : 0 < s.card)
     {f : E → ℝ} {x y g v r : E} (gr : ι → E) {μ G L : ℝ}
     (hμ : 0 < μ) (hL : 0 ≤ L) (hc : StrongConvexOn univ μ f)
     (hx : HasGradientAt f 0 x) (hy : HasGradientAt f g y)
     (hv : ‖v‖ ≤ G) (hgr : ∀ i ∈ s, ‖gr i‖ ≤ G)
     (hret : (s.card : ℝ)⁻¹ • ∑ i ∈ s, gr i + r = 0)
     (hfull : g = ((s.card : ℝ)+1)⁻¹ • ((∑ i ∈ s, gr i) + v) + r)
     {predict : E → ℝ} (hp : |predict x - predict y| ≤ L * ‖y-x‖) :
     |predict x - predict y| ≤ 2*G*L/(μ*((s.card : ℝ)+1)) := by
   have hn : 0 < (s.card : ℝ)+1 := by positivity
   have hs' : 0 < (s.card : ℝ) := by exact_mod_cast hs
   have he := gradient_cancellation (n := (s.card : ℝ)+1) hn.ne' (by simpa using hs'.ne') v (∑ i ∈ s, gr i) r (by simpa using hret)
   have hg : g = ((s.card : ℝ)+1)⁻¹ • (v - (s.card : ℝ)⁻¹ • ∑ i ∈ s, gr i) := by simpa [hfull] using he
   have hmean := average_gradient_bound s hs gr hgr
   have hnorm : ‖g‖ ≤ 2*G/((s.card : ℝ)+1) := by
     rw [hg, norm_smul, Real.norm_eq_abs, abs_of_nonneg (by positivity)]
     have hsub := norm_sub_le v ((s.card : ℝ)⁻¹ • ∑ i ∈ s, gr i)
     calc
       _ ≤ ((s.card : ℝ)+1)⁻¹ * (2*G) := mul_le_mul_of_nonneg_left (by linarith) (by positivity)
       _ = _ := by ring
   have hd := strong_gradient_distance hμ hc hx hy
   apply (le_div_iff₀ (mul_pos hμ hn)).mpr
   have hh := (le_div_iff₀ hn).mp (hd.trans hnorm)
   have hmul := mul_le_mul_of_nonneg_left hh hL
   have hpred := mul_le_mul_of_nonneg_right hp (mul_pos hμ hn).le
   nlinarith

 theorem gradient_add {f k : E → ℝ} {g h x : E}
     (hf : HasGradientAt f g x) (hk : HasGradientAt k h x) :
     HasGradientAt (fun z ↦ f z + k z) (g+h) x := by
   apply hasGradientAt_iff_hasFDerivAt.mpr
   convert! hf.hasFDerivAt.add hk.hasFDerivAt using 1 <;> simp

 theorem gradient_const_mul {f : E → ℝ} {g x : E} (c : ℝ)
     (hf : HasGradientAt f g x) : HasGradientAt (fun z ↦ c * f z) (c • g) x := by
   apply hasGradientAt_iff_hasFDerivAt.mpr
   simpa using hf.hasFDerivAt.const_mul c

 theorem gradient_sum {ι : Type*} (s : Finset ι) (f : ι → E → ℝ)
     {g : ι → E} {x : E} (hg : ∀ i ∈ s, HasGradientAt (f i) (g i) x) :
     HasGradientAt (fun z ↦ ∑ i ∈ s, f i z) (∑ i ∈ s, g i) x := by
   apply hasGradientAt_iff_hasFDerivAt.mpr
   simpa using HasFDerivAt.fun_sum (fun i hi ↦ (hg i hi).hasFDerivAt)

 theorem stability_from_objectives {ι : Type*} (s : Finset ι) (hs : 0 < s.card)
     (loss : ι → E → ℝ) (deleted reg : E → ℝ)
     (grad : ι → E → E) (gradDeleted gradReg : E → E)
     (hgrad : ∀ i ∈ s, ∀ w, HasGradientAt (loss i) (grad i w) w)
     (hdel : ∀ w, HasGradientAt deleted (gradDeleted w) w)
     (hreg : ∀ w, HasGradientAt reg (gradReg w) w)
     {μ G L : ℝ} (hμ : 0 < μ) (hL : 0 ≤ L)
     (hbound : ∀ i ∈ s, ∀ w, ‖grad i w‖ ≤ G)
     (hboundDel : ∀ w, ‖gradDeleted w‖ ≤ G)
     (x y : E)
     (hcx : StrongConvexOn univ μ
       (fun w ↦ ((s.card : ℝ)+1)⁻¹ * ((∑ i ∈ s, loss i w) + deleted w) + reg w))
     (hminx : ∀ w, ((s.card : ℝ)+1)⁻¹ * ((∑ i ∈ s, loss i x) + deleted x) + reg x ≤
       ((s.card : ℝ)+1)⁻¹ * ((∑ i ∈ s, loss i w) + deleted w) + reg w)
     (hminy : ∀ w, (s.card : ℝ)⁻¹ * (∑ i ∈ s, loss i y) + reg y ≤
       (s.card : ℝ)⁻¹ * (∑ i ∈ s, loss i w) + reg w)
     {predict : E → ℝ} (hp : |predict x - predict y| ≤ L * ‖y-x‖) :
     |predict x - predict y| ≤ 2*G*L/(μ*((s.card : ℝ)+1)) := by
   have hf := fun w ↦ gradient_add
     (gradient_const_mul (((s.card : ℝ)+1)⁻¹)
       (gradient_add (gradient_sum s loss (fun i hi ↦ hgrad i hi w)) (hdel w))) (hreg w)
   have hr := fun w ↦ gradient_add
     (gradient_const_mul ((s.card : ℝ)⁻¹) (gradient_sum s loss (fun i hi ↦ hgrad i hi w))) (hreg w)
   have hx0 := gradient_stationary hminx (hf x)
   have hy0 := gradient_stationary hminy (hr y)
   have hx : HasGradientAt
       (fun w ↦ ((s.card : ℝ)+1)⁻¹ * ((∑ i ∈ s, loss i w) + deleted w) + reg w) 0 x := by
     simpa only [hx0] using hf x
   exact leave_one_out_stability s hs (fun i ↦ grad i y) hμ hL hcx hx (hf y)
     (hboundDel y) (fun i hi ↦ hbound i hi y) hy0 rfl hp

end DeletionFloor
