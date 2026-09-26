import Mathlib

/-! Scalar part of the sharp transfer interval. The event-to-expectation step
is proved separately in Transfer. All losses here are normalized by B. -/
noncomputable section
namespace DeletionFloor

 def lower (a δ q : ℝ) : ℝ := max 0 (max ((q - δ) / a) (1 - a * (1 - q) - δ))
 def upper (a δ q : ℝ) : ℝ := min 1 (min (a * q + δ) (1 - (1 - q - δ) / a))
 def Feasible (a δ p q : ℝ) : Prop :=
   0 ≤ p ∧ p ≤ 1 ∧ p ≤ a * q + δ ∧ q ≤ a * p + δ ∧
   1 - p ≤ a * (1 - q) + δ ∧ 1 - q ≤ a * (1 - p) + δ

 theorem feasible_iff {a δ p q : ℝ} (ha : 0 < a) :
     Feasible a δ p q ↔ lower a δ q ≤ p ∧ p ≤ upper a δ q := by
   have h₁ := div_mul_cancel₀ (q - δ) (ne_of_gt ha)
   have h₂ := div_mul_cancel₀ (1 - q - δ) (ne_of_gt ha)
   simp only [Feasible, lower, upper, max_le_iff, le_min_iff]
   constructor
   · rintro ⟨h0, h1, h2, h3, h4, h5⟩
     exact ⟨⟨h0, ⟨by nlinarith, by linarith⟩⟩,
       ⟨h1, ⟨h2, by nlinarith⟩⟩⟩
   · rintro ⟨⟨h0, h3, h4⟩, h1, h2, h5⟩
     exact ⟨h0, h1, h2, by nlinarith, by linarith, by nlinarith⟩

 theorem reference_feasible {a δ q : ℝ} (ha : 1 ≤ a) (hd : 0 ≤ δ)
     (hq : 0 ≤ q) (hq1 : q ≤ 1) : Feasible a δ q q := by
   unfold Feasible
   have h₁ : 0 ≤ (a - 1) * q := mul_nonneg (by linarith) hq
   have h₂ : 0 ≤ (a - 1) * (1 - q) := mul_nonneg (by linarith) (by linarith)
   exact ⟨hq, hq1, by nlinarith, by nlinarith, by nlinarith, by nlinarith⟩

 theorem endpoints_feasible {a δ q : ℝ} (ha : 1 ≤ a) (hd : 0 ≤ δ)
     (hq : 0 ≤ q) (hq1 : q ≤ 1) :
     Feasible a δ (lower a δ q) q ∧ Feasible a δ (upper a δ q) q := by
   have hap : 0 < a := by linarith
   obtain ⟨hl, hu⟩ := (feasible_iff hap).mp (reference_feasible ha hd hq hq1)
   exact ⟨(feasible_iff hap).mpr ⟨le_rfl, hl.trans hu⟩,
     (feasible_iff hap).mpr ⟨hl.trans hu, le_rfl⟩⟩

 theorem lower_at_identity {q : ℝ} (hq : 0 ≤ q) : lower 1 0 q = q := by
   simp [lower, max_eq_right hq]
 theorem upper_at_identity {q : ℝ} (hq : q ≤ 1) : upper 1 0 q = q := by
   simp [upper, min_eq_right hq]

 theorem lower_continuousAt {q : ℝ} :
     ContinuousAt (fun z : ℝ × ℝ ↦ lower (Real.exp z.1) z.2 q) (0, 0) := by
   unfold lower
   fun_prop (disch := positivity)
 theorem upper_continuousAt {q : ℝ} :
     ContinuousAt (fun z : ℝ × ℝ ↦ upper (Real.exp z.1) z.2 q) (0, 0) := by
   unfold upper
   fun_prop (disch := positivity)

 theorem endpoints_tendsto {q : ℝ} (hq : 0 ≤ q) (hq1 : q ≤ 1) :
     Filter.Tendsto (fun z : ℝ × ℝ ↦ lower (Real.exp z.1) z.2 q)
       (nhds (0, 0)) (nhds q) ∧
     Filter.Tendsto (fun z : ℝ × ℝ ↦ upper (Real.exp z.1) z.2 q)
       (nhds (0, 0)) (nhds q) := by
   constructor
   · simpa [Real.exp_zero, lower_at_identity hq] using (lower_continuousAt (q := q)).tendsto
   · simpa [Real.exp_zero, upper_at_identity hq1] using (upper_continuousAt (q := q)).tendsto

 theorem target_threshold {ε B φ τ m : ℝ} (hφ : 0 < φ) (hτ : φ < τ)
     (hB : τ < B) (hm : τ ≤ m)
     (hupper : m ≤ Real.exp ε * φ)
     (hcomp : m ≤ B - Real.exp (-ε) * (B - φ)) :
     max (Real.log (τ / φ)) (Real.log ((B - φ) / (B - τ))) ≤ ε := by
   apply max_le
   · apply (Real.log_le_iff_le_exp (div_pos (lt_trans hφ hτ) hφ)).mpr
     apply (div_le_iff₀ hφ).mpr
     linarith
   · apply (Real.log_le_iff_le_exp (div_pos (by linarith) (by linarith))).mpr
     apply (div_le_iff₀ (show 0 < B - τ by linarith)).mpr
     have hexp : 0 < Real.exp ε := Real.exp_pos ε
     have hinv : Real.exp (-ε) * Real.exp ε = 1 := by rw [← Real.exp_add]; simp
     have h := mul_le_mul_of_nonneg_right (show Real.exp (-ε) * (B - φ) ≤ B - τ by linarith) hexp.le
     nlinarith

 theorem scaled_lower {B a δ φ : ℝ} (hB : 0 < B) (ha : 0 < a) :
     B * lower a δ (φ/B) = max 0 (max ((φ-δ*B)/a) (B-a*(B-φ)-δ*B)) := by
   unfold lower
   rw [mul_max_of_nonneg _ _ hB.le, mul_zero, mul_max_of_nonneg _ _ hB.le]
   congr 2
   · field_simp
   · field_simp

 theorem scaled_upper {B a δ φ : ℝ} (hB : 0 < B) (ha : 0 < a) :
     B * upper a δ (φ/B) = min B (min (a*φ+δ*B) (B-(B-φ-δ*B)/a)) := by
   unfold upper
   rw [mul_min_of_nonneg _ _ hB.le, mul_one, mul_min_of_nonneg _ _ hB.le]
   congr 2
   · field_simp
   · field_simp

 theorem scaled_endpoints_tendsto {B φ : ℝ} (hB : 0 < B) (hφ : 0 ≤ φ) (hφB : φ ≤ B) :
     Filter.Tendsto (fun z : ℝ × ℝ ↦ B * lower (Real.exp z.1) z.2 (φ/B))
       (nhds (0,0)) (nhds φ) ∧
     Filter.Tendsto (fun z : ℝ × ℝ ↦ B * upper (Real.exp z.1) z.2 (φ/B))
       (nhds (0,0)) (nhds φ) := by
   obtain ⟨hl, hu⟩ := endpoints_tendsto (div_nonneg hφ hB.le) ((div_le_one hB).mpr hφB)
   constructor
   · simpa [mul_div_cancel₀ φ hB.ne'] using hl.const_mul B
   · simpa [mul_div_cancel₀ φ hB.ne'] using hu.const_mul B

end DeletionFloor
