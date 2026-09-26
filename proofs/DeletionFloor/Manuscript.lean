import DeletionFloor.Definitions
import DeletionFloor.Sharpness
import DeletionFloor.RidgeFit
import DeletionFloor.Stability

open MeasureTheory Filter Matrix
noncomputable section
namespace DeletionFloor

/-- Equation (1), starting from the model laws and the target-loss objective. -/
theorem target_threshold_from_laws {Θ : Type*} [MeasurableSpace Θ]
    (P Q : Measure Θ) [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    {h : Θ → ℝ} {B ε τ : ℝ} (hm : Measurable h)
    (hb : ∀ θ, 0 ≤ h θ ∧ h θ ≤ B) (he : Equivalent P Q ε 0)
    (hφ : 0 < floor Q h) (hτ : floor Q h < τ) (hB : τ < B)
    (hobj : τ ≤ floor P h) :
    max (Real.log (τ / floor Q h))
      (Real.log ((B - floor Q h) / (B - τ))) ≤ ε := by
  have hBp : 0 < B := lt_trans (lt_trans hφ hτ) hB
  have ht := (bounded_transfer P Q hBp hm hb he).2
  simp only [zero_mul, add_zero, sub_zero] at ht
  exact target_threshold hφ hτ hB hobj
    (ht.trans ((min_le_right _ _).trans (min_le_left _ _)))
    (ht.trans ((min_le_right _ _).trans (min_le_right _ _)))

/-- Appendix A.2: the minimum over neighbors bounds both expected losses. -/
theorem finite_neighbor_expected_min {Θ ι : Type*} [MeasurableSpace Θ]
    (Q : Measure Θ) [IsProbabilityMeasure Q] {h : Θ → ℝ} {C : ι → ℝ}
    (s : Finset ι) (hs : s.Nonempty) (hm : Measurable h)
    (hC : ∀ i ∈ s, 0 ≤ C i) (hb : ∀ i ∈ s, ∀ᵐ θ ∂Q, |h θ| ≤ C i) :
    floor Q (fun θ ↦ |h θ|) ≤ s.inf' hs C ∧
    floor Q (fun θ ↦ (h θ)^2) ≤ (s.inf' hs C)^2 := by
  exact expected_error Q hm (finite_neighbor_min Q s hs hb)
    ((Finset.le_inf'_iff hs C).mpr hC)

/-- Appendix A.2: the same minimum also bounds clipped expected squared loss. -/
theorem finite_neighbor_clipped_min {Θ ι : Type*} [MeasurableSpace Θ]
    (Q : Measure Θ) [IsProbabilityMeasure Q] {h : Θ → ℝ} {C : ι → ℝ} {B : ℝ}
    (s : Finset ι) (hs : s.Nonempty) (hm : Measurable h)
    (hC : ∀ i ∈ s, 0 ≤ C i) (hB : 0 ≤ B)
    (hb : ∀ i ∈ s, ∀ᵐ θ ∂Q, |h θ| ≤ C i) :
    floor Q (fun θ ↦ min ((h θ)^2) B) ≤ min ((s.inf' hs C)^2) B := by
  exact clipped_expected_error Q hm (finite_neighbor_min Q s hs hb)
    ((Finset.le_inf'_iff hs C).mpr hC) hB

/-- Equation (4): the retained objective's minimizer gives the stated floor. -/
theorem indexed_ridge_floor {p ι : Type*} [Fintype p] [DecidableEq p] [DecidableEq ι]
    (s : Finset ι) (a : ι → p → ℝ) (y : ι → ℝ)
    (lam : ℝ) (i : ι) (hi : i ∈ s) (hs : 2 ≤ s.card) (hl : 0 < lam)
    (w : p → ℝ)
    (hw : ∀ z, meanObjective (s.erase i) a y lam w ≤ meanObjective (s.erase i) a y lam z) :
    let M := normalMatrix s a ((s.card - 1 : ℝ)*lam)
    floor (Measure.dirac w) (fun z ↦ (a i ⬝ᵥ z - y i)^2) =
      ((a i ⬝ᵥ (M⁻¹ *ᵥ normalRhs s a y) - y i) /
        (1 - a i ⬝ᵥ (M⁻¹ *ᵥ a i)))^2 := by
  dsimp only
  rw [floor, integral_dirac]
  rw [indexed_ridge_residual s a y lam i hi hs hl w hw]

/-- Appendix A.3: the fixed-penalty ratio uses the actual deterministic floor. -/
theorem fixed_penalty_floor_ratio {p ι : Type*} [Fintype p] [DecidableEq p] [DecidableEq ι]
    (s : Finset ι) (a : ι → p → ℝ) (y : ι → ℝ)
    (α : ℝ) (i : ι) (hi : i ∈ s) (hα : 0 < α) (w wd : p → ℝ)
    (hw : ∀ z, squareObjective (s.erase i) a y α w ≤ squareObjective (s.erase i) a y α z)
    (hwd : ∀ z, squareObjective s a y α wd ≤ squareObjective s a y α z)
    (hpos : 0 < floor (Measure.dirac w) (fun z ↦ (a i ⬝ᵥ z - y i)^2)) :
    (a i ⬝ᵥ wd - y i)^2 / floor (Measure.dirac w) (fun z ↦ (a i ⬝ᵥ z - y i)^2) =
      (1 - a i ⬝ᵥ ((normalMatrix s a α)⁻¹ *ᵥ a i))^2 := by
  have hH := normalMatrix_posDef (s.erase i) a hα
  rw [erase_normalMatrix s a α i hi] at hH
  have hh := leverage_lt_one _ (a i) (normalMatrix_posDef s a hα) hH
  simp only [floor, integral_dirac] at hpos ⊢
  exact fixed_penalty_ratio (fixed_penalty_residual s a y α i hi hα w wd hw hwd) hh hpos

/-- The lower endpoint printed in Theorem 1. -/
def lowerEndpoint (B ε δ φ : ℝ) : ℝ :=
  max 0 (max (Real.exp (-ε) * (φ - δ*B)) (B - Real.exp ε * (B-φ) - δ*B))

/-- The upper endpoint printed in Theorem 1. -/
def upperEndpoint (B ε δ φ : ℝ) : ℝ :=
  min B (min (Real.exp ε * φ + δ*B) (B - Real.exp (-ε) * (B-φ-δ*B)))

theorem normalized_lower_eq_endpoint {B ε δ φ : ℝ} (hB : 0 < B) :
    B * lower (Real.exp ε) δ (φ/B) = lowerEndpoint B ε δ φ := by
  rw [scaled_lower hB (Real.exp_pos ε)]
  simp only [lowerEndpoint, Real.exp_neg, div_eq_mul_inv, mul_comm]

theorem normalized_upper_eq_endpoint {B ε δ φ : ℝ} (hB : 0 < B) :
    B * upper (Real.exp ε) δ (φ/B) = upperEndpoint B ε δ φ := by
  rw [scaled_upper hB (Real.exp_pos ε)]
  simp only [upperEndpoint, Real.exp_neg, div_eq_mul_inv, mul_comm]

/-- Theorem 1's transfer interval, using the manuscript's floor definition. -/
theorem manuscript_transfer {Θ : Type*} [MeasurableSpace Θ]
    (P Q : Measure Θ) [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    {h : Θ → ℝ} {B ε δ : ℝ} (hB : 0 < B) (hm : Measurable h)
    (hb : ∀ θ, 0 ≤ h θ ∧ h θ ≤ B) (he : Equivalent P Q ε δ) :
    lowerEndpoint B ε δ (floor Q h) ≤ floor P h ∧
    floor P h ≤ upperEndpoint B ε δ (floor Q h) := by
  exact bounded_transfer P Q hB hm hb he

/-- Theorem 1's sharpness, for the same endpoints as its transfer statement. -/
theorem manuscript_sharpness {ε δ B φ : ℝ} (hε : 0 ≤ ε) (hδ : 0 ≤ δ)
    (hB : 0 < B) (hφ : 0 ≤ φ) (hφB : φ ≤ B) :
    ∃ (Plo Phi Q : Measure Bool) (h : Bool → ℝ),
      IsProbabilityMeasure Plo ∧ IsProbabilityMeasure Phi ∧ IsProbabilityMeasure Q ∧
      Measurable h ∧ (∀ θ, 0 ≤ h θ ∧ h θ ≤ B) ∧
      Equivalent Plo Q ε δ ∧ Equivalent Phi Q ε δ ∧ floor Q h = φ ∧
      floor Plo h = lowerEndpoint B ε δ φ ∧ floor Phi h = upperEndpoint B ε δ φ := by
  simpa only [floor, normalized_lower_eq_endpoint hB, normalized_upper_eq_endpoint hB]
    using sharp_endpoints hε hδ hB hφ hφB

/-- Theorem 1's convergence for its printed endpoints, including boundary floors. -/
theorem manuscript_endpoint_limit {B φ : ℝ} (hB : 0 < B) (hφ : 0 ≤ φ) (hφB : φ ≤ B) :
    Tendsto (fun z : ℝ × ℝ ↦ lowerEndpoint B z.1 z.2 φ) (nhds (0,0)) (nhds φ) ∧
    Tendsto (fun z : ℝ × ℝ ↦ upperEndpoint B z.1 z.2 φ) (nhds (0,0)) (nhds φ) := by
  simpa only [normalized_lower_eq_endpoint hB, normalized_upper_eq_endpoint hB]
    using scaled_endpoints_tendsto hB hφ hφB

end DeletionFloor
