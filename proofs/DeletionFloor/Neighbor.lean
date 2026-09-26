import Mathlib

open MeasureTheory Filter
namespace DeletionFloor

 theorem neighbor_error {u v yf yr e L d ρ η : ℝ}
     (he : |v - yr| ≤ e) (hv : |u - v| ≤ L * d)
     (hy : |yf - yr| ≤ η) (hd : d ≤ ρ) (hL : 0 ≤ L) :
     |u - yf| ≤ e + L * ρ + η := by
   calc
     |u - yf| = |(u - v) + (v - yr) + (yr - yf)| := by congr 1 <;> ring
     _ ≤ |u - v| + |v - yr| + |yr - yf| := by
       have h1 := abs_add_le (u - v) (v - yr)
       have h2 := abs_add_le ((u - v) + (v - yr)) (yr - yf)
       linarith
     _ ≤ e + L * ρ + η := by rw [abs_sub_comm yr yf]; nlinarith [mul_le_mul_of_nonneg_left hd hL]

 theorem integrable_bounded {α : Type*} [MeasurableSpace α]
     (Q : Measure α) [IsProbabilityMeasure Q] {h : α → ℝ} {C : ℝ}
     (hm : Measurable h) (hb : ∀ᵐ x ∂Q, |h x| ≤ C) : Integrable h Q := by
   apply (integrable_const C).mono' hm.aestronglyMeasurable
   simpa only [Real.norm_eq_abs] using hb

 theorem expected_error {α : Type*} [MeasurableSpace α]
     (Q : Measure α) [IsProbabilityMeasure Q] {h : α → ℝ} {C : ℝ}
     (hm : Measurable h) (hb : ∀ᵐ x ∂Q, |h x| ≤ C) (hC : 0 ≤ C) :
     (∫ x, |h x| ∂Q) ≤ C ∧ (∫ x, (h x)^2 ∂Q) ≤ C^2 := by
   have hi : Integrable (fun x ↦ |h x|) Q :=
     integrable_bounded Q hm.abs (by simpa only [abs_abs] using hb)
   have hs : ∀ᵐ x ∂Q, |(h x)^2| ≤ C^2 := by
     filter_upwards [hb] with x hx
     rw [abs_of_nonneg (sq_nonneg _)]
     nlinarith [sq_abs (h x), abs_nonneg (h x)]
   have hsi := integrable_bounded Q (hm.pow_const 2) hs
   constructor
   · calc
       (∫ x, |h x| ∂Q) ≤ ∫ _, C ∂Q := integral_mono_ae hi (integrable_const C) hb
       _ = C := by simp
   · calc
       (∫ x, (h x)^2 ∂Q) ≤ ∫ _, C^2 ∂Q := integral_mono_ae hsi (integrable_const _) (by
         filter_upwards [hs] with x hx
         exact (le_abs_self _).trans hx)
       _ = C^2 := by simp

 theorem clipped_expected_error {α : Type*} [MeasurableSpace α]
     (Q : Measure α) [IsProbabilityMeasure Q] {h : α → ℝ} {C B : ℝ}
     (hm : Measurable h) (hb : ∀ᵐ x ∂Q, |h x| ≤ C) (hC : 0 ≤ C) (hB : 0 ≤ B) :
     (∫ x, min ((h x)^2) B ∂Q) ≤ min (C^2) B := by
   have hbound : ∀ᵐ x ∂Q, |min ((h x)^2) B| ≤ min (C^2) B := by
     filter_upwards [hb] with x hx
     rw [abs_of_nonneg (le_min (sq_nonneg _) hB)]
     apply min_le_min _ le_rfl
     nlinarith [sq_abs (h x), abs_nonneg (h x)]
   have hi := integrable_bounded Q ((hm.pow_const 2).min measurable_const) hbound
   calc
     (∫ x, min ((h x)^2) B ∂Q) ≤ ∫ _, min (C^2) B ∂Q :=
       integral_mono_ae hi (integrable_const _) (by
         filter_upwards [hbound] with x hx
         exact (le_abs_self _).trans hx)
     _ = min (C^2) B := by simp

 theorem neighbor_floor {α : Type*} [MeasurableSpace α]
     (Q : Measure α) [IsProbabilityMeasure Q] {u v : α → ℝ}
     {yf yr e L d ρ η : ℝ} (hu : Measurable u)
     (he : ∀ᵐ θ ∂Q, |v θ - yr| ≤ e) (hv : ∀ᵐ θ ∂Q, |u θ - v θ| ≤ L*d)
     (hy : |yf - yr| ≤ η) (hd : d ≤ ρ) (hL : 0 ≤ L)
     (he0 : 0 ≤ e) (hρ : 0 ≤ ρ) (hη : 0 ≤ η) :
     (∫ θ, |u θ - yf| ∂Q) ≤ e + L*ρ + η ∧
     (∫ θ, (u θ - yf)^2 ∂Q) ≤ (e + L*ρ + η)^2 := by
   apply expected_error Q (hu.sub measurable_const) _ (by positivity)
   filter_upwards [he, hv] with θ heθ hvθ
   exact neighbor_error heθ hvθ hy hd hL

 theorem finite_neighbor_min {α ι : Type*} [MeasurableSpace α]
     (Q : Measure α) {h : α → ℝ} {C : ι → ℝ} (s : Finset ι) (hs : s.Nonempty)
     (hb : ∀ i ∈ s, ∀ᵐ θ ∂Q, |h θ| ≤ C i) :
     ∀ᵐ θ ∂Q, |h θ| ≤ s.inf' hs C := by
   have hall : ∀ᵐ θ ∂Q, ∀ i ∈ s, |h θ| ≤ C i := by
     exact (s.eventually_all).mpr hb
   filter_upwards [hall] with θ hθ
   exact (Finset.le_inf'_iff hs C).mpr hθ
end DeletionFloor
