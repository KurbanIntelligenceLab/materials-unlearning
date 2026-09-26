import DeletionFloor.Interval
import DeletionFloor.Neighbor

open MeasureTheory Set Filter
namespace DeletionFloor

 def Equivalent {α : Type*} [MeasurableSpace α] (P Q : Measure α) (ε δ : ℝ) : Prop :=
   ∀ S, MeasurableSet S →
     P.real S ≤ Real.exp ε * Q.real S + δ ∧ Q.real S ≤ Real.exp ε * P.real S + δ

 theorem tail_integrable {α : Type*} [MeasurableSpace α]
     (Q : Measure α) [IsProbabilityMeasure Q] (h : α → ℝ) (B : ℝ) :
     Integrable (fun t : ℝ ↦ Q.real {x | t ≤ h x}) (volume.restrict (Ioc 0 B)) := by
   have hm : Measurable (fun t : ℝ ↦ Q.real {x | t ≤ h x}) := by
     apply Measurable.ennreal_toReal
     exact Antitone.measurable (fun s t hst ↦ measure_mono (fun x hx ↦ hst.trans hx))
   apply (integrable_const (1 : ℝ)).mono' hm.aestronglyMeasurable
   exact Eventually.of_forall (fun t ↦ by
     rw [Real.norm_eq_abs, abs_of_nonneg (measureReal_nonneg)]
     exact measureReal_le_one)

 theorem one_way_transfer {α : Type*} [MeasurableSpace α]
     (P Q : Measure α) [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
     {h : α → ℝ} {B a δ : ℝ} (hB : 0 < B) (hm : Measurable h)
     (hb : ∀ x, 0 ≤ h x ∧ h x ≤ B)
     (he : ∀ S, MeasurableSet S → P.real S ≤ a * Q.real S + δ) :
     (∫ x, h x ∂P) ≤ a * (∫ x, h x ∂Q) + δ * B := by
   have hiP := integrable_bounded P hm (Eventually.of_forall (fun x ↦ by
     rw [abs_of_nonneg (hb x).1]; exact (hb x).2))
   have hiQ := integrable_bounded Q hm (Eventually.of_forall (fun x ↦ by
     rw [abs_of_nonneg (hb x).1]; exact (hb x).2))
   rw [hiP.integral_eq_integral_Ioc_meas_le
       (Eventually.of_forall (fun x ↦ (hb x).1)) (Eventually.of_forall (fun x ↦ (hb x).2)),
     hiQ.integral_eq_integral_Ioc_meas_le
       (Eventually.of_forall (fun x ↦ (hb x).1)) (Eventually.of_forall (fun x ↦ (hb x).2))]
   calc
     (∫ t in Ioc 0 B, P.real {x | t ≤ h x}) ≤
         ∫ t in Ioc 0 B, (a * Q.real {x | t ≤ h x} + δ) := by
       apply integral_mono (tail_integrable P h B) ((tail_integrable Q h B).const_mul a |>.add (integrable_const δ))
       intro t
       exact he _ (measurableSet_le measurable_const hm)
     _ = a * (∫ t in Ioc 0 B, Q.real {x | t ≤ h x}) + δ * B := by
       rw [integral_add ((tail_integrable Q h B).const_mul a) (integrable_const δ), integral_const_mul]
       simp [Real.volume_real_Ioc, max_eq_left hB.le, mul_comm]

 theorem bounded_transfer {α : Type*} [MeasurableSpace α]
     (P Q : Measure α) [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
     {h : α → ℝ} {B ε δ : ℝ} (hB : 0 < B) (hm : Measurable h)
     (hb : ∀ x, 0 ≤ h x ∧ h x ≤ B) (he : Equivalent P Q ε δ) :
     max 0 (max (Real.exp (-ε) * ((∫ x, h x ∂Q) - δ*B))
         (B - Real.exp ε * (B - (∫ x, h x ∂Q)) - δ*B)) ≤ (∫ x, h x ∂P) ∧
     (∫ x, h x ∂P) ≤ min B (min (Real.exp ε * (∫ x, h x ∂Q) + δ*B)
         (B - Real.exp (-ε) * (B - (∫ x, h x ∂Q) - δ*B))) := by
   have hiP := integrable_bounded P hm (Eventually.of_forall (fun x ↦ by
     rw [abs_of_nonneg (hb x).1]; exact (hb x).2))
   have hiQ := integrable_bounded Q hm (Eventually.of_forall (fun x ↦ by
     rw [abs_of_nonneg (hb x).1]; exact (hb x).2))
   have h₁ := one_way_transfer P Q hB hm hb (fun S hS ↦ (he S hS).1)
   have h₂ := one_way_transfer Q P hB hm hb (fun S hS ↦ (he S hS).2)
   have hc : ∀ x, 0 ≤ B - h x ∧ B - h x ≤ B := fun x ↦ ⟨by linarith [(hb x).2], by linarith [(hb x).1]⟩
   have h₃ := one_way_transfer P Q hB (measurable_const.sub hm) hc (fun S hS ↦ (he S hS).1)
   have h₄ := one_way_transfer Q P hB (measurable_const.sub hm) hc (fun S hS ↦ (he S hS).2)
   simp [integral_sub (integrable_const B) hiP, integral_sub (integrable_const B) hiQ] at h₃ h₄
   have h0 : 0 ≤ ∫ x, h x ∂P := integral_nonneg (fun x ↦ (hb x).1)
   have htop : (∫ x, h x ∂P) ≤ B := by
     simpa using integral_mono hiP (integrable_const B) (fun x ↦ (hb x).2)
   have hinv : Real.exp ε * Real.exp (-ε) = 1 := by rw [← Real.exp_add]; simp
   have hx := Real.exp_pos (-ε)
   have h₂' := mul_le_mul_of_nonneg_right h₂ hx.le
   have h₄' := mul_le_mul_of_nonneg_right h₄ hx.le
   constructor
   · exact max_le h0 (max_le (by nlinarith) (by linarith))
   · exact le_min htop (le_min h₁ (by nlinarith))
end DeletionFloor
