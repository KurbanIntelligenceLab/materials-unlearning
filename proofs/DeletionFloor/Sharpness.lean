import DeletionFloor.Transfer
open MeasureTheory ProbabilityTheory Set
namespace DeletionFloor

 theorem bernoulli_equivalent {ε δ : ℝ} (hε : 0 ≤ ε) (hδ : 0 ≤ δ)
     (p q : unitInterval) (hf : Feasible (Real.exp ε) δ p q) :
     Equivalent (bernoulliMeasure true false p) (bernoulliMeasure true false q) ε δ := by
   classical
   intro S hS
   have ha : 1 ≤ Real.exp ε := Real.one_le_exp_iff.mpr hε
   rcases hf with ⟨_, _, h1, h2, h3, h4⟩
   by_cases ht : true ∈ S <;> by_cases hf : false ∈ S
   · simp only [bernoulliMeasure_real_apply_of_mem_of_mem _ hS ht hf]
     constructor <;> linarith
   · simpa only [bernoulliMeasure_real_apply_of_mem_of_notMem _ hS ht hf] using And.intro h1 h2
   · simpa only [bernoulliMeasure_real_apply_of_notMem_of_mem _ hS ht hf] using And.intro h3 h4
   · simp only [bernoulliMeasure_real_apply_of_notMem_of_notMem _ hS ht hf]
     constructor <;> linarith

 theorem sharp_endpoints {ε δ B φ : ℝ} (hε : 0 ≤ ε) (hδ : 0 ≤ δ)
     (hB : 0 < B) (hφ : 0 ≤ φ) (hφB : φ ≤ B) :
     ∃ (Plo Phi Q : Measure Bool) (h : Bool → ℝ),
       IsProbabilityMeasure Plo ∧ IsProbabilityMeasure Phi ∧ IsProbabilityMeasure Q ∧
       Measurable h ∧ (∀ x, 0 ≤ h x ∧ h x ≤ B) ∧
       Equivalent Plo Q ε δ ∧ Equivalent Phi Q ε δ ∧
       (∫ x, h x ∂Q) = φ ∧
       (∫ x, h x ∂Plo) = B * lower (Real.exp ε) δ (φ/B) ∧
       (∫ x, h x ∂Phi) = B * upper (Real.exp ε) δ (φ/B) := by
   have hq : 0 ≤ φ / B := div_nonneg hφ hB.le
   have hq1 : φ / B ≤ 1 := (div_le_one hB).mpr hφB
   obtain ⟨hl, hu⟩ := endpoints_feasible (Real.one_le_exp_iff.mpr hε) hδ hq hq1
   let q : unitInterval := ⟨φ/B, hq, hq1⟩
   let pl : unitInterval := ⟨lower (Real.exp ε) δ (φ/B), hl.1, hl.2.1⟩
   let pu : unitInterval := ⟨upper (Real.exp ε) δ (φ/B), hu.1, hu.2.1⟩
   let h : Bool → ℝ := fun x ↦ if x then B else 0
   refine ⟨bernoulliMeasure true false pl, bernoulliMeasure true false pu,
     bernoulliMeasure true false q, h, inferInstance, inferInstance, inferInstance,
     measurable_of_finite h, ?_, bernoulli_equivalent hε hδ pl q hl,
     bernoulli_equivalent hε hδ pu q hu, ?_, ?_, ?_⟩
   · intro x; cases x <;> simp [h, hB.le]
   · rw [integral_bernoulliMeasure]; dsimp [h, q]; field_simp; simp
   · rw [integral_bernoulliMeasure]; simp [h, pl, mul_comm]
   · rw [integral_bernoulliMeasure]; simp [h, pu, mul_comm]
end DeletionFloor
