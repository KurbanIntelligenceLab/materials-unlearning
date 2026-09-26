import Mathlib
open MeasureTheory
noncomputable section
namespace DeletionFloor

/-- Expected target loss under the specified retraining law. -/
def floor {Θ : Type*} [MeasurableSpace Θ] (Q : Measure Θ) (loss : Θ → ℝ) : ℝ :=
  ∫ θ, loss θ ∂Q

/-- A deterministic retraining law evaluates the loss at its fitted model. -/
theorem deterministic_floor {Θ : Type*} [MeasurableSpace Θ]
    (loss : Θ → ℝ) (hm : Measurable loss) (θ : Θ) :
    floor (Measure.dirac θ) loss = loss θ := by
  exact integral_dirac' loss θ hm.stronglyMeasurable

end DeletionFloor
