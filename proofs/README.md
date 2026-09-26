# Lean proofs for the deletion-floor manuscript

This project formalizes the mathematical results in the submitted manuscript’s Method and Appendix A. The root module is `DeletionFloor.lean`; `DeletionFloor/Manuscript.lean` assembles manuscript-level claims from their supporting lemmas. Experimental implementations, datasets, measurements, and the empirical FloorScore diagnostic are outside the formalization.

## Reproduce the verification

Install [elan](https://github.com/leanprover/elan#installation), then run from this directory:

```sh
lake exe cache get
python3 verify.py
```

The first command obtains mathlib's compiled dependency cache. The second builds all project modules and runs `#print axioms` on every project theorem, checking that its transitive axiom dependencies are a subset of `propext`, `Classical.choice`, and `Quot.sound`. It fails if a theorem is missing from the audit, a proof placeholder or project axiom appears, or any other axiom dependency is reported. If `lake` is not on PATH, set `LAKE` to its absolute executable path when running the script. Python 3 uses only its standard library.

`lean-toolchain` pins Lean **4.33.1**. `lakefile.toml` selects mathlib **v4.33.1**, and `lake-manifest.json` locks mathlib to **0df444a360eaa60ab8c11dca51a86af692955474** and records all transitive package revisions. Retain the lockfile; do not run `lake update` to reproduce this version. Generated files and downloaded dependencies live in `.lake/` and are excluded from version control.

To repeat with no project build artifacts, copy this directory excluding `.lake/`, obtain the same dependency cache, and run `python3 verify.py` in that copy. A dependency cache can be reused without reusing any compiled `DeletionFloor` module.

## Manuscript-to-theorem map

All declarations below are in namespace `DeletionFloor`. Supporting lemmas in the same modules are also built and audited.

| Manuscript result | Formal declarations and correspondence |
|---|---|
| Definition 1 (`def:floor`) | `floor` is the real integral of the target-loss observable under the specified retraining measure. `deterministic_floor` reduces a Dirac retraining law to the loss at its fitted model. |
| Theorem 1 (`thm:floor`), Appendix A.1: transfer | `Equivalent` states both measurable-event probability inequalities. `one_way_transfer` derives expectation transfer by integrating event probabilities; `bounded_transfer` gives the full lower and upper bounds, including the complement inequalities and constant bounds. `manuscript_transfer` states this interval using `floor` and the printed endpoint definitions. |
| Theorem 1: sharpness | `feasible_iff`, `reference_feasible`, and `endpoints_feasible` characterize the feasible Bernoulli masses. `bernoulli_equivalent` checks all measurable events. `sharp_endpoints` constructs probability measures and a bounded measurable observable attaining both endpoints. `scaled_lower` and `scaled_upper` identify its normalized endpoints with the manuscript's formulas. Floors equal to zero or B are included. `manuscript_sharpness` connects the constructed laws directly to the same endpoint definitions used by `manuscript_transfer`. |
| Theorem 1: vanishing parameters | `scaled_endpoints_tendsto` proves convergence of both endpoints to the reference floor, including the boundary floors. `endpoints_tendsto` is its normalized version. `manuscript_endpoint_limit` proves convergence of the printed endpoints. |
| Equation (`eq:target-threshold`) | `target_threshold` derives the maximum of both logarithmic lower bounds from the two upper inequalities supplied by `bounded_transfer`, with delta zero and the paper's strict floor/target/B inequalities. `target_threshold_from_laws` proves the complete claim starting from equivalence of probability measures and the expected-loss objective. |
| Assumption (`as:lip`) and Equation (`eq:neighbor-bound`), Appendix A.2 | `neighbor_error` proves the triangle bound; `neighbor_floor` derives absolute- and squared-loss expectation bounds from measurable predictions and uniform almost-sure control. `clipped_expected_error` handles clipping; `finite_neighbor_min` handles a finite nonempty collection of valid bounds. `finite_neighbor_expected_min` and `finite_neighbor_clipped_min` carry that minimum through to the expected absolute, squared and clipped losses. `integrable_bounded` and `expected_error` establish the necessary integrability and integral inequalities. |
| Equations (`eq:ridge-adjust`, `eq:ridge-loo`), Appendix A.3 | `meanObjective_scale` turns the averaged squared loss into an unnormalized objective with penalty cardinality times lambda. `mean_minimizer_normal_equation` derives its normal equation from global minimization. `normalMatrix_posDef`, `erase_normalMatrix`, and `erase_normalRhs` identify the retained system. `indexed_ridge_residual` proves the residual formula with the (n−1)lambda matrix. `leverage_lt_one` proves the positive denominator directly from positive definiteness. `indexed_ridge_floor` connects the retained minimizer directly to the deterministic expected-loss floor. |
| Equation (`eq:ridge-change`) | `indexed_prediction_change` derives the formula from full and retained mean-loss minimizers, including the penalty-rescaling term. `ridge_prediction_change` supplies the matrix identity. |
| Appendix A.3: fixed unnormalized penalty | `square_minimizer_normal_equation` derives normal equations for that objective. `fixed_penalty_residual` connects the retained residual to the actual full-fit residual with unchanged penalty. `fixed_penalty_ratio` then proves the squared-loss ratio when the retained squared residual is positive; `deterministic_floor` identifies that residual square with its floor. `fixed_penalty_floor_ratio` assembles the complete ratio statement from the actual full and retained objective minimizers. |
| Equation (`eq:stability`), Appendix A.4 | `stability_from_objectives` starts from the full and retained averaged loss objectives with a shared regularizer, their gradients and minima, the gradient bound, strong convexity, and prediction regularity. It proves the stated 2GL/(mu n) prediction bound. `convex_support`, `strong_support`, `gradient_stationary`, `strong_gradient_distance`, `gradient_cancellation`, and `leave_one_out_stability` supply the analytic steps, including coincident minimizers. |

## Statement choices and scope

Records are indexed by a finite set of identifiers; feature vectors and labels may coincide across distinct identifiers. Erasing one identifier therefore preserves other copies. The ridge feature map is fixed across the full and retained objectives, and lambda is positive. The mean-loss and fixed unnormalized-penalty conventions are separate definitions.

The neighbor hypothesis uses the same constants for almost every predictor under Q. This replaces undefined topological support language in the earlier manuscript and requires measurable prediction observables. A finite intersection of validity sets justifies taking the minimum over neighbors. A large upper bound does not establish a large actual loss.

The transfer theorem uses arbitrary probability measures on a common measurable space. Event probabilities are written using `Measure.real`; probability-measure assumptions ensure these are finite. The proof integrates non-strict superlevel sets, an equivalent layer-cake convention to the manuscript's strict superlevel sets. Sharpness is an existence result on a two-outcome space, as in the manuscript, rather than an assertion of attainability in every model class.

Stability is proved on a complete real inner-product space, which includes the manuscript's finite-dimensional Euclidean setting. Mathlib's strong-convexity convention supplies the mu/2 quadratic term in the supporting inequality, yielding mu times squared distance after adding both directions. The formal theorem only needs strong convexity of the full objective; the manuscript assumes it for both objectives and therefore implies the formal hypothesis. Its prediction assumption is the stated Lipschitz inequality at the two minimizers, implied by the manuscript's global condition. No bounded-gradient or strong-convexity assumption is asserted for the measured neural model.

The audit checks proof dependencies; the theorem map supplies the separate human-readable correspondence to the manuscript. It does not certify experiments or establish that measured model laws satisfy the theorem hypotheses. Lean's [axiom reference](https://lean-lang.org/doc/reference/latest/Axioms/) explains the standard mathematical axioms used here.

## Consistency verification

All 63 project theorems, including the manuscript-level connections, are included in the build and transitive-axiom audit. The transfer theorem needs fewer parameter restrictions than the manuscript: transfer itself does not require epsilon or delta nonnegative once the event inequalities hold, while sharpness requires epsilon and delta nonnegative. The manuscript's delta-at-most-one condition is a permitted restriction of the formal result. The normalized and printed endpoint identities are proved explicitly.

The `floor` definition uses Lean's total real integral. Its paper interpretation requires the probability, measurability, nonnegativity and integrability conditions stated in the manuscript. Those conditions hold in the mapped results: bounded or almost-surely bounded observables are proved integrable, and deterministic ridge uses a Dirac law. The finite-neighbor result uses one target residual and different valid nonnegative bounds on it; it does not minimize over different target losses.

The ridge proofs use finite sums of row outer products for the normal matrix and sums of label-scaled rows for its right-hand side. These are the coordinate expansions of the manuscript's transposed-matrix products. The prediction-change identity is signed; the experiment reports its absolute value. Its mean-loss penalty is lambda, with normal-equation coefficient n times lambda, and the strong-convexity convention has the separate coefficient mu.

The source review compared every formal conclusion and hypothesis against Method and Appendix A. Targeted execution also compared the actual Python ridge backend against the retained-residual and signed prediction-change identities, and checked recorded request quantities against their definitions. These numerical checks supplement the formal proofs; they are not formal verification of Python or a rerun of the materials experiments.
