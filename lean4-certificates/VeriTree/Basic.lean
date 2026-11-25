/-
  VeriTree: Certified Machine Unlearning and Quasi-ML Residual Propagation in GBDTs
  Layer 1 & 2 Formal Proofs
  File: VeriTree/Basic.lean
  Phase 1 Step 1.1: Base Convexity & Gain Definitions
-/

import Mathlib

namespace VeriTree

/-- The regularized single-child split score ψ(G, H) = G² / (H + λ).
    Defined on the positive regularized half-space H + λ > 0. -/
noncomputable def psi (G H lambda_reg : ℝ) : ℝ :=
  (G * G) / (H + lambda_reg)

/-- The total regularized two-child split score:
    Φ(s) = ψ(G_L(s), H_L(s)) + ψ(G_R(s), H_R(s))
         = G_L(s)² / (H_L(s) + λ) + G_R(s)² / (H_R(s) + λ) -/
noncomputable def Phi (G_L H_L G_R H_R lambda_reg : ℝ) : ℝ :=
  psi G_L H_L lambda_reg + psi G_R H_R lambda_reg

/-- Node split gain:
    G(s; G_v, H_v) = 1/2 Φ(s) - G_v² / (2(H_v + λ)) - γ -/
noncomputable def NodeGain (G_L H_L G_R H_R G_v H_v lambda_reg gamma : ℝ) : ℝ :=
  (Phi G_L H_L G_R H_R lambda_reg) / 2 - (G_v * G_v) / (2 * (H_v + lambda_reg)) - gamma

/-- Structure representing a discrete split candidate threshold at node v -/
structure SplitCandidate where
  candidate_id : Nat
  feature_idx  : Nat
  bin_idx      : Nat
  G_L          : ℝ
  H_L          : ℝ
  G_R          : ℝ
  H_R          : ℝ

/-- Evaluation of split score for a candidate -/
noncomputable def SplitCandidate.evalScore (s : SplitCandidate) (lambda_reg : ℝ) : ℝ :=
  Phi s.G_L s.H_L s.G_R s.H_R lambda_reg

/-- Foundational Lemma: ψ(G, H) = G² / (H + λ) is jointly convex on the half-space H + λ > 0.
    Proof: For any (G₁,H₁), (G₂,H₂) with H₁+λ > 0, H₂+λ > 0, and θ ∈ [0,1]:
      ψ(θG₁+(1-θ)G₂, θH₁+(1-θ)H₂, λ) ≤ θ·ψ(G₁,H₁,λ) + (1-θ)·ψ(G₂,H₂,λ)
    Let a = H₁+λ > 0, b = H₂+λ > 0, c = θa+(1-θ)b > 0. Note c = θH₁+(1-θ)H₂+λ.
    Clearing denominators, we need:
      (θG₁+(1-θ)G₂)² · a · b ≤ c · (θ·G₁²·b + (1-θ)·G₂²·a)
    The difference RHS − LHS = θ(1−θ)·(a·G₂ − b·G₁)² ≥ 0, which proves the result. -/

theorem psi_jointly_convex_inequality
    {G1 H1 G2 H2 lambda_reg : ℝ}
    (h1 : 0 < H1 + lambda_reg)
    (h2 : 0 < H2 + lambda_reg)
    {theta : ℝ}
    (htheta : 0 ≤ theta)
    (htheta' : theta ≤ 1) :
    psi (theta * G1 + (1 - theta) * G2) (theta * H1 + (1 - theta) * H2) lambda_reg ≤
      theta * psi G1 H1 lambda_reg + (1 - theta) * psi G2 H2 lambda_reg := by
  have h_c_pos : 0 < theta * H1 + (1 - theta) * H2 + lambda_reg := by
    have h_id : theta * H1 + (1 - theta) * H2 + lambda_reg =
        theta * (H1 + lambda_reg) + (1 - theta) * (H2 + lambda_reg) := by ring
    rw [h_id]
    rcases eq_or_lt_of_le htheta with rfl | h_pos
    · simpa using h2
    · have h_t1 : 0 < theta * (H1 + lambda_reg) := mul_pos h_pos h1
      have h_t2 : 0 ≤ (1 - theta) * (H2 + lambda_reg) := mul_nonneg (sub_nonneg.mpr htheta') h2.le
      linarith
  -- Define a = H1 + λ, b = H2 + λ, c = θH1 + (1-θ)H2 + λ = θa + (1-θ)b
  have h_a_pos : 0 < H1 + lambda_reg := h1
  have h_b_pos : 0 < H2 + lambda_reg := h2
  -- Key algebraic inequality: (θG₁+(1-θ)G₂)² · a · b ≤ c · (θ·G₁²·b + (1-θ)·G₂²·a)
  -- because RHS − LHS = θ(1−θ)·(a·G₂ − b·G₁)² ≥ 0
  have h_main : (theta * G1 + (1 - theta) * G2) ^ 2 * (H1 + lambda_reg) * (H2 + lambda_reg) ≤
      (theta * H1 + (1 - theta) * H2 + lambda_reg) *
        (theta * G1 ^ 2 * (H2 + lambda_reg) + (1 - theta) * G2 ^ 2 * (H1 + lambda_reg)) := by
    have h₁ : 0 ≤ theta * (1 - theta) := by nlinarith
    -- Verify RHS − LHS = θ(1−θ)·((H₁+λ)·G₂ − (H₂+λ)·G₁)² ≥ 0
    nlinarith [sq_nonneg ((H1 + lambda_reg) * G2 - (H2 + lambda_reg) * G1),
      mul_nonneg htheta (sub_nonneg.mpr htheta'),
      mul_pos h1 h2]
  -- Divide both sides by c · a · b > 0 to get the convexity inequality
  have h_denom_pos : 0 < (theta * H1 + (1 - theta) * H2 + lambda_reg) * (H1 + lambda_reg) * (H2 + lambda_reg) := by
    positivity
  have h₆ : (theta * G1 + (1 - theta) * G2) ^ 2 / (theta * H1 + (1 - theta) * H2 + lambda_reg) ≤
      theta * (G1 ^ 2 / (H1 + lambda_reg)) + (1 - theta) * (G2 ^ 2 / (H2 + lambda_reg)) := by
    -- h_main: LHS_num ≤ RHS_num where both sides are multiplied by c·a·b
    -- Divide both sides by c·a·b > 0 using div_le_div_of_nonneg_right
    have h₇ : 0 < (theta * H1 + (1 - theta) * H2 + lambda_reg) * (H1 + lambda_reg) * (H2 + lambda_reg) := by positivity
    have h₈ : ((theta * G1 + (1 - theta) * G2) ^ 2 * (H1 + lambda_reg) * (H2 + lambda_reg)) /
        ((theta * H1 + (1 - theta) * H2 + lambda_reg) * (H1 + lambda_reg) * (H2 + lambda_reg)) ≤
        ((theta * H1 + (1 - theta) * H2 + lambda_reg) *
          (theta * G1 ^ 2 * (H2 + lambda_reg) + (1 - theta) * G2 ^ 2 * (H1 + lambda_reg))) /
        ((theta * H1 + (1 - theta) * H2 + lambda_reg) * (H1 + lambda_reg) * (H2 + lambda_reg)) := by
      exact div_le_div_of_nonneg_right h_main (by positivity)
    -- Simplify both sides
    calc
      (theta * G1 + (1 - theta) * G2) ^ 2 / (theta * H1 + (1 - theta) * H2 + lambda_reg)
        = ((theta * G1 + (1 - theta) * G2) ^ 2 * (H1 + lambda_reg) * (H2 + lambda_reg)) /
            ((theta * H1 + (1 - theta) * H2 + lambda_reg) * (H1 + lambda_reg) * (H2 + lambda_reg)) := by
          field_simp [h_c_pos.ne', h1.ne', h2.ne']
      _ ≤ ((theta * H1 + (1 - theta) * H2 + lambda_reg) *
          (theta * G1 ^ 2 * (H2 + lambda_reg) + (1 - theta) * G2 ^ 2 * (H1 + lambda_reg))) /
          ((theta * H1 + (1 - theta) * H2 + lambda_reg) * (H1 + lambda_reg) * (H2 + lambda_reg)) := by
        exact h₈
      _ = theta * (G1 ^ 2 / (H1 + lambda_reg)) + (1 - theta) * (G2 ^ 2 / (H2 + lambda_reg)) := by
        field_simp [h_c_pos.ne', h1.ne', h2.ne']
  -- Convert back to psi notation: h₆ is exactly theta*psi G1 H1 + (1-theta)*psi G2 H2
  calc
    psi (theta * G1 + (1 - theta) * G2) (theta * H1 + (1 - theta) * H2) lambda_reg
      = (theta * G1 + (1 - theta) * G2) ^ 2 / (theta * H1 + (1 - theta) * H2 + lambda_reg) := by
        simp [psi, sq]
    _ ≤ theta * (G1 ^ 2 / (H1 + lambda_reg)) + (1 - theta) * (G2 ^ 2 / (H2 + lambda_reg)) := by
      exact h₆
    _ = theta * psi G1 H1 lambda_reg + (1 - theta) * psi G2 H2 lambda_reg := by
      simp [psi, sq]

/-- Perturbation box envelope: [-m_v * g_max, m_v * g_max] × [0, m_v * h_max].
    By joint convexity, the maximum perturbation is attained at the extreme vertices. -/
structure VertexPerturbationBound (m_v g_max h_max H_min lambda_reg : ℝ) where
  h_admissible : 0 < H_min - m_v * h_max + lambda_reg
  max_num_shift : ℝ
  denom_lower   : ℝ

/-- Construction of certified extreme vertex envelope -/
def vertex_bound_admissible (m_v g_max h_max H_min lambda_reg : ℝ)
    (h_adm : 0 < H_min - m_v * h_max + lambda_reg) :
    VertexPerturbationBound m_v g_max h_max H_min lambda_reg :=
  {
    h_admissible := h_adm,
    max_num_shift := (2 : ℝ) * m_v * g_max,
    denom_lower := H_min - m_v * h_max + lambda_reg
  }

end VeriTree
