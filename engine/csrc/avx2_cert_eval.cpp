#include "ring_buffer.hpp"
#include <immintrin.h>
#include <cmath>
#include <algorithm>

namespace veritree {

// Inline helper for IEEE-754 division with strict determinism
static inline float compute_psi(float G, float H, float lambda_reg) {
    return (G * G) / (H + lambda_reg);
}

// Compute Φ(s) = ψ(G_L, H_L, λ) + ψ(G_R, H_R, λ)
static inline float compute_phi(float G_L, float H_L, float G_node, float H_node, float lambda_reg) {
    float G_R = G_node - G_L;
    float H_R = H_node - H_L;
    return compute_psi(G_L, H_L, lambda_reg) + compute_psi(G_R, H_R, lambda_reg);
}

// Compute uncached tail upper envelope Φ̄_tail(m_v)
static inline float compute_tail_upper_bound(
    float phi_tail,
    float H_min_tail,
    float G_abs_max,
    float g_max,
    float h_max,
    uint32_t m_v,
    float lambda_reg
) {
    float m_f = static_cast<float>(m_v);
    float denom = H_min_tail - m_f * h_max + lambda_reg;
    if (denom <= 0.0f) {
        return 1e30f; // Denominator breached: must trigger fallback
    }
    float delta_phi = 2.0f * m_f * g_max * G_abs_max + (m_f * m_f) * (g_max * g_max);
    float num = phi_tail * (H_min_tail + lambda_reg) + delta_phi;
    return num / denom;
}

// SIMD 4-way parallel score evaluation using AVX2/SSE registers
extern "C" CertEvaluationOutput evaluate_layer1_certificate_avx2(
    const SplitSensitivityRingBuffer* __restrict node,
    uint32_t m_v,
    const float* __restrict dG_left,
    const float* __restrict dH_left,
    float dG_node,
    float dH_node,
    float taylor_interval_E,
    float lambda_reg
) {
    CertEvaluationOutput out;
    out.winning_candidate_idx = 0;

    // Fast-path check: if m_v <= safe_radius_m_star and no Taylor error, return immediately
    if (m_v <= node->safe_radius_m_star && taylor_interval_E == 0.0f) {
        out.result = CERT_INVARIANT;
        out.winning_score = compute_phi(node->G_left[0], node->H_left[0], node->G_node, node->H_node, lambda_reg);
        out.tail_bound = node->phi_tail;
        return out;
    }

    // Compute post-deletion totals for the node
    float tilde_G_node = node->G_node - dG_node;
    float tilde_H_node = node->H_node - dH_node;

    // Load left statistics into 128-bit SIMD registers (4 x float)
    __m128 v_G_L = _mm_loadu_ps(node->G_left);
    __m128 v_H_L = _mm_loadu_ps(node->H_left);
    __m128 v_dG_L = _mm_loadu_ps(dG_left);
    __m128 v_dH_L = _mm_loadu_ps(dH_left);

    // Subtract perturbations: G̃_L = G_L - ΔG_L, H̃_L = H_L - ΔH_L
    __m128 v_tilde_G_L = _mm_sub_ps(v_G_L, v_dG_L);
    __m128 v_tilde_H_L = _mm_sub_ps(v_H_L, v_dH_L);

    // Compute right statistics: G̃_R = G̃_node - G̃_L, H̃_R = H̃_node - H̃_L
    __m128 v_tilde_G_node = _mm_set1_ps(tilde_G_node);
    __m128 v_tilde_H_node = _mm_set1_ps(tilde_H_node);
    __m128 v_tilde_G_R = _mm_sub_ps(v_tilde_G_node, v_tilde_G_L);
    __m128 v_tilde_H_R = _mm_sub_ps(v_tilde_H_node, v_tilde_H_L);

    // Add regularization: H̃_L + λ, H̃_R + λ
    __m128 v_lambda = _mm_set1_ps(lambda_reg);
    __m128 v_den_L = _mm_add_ps(v_tilde_H_L, v_lambda);
    __m128 v_den_R = _mm_add_ps(v_tilde_H_R, v_lambda);

    // Numerators: G̃_L^2, G̃_R^2
    __m128 v_num_L = _mm_mul_ps(v_tilde_G_L, v_tilde_G_L);
    __m128 v_num_R = _mm_mul_ps(v_tilde_G_R, v_tilde_G_R);

    // Exact IEEE-754 divisions
    __m128 v_psi_L = _mm_div_ps(v_num_L, v_den_L);
    __m128 v_psi_R = _mm_div_ps(v_num_R, v_den_R);

    // Total post-deletion scores Φ̃(s_r) for r ∈ {0, 1, 2, 3}
    __m128 v_scores = _mm_add_ps(v_psi_L, v_psi_R);

    // Store scores back to stack
    alignas(16) float scores[4];
    _mm_storeu_ps(scores, v_scores);

    // Adjust for Taylor error interval bound if applicable:
    // For s_1 we test the lower bound: Φ̃(s_1) - Taylor margin
    float score_s1 = scores[0];
    if (taylor_interval_E > 0.0f) {
        score_s1 -= taylor_interval_E;
    }

    // Evaluate uncached tail bound
    float tail_bound = compute_tail_upper_bound(
        node->phi_tail,
        node->H_min_tail,
        node->G_abs_max,
        node->g_max,
        node->h_max,
        m_v,
        lambda_reg
    );
    out.tail_bound = tail_bound;

    // Find max among runner-ups {s_2, s_3, s_4}
    uint32_t best_runner_up_idx = 1;
    float max_runner_up_score = scores[1];
    for (uint32_t r = 2; r < 4; ++r) {
        if (scores[r] > max_runner_up_score) {
            max_runner_up_score = scores[r];
            best_runner_up_idx = r;
        }
    }

    // Layer 1 Test: Is s_1 invariant?
    if (score_s1 > max_runner_up_score && score_s1 > tail_bound) {
        out.result = CERT_INVARIANT;
        out.winning_candidate_idx = 0;
        out.winning_score = scores[0];
        return out;
    }

    // Layer 2 Test (Theorem 2): Is runner-up dominant over the tail?
    if (max_runner_up_score > tail_bound) {
        out.result = CERT_RUNNER_UP_DOMINANT;
        out.winning_candidate_idx = best_runner_up_idx;
        out.winning_score = max_runner_up_score;
        return out;
    }

    // Neither certificate held: Fallback to raw data scan
    out.result = CERT_FALLBACK_SCAN;
    out.winning_candidate_idx = 0;
    out.winning_score = scores[0];
    return out;
}

} // namespace veritree

