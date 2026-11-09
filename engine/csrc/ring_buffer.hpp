#pragma once

#include <cstdint>
#include <cstddef>

namespace veritree {

constexpr uint32_t RING_BUFFER_K = 4;
constexpr float DEFAULT_LAMBDA = 1.0f;
constexpr float DEFAULT_GAMMA = 0.0f;

// Cache-aligned Split-Sensitivity Ring Buffer (k=4)
// Fits in exactly two 64-byte L1 cache lines (128 bytes per node)
struct alignas(64) SplitSensitivityRingBuffer {
    // Cache Line 0 (64 bytes): Top-4 candidate splits + slice statistics
    uint16_t feature_idx[4];       // 8B: Candidate feature indices f_1..f_4
    uint16_t bin_idx[4];           // 8B: Candidate threshold bins \tau_1..\tau_4
    float    G_left[4];            // 16B: Left child gradient sums G_{L,1..4}
    float    H_left[4];            // 16B: Left child Hessian sums H_{L,1..4}
    float    dG_slice[2];          // 8B: Adjacent bin slice \Delta G_{slice} (left/right)
    float    dH_slice[2];          // 8B: Adjacent bin slice \Delta H_{slice} (left/right)

    // Cache Line 1 (64 bytes): Parent totals, tail bounds & safe radius
    float    G_node;               // 4B: Total node gradient G_v
    float    H_node;               // 4B: Total node Hessian H_v
    float    phi_tail;             // 4B: Uncached tail ceiling \Phi_{tail}
    float    H_min_tail;           // 4B: Minimum Hessian across uncached bins
    float    G_abs_max;            // 4B: |G|_{max} across uncached bins
    float    g_max;                // 4B: Max single-sample |g_i| in I_v
    float    h_max;                // 4B: Max single-sample h_i in I_v
    uint32_t safe_radius_m_star;   // 4B: Pre-certified safe deletion radius m_v^*
    int32_t  left_child_id;        // 4B: Pointer/index to left subtree
    int32_t  right_child_id;       // 4B: Pointer/index to right subtree
    uint32_t node_sample_count;    // 4B: |I_v|
    uint8_t  reserved[20];         // 20B: Padding to exact 128-byte boundary
};

static_assert(sizeof(SplitSensitivityRingBuffer) == 128, "Must occupy 2 L1 cache lines");
static_assert(alignof(SplitSensitivityRingBuffer) == 64, "Must align to 64-byte cache line");

enum CertResult : int32_t {
    CERT_INVARIANT = 0,             // Layer 1 holds: s̃* = s_1, O(depth) pointer splice
    CERT_RUNNER_UP_DOMINANT = 1,    // Layer 2 holds: runner-up r* beats tail ceiling
    CERT_FALLBACK_SCAN = 2          // Neither holds: raw scan fallback required
};

struct CertEvaluationOutput {
    CertResult result;
    uint32_t winning_candidate_idx; // 0 for s_1, 1..3 for runner-ups
    float winning_score;
    float tail_bound;
};

} // namespace veritree

