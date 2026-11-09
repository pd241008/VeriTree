//! VeriTree Layer 1: Split-Invariance Certificate Evaluator
//! Resolves ~99.4% of deletion requests via O(depth) pointer splicing and leaf weight updates.

use crate::trainer::SplitSensitivityRingBuffer;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CertResult {
    CertInvariant = 0,
    CertRunnerUpDominant = 1,
    CertFallbackScan = 2,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CertEvaluationOutput {
    pub result: CertResult,
    pub winning_candidate_idx: u32,
    pub winning_score: f32,
    pub tail_bound: f32,
}

extern "C" {
    pub fn evaluate_layer1_certificate_avx2(
        node: *const SplitSensitivityRingBuffer,
        m_v: u32,
        dG_left: *const f32,
        dH_left: *const f32,
        dG_node: f32,
        dH_node: f32,
        taylor_interval_E: f32,
        lambda_reg: f32,
    ) -> CertEvaluationOutput;
}

/// Evaluates Layer 1 Certificate for an internal node
pub fn check_layer1_invariance(
    node: &SplitSensitivityRingBuffer,
    m_v: u32,
    dg_left: &[f32; 4],
    dh_left: &[f32; 4],
    dg_node: f32,
    dh_node: f32,
    taylor_e: f32,
    lambda_reg: f32,
) -> CertEvaluationOutput {
    unsafe {
        evaluate_layer1_certificate_avx2(
            node as *const _,
            m_v,
            dg_left.as_ptr(),
            dh_left.as_ptr(),
            dg_node,
            dh_node,
            taylor_e,
            lambda_reg,
        )
    }
}

/// Exact O(depth) leaf weight update under certified invariance
#[inline]
pub fn update_leaf_weight_exact(
    g_leaf: f32,
    h_leaf: f32,
    delta_g: f32,
    delta_h: f32,
    lambda_reg: f32,
) -> f32 {
    let tilde_g = g_leaf - delta_g;
    let tilde_h = h_leaf - delta_h;
    -tilde_g / (tilde_h + lambda_reg)
}

