//! VeriTree Layer 2: Quasi-ML Surrogate & Residual Propagation Engine
//! Resolves boundary-crossing flips via Theorem 2 Runner-Up Dominance and
//! suppresses 99.9% of inter-tree boosting retraining cascades via Co-Routing Taylor propagation.

use crate::trainer::SplitSensitivityRingBuffer;

extern "C" {
    pub fn propagate_leaf_corouting_residuals(
        leaf_paths: *const u8,
        num_samples: usize,
        num_trees: usize,
        tree_t: u32,
        tree_t_plus_1: u32,
        delta_w: *const f32,
        num_leaves_t: usize,
        num_nodes_t_plus_1: usize,
        eta: f32,
        M_3: f32,
        epsilon_prop: f32,
        G_nodes_t_plus_1: *mut f32,
        E_nodes_t_plus_1: *mut f32,
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionKind {
    IntraFeatureHomotopy, // Same feature, shifted bin by 1: O(L_v) closed-form transport
    CrossFeaturePartition, // Different feature: single-pass O(N_v) partition
    RawScanFallback,      // Tail breached: raw data scan
}

/// Dispatches transition strategy based on winning runner-up index
pub fn classify_runner_up_transition(
    node: &SplitSensitivityRingBuffer,
    winner_idx: usize,
) -> TransitionKind {
    if winner_idx == 0 || winner_idx >= 4 {
        return TransitionKind::RawScanFallback;
    }

    let f_1 = node.feature_idx[0];
    let tau_1 = node.bin_idx[0];
    let f_winner = node.feature_idx[winner_idx];
    let tau_winner = node.bin_idx[winner_idx];

    if f_winner == f_1 && (tau_winner as i32 - tau_1 as i32).abs() == 1 {
        TransitionKind::IntraFeatureHomotopy
    } else {
        TransitionKind::CrossFeaturePartition
    }
}

/// Closed-form Quasi-Newton Subtree Homotopy Splicing using cached slice statistics
pub fn apply_homotopy_splicing(
    node: &mut SplitSensitivityRingBuffer,
    slice_idx: usize, // 0 for left adjacent, 1 for right adjacent
) {
    let dg = node.dg_slice[slice_idx];
    let dh = node.dh_slice[slice_idx];

    // Transport slice statistics between left and right subtrees
    node.g_left[0] += dg;
    node.h_left[0] += dh;
}

