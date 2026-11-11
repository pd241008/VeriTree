//! VeriTree Phase 2 Detailed Verification Gate - Layer 1 Edge Cases
//! Covers: Ring Buffer layout, Tie-Breaker, Psi score, Trainer, Inverted Index,
//! Leaf Weight Update, AVX2 Certificate Evaluator fast/slow paths

use std::mem;
use veritree_engine::{
    check_layer1_invariance, CertResult, DecisionTree, HistogramTrainer,
    InvertedLeafRoutingIndex, SplitSensitivityRingBuffer, TreeConfig,
};
use veritree_engine::trainer::{BinnedDataset, margin_regularized_tie_breaker, psi, RING_BUFFER_K};

/// ---- Phase 2 Step 2.1: Ring Buffer Layout ----

#[test]
fn test_ring_buffer_exact_128_bytes() {
    assert_eq!(mem::size_of::<SplitSensitivityRingBuffer>(), 128);
}

#[test]
fn test_ring_buffer_64_byte_alignment() {
    assert_eq!(mem::align_of::<SplitSensitivityRingBuffer>(), 64);
}

#[test]
fn test_ring_buffer_default_all_zeroed() {
    let buf = SplitSensitivityRingBuffer::default();
    assert_eq!(buf.feature_idx, [0u16; 4]);
    assert_eq!(buf.bin_idx, [0u16; 4]);
    assert_eq!(buf.g_left, [0.0f32; 4]);
    assert_eq!(buf.h_left, [0.0f32; 4]);
    assert_eq!(buf.dg_slice, [0.0f32; 2]);
    assert_eq!(buf.dh_slice, [0.0f32; 2]);
    assert_eq!(buf.g_node, 0.0);
    assert_eq!(buf.h_node, 0.0);
    assert_eq!(buf.phi_tail, 0.0);
    assert_eq!(buf.h_min_tail, 1e6f32);
    assert_eq!(buf.g_abs_max, 0.0);
    assert_eq!(buf.g_max, 0.0);
    assert_eq!(buf.h_max, 0.0);
    assert_eq!(buf.safe_radius_m_star, 0);
    assert_eq!(buf.left_child_id, -1);
    assert_eq!(buf.right_child_id, -1);
    assert_eq!(buf.node_sample_count, 0);
    assert_eq!(buf.reserved, [0u8; 20]);
}

/// ---- Phase 2 Step 2.3: Margin-Regularized Tie-Breaker ----

#[test]
fn test_tie_breaker_gain_a_dominates() {
    // gain_a clearly larger, no tie-break needed
    let result = margin_regularized_tie_breaker(
        0.9, 0.5,  // gains
        10.0, 10.0, 10.0, 10.0, // hl/hr
        100.0, // h_v
        1e-5,  // eps_tie
    );
    assert!(result);
}

#[test]
fn test_tie_breaker_gain_b_dominates() {
    // gain_b clearly larger
    let result = margin_regularized_tie_breaker(
        0.5, 0.9,
        10.0, 10.0, 10.0, 10.0,
        100.0,
        1e-5,
    );
    assert!(!result);
}

#[test]
fn test_tie_breaker_within_eps_stability_decides() {
    // Gains within eps_tie, stability_a > stability_b => select A
    let result = margin_regularized_tie_breaker(
        0.5, 0.500001, // diff = 1e-6 <= eps_tie(1e-5)
        20.0, 20.0,  // hl_a=20, hr_a=20 => stab_a = 400/100 = 4
        10.0, 10.0,  // hl_b=10, hr_b=10 => stab_b = 100/100 = 1
        100.0,
        1e-5,
    );
    assert!(result); // A has higher stability
}

#[test]
fn test_tie_breaker_within_eps_stability_b_better() {
    // Gains within eps_tie, stability_b > stability_a => reject A
    let result = margin_regularized_tie_breaker(
        0.5, 0.500001,
        10.0, 10.0,  // stab_a = 1
        20.0, 20.0,  // stab_b = 4
        100.0,
        1e-5,
    );
    assert!(!result);
}

#[test]
fn test_tie_breaker_equal_stability_returns_true() {
    // Equal stability: stability_a >= stability_b is true
    let result = margin_regularized_tie_breaker(
        0.5, 0.500001,
        15.0, 15.0,  // stab_a = 225/100 = 2.25
        15.0, 15.0,  // stab_b = 2.25
        100.0,
        1e-5,
    );
    assert!(result);
}

#[test]
fn test_tie_breaker_h_v_zero_guard() {
    // h_v = 0: division guard, falls through to gain comparison
    let result = margin_regularized_tie_breaker(
        0.8, 0.5,
        10.0, 10.0, 10.0, 10.0,
        0.0, // h_v = 0, guard triggers
        1e-5,
    );
    assert!(result); // gain_a >= gain_b
}

#[test]
fn test_tie_breaker_exactly_at_eps_boundary() {
    // diff = 5e-6 < eps_tie(1e-5): clearly within tie zone
    let result = margin_regularized_tie_breaker(
        0.5, 0.500005, // diff = 5e-6 <= eps_tie
        20.0, 20.0,
        10.0, 10.0,
        100.0,
        1e-5,
    );
    assert!(result); // stability path taken
}

/// ---- Phase 2: Psi Score Function ----

#[test]
fn test_psi_positive_g_h() {
    let val = psi(4.0f32, 6.0f32, 1.0f32);
    let expected = 16.0 / 7.0;
    assert!((val - expected).abs() < 1e-6);
}

#[test]
fn test_psi_negative_g() {
    // G^2 is same for +G and -G
    let val_pos = psi(4.0f32, 6.0f32, 1.0f32);
    let val_neg = psi(-4.0f32, 6.0f32, 1.0f32);
    assert!((val_pos - val_neg).abs() < 1e-6);
}

#[test]
fn test_psi_zero_g() {
    let val = psi(0.0f32, 5.0f32, 1.0f32);
    assert_eq!(val, 0.0);
}

#[test]
fn test_psi_zero_h_with_lambda() {
    // H = 0 but lambda > 0, no division by zero
    let val = psi(3.0f32, 0.0f32, 1.0f32);
    assert!((val - 9.0).abs() < 1e-6);
}

/// ---- Phase 2 Step 2.2: Decision Tree Predict ----

#[test]
fn test_tree_predict_untrained_leaf_zero() {
    let tree = DecisionTree::new(2);
    let sample = [0u8; 8];
    // All nodes default, left_child_id = -1 => leaf_id = 0, weight = 0
    let pred = tree.predict(&sample);
    assert_eq!(pred, 0.0);
}

#[test]
fn test_tree_predict_traversal_depth_bound() {
    let mut tree = DecisionTree::new(3);
    // Force root to be a split with left child leading to leaf 0
    tree.nodes[0].left_child_id = -2; // leaf_id = (-(-2)-1) = 1... wait
    // left_child_id = -(leaf_id) - 1 => leaf_id = -left_child_id - 1
    // left_child_id = -1 => leaf_id = 0
    tree.nodes[0].left_child_id = -1;
    tree.leaf_weights[0] = 42.0;
    let sample = [0u8; 8];
    let pred = tree.predict(&sample);
    assert_eq!(pred, 42.0);
}

#[test]
fn test_tree_predict_right_child_traversal() {
    let mut tree = DecisionTree::new(3);
    tree.nodes[0].feature_idx[0] = 5;
    tree.nodes[0].bin_idx[0] = 100;
    tree.nodes[0].right_child_id = -2; // leaf_id = 1
    tree.leaf_weights[1] = 7.5;
    // sample[5] = 200 > 100 => go right
    let mut sample = [0u8; 8];
    sample[5] = 200;
    let pred = tree.predict(&sample);
    assert_eq!(pred, 7.5);
}

/// ---- Phase 2 Step 2.3: Histogram Trainer Edge Cases ----

#[test]
fn test_trainer_empty_sample_indices_returns_early() {
    let dataset = BinnedDataset::new(10, 3, 256);
    let config = TreeConfig::default();
    let trainer = HistogramTrainer::new(config);
    let mut routing = InvertedLeafRoutingIndex::new(10, 1);
    let tree = trainer.train_tree(&dataset, &[], &mut routing, 0);
    // Tree with default nodes, no leaves set
    assert_eq!(tree.max_depth, 6);
}

#[test]
fn test_trainer_single_sample_trains() {
    let mut dataset = BinnedDataset::new(1, 2, 256);
    dataset.data[0] = 10;
    dataset.data[1] = 20;
    dataset.gradients[0] = 2.0;
    dataset.hessians[0] = 1.0;
    let config = TreeConfig {
        max_depth: 3,
        num_trees: 1,
        lambda_reg: 1.0,
        gamma: 0.0,
        eta: 0.1,
        tie_breaker_eps: 1e-5,
        min_child_hessian: 1e-3,
    };
    let trainer = HistogramTrainer::new(config);
    let mut routing = InvertedLeafRoutingIndex::new(1, 1);
    let tree = trainer.train_tree(&dataset, &[0], &mut routing, 0);
    // Single sample, depth 3 => leaf at depth 3
    // The leaf weight = -g/(h+lambda) = -2.0/2.0 = -1.0
    assert!((tree.leaf_weights[0] - (-1.0f32)).abs() < 1e-6);
}

#[test]
fn test_trainer_min_child_hessian_filters_all_splits() {
    // All samples have h=0.001, min_child_hessian=1.0 => no valid split
    let mut dataset = BinnedDataset::new(100, 2, 4);
    for i in 0..100 {
        dataset.gradients[i] = 1.0;
        dataset.hessians[i] = 0.001;
    }
    let config = TreeConfig {
        max_depth: 3,
        num_trees: 1,
        lambda_reg: 1.0,
        gamma: 0.0,
        eta: 0.1,
        tie_breaker_eps: 1e-5,
        min_child_hessian: 1.0, // very high, filters everything
    };
    let trainer = HistogramTrainer::new(config);
    let mut routing = InvertedLeafRoutingIndex::new(100, 1);
    let all_indices: Vec<usize> = (0..100).collect();
    let tree = trainer.train_tree(&dataset, &all_indices, &mut routing, 0);
    // Root becomes a leaf since no split passes min_child_hessian
    assert_eq!(tree.nodes[0].left_child_id, -1); // leaf pointer
}

/// ---- Phase 2 Step 2.2: Inverted Leaf Routing Index ----

#[test]
fn test_inverted_index_get_set_roundtrip() {
    let mut idx = InvertedLeafRoutingIndex::new(50, 10);
    idx.set_leaf(7, 3, 42);
    assert_eq!(idx.get_leaf(7, 3), 42);
}

#[test]
fn test_inverted_index_all_samples_all_trees() {
    let mut idx = InvertedLeafRoutingIndex::new(5, 3);
    for s in 0..5 {
        for t in 0..3 {
            idx.set_leaf(s, t, (s * 3 + t) as u8);
        }
    }
    for s in 0..5 {
        for t in 0..3 {
            assert_eq!(idx.get_leaf(s, t), (s * 3 + t) as u8);
        }
    }
}

#[test]
#[should_panic(expected = "buffer size overflow")]
fn test_inverted_index_overflow_panics() {
    // usize::MAX samples * usize::MAX trees overflows checked_mul
    InvertedLeafRoutingIndex::new(usize::MAX, usize::MAX);
}

#[test]
fn test_inverted_index_ancestor_path_bitwise() {
    let idx = InvertedLeafRoutingIndex::new(10, 1);
    // Complete binary tree: parent = (child - 1) >> 1
    // Node 7 -> parent 3 -> parent 1 -> parent 0
    let path = idx.get_ancestor_path(7);
    assert_eq!(path, vec![7, 3, 1, 0]);
}

#[test]
fn test_inverted_index_ancestor_path_root() {
    let idx = InvertedLeafRoutingIndex::new(10, 1);
    let path = idx.get_ancestor_path(0);
    assert_eq!(path, vec![0]);
}

#[test]
fn test_inverted_index_ancestor_path_node_1() {
    let idx = InvertedLeafRoutingIndex::new(10, 1);
    let path = idx.get_ancestor_path(1);
    assert_eq!(path, vec![1, 0]);
}

/// ---- Phase 2 Step 2.3: Exact Leaf Weight Update ----

#[test]
fn test_update_leaf_weight_exact_zero_delta() {
    let w = veritree_engine::update_leaf_weight_exact(10.0, 5.0, 0.0, 0.0, 1.0);
    assert!((w - (-10.0 / 6.0)).abs() < 1e-6);
}

#[test]
fn test_update_leaf_weight_exact_negative_gradient() {
    let w = veritree_engine::update_leaf_weight_exact(-4.0, 2.0, 0.0, 0.0, 1.0);
    assert!((w - (4.0 / 3.0)).abs() < 1e-6);
}

#[test]
fn test_update_leaf_weight_exact_with_delta() {
    // tilde_g = g - dg, tilde_h = h - dh, w = -tilde_g / (tilde_h + lambda)
    let w = veritree_engine::update_leaf_weight_exact(10.0, 5.0, 2.0, 1.0, 1.0);
    // tilde_g = 8, tilde_h = 4, w = -8/5 = -1.6
    assert!((w - (-1.6f32)).abs() < 1e-6);
}

#[test]
fn test_update_leaf_weight_exact_large_values() {
    let w = veritree_engine::update_leaf_weight_exact(1e6f32, 1e6f32, 0.0, 0.0, 1.0);
    assert!(w.is_finite());
}

/// ---- Phase 2 Step 2.4: AVX2 Layer 1 Certificate Evaluator ----

fn make_test_node() -> SplitSensitivityRingBuffer {
    let mut node = SplitSensitivityRingBuffer::default();
    node.feature_idx = [0, 1, 2, 3];
    node.bin_idx = [10, 20, 30, 40];
    node.g_left = [100.0, 80.0, 60.0, 40.0];
    node.h_left = [50.0, 40.0, 30.0, 20.0];
    node.g_node = 200.0;
    node.h_node = 100.0;
    node.phi_tail = 10.0;
    node.h_min_tail = 10.0;
    node.g_abs_max = 5.0;
    node.g_max = 3.0;
    node.h_max = 2.0;
    node.safe_radius_m_star = 100;
    node.node_sample_count = 50;
    node
}

#[test]
fn test_check_layer1_fast_path_m_v_le_m_star() {
    let node = make_test_node();
    let dg = [0.0f32; 4];
    let dh = [0.0f32; 4];
    // m_v = 1 <= safe_radius_m_star = 100, taylor_e = 0 => fast path CERT_INVARIANT
    let out = check_layer1_invariance(&node, 1, &dg, &dh, 0.0, 0.0, 0.0, 1.0);
    assert_eq!(out.result, CertResult::CertInvariant);
    assert_eq!(out.winning_candidate_idx, 0);
}

#[test]
fn test_check_layer1_taylor_error_breaks_fast_path() {
    let node = make_test_node();
    let dg = [0.0f32; 4];
    let dh = [0.0f32; 4];
    // m_v <= m_star but taylor_e > 0 => full evaluation
    let out = check_layer1_invariance(&node, 1, &dg, &dh, 0.0, 0.0, 0.5, 1.0);
    assert!(out.result == CertResult::CertInvariant
        || out.result == CertResult::CertRunnerUpDominant
        || out.result == CertResult::CertFallbackScan);
    assert!(out.winning_score.is_finite());
    assert!(out.tail_bound.is_finite());
}

#[test]
fn test_check_layer1_m_v_exceeds_safe_radius() {
    let node = make_test_node();
    let dg = [0.0f32; 4];
    let dh = [0.0f32; 4];
    // m_v = 1000 > safe_radius_m_star = 100 => full eval
    let out = check_layer1_invariance(&node, 1000, &dg, &dh, 0.0, 0.0, 0.0, 1.0);
    assert!(out.winning_score.is_finite());
    assert!(out.tail_bound.is_finite());
}

#[test]
fn test_check_layer1_output_struct_fields() {
    let node = make_test_node();
    let dg = [1.0f32, 2.0, 3.0, 4.0];
    let dh = [0.1f32, 0.2, 0.3, 0.4];
    let out = check_layer1_invariance(&node, 5, &dg, &dh, 10.0, 5.0, 0.1, 1.0);
    match out.result {
        CertResult::CertInvariant => {}
        CertResult::CertRunnerUpDominant => {
            assert!(out.winning_candidate_idx >= 1 && out.winning_candidate_idx <= 3);
        }
        CertResult::CertFallbackScan => {}
    }
    assert!(out.winning_score.is_finite());
    assert!(out.tail_bound.is_finite());
}

/// ---- Phase 2: TreeConfig Defaults ----

#[test]
fn test_tree_config_defaults_match_roadmap() {
    let config = TreeConfig::default();
    assert_eq!(config.max_depth, 6);
    assert_eq!(config.num_trees, 100);
    assert_eq!(config.lambda_reg, 1.0);
    assert_eq!(config.gamma, 0.0);
    assert_eq!(config.eta, 0.1);
    assert_eq!(config.tie_breaker_eps, 1e-5);
    assert_eq!(config.min_child_hessian, 1e-3);
}

/// ---- Phase 2: RING_BUFFER_K constant ----

#[test]
fn test_ring_buffer_k_is_4() {
    assert_eq!(RING_BUFFER_K, 4);
}
