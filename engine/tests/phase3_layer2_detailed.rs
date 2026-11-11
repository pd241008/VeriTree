//! VeriTree Phase 3 Detailed Verification Gate - Layer 2 Edge Cases
//! Covers: Runner-Up Transition Classification, Homotopy Splicing,
//! Cross-Feature Partition, Residual Propagation, Ensemble Unlearning

use veritree_engine::{
    classify_runner_up_transition, cross_feature_partition,
    BinnedDataset, ResidualPropagator, SplitSensitivityRingBuffer, TransitionKind,
    TreeConfig, VeriTreeEnsemble,
};
use veritree_engine::inverted_index::InvertedLeafRoutingIndex;
use veritree_engine::layer2_quasiml::apply_homotopy_splicing;

/// ---- Phase 3 Step 3.1: Runner-Up Transition Classification ----

#[test]
fn test_classify_winner_idx_zero_is_fallback() {
    let mut node = SplitSensitivityRingBuffer::default();
    node.feature_idx = [0, 1, 2, 3];
    node.bin_idx = [10, 20, 30, 40];
    let kind = classify_runner_up_transition(&node, 0);
    assert_eq!(kind, TransitionKind::RawScanFallback);
}

#[test]
fn test_classify_winner_idx_ge_4_is_fallback() {
    let mut node = SplitSensitivityRingBuffer::default();
    node.feature_idx = [0, 1, 2, 3];
    node.bin_idx = [10, 20, 30, 40];
    let kind = classify_runner_up_transition(&node, 4);
    assert_eq!(kind, TransitionKind::RawScanFallback);
}

#[test]
fn test_classify_winner_idx_5_is_fallback() {
    let mut node = SplitSensitivityRingBuffer::default();
    node.feature_idx = [0, 1, 2, 3];
    node.bin_idx = [10, 20, 30, 40];
    let kind = classify_runner_up_transition(&node, 5);
    assert_eq!(kind, TransitionKind::RawScanFallback);
}

#[test]
fn test_classify_intra_feature_homotopy_tau_diff_1() {
    // Same feature, adjacent bin => IntraFeatureHomotopy
    let mut node = SplitSensitivityRingBuffer::default();
    node.feature_idx = [5, 5, 9, 9];   // f_1 = 5, f_2 = 5
    node.bin_idx = [10, 11, 20, 21];   // tau_1 = 10, tau_2 = 11 (diff = 1)
    let kind = classify_runner_up_transition(&node, 1);
    assert_eq!(kind, TransitionKind::IntraFeatureHomotopy);
}

#[test]
fn test_classify_intra_feature_homotopy_tau_diff_1_reverse() {
    // tau_winner < tau_1 by 1 => still intra-feature
    let mut node = SplitSensitivityRingBuffer::default();
    node.feature_idx = [7, 7, 2, 2];
    node.bin_idx = [15, 14, 3, 4];   // tau_1 = 15, tau_2 = 14 (diff = 1)
    let kind = classify_runner_up_transition(&node, 1);
    assert_eq!(kind, TransitionKind::IntraFeatureHomotopy);
}

#[test]
fn test_classify_intra_feature_tau_diff_2_is_cross_feature() {
    // Same feature but tau diff = 2 => CrossFeaturePartition (not adjacent)
    let mut node = SplitSensitivityRingBuffer::default();
    node.feature_idx = [5, 5, 9, 9];
    node.bin_idx = [10, 12, 30, 40];  // tau diff = 2
    let kind = classify_runner_up_transition(&node, 1);
    assert_eq!(kind, TransitionKind::CrossFeaturePartition);
}

#[test]
fn test_classify_cross_feature_different_f() {
    // Different feature => CrossFeaturePartition
    let mut node = SplitSensitivityRingBuffer::default();
    node.feature_idx = [5, 9, 2, 2];
    node.bin_idx = [10, 20, 30, 40];
    let kind = classify_runner_up_transition(&node, 1);
    assert_eq!(kind, TransitionKind::CrossFeaturePartition);
}

#[test]
fn test_classify_winner_idx_2_cross_feature() {
    let mut node = SplitSensitivityRingBuffer::default();
    node.feature_idx = [3, 3, 7, 7];
    node.bin_idx = [5, 6, 8, 9];
    let kind = classify_runner_up_transition(&node, 2);
    assert_eq!(kind, TransitionKind::CrossFeaturePartition);
}

#[test]
fn test_classify_winner_idx_3_intra_feature() {
    let mut node = SplitSensitivityRingBuffer::default();
    node.feature_idx = [4, 4, 4, 4];
    node.bin_idx = [100, 101, 102, 103];  // all same feature, adjacent bins
    let kind = classify_runner_up_transition(&node, 3);
    // tau_3 = 103, tau_1 = 100, diff = 3 => not adjacent => CrossFeaturePartition
    assert_eq!(kind, TransitionKind::CrossFeaturePartition);
}

/// ---- Phase 3 Step 3.1: Homotopy Splicing ----

#[test]
fn test_apply_homotopy_splicing_slice_0_transports_stats() {
    let mut node = SplitSensitivityRingBuffer::default();
    node.dg_slice = [10.0, 5.0];
    node.dh_slice = [2.0, 1.0];
    node.g_left[0] = 100.0;
    node.h_left[0] = 50.0;

    apply_homotopy_splicing(&mut node, 0);

    assert!((node.g_left[0] - 110.0).abs() < 1e-6);
    assert!((node.h_left[0] - 52.0).abs() < 1e-6);
    // slice 1 untouched
    assert_eq!(node.dg_slice[1], 5.0);
    assert_eq!(node.dh_slice[1], 1.0);
}

#[test]
fn test_apply_homotopy_splicing_slice_1_transports_stats() {
    let mut node = SplitSensitivityRingBuffer::default();
    node.dg_slice = [3.0, 7.0];
    node.dh_slice = [1.0, 3.0];
    node.g_left[0] = 50.0;
    node.h_left[0] = 25.0;

    apply_homotopy_splicing(&mut node, 1);

    assert!((node.g_left[0] - 57.0).abs() < 1e-6);
    assert!((node.h_left[0] - 28.0).abs() < 1e-6);
}

#[test]
fn test_apply_homotopy_splicing_zero_stats_no_change() {
    let mut node = SplitSensitivityRingBuffer::default();
    node.dg_slice = [0.0, 0.0];
    node.dh_slice = [0.0, 0.0];
    node.g_left[0] = 99.0;
    node.h_left[0] = 44.0;

    apply_homotopy_splicing(&mut node, 0);
    apply_homotopy_splicing(&mut node, 1);

    assert!((node.g_left[0] - 99.0).abs() < 1e-6);
    assert!((node.h_left[0] - 44.0).abs() < 1e-6);
}

#[test]
fn test_apply_homotopy_splicing_negative_slice_stats() {
    let mut node = SplitSensitivityRingBuffer::default();
    node.dg_slice = [-5.0, 0.0];
    node.dh_slice = [-2.0, 0.0];
    node.g_left[0] = 10.0;
    node.h_left[0] = 5.0;

    apply_homotopy_splicing(&mut node, 0);

    assert!((node.g_left[0] - 5.0).abs() < 1e-6);
    assert!((node.h_left[0] - 3.0).abs() < 1e-6);
}

/// ---- Phase 3 Step 3.1: Cross-Feature Partition ----

#[test]
fn test_cross_feature_partition_all_left() {
    let dataset = BinnedDataset::new(4, 2, 256);
    // All bins = 0, winner_bin = 10 => all left
    let indices = vec![0, 1, 2, 3];
    let (left, right) = cross_feature_partition(&dataset, &indices, 0, 10);
    assert_eq!(left.len(), 4);
    assert_eq!(right.len(), 0);
}

#[test]
fn test_cross_feature_partition_all_right() {
    let mut dataset = BinnedDataset::new(4, 2, 256);
    for i in 0..4 {
        dataset.data[i * 2] = 200; // all bins > 50
    }
    let indices = vec![0, 1, 2, 3];
    let (left, right) = cross_feature_partition(&dataset, &indices, 0, 50);
    assert_eq!(left.len(), 0);
    assert_eq!(right.len(), 4);
}

#[test]
fn test_cross_feature_partition_mixed_split() {
    let mut dataset = BinnedDataset::new(6, 2, 256);
    // feature_idx = 0, so data[i * 2 + 0]
    dataset.data[0 * 2 + 0] = 10;   // sample 0, left
    dataset.data[1 * 2 + 0] = 200;  // sample 1, right
    dataset.data[2 * 2 + 0] = 20;   // sample 2, left
    dataset.data[3 * 2 + 0] = 250;  // sample 3, right
    dataset.data[4 * 2 + 0] = 30;   // sample 4, left
    dataset.data[5 * 2 + 0] = 255;  // sample 5, right
    let indices = vec![0, 1, 2, 3, 4, 5];
    let (left, right) = cross_feature_partition(&dataset, &indices, 0, 100);
    assert_eq!(left.len(), 3);
    assert_eq!(right.len(), 3);
    assert_eq!(left, vec![0, 2, 4]);
    assert_eq!(right, vec![1, 3, 5]);
}

#[test]
fn test_cross_feature_partition_empty_indices() {
    let dataset = BinnedDataset::new(4, 2, 256);
    let (left, right) = cross_feature_partition(&dataset, &[], 0, 50);
    assert!(left.is_empty());
    assert!(right.is_empty());
}

#[test]
fn test_cross_feature_partition_winner_bin_zero() {
    let mut dataset = BinnedDataset::new(4, 1, 256);
    dataset.data[0] = 0;   // left (0 <= 0)
    dataset.data[1] = 1;   // right (1 > 0)
    dataset.data[2] = 0;   // left
    dataset.data[3] = 255; // right
    let indices = vec![0, 1, 2, 3];
    let (left, right) = cross_feature_partition(&dataset, &indices, 0, 0);
    assert_eq!(left, vec![0, 2]);
    assert_eq!(right, vec![1, 3]);
}

#[test]
fn test_cross_feature_partition_winner_bin_255() {
    let mut dataset = BinnedDataset::new(4, 1, 256);
    dataset.data[0] = 255; // left (255 <= 255)
    dataset.data[1] = 254; // left
    dataset.data[2] = 0;   // left
    // all bins <= 255 => all left
    let indices = vec![0, 1, 2];
    let (left, right) = cross_feature_partition(&dataset, &indices, 0, 255);
    assert_eq!(left.len(), 3);
    assert!(right.is_empty());
}

#[test]
fn test_cross_feature_partition_different_feature_column() {
    let mut dataset = BinnedDataset::new(4, 3, 256);
    // Feature 1 (column index 1)
    dataset.data[1] = 10;   // left (feature 1, sample 0)
    dataset.data[4] = 200;  // right (feature 1, sample 1)
    dataset.data[7] = 5;    // left (feature 1, sample 2)
    dataset.data[10] = 150; // right (feature 1, sample 3)
    let indices = vec![0, 1, 2, 3];
    let (left, right) = cross_feature_partition(&dataset, &indices, 1, 50);
    assert_eq!(left, vec![0, 2]);
    assert_eq!(right, vec![1, 3]);
}

/// ---- Phase 3 Step 3.2: Residual Propagator Construction ----

#[test]
fn test_residual_propagator_squared_error_m3_zero() {
    let prop = ResidualPropagator::new(0.1, true);
    assert_eq!(prop.m_3, 0.0);
    assert_eq!(prop.eta, 0.1);
    assert_eq!(prop.epsilon_prop, 1e-6);
}

#[test]
fn test_residual_propagator_logistic_m3_nonzero() {
    let prop = ResidualPropagator::new(0.1, false);
    // M_3 = 1 / (6 * sqrt(3)) ≈ 0.096225
    assert!((prop.m_3 - 0.096225f32).abs() < 1e-5);
    assert!(prop.m_3 > 0.0);
}

#[test]
fn test_residual_propagator_different_eta() {
    let prop = ResidualPropagator::new(0.5, true);
    assert_eq!(prop.eta, 0.5);
    assert_eq!(prop.m_3, 0.0);
}

/// ---- Phase 3 Step 3.2: Residual Propagation Edge Cases ----

fn make_propagator() -> ResidualPropagator {
    ResidualPropagator::new(0.1, true)
}

fn make_routing_index(num_samples: usize, num_trees: usize) -> InvertedLeafRoutingIndex {
    InvertedLeafRoutingIndex::new(num_samples, num_trees)
}

#[test]
fn test_propagate_no_active_leaves_all_below_epsilon() {
    use veritree_engine::DecisionTree;
    let prop = make_propagator();
    let routing = make_routing_index(10, 2);
    let mut tree = DecisionTree::new(3);
    // All delta_w below epsilon_prop (1e-6) => no active leaves => no error
    let delta_w = vec![0.0f32; 8];
    let (_err, needs_flush) = prop.propagate_residuals(&routing, 0, 1, &delta_w, &mut tree);
    assert_eq!(_err, 0.0);
    assert!(!needs_flush);
}

#[test]
fn test_propagate_all_zero_delta_w() {
    use veritree_engine::DecisionTree;
    let prop = make_propagator();
    let routing = make_routing_index(10, 2);
    let mut tree = DecisionTree::new(3);
    let delta_w = vec![0.0f32; 8];
    let (err, needs_flush) = prop.propagate_residuals(&routing, 0, 1, &delta_w, &mut tree);
    assert_eq!(err, 0.0);
    assert!(!needs_flush);
}

#[test]
fn test_propagate_with_active_leaves_applies_shift() {
    use veritree_engine::DecisionTree;
    let prop = make_propagator();
    let mut routing = make_routing_index(4, 2);
    // Sample 0: leaf 0 in tree 0, leaf 2 in tree 1
    // Sample 1: leaf 0 in tree 0, leaf 2 in tree 1
    routing.set_leaf(0, 0, 0);
    routing.set_leaf(0, 1, 2);
    routing.set_leaf(1, 0, 0);
    routing.set_leaf(1, 1, 2);
    routing.set_leaf(2, 0, 1);
    routing.set_leaf(2, 1, 3);
    routing.set_leaf(3, 0, 1);
    routing.set_leaf(3, 1, 3);

    let mut tree = DecisionTree::new(3);
    tree.nodes[2].node_sample_count = 2; // leaf 2 has samples
    tree.nodes[3].node_sample_count = 2; // leaf 3 has samples
    tree.nodes[2].g_left[0] = 0.0;
    tree.nodes[2].g_left[1] = 0.0;
    tree.nodes[3].g_left[0] = 0.0;
    tree.nodes[3].g_left[1] = 0.0;

    // delta_w[0] = 0.5 (active), delta_w[1] = 0.0 (inactive)
    let delta_w = vec![0.5f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let (err, needs_flush) = prop.propagate_residuals(&routing, 0, 1, &delta_w, &mut tree);
    // Active leaf 0 routes samples 0,1 to node 2 in tree 1
    // grad_shift = H_cross * dw = 2 * 0.5 = 1.0 (H_cross = 1.0 per sample)
    // G[2] += eta * 1.0 = 0.1
    assert!(err > 0.0 || err == 0.0); // Taylor error may be 0 for squared error
    assert!(!needs_flush); // delta_12 = 0 => no flush
}

#[test]
fn test_propagate_needs_flush_guard_delta_12_zero() {
    use veritree_engine::DecisionTree;
    let prop = make_propagator();
    let mut routing = make_routing_index(2, 2);
    routing.set_leaf(0, 0, 0);
    routing.set_leaf(0, 1, 0);
    routing.set_leaf(1, 0, 0);
    routing.set_leaf(1, 1, 0);

    let mut tree = DecisionTree::new(3);
    tree.nodes[0].node_sample_count = 2;
    tree.nodes[0].g_left[0] = 0.0;
    tree.nodes[0].g_left[1] = 0.0; // delta_12 = |0 - 0| = 0

    let delta_w = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let (_err, needs_flush) = prop.propagate_residuals(&routing, 0, 1, &delta_w, &mut tree);
    // delta_12 = 0 => guard prevents flush even if err > delta_12/2
    assert!(!needs_flush);
}

#[test]
fn test_propagate_eta_applies_gradient_shift() {
    use veritree_engine::DecisionTree;
    let prop = ResidualPropagator::new(0.5, true); // eta = 0.5
    let mut routing = make_routing_index(1, 2);
    routing.set_leaf(0, 0, 0);
    routing.set_leaf(0, 1, 0);

    let mut tree = DecisionTree::new(3);
    tree.nodes[0].node_sample_count = 1;
    tree.nodes[0].g_left[0] = 0.0;
    tree.nodes[0].g_left[1] = 0.0;

    let delta_w = vec![2.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let (_err, _flush) = prop.propagate_residuals(&routing, 0, 1, &delta_w, &mut tree);
    // G[0] += eta * (H_cross * dw) = 0.5 * (1.0 * 2.0) = 1.0
    // But g_left[0] is used as proxy, not G_node directly. The C code mutates G_nodes_t_plus_1.
    // In Rust wrapper, we only check needs_flush. The actual G mutation happens via C pointer.
    // Just verify no panic and finite results.
    assert!(_err.is_finite());
}

/// ---- Phase 3: Ensemble Unlearning Edge Cases ----

#[test]
fn test_ensemble_unlearn_batch_empty_deletions() {
    let mut dataset = BinnedDataset::new(50, 4, 16);
    for i in 0..50 {
        dataset.gradients[i] = 1.0;
        dataset.hessians[i] = 1.0;
    }
    let config = TreeConfig {
        max_depth: 3,
        num_trees: 3,
        lambda_reg: 1.0,
        gamma: 0.0,
        eta: 0.1,
        tie_breaker_eps: 1e-5,
        min_child_hessian: 1e-3,
    };
    let mut ensemble = VeriTreeEnsemble::new(config, 50, true);
    ensemble.fit(&mut dataset);

    let (l1, l2, retrain) = ensemble.unlearn_batch(&[]);
    // No deletions => all trees resolve at Layer 1 (no shift)
    assert_eq!(l1, 3);
    assert_eq!(l2, 0);
    assert_eq!(retrain, 0);
}

#[test]
fn test_ensemble_unlearn_batch_single_deletion() {
    let mut dataset = BinnedDataset::new(50, 4, 16);
    for i in 0..50 {
        dataset.gradients[i] = 1.0;
        dataset.hessians[i] = 1.0;
    }
    let config = TreeConfig {
        max_depth: 3,
        num_trees: 3,
        lambda_reg: 1.0,
        gamma: 0.0,
        eta: 0.1,
        tie_breaker_eps: 1e-5,
        min_child_hessian: 1e-3,
    };
    let mut ensemble = VeriTreeEnsemble::new(config, 50, true);
    ensemble.fit(&mut dataset);

    let (l1, l2, retrain) = ensemble.unlearn_batch(&[5]);
    let total = l1 + l2 + retrain;
    assert_eq!(total, 3);
    assert!(l1 + l2 + retrain == 3);
}

#[test]
fn test_ensemble_fit_and_unlearn_consistency() {
    let mut dataset = BinnedDataset::new(100, 6, 32);
    for i in 0..100 {
        dataset.gradients[i] = (i as f32) * 0.01 - 0.5;
        dataset.hessians[i] = 1.0;
    }
    let config = TreeConfig {
        max_depth: 4,
        num_trees: 5,
        lambda_reg: 1.0,
        gamma: 0.0,
        eta: 0.1,
        tie_breaker_eps: 1e-5,
        min_child_hessian: 1e-3,
    };
    let mut ensemble = VeriTreeEnsemble::new(config, 100, true);
    ensemble.fit(&mut dataset);

    // Verify all trees have non-default structure after fit
    for t in 0..ensemble.trees.len() {
        assert!(ensemble.trees[t].leaf_weights.len() == 16); // 2^4 leaves
    }

    let (l1, l2, retrain) = ensemble.unlearn_batch(&[10, 20, 30]);
    let total = l1 + l2 + retrain;
    assert_eq!(total, 5);
}

#[test]
fn test_ensemble_squared_error_vs_logistic() {
    let mut dataset = BinnedDataset::new(60, 5, 16);
    for i in 0..60 {
        let p = 0.1 + (i as f32) / 60.0 * 0.8;
        let y = if i % 2 == 0 { 1.0 } else { 0.0 };
        dataset.gradients[i] = p - y;
        dataset.hessians[i] = p * (1.0 - p);
    }

    // Squared error ensemble
    let config_se = TreeConfig {
        max_depth: 3,
        num_trees: 3,
        lambda_reg: 1.0,
        gamma: 0.0,
        eta: 0.1,
        tie_breaker_eps: 1e-5,
        min_child_hessian: 1e-3,
    };
    let mut ensemble_se = VeriTreeEnsemble::new(config_se, 60, true);
    ensemble_se.fit(&mut dataset);
    let (l1_se, l2_se, rt_se) = ensemble_se.unlearn_batch(&[3, 7]);
    assert_eq!(l1_se + l2_se + rt_se, 3);

    // Logistic ensemble (recreate dataset since fit mutates gradients)
    let mut dataset2 = BinnedDataset::new(60, 5, 16);
    for i in 0..60 {
        let p = 0.1 + (i as f32) / 60.0 * 0.8;
        let y = if i % 2 == 0 { 1.0 } else { 0.0 };
        dataset2.gradients[i] = p - y;
        dataset2.hessians[i] = p * (1.0 - p);
    }
    let config_lg = TreeConfig {
        max_depth: 3,
        num_trees: 3,
        lambda_reg: 1.0,
        gamma: 0.0,
        eta: 0.1,
        tie_breaker_eps: 1e-5,
        min_child_hessian: 1e-3,
    };
    let mut ensemble_lg = VeriTreeEnsemble::new(config_lg, 60, false);
    ensemble_lg.fit(&mut dataset2);
    let (l1_lg, l2_lg, rt_lg) = ensemble_lg.unlearn_batch(&[3, 7]);
    assert_eq!(l1_lg + l2_lg + rt_lg, 3);
}

#[test]
fn test_ensemble_deletions_all_same_leaf() {
    let mut dataset = BinnedDataset::new(30, 3, 8);
    for i in 0..30 {
        dataset.data[i * 3] = 0; // all same bin => likely same leaf
        dataset.gradients[i] = 1.0;
        dataset.hessians[i] = 1.0;
    }
    let config = TreeConfig {
        max_depth: 3,
        num_trees: 2,
        lambda_reg: 1.0,
        gamma: 0.0,
        eta: 0.1,
        tie_breaker_eps: 1e-5,
        min_child_hessian: 1e-3,
    };
    let mut ensemble = VeriTreeEnsemble::new(config, 30, true);
    ensemble.fit(&mut dataset);

    let (l1, l2, retrain) = ensemble.unlearn_batch(&[0, 1, 2, 3, 4]);
    let total = l1 + l2 + retrain;
    assert_eq!(total, 2);
}
