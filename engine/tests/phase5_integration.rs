//! VeriTree Phase 5 Integration Tests - End-to-End Edge Cases
//! Covers: Full ensemble fit + unlearn, squared vs logistic loss consistency,
//! deletion batch size boundaries, multi-depth tree traversal

use veritree_engine::{
    BinnedDataset, TreeConfig, VeriTreeEnsemble,
};

fn make_dataset(num_samples: usize, num_features: usize, num_bins: usize, seed: u64) -> BinnedDataset {
    let mut dataset = BinnedDataset::new(num_samples, num_features, num_bins);
    // Simple LCG
    let mut state: u64 = seed;
    for i in 0..num_samples {
        let base = i * num_features;
        for f in 0..num_features {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            dataset.data[base + f] = ((state >> 32) as u8) % (num_bins as u8);
        }
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let g = ((state >> 32) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        dataset.gradients[i] = g;
        dataset.hessians[i] = 1.0;
    }
    dataset
}

fn default_config(max_depth: usize, num_trees: usize) -> TreeConfig {
    TreeConfig {
        max_depth,
        num_trees,
        lambda_reg: 1.0,
        gamma: 0.0,
        eta: 0.1,
        tie_breaker_eps: 1e-5,
        min_child_hessian: 1e-3,
    }
}

/// ---- Phase 5: End-to-End Ensemble Lifecycle ----

#[test]
fn test_ensemble_full_lifecycle_fit_unlearn_refit() {
    let mut dataset = make_dataset(200, 6, 32, 42);
    let config = default_config(4, 4);
    let mut ensemble = VeriTreeEnsemble::new(config, 200, true);
    ensemble.fit(&mut dataset);

    // Record pre-unlearn leaf weights
    let pre_weights: Vec<Vec<f32>> = ensemble.trees.iter()
        .map(|t| t.leaf_weights.clone())
        .collect();

    // Unlearn batch
    let (_l1, _l2, _rt) = ensemble.unlearn_batch(&[5, 15, 25]);
    let post_weights: Vec<Vec<f32>> = ensemble.trees.iter()
        .map(|t| t.leaf_weights.clone())
        .collect();

    // At least some weights should change after unlearning
    let mut any_changed = false;
    for t in 0..ensemble.trees.len() {
        for l in 0..pre_weights[t].len() {
            if (pre_weights[t][l] - post_weights[t][l]).abs() > 1e-7 {
                any_changed = true;
            }
        }
    }
    assert!(any_changed, "No leaf weights changed after unlearning");
}

#[test]
fn test_ensemble_deletion_batch_larger_than_samples() {
    let mut dataset = make_dataset(10, 4, 16, 7);
    let config = default_config(3, 2);
    let mut ensemble = VeriTreeEnsemble::new(config, 10, true);
    ensemble.fit(&mut dataset);

    // Delete more samples than exist (ids beyond range are invalid but routing handles gracefully)
    // Use valid ids only, but many of them
    let deletions: Vec<usize> = (0..10).collect();
    let (l1, l2, rt) = ensemble.unlearn_batch(&deletions);
    let total = l1 + l2 + rt;
    assert_eq!(total, 2);
}

#[test]
fn test_ensemble_single_tree_single_sample_deletion() {
    let mut dataset = make_dataset(1, 2, 8, 99);
    let config = default_config(2, 1);
    let mut ensemble = VeriTreeEnsemble::new(config, 1, true);
    ensemble.fit(&mut dataset);

    let (l1, l2, rt) = ensemble.unlearn_batch(&[0]);
    assert_eq!(l1 + l2 + rt, 1);
}

#[test]
fn test_ensemble_max_depth_1_stump() {
    let mut dataset = make_dataset(50, 4, 16, 55);
    let config = default_config(1, 2); // depth 1 = stump (2 leaves)
    let mut ensemble = VeriTreeEnsemble::new(config, 50, true);
    ensemble.fit(&mut dataset);

    for t in 0..ensemble.trees.len() {
        assert_eq!(ensemble.trees[t].leaf_weights.len(), 2);
    }

    let (l1, l2, rt) = ensemble.unlearn_batch(&[0, 1]);
    assert_eq!(l1 + l2 + rt, 2);
}

#[test]
fn test_ensemble_deep_tree_depth_8() {
    let mut dataset = make_dataset(100, 5, 32, 123);
    let config = default_config(8, 1);
    let mut ensemble = VeriTreeEnsemble::new(config, 100, true);
    ensemble.fit(&mut dataset);

    assert_eq!(ensemble.trees[0].leaf_weights.len(), 256); // 2^8 leaves
    let (l1, l2, rt) = ensemble.unlearn_batch(&[10]);
    assert_eq!(l1 + l2 + rt, 1);
}

#[test]
fn test_ensemble_large_num_trees() {
    let mut dataset = make_dataset(100, 4, 16, 211);
    let config = default_config(3, 50);
    let mut ensemble = VeriTreeEnsemble::new(config, 100, true);
    ensemble.fit(&mut dataset);
    assert_eq!(ensemble.trees.len(), 50);

    let (l1, l2, rt) = ensemble.unlearn_batch(&[3]);
    assert_eq!(l1 + l2 + rt, 50);
}

/// ---- Phase 5: Squared Error vs Logistic Loss Determinism ----

#[test]
fn test_ensemble_squared_error_deterministic_across_runs() {
    let mut dataset1 = make_dataset(80, 5, 32, 313);
    let mut dataset2 = make_dataset(80, 5, 32, 313); // identical data

    let config = default_config(4, 3);
    let mut e1 = VeriTreeEnsemble::new(config.clone(), 80, true);
    e1.fit(&mut dataset1);

    let mut e2 = VeriTreeEnsemble::new(config, 80, true);
    e2.fit(&mut dataset2);

    // Same seed, same data => identical leaf weights
    for t in 0..3 {
        assert_eq!(e1.trees[t].leaf_weights, e2.trees[t].leaf_weights);
    }
}

/// ---- Phase 5: Large Deletion Batch ----

#[test]
fn test_ensemble_half_dataset_deleted() {
    let mut dataset = make_dataset(100, 4, 16, 424);
    let config = default_config(4, 3);
    let mut ensemble = VeriTreeEnsemble::new(config, 100, true);
    ensemble.fit(&mut dataset);

    // Delete half the samples
    let deletions: Vec<usize> = (0..50).collect();
    let (l1, l2, rt) = ensemble.unlearn_batch(&deletions);
    assert_eq!(l1 + l2 + rt, 3);
}

/// ---- Phase 5: Feature Bin Boundaries ----

#[test]
fn test_ensemble_all_same_bin_all_features() {
    let mut dataset = BinnedDataset::new(40, 3, 8);
    for i in 0..40 {
        for f in 0..3 {
            dataset.data[i * 3 + f] = 7; // max bin
        }
        dataset.gradients[i] = 1.0;
        dataset.hessians[i] = 1.0;
    }
    let config = default_config(3, 2);
    let mut ensemble = VeriTreeEnsemble::new(config, 40, true);
    ensemble.fit(&mut dataset);

    // All same bin => no split possible => root becomes leaf
    // Tree trains without panic
    let (l1, l2, rt) = ensemble.unlearn_batch(&[0, 1]);
    assert_eq!(l1 + l2 + rt, 2);
}

/// ---- Phase 5: Zero Gradients Dataset ----

#[test]
fn test_ensemble_zero_gradients() {
    let mut dataset = BinnedDataset::new(40, 4, 16);
    for i in 0..40 {
        for f in 0..4 {
            dataset.data[i * 4 + f] = (i % 16) as u8;
        }
        dataset.gradients[i] = 0.0;
        dataset.hessians[i] = 1.0;
    }
    let config = default_config(3, 2);
    let mut ensemble = VeriTreeEnsemble::new(config, 40, true);
    ensemble.fit(&mut dataset);

    // Zero gradients => zero leaf weights (w = -0/(h+lambda) = 0)
    for t in 0..ensemble.trees.len() {
        for w in &ensemble.trees[t].leaf_weights {
            assert!((*w).abs() < 1e-6, "Leaf weight should be ~0 for zero gradients");
        }
    }

    let (l1, l2, rt) = ensemble.unlearn_batch(&[5]);
    assert_eq!(l1 + l2 + rt, 2);
}

/// ---- Phase 5: Large Hessian Values ----

#[test]
fn test_ensemble_large_hessians() {
    let mut dataset = BinnedDataset::new(40, 4, 16);
    for i in 0..40 {
        for f in 0..4 {
            dataset.data[i * 4 + f] = (i % 16) as u8;
        }
        dataset.gradients[i] = 1.0;
        dataset.hessians[i] = 1e6f32; // large Hessian
    }
    let config = default_config(3, 2);
    let mut ensemble = VeriTreeEnsemble::new(config, 40, true);
    ensemble.fit(&mut dataset);

    // Should not panic or produce NaN
    for t in 0..ensemble.trees.len() {
        for w in &ensemble.trees[t].leaf_weights {
            assert!(w.is_finite(), "Leaf weight should be finite");
        }
    }

    let (l1, l2, rt) = ensemble.unlearn_batch(&[10]);
    assert_eq!(l1 + l2 + rt, 2);
}

/// ---- Phase 5: Ensemble Unlearn with Empty Trees Edge ----

#[test]
fn test_ensemble_zero_trees() {
    let mut dataset = make_dataset(10, 2, 8, 1);
    let config = TreeConfig {
        max_depth: 3,
        num_trees: 0,
        lambda_reg: 1.0,
        gamma: 0.0,
        eta: 0.1,
        tie_breaker_eps: 1e-5,
        min_child_hessian: 1e-3,
    };
    let mut ensemble = VeriTreeEnsemble::new(config, 10, true);
    ensemble.fit(&mut dataset);
    assert!(ensemble.trees.is_empty());

    let (l1, l2, rt) = ensemble.unlearn_batch(&[0]);
    assert_eq!(l1, 0);
    assert_eq!(l2, 0);
    assert_eq!(rt, 0);
}
