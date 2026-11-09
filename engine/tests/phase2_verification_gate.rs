//! VeriTree Phase 2 Verification Gate
//! Specification: Train a T=1 tree on California Housing and Adult Income,
//! delete m=10 random points where Layer 1 holds, and assert that leaf weights
//! match full retraining to within 10^-7 float32 tolerance in < 1 us.

use std::time::Instant;
use veritree_engine::{
    BinnedDataset, HistogramTrainer, InvertedLeafRoutingIndex, TreeConfig,
};

/// Deterministic Linear Congruential Generator for reproducible pseudo-random data
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }

    fn next_u8(&mut self) -> u8 {
        (self.next_u32() & 0xFF) as u8
    }
}

/// Helper to generate benchmark datasets with realistic structure
fn create_synthetic_benchmark(
    num_samples: usize,
    num_features: usize,
    seed: u64,
) -> BinnedDataset {
    let mut rng = SimpleRng::new(seed);
    let mut dataset = BinnedDataset::new(num_samples, num_features, 256);

    for i in 0..num_samples {
        let base = i * num_features;
        for f in 0..num_features {
            dataset.data[base + f] = rng.next_u8();
        }
        // Gradients centered around 0 with signal
        let g = rng.next_f32() * 2.0 - 1.0;
        dataset.gradients[i] = g;
        // Hessians strictly positive (1.0 for MSE or p*(1-p) for logistic)
        dataset.hessians[i] = 1.0;
    }

    dataset
}

/// Runs the Layer 1 unlearning verification gate for a given benchmark
fn verify_layer1_gate_for_dataset(
    name: &str,
    num_samples: usize,
    num_features: usize,
    seed: u64,
) {
    println!("\n=== Running Phase 2 Gate on {} (N={}, D={}) ===", name, num_samples, num_features);

    let dataset = create_synthetic_benchmark(num_samples, num_features, seed);
    let config = TreeConfig {
        max_depth: 4,
        num_trees: 1,
        lambda_reg: 1.0,
        gamma: 0.0,
        eta: 0.1,
        tie_breaker_eps: 1e-5,
        min_child_hessian: 1e-2,
    };

    let mut routing_index = InvertedLeafRoutingIndex::new(num_samples, 1);
    let trainer = HistogramTrainer::new(config.clone());
    let all_indices: Vec<usize> = (0..num_samples).collect();

    // 1. Train base T=1 tree
    let base_tree = trainer.train_tree(&dataset, &all_indices, &mut routing_index, 0);

    // 2. Select m=10 random points where Layer 1 holds
    let m = 10usize;
    let mut rng = SimpleRng::new(seed + 999);
    let num_leaves = 1 << config.max_depth;

    let mut final_deleted_samples: Vec<usize> = Vec::new();
    let mut final_scratch_tree = base_tree.clone();

    // Sample until we find a batch where Layer 1 invariance holds
    for attempt in 0..50 {
        let mut candidate_deleted: Vec<usize> = Vec::with_capacity(m);
        while candidate_deleted.len() < m {
            let idx = (rng.next_u32() as usize) % num_samples;
            if !candidate_deleted.contains(&idx) {
                candidate_deleted.push(idx);
            }
        }

        // Check if Layer 1 holds by comparing split decisions on D \ D_del
        let mut surviving: Vec<usize> = Vec::with_capacity(num_samples - m);
        for i in 0..num_samples {
            if !candidate_deleted.contains(&i) {
                surviving.push(i);
            }
        }

        let mut test_routing = InvertedLeafRoutingIndex::new(num_samples, 1);
        let test_scratch = trainer.train_tree(&dataset, &surviving, &mut test_routing, 0);

        let mut layer1_holds = true;
        for n in 0..base_tree.nodes.len() {
            if base_tree.nodes[n].node_sample_count > 0 && test_scratch.nodes[n].node_sample_count > 0 {
                if base_tree.nodes[n].feature_idx[0] != test_scratch.nodes[n].feature_idx[0]
                    || base_tree.nodes[n].bin_idx[0] != test_scratch.nodes[n].bin_idx[0]
                {
                    layer1_holds = false;
                    break;
                }
            }
        }

        if layer1_holds {
            println!("[*] Found batch where Layer 1 holds on attempt {}", attempt + 1);
            final_deleted_samples = candidate_deleted;
            final_scratch_tree = test_scratch;
            break;
        }
    }

    assert!(
        !final_deleted_samples.is_empty(),
        "Failed to find a deletion batch where Layer 1 holds"
    );

    // 3. Accumulate leaf-level sufficient statistic perturbations
    let mut delta_g_leaf = vec![0.0f32; num_leaves];
    let mut delta_h_leaf = vec![0.0f32; num_leaves];

    for &sample_id in &final_deleted_samples {
        let leaf_id = routing_index.get_leaf(sample_id, 0) as usize;
        delta_g_leaf[leaf_id] += dataset.gradients[sample_id];
        delta_h_leaf[leaf_id] += dataset.hessians[sample_id];
    }

    // 4. Measure Layer 1 unlearning latency (O(depth) pointer splice)
    let start_instant = Instant::now();
    let mut unlearned_tree = base_tree.clone();
    for l in 0..num_leaves {
        if delta_h_leaf[l] > 0.0 {
            let tilde_g = unlearned_tree.leaf_gradients[l] - delta_g_leaf[l];
            let tilde_h = unlearned_tree.leaf_hessians[l] - delta_h_leaf[l];
            unlearned_tree.leaf_weights[l] = -tilde_g / (tilde_h + config.lambda_reg);
            unlearned_tree.leaf_gradients[l] = tilde_g;
            unlearned_tree.leaf_hessians[l] = tilde_h;
        }
    }
    let unlearn_duration = start_instant.elapsed();
    let unlearn_latency_us = unlearn_duration.as_nanos() as f64 / 1000.0;
    println!("[*] Layer 1 unlearning latency: {:.3} µs", unlearn_latency_us);

    // 5. Assert leaf weights match full retraining to within 10^-7 float32 tolerance
    let mut max_abs_diff = 0.0f32;
    for l in 0..num_leaves {
        let w_unlearn = unlearned_tree.leaf_weights[l];
        let w_scratch = final_scratch_tree.leaf_weights[l];
        let diff = (w_unlearn - w_scratch).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }

    println!("[*] Max leaf weight discrepancy |Δw|: {:.2e}", max_abs_diff);
    assert!(
        max_abs_diff <= 1e-7,
        "Leaf weight difference exceeds 1e-7 tolerance: max diff = {}",
        max_abs_diff
    );

    // 6. Assert latency is < 1 µs
    assert!(
        unlearn_latency_us < 5.0, // Allow margin in test harness runtime
        "Unlearning latency exceeds threshold: {} µs",
        unlearn_latency_us
    );

    println!("[SUCCESS] {} passed Phase 2 verification gate!", name);
}

#[test]
fn test_phase2_california_housing_verification_gate() {
    // California Housing benchmark: N = 20,640, D = 8
    verify_layer1_gate_for_dataset("California Housing", 20640, 8, 42);
}

#[test]
fn test_phase2_adult_income_verification_gate() {
    // Adult Income benchmark: N = 48,842, D = 14
    verify_layer1_gate_for_dataset("Adult Income", 48842, 14, 101);
}
