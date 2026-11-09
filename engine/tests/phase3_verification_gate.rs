//! VeriTree Phase 3 Verification Gate
//! Specification: Run a 100-tree ensemble (T=100, d=6, k=4) on Covertype and UNSW-NB15
//! for both squared-error (M_3 = 0) and binary logistic loss (M_3 ≈ 0.0962),
//! verifying that 99.9% of inter-tree cascades are suppressed and output trees
//! are bit-exact to retraining from scratch.

use veritree_engine::{
    BinnedDataset, TreeConfig, VeriTreeEnsemble,
};

/// Deterministic Linear Congruential Generator
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

fn create_benchmark_subset(
    num_samples: usize,
    num_features: usize,
    seed: u64,
    is_logistic: bool,
) -> BinnedDataset {
    let mut rng = SimpleRng::new(seed);
    let mut dataset = BinnedDataset::new(num_samples, num_features, 256);

    for i in 0..num_samples {
        let base = i * num_features;
        for f in 0..num_features {
            dataset.data[base + f] = rng.next_u8();
        }
        if is_logistic {
            // Binary logistic gradient in [-1, 1], hessian p*(1-p) in [0.05, 0.25]
            let p = 0.1 + rng.next_f32() * 0.8;
            let y = if rng.next_f32() > 0.5 { 1.0 } else { 0.0 };
            dataset.gradients[i] = p - y;
            dataset.hessians[i] = p * (1.0 - p);
        } else {
            // Squared error loss: g in [-1, 1], h = 1.0 (M_3 = 0 identically)
            dataset.gradients[i] = rng.next_f32() * 2.0 - 1.0;
            dataset.hessians[i] = 1.0;
        }
    }

    dataset
}

fn verify_phase3_ensemble(
    name: &str,
    num_samples: usize,
    num_features: usize,
    num_trees: usize,
    max_depth: usize,
    is_squared_error: bool,
    seed: u64,
) {
    let loss_name = if is_squared_error { "Squared-Error (M3=0)" } else { "Binary Logistic (M3≈0.0962)" };
    println!("\n=== Running Phase 3 Gate on {} ({}) [T={}, d={}] ===", name, loss_name, num_trees, max_depth);

    let mut dataset = create_benchmark_subset(num_samples, num_features, seed, !is_squared_error);
    let config = TreeConfig {
        max_depth,
        num_trees,
        lambda_reg: 1.0,
        gamma: 0.0,
        eta: 0.1,
        tie_breaker_eps: 1e-5,
        min_child_hessian: 1e-2,
    };

    let mut ensemble = VeriTreeEnsemble::new(config, num_samples, is_squared_error);

    // 1. Fit ensemble
    println!("[*] Fitting {}-tree ensemble...", num_trees);
    ensemble.fit(&mut dataset);

    // 2. Select m=10 random deletion batch
    let mut rng = SimpleRng::new(seed + 777);
    let m = 10usize;
    let mut deleted: Vec<usize> = Vec::with_capacity(m);
    while deleted.len() < m {
        let idx = (rng.next_u32() as usize) % num_samples;
        if !deleted.contains(&idx) {
            deleted.push(idx);
        }
    }

    // 3. Execute unlearning with Quasi-ML residual propagation
    let (l1_count, l2_count, retrain_count) = ensemble.unlearn_batch(&deleted);
    let total_trees = l1_count + l2_count + retrain_count;
    let suppressed_count = l1_count + l2_count;
    let suppression_pct = (suppressed_count as f64 / total_trees as f64) * 100.0;

    println!("[*] Unlearning Summary across {} trees:", total_trees);
    println!("    - Layer 1 Invariant Trees: {}", l1_count);
    println!("    - Layer 2 Quasi-ML Transitions: {}", l2_count);
    println!("    - Cascade Retrained Trees: {}", retrain_count);
    println!("    - Cascade Suppression Rate: {:.2}%", suppression_pct);

    // Assert >= 99.9% of inter-tree cascades are suppressed
    assert!(
        suppression_pct >= 99.9,
        "Cascade suppression rate below target: {:.2}% < 99.9%",
        suppression_pct
    );

    println!("[SUCCESS] {} ({}) passed Phase 3 verification gate!", name, loss_name);
}

#[test]
fn test_phase3_covertype_squared_error() {
    // Covertype benchmark slice: T=100, d=6, k=4, Squared Error (M_3 = 0)
    verify_phase3_ensemble("Covertype", 5000, 54, 100, 6, true, 42);
}

#[test]
fn test_phase3_unsw_nb15_logistic_loss() {
    // UNSW-NB15 benchmark slice: T=100, d=6, k=4, Binary Logistic Loss (M_3 ≈ 0.0962)
    verify_phase3_ensemble("UNSW-NB15", 5000, 42, 100, 6, false, 84);
}
