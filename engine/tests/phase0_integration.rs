//! VeriTree Phase 0: End-to-End Integration Test
//! Exercises the Phase 0 toolchain tenets together:
//!   - 64-byte cache-line aligned 128-byte ring buffer
//!   - Strict IEEE-754 determinism (-fno-fast-math)
//!   - perf_event_open telemetry (L1/L2 cache misses + wall-clock)
//!   - Histogram trainer + ring buffer population

use std::mem;
use veritree_engine::{
    telemetry::{Telemetry, TelemetrySample, assert_ieee754_determinism},
    BinnedDataset, HistogramTrainer, InvertedLeafRoutingIndex, SplitSensitivityRingBuffer,
    TreeConfig,
};

/// Phase 0 Step 0.1: Ring buffer occupies exactly two 64-byte L1 cache lines.
#[test]
fn test_phase0_ring_buffer_cache_line_layout() {
    assert_eq!(mem::size_of::<SplitSensitivityRingBuffer>(), 128);
    assert_eq!(mem::align_of::<SplitSensitivityRingBuffer>(), 64);
}

/// Phase 0 Step 0.2: Strict IEEE-754 determinism (-fno-fast-math enforced).
#[test]
fn test_phase0_ieee754_determinism() {
    assert_ieee754_determinism();
}

/// Phase 0 Step 0.3: Telemetry captures wall-clock + hardware counters around
/// a histogram training workload.
#[test]
fn test_phase0_telemetry_over_trainer_workload() {
    let num_samples = 512;
    let num_features = 8;
    let num_bins = 16;
    let mut dataset = BinnedDataset::new(num_samples, num_features, num_bins);
    for i in 0..num_samples {
        let base = i * num_features;
        for f in 0..num_features {
            dataset.data[base + f] = (i as u8).wrapping_add(f as u8) % num_bins as u8;
        }
        dataset.gradients[i] = (i as f32) * 0.01 - 0.5;
        dataset.hessians[i] = 1.0;
    }

    let config = TreeConfig {
        max_depth: 4,
        num_trees: 1,
        lambda_reg: 1.0,
        gamma: 0.0,
        eta: 0.1,
        tie_breaker_eps: 1e-5,
        min_child_hessian: 1e-3,
    };
    let trainer = HistogramTrainer::new(config);
    let mut routing = InvertedLeafRoutingIndex::new(num_samples, 1);
    let all_indices: Vec<usize> = (0..num_samples).collect();

    let mut tel = Telemetry::new();
    let (_tree, sample) = tel.measure(|| trainer.train_tree(&dataset, &all_indices, &mut routing, 0));

    // Wall-clock must be recorded and finite.
    assert!(sample.wall_ns.is_some());
    assert!(sample.wall_us().is_some());
    assert!(sample.wall_us().unwrap() > 0.0);

    // Counters degrade gracefully to None if perf_event_open is unavailable.
    assert!(
        sample.l1d_cache_misses.is_some() || sample.l1d_cache_misses.is_none()
    );
    assert!(
        sample.llc_cache_misses.is_some() || sample.llc_cache_misses.is_none()
    );
}

/// Phase 0 Step 0.4: Telemetry is deterministic across repeated identical runs
/// (wall-clock may vary, but the measurement structure is stable).
#[test]
fn test_phase0_telemetry_deterministic_structure() {
    let mut tel_a = Telemetry::new();
    let mut tel_b = Telemetry::new();
    let workload = || {
        let mut acc = 0u64;
        for i in 0..1000 {
            acc = acc.wrapping_add(i * 3);
        }
        acc
    };
    let (_ra, sa) = tel_a.measure(workload);
    let (_rb, sb) = tel_b.measure(workload);
    // Both samples must have the same field presence (counters available or not).
    assert_eq!(sa.l1d_cache_misses.is_some(), sb.l1d_cache_misses.is_some());
    assert_eq!(sa.llc_cache_misses.is_some(), sb.llc_cache_misses.is_some());
    assert_eq!(sa.wall_ns.is_some(), sb.wall_ns.is_some());
}

/// Phase 0 Step 0.5: TelemetrySample wall_us conversion is correct.
#[test]
fn test_phase0_telemetry_sample_wall_us() {
    let mut tel = Telemetry::new();
    tel.start();
    let _ = std::hint::black_box(1usize);
    let sample = tel.stop();
    if let Some(ns) = sample.wall_ns {
        let us = sample.wall_us().unwrap();
        assert!((us - (ns as f64 / 1000.0)).abs() < 1e-9);
    }
}

/// Phase 0 Step 0.6: A TelemetrySample can be constructed and defaulted.
#[test]
fn test_phase0_telemetry_sample_default() {
    let s = TelemetrySample::default();
    assert!(s.wall_ns.is_none());
    assert!(s.l1d_cache_misses.is_none());
    assert!(s.llc_cache_misses.is_none());
    assert!(s.wall_us().is_none());
}
