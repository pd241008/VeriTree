//! VeriTree Deterministic Histogram GBDT Trainer & Ring Buffer Population
//! Implements strict IEEE-754 deterministic split finding, Margin-Regularized Tie-Breaker,
//! and 128-byte cache-line ring buffer construction.

use std::mem;

pub const RING_BUFFER_K: usize = 4;

/// 128-byte cache-aligned node layout: packs exactly into two 64-byte L1 cache lines
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct SplitSensitivityRingBuffer {
    // Cache Line 0 (64 bytes): Top-4 candidate splits + slice statistics
    pub feature_idx: [u16; 4],     // 8B: Candidate feature indices f_1..f_4
    pub bin_idx: [u16; 4],         // 8B: Candidate threshold bins \tau_1..\tau_4
    pub g_left: [f32; 4],          // 16B: Left child gradient sums G_{L,1..4}
    pub h_left: [f32; 4],          // 16B: Left child Hessian sums H_{L,1..4}
    pub dg_slice: [f32; 2],        // 8B: Adjacent bin slice \Delta G_{slice} (left/right)
    pub dh_slice: [f32; 2],        // 8B: Adjacent bin slice \Delta H_{slice} (left/right)

    // Cache Line 1 (64 bytes): Parent totals, tail bounds & safe radius
    pub g_node: f32,               // 4B: Total node gradient G_v
    pub h_node: f32,               // 4B: Total node Hessian H_v
    pub phi_tail: f32,             // 4B: Uncached tail ceiling \Phi_{tail}
    pub h_min_tail: f32,           // 4B: Minimum Hessian across uncached bins
    pub g_abs_max: f32,            // 4B: |G|_{max} across uncached bins
    pub g_max: f32,                // 4B: Max single-sample |g_i| in I_v
    pub h_max: f32,                // 4B: Max single-sample h_i in I_v
    pub safe_radius_m_star: u32,   // 4B: Pre-certified safe deletion radius m_v^*
    pub left_child_id: i32,        // 4B: Pointer/index to left subtree
    pub right_child_id: i32,       // 4B: Pointer/index to right subtree
    pub node_sample_count: u32,    // 4B: |I_v|
    pub reserved: [u8; 20],        // 20B: Padding to exact 128-byte boundary
}

// Compile-time assertion for 128-byte boundary and 64-byte alignment
const _: () = {
    assert!(mem::size_of::<SplitSensitivityRingBuffer>() == 128);
    assert!(mem::align_of::<SplitSensitivityRingBuffer>() == 64);
};

impl Default for SplitSensitivityRingBuffer {
    fn default() -> Self {
        Self {
            feature_idx: [0; 4],
            bin_idx: [0; 4],
            g_left: [0.0; 4],
            h_left: [0.0; 4],
            dg_slice: [0.0; 2],
            dh_slice: [0.0; 2],
            g_node: 0.0,
            h_node: 0.0,
            phi_tail: 0.0,
            h_min_tail: 1e6,
            g_abs_max: 0.0,
            g_max: 0.0,
            h_max: 0.0,
            safe_radius_m_star: 0,
            left_child_id: -1,
            right_child_id: -1,
            node_sample_count: 0,
            reserved: [0u8; 20],
        }
    }
}

/// GBDT Tree Hyperparameters & Execution Config
#[derive(Clone, Debug)]
pub struct TreeConfig {
    pub max_depth: usize,
    pub num_trees: usize,
    pub lambda_reg: f32,
    pub gamma: f32,
    pub eta: f32,
    pub tie_breaker_eps: f32,
    pub min_child_hessian: f32,
}

impl Default for TreeConfig {
    fn default() -> Self {
        Self {
            max_depth: 6,
            num_trees: 100,
            lambda_reg: 1.0,
            gamma: 0.0,
            eta: 0.1,
            tie_breaker_eps: 1e-5,
            min_child_hessian: 1e-3,
        }
    }
}

/// A binned dataset for histogram-based training
#[derive(Clone, Debug)]
pub struct BinnedDataset {
    pub num_samples: usize,
    pub num_features: usize,
    pub num_bins: usize,
    /// Row-major uint8_t matrix: shape [num_samples, num_features]
    pub data: Vec<u8>,
    pub gradients: Vec<f32>,
    pub hessians: Vec<f32>,
}

impl BinnedDataset {
    pub fn new(num_samples: usize, num_features: usize, num_bins: usize) -> Self {
        Self {
            num_samples,
            num_features,
            num_bins,
            data: vec![0u8; num_samples * num_features],
            gradients: vec![0.0f32; num_samples],
            hessians: vec![1.0f32; num_samples],
        }
    }

    #[inline(always)]
    pub fn get_bin(&self, sample_idx: usize, feature_idx: usize) -> u8 {
        self.data[sample_idx * self.num_features + feature_idx]
    }
}

/// Candidate split evaluated during histogram scan
#[derive(Clone, Copy, Debug)]
pub struct CandidateSplit {
    pub feature_idx: u16,
    pub bin_idx: u16,
    pub g_left: f32,
    pub h_left: f32,
    pub phi: f32,
    pub gain: f32,
}

/// A trained decision tree storing nodes in cache-aligned array format
#[derive(Clone, Debug)]
pub struct DecisionTree {
    pub nodes: Vec<SplitSensitivityRingBuffer>,
    pub leaf_weights: Vec<f32>,
    pub leaf_gradients: Vec<f32>,
    pub leaf_hessians: Vec<f32>,
    pub max_depth: usize,
}

impl DecisionTree {
    pub fn new(max_depth: usize) -> Self {
        let max_internal_nodes = (1 << max_depth) - 1;
        let num_leaves = 1 << max_depth;
        Self {
            nodes: vec![SplitSensitivityRingBuffer::default(); max_internal_nodes],
            leaf_weights: vec![0.0; num_leaves],
            leaf_gradients: vec![0.0; num_leaves],
            leaf_hessians: vec![0.0; num_leaves],
            max_depth,
        }
    }

    /// Evaluates tree prediction for a single sample
    pub fn predict(&self, sample: &[u8]) -> f32 {
        let mut node_idx = 0;
        for _ in 0..self.max_depth {
            let node = &self.nodes[node_idx];
            let f = node.feature_idx[0] as usize;
            let tau = node.bin_idx[0] as u8;
            if sample[f] <= tau {
                let next = node.left_child_id;
                if next < 0 {
                    let leaf_id = (-next - 1) as usize;
                    return self.leaf_weights[leaf_id];
                }
                node_idx = next as usize;
            } else {
                let next = node.right_child_id;
                if next < 0 {
                    let leaf_id = (-next - 1) as usize;
                    return self.leaf_weights[leaf_id];
                }
                node_idx = next as usize;
            }
        }
        0.0
    }
}

/// Helper: regularized split score psi(G, H) = G^2 / (H + lambda)
#[inline(always)]
pub fn psi(g: f32, h: f32, lambda_reg: f32) -> f32 {
    (g * g) / (h + lambda_reg)
}

/// Margin-Regularized Split Tie-Breaker (Layer 3 Defense):
/// If two candidate splits have gain within eps_tie, select the one
/// with the larger second-order Hessian stability: (H_L * H_R) / H_v.
#[inline]
pub fn margin_regularized_tie_breaker(
    gain_a: f32,
    gain_b: f32,
    hl_a: f32,
    hr_a: f32,
    hl_b: f32,
    hr_b: f32,
    h_v: f32,
    eps_tie: f32,
) -> bool {
    let diff = (gain_a - gain_b).abs();
    if diff <= eps_tie && h_v > 0.0 {
        let stability_a = (hl_a * hr_a) / h_v;
        let stability_b = (hl_b * hr_b) / h_v;
        stability_a >= stability_b
    } else {
        gain_a >= gain_b
    }
}

/// Deterministic Histogram GBDT Trainer
pub struct HistogramTrainer {
    pub config: TreeConfig,
}

impl HistogramTrainer {
    pub fn new(config: TreeConfig) -> Self {
        Self { config }
    }

    /// Train a single deterministic tree on sample subset
    pub fn train_tree(
        &self,
        dataset: &BinnedDataset,
        sample_indices: &[usize],
        routing_index: &mut crate::inverted_index::InvertedLeafRoutingIndex,
        tree_idx: usize,
    ) -> DecisionTree {
        let mut tree = DecisionTree::new(self.config.max_depth);
        let mut next_leaf_id = 0usize;

        self.build_subtree(
            dataset,
            sample_indices,
            0,
            0,
            &mut tree,
            &mut next_leaf_id,
            routing_index,
            tree_idx,
        );

        tree
    }

    fn build_subtree(
        &self,
        dataset: &BinnedDataset,
        sample_indices: &[usize],
        node_idx: usize,
        depth: usize,
        tree: &mut DecisionTree,
        next_leaf_id: &mut usize,
        routing_index: &mut crate::inverted_index::InvertedLeafRoutingIndex,
        tree_idx: usize,
    ) {
        let num_samples = sample_indices.len();
        if num_samples == 0 {
            return;
        }

        // Node totals and envelopes
        let mut g_node = 0.0f32;
        let mut h_node = 0.0f32;
        let mut g_max = 0.0f32;
        let mut h_max = 0.0f32;

        for &i in sample_indices {
            let g = dataset.gradients[i];
            let h = dataset.hessians[i];
            g_node += g;
            h_node += h;
            let abs_g = g.abs();
            if abs_g > g_max {
                g_max = abs_g;
            }
            if h > h_max {
                h_max = h;
            }
        }

        // Terminal condition: max depth reached
        if depth >= self.config.max_depth {
            let leaf_id = *next_leaf_id;
            *next_leaf_id += 1;

            let weight = -g_node / (h_node + self.config.lambda_reg);
            tree.leaf_weights[leaf_id] = weight;
            tree.leaf_gradients[leaf_id] = g_node;
            tree.leaf_hessians[leaf_id] = h_node;

            // Route all samples to this terminal leaf
            for &i in sample_indices {
                routing_index.set_leaf(i, tree_idx, leaf_id as u8);
            }
            return;
        }

        // Build 2D histogram across features and bins
        let num_feats = dataset.num_features;
        let num_bins = dataset.num_bins;
        let mut hist_g = vec![vec![0.0f32; num_bins]; num_feats];
        let mut hist_h = vec![vec![0.0f32; num_bins]; num_feats];

        for &i in sample_indices {
            let g = dataset.gradients[i];
            let h = dataset.hessians[i];
            let base = i * num_feats;
            for f in 0..num_feats {
                let bin = dataset.data[base + f] as usize;
                hist_g[f][bin] += g;
                hist_h[f][bin] += h;
            }
        }

        // Scan candidate splits across all (features x bins)
        let mut all_candidates: Vec<CandidateSplit> = Vec::with_capacity(num_feats * num_bins);

        for f in 0..num_feats {
            let mut g_left = 0.0f32;
            let mut h_left = 0.0f32;

            for b in 0..(num_bins - 1) {
                g_left += hist_g[f][b];
                h_left += hist_h[f][b];

                let g_right = g_node - g_left;
                let h_right = h_node - h_left;

                if h_left < self.config.min_child_hessian || h_right < self.config.min_child_hessian {
                    continue;
                }

                let phi = psi(g_left, h_left, self.config.lambda_reg)
                    + psi(g_right, h_right, self.config.lambda_reg);
                let gain = 0.5 * phi
                    - 0.5 * psi(g_node, h_node, self.config.lambda_reg)
                    - self.config.gamma;

                if gain > 0.0 {
                    all_candidates.push(CandidateSplit {
                        feature_idx: f as u16,
                        bin_idx: b as u16,
                        g_left,
                        h_left,
                        phi,
                        gain,
                    });
                }
            }
        }

        // If no valid split found, turn node into leaf
        if all_candidates.is_empty() {
            let leaf_id = *next_leaf_id;
            *next_leaf_id += 1;

            let weight = -g_node / (h_node + self.config.lambda_reg);
            tree.leaf_weights[leaf_id] = weight;
            tree.leaf_gradients[leaf_id] = g_node;
            tree.leaf_hessians[leaf_id] = h_node;

            for &i in sample_indices {
                routing_index.set_leaf(i, tree_idx, leaf_id as u8);
            }
            return;
        }

        // Sort candidates descending by gain with Margin-Regularized Tie-Breaker
        // Quantize gain into eps_tie buckets to guarantee a mathematically strict total order
        let eps_tie = self.config.tie_breaker_eps.max(1e-7);
        all_candidates.sort_by(|a, b| {
            let bucket_a = (a.gain / eps_tie).floor() as i64;
            let bucket_b = (b.gain / eps_tie).floor() as i64;

            bucket_b.cmp(&bucket_a)
                .then_with(|| {
                    if h_node > 0.0 {
                        let hr_a = h_node - a.h_left;
                        let hr_b = h_node - b.h_left;
                        let stab_a = (a.h_left * hr_a) / h_node;
                        let stab_b = (b.h_left * hr_b) / h_node;
                        stab_b.total_cmp(&stab_a)
                    } else {
                        std::cmp::Ordering::Equal
                    }
                })
                .then_with(|| b.gain.total_cmp(&a.gain))
                .then_with(|| a.feature_idx.cmp(&b.feature_idx))
                .then_with(|| a.bin_idx.cmp(&b.bin_idx))
        });

        // Pack top-4 candidates into Ring Buffer
        let mut node = SplitSensitivityRingBuffer::default();
        node.g_node = g_node;
        node.h_node = h_node;
        node.g_max = g_max;
        node.h_max = h_max;
        node.node_sample_count = num_samples as u32;

        let k = std::cmp::min(RING_BUFFER_K, all_candidates.len());
        for r in 0..k {
            let cand = &all_candidates[r];
            node.feature_idx[r] = cand.feature_idx;
            node.bin_idx[r] = cand.bin_idx;
            node.g_left[r] = cand.g_left;
            node.h_left[r] = cand.h_left;
        }

        // Tail envelope across all splits outside top-k
        let mut phi_tail = 0.0f32;
        let mut h_min_tail = 1e6f32;
        let mut g_abs_max = 0.0f32;

        if all_candidates.len() > k {
            for cand in &all_candidates[k..] {
                if cand.phi > phi_tail {
                    phi_tail = cand.phi;
                }
                let h_r = h_node - cand.h_left;
                let min_h = cand.h_left.min(h_r);
                if min_h < h_min_tail {
                    h_min_tail = min_h;
                }
                let g_r = g_node - cand.g_left;
                let max_g = cand.g_left.abs().max(g_r.abs());
                if max_g > g_abs_max {
                    g_abs_max = max_g;
                }
            }
        }
        node.phi_tail = phi_tail;
        node.h_min_tail = h_min_tail;
        node.g_abs_max = g_abs_max;

        // Compute certified safe deletion radius m_v^*
        let best_phi = all_candidates[0].phi;
        let runner_up_phi = if all_candidates.len() > 1 {
            all_candidates[1].phi
        } else {
            phi_tail
        };
        let delta_12 = (best_phi - runner_up_phi).max(0.0);

        // Gradient norms:
        let s1 = &all_candidates[0];
        let h1_r = h_node - s1.h_left;
        let lambda = self.config.lambda_reg;
        let grad_g1 = (2.0 * s1.g_left.abs() / (s1.h_left + lambda))
            .max(2.0 * (g_node - s1.g_left).abs() / (h1_r + lambda));
        let grad_h1 = (s1.g_left * s1.g_left / ((s1.h_left + lambda) * (s1.h_left + lambda)))
            .max((g_node - s1.g_left) * (g_node - s1.g_left) / ((h1_r + lambda) * (h1_r + lambda)));

        let mut grad_g_sum = grad_g1;
        let mut grad_h_max = grad_h1;

        if all_candidates.len() > 1 {
            let s2 = &all_candidates[1];
            let h2_r = h_node - s2.h_left;
            let grad_g2 = (2.0 * s2.g_left.abs() / (s2.h_left + lambda))
                .max(2.0 * (g_node - s2.g_left).abs() / (h2_r + lambda));
            let grad_h2 = (s2.g_left * s2.g_left / ((s2.h_left + lambda) * (s2.h_left + lambda)))
                .max((g_node - s2.g_left) * (g_node - s2.g_left) / ((h2_r + lambda) * (h2_r + lambda)));
            grad_g_sum += grad_g2;
            if grad_h2 > grad_h_max {
                grad_h_max = grad_h2;
            }
        }

        let denom = 2.0 * g_max * grad_g_sum + h_max * grad_h_max;
        if denom > 0.0 {
            node.safe_radius_m_star = (delta_12 / denom).floor() as u32;
        } else {
            node.safe_radius_m_star = 0;
        }

        // Cache adjacent bin slice statistics for homotopy splicing
        let f_winner = all_candidates[0].feature_idx as usize;
        let tau_winner = all_candidates[0].bin_idx as usize;
        node.dg_slice[0] = hist_g[f_winner][tau_winner];
        node.dh_slice[0] = hist_h[f_winner][tau_winner];
        if tau_winner + 1 < num_bins {
            node.dg_slice[1] = hist_g[f_winner][tau_winner + 1];
            node.dh_slice[1] = hist_h[f_winner][tau_winner + 1];
        }

        // Partition samples into left and right children
        let best_feat = all_candidates[0].feature_idx as usize;
        let best_bin = all_candidates[0].bin_idx as u8;

        let mut left_samples = Vec::with_capacity(num_samples / 2);
        let mut right_samples = Vec::with_capacity(num_samples / 2);

        for &i in sample_indices {
            let bin = dataset.data[i * num_feats + best_feat];
            if bin <= best_bin {
                left_samples.push(i);
            } else {
                right_samples.push(i);
            }
        }

        // Child indices in complete binary tree
        let left_child_idx = 2 * node_idx + 1;
        let right_child_idx = 2 * node_idx + 2;

        let is_next_level_leaf = depth + 1 >= self.config.max_depth;
        if is_next_level_leaf {
            let left_leaf = *next_leaf_id;
            let right_leaf = *next_leaf_id + 1;
            node.left_child_id = -(left_leaf as i32) - 1;
            node.right_child_id = -(right_leaf as i32) - 1;
        } else {
            node.left_child_id = left_child_idx as i32;
            node.right_child_id = right_child_idx as i32;
        }

        tree.nodes[node_idx] = node;

        // Recurse left
        self.build_subtree(
            dataset,
            &left_samples,
            left_child_idx,
            depth + 1,
            tree,
            next_leaf_id,
            routing_index,
            tree_idx,
        );

        // Recurse right
        self.build_subtree(
            dataset,
            &right_samples,
            right_child_idx,
            depth + 1,
            tree,
            next_leaf_id,
            routing_index,
            tree_idx,
        );
    }
}
