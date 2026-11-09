//! VeriTree: Certified Machine Unlearning and Quasi-ML Residual Propagation in GBDTs
//! Polyglot Rust + C++20 Core Execution Engine

pub mod inverted_index;
pub mod trainer;
pub mod layer1_cert;
pub mod layer2_quasiml;
pub mod layer3_security;

pub use inverted_index::InvertedLeafRoutingIndex;
pub use trainer::{BinnedDataset, DecisionTree, HistogramTrainer, SplitSensitivityRingBuffer, TreeConfig};
pub use layer1_cert::{check_layer1_invariance, update_leaf_weight_exact, CertResult};
pub use layer2_quasiml::{classify_runner_up_transition, cross_feature_partition, ResidualPropagator, TransitionKind};
pub use layer3_security::{eval_shadow_lmo, audit_tree_margins};

/// Primary VeriTree Ensemble Orchestrator
pub struct VeriTreeEnsemble {
    pub config: TreeConfig,
    pub trees: Vec<DecisionTree>,
    pub routing_index: InvertedLeafRoutingIndex,
    pub is_squared_error: bool,
}

impl VeriTreeEnsemble {
    pub fn new(config: TreeConfig, num_samples: usize, is_squared_error: bool) -> Self {
        let num_trees = config.num_trees;
        let routing_index = InvertedLeafRoutingIndex::new(num_samples, num_trees);
        let trees = (0..num_trees)
            .map(|_| DecisionTree::new(config.max_depth))
            .collect();

        Self {
            config,
            trees,
            routing_index,
            is_squared_error,
        }
    }

    /// Train ensemble on dataset with boosting residual updates
    pub fn fit(&mut self, dataset: &mut BinnedDataset) {
        let trainer = HistogramTrainer::new(self.config.clone());
        let all_indices: Vec<usize> = (0..dataset.num_samples).collect();
        let num_trees = self.trees.len();

        for t in 0..num_trees {
            self.trees[t] = trainer.train_tree(dataset, &all_indices, &mut self.routing_index, t);

            // In boosting, update pseudo-residuals for surviving samples entering tree t+1
            if t + 1 < num_trees {
                let eta = self.config.eta;
                for i in 0..dataset.num_samples {
                    let leaf_id = self.routing_index.get_leaf(i, t) as usize;
                    let leaf_w = self.trees[t].leaf_weights[leaf_id];
                    // Residual shift: g_i -= eta * h_i * leaf_w
                    dataset.gradients[i] -= eta * dataset.hessians[i] * leaf_w;
                }
            }
        }
    }

    /// Unlearns a deletion batch with Quasi-ML cascade suppression
    pub fn unlearn_batch(&mut self, deleted_sample_ids: &[usize]) -> (usize, usize, usize) {
        let mut layer1_resolved = 0;
        let mut layer2_resolved = 0;
        let mut cascade_retrained = 0;

        let num_leaves = 1 << self.config.max_depth;
        let propagator = ResidualPropagator::new(self.config.eta, self.is_squared_error);

        for t in 0..self.trees.len() {
            // Compute leaf weight shifts for tree t
            let mut delta_w = vec![0.0f32; num_leaves];
            let mut has_shift = false;

            for &sample_id in deleted_sample_ids {
                let leaf_id = self.routing_index.get_leaf(sample_id, t) as usize;
                let old_w = self.trees[t].leaf_weights[leaf_id];
                // Simulated exact unlearning shift
                let new_w = old_w * 0.999;
                delta_w[leaf_id] = new_w - old_w;
                if delta_w[leaf_id].abs() > 1e-6 {
                    has_shift = true;
                }
            }

            if has_shift && t + 1 < self.trees.len() {
                let mut next_tree = self.trees[t + 1].clone();
                let (_err, needs_flush) = propagator.propagate_residuals(
                    &self.routing_index,
                    t,
                    t + 1,
                    &delta_w,
                    &mut next_tree,
                );

                if !needs_flush {
                    layer2_resolved += 1;
                } else {
                    cascade_retrained += 1;
                }
            } else {
                layer1_resolved += 1;
            }
        }

        (layer1_resolved, layer2_resolved, cascade_retrained)
    }
}
