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
pub use layer2_quasiml::{classify_runner_up_transition, TransitionKind};
pub use layer3_security::{eval_shadow_lmo, audit_tree_margins};

/// Primary VeriTree Ensemble Orchestrator
pub struct VeriTreeEnsemble {
    pub config: TreeConfig,
    pub trees: Vec<DecisionTree>,
    pub routing_index: InvertedLeafRoutingIndex,
}

impl VeriTreeEnsemble {
    pub fn new(config: TreeConfig, num_samples: usize) -> Self {
        let num_trees = config.num_trees;
        let routing_index = InvertedLeafRoutingIndex::new(num_samples, num_trees);
        let trees = (0..num_trees)
            .map(|_| DecisionTree::new(config.max_depth))
            .collect();

        Self {
            config,
            trees,
            routing_index,
        }
    }

    /// Train ensemble on dataset
    pub fn fit(&mut self, dataset: &BinnedDataset) {
        let trainer = HistogramTrainer::new(self.config.clone());
        let all_indices: Vec<usize> = (0..dataset.num_samples).collect();
        for t in 0..self.trees.len() {
            self.trees[t] = trainer.train_tree(dataset, &all_indices, &mut self.routing_index, t);
        }
    }

    /// Amortized unlearning entry point
    pub fn unlearn_batch(&mut self, deleted_sample_ids: &[usize]) -> usize {
        let mut invariant_node_count = 0;
        for tree_idx in 0..self.trees.len() {
            for &sample_id in deleted_sample_ids {
                let leaf_id = self.routing_index.get_leaf(sample_id, tree_idx);
                let path = self.routing_index.get_ancestor_path(leaf_id as usize);
                invariant_node_count += path.len();
            }
        }
        invariant_node_count
    }
}
