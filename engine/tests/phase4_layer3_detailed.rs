//! VeriTree Phase 4 Detailed Verification Gate - Layer 3 Edge Cases
//! Covers: CoW Shadow LmO Evaluation, Tree Margin Auditing

use veritree_engine::{audit_tree_margins, eval_shadow_lmo, DecisionTree};

/// ---- Phase 4 Step 4.1: CoW Shadow LmO Evaluation ----

#[test]
fn test_eval_shadow_lmo_empty_clusters() {
    let tree = DecisionTree::new(3);
    let trees = vec![tree];
    let clusters: Vec<Vec<usize>> = vec![];
    let targets: Vec<f32> = vec![];

    let scores = eval_shadow_lmo(&trees, &clusters, &targets);
    assert!(scores.is_empty());
}

#[test]
fn test_eval_shadow_lmo_single_cluster() {
    let mut tree = DecisionTree::new(3);
    tree.nodes[0].h_node = 100.0;
    let trees = vec![tree];
    let clusters = vec![vec![0usize, 1, 2]]; // m = 3
    let targets = vec![0.5f32];

    let scores = eval_shadow_lmo(&trees, &clusters, &targets);
    assert_eq!(scores.len(), 1);
    assert!(scores[0] >= 0.0);
    assert!(scores[0].is_finite());
}

#[test]
fn test_eval_shadow_lmo_multiple_clusters() {
    let mut tree = DecisionTree::new(3);
    tree.nodes[0].h_node = 50.0;
    let trees = vec![tree];
    let clusters = vec![
        vec![0usize, 1],
        vec![2, 3, 4, 5],
        vec![6],
        vec![7, 8, 9, 10, 11],
    ];
    let targets = vec![0.1f32, 0.2, 0.3, 0.4];

    let scores = eval_shadow_lmo(&trees, &clusters, &targets);
    assert_eq!(scores.len(), 4);
    for s in &scores {
        assert!(s.is_finite());
        assert!(*s >= 0.0);
    }
}

#[test]
fn test_eval_shadow_lmo_multiple_trees() {
    let mut tree1 = DecisionTree::new(2);
    tree1.nodes[0].h_node = 10.0;
    let mut tree2 = DecisionTree::new(2);
    tree2.nodes[0].h_node = 20.0;
    let trees = vec![tree1, tree2];
    let clusters = vec![vec![0usize, 1, 2]];
    let targets = vec![0.5f32];

    let scores = eval_shadow_lmo(&trees, &clusters, &targets);
    assert_eq!(scores.len(), 1);
    assert!(scores[0].is_finite());
    // Score accumulates over both trees
    assert!(scores[0] > 0.0);
}

#[test]
fn test_eval_shadow_lmo_cluster_size_zero() {
    let mut tree = DecisionTree::new(3);
    tree.nodes[0].h_node = 10.0;
    let trees = vec![tree];
    // Empty cluster => m = 0 => delta = 0 => score = 0
    let clusters = vec![vec![]];
    let targets = vec![0.5f32];

    let scores = eval_shadow_lmo(&trees, &clusters, &targets);
    assert_eq!(scores.len(), 1);
    assert_eq!(scores[0], 0.0);
}

#[test]
fn test_eval_shadow_lmo_determinism() {
    let mut tree = DecisionTree::new(3);
    tree.nodes[0].h_node = 42.0;
    let trees = vec![tree];
    let clusters = vec![vec![0usize, 1, 2, 3, 4]];
    let targets = vec![0.5f32];

    let scores1 = eval_shadow_lmo(&trees, &clusters, &targets);
    let scores2 = eval_shadow_lmo(&trees, &clusters, &targets);
    assert_eq!(scores1, scores2);
}

#[test]
fn test_eval_shadow_lmo_zero_h_node() {
    let mut tree = DecisionTree::new(3);
    tree.nodes[0].h_node = 0.0;
    let trees = vec![tree];
    let clusters = vec![vec![0usize, 1]];
    let targets = vec![0.5f32];

    let scores = eval_shadow_lmo(&trees, &clusters, &targets);
    assert_eq!(scores.len(), 1);
    // delta = (2 * 0.01) / (0.0 + 1.0) = 0.02, score = 0.0004
    assert!((scores[0] - 0.0004).abs() < 1e-6);
}

/// ---- Phase 4 Step 4.2: Tree Margin Auditing ----

fn make_tree_with_safe_radius(safe_radius: u32, node_count: u32) -> DecisionTree {
    let mut tree = DecisionTree::new(3);
    for i in 0..(node_count as usize) {
        if i < tree.nodes.len() {
            tree.nodes[i].safe_radius_m_star = safe_radius;
            tree.nodes[i].node_sample_count = 10;
        }
    }
    tree
}

#[test]
fn test_audit_tree_margins_all_brittle() {
    let tree = make_tree_with_safe_radius(1, 7); // all safe_radius <= 1
    let (brittle, secure) = audit_tree_margins(&tree);
    assert_eq!(brittle, 7);
    assert_eq!(secure, 0);
}

#[test]
fn test_audit_tree_margins_all_secure() {
    let tree = make_tree_with_safe_radius(5, 7); // all safe_radius > 1
    let (brittle, secure) = audit_tree_margins(&tree);
    assert_eq!(brittle, 0);
    assert_eq!(secure, 7);
}

#[test]
fn test_audit_tree_margins_mixed() {
    let mut tree = DecisionTree::new(3);
    // Node 0: brittle (radius 0)
    tree.nodes[0].safe_radius_m_star = 0;
    tree.nodes[0].node_sample_count = 10;
    // Node 1: brittle (radius 1)
    tree.nodes[1].safe_radius_m_star = 1;
    tree.nodes[1].node_sample_count = 10;
    // Node 2: secure (radius 2)
    tree.nodes[2].safe_radius_m_star = 2;
    tree.nodes[2].node_sample_count = 10;
    // Node 3: secure (radius 100)
    tree.nodes[3].safe_radius_m_star = 100;
    tree.nodes[3].node_sample_count = 10;
    // Nodes 4-6: empty (sample_count = 0, skipped)
    for i in 4..7 {
        tree.nodes[i].node_sample_count = 0;
    }

    let (brittle, secure) = audit_tree_margins(&tree);
    assert_eq!(brittle, 2);
    assert_eq!(secure, 2);
}

#[test]
fn test_audit_tree_margins_empty_nodes_skipped() {
    let tree = DecisionTree::new(3);
    // All nodes have node_sample_count = 0 by default
    let (brittle, secure) = audit_tree_margins(&tree);
    assert_eq!(brittle, 0);
    assert_eq!(secure, 0);
}

#[test]
fn test_audit_tree_margins_single_node_brittle() {
    let mut tree = DecisionTree::new(3);
    tree.nodes[0].safe_radius_m_star = 1;
    tree.nodes[0].node_sample_count = 1;
    let (brittle, secure) = audit_tree_margins(&tree);
    assert_eq!(brittle, 1);
    assert_eq!(secure, 0);
}

#[test]
fn test_audit_tree_margins_single_node_secure() {
    let mut tree = DecisionTree::new(3);
    tree.nodes[0].safe_radius_m_star = 2;
    tree.nodes[0].node_sample_count = 1;
    let (brittle, secure) = audit_tree_margins(&tree);
    assert_eq!(brittle, 0);
    assert_eq!(secure, 1);
}

#[test]
fn test_audit_tree_margins_boundary_radius_one_is_brittle() {
    // safe_radius_m_star <= 1 => brittle
    let mut tree = DecisionTree::new(3);
    tree.nodes[0].safe_radius_m_star = 1;
    tree.nodes[0].node_sample_count = 5;
    let (brittle, _secure) = audit_tree_margins(&tree);
    assert_eq!(brittle, 1);

    let mut tree2 = DecisionTree::new(3);
    tree2.nodes[0].safe_radius_m_star = 0;
    tree2.nodes[0].node_sample_count = 5;
    let (brittle2, _secure2) = audit_tree_margins(&tree2);
    assert_eq!(brittle2, 1);
}

#[test]
fn test_audit_tree_margins_boundary_radius_two_is_secure() {
    // safe_radius_m_star > 1 => secure
    let mut tree = DecisionTree::new(3);
    tree.nodes[0].safe_radius_m_star = 2;
    tree.nodes[0].node_sample_count = 5;
    let (_brittle, secure) = audit_tree_margins(&tree);
    assert_eq!(secure, 1);
}
