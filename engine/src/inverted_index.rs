//! VeriTree Inverted Leaf Routing Index
//! Zero-copy uint8_t N x T routing table supporting O(1) leaf lookup
//! and bitwise parent indexing p = (l - 1) >> 1.

#[derive(Clone, Debug)]
pub struct InvertedLeafRoutingIndex {
    pub num_samples: usize,
    pub num_trees: usize,
    /// Flat contiguous row-major buffer: shape [num_samples, num_trees]
    pub leaf_paths: Vec<u8>,
}

impl InvertedLeafRoutingIndex {
    /// Allocate an uninitialized or zeroed routing index
    pub fn new(num_samples: usize, num_trees: usize) -> Self {
        let total_size = num_samples
            .checked_mul(num_trees)
            .expect("Inverted index buffer size overflow");
        Self {
            num_samples,
            num_trees,
            leaf_paths: vec![0u8; total_size],
        }
    }

    /// O(1) direct lookup of terminal leaf for sample i in tree t
    #[inline(always)]
    pub fn get_leaf(&self, sample_idx: usize, tree_idx: usize) -> u8 {
        debug_assert!(sample_idx < self.num_samples);
        debug_assert!(tree_idx < self.num_trees);
        unsafe {
            *self.leaf_paths.get_unchecked(sample_idx * self.num_trees + tree_idx)
        }
    }

    /// Update terminal leaf for sample i in tree t
    #[inline(always)]
    pub fn set_leaf(&mut self, sample_idx: usize, tree_idx: usize, leaf_id: u8) {
        debug_assert!(sample_idx < self.num_samples);
        debug_assert!(tree_idx < self.num_trees);
        let idx = sample_idx * self.num_trees + tree_idx;
        self.leaf_paths[idx] = leaf_id;
    }

    /// Traverse upward from leaf to root via bitwise parent indexing: p = (l - 1) >> 1
    #[inline]
    pub fn get_ancestor_path(&self, mut node_idx: usize) -> Vec<usize> {
        let mut path = Vec::with_capacity(8);
        while node_idx > 0 {
            path.push(node_idx);
            node_idx = (node_idx - 1) >> 1;
        }
        path.push(0); // Root is always index 0
        path
    }

    /// Raw pointer access for C++ AVX2 kernels
    pub fn as_ptr(&self) -> *const u8 {
        self.leaf_paths.as_ptr()
    }
}

