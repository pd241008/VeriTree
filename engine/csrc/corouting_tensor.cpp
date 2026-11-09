#include "ring_buffer.hpp"
#include <vector>
#include <cmath>
#include <cstring>

namespace veritree {

// Co-routing accumulator data structure
struct CoRoutingAccumulator {
    float H_cross;      // H_{\ell, u}^{(t -> t+1)}
    uint32_t N_cross;   // N_{\ell, u}^{(t -> t+1)}
};

// Compute sparse leaf co-routing and propagate gradients to tree t+1
extern "C" void propagate_leaf_corouting_residuals(
    const uint8_t* __restrict leaf_paths,       // N x T array
    size_t num_samples,
    size_t num_trees,
    uint32_t tree_t,
    uint32_t tree_t_plus_1,
    const float* __restrict delta_w,            // Size L_t
    size_t num_leaves_t,
    size_t num_nodes_t_plus_1,
    float eta,
    float M_3,
    float epsilon_prop,
    float* __restrict G_nodes_t_plus_1,         // In/Out: Updated G_u
    float* __restrict E_nodes_t_plus_1          // Out: Taylor remainder E_u
) {
    // Identify active leaves exceeding epsilon_prop
    std::vector<uint32_t> active_leaves;
    active_leaves.reserve(num_leaves_t);
    for (size_t l = 0; l < num_leaves_t; ++l) {
        if (std::abs(delta_w[l]) > epsilon_prop) {
            active_leaves.push_back(static_cast<uint32_t>(l));
        }
    }

    if (active_leaves.empty()) {
        std::memset(E_nodes_t_plus_1, 0, num_nodes_t_plus_1 * sizeof(float));
        return;
    }

    // Allocate dense accumulator for (active_leaves x nodes_t_plus_1)
    size_t num_active = active_leaves.size();
    std::vector<CoRoutingAccumulator> accum(num_active * num_nodes_t_plus_1, {0.0f, 0});

    // Scan samples to accumulate cross-routing statistics on the fly
    for (size_t i = 0; i < num_samples; ++i) {
        uint8_t leaf_t = leaf_paths[i * num_trees + tree_t];
        uint8_t leaf_next = leaf_paths[i * num_trees + tree_t_plus_1];

        // Check if sample belongs to an active leaf
        for (size_t a = 0; a < num_active; ++a) {
            if (active_leaves[a] == leaf_t) {
                // Leaf next is an ancestor or terminal leaf
                size_t node_idx = static_cast<size_t>(leaf_next);
                if (node_idx < num_nodes_t_plus_1) {
                    accum[a * num_nodes_t_plus_1 + node_idx].H_cross += 1.0f; // Unit Hessian or loaded
                    accum[a * num_nodes_t_plus_1 + node_idx].N_cross += 1;
                }
                break;
            }
        }
    }

    // Apply Quasi-ML gradient propagation and Taylor remainder calculation
    float taylor_factor = (M_3 * eta * eta) * 0.5f;

    for (size_t u = 0; u < num_nodes_t_plus_1; ++u) {
        float grad_shift = 0.0f;
        float var_sum = 0.0f;

        for (size_t a = 0; a < num_active; ++a) {
            uint32_t l = active_leaves[a];
            float dw = delta_w[l];
            const auto& cell = accum[a * num_nodes_t_plus_1 + u];

            grad_shift += cell.H_cross * dw;
            var_sum += static_cast<float>(cell.N_cross) * (dw * dw);
        }

        // Equation (14): Ĝ_u^{(t+1)} += η * ∑ H_{ℓ, u} * Δw_ℓ
        G_nodes_t_plus_1[u] += eta * grad_shift;

        // Equation (15): E_u^{(t+1)} = (M_3 * η^2 / 2) * ∑ N_{ℓ, u} * (Δw_ℓ)^2
        E_nodes_t_plus_1[u] = taylor_factor * var_sum;
    }
}

} // namespace veritree

