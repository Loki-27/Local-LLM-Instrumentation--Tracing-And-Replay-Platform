#pragma once
/*
 * include/topo/topology_parser.hpp
 *
 * Builds a navigable tree of the model's computation graph for Panel 1.
 *
 * HOW IT WORKS:
 *   After the first llama_decode() call, the HookManager has already captured
 *   every tensor name in the ring buffer. TopologyParser reads those names,
 *   extracts layer indices from suffixes like "-N" and "_lN", deduplicates,
 *   and constructs a three-level tree:
 *
 *     Root (model name)
 *     ├── embd                    (Embedding — root level)
 *     ├── Layers (22)             (group — collapsible)
 *     │   ├── Layer 0             (block — collapsible)
 *     │   │   ├── norm-0          (leaf — navigable, can be capture target)
 *     │   │   ├── kqv_out-0       (leaf — most useful capture target)
 *     │   │   └── ...             (11 tensors per layer)
 *     │   └── Layer 21
 *     └── Output                  (group — collapsible)
 *         ├── result_norm
 *         └── result_output
 *
 * WHY NOT WALK ggml_cgraph?
 *   That would require including llama-impl.h (private headers). Parsing the
 *   ring buffer achieves the same result using only our public hook data.
 *
 * TUI INTEGRATION (Phase 5):
 *   flatten_visible() converts the tree into a flat list respecting is_expanded
 *   state, so Panel 1 can render only visible rows and support j/k scrolling.
 *   Each FlatNode carries a depth field for indentation.
 */

#include "core/layer_event.hpp"
#include <string>
#include <vector>

// ── TopoNode ──────────────────────────────────────────────────────────────────

struct TopoNode {
    // Display
    std::string name;               // "Layer 0", "kqv_out-0", "Layers (22)", etc.
    std::string layer_type;         // Semantic: "block", "group", "LayerNorm", etc.
    int         layer_idx  = -1;    // 0..N for per-layer tensors; -1 for structural

    // Structure
    bool        is_leaf    = true;  // No children — a navigable row the user can select
    std::vector<TopoNode> children;

    // TUI state (mutated by Phase 5 keyboard handler)
    bool is_expanded       = false; // Non-leaf: show/hide children. Default: collapsed.
    bool is_capture_target = false; // Space pressed: feeds Panel 3 & 4 with this tensor's data.
};

// ── TopologyParser ────────────────────────────────────────────────────────────

class TopologyParser {
public:
    // ── Primary builder ───────────────────────────────────────────────────────

    // Call after first llama_decode() so the ring buffer has at least one full
    // forward pass worth of events.
    //
    // model_name: displayed as the root node label. Pass llama_model_desc() here.
    // events:     ring.snapshot() immediately after the first decode.
    //
    // Returns the fully constructed tree. The root node is always expanded.
    // Layer nodes are collapsed by default; "Layers" group is expanded.
    static TopoNode build_from_events(
        const std::vector<LayerEvent>& events,
        const std::string& model_name = "LLaMA"
    );

    // ── TUI helpers ───────────────────────────────────────────────────────────

    // A node as it appears in the flat Panel 1 scroll list.
    struct FlatNode {
        TopoNode* node;   // Raw pointer — valid as long as the root TopoNode lives.
        int       depth;  // Indentation level (0 = root, 1 = top children, etc.)
    };

    // Flatten the tree depth-first into visible rows only.
    // Collapsed nodes (is_expanded == false) hide their children.
    // Used by the TUI every render frame to build the scroll list.
    static std::vector<FlatNode> flatten_visible(TopoNode& root);

    // Count all leaf nodes reachable from a node (for "[11 tensors]" hints).
    static int count_leaves(const TopoNode& node);

    // ── Utility ───────────────────────────────────────────────────────────────

    // Extract the layer index from a tensor name suffix.
    // "kqv_out-7"          → 7
    // "cache_k_l12 (view)" → 12
    // "embd"               → -1  (not a per-layer tensor)
    static int extract_layer_idx(const std::string& name);
};
