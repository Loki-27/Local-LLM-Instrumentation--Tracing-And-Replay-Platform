/*
 * src/topo/topology_parser.cpp
 *
 * Builds the Panel 1 topology tree from LayerEvents captured by HookManager.
 *
 * Design note: we deliberately include only the 11 most semantically meaningful
 * tensors per layer (not all 23+). The others (Qcur views, KV cache reshapes,
 * __fattn__ intermediates) are internal bookkeeping — not useful as capture
 * targets and would clutter Panel 1. The 11 we keep map cleanly to the
 * conceptual transformer architecture a developer thinks in.
 */

#include "topo/topology_parser.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

// ── extract_layer_idx ─────────────────────────────────────────────────────────

int TopologyParser::extract_layer_idx(const std::string& name) {
    // Pattern 1: "name-N" or "name-N (suffix)"  e.g. "kqv_out-7", "norm-21"
    auto dash = name.rfind('-');
    if (dash != std::string::npos && dash + 1 < name.size()) {
        std::string after = name.substr(dash + 1);
        auto sp = after.find(' ');                  // strip " (view)" etc.
        if (sp != std::string::npos) after = after.substr(0, sp);
        try { return std::stoi(after); } catch (...) {}
    }
    // Pattern 2: "name_lN" or "name_lN (suffix)"  e.g. "cache_k_l12 (view)"
    auto lpos = name.rfind("_l");
    if (lpos != std::string::npos && lpos + 2 < name.size()) {
        std::string after = name.substr(lpos + 2);
        auto sp = after.find(' ');
        if (sp != std::string::npos) after = after.substr(0, sp);
        try { return std::stoi(after); } catch (...) {}
    }
    return -1;
}

// ── count_leaves ─────────────────────────────────────────────────────────────

int TopologyParser::count_leaves(const TopoNode& node) {
    if (node.is_leaf) return 1;
    int total = 0;
    for (auto& child : node.children) total += count_leaves(child);
    return total;
}

// ── flatten_visible ───────────────────────────────────────────────────────────

static void flatten_recursive(TopoNode& node, int depth,
                               std::vector<TopologyParser::FlatNode>& out) {
    out.push_back({&node, depth});
    if (!node.is_leaf && node.is_expanded) {
        for (auto& child : node.children) {
            flatten_recursive(child, depth + 1, out);
        }
    }
}

std::vector<TopologyParser::FlatNode>
TopologyParser::flatten_visible(TopoNode& root) {
    std::vector<FlatNode> result;
    result.reserve(64);
    flatten_recursive(root, 0, result);
    return result;
}

// ── build_from_events ─────────────────────────────────────────────────────────

TopoNode TopologyParser::build_from_events(
    const std::vector<LayerEvent>& events,
    const std::string& model_name
) {
    // ── Step 1: collect unique tensor names → layer_type ─────────────────────
    // The ring buffer has many duplicates (same tensor fires once per token).
    // We deduplicate by taking the first occurrence of each name.
    std::map<std::string, std::string> name_to_type;  // name → layer_type
    for (auto& ev : events) {
        if (!name_to_type.count(ev.layer_name)) {
            name_to_type[ev.layer_name] = ev.layer_type;
        }
    }

    // ── Step 2: find max layer index ─────────────────────────────────────────
    int max_layer = -1;
    for (auto& [name, type] : name_to_type) {
        int idx = extract_layer_idx(name);
        if (idx > max_layer) max_layer = idx;
    }

    // ── Step 3: build root node ───────────────────────────────────────────────
    TopoNode root;
    root.name        = model_name;
    root.layer_type  = "root";
    root.layer_idx   = -1;
    root.is_leaf     = false;
    root.is_expanded = true;   // root is always expanded

    // ── Step 4: embedding (root-level, appears before any layer) ─────────────
    for (auto& embd_name : {"embd", "token_embd", "inp_embd"}) {
        if (name_to_type.count(embd_name)) {
            TopoNode n;
            n.name       = embd_name;
            n.layer_type = name_to_type[embd_name];
            n.is_leaf    = true;
            root.children.push_back(n);
            break;  // only one embedding node needed
        }
    }

    // ── Step 5: Layers group ──────────────────────────────────────────────────
    if (max_layer >= 0) {
        TopoNode layers_group;
        layers_group.name       = "Layers (" + std::to_string(max_layer + 1) + ")";
        layers_group.layer_type = "group";
        layers_group.layer_idx  = -1;
        layers_group.is_leaf    = false;
        layers_group.is_expanded = true;   // show all layers by default

        // The 11 semantically meaningful tensors per layer, in forward-pass order.
        // These are the ones useful as Panel 3/4 capture targets.
        // Prefixes are matched as: prefix + std::to_string(layer_idx).
        static const std::vector<std::pair<std::string, std::string>> LAYER_TENSORS = {
            {"norm-",        "LayerNorm"},     // pre-attention RMS norm
            {"attn_norm-",   "LayerNorm"},     // attention weight norm
            {"kqv_out-",     "Attn (Self)"},   // ← best capture target for attention
            {"attn_out-",    "Attn Out Proj"}, // attention output projection
            {"ffn_inp-",     "Residual"},      // residual entering FFN block
            {"ffn_norm-",    "LayerNorm"},     // pre-FFN RMS norm
            {"ffn_gate-",    "MLP Gate"},      // SwiGLU gate projection
            {"ffn_up-",      "MLP Up"},        // SwiGLU up projection
            {"ffn_swiglu-",  "SwiGLU Act"},    // fused SwiGLU activation
            {"ffn_out-",     "MLP Down"},      // FFN down projection
            {"l_out-",       "Residual"},      // residual out of transformer block
        };

        for (int i = 0; i <= max_layer; ++i) {
            TopoNode layer;
            layer.name        = "Layer " + std::to_string(i);
            layer.layer_type  = "block";
            layer.layer_idx   = i;
            layer.is_leaf     = false;
            layer.is_expanded = false;   // collapsed by default — 22 open at once is too much

            for (auto& [prefix, default_type] : LAYER_TENSORS) {
                std::string full_name = prefix + std::to_string(i);
                // Use the layer_type we captured from the hook (more accurate),
                // fall back to the hardcoded default if not found.
                if (name_to_type.count(full_name)) {
                    TopoNode tensor;
                    tensor.name       = full_name;
                    tensor.layer_type = name_to_type[full_name];
                    tensor.layer_idx  = i;
                    tensor.is_leaf    = true;
                    layer.children.push_back(tensor);
                }
            }

            // Append "[N tensors]" hint to layer name
            int n = count_leaves(layer);
            if (n > 0) {
                layer.name += " [" + std::to_string(n) + "]";
            }

            layers_group.children.push_back(layer);
        }

        root.children.push_back(layers_group);
    }

    // ── Step 6: Output group ──────────────────────────────────────────────────
    static const std::vector<std::string> OUTPUT_TENSORS = {
        "norm", "result_norm", "result_output"
    };

    TopoNode output_group;
    output_group.name        = "Output";
    output_group.layer_type  = "group";
    output_group.layer_idx   = -1;
    output_group.is_leaf     = false;
    output_group.is_expanded = false;  // collapsed by default

    for (auto& out_name : OUTPUT_TENSORS) {
        if (name_to_type.count(out_name)) {
            TopoNode n;
            n.name       = out_name;
            n.layer_type = name_to_type[out_name];
            n.is_leaf    = true;
            output_group.children.push_back(n);
        }
    }
    if (!output_group.children.empty()) {
        int n = count_leaves(output_group);
        output_group.name += " [" + std::to_string(n) + "]";
        root.children.push_back(output_group);
    }

    return root;
}
