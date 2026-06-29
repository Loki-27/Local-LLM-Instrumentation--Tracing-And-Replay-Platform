/*
 * tests/test_topology_print.cpp  —  Phase 3 test
 *
 * Runs one decode with HookManager attached, builds the topology tree from
 * the captured events, and prints it to stdout in two formats:
 *   1. Full expanded tree (all nodes visible, with types)
 *   2. Default collapsed view (what Panel 1 shows on startup)
 *
 * Build:  cmake --build build --target test_topology_print
 * Run:    ./build/test_topology_print 2>/dev/null
 *
 * Expected:
 *   ▼ TinyLlama-1.1B (or whatever llama_model_desc returns)
 *     ● embd                [Embedding]
 *     ▼ Layers (22)
 *       ▼ Layer 0 [11]
 *         ● norm-0          [LayerNorm]
 *         ● attn_norm-0     [LayerNorm]
 *         ● kqv_out-0       [Attn (Self)]
 *         ...
 *       ► Layer 1 [11]
 *       ...
 *     ► Output [3]
 */

#include "core/ring_buffer.hpp"
#include "core/attention_store.hpp"
#include "hook/hook_manager.hpp"
#include "topo/topology_parser.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"

// ── Tree printing helpers ─────────────────────────────────────────────────────

static void print_node(const TopoNode& node, int depth, bool expand_all) {
    std::string indent(depth * 2, ' ');
    std::string marker;

    if (node.is_leaf) {
        marker = "●";
    } else if (node.is_expanded || expand_all) {
        marker = "▼";
    } else {
        marker = "►";
    }

    // Name + type annotation
    fprintf(stdout, "%s%s %-32s  [%s]\n",
        indent.c_str(),
        marker.c_str(),
        node.name.c_str(),
        node.layer_type.c_str());

    // Recurse into children if expanded or forcing all open
    if (!node.is_leaf && (node.is_expanded || expand_all)) {
        for (auto& child : node.children) {
            print_node(child, depth + 1, expand_all);
        }
    }
}

// ── Flat navigation list ──────────────────────────────────────────────────────

static void print_flat_view(TopoNode& root) {
    auto flat = TopologyParser::flatten_visible(root);
    fprintf(stdout,
        "\n── Flat navigation list (Panel 1 default view) ──────────────────────\n"
        "   This is what j/k scrolls through in the TUI:\n\n");
    for (size_t i = 0; i < flat.size(); ++i) {
        auto* n = flat[i].node;
        std::string indent(flat[i].depth * 2, ' ');
        std::string marker = n->is_leaf ? "●" : (n->is_expanded ? "▼" : "►");
        fprintf(stdout, "  [%3zu] %s%s %s\n",
            i, indent.c_str(), marker.c_str(), n->name.c_str());
    }
    fprintf(stdout, "\n  Total visible rows: %zu\n", flat.size());
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--model") && i+1<argc) model_path = argv[++i];
    }

    // ── Setup (same as test_hook_dump) ────────────────────────────────────────
    EventBuffer    ring;
    AttentionStore attn_store;
    HookManager    hook(ring, attn_store);

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "FAIL — could not load: %s\n", model_path.c_str());
        return 1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);

    // Get model description for the root node label
    char model_desc[256] = "LLaMA";
    llama_model_desc(model, model_desc, sizeof(model_desc));

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;
    hook.attach_via_params(cparams);

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "FAIL — could not create context\n");
        llama_model_free(model); llama_backend_free(); return 1;
    }

    // ── Run ONE decode to populate ring buffer ────────────────────────────────
    const char* prompt = "The sky is";
    std::vector<llama_token> tokens(512);
    int n = llama_tokenize(vocab, prompt, (int)strlen(prompt),
                           tokens.data(), 512, true, false);
    tokens.resize(n);
    llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "FAIL — decode failed\n");
        llama_free(ctx); llama_model_free(model); llama_backend_free(); return 1;
    }

    // ── Build topology tree ───────────────────────────────────────────────────
    auto events = ring.snapshot();
    TopoNode root = TopologyParser::build_from_events(events, model_desc);

    fprintf(stdout,
        "═══════════════════════════════════════════════════════════════\n"
        " Topology tree: %s\n"
        " Built from %zu LayerEvents\n"
        " Layers detected: %zu\n"
        "═══════════════════════════════════════════════════════════════\n\n",
        model_desc,
        events.size(),
        root.children.size() >= 2 ? root.children[1].children.size() : 0
    );

    // ── Print 1: Full expanded tree ───────────────────────────────────────────
    fprintf(stdout, "── Full tree (all nodes expanded) ──────────────────────────────\n\n");
    print_node(root, 0, /*expand_all=*/true);

    // ── Print 2: Default collapsed view ──────────────────────────────────────
    fprintf(stdout,
        "\n── Default view (layers collapsed, as Panel 1 starts) ──────────\n\n");
    print_node(root, 0, /*expand_all=*/false);
    print_flat_view(root);

    // ── Validation ────────────────────────────────────────────────────────────
    fprintf(stdout,
        "\n── Validation ──────────────────────────────────────────────────\n");

    bool ok = true;

    // Root should have at least 2 children (embd + Layers or Layers + Output)
    if (root.children.size() < 2) {
        fprintf(stdout, "  FAIL  root has only %zu children (expected ≥2)\n",
                root.children.size());
        ok = false;
    } else {
        fprintf(stdout, "  PASS  root has %zu top-level children\n",
                root.children.size());
    }

    // Find "Layers" group
    TopoNode* layers_group = nullptr;
    for (auto& c : root.children) {
        if (c.layer_type == "group" && c.name.find("Layers") != std::string::npos) {
            layers_group = &c;
        }
    }

    if (!layers_group) {
        fprintf(stdout, "  FAIL  no 'Layers' group found\n");
        ok = false;
    } else {
        fprintf(stdout, "  PASS  Layers group found with %zu layer children\n",
                layers_group->children.size());

        // Each layer should have children
        if (!layers_group->children.empty()) {
            auto& layer0 = layers_group->children[0];
            int leaves = TopologyParser::count_leaves(layer0);
            if (leaves >= 11) {
                fprintf(stdout, "  PASS  Layer 0 has %d tensors (expected 11)\n", leaves);
            } else {
                fprintf(stdout, "  FAIL  Layer 0 has only %d tensors\n", leaves);
                ok = false;
            }

            // kqv_out-0 should be in Layer 0 (primary capture target)
            bool found_kqv = false;
            for (auto& t : layer0.children) {
                if (t.name == "kqv_out-0") { found_kqv = true; break; }
            }
            if (found_kqv) {
                fprintf(stdout, "  PASS  kqv_out-0 present in Layer 0\n");
            } else {
                fprintf(stdout, "  FAIL  kqv_out-0 missing from Layer 0\n");
                ok = false;
            }
        }
    }

    // flatten_visible with default state should show ~27 rows
    // (root + embd + Layers group + 22 layer rows + Output group)
    auto flat = TopologyParser::flatten_visible(root);
    if (flat.size() >= 25) {
        fprintf(stdout, "  PASS  flatten_visible returns %zu rows\n", flat.size());
    } else {
        fprintf(stdout, "  FAIL  flatten_visible only %zu rows (expected ~27)\n",
                flat.size());
        ok = false;
    }

    fprintf(stdout,
        "\n═══════════════════════════════════════════════════════════════\n");
    if (ok) {
        fprintf(stdout, " ✓  TOPOLOGY TEST PASSED — ready for Phase 4 (TUI)\n");
    } else {
        fprintf(stdout, " ✗  TOPOLOGY TEST FAILED\n");
    }
    fprintf(stdout, "═══════════════════════════════════════════════════════════════\n\n");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}
