/*
 * tests/test_tui_mock.cpp  —  Phase 4 TUI skeleton test
 *
 * Launches the full five-panel TUI with FAKE data — no llama model needed.
 * Tests that:
 *   - ftxui renders without crashing
 *   - All five panels appear
 *   - Tab, j/k, Space, Q work
 *   - The refresh thread runs and stops cleanly
 *
 * Build:  cmake --build build --target test_tui_mock
 * Run:    ./build/test_tui_mock
 *
 * You should see the TUI appear. Try:
 *   Tab           → cycle between panels (title turns cyan)
 *   j / k         → navigate topology tree in Panel 1
 *   Space         → expand/collapse a layer node, or set leaf as capture target
 *   Q             → quit
 *
 * This is a visual test — it passes if you can launch it, navigate, and quit.
 */

#include "core/ring_buffer.hpp"
#include "core/attention_store.hpp"
#include "hook/hook_manager.hpp"
#include "topo/topology_parser.hpp"
#include "tui/tui_app.hpp"

#include <chrono>
#include <cmath>
#include <string>
#include <vector>

// ── Build a fake topology tree (no llama needed) ──────────────────────────────

static TopoNode build_fake_topology() {
    // Build a 3-layer fake model for quick testing
    // Uses same structure as real TinyLlama output from test_topology_print

    // Simulate what build_from_events would produce
    std::vector<LayerEvent> fake_events;

    auto make_ev = [](const std::string& name, const std::string& type) {
        LayerEvent ev;
        ev.id          = 0;
        ev.layer_name  = name;
        ev.layer_type  = type;
        ev.timestamp   = std::chrono::system_clock::now();
        ev.latency_ms  = 0.25f;
        ev.shape       = {2048, 1, 1, 1};
        ev.dtype       = "f16";
        ev.stats_valid = false;
        ev.has_anomaly = false;
        return ev;
    };

    // Add root-level tensors
    fake_events.push_back(make_ev("embd",          "Embedding"));

    // Add 3 fake layers
    for (int i = 0; i < 3; i++) {
        auto s = std::to_string(i);
        fake_events.push_back(make_ev("norm-"       + s, "LayerNorm"));
        fake_events.push_back(make_ev("attn_norm-"  + s, "LayerNorm"));
        fake_events.push_back(make_ev("kqv_out-"    + s, "Attn (Self)"));
        fake_events.push_back(make_ev("attn_out-"   + s, "Attn Out Proj"));
        fake_events.push_back(make_ev("ffn_inp-"    + s, "Residual"));
        fake_events.push_back(make_ev("ffn_norm-"   + s, "LayerNorm"));
        fake_events.push_back(make_ev("ffn_gate-"   + s, "MLP Gate"));
        fake_events.push_back(make_ev("ffn_up-"     + s, "MLP Up"));
        fake_events.push_back(make_ev("ffn_swiglu-" + s, "SwiGLU Act"));
        fake_events.push_back(make_ev("ffn_out-"    + s, "MLP Down"));
        fake_events.push_back(make_ev("l_out-"      + s, "Residual"));
    }

    fake_events.push_back(make_ev("norm",          "LayerNorm"));
    fake_events.push_back(make_ev("result_norm",   "LayerNorm"));
    fake_events.push_back(make_ev("result_output", "Logits"));

    return TopologyParser::build_from_events(fake_events, "TestModel-3L (fake)");
}

// ── Build a fake ring buffer ──────────────────────────────────────────────────

static void populate_fake_ring(EventBuffer& ring) {
    auto make_ev = [](uint64_t id, const std::string& name,
                      const std::string& type, float lat, bool anomaly = false) {
        LayerEvent ev;
        ev.id          = id;
        ev.layer_name  = name;
        ev.layer_type  = type;
        ev.timestamp   = std::chrono::system_clock::now();
        ev.latency_ms  = lat;
        ev.shape       = {2048, 1, 1, 1};
        ev.dtype       = "f16";
        ev.stats_valid = false;
        ev.has_anomaly = anomaly;
        if (anomaly) ev.anomaly_msg = "⚠ Latency spike: " + name;
        return ev;
    };

    uint64_t id = 0;
    ring.push(make_ev(id++, "embd",          "Embedding",    0.40f));
    ring.push(make_ev(id++, "norm-0",        "LayerNorm",    0.21f));
    ring.push(make_ev(id++, "attn_norm-0",   "LayerNorm",    0.22f));
    ring.push(make_ev(id++, "kqv_out-0",     "Attn (Self)",  0.23f));
    ring.push(make_ev(id++, "attn_out-0",    "Attn Out",     0.24f));
    ring.push(make_ev(id++, "ffn_inp-0",     "Residual",     0.20f));
    ring.push(make_ev(id++, "ffn_gate-0",    "MLP Gate",     0.28f));
    ring.push(make_ev(id++, "ffn_out-0",     "MLP Down",     0.29f));
    ring.push(make_ev(id++, "l_out-0",       "Residual",     0.19f));
    ring.push(make_ev(id++, "norm-1",        "LayerNorm",    0.21f));
    ring.push(make_ev(id++, "kqv_out-1",     "Attn (Self)",  12.5f, true));  // anomaly
    ring.push(make_ev(id++, "ffn_gate-1",    "MLP Gate",     0.27f));
    ring.push(make_ev(id++, "ffn_out-1",     "MLP Down",     0.26f));
    ring.push(make_ev(id++, "result_output", "Logits",       0.61f));
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    // Build fake data
    EventBuffer    ring;
    AttentionStore attn_store;

    populate_fake_ring(ring);
    TopoNode topo_root = build_fake_topology();

    fprintf(stderr,
        "[test_tui_mock] Launching TUI with fake data:\n"
        "  Ring buffer: %zu events\n"
        "  Topology: %zu top-level nodes\n"
        "\n"
        "  Controls:\n"
        "    Tab       → cycle panel focus\n"
        "    j / k     → navigate topology (Panel 1)\n"
        "    Space     → expand/collapse node OR set capture target\n"
        "    Q         → quit\n"
        "\n"
        "  Expected: five-panel TUI appears, all controls work, Q quits cleanly.\n\n",
        ring.size(),
        topo_root.children.size()
    );

    // Launch TUI — blocks until Q
    TuiApp app(ring, attn_store, topo_root);
    app.run();

    fprintf(stderr, "[test_tui_mock] Quit cleanly. Phase 4 TUI skeleton: PASS\n\n");
    return 0;
}
