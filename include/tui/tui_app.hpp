#pragma once
/*
 * include/tui/tui_app.hpp
 *
 * The top-level TUI orchestrator. Owns the ftxui screen loop, keyboard
 * handling, and all five panels.
 *
 * PANEL MAP:
 *   ┌─── P1: Model Topology ─────┬──── P2: Live Packet Stream ────────────┐
 *   │ j/k navigate, Space expand │ events scroll here in real time        │
 *   ├────────────────────────────┴────────────────────────────────────────┤
 *   │                P3: Attention Matrix Visualizer                      │
 *   ├───────────────────────────────┬─────────────────────────────────────┤
 *   │  P4: Runtime Metrics          │  P5: Anomaly Ledger                 │
 *   └───────────────────────────────┴─────────────────────────────────────┘
 *
 * KEYBOARD BINDINGS:
 *   Tab         → cycle focused panel (0→1→2→3→4→0)
 *   j / ↓       → Panel 1: cursor down
 *   k / ↑       → Panel 1: cursor up
 *   Space       → Panel 1: expand/collapse node OR set leaf as capture target
 *   Q / q       → quit
 *
 * THREADING:
 *   run() blocks on the main thread (ftxui's screen.Loop).
 *   A background refresh thread posts Event::Custom every 100 ms so the
 *   renderer re-reads the ring buffer and redraws live data.
 *   All shared data (ring_, attn_store_) is already mutex-protected.
 */

#include "core/ring_buffer.hpp"
#include "core/attention_store.hpp"
#include "hook/hook_manager.hpp"
#include "topo/topology_parser.hpp"

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include <ftxui/dom/elements.hpp>

class TuiApp {
public:
    // Construct before calling run().
    // ring, attn_store, topo_root must outlive the TuiApp.
    // topo_root must already be built (call TopologyParser::build_from_events first).
    TuiApp(EventBuffer&    ring,
           AttentionStore& attn_store,
           TopoNode&       topo_root);

    // Blocks until the user presses Q.
    // Launches a background refresh thread internally.
    void run();

private:
    // ── Data (owned elsewhere, read here) ────────────────────────────────────
    EventBuffer&    ring_;
    AttentionStore& attn_store_;
    TopoNode&       topo_root_;

    // ── TUI state (main thread only) ─────────────────────────────────────────
    int focused_panel_ = 0;      // 0–4
    int topo_cursor_   = 0;      // index into flat_nodes_
    std::string capture_target_; // name of the selected leaf node, "" if none

    // Flattened, visibility-respecting list of topo_root_ for Panel 1 rendering
    std::vector<TopologyParser::FlatNode> flat_nodes_;

    // ── Thread control ────────────────────────────────────────────────────────
    std::atomic<bool> running_{false};

    // ── Layout renderer ───────────────────────────────────────────────────────
    ftxui::Element render_layout();

    // ── Per-panel renderers ───────────────────────────────────────────────────
    // Panel 1: topology tree — fully implemented in Phase 4
    ftxui::Element render_panel1();

    // Panels 2–5: stubs in Phase 4, wired with real data in Phase 5
    ftxui::Element render_panel2();
    ftxui::Element render_panel3();
    ftxui::Element render_panel4();
    ftxui::Element render_panel5();

    // ── Helpers ───────────────────────────────────────────────────────────────

    // Rebuild flat_nodes_ from topo_root_ (call after any is_expanded change)
    void rebuild_flat();

    // Clear all is_capture_target flags in the tree
    void clear_capture_targets(TopoNode& node);

    // Panel title with active highlight when focused
    ftxui::Element panel_title(const std::string& label, int panel_idx);
};
