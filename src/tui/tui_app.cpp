/*
 * src/tui/tui_app.cpp
 *
 * Five-panel TUI implementation using ftxui.
 *
 * Phase 4 status:
 *   Panel 1 (Topology): FULLY IMPLEMENTED — j/k nav, Space expand/collapse/capture
 *   Panels 2–5:         STUBS — replaced with real data in Phase 5
 *
 * The overall structure is fixed here so Phase 5 only needs to replace
 * the render_panelN() body, not the layout or keyboard logic.
 */

#include "tui/tui_app.hpp"

#include <algorithm>
#include <map>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>

using namespace ftxui;

// ── Constructor ───────────────────────────────────────────────────────────────

TuiApp::TuiApp(EventBuffer&    ring,
               AttentionStore& attn_store,
               TopoNode&       topo_root)
    : ring_(ring), attn_store_(attn_store), topo_root_(topo_root)
{
    rebuild_flat();
}

// ── Helpers ───────────────────────────────────────────────────────────────────

void TuiApp::rebuild_flat() {
    flat_nodes_ = TopologyParser::flatten_visible(topo_root_);
    // Clamp cursor to valid range after rebuild
    if (!flat_nodes_.empty())
        topo_cursor_ = std::clamp(topo_cursor_, 0, (int)flat_nodes_.size() - 1);
    else
        topo_cursor_ = 0;
}

void TuiApp::clear_capture_targets(TopoNode& node) {
    node.is_capture_target = false;
    for (auto& child : node.children)
        clear_capture_targets(child);
}

Element TuiApp::panel_title(const std::string& label, int panel_idx) {
    bool active = (focused_panel_ == panel_idx);
    if (active)
        return text(" " + label + " ") | color(Color::Cyan) | bold;
    return text(" " + label + " ");
}

// ── Panel 1: Model Topology (fully implemented) ───────────────────────────────

Element TuiApp::render_panel1() {
    auto title = panel_title("1. MODEL TOPOLOGY", 0);
    bool focused = (focused_panel_ == 0);

    std::vector<Element> rows;

    if (flat_nodes_.empty()) {
        rows.push_back(text("  Waiting for inference...") | dim);
        rows.push_back(text("  (ring buffer empty)") | dim);
    } else {
        for (int i = 0; i < (int)flat_nodes_.size(); i++) {
            auto* node  = flat_nodes_[i].node;
            int   depth = flat_nodes_[i].depth;

            // Indentation
            std::string indent(depth * 2, ' ');

            // Node symbol
            std::string sym;
            if (node->is_leaf) {
                sym = node->is_capture_target ? "★" : "●";
            } else {
                sym = node->is_expanded ? "▼" : "►";
            }

            std::string line = indent + sym + " " + node->name;

            // Style
            Element row;
            if (focused && i == topo_cursor_) {
                // Cursor row
                row = text(line) | inverted;
            } else if (node->is_capture_target) {
                // Capture target highlight
                row = text(line) | color(Color::Yellow);
            } else if (!node->is_leaf) {
                // Group/block nodes slightly brighter
                row = text(line) | bold;
            } else {
                row = text(line);
            }

            // Tell ftxui to scroll this row into view when it's the cursor
            if (focused && i == topo_cursor_) {
                row = row | focus;
            }

            rows.push_back(row);
        }
    }

    // Navigation hint
    auto hint = hbox({
        text(" [j/k]") | color(Color::GrayLight),
        text(" navigate  ") | dim,
        text("[Space]") | color(Color::GrayLight),
        text(" expand/select") | dim,
    });

    auto content = vbox({
        vbox(rows) | frame | flex,
        separator(),
        hint,
    });

    return window(title, content | flex);
}

// ── Panel 2: Live Packet Stream (stub) ───────────────────────────────────────

Element TuiApp::render_panel2() {
    auto title = panel_title("2. LIVE PACKET STREAM", 1);

    // Phase 5: replace with ring_.latest(30) rendered as a scrolling table
    auto snap = ring_.snapshot();
    size_t total = snap.size();

    std::vector<Element> rows;
    rows.push_back(
        hbox({
            text("  ID  ") | bold,
            text("│ TENSOR NAME                    ") | bold,
            text("│ TYPE          ") | bold,
            text("│ LAT(ms)") | bold,
        }) | color(Color::GrayLight)
    );
    rows.push_back(separator());

    if (total == 0) {
        rows.push_back(text("  Waiting for events...") | dim);
    } else {
        // Show last 15 events
        int start = std::max(0, (int)total - 15);
        for (int i = start; i < (int)total; i++) {
            auto& ev = snap[i];
            std::string nm = ev.layer_name;
            if (nm.size() > 30) nm = nm.substr(0, 27) + "...";
            std::string tp = ev.layer_type;
            if (tp.size() > 13) tp = tp.substr(0, 13);

            char buf[128];
            std::snprintf(buf, sizeof(buf), "  %4llu  %-30s  %-13s  %6.3f",
                (unsigned long long)ev.id % 10000,
                nm.c_str(), tp.c_str(), ev.latency_ms);

            Element row = text(buf);
            if (ev.has_anomaly) row = row | color(Color::Red);
            rows.push_back(row);
        }
        rows.push_back(separator());
        rows.push_back(
            text("  Total: " + std::to_string(total) + " events captured") | dim
        );
    }

    return window(title, vbox(rows) | frame | flex);
}

// ── Panel 3: Attention Matrix (stub) ─────────────────────────────────────────

// Element TuiApp::render_panel3() {
//     auto title = panel_title("3. ATTENTION MATRIX VISUALIZER", 2);

//     // Phase 5: render AttentionStore snapshot as unicode block grid
//     // Note: with Metal, attention weights are fused inside __fattn__ kernel
//     // and not accessible as individual tensors. We show latency data instead.
//     std::vector<Element> rows;

//     if (capture_target_.empty()) {
//         rows.push_back(text("  Select a layer in Panel 1 and press [Space]") | dim);
//         rows.push_back(text("  Then latency timeline will appear here") | dim);
//     } else {
//         rows.push_back(hbox({
//             text("  Capture target: ") | dim,
//             text(capture_target_) | color(Color::Yellow) | bold,
//         }));
//         rows.push_back(text("  (Attention weights not accessible for Metal tensors)") | dim);
//         rows.push_back(text("  Phase 5: will show per-token latency sparkline here") | dim);
//     }

//     return window(title, hbox(rows));
// }




Element TuiApp::render_panel3() {
    auto title = panel_title("3. LAYER LATENCY PROFILE", 2);

    auto snap = ring_.snapshot();
    if (snap.empty()) {
        return window(title,
            text("  Waiting for inference data...") | dim);
    }

    // Average latency per layer type across all captured events
    std::map<std::string, std::pair<float,int>> type_lat; // type -> {sum, count}
    for (auto& ev : snap) {
        auto& p = type_lat[ev.layer_type];
        p.first += ev.latency_ms;
        p.second++;
    }

    // Ordered display list
    static const std::vector<std::string> ORDER = {
        "Embedding", "LayerNorm", "Attn QKV", "Flash Attn", "Attn (Self)",
        "Attn Out Proj", "Residual", "MLP Gate", "MLP Up", "SwiGLU Act",
        "MLP Down", "Logits"
    };

    float max_avg = 0.01f;
    for (auto& [type, p] : type_lat) {
        if (p.second > 0) max_avg = std::max(max_avg, p.first / p.second);
    }

    std::vector<Element> rows;
    rows.push_back(hbox({
        text("  Layer Type         ") | bold | color(Color::GrayLight),
        text("  Avg(ms)  ") | bold | color(Color::GrayLight),
        text("Distribution") | bold | color(Color::GrayLight),
    }));
    rows.push_back(separator());

    for (auto& type : ORDER) {
        if (!type_lat.count(type)) continue;
        auto& [sum, cnt] = type_lat[type];
        float avg = sum / cnt;
        int bar_w = std::max(1, (int)(avg / max_avg * 28));
        std::string bar(bar_w, '|');

        Color bar_col = Color::Cyan;
        if (type.find("MLP") != std::string::npos ||
            type.find("SwiGLU") != std::string::npos) bar_col = Color::Green;
        if (type.find("Norm") != std::string::npos)   bar_col = Color::GrayLight;
        if (type.find("Attn") != std::string::npos)   bar_col = Color::Yellow;

        std::string lbl = type;
        if (lbl.size() < 18) lbl.resize(18, ' ');
        char lat_str[16];
        std::snprintf(lat_str, sizeof(lat_str), "%6.3fms  ", avg);

        rows.push_back(hbox({
            text("  " + lbl) | dim,
            text(lat_str) | color(Color::Yellow),
            text(bar) | color(bar_col),
        }));
    }

    rows.push_back(separator());
    rows.push_back(hbox({
        text("  Note: Flash Attention active — attention weights fused in Metal kernel") | dim,
    }));

    return window(title, vbox(rows));
}




// ── Panel 4: Runtime Metrics (stub) ──────────────────────────────────────────

Element TuiApp::render_panel4() {
    auto title = panel_title("4. RUNTIME METRICS", 3);

    // Phase 5: show latest LayerEvent for the capture target
    auto snap = ring_.snapshot();

    std::vector<Element> rows;

    if (capture_target_.empty() || snap.empty()) {
        rows.push_back(text("  Tensor Shape  : [awaiting capture]") | dim);
        rows.push_back(text("  Dtype         : --") | dim);
        rows.push_back(text("  Latency Delta : --") | dim);
        rows.push_back(text("  Sparsity Rate : --") | dim);
    } else {
        // Find most recent event matching capture target
        const LayerEvent* latest = nullptr;
        for (auto it = snap.rbegin(); it != snap.rend(); ++it) {
            if (it->layer_name == capture_target_) { latest = &*it; break; }
        }
        if (latest) {
            rows.push_back(hbox({
                text("  Tensor Shape  : ") | dim,
                text(latest->shape_str()) | bold,
            }));
            rows.push_back(hbox({
                text("  Dtype         : ") | dim,
                text(latest->dtype) | bold,
            }));
            char lat[64];
            std::snprintf(lat, sizeof(lat), "%.3f ms", latest->latency_ms);
            rows.push_back(hbox({
                text("  Latency Delta : ") | dim,
                text(lat) | color(Color::Cyan),
            }));
            if (latest->stats_valid) {
                char sp[64];
                std::snprintf(sp, sizeof(sp), "%.1f%%", latest->sparsity_rate * 100.0f);
                rows.push_back(hbox({
                    text("  Sparsity Rate : ") | dim,
                    text(sp) | bold,
                }));
            } else {
                rows.push_back(text("  Sparsity Rate : N/A (Metal tensor)") | dim);
            }
        }
    }

    return window(title, vbox(rows));
}

// ── Panel 5: Anomaly Ledger (stub) ────────────────────────────────────────────

Element TuiApp::render_panel5() {
    auto title = panel_title("5. ANOMALY LEDGER", 4);

    // Phase 5: filter ring_ for has_anomaly == true
    auto snap = ring_.snapshot();
    std::vector<Element> rows;

    int anomaly_count = 0;
    for (auto it = snap.rbegin(); it != snap.rend(); ++it) {
        if (it->has_anomaly) {
            ++anomaly_count;
            if (anomaly_count <= 8) {  // show latest 8
                std::string ts = it->timestamp_str();
                Element row = hbox({
                    text("  " + ts + " ") | color(Color::GrayLight),
                    text(it->anomaly_msg),
                });
                // Colour by severity
                if (it->anomaly_msg.find("✖") != std::string::npos)
                    row = row | color(Color::Red);
                else
                    row = row | color(Color::Yellow);
                rows.push_back(row);
            }
        }
    }

    if (anomaly_count == 0) {
        rows.push_back(text("  ✓ No anomalies detected") | color(Color::Green));
    } else {
        rows.push_back(separator());
        rows.push_back(
            text("  Total: " + std::to_string(anomaly_count) + " anomalies") | dim
        );
    }

    return window(title, vbox(rows));
}

// ── Main layout ───────────────────────────────────────────────────────────────

Element TuiApp::render_layout() {
    // Header bar
    auto header = hbox({
        text(" [Tab] Panel  [j/k] Nav  [Space] Select  [Q] Quit ") | dim,
        filler(),
        text(" Focus: P" + std::to_string(focused_panel_ + 1) + " ") | color(Color::Cyan),
        text(" │ ") | dim,
        text(" Events: " + std::to_string(ring_.size()) + " "),
    });

    // Top row: Panel 1 (narrow) + Panel 2 (wide), takes most vertical space
    auto top = hbox({
        render_panel1() | size(WIDTH, GREATER_THAN, 32),
        render_panel2() | flex,
    }) | flex;

    // Middle: Panel 3 (full width, fixed height)
    auto mid = render_panel3() | size(HEIGHT, EQUAL, 16);

    // Bottom row: Panel 4 + Panel 5, fixed height
    auto bot = hbox({
        render_panel4() | flex,
        render_panel5() | flex,
    }) | size(HEIGHT, EQUAL, 9);

    return vbox({
        header,
        separator(),
        top,
        mid,
        bot,
    });
}

// ── run() ─────────────────────────────────────────────────────────────────────

void TuiApp::run() {
    auto screen = ScreenInteractive::Fullscreen();

    // Background refresh thread — triggers a re-render every 100 ms so live
    // data (ring buffer, anomaly ledger) updates without user input
    running_.store(true);
    std::thread refresh([&] {
        while (running_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            screen.PostEvent(Event::Custom);
        }
    });

    // Build the full component: renderer + keyboard handler
    auto component = CatchEvent(
        Renderer([&] { return render_layout(); }),
        [&](Event e) -> bool {

            // ── Global: Tab cycles focus ──────────────────────────────────
            if (e == Event::Tab) {
                focused_panel_ = (focused_panel_ + 1) % 5;
                return true;
            }

            // ── Global: Q quits ───────────────────────────────────────────
            if (e == Event::Character('q') || e == Event::Character('Q')) {
                screen.ExitLoopClosure()();
                return true;
            }

            // ── Panel 1 keys ──────────────────────────────────────────────
            if (focused_panel_ == 0) {

                // j / ↓ : cursor down
                if (e == Event::Character('j') || e == Event::ArrowDown) {
                    if (!flat_nodes_.empty())
                        topo_cursor_ = std::min(topo_cursor_ + 1,
                                                (int)flat_nodes_.size() - 1);
                    return true;
                }

                // k / ↑ : cursor up
                if (e == Event::Character('k') || e == Event::ArrowUp) {
                    topo_cursor_ = std::max(topo_cursor_ - 1, 0);
                    return true;
                }

                // Space: expand/collapse non-leaf; set capture target on leaf
                if (e == Event::Character(' ')) {
                    if (!flat_nodes_.empty() &&
                        topo_cursor_ < (int)flat_nodes_.size()) {
                        auto* node = flat_nodes_[topo_cursor_].node;
                        if (!node->is_leaf) {
                            // Toggle expand/collapse
                            node->is_expanded = !node->is_expanded;
                            rebuild_flat();
                        } else {
                            // Set as capture target (clears previous)
                            clear_capture_targets(topo_root_);
                            node->is_capture_target = true;
                            capture_target_ = node->name;
                        }
                    }
                    return true;
                }
            }

            return false;
        }
    );

    // Main loop (blocks)
    screen.Loop(component);

    // Cleanup
    running_.store(false);
    refresh.join();
}
