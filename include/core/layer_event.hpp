#pragma once
/*
 * include/core/layer_event.hpp
 *
 * The atomic unit of data produced by HookManager and stored in RingBuffer.
 * Every tensor evaluation that llama.cpp performs produces one LayerEvent.
 *
 * All fields are populated in HookManager::on_tensor_ready() (Phase 2).
 * The helper methods (timestamp_str, shape_str) are used by TUI panels (Phase 5).
 */

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

struct LayerEvent {

    // ── Identity ─────────────────────────────────────────────────────────────
    uint64_t    id;               // Monotonically increasing, unique per run
    std::string layer_name;       // Raw tensor name from llama.cpp, e.g. "blk.1.attn_q"
    std::string layer_type;       // Human-readable classification:
                                  //   "Attn (Self)" | "MLP (SwiGLU)" | "LayerNorm"
                                  //   "Embedding"   | "Other"

    // ── Timing ───────────────────────────────────────────────────────────────
    std::chrono::system_clock::time_point timestamp;  // Wall-clock time of capture
    float latency_ms;             // Time between ask=true and ask=false for this tensor
                                  // Measures actual compute time for this layer

    // ── Hardware ─────────────────────────────────────────────────────────────
    std::string compute_device;   // "CUDA [GPU 0]" | "Metal" | "CPU"
                                  // Populated from ggml_backend_buffer type

    // ── Tensor metadata ───────────────────────────────────────────────────────
    std::vector<int64_t> shape;   // Dimensions from t->ne[0..3], e.g. {4096, 32, 1, 1}
    std::string dtype;            // From ggml_type_name(): "f32" | "f16" | "q4_0" etc

    // ── Activation statistics ─────────────────────────────────────────────────
    // Only valid when stats_valid == true.
    // Set to false for: GPU tensors (can't dereference device ptr on CPU),
    //                   quantized types (can't iterate as float directly).
    bool  stats_valid = false;
    float sparsity_rate = 0.0f;  // Fraction of elements where |x| < 1e-6
    float mean          = 0.0f;  // Arithmetic mean over all elements
    float max_val       = 0.0f;  // Maximum absolute value across all elements

    // ── Anomaly detection ─────────────────────────────────────────────────────
    // Populated by AnomalyDetector::check() immediately after stats are computed.
    bool        has_anomaly  = false;
    std::string anomaly_msg;      // Human-readable, shown in Panel 5

    // ── Display helpers (implemented in src/core/layer_event.cpp) ────────────

    // Returns "HH:MM:SS.mmm" formatted wall-clock time
    std::string timestamp_str() const;

    // Returns shape as "[4096, 32]" — trailing 1s stripped (GGML always has 4 dims)
    std::string shape_str() const;

    // Returns a single-line summary for Panel 2 (Live Packet Stream)
    // Format: "[  42] blk.1.attn_q     Attn (Self)    CUDA [GPU 0]   0.842ms"
    std::string summary_str() const;
};