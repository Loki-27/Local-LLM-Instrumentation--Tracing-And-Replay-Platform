/*
 * src/hook/anomaly_detector.cpp
 */

#include "hook/anomaly_detector.hpp"

#include <cmath>
#include <string>

void AnomalyDetector::check(LayerEvent& ev, bool is_gpu_run) {
    ev.has_anomaly = false;
    ev.anomaly_msg.clear();

    // ── Rule 1: Outlier activation ─────────────────────────────────────────────
    // High max activation can indicate numerical instability, gradient explosion
    // in the residual stream, or clipping risks. Only meaningful for float types.
    if (ev.stats_valid && ev.max_val > OUTLIER_MAX_THRESHOLD) {
        ev.has_anomaly = true;
        ev.anomaly_msg = "⚠ Outlier feature in " + ev.layer_name
                       + ": max=" + std::to_string(ev.max_val).substr(0, 5);
        return;
    }

    // ── Rule 2: CPU fallback ───────────────────────────────────────────────────
    // When the model is loaded on GPU (Metal/CUDA), a tensor landing on CPU means
    // the GPU scheduler couldn't handle it — memory pressure or unsupported op.
    // Only relevant when we're in a GPU run; CPU-only runs are expected on CPU.
    if (is_gpu_run) {
        bool on_cpu = (ev.compute_device.find("CPU") != std::string::npos &&
                       ev.compute_device.find("GPU") == std::string::npos &&
                       ev.compute_device != "Metal (M5)");
        // Embedding lookup is always CPU-mapped (mmap weights) — not a real fallback
        bool is_embed = (ev.layer_type == "Embedding" || ev.layer_name == "embd");
        if (on_cpu && !is_embed) {
            ev.has_anomaly = true;
            ev.anomaly_msg = "✖ CPU fallback: " + ev.layer_name
                           + " ran on CPU (GPU expected)";
            return;
        }
    }

    // ── Rule 3: High sparsity ──────────────────────────────────────────────────
    // Very high sparsity (>90% near-zero activations) can indicate dead neurons,
    // over-aggressive ReLU, or that a layer is contributing almost nothing.
    if (ev.stats_valid && ev.sparsity_rate > HIGH_SPARSITY_THRESHOLD) {
        ev.has_anomaly = true;
        ev.anomaly_msg = "⚠ High sparsity in " + ev.layer_name
                       + ": " + std::to_string(int(ev.sparsity_rate * 100)) + "%";
        return;
    }

    // ── Rule 4: Latency spike ──────────────────────────────────────────────────
    // A single tensor taking >10ms is unusual. Could indicate:
    //   - Memory bandwidth saturation
    //   - GPU pipeline stall
    //   - Unexpectedly large tensor for context length
    if (ev.latency_ms > LATENCY_SPIKE_MS) {
        ev.has_anomaly = true;
        ev.anomaly_msg = "⚠ Latency spike in " + ev.layer_name
                       + ": " + std::to_string(ev.latency_ms).substr(0, 6) + "ms";
        return;
    }
}
