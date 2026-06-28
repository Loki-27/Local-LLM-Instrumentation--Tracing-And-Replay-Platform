#pragma once
/*
 * include/hook/anomaly_detector.hpp
 *
 * Lightweight rules engine that inspects a LayerEvent and sets has_anomaly
 * and anomaly_msg. Called synchronously inside HookManager::on_tensor_ready()
 * before pushing to the ring buffer.
 *
 * Rules (first match wins):
 *   1. Outlier activation  — max_val > 6.0 (only when stats_valid)
 *   2. CPU fallback        — tensor on CPU when the run is GPU (Metal/CUDA)
 *   3. High sparsity       — sparsity_rate > 90% (only when stats_valid)
 *   4. Latency spike       — latency_ms > 10ms (always applicable)
 *
 * The is_gpu_run flag is provided by HookManager so that the CPU fallback
 * rule only fires when we expect GPU but get CPU — not when the whole run
 * is CPU-only.
 *
 * All thresholds are compile-time constants so they're easy to tune.
 */

#include "core/layer_event.hpp"

struct AnomalyDetector {

    // Thresholds — adjust to taste
    static constexpr float OUTLIER_MAX_THRESHOLD  = 6.0f;    // max activation
    static constexpr float HIGH_SPARSITY_THRESHOLD = 0.90f;  // 90% zeros
    static constexpr float LATENCY_SPIKE_MS        = 10.0f;  // ms per tensor

    // Inspect ev, set ev.has_anomaly and ev.anomaly_msg if a rule fires.
    // is_gpu_run: pass HookManager::is_gpu_run() so the CPU-fallback rule
    //             only fires when appropriate.
    static void check(LayerEvent& ev, bool is_gpu_run = false);
};
