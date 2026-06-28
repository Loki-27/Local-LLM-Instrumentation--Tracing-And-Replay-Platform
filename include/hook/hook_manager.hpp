#pragma once
/*
 * include/hook/hook_manager.hpp
 *
 * The non-invasive hook into llama.cpp's execution pipeline.
 *
 * HOW IT WORKS:
 *   llama_context_params has two fields: cb_eval and cb_eval_user_data.
 *   Set these BEFORE calling llama_init_from_model() and llama.cpp will
 *   call our static callback for every tensor it computes — before (ask=true)
 *   and after (ask=false). No source modification required.
 *
 * THREAD MODEL:
 *   The callback fires on whichever thread calls llama_decode() — the inference
 *   thread. The TUI reads from ring_ and attn_store_ on the main thread.
 *   Both are thread-safe (mutex-protected). timers_ is only accessed from the
 *   callback (single thread), so it needs no mutex.
 *
 * METAL NOTE (M5 MacBook):
 *   All tensors run on Metal GPU. t->data is a device pointer — dereferencing
 *   it on CPU causes a segfault. We detect this via ggml_backend_buffer_is_host()
 *   and skip float stat computation for GPU tensors. Shape, dtype, and latency
 *   are still captured for every tensor regardless of device.
 */

#include "core/layer_event.hpp"
#include "core/ring_buffer.hpp"
#include "core/attention_store.hpp"

#include <atomic>
#include <chrono>
#include <string>
#include <unordered_map>

#include "llama.h"    // llama_context_params, ggml_backend_sched_eval_callback
#include "ggml.h"     // ggml_tensor, ggml_nelements, ggml_type_name, ggml_get_f32_1d

// Convenience alias used throughout the project
// 73 events/token x ~56 tokens = 4096 events of history
using EventBuffer = RingBuffer<LayerEvent, 4096>;

class HookManager {
public:
    HookManager(EventBuffer& ring, AttentionStore& attn_store);

    // ── Attachment ────────────────────────────────────────────────────────────

    // Call this BEFORE llama_init_from_model().
    // Sets cparams.cb_eval and cparams.cb_eval_user_data so that llama.cpp
    // calls our callback for every tensor during inference.
    void attach_via_params(llama_context_params& cparams);

    // ── Runtime state (read by TUI) ───────────────────────────────────────────

    // Total LayerEvents pushed to the ring buffer since construction
    uint64_t event_count() const { return event_id_.load(); }

    // True once the first non-CPU-host tensor has been seen (Metal/CUDA run)
    bool is_gpu_run() const { return is_gpu_run_.load(); }

private:
    // ── The callback llama.cpp calls for every tensor ─────────────────────────

    // Static: llama.cpp expects a plain function pointer, not a member pointer.
    // user_data is always `this` (set in attach_via_params).
    // Returns true to allow computation to proceed (always true here — we observe,
    // never block).
    static bool ggml_callback(ggml_tensor* t, bool ask, void* user_data);

    // ── Event processing ──────────────────────────────────────────────────────

    // ask=true: tensor is ABOUT to be computed — record start time
    void on_tensor_start(ggml_tensor* t);

    // ask=false: tensor has FINISHED computing — read metadata + push event
    void on_tensor_ready(ggml_tensor* t);

    // ── Per-tensor helpers ────────────────────────────────────────────────────

    // Map raw tensor name → human-readable layer type string
    // "blk.1.attn_q"   → "Attn (Self)"
    // "blk.1.ffn_gate" → "MLP (SwiGLU)"
    // "blk.1.attn_norm"→ "LayerNorm"
    // "token_embd"     → "Embedding"
    static std::string classify_layer(const std::string& name);

    // Detect backend device from the tensor's buffer type
    // Returns "Metal (M5)", "CUDA [GPU 0]", "CPU", or "Unknown"
    static std::string detect_device(ggml_tensor* t);

    // Compute sparsity, mean, max — ONLY safe for CPU host tensors
    // Caller must check ggml_backend_buffer_is_host() before calling
    static void compute_stats(ggml_tensor* t, LayerEvent& ev);

    // ── Members ───────────────────────────────────────────────────────────────

    EventBuffer&    ring_;
    AttentionStore& attn_store_;

    // Per-tensor start times for latency measurement.
    // Keyed by tensor name (same thread only — no mutex needed).
    std::unordered_map<std::string,
        std::chrono::high_resolution_clock::time_point> timers_;

    std::atomic<uint64_t> event_id_{0};
    std::atomic<bool>     is_gpu_run_{false};
};
