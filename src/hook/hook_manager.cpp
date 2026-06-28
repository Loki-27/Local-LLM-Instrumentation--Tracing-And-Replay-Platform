/*
 * src/hook/hook_manager.cpp
 *
 * Implementation of HookManager — the non-invasive llama.cpp hook.
 *
 * Key design decisions documented here:
 *
 * 1. CALLBACK REGISTRATION via cparams.cb_eval
 *    We set llama_context_params::cb_eval before calling llama_init_from_model().
 *    This is the public API — no internal headers, no patching.
 *
 * 2. LATENCY MEASUREMENT
 *    ask=true fires immediately before ggml computes the tensor.
 *    ask=false fires immediately after. We store a high_resolution_clock
 *    timestamp keyed by tensor name on ask=true, compute delta on ask=false.
 *    Same thread → no mutex needed on timers_.
 *
 * 3. GPU/METAL TENSOR SAFETY
 *    ggml_backend_buffer_is_host(t->buffer) returns false for Metal/CUDA tensors.
 *    Reading t->data on those would segfault. We set stats_valid=false and skip
 *    float iteration entirely. Shape, dtype, device, and latency still captured.
 *
 * 4. PERFORMANCE
 *    The callback fires for every tensor — hundreds per token on a 1B model.
 *    Keep on_tensor_ready() cheap: no allocations in the hot path, no syscalls,
 *    no heavy computation. The ring buffer push is a mutex lock + array write.
 *    For CPU tensors, float iteration is bounded by a sample rate.
 */

#include "hook/hook_manager.hpp"
#include "hook/anomaly_detector.hpp"

#include "ggml-backend.h"   // ggml_backend_buffer_is_host, ggml_backend_buffer_get_type
                             // ggml_backend_buft_name

#include <cmath>
#include <cstring>

// ── Constructor ───────────────────────────────────────────────────────────────

HookManager::HookManager(EventBuffer& ring, AttentionStore& attn_store)
    : ring_(ring), attn_store_(attn_store)
{}

// ── Attachment ────────────────────────────────────────────────────────────────

void HookManager::attach_via_params(llama_context_params& cparams) {
    cparams.cb_eval           = HookManager::ggml_callback;
    cparams.cb_eval_user_data = this;
}

// ── Static callback (called by llama.cpp on inference thread) ─────────────────

bool HookManager::ggml_callback(ggml_tensor* t, bool ask, void* user_data) {
    // Always return true — we observe, never block computation
    if (!t || t->name[0] == '\0') return true;   // skip unnamed tensors

    auto* self = static_cast<HookManager*>(user_data);
    if (ask) {
        self->on_tensor_start(t);
    } else {
        self->on_tensor_ready(t);
    }
    return true;
}

// ── on_tensor_start (ask=true) ────────────────────────────────────────────────

void HookManager::on_tensor_start(ggml_tensor* t) {
    timers_[t->name] = std::chrono::high_resolution_clock::now();
}

// ── on_tensor_ready (ask=false) ───────────────────────────────────────────────

void HookManager::on_tensor_ready(ggml_tensor* t) {

    // ── Build LayerEvent ───────────────────────────────────────────────────────
    LayerEvent ev;
    ev.id         = event_id_++;
    ev.layer_name = t->name;
    ev.layer_type = classify_layer(t->name);
    ev.timestamp  = std::chrono::system_clock::now();

    // ── Latency ────────────────────────────────────────────────────────────────
    auto it = timers_.find(t->name);
    if (it != timers_.end()) {
        auto now     = std::chrono::high_resolution_clock::now();
        ev.latency_ms = std::chrono::duration<float, std::milli>(
                            now - it->second).count();
        timers_.erase(it);
    }

    // ── Tensor metadata ────────────────────────────────────────────────────────
    ev.shape = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
    ev.dtype = ggml_type_name(t->type);

    // ── Device detection ───────────────────────────────────────────────────────
    ev.compute_device = detect_device(t);

    // Track whether this is a GPU run (first non-host tensor sets the flag)
    if (!is_gpu_run_.load()) {
        if (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) {
            is_gpu_run_.store(true);
        }
    }

    // ── Float statistics ───────────────────────────────────────────────────────
    // ONLY safe for CPU host memory. Metal/CUDA tensors: skip, mark invalid.
    bool is_host = t->buffer && ggml_backend_buffer_is_host(t->buffer);
    if (is_host && t->data && ggml_nelements(t) > 0) {
        compute_stats(t, ev);
        ev.stats_valid = true;
    } else {
        ev.stats_valid    = false;
        ev.sparsity_rate  = -1.0f;
        ev.mean           = -1.0f;
        ev.max_val        = -1.0f;
    }

    // ── Anomaly check ──────────────────────────────────────────────────────────
    AnomalyDetector::check(ev, is_gpu_run_.load());

    // ── Push to ring buffer ────────────────────────────────────────────────────
    ring_.push(std::move(ev));
}

// ── classify_layer ────────────────────────────────────────────────────────────

std::string HookManager::classify_layer(const std::string& name) {
    // Ground truth from hook_dump.txt on TinyLlama / LLaMA-family models.
    // Each rule maps the actual tensor names seen in the ggml callback to a
    // human-readable type shown in Panel 2.

    // ── Attention internals ───────────────────────────────────────────────────
    // kqv_out-N  : output projection after Flash Attention → the main attn result
    // Qcur/Kcur/Vcur : Q, K, V projection intermediates (multiple reshapes)
    // __fattn__-N: Flash Attention kernel fused output
    // cache_k/v  : KV cache read/write views
    // (view)/(permuted): tensor reshape operations for head splitting
    if (name.find("kqv_out")     != std::string::npos) return "Attn (Self)";
    if (name.find("Qcur")        != std::string::npos ||
        name.find("Kcur")        != std::string::npos ||
        name.find("Vcur")        != std::string::npos) return "Attn QKV";
    if (name.find("__fattn__")   != std::string::npos) return "Flash Attn";
    if (name.find("cache_k")     != std::string::npos ||
        name.find("cache_v")     != std::string::npos) return "KV Cache";
    if (name.find("attn_out")    != std::string::npos) return "Attn Out Proj";

    // ── MLP / FFN ─────────────────────────────────────────────────────────────
    // ffn_gate / ffn_up  : SwiGLU gate and up projections
    // ffn_swiglu-N       : fused SwiGLU activation (gate * silu(up))
    // ffn_out-N          : down projection output
    // ffn_inp-N          : residual stream entering FFN block
    if (name.find("ffn_gate")    != std::string::npos) return "MLP Gate";
    if (name.find("ffn_up")      != std::string::npos) return "MLP Up";
    if (name.find("ffn_swiglu")  != std::string::npos) return "SwiGLU Act";
    if (name.find("ffn_out")     != std::string::npos) return "MLP Down";
    if (name.find("ffn_inp")     != std::string::npos) return "Residual";

    // ── Layer norm ────────────────────────────────────────────────────────────
    // norm-N / attn_norm-N / ffn_norm-N / result_norm
    if (name.find("norm")        != std::string::npos) return "LayerNorm";

    // ── Residual stream ───────────────────────────────────────────────────────
    // l_out-N: residual add at the end of each transformer block
    // attn_out is already caught above
    if (name.find("l_out")       != std::string::npos) return "Residual";



    // ── Embedding ─────────────────────────────────────────────────────────────
    if (name == "embd"               ||   // ← ADD THIS LINE
        name.find("token_embd")  != std::string::npos ||
        name.find("inp_embd")    != std::string::npos ||
        name.find("inp_pos")     != std::string::npos) return "Embedding";

    // ── Output / logits ───────────────────────────────────────────────────────
    if (name.find("result_output")!= std::string::npos) return "Logits";
    if (name.find("result_norm")  != std::string::npos) return "LayerNorm";

    // ── Generic view / permute ops (KV cache reshapes) ────────────────────────
    if (name.find("(view)")      != std::string::npos ||
        name.find("(permuted)")  != std::string::npos) return "Reshape";

    // ── Unnamed graph nodes ───────────────────────────────────────────────────
    if (name.substr(0, 4) == "node") return "Graph Node";

    return "Other";
}

// ── detect_device ─────────────────────────────────────────────────────────────

std::string HookManager::detect_device(ggml_tensor* t) {
    if (!t->buffer) return "CPU";

    // Host memory = CPU
    if (ggml_backend_buffer_is_host(t->buffer)) return "CPU";

    // GPU buffer — get backend type name for display
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t->buffer);
    if (buft) {
        const char* name = ggml_backend_buft_name(buft);
        if (name) {
            std::string n = name;
            if (n.find("Metal") != std::string::npos) return "Metal (M5)";
            if (n.find("CUDA")  != std::string::npos) return "CUDA [GPU 0]";
            if (n.find("Vulkan")!= std::string::npos) return "Vulkan";
            if (n.find("ROCm")  != std::string::npos) return "ROCm";
            // Return whatever the backend reports
            if (n.size() < 20) return n;
        }
    }
    return "GPU";
}

// ── compute_stats ─────────────────────────────────────────────────────────────
// PRE: t->buffer is host memory and t->data is valid CPU pointer.

void HookManager::compute_stats(ggml_tensor* t, LayerEvent& ev) {
    size_t n = static_cast<size_t>(ggml_nelements(t));
    if (n == 0) return;

    // For large tensors, sample at most 8192 elements to keep the callback fast.
    // This still gives a statistically representative picture of sparsity/mean/max.
    const size_t MAX_SAMPLE = 8192;
    size_t step   = (n > MAX_SAMPLE) ? (n / MAX_SAMPLE) : 1;
    size_t n_samp = 0;

    float sum      = 0.0f;
    float max_abs  = 0.0f;
    size_t n_zero  = 0;

    if (t->type == GGML_TYPE_F32) {
        const float* data = static_cast<const float*>(t->data);
        for (size_t i = 0; i < n; i += step, ++n_samp) {
            float v = data[i];
            float a = std::fabs(v);
            sum    += v;
            if (a < 1e-6f) ++n_zero;
            if (a > max_abs) max_abs = a;
        }
    } else if (t->type == GGML_TYPE_F16) {
        // ggml_get_f32_1d handles F16→F32 conversion and respects tensor strides
        for (size_t i = 0; i < n; i += step, ++n_samp) {
            float v = ggml_get_f32_1d(t, static_cast<int>(i));
            float a = std::fabs(v);
            sum    += v;
            if (a < 1e-6f) ++n_zero;
            if (a > max_abs) max_abs = a;
        }
    } else {
        // Quantized types (q4_K, q6_K, etc.) — cannot iterate as float without
        // dequantizing. Skip stats but caller already set stats_valid=true;
        // mark as invalid here.
        ev.stats_valid = false;
        return;
    }

    if (n_samp > 0) {
        ev.mean          = sum   / static_cast<float>(n_samp);
        ev.max_val       = max_abs;
        ev.sparsity_rate = static_cast<float>(n_zero) / static_cast<float>(n_samp);
    }
}
