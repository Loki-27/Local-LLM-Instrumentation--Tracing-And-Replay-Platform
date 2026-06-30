/*
 * src/main.cpp — Phase 5: Full integration
 *
 * Loads a GGUF model, runs inference in a background thread,
 * and displays a live five-panel TUI:
 *   Panel 1: Model computation graph (interactive tree, j/k/Space)
 *   Panel 2: Live tensor event stream (auto-scrolls)
 *   Panel 3: Capture target info / latency
 *   Panel 4: Per-tensor runtime metrics
 *   Panel 5: Numerical anomaly ledger
 *
 * Run: ./build/llm_tracer --model models/tinyllama.gguf
 *      ./build/llm_tracer --model models/tinyllama.gguf --prompt "Once upon a time"
 *
 * llama.cpp startup noise is redirected to llm-tracer.log
 */

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "llama.h"

#include "core/ring_buffer.hpp"
#include "core/attention_store.hpp"
#include "hook/hook_manager.hpp"
#include "topo/topology_parser.hpp"
#include "tui/tui_app.hpp"

static void print_usage(const char* prog) {
    printf("Usage: %s --model <path.gguf> [options]\n"
           "  --model    <path>   GGUF model file (required)\n"
           "  --prompt   <text>   Generation prompt\n"
           "  --n-tokens <N>      Max tokens to generate (default: 500)\n"
           "  --help\n",
           prog);
}

int main(int argc, char** argv) {

    
    std::string model_path;
    std::string prompt =
        "The transformer architecture works by";
    int n_tokens = 500;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--model")    && i+1<argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--prompt")   && i+1<argc) prompt     = argv[++i];
        else if (!strcmp(argv[i], "--n-tokens") && i+1<argc) n_tokens   = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--help"))    { print_usage(argv[0]); return 0; }
    }

    if (model_path.empty()) {
        fprintf(stderr, "[error] --model is required\n");
        print_usage(argv[0]);
        return 1;
    }

    // Print startup banner BEFORE redirecting stderr
    printf("\n  llm-tracer — LLM Instrumentation Platform\n");
    printf("  ─────────────────────────────────────────\n");
    printf("  Model:  %s\n", model_path.c_str());
    printf("  Prompt: %.55s%s\n", prompt.c_str(), prompt.size() > 55 ? "..." : "");
    printf("  Tokens: up to %d\n\n", n_tokens);
    printf("  Loading model... (details → llm-tracer.log)\n\n");
    fflush(stdout);

    // Redirect llama.cpp startup noise to log file — keeps TUI display clean
    freopen("llm-tracer.log", "w", stderr);

    // ── Data structures (outlive both threads) ────────────────────────────────
    EventBuffer    ring;
    AttentionStore attn_store;
    HookManager    hook(ring, attn_store);

    // ── Load model ────────────────────────────────────────────────────────────
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        printf("[error] Failed to load model. See llm-tracer.log for details.\n");
        llama_backend_free();
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    char model_desc[256] = "LLaMA";
    llama_model_desc(model, model_desc, sizeof(model_desc));

    // ── Create context WITH hook registered ───────────────────────────────────
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 2048;
    cparams.n_batch = 512;
    // cparams.flash_attn  = false; 
    // cparams.flash_attn_type = LLAMA_FLASH_ATTN_OFF;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    hook.attach_via_params(cparams);   // MUST be called before llama_init_from_model

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        printf("[error] Failed to create context. See llm-tracer.log.\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // ── Tokenize + prefill ────────────────────────────────────────────────────
    std::vector<llama_token> tokens(cparams.n_ctx);
    int n_prompt = llama_tokenize(
        vocab, prompt.c_str(), (int)prompt.size(),
        tokens.data(), (int)tokens.size(), true, false);
    if (n_prompt < 0) {
        printf("[error] Tokenization failed.\n");
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 1;
    }
    tokens.resize(n_prompt);

    // Prefill: this is when the hook fires for the first time.
    // After this call, ring buffer has events for every tensor in the forward pass.
    llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        printf("[error] Prefill decode failed. See llm-tracer.log.\n");
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 1;
    }

    // ── Build topology tree from prefill events ───────────────────────────────
    // snapshot() is thread-safe; topology is read-only after construction.
    TopoNode topo_root = TopologyParser::build_from_events(
        ring.snapshot(), model_desc);

    // ── Background inference thread ───────────────────────────────────────────
    // Runs greedily, pushing events to ring buffer each token.
    // Stops when: n_tokens reached | context full | EOS | stop_inference set.
    std::atomic<bool> stop_inference{false};
    std::atomic<int>  tokens_generated{0};

    std::thread inference_thread([&] {
        const int32_t   nv  = llama_vocab_n_tokens(vocab);
        const llama_token eos = llama_vocab_eos(vocab);
        int n_past = n_prompt;

        while (!stop_inference.load() &&
               tokens_generated.load() < n_tokens &&
               n_past < (int)cparams.n_ctx - 1) {

            // Greedy sampling
            float* logits = llama_get_logits(ctx);
            llama_token next = 0;
            float best = logits[0];
            for (int32_t j = 1; j < nv; ++j) {
                if (logits[j] > best) { best = logits[j]; next = j; }
            }
            if (next == eos) break;

            // Each llama_decode call triggers the ggml eval callback,
            // which calls HookManager::on_tensor_ready() for every tensor.
            llama_batch nb = llama_batch_get_one(&next, 1);
            if (llama_decode(ctx, nb) != 0) break;

            ++tokens_generated;
            ++n_past;
        }
    });

    // ── Launch TUI (blocks until Q pressed) ──────────────────────────────────
    TuiApp app(ring, attn_store, topo_root);
    app.run();

    // ── Cleanup ───────────────────────────────────────────────────────────────
    stop_inference.store(true);
    inference_thread.join();

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    printf("\nllm-tracer: %d tokens generated  |  %zu events captured  |  log: llm-tracer.log\n\n",
           tokens_generated.load(), ring.size());
    return 0;
}
