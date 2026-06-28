/*
 * tests/test_hook_dump.cpp  —  Phase 0 baseline test (API-corrected)
 *
 * Phase 2: add HookManager here and print captured LayerEvents.
 */

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"

int main(int argc, char** argv) {
    std::string model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--model") && i+1<argc) model_path = argv[++i];
    }
    fprintf(stderr, "[test_hook_dump] model: %s\n", model_path.c_str());

    // ── PHASE 2: insert data structures here ────────────────────────────────
    // #include "core/ring_buffer.hpp"
    // #include "core/attention_store.hpp"
    // #include "hook/hook_manager.hpp"
    // using EventBuffer = RingBuffer<LayerEvent, 512>;
    // EventBuffer ring;
    // AttentionStore attn_store;
    // HookManager hook(ring, attn_store);
    // ────────────────────────────────────────────────────────────────────────

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "[test_hook_dump] FAIL — could not load model\n");
        fprintf(stderr, "                Run: bash scripts/download_model.sh\n");
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512; cparams.n_batch = 512;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "[test_hook_dump] FAIL — could not create context\n");
        llama_model_free(model); llama_backend_free();
        return 1;
    }

    // ── PHASE 2: attach hook here ─────────────────────────────────────────
    // hook.attach(llama_get_sched(ctx));
    // ─────────────────────────────────────────────────────────────────────

    const char* prompt = "The sky is";
    std::vector<llama_token> tokens(512);
    int n = llama_tokenize(vocab, prompt, (int)strlen(prompt),
                           tokens.data(), 512, true, false);
    if (n < 0) {
        fprintf(stderr, "[test_hook_dump] FAIL — tokenization failed\n");
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 1;
    }
    tokens.resize(n);

    llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "[test_hook_dump] FAIL — decode failed\n");
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 1;
    }

    // 5 greedy steps
    const int32_t   nv  = llama_vocab_n_tokens(vocab);
    const llama_token eos = llama_vocab_eos(vocab);
    for (int i = 0; i < 5; ++i) {
        float* logits = llama_get_logits(ctx);
        llama_token next = 0; float best = logits[0];
        for (int32_t j = 1; j < nv; ++j) if (logits[j] > best) { best=logits[j]; next=j; }
        if (next == eos) break;
        llama_batch nb = llama_batch_get_one(&next, 1);
        llama_decode(ctx, nb);
    }

    // ── PHASE 2: print captured events here ───────────────────────────────
    // auto events = ring.snapshot();
    // fprintf(stdout, "Captured %zu LayerEvents\n", events.size());
    // for (auto& ev : events) { fprintf(stdout, "%s\n", ev.summary_str().c_str()); }
    // ─────────────────────────────────────────────────────────────────────

    fprintf(stdout, "\n[test_hook_dump] BASELINE OK\n");
    fprintf(stdout, "  Model loaded:    yes\n");
    fprintf(stdout, "  Context created: yes\n");
    fprintf(stdout, "  Decode ran:      yes (5 tokens)\n");
    fprintf(stdout, "  Next step:       Phase 2 — attach HookManager\n\n");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
