/*
 * src/main.cpp  —  Phase 0 baseline inference (API-corrected for llama.cpp b5659+)
 *
 * API changes from older llama.cpp:
 *   llama_load_model_from_file    → llama_model_load_from_file
 *   llama_new_context_with_model  → llama_init_from_model
 *   llama_free_model              → llama_model_free
 *   llama_tokenize(model, ...)    → llama_tokenize(vocab, ...)
 *   llama_n_vocab(model)          → llama_vocab_n_tokens(vocab)
 *   llama_token_eos(model)        → llama_vocab_eos(vocab)
 *   llama_token_to_piece(model,..)→ llama_token_to_piece(vocab,..)
 */

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --model <path.gguf> [--prompt <text>] [--n-tokens <N>]\n",
        prog);
}

int main(int argc, char** argv) {

    // ── 1. Parse CLI ────────────────────────────────────────────────────────
    std::string model_path;
    std::string prompt  = "The sky is";
    int         n_gen   = 10;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--model")    && i+1<argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--prompt")   && i+1<argc) prompt     = argv[++i];
        else if (!strcmp(argv[i], "--n-tokens") && i+1<argc) n_gen      = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--help"))    { print_usage(argv[0]); return 0; }
    }
    if (model_path.empty()) {
        fprintf(stderr, "[error] --model is required\n");
        print_usage(argv[0]);
        return 1;
    }

    // ── 2. Backend ──────────────────────────────────────────────────────────
    llama_backend_init();
    fprintf(stderr, "[llm-tracer] backend initialized\n");

    // ── 3. Load model ───────────────────────────────────────────────────────
    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "[error] failed to load: %s\n", model_path.c_str());
        llama_backend_free();
        return 1;
    }
    fprintf(stderr, "[llm-tracer] model loaded\n");

    // ── 4. Get vocab (needed for tokenize, eos, n_vocab, token_to_piece) ───
    const llama_vocab* vocab = llama_model_get_vocab(model);

    // ── 5. Create context ───────────────────────────────────────────────────
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "[error] failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    fprintf(stderr, "[llm-tracer] context created (n_ctx=%d)\n", (int)cparams.n_ctx);

    // ── PHASE 2 INSERTION POINT ─────────────────────────────────────────────
    // HookManager hook(event_buffer, attn_store);
    // hook.attach(llama_get_sched(ctx));
    // ────────────────────────────────────────────────────────────────────────

    // ── 6. Tokenize ─────────────────────────────────────────────────────────
    std::vector<llama_token> tokens(cparams.n_ctx);
    int n_prompt = llama_tokenize(
        vocab,
        prompt.c_str(), (int)prompt.size(),
        tokens.data(), (int)tokens.size(),
        /*add_special=*/true, /*parse_special=*/false
    );
    if (n_prompt < 0) {
        fprintf(stderr, "[error] tokenization failed\n");
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 1;
    }
    tokens.resize(n_prompt);
    fprintf(stderr, "[llm-tracer] prompt: %d tokens — generating %d...\n\n", n_prompt, n_gen);

    // ── 7. Prefill ──────────────────────────────────────────────────────────
    llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "[error] prefill failed\n");
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 1;
    }

    // ── 8. Greedy generation loop ───────────────────────────────────────────
    fprintf(stdout, "[prompt] %s\n[output]", prompt.c_str());

    const int32_t   n_vocab = llama_vocab_n_tokens(vocab);
    const llama_token eos   = llama_vocab_eos(vocab);

    for (int i = 0; i < n_gen; ++i) {
        float* logits = llama_get_logits(ctx);
        llama_token next = 0;
        float best = logits[0];
        for (int32_t j = 1; j < n_vocab; ++j) {
            if (logits[j] > best) { best = logits[j]; next = j; }
        }
        if (next == eos) { fprintf(stdout, " [EOS]"); break; }

        char piece[256] = {};
        int n = llama_token_to_piece(vocab, next, piece, sizeof(piece)-1, 0, false);
        if (n > 0) { piece[n] = '\0'; fprintf(stdout, "%s", piece); fflush(stdout); }

        llama_batch nb = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, nb) != 0) {
            fprintf(stderr, "\n[error] decode failed at step %d\n", i);
            break;
        }
    }
    fprintf(stdout, "\n\n[llm-tracer] done ✓\n");

    // ── 9. Cleanup ──────────────────────────────────────────────────────────
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
