/*
 * tests/test_hook_dump.cpp  —  Phase 2: Hook ground truth dump
 *
 * Wires HookManager into real inference and prints every LayerEvent captured.
 * This output is the ground truth for:
 *   - What tensor names the callback actually fires for
 *   - How many events per token
 *   - Whether stats_valid is true (Metal prevents float access)
 *   - Which layer types actually appear
 *
 * Build:  cmake --build build --target test_hook_dump
 * Run:    ./build/test_hook_dump 2>/dev/null > tests/hook_dump.txt
 *         (stderr=llama noise goes to terminal, stdout=events go to file)
 *
 * Commit tests/hook_dump.txt — it documents the ground truth for reviewers.
 */

#include "core/ring_buffer.hpp"
#include "core/attention_store.hpp"
#include "hook/hook_manager.hpp"
#include "hook/anomaly_detector.hpp"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "llama.h"

int main(int argc, char** argv) {
    std::string model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    int n_gen = 3;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--model")    && i+1<argc) model_path = argv[++i];
        if (!strcmp(argv[i], "--n-tokens") && i+1<argc) n_gen      = std::stoi(argv[++i]);
    }

    // Data structures
    EventBuffer    ring;
    AttentionStore attn_store;
    HookManager    hook(ring, attn_store);

    // Load model
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "FAIL — could not load: %s\n", model_path.c_str());
        return 1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);

    // Create context WITH hook — must call attach_via_params BEFORE llama_init_from_model
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;
    hook.attach_via_params(cparams);

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "FAIL — could not create context\n");
        llama_model_free(model); llama_backend_free(); return 1;
    }

    // Tokenize + prefill
    const char* prompt = "The sky is";
    std::vector<llama_token> tokens(512);
    int n = llama_tokenize(vocab, prompt, (int)strlen(prompt),
                           tokens.data(), 512, true, false);
    tokens.resize(n);
    llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "FAIL — prefill failed\n");
        llama_free(ctx); llama_model_free(model); llama_backend_free(); return 1;
    }

    // Generate tokens so the hook fires multiple times
    const int32_t   nv  = llama_vocab_n_tokens(vocab);
    const llama_token eos = llama_vocab_eos(vocab);
    for (int i = 0; i < n_gen; ++i) {
        float* logits = llama_get_logits(ctx);
        llama_token next = 0; float best = logits[0];
        for (int32_t j = 1; j < nv; ++j) if (logits[j]>best){best=logits[j];next=j;}
        if (next == eos) break;
        llama_batch nb = llama_batch_get_one(&next, 1);
        llama_decode(ctx, nb);
    }

    // Print all captured events
    auto events = ring.snapshot();

    fprintf(stdout,
        "=======================================================================\n"
        " Hook dump: %zu LayerEvents (prompt + %d gen tokens) | GPU: %s\n"
        "=======================================================================\n",
        events.size(), n_gen,
        hook.is_gpu_run() ? "YES" : "NO (CPU)");

    fprintf(stdout, " %-4s  %-36s  %-14s  %-12s  %8s  %s\n",
            "ID", "TENSOR NAME", "TYPE", "DEVICE", "LAT(ms)", "SHAPE");
    fprintf(stdout, "%s\n", std::string(100, '-').c_str());

    for (auto& ev : events) {
        // Trim long names
        std::string nm = ev.layer_name;
        if (nm.size() > 36) nm = nm.substr(0,33) + "...";

        fprintf(stdout, " %-4llu  %-36s  %-14s  %-12s  %7.3f  %s%s\n",
            (unsigned long long)ev.id,
            nm.c_str(),
            ev.layer_type.c_str(),
            ev.compute_device.c_str(),
            ev.latency_ms,
            ev.shape_str().c_str(),
            ev.has_anomaly ? ("  ANOMALY: " + ev.anomaly_msg).c_str() : "");
    }

    // Breakdown
    std::map<std::string,int> type_counts;
    size_t stats_ok = 0, anomalies = 0;
    for (auto& ev : events) {
        type_counts[ev.layer_type]++;
        if (ev.stats_valid) stats_ok++;
        if (ev.has_anomaly) anomalies++;
    }

    fprintf(stdout,
        "\n--- Layer type breakdown ---\n");
    for (auto& [t,c] : type_counts)
        fprintf(stdout, "  %-16s %d\n", t.c_str(), c);

    fprintf(stdout,
        "\nTotal events  : %zu\n"
        "Stats valid   : %zu  (0 = all on GPU, expected for Metal)\n"
        "Anomalies     : %zu\n"
        "\nNext steps:\n"
        "  1. Any tensor showing 'Other' type? Update classify_layer() in hook_manager.cpp\n"
        "  2. Events/token = total / (n_prompt_tokens + n_gen) — check ring buffer capacity\n"
        "  3. git add tests/hook_dump.txt && git commit -m 'chore: add hook ground truth'\n",
        events.size(), stats_ok, anomalies);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
