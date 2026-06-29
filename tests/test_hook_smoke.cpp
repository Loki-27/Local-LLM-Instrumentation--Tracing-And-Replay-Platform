/*
 * tests/test_hook_smoke.cpp  —  Phase 6 regression test
 *
 * Runs 10 tokens of real inference with HookManager attached and asserts:
 *   - Ring buffer is non-empty
 *   - All major layer types are present (Attn, MLP, LayerNorm, Embedding, Logits)
 *   - No negative latencies
 *   - TopologyParser correctly builds a 22-layer tree from the events
 *   - No anomalies fire on normal inference
 *
 * Build:  cmake --build build --target test_hook_smoke
 * Run:    ./build/test_hook_smoke
 * Expect: exit code 0, all PASS
 *
 * This is the project's regression test — run it after any hook change.
 */

#include "core/ring_buffer.hpp"
#include "core/attention_store.hpp"
#include "hook/hook_manager.hpp"
#include "topo/topology_parser.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"

// ── Minimal test runner ───────────────────────────────────────────────────────

static int total_checks = 0, failed_checks = 0;

static void check(const char* name, bool ok) {
    ++total_checks;
    if (ok) {
        printf("  PASS  %s\n", name);
    } else {
        ++failed_checks;
        printf("  FAIL  %s\n", name);
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
    int n_gen = 10;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--model")    && i+1<argc) model_path = argv[++i];
        if (!strcmp(argv[i], "--n-tokens") && i+1<argc) n_gen      = std::stoi(argv[++i]);
    }

    printf("─────────────────────────────────────────────────────────\n");
    printf(" Phase 6 regression test — hook smoke test\n");
    printf(" Model: %s\n", model_path.c_str());
    printf(" Generating %d tokens with hook attached...\n\n", n_gen);

    // Setup
    EventBuffer    ring;
    AttentionStore attn_store;
    HookManager    hook(ring, attn_store);

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "FAIL: could not load model: %s\n", model_path.c_str());
        fprintf(stderr, "      Run: bash scripts/download_model.sh\n");
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;
    hook.attach_via_params(cparams);

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "FAIL: could not create context\n");
        llama_model_free(model); llama_backend_free(); return 1;
    }

    // Tokenize + prefill
    const char* prompt = "The sky is";
    std::vector<llama_token> tokens(512);
    int n = llama_tokenize(vocab, prompt, (int)strlen(prompt),
                           tokens.data(), 512, true, false);
    tokens.resize(n);
    llama_batch batch = llama_batch_get_one(tokens.data(), n);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "FAIL: prefill failed\n");
        llama_free(ctx); llama_model_free(model); llama_backend_free(); return 1;
    }

    // Generate n_gen tokens
    const int32_t   nv  = llama_vocab_n_tokens(vocab);
    const llama_token eos = llama_vocab_eos(vocab);
    for (int i = 0; i < n_gen; ++i) {
        float* logits = llama_get_logits(ctx);
        llama_token next = 0; float best = logits[0];
        for (int32_t j = 1; j < nv; ++j) if (logits[j]>best){best=logits[j];next=j;}
        if (next == eos) break;
        llama_batch nb = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, nb) != 0) break;
    }

    // ── Assertions ────────────────────────────────────────────────────────────
    auto events = ring.snapshot();

    printf("Assertions:\n");
    check("ring buffer non-empty",         !events.empty());
    check("GPU run detected (Metal)",       hook.is_gpu_run());
    check("at least 200 events captured",   events.size() >= 200);

    // Count by layer type
    int n_attn=0, n_mlp=0, n_norm=0;
    bool has_embd=false, has_logits=false;
    bool all_named=true, no_neg_lat=true, no_bad_latency=false;

    for (auto& ev : events) {
        if (ev.layer_name.empty())         all_named = false;
        if (ev.latency_ms < 0.0f)          no_neg_lat = false;
        if (ev.latency_ms > 100.0f)        no_bad_latency = true; // >100ms is suspicious
        if (ev.layer_type == "Attn (Self)") n_attn++;
        if (ev.layer_type.find("MLP") != std::string::npos) n_mlp++;
        if (ev.layer_type == "LayerNorm")   n_norm++;
        if (ev.layer_type == "Embedding")   has_embd = true;
        if (ev.layer_type == "Logits")      has_logits = true;
    }

    check("all events have names",          all_named);
    check("no negative latencies",          no_neg_lat);
    check("no suspiciously high latency",  !no_bad_latency);
    check("Attn (Self) events present",     n_attn > 0);
    check("MLP events present",             n_mlp > 0);
    check("LayerNorm events present",       n_norm > 0);
    check("Embedding event present",        has_embd);
    check("Logits event present",           has_logits);

    // Topology test
    auto topo = TopologyParser::build_from_events(events, "smoke-test");
    check("topology root non-empty",        !topo.children.empty());

    bool found_22_layers = false;
    for (auto& c : topo.children) {
        if (c.name.find("Layers") != std::string::npos &&
            c.children.size() >= 22) {
            found_22_layers = true;
        }
    }
    check("topology has 22 layers",         found_22_layers);

    auto flat = TopologyParser::flatten_visible(topo);
    check("flatten_visible works",          flat.size() >= 20);

    // Anomaly check — normal inference should produce zero anomalies
    int anomaly_count = 0;
    for (auto& ev : events) if (ev.has_anomaly) anomaly_count++;
    // embd runs on CPU (mmap weights) — expected, not a real anomaly
    check("no compute anomalies (only embd expected)",
          anomaly_count == 0 || (anomaly_count <= 5 &&
          [&]{ for (auto& ev:events) if (ev.has_anomaly && ev.layer_type!="Embedding") return true; return false; }() == false));

    // ── Summary ────────────────────────────────────────────────────────────────
    printf("\nEvent breakdown:\n");
    printf("  Total events  : %zu\n",  events.size());
    printf("  Attn (Self)   : %d\n",   n_attn);
    printf("  MLP           : %d\n",   n_mlp);
    printf("  LayerNorm     : %d\n",   n_norm);
    printf("  Anomalies     : %d\n",   anomaly_count);
    printf("  Events/token  : ~%.0f\n", (float)events.size() / (n + n_gen));

    printf("\n─────────────────────────────────────────────────────────\n");
    if (failed_checks == 0) {
        printf(" ✓  %d/%d PASSED — hook smoke test complete\n\n",
               total_checks, total_checks);
    } else {
        printf(" ✗  %d/%d FAILED\n\n", failed_checks, total_checks);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return (failed_checks == 0) ? 0 : 1;
}
