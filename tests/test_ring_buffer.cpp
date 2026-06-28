/*
 * tests/test_ring_buffer.cpp
 *
 * Phase 1 smoke test — verifies RingBuffer<T, Cap> and LayerEvent before
 * wiring them into HookManager.
 *
 * Build:  cmake --build build --target test_ring_buffer
 * Run:    ./build/test_ring_buffer
 * Expect: all lines print PASS, exit code 0
 *
 * If any test FAILs, fix the data structure before moving to Phase 2.
 */

 #include "core/ring_buffer.hpp"
 #include "core/layer_event.hpp"
 #include "core/attention_store.hpp"
 
 #include <atomic>
 #include <cassert>
 #include <chrono>
 #include <cstdio>
 #include <string>
 #include <thread>
 #include <vector>
 
 // ── Test runner helpers ───────────────────────────────────────────────────────
 
 static int total = 0, passed = 0;
 
 static void check(const char* name, bool ok) {
     ++total;
     if (ok) {
         ++passed;
         fprintf(stdout, "  PASS  %s\n", name);
     } else {
         fprintf(stdout, "  FAIL  %s   ← fix this before Phase 2\n", name);
     }
 }
 
 // ── Test 1: Basic push and size ───────────────────────────────────────────────
 
 static void test_basic_push() {
     fprintf(stdout, "\n[1] Basic push and size\n");
     RingBuffer<int, 8> buf;
 
     check("empty on construction",     buf.empty());
     check("size() == 0 initially",     buf.size() == 0);
 
     buf.push(10);
     check("size() == 1 after one push", buf.size() == 1);
     check("not empty after push",       !buf.empty());
 
     buf.push(20);
     buf.push(30);
     check("size() == 3 after three pushes", buf.size() == 3);
 }
 
 // ── Test 2: Capacity cap ─────────────────────────────────────────────────────
 
 static void test_capacity_cap() {
     fprintf(stdout, "\n[2] Capacity cap (push more than Cap)\n");
     constexpr size_t CAP = 512;
     RingBuffer<int, CAP> buf;
 
     // Push 600 items — 88 more than capacity
     for (int i = 0; i < 600; ++i) buf.push(i);
 
     check("size() == Cap after overflow",          buf.size() == CAP);
     check("capacity() returns compile-time value", buf.capacity() == CAP);
 }
 
 // ── Test 3: Snapshot order ────────────────────────────────────────────────────
 
 static void test_snapshot_order() {
     fprintf(stdout, "\n[3] Snapshot order (oldest → newest, wrapping)\n");
     constexpr size_t CAP = 8;
     RingBuffer<int, CAP> buf;
 
     // Push 0..14 (CAP=8, so only 7..14 survive)
     for (int i = 0; i < 15; ++i) buf.push(i);
 
     auto snap = buf.snapshot();
     check("snapshot has exactly Cap entries",   snap.size() == CAP);
     check("first entry is oldest surviving (7)", snap.front() == 7);
     check("last entry is newest (14)",           snap.back()  == 14);
 
     // Verify strict ascending order
     bool ordered = true;
     for (size_t i = 1; i < snap.size(); ++i) {
         if (snap[i] != snap[i-1] + 1) { ordered = false; break; }
     }
     check("snapshot is in insertion order", ordered);
 }
 
 // ── Test 4: latest(N) helper ──────────────────────────────────────────────────
 
 static void test_latest() {
     fprintf(stdout, "\n[4] latest(N) returns N most-recent entries\n");
     RingBuffer<int, 100> buf;
     for (int i = 0; i < 50; ++i) buf.push(i);
 
     auto last5 = buf.latest(5);
     check("latest(5) returns 5 items",   last5.size() == 5);
     check("latest(5)[0] is 45th entry",  last5[0] == 45);
     check("latest(5)[4] is 49th entry",  last5[4] == 49);
 
     // Request more than available
     auto allOfThem = buf.latest(200);
     check("latest(200) capped at actual size", allOfThem.size() == 50);
 }
 
 // ── Test 5: clear() resets state ─────────────────────────────────────────────
 
 static void test_clear() {
     fprintf(stdout, "\n[5] clear() resets to empty\n");
     RingBuffer<int, 16> buf;
     for (int i = 0; i < 10; ++i) buf.push(i);
     buf.clear();
 
     check("empty after clear",               buf.empty());
     check("size() == 0 after clear",         buf.size() == 0);
     check("snapshot empty after clear",      buf.snapshot().empty());
 
     // Should be able to push again normally after clear
     buf.push(99);
     check("push works after clear",          buf.size() == 1);
     check("snapshot[0] == 99 after clear",   buf.snapshot()[0] == 99);
 }
 
 // ── Test 6: Thread safety ─────────────────────────────────────────────────────
 
 static void test_thread_safety() {
     fprintf(stdout, "\n[6] Thread safety (two threads pushing concurrently)\n");
     constexpr size_t CAP = 256;
     RingBuffer<int, CAP> buf;
     std::atomic<bool> any_exception{false};
 
     auto pusher = [&](int base) {
         for (int i = 0; i < 1000; ++i) {
             buf.push(base + i);
         }
     };
 
     std::thread t1(pusher, 0);
     std::thread t2(pusher, 100000);
     t1.join();
     t2.join();
 
     // No crash + size is exactly Cap after 2000 pushes into a 256-cap buffer
     bool size_ok  = (buf.size() == CAP);
     auto snap     = buf.snapshot();
     bool snap_ok  = (snap.size() == CAP);
 
     check("no crash under concurrent pushes",     !any_exception.load());
     check("size() == Cap after concurrent pushes", size_ok);
     check("snapshot() returns Cap items",          snap_ok);
 }
 
 // ── Test 7: RingBuffer<LayerEvent> ───────────────────────────────────────────
 
 static void test_layer_event_push() {
     fprintf(stdout, "\n[7] RingBuffer<LayerEvent> — push and retrieve real structs\n");
     using EventBuffer = RingBuffer<LayerEvent, 512>;
     EventBuffer ring;
 
     // Build a realistic LayerEvent
     LayerEvent ev;
     ev.id             = 42;
     ev.layer_name     = "blk.1.attn_q";
     ev.layer_type     = "Attn (Self)";
     ev.compute_device = "CPU";
     ev.timestamp      = std::chrono::system_clock::now();
     ev.latency_ms     = 1.142f;
     ev.shape          = {4096, 32, 1, 1};
     ev.dtype          = "f16";
     ev.stats_valid    = true;
     ev.sparsity_rate  = 0.542f;
     ev.mean           = 0.001f;
     ev.max_val        = 3.7f;
     ev.has_anomaly    = false;
 
     ring.push(ev);
     auto snap = ring.snapshot();
 
     check("snapshot has 1 event",                   snap.size() == 1);
     check("id survives round-trip",                  snap[0].id == 42);
     check("layer_name survives round-trip",          snap[0].layer_name == "blk.1.attn_q");
     check("latency_ms survives round-trip",          snap[0].latency_ms == 1.142f);
     check("sparsity_rate survives round-trip",       snap[0].sparsity_rate == 0.542f);
     check("shape survives round-trip",               snap[0].shape == std::vector<int64_t>{4096,32,1,1});
 }
 
 // ── Test 8: LayerEvent helper methods ────────────────────────────────────────
 
 static void test_layer_event_helpers() {
     fprintf(stdout, "\n[8] LayerEvent display helpers\n");
 
     LayerEvent ev;
     ev.id             = 7;
     ev.layer_name     = "blk.2.ffn_gate";
     ev.layer_type     = "MLP (SwiGLU)";
     ev.compute_device = "CUDA [GPU 0]";
     ev.timestamp      = std::chrono::system_clock::now();
     ev.latency_ms     = 0.312f;
     ev.shape          = {4096, 1, 1, 1};  // 3 trailing 1s should be stripped
     ev.dtype          = "f32";
     ev.stats_valid    = true;
     ev.sparsity_rate  = 0.1f;
     ev.mean           = 0.002f;
     ev.max_val        = 2.1f;
     ev.has_anomaly    = false;
 
     auto ts  = ev.timestamp_str();
     auto shp = ev.shape_str();
     auto sum = ev.summary_str();
 
     // timestamp_str should be "HH:MM:SS.mmm" — 12 chars
     check("timestamp_str is 12 chars",       ts.size() == 12);
     check("timestamp_str has colons",        ts[2] == ':' && ts[5] == ':');
 
     // shape_str should strip trailing 1s: "[4096]" not "[4096, 1, 1, 1]"
     check("shape_str strips trailing 1s",    shp == "[4096]");
 
     // summary_str should contain key fields
     check("summary_str contains id",         sum.find("7") != std::string::npos);
     check("summary_str contains layer_name", sum.find("blk.2.ffn_gate") != std::string::npos);
 
     fprintf(stdout, "         timestamp_str: %s\n", ts.c_str());
     fprintf(stdout, "         shape_str:     %s\n", shp.c_str());
     fprintf(stdout, "         summary_str:   %s\n", sum.c_str());
 }
 
 // ── Test 9: AttentionStore ────────────────────────────────────────────────────
 
 static void test_attention_store() {
     fprintf(stdout, "\n[9] AttentionStore — store and retrieve\n");
     AttentionStore store;
 
     check("has_data() false initially",     !store.has_data());
 
     // Snapshot when empty
     auto empty_snap = store.get();
     check("Snapshot::valid false initially", !empty_snap.valid);
 
     // Store a 4x4 matrix
     std::vector<std::vector<float>> mat = {
         {0.9f, 0.05f, 0.03f, 0.02f},
         {0.1f, 0.80f, 0.06f, 0.04f},
         {0.1f, 0.10f, 0.70f, 0.10f},
         {0.1f, 0.10f, 0.10f, 0.70f},
     };
     std::vector<std::string> labels = {"[I]", "[want]", "[it]", "[to]"};
     store.store(mat, labels, "blk.1.attn_q", 0);
 
     check("has_data() true after store",    store.has_data());
 
     auto snap = store.get();
     check("Snapshot::valid true after store",         snap.valid);
     check("Snapshot matrix is 4x4",                   snap.matrix.size() == 4 && snap.matrix[0].size() == 4);
     check("Snapshot[0][0] == 0.9f",                   snap.matrix[0][0] == 0.9f);
     check("Snapshot layer_name correct",              snap.layer_name == "blk.1.attn_q");
     check("Snapshot token_labels[1] == \"[want]\"",   snap.token_labels[1] == "[want]");
 
     // Clear
     store.clear();
     check("has_data() false after clear",  !store.has_data());
     check("Snapshot invalid after clear",  !store.get().valid);
 }
 
 // ── Test 10: AttentionStore thread safety ─────────────────────────────────────
 
 static void test_attention_store_threads() {
     fprintf(stdout, "\n[10] AttentionStore concurrent store + get\n");
     AttentionStore store;
     std::atomic<bool> running{true};
     std::atomic<int>  get_count{0};
 
     // Reader thread: repeatedly calls get()
     std::thread reader([&] {
         while (running.load()) {
             auto snap = store.get();  // must not crash
             ++get_count;
         }
     });
 
     // Writer: stores 100 matrices
     for (int i = 0; i < 100; ++i) {
         std::vector<std::vector<float>> m = {{float(i), 0.0f}, {0.0f, float(i)}};
         std::vector<std::string> lbs = {"a", "b"};
         store.store(m, lbs, "blk." + std::to_string(i) + ".attn_q", 0);
     }
 
     running.store(false);
     reader.join();
 
     check("no crash under concurrent store+get", true);   // reaching here = no crash
     check("reader thread ran (get_count > 0)",   get_count.load() > 0);
 }
 
 // ── Main ──────────────────────────────────────────────────────────────────────
 
 int main() {
     fprintf(stdout, "─────────────────────────────────────────────────────────\n");
     fprintf(stdout, " Phase 1 smoke test — RingBuffer + LayerEvent + AttentionStore\n");
     fprintf(stdout, "─────────────────────────────────────────────────────────\n");
 
     test_basic_push();
     test_capacity_cap();
     test_snapshot_order();
     test_latest();
     test_clear();
     test_thread_safety();
     test_layer_event_push();
     test_layer_event_helpers();
     test_attention_store();
     test_attention_store_threads();
 
     fprintf(stdout, "\n─────────────────────────────────────────────────────────\n");
     fprintf(stdout, " Results: %d / %d tests passed\n", passed, total);
     fprintf(stdout, "─────────────────────────────────────────────────────────\n");
 
     if (passed == total) {
         fprintf(stdout, " ✓  All tests passed — ready for Phase 2\n\n");
         return 0;
     } else {
         fprintf(stdout, " ✗  %d test(s) failed — fix before Phase 2\n\n", total - passed);
         return 1;
     }
 }