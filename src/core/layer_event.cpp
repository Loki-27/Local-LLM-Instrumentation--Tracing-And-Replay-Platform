/*
 * src/core/layer_event.cpp
 *
 * Implements the display-helper methods declared in layer_event.hpp.
 * These are used by TUI panels in Phase 5. No llama.cpp dependency here —
 * just string formatting and chrono arithmetic.
 */

 #include "core/layer_event.hpp"

 #include <cstdio>
 #include <ctime>
 #include <sstream>
 
 // ── timestamp_str ─────────────────────────────────────────────────────────────
 
 std::string LayerEvent::timestamp_str() const {
     // Extract wall-clock time and millisecond sub-second component
     auto t  = std::chrono::system_clock::to_time_t(timestamp);
     auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                   timestamp.time_since_epoch()) % 1000;
 
     std::tm tm_info = {};
 #if defined(_WIN32)
     localtime_s(&tm_info, &t);
 #else
     localtime_r(&t, &tm_info);
 #endif
 
     char buf[16];
     std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03d",
                   tm_info.tm_hour,
                   tm_info.tm_min,
                   tm_info.tm_sec,
                   static_cast<int>(ms.count()));
     return buf;
 }
 
 // ── shape_str ─────────────────────────────────────────────────────────────────
 
 std::string LayerEvent::shape_str() const {
     if (shape.empty()) return "[]";
 
     // GGML always provides 4 dimensions. Unused dims are padded with 1.
     // Find the last dimension that isn't 1 so we don't print "[4096, 1, 1, 1]".
     int last = 0;
     for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
         if (shape[i] != 1) { last = i; break; }
     }
 
     std::string s = "[";
     for (int i = 0; i <= last; ++i) {
         if (i > 0) s += ", ";
         s += std::to_string(shape[i]);
     }
     s += "]";
     return s;
 }
 
 // ── summary_str ───────────────────────────────────────────────────────────────
 
 std::string LayerEvent::summary_str() const {
     // Fixed-width columns for Panel 2 table alignment
     // Format: "[  42] blk.1.attn_q          | Attn (Self)   | CUDA [GPU 0] | 0.842ms"
     char buf[128];
     std::snprintf(buf, sizeof(buf),
                   "[%4llu] %-28s | %-14s | %-14s | %.3fms",
                   static_cast<unsigned long long>(id),
                   layer_name.c_str(),
                   layer_type.c_str(),
                   compute_device.c_str(),
                   latency_ms);
     return buf;
 }

 