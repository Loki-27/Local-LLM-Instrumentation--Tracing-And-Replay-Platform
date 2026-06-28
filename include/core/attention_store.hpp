#pragma once
/*
 * include/core/attention_store.hpp
 *
 * Stores the single most recently captured attention weight matrix.
 *
 * Why only one matrix?
 *   A full 8B-parameter model can have 32 layers × 32 heads × 512² tokens
 *   of attention weights. Storing all of them would be gigabytes. Instead,
 *   the user selects a "capture target" layer in Panel 1 (via Space), and
 *   HookManager calls store() only when that layer fires. Panel 3 then reads
 *   it via get().
 *
 * Thread model:
 *   store() is called from the inference thread (inside the ggml eval callback).
 *   get()   is called from the TUI main thread (inside the render loop).
 *   A single std::mutex protects both. The critical section is a vector copy —
 *   short enough that there's no meaningful contention.
 *
 * Matrix format:
 *   matrix[row][col] = attention weight from query token `row` to key token `col`.
 *   Values are in [0, 1] after softmax. Diagonal is typically the highest for
 *   causal (autoregressive) attention.
 */

#include <mutex>
#include <string>
#include <vector>

class AttentionStore {
public:
    // Snapshot returned to the TUI — all fields are copies, safe to use off-thread
    struct Snapshot {
        std::vector<std::vector<float>> matrix;        // [n_query][n_key], values ∈ [0,1]
        std::vector<std::string>        token_labels;  // Display labels for row/col headers
        std::string                     layer_name;    // e.g. "blk.1.attn_q"
        int                             head_idx = 0;  // Which attention head this is
        bool                            valid    = false; // false if store() has never been called
    };

    // ── Write (inference thread) ──────────────────────────────────────────────

    // Replace the stored matrix with a new one.
    // Moves data in — the caller's vectors are left empty after this call.
    void store(
        std::vector<std::vector<float>> matrix,
        std::vector<std::string>        token_labels,
        std::string                     layer_name,
        int                             head_idx = 0
    );

    // ── Read (TUI thread) ─────────────────────────────────────────────────────

    // Returns a full copy of the current snapshot.
    // If store() has never been called, Snapshot::valid == false.
    Snapshot get() const;

    // Lightweight check before calling get() — avoids the copy when Panel 3 is
    // not focused and doesn't need to render
    bool has_data() const {
        std::lock_guard<std::mutex> lk(mu_);
        return has_data_;
    }

    // Reset to empty state (e.g. when model is reloaded)
    void clear();

private:
    std::vector<std::vector<float>> matrix_;
    std::vector<std::string>        token_labels_;
    std::string                     layer_name_;
    int                             head_idx_ = 0;
    bool                            has_data_ = false;
    mutable std::mutex              mu_;
};