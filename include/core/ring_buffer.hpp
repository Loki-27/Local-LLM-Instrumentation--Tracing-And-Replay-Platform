#pragma once
/*
 * include/core/ring_buffer.hpp
 *
 * Fixed-capacity circular (ring) buffer. Header-only because it's a template.
 *
 * Design decisions:
 *   - Fixed capacity set at compile time via template parameter — no heap
 *     allocation after construction, predictable memory usage.
 *   - When full, push() overwrites the oldest entry (no blocking, no dropping).
 *     This is the "bounded telemetry" contract: you always get the most recent
 *     Cap events, never more, never stalled.
 *   - snapshot() returns a std::vector<T> copy — NOT a reference or iterator.
 *     The TUI thread holds the copy while rendering; the inference thread can
 *     keep pushing without any contention beyond the brief mutex lock.
 *   - std::mutex protects both push() and snapshot(). The critical sections are
 *     short (array write + index update), so contention is negligible.
 *
 * Usage:
 *   using EventBuffer = RingBuffer<LayerEvent, 512>;
 *   EventBuffer ring;
 *
 *   // inference thread:
 *   ring.push(ev);
 *
 *   // TUI thread:
 *   auto snap = ring.snapshot();   // safe copy
 *   for (auto& ev : snap) { ... }
 */

#include <array>
#include <cstddef>
#include <mutex>
#include <vector>

template<typename T, size_t Cap>
class RingBuffer {
    static_assert(Cap > 0, "RingBuffer capacity must be greater than 0");

public:
    // ── Write (inference thread) ──────────────────────────────────────────────

    // Push one item. If the buffer is full, the oldest entry is silently
    // overwritten. Never blocks. O(1).
    void push(T item) {
        std::lock_guard<std::mutex> lk(mu_);
        buf_[head_] = std::move(item);
        head_ = (head_ + 1) % Cap;
        if (count_ < Cap) ++count_;
    }

    // ── Read (TUI thread) ─────────────────────────────────────────────────────

    // Returns all current entries as a vector, in insertion order (oldest first).
    // Thread-safe: takes a full copy under lock so the caller can iterate freely.
    // If the buffer has never been full, returns only the count_ valid entries.
    std::vector<T> snapshot() const {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<T> out;
        out.reserve(count_);
        // Start index of the oldest valid entry
        size_t start = (head_ + Cap - count_) % Cap;
        for (size_t i = 0; i < count_; ++i) {
            out.push_back(buf_[(start + i) % Cap]);
        }
        return out;
    }

    // Returns the most recent N entries (or fewer if buffer has less than N).
    // Useful for Panel 2 which only displays the last ~20 events on screen.
    std::vector<T> latest(size_t n) const {
        std::lock_guard<std::mutex> lk(mu_);
        size_t take  = (n < count_) ? n : count_;
        std::vector<T> out;
        out.reserve(take);
        // Most recent `take` entries end at head_-1
        size_t start = (head_ + Cap - take) % Cap;
        for (size_t i = 0; i < take; ++i) {
            out.push_back(buf_[(start + i) % Cap]);
        }
        return out;
    }

    // ── Metadata ─────────────────────────────────────────────────────────────

    size_t size() const {
        std::lock_guard<std::mutex> lk(mu_);
        return count_;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lk(mu_);
        return count_ == 0;
    }

    void clear() {
        std::lock_guard<std::mutex> lk(mu_);
        count_ = 0;
        head_  = 0;
    }

    // Compile-time capacity — no lock needed
    static constexpr size_t capacity() { return Cap; }

private:
    std::array<T, Cap>  buf_;
    size_t              head_  = 0;   // Index where the NEXT write goes
    size_t              count_ = 0;   // Number of valid entries currently stored
    mutable std::mutex  mu_;
};
