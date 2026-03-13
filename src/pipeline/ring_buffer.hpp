#pragma once

// ============================================================
// ring_buffer.hpp  –  Single-Producer / Single-Consumer
// lock-free ring buffer for DMS Euro NCAP 2026 PoC
// Step 2: multi-threaded capture pipeline
// ============================================================

#include <array>
#include <atomic>
#include <cstddef>
#include <optional>

namespace dms {

/**
 * @brief SPSC lock-free ring buffer.
 *
 * Capacity must be a power-of-two.  The extra slot technique is used so
 * that full/empty states are always distinguishable without a size counter.
 *
 * Thread safety: push() may only be called from ONE thread (producer).
 *                pop()  may only be called from ONE thread (consumer).
 *
 * @tparam T      Element type (must be move-constructible).
 * @tparam N      Ring capacity (power-of-two, e.g. 8, 16, 32).
 */
template <typename T, std::size_t N>
class RingBuffer {
  static_assert((N & (N - 1)) == 0, "RingBuffer capacity must be power-of-two");

public:
  RingBuffer() : head_(0), tail_(0) {}

  // ---- Producer side ----

  /**
   * @brief Try to push an element.  Non-blocking.
   * @return true on success, false if buffer is full (oldest frame dropped by caller).
   */
  bool push(T item) noexcept {
    const std::size_t h = head_.load(std::memory_order_relaxed);
    const std::size_t next_h = advance(h);
    if (next_h == tail_.load(std::memory_order_acquire)) {
      return false; // full
    }
    buf_[h] = std::move(item);
    head_.store(next_h, std::memory_order_release);
    return true;
  }

  // ---- Consumer side ----

  /**
   * @brief Try to pop an element.  Non-blocking.
   * @return The element wrapped in std::optional, or std::nullopt if empty.
   */
  std::optional<T> pop() noexcept {
    const std::size_t t = tail_.load(std::memory_order_relaxed);
    if (t == head_.load(std::memory_order_acquire)) {
      return std::nullopt; // empty
    }
    T item = std::move(buf_[t]);
    tail_.store(advance(t), std::memory_order_release);
    return item;
  }

  // ---- Queries (approximate, not synchronised) ----
  bool empty() const noexcept {
    return head_.load(std::memory_order_relaxed) ==
           tail_.load(std::memory_order_relaxed);
  }

  bool full() const noexcept {
    return advance(head_.load(std::memory_order_relaxed)) ==
           tail_.load(std::memory_order_relaxed);
  }

  static constexpr std::size_t capacity() noexcept { return N - 1; }

private:
  static constexpr std::size_t advance(std::size_t idx) noexcept {
    return (idx + 1) & (N - 1);
  }

  // Pad to separate cache lines – avoids false sharing between producer
  // (writes head_) and consumer (writes tail_).
  alignas(64) std::atomic<std::size_t> head_;
  alignas(64) std::atomic<std::size_t> tail_;
  std::array<T, N> buf_;
};

} // namespace dms
