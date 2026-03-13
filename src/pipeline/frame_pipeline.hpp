#pragma once

// ============================================================
// frame_pipeline.hpp  –  Producer / Consumer thread management
// DMS Euro NCAP 2026 PoC  –  Step 2
// ============================================================
//
// Architecture:
//   Producer thread  –  grabs frames from cv::VideoCapture at
//                       camera FPS and pushes into ring buffer.
//   Consumer thread  –  pops frames and calls a user-supplied
//                       callback (e.g. display / inference).
//
// If the ring buffer is full the producer drops the frame and
// increments a drop counter (no blocking, no frame tearing).
// ============================================================

#include "ring_buffer.hpp"

#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <thread>

#include <opencv2/videoio.hpp>
#include <opencv2/core/mat.hpp>

namespace dms {

// Ring buffer capacity (power-of-two).  8 slots ~= 266 ms @ 30 fps.
static constexpr std::size_t kRingCap = 8;

/**
 * @brief Manages the producer (capture) and consumer (process) threads.
 *
 * Usage:
 *   FramePipeline pipeline(cameraIndex, callback);
 *   pipeline.start();
 *   // ...run until user presses 'q'...
 *   pipeline.stop();
 */
class FramePipeline {
public:
  using FrameCallback = std::function<bool(cv::Mat&)>;
  // callback returns false  →  pipeline stops

  /**
   * @param camera_index  OpenCV camera index (0 = default webcam)
   * @param on_frame      Called on every captured frame from consumer thread.
   *                      Must return true to continue, false to stop pipeline.
   */
  explicit FramePipeline(int camera_index, FrameCallback on_frame);
  ~FramePipeline();

  // Non-copyable, non-movable
  FramePipeline(const FramePipeline&) = delete;
  FramePipeline& operator=(const FramePipeline&) = delete;

  void start();
  void stop();

  // Diagnostics
  uint64_t droppedFrames()  const noexcept { return dropped_.load(); }
  uint64_t capturedFrames() const noexcept { return captured_.load(); }
  uint64_t processedFrames()const noexcept { return processed_.load(); }

private:
  void producerLoop();
  void consumerLoop();

  int                              camera_index_;
  FrameCallback                    on_frame_;
  RingBuffer<cv::Mat, kRingCap>    ring_;
  std::atomic<bool>                running_{false};
  std::thread                      producer_thread_;
  std::thread                      consumer_thread_;
  std::atomic<uint64_t>            dropped_{0};
  std::atomic<uint64_t>            captured_{0};
  std::atomic<uint64_t>            processed_{0};
};

} // namespace dms
