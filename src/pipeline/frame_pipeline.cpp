// ============================================================
// frame_pipeline.cpp  –  FramePipeline implementation
// DMS Euro NCAP 2026 PoC  –  Step 2
// ============================================================

#include "frame_pipeline.hpp"

#include <iostream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace dms {

// ---- Constructor / Destructor --------------------------------

FramePipeline::FramePipeline(int camera_index, FrameCallback on_frame)
    : camera_index_(camera_index), on_frame_(std::move(on_frame)) {}

FramePipeline::~FramePipeline() {
  stop();
}

// ---- Public API ----------------------------------------------

void FramePipeline::start() {
  if (running_.exchange(true)) {
    return; // already running
  }
  producer_thread_ = std::thread(&FramePipeline::producerLoop, this);
  consumer_thread_ = std::thread(&FramePipeline::consumerLoop, this);
}

void FramePipeline::stop() {
  if (!running_.exchange(false)) {
    return; // was not running
  }
  if (producer_thread_.joinable()) producer_thread_.join();
  if (consumer_thread_.joinable()) consumer_thread_.join();

  std::cout << "[FramePipeline] Stopped.\n"
            << "  Captured : " << captured_.load()  << "\n"
            << "  Processed: " << processed_.load() << "\n"
            << "  Dropped  : " << dropped_.load()   << "\n";
}

// ---- Producer (capture) thread -------------------------------

void FramePipeline::producerLoop() {
  cv::VideoCapture cap(camera_index_, cv::CAP_DSHOW);
  if (!cap.isOpened()) {
    // Fallback: try without backend hint
    cap.open(camera_index_);
  }
  if (!cap.isOpened()) {
    std::cerr << "[FramePipeline] ERROR: Cannot open camera " << camera_index_ << "\n";
    running_.store(false);
    return;
  }

  // Request 30 fps from the driver (best-effort)
  cap.set(cv::CAP_PROP_FPS, 30.0);

  cv::Mat frame;
  while (running_.load(std::memory_order_relaxed)) {
    if (!cap.read(frame) || frame.empty()) {
      std::cerr << "[FramePipeline] WARNING: Empty frame, skipping.\n";
      continue;
    }
    ++captured_;

    // Try to push; if ring is full, drop oldest attempt
    if (!ring_.push(frame.clone())) {
      ++dropped_;
    }
  }
}

// ---- Consumer (process/display) thread -----------------------

void FramePipeline::consumerLoop() {
  while (running_.load(std::memory_order_relaxed)) {
    auto maybe = ring_.pop();
    if (!maybe) {
      // Buffer empty – yield to avoid busy-spin
      std::this_thread::sleep_for(std::chrono::microseconds(500));
      continue;
    }

    cv::Mat& frame = *maybe;
    ++processed_;

    // Invoke user callback; if it returns false, request pipeline stop
    if (!on_frame_(frame)) {
      running_.store(false);
    }
  }
}

} // namespace dms
