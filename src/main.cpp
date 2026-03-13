// ============================================================
// main.cpp  –  DMS Euro NCAP 2026 PoC
// Step 2: Multi-threaded producer/consumer pipeline
// ============================================================
//
// THREADING MODEL (Windows-safe):
//   producer thread  –  cv::VideoCapture → ring buffer
//   consumer thread  –  ring buffer → deposits frame into shared slot
//   main thread      –  picks up frame from shared slot, calls
//                        cv::imshow + cv::waitKey  <-- MUST be main thread on Windows
//
// Key fix: cv::imshow/waitKey called ONLY from main thread.
// ============================================================

#include "pipeline/frame_pipeline.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <mutex>
#include <thread>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// ---- Shared frame slot (consumer thread writes, main thread reads) ----
static std::mutex       g_frame_mutex;
static cv::Mat          g_latest_frame;
static std::atomic<bool> g_frame_ready{false};
static std::atomic<bool> g_quit{false};

static void signalHandler(int) { g_quit.store(true); }

int main() {
  std::signal(SIGINT,  signalHandler);
  std::signal(SIGTERM, signalHandler);

  constexpr int  kCamera = 0;
  constexpr char kWindow[] = "DMS Euro NCAP 2026 - Step 2";

  // Create window on main thread (required on Windows)
  cv::namedWindow(kWindow, cv::WINDOW_AUTOSIZE);

  // ---- Consumer callback: runs on consumer thread ------------------
  // MUST NOT call cv::imshow here on Windows.
  // Just deposit the frame into the shared slot.
  auto onFrame = [&](cv::Mat& frame) -> bool {
    if (g_quit.load(std::memory_order_relaxed)) {
      return false;
    }
    // Add overlay on consumer thread (pure pixel math – safe)
    cv::putText(frame,
                "DMS PoC - Step 2",
                {10, 30},
                cv::FONT_HERSHEY_SIMPLEX,
                0.9, {0, 255, 0}, 2);

    // Deposit into shared slot
    {
      std::lock_guard<std::mutex> lock(g_frame_mutex);
      g_latest_frame = frame.clone();
    }
    g_frame_ready.store(true, std::memory_order_release);
    return true;
  };

  // ---- Start pipeline ----------------------------------------------
  dms::FramePipeline pipeline(kCamera, onFrame);
  pipeline.start();
  std::cout << "[main] Pipeline started. Press 'q' or ESC to quit.\n";
  std::cout << "[main] Waiting for first frame...\n";

  // ---- Main thread display loop ------------------------------------
  // cv::imshow and cv::waitKey MUST run here (main thread).
  while (!g_quit.load(std::memory_order_relaxed)) {

    if (g_frame_ready.exchange(false, std::memory_order_acquire)) {
      cv::Mat display;
      {
        std::lock_guard<std::mutex> lock(g_frame_mutex);
        display = g_latest_frame.clone();
      }

      if (!display.empty()) {
        cv::imshow(kWindow, display);
      }
    }

    // waitKey drives the HighGUI event loop; 1 ms timeout
    const int key = cv::waitKey(1);
    if (key == 'q' || key == 'Q' || key == 27 /* ESC */) {
      std::cout << "[main] Exit key pressed.\n";
      g_quit.store(true);
      break;
    }
  }

  // ---- Shutdown ----------------------------------------------------
  pipeline.stop();
  cv::destroyAllWindows();

  std::cout << "[main] Exiting.\n"
            << "  Captured : " << pipeline.capturedFrames()  << "\n"
            << "  Processed: " << pipeline.processedFrames() << "\n"
            << "  Dropped  : " << pipeline.droppedFrames()   << "\n";
  return 0;
}
