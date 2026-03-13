// ============================================================
// main.cpp  –  DMS Euro NCAP 2026 PoC
// Step 2: Multi-threaded producer/consumer pipeline
// ============================================================
//
// Architecture:
//   main thread     –  initialises pipeline, runs display loop,
//                      handles keyboard quit ('q' / ESC).
//   producer thread –  cv::VideoCapture → ring buffer  (inside FramePipeline)
//   consumer thread –  ring buffer → user callback      (inside FramePipeline)
//
// Build:  cmake -B build -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
//         cmake --build build --config Release
// Run:    .\bin\Release\dms_app.exe
// Press 'q' or ESC to quit.
// ============================================================

#include "pipeline/frame_pipeline.hpp"

#include <atomic>
#include <csignal>
#include <cstdio>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

static std::atomic<bool> g_quit{false};

static void signalHandler(int) { g_quit.store(true); }

int main() {
  std::signal(SIGINT,  signalHandler);
  std::signal(SIGTERM, signalHandler);

  constexpr int kCamera = 0; // default webcam
  constexpr char kWindow[] = "DMS Euro NCAP 2026 – Step 2 (pipeline)";

  cv::namedWindow(kWindow, cv::WINDOW_AUTOSIZE);

  // ---- Consumer callback (called from consumer thread) --------
  // Returns false to stop the pipeline.
  auto onFrame = [&](cv::Mat& frame) -> bool {
    if (g_quit.load(std::memory_order_relaxed)) {
      return false;
    }

    // Overlay FPS text (cheap – just a visual sanity check)
    cv::putText(frame,
                "DMS PoC - Step 2",
                {10, 30},
                cv::FONT_HERSHEY_SIMPLEX,
                0.9,
                {0, 255, 0},
                2);

    cv::imshow(kWindow, frame);

    // Poll for key press on the main/consumer thread
    const int key = cv::waitKey(1);
    if (key == 'q' || key == 27 /* ESC */) {
      g_quit.store(true);
      return false;
    }
    return true;
  };

  // ---- Build and run pipeline ---------------------------------
  dms::FramePipeline pipeline(kCamera, onFrame);
  pipeline.start();

  std::cout << "[main] Pipeline started. Press 'q' or ESC to quit.\n";

  // Block main thread until pipeline stops (consumer returned false
  // or SIGINT received)
  while (!g_quit.load(std::memory_order_relaxed) &&
         pipeline.capturedFrames() == 0 &&
         pipeline.processedFrames() == 0) {
    // just waiting for first frame
  }

  // The pipeline runs until onFrame() returns false or g_quit is set
  // We need to wait for a stop signal ourselves:
  while (!g_quit.load(std::memory_order_relaxed)) {
    // Sleep main thread – all work is in producer/consumer threads
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  pipeline.stop();
  cv::destroyAllWindows();

  std::cout << "[main] Exiting.  Captured=" << pipeline.capturedFrames()
            << "  Processed=" << pipeline.processedFrames()
            << "  Dropped=" << pipeline.droppedFrames() << "\n";
  return 0;
}
