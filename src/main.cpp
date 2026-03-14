// =============================================================
// main.cpp  -  DMS Euro NCAP 2026 PoC
// Phase 2: ONNX model integration with command-line support
// =============================================================
#include <cstdio>
#include <cstdlib>
#include <string>
#ifdef _WIN32
#include <windows.h>
static void printLastError(const char* ctx) {
    DWORD err = GetLastError();
    char buf[512] = {};
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, nullptr, err, 0, buf, sizeof(buf), nullptr);
    printf("[ERROR] %s: code=%lu %s\n", ctx, err, buf);
    fflush(stdout);
}
#else
static void printLastError(const char*) {}
#endif

#include "pipeline/frame_pipeline.hpp"
#include "dms/dms_processor.hpp"  // DMS processor with ONNX support

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <mutex>
#include <thread>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

static std::mutex g_frame_mutex;
static cv::Mat g_latest_frame;
static std::atomic<bool> g_frame_ready{false};
static std::atomic<bool> g_quit{false};

static void signalHandler(int) { g_quit.store(true); }

// Print usage information
static void printUsage(const char* prog) {
    printf("\nUsage: %s [model_path]\n", prog);
    printf("\nArguments:\n");
    printf("  model_path    Path to ONNX face detection model (optional)\n");
    printf("                Default: ../../../models/face_detection_yunet_2023mar.onnx\n");
    printf("\nExamples:\n");
    printf("  %s                                          # Use default model\n", prog);
    printf("  %s models/face_detection_yunet_2023mar.onnx  # Use custom path\n", prog);
    printf("  %s ""                                         # Stub mode (no detection)\n", prog);
    printf("\nControls:\n");
    printf("  q or ESC      Quit application\n");
    printf("\n");
}

int main(int argc, char* argv[]) {
    printf("[main] ===== DMS Euro NCAP 2026 - Phase 2 starting =====\n");
    fflush(stdout);

    // Parse command-line arguments
    std::string modelPath;
    if (argc > 1) {
        if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        if (std::string(argv[1]) == "--model" && argc > 2) {
            modelPath = argv[2];
            printf("[main] Using model from command line: %s\n", modelPath.c_str());
        } else {
            modelPath = argv[1];
            printf("[main] Using model from command line: %s\n", modelPath.c_str());
                 }
    }
    fflush(stdout);

    // Register signal handlers
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    printf("[main] Signal handlers registered\n");
    fflush(stdout);

    // Configuration
    constexpr int kCamera = 0;
    constexpr char kWindow[] = "DMS Euro NCAP 2026 - Phase 2";

    // Create OpenCV window
    printf("[main] Creating OpenCV window...\n");
    fflush(stdout);
    try {
        cv::namedWindow(kWindow, cv::WINDOW_AUTOSIZE);
    } catch (const cv::Exception& e) {
        printf("[main] cv::namedWindow EXCEPTION: %s\n", e.what());
        fflush(stdout);
        return 1;
    } catch (...) {
        printf("[main] cv::namedWindow UNKNOWN EXCEPTION\n");
        fflush(stdout);
        return 1;
    }
    printf("[main] Window created OK\n");
    fflush(stdout);

    // Initialize DMS processor with ONNX model
    printf("[main] Initializing DMS processor...\n");
    fflush(stdout);
    dms::DmsProcessor dmsProcessor(modelPath);
    printf("[main] DMS processor ready\n");
    fflush(stdout);

    // Consumer callback: deposit frame + run DMS inference
    auto onFrame = [&](cv::Mat& frame) -> bool {
        if (g_quit.load(std::memory_order_relaxed))
            return false;

        // Run DMS inference on consumer thread
        dms::DmsResult result = dmsProcessor.process(frame);

        // Deposit frame into shared slot for main thread display
        {
            std::lock_guard<std::mutex> lk(g_frame_mutex);
            g_latest_frame = frame.clone();
        }
        g_frame_ready.store(true, std::memory_order_release);
        return true;
    };

    // Start pipeline
    printf("[main] Starting pipeline on camera index %d...\n", kCamera);
    fflush(stdout);
    dms::FramePipeline pipeline(kCamera, onFrame);
    pipeline.start();
    printf("[main] Pipeline started. Press q or ESC to quit.\n");
    fflush(stdout);

    // Main thread display loop
    while (!g_quit.load(std::memory_order_relaxed)) {
        if (g_frame_ready.exchange(false, std::memory_order_acquire)) {
            cv::Mat display;
            {
                std::lock_guard<std::mutex> lk(g_frame_mutex);
                display = g_latest_frame.clone();
            }

            if (!display.empty()) {
                // Run DMS inference for overlay rendering
                dms::DmsResult result = dmsProcessor.process(display);

                // Render HMI overlay
                dms::DmsProcessor::renderOverlay(display, result);

                try {
                    cv::imshow(kWindow, display);
                } catch (const cv::Exception& e) {
                    printf("[main] cv::imshow exception: %s\n", e.what());
                    fflush(stdout);
                }
            }
        }

        const int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {
            printf("[main] Exit key pressed.\n");
            fflush(stdout);
            g_quit.store(true);
            break;
        }
    }

    // Cleanup
    printf("[main] Stopping pipeline...\n");
    fflush(stdout);
    pipeline.stop();
    cv::destroyAllWindows();

    printf("[main] Done. Captured=%llu Processed=%llu Dropped=%llu\n",
           (unsigned long long)pipeline.capturedFrames(),
           (unsigned long long)pipeline.processedFrames(),
           (unsigned long long)pipeline.droppedFrames());
    fflush(stdout);

    return 0;
}
