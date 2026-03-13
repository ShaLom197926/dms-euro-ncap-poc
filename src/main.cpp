// =============================================================
// main.cpp  -  DMS Euro NCAP 2026 PoC
// Step 2: Multi-Threaded producer/consumer pipeline
// =============================================================
//
// THREADING MODEL (Windows-safe):
//   producer thread  - cv::VideoCapture + ring buffer
//   consumer thread  - ring buffer -> deposits frame into shared slot
//   main thread      - picks up frame from shared slot, calls
//                      cv::imshow + cv::waitKey  <-- MUST be main thread on Windows
//
// Key fix: cv::imshow/waitKey called ONLY from main thread.
// DLL fix:  CMakeLists copies OpenCV DLLs next to EXE at build time.
// =============================================================

// --- Sanity check: if this printf never appears, the EXE crashes before main() ---
// (indicates missing DLL; run 'where opencv_world*.dll' or check PATH)

#include <cstdio>
#include <cstdlib>

// Windows-specific: print last error as string
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

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <mutex>
#include <thread>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// ---- Shared frame slot (consumer thread writes, main thread reads) ----
static std::mutex          g_frame_mutex;
static cv::Mat             g_latest_frame;
static std::atomic<bool>   g_frame_ready{false};
static std::atomic<bool>   g_quit{false};

static void signalHandler(int) { g_quit.store(true); }

int main() {
    printf("[main] ===== DMS Euro NCAP 2026 - Step 2 starting =====\n");
    fflush(stdout);

    // Register signal handlers
    std::signal(SIGINT,  signalHandler);
    std::signal(SIGTERM, signalHandler);

    printf("[main] Signal handlers registered\n");
    fflush(stdout);

    constexpr int  kCamera  = 0;
    constexpr char kWindow[] = "DMS Euro NCAP 2026 - Step 2";

    // Create window on main thread (required on Windows)
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

    // ---- Consumer callback: runs on consumer thread ------------------
    // MUST NOT call cv::imshow here on Windows.
    // Just deposit the frame into the shared slot.
    auto onFrame = [&](cv::Mat& frame) -> bool {
        if (g_quit.load(std::memory_order_relaxed))
            return false;

        // Deposit into shared slot for main thread to display
        {
            std::lock_guard<std::mutex> lk(g_frame_mutex);
            g_latest_frame = frame.clone();
        }
        g_frame_ready.store(true, std::memory_order_release);
        return true;
    };

    // ---- Start pipeline ---------------------------------------------
    printf("[main] Starting pipeline on camera index %d...\n", kCamera);
    fflush(stdout);

    dms::FramePipeline pipeline(kCamera, onFrame);
    pipeline.start();

    printf("[main] Pipeline started. Press q or ESC to quit.\n");
    fflush(stdout);

    // ---- Main thread display loop (Windows: imshow MUST be here) ----
    while (!g_quit.load(std::memory_order_relaxed)) {

        if (g_frame_ready.exchange(false, std::memory_order_acquire)) {
            cv::Mat display;
            {
                std::lock_guard<std::mutex> lk(g_frame_mutex);
                display = g_latest_frame.clone();
            }
            if (!display.empty()) {
                try {
                    cv::imshow(kWindow, display);
                } catch (const cv::Exception& e) {
                    printf("[main] cv::imshow exception: %s\n", e.what());
                    fflush(stdout);
                }
            }
        }

        // waitKey pumps the Win32 message queue; 1 ms timeout
        const int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {
            printf("[main] Exit key pressed.\n"); fflush(stdout);
            g_quit.store(true);
            break;
        }
    }

    // ---- Shutdown ---------------------------------------------------
    printf("[main] Stopping pipeline...\n"); fflush(stdout);
    pipeline.stop();
    cv::destroyAllWindows();

    printf("[main] Done. Captured=%llu  Processed=%llu  Dropped=%llu\n",
           (unsigned long long)pipeline.capturedFrames(),
           (unsigned long long)pipeline.processedFrames(),
           (unsigned long long)pipeline.droppedFrames());
    fflush(stdout);

    return 0;
}
