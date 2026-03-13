// =============================================================
// main.cpp  -  DMS Euro NCAP 2026 PoC
// Step 3: DMS HMI overlay integration
// =============================================================

#include <cstdio>
#include <cstdlib>

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
#include "dms/dms_processor.hpp"  // Step 3: DMS overlay

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <mutex>
#include <thread>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

static std::mutex          g_frame_mutex;
static cv::Mat             g_latest_frame;
static std::atomic<bool>   g_frame_ready{false};
static std::atomic<bool>   g_quit{false};

static void signalHandler(int) { g_quit.store(true); }

int main() {
    printf("[main] ===== DMS Euro NCAP 2026 - Step 3 starting =====\n");
    fflush(stdout);

    std::signal(SIGINT,  signalHandler);
    std::signal(SIGTERM, signalHandler);
    printf("[main] Signal handlers registered\n");
    fflush(stdout);

    constexpr int  kCamera  = 0;
    constexpr char kWindow[] = "DMS Euro NCAP 2026 - Step 3";

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

    // ---- Step 3: Instantiate DMS processor ----------------------
    // Pass empty string to skip Haar cascade (overlay-only mode),
    // OR pass path to haarcascade_frontalface_default.xml for face detection.
    dms::DmsProcessor dmsProcessor("");  // stub mode for now

    printf("[main] DMS processor ready\n");
    fflush(stdout);

    // ---- Consumer callback: deposit frame + run DMS inference ---
    auto onFrame = [&](cv::Mat& frame) -> bool {
        if (g_quit.load(std::memory_order_relaxed))
            return false;

        // Step 3: Run DMS inference on consumer thread
        // (renderOverlay is called on main thread later)
        dms::DmsResult result = dmsProcessor.process(frame);

        // Deposit raw frame into shared slot for main thread to display
        {
            std::lock_guard<std::mutex> lk(g_frame_mutex);
            g_latest_frame = frame.clone();
        }
        g_frame_ready.store(true, std::memory_order_release);
        return true;
    };

    printf("[main] Starting pipeline on camera index %d...\n", kCamera);
    fflush(stdout);

    dms::FramePipeline pipeline(kCamera, onFrame);
    pipeline.start();

    printf("[main] Pipeline started. Press q or ESC to quit.\n");
    fflush(stdout);

    // ---- Main thread display loop -------------------------------
    while (!g_quit.load(std::memory_order_relaxed)) {

        if (g_frame_ready.exchange(false, std::memory_order_acquire)) {
            cv::Mat display;
            {
                std::lock_guard<std::mutex> lk(g_frame_mutex);
                display = g_latest_frame.clone();
            }
            if (!display.empty()) {
                // Step 3: Run DMS inference again for overlay
                // (or store result from consumer thread — for PoC we re-run)
                dms::DmsResult result = dmsProcessor.process(display);

                // Step 3: Render HMI overlay
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
            printf("[main] Exit key pressed.\n"); fflush(stdout);
            g_quit.store(true);
            break;
        }
    }

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
