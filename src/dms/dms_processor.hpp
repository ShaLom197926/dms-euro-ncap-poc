#pragma once
// =============================================================
// dms_processor.hpp  -  DMS Euro NCAP 2026 PoC
// Step 3: Stub DMS inference engine + HMI overlay renderer
// =============================================================
// Architecture:
//   process(frame) -> runs stub inference, returns DmsResult
//   renderOverlay(frame, result) -> draws HMI on top of frame
//
// Stub strategy (no ML model yet):
//   - Uses OpenCV Haar cascade for face detection
//   - Derives head-pose proxy from face bounding box position
//   - Simulates PERCLOS via a deterministic sine wave (for PoC demo)
//   - All real inference slots marked TODO for Step 4 (ONNX model)
// =============================================================

#include "dms_types.hpp"
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <string>

namespace dms {

class DmsProcessor {
public:
    // cascade_path: path to haarcascade_frontalface_default.xml
    // Pass empty string to skip face detection (overlay-only mode)
    explicit DmsProcessor(const std::string& cascade_path = "");
    ~DmsProcessor() = default;

    // Run stub inference on a single BGR frame.
    // Thread-safe: may be called from any thread.
    DmsResult process(const cv::Mat& frame);

    // Render the HMI overlay onto 'frame' in-place.
    // Call ONLY from main/display thread.
    static void renderOverlay(cv::Mat& frame, const DmsResult& result);

private:
    cv::CascadeClassifier m_faceCascade;
    bool                  m_hasCascade = false;

    // Rolling frame counter for PERCLOS simulation
    uint64_t m_frameCount = 0;

    // Helpers
    GazeZone  inferGazeZone(const cv::Rect& face, const cv::Size& frameSize) const;
    float     simulatePerclos() const;
    DrowsinessLevel  classifyDrowsiness(float perclos) const;
    DistractionLevel classifyDistraction(GazeZone zone, float offRoadSec) const;

    // Cumulative off-road gaze timer
    float m_offRoadGazeSec  = 0.f;
    bool  m_wasOffRoad      = false;
    std::chrono::steady_clock::time_point m_offRoadStart;
};

} // namespace dms
