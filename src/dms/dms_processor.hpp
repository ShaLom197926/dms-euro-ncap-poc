#pragma once
// =============================================================
// dms_processor.hpp - DMS Euro NCAP 2026 PoC
// Phase 2: ONNX model integration with YuNet face detector
// =============================================================
// Architecture:
//  process(frame) -> runs DMS inference, returns DmsResult
//  renderOverlay(frame, result) -> draws HMI on top of frame
//
// Uses ONNX YOLOv8 face detector for real-time face detection
// =============================================================
#include "dms_types.hpp"
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <string>
#include <memory>
#include "onnx_face_detector.hpp"  // ONNX YOLOv8 face detector

namespace dms {

class DmsProcessor {
public:
    // model_path: path to ONNX face detection model
    // Pass empty string to skip face detection (overlay-only mode)
    explicit DmsProcessor(const std::string& model_path = "");
    ~DmsProcessor() = default;

    // Run DMS inference on a single BGR frame.
    // Thread-safe: may be called from any thread.
    DmsResult process(const cv::Mat& frame);

    // Render the HMI overlay onto 'frame' in-place.
    // Call ONLY from main/display thread.
    static void renderOverlay(cv::Mat& frame, const DmsResult& result);

private:
    // ONNX face detector
    std::unique_ptr<OnnxFaceDetector> m_faceDetector;

    // Rolling frame counter for PERCLOS simulation
    uint64_t m_frameCount = 0;

    // Helpers
    GazeZone inferGazeZone(const cv::Rect& face, const cv::Size& frameSize) const;
    float simulatePerclos() const;
    DrowsinessLevel classifyDrowsiness(float perclos) const;
    DistractionLevel classifyDistraction(GazeZone zone, float offRoadSec) const;

    // Cumulative off-road gaze timer
    float m_offRoadGazeSec = 0.f;
    bool m_wasOffRoad = false;
    std::chrono::steady_clock::time_point m_offRoadStart;
};

} // namespace dms
