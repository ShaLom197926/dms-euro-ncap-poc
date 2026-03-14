// ================================================================
// dms_processor.cpp - DMS Euro NCAP 2026 PoC
// Phase 2: ONNX model integration with YuNet face detector
// ================================================================
#include "dms_processor.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <cmath>

using namespace dms;

// ================================================================
// Constructor
// ================================================================
DmsProcessor::DmsProcessor(const std::string& model_path) {
    // Initialize ONNX face detector if model path provided
    if (!model_path.empty()) {
        try {
            m_faceDetector = std::make_unique<OnnxFaceDetector>(model_path);
            printf("[DmsProcessor] ONNX face detector initialized\n");
        } catch (const std::exception& e) {
            fprintf(stderr, "[DmsProcessor] Failed to initialize ONNX detector: %s\n", e.what());
            m_faceDetector = nullptr;
        }
    } else {
        printf("[DmsProcessor] Running in stub mode (no face detection)\n");
    }
}

// ================================================================
// process() - Run DMS inference per-frame
// ================================================================
DmsResult DmsProcessor::process(const cv::Mat& frame) {
    DmsResult result;
    result.timestamp = std::chrono::steady_clock::now();
    m_frameCount++;

    // ---- ONNX face detection ----------
    if (m_faceDetector && !frame.empty()) {
        auto detections = m_faceDetector->detect(frame);
                
        if (!detections.empty()) {
            // Use first detection (highest confidence)
            result.faceDetected = true;
            
            // Store bounding boxes for visualization in renderOverlay()
            for (const auto& det : detections) {
                result.faceDetections.push_back(det.bbox);
            }
            
            // Store face bounding box for head pose estimation
            const auto& face = detections[0].bbox;
            // TODO: Implement real head pose estimation using landmarks
            // For now, use face position as proxy
        }
    }

    return result;
}

// ================================================================
// renderOverlay() - static HMI overlay renderer
// ================================================================
void DmsProcessor::renderOverlay(cv::Mat& frame, const DmsResult& result) {
    if (frame.empty()) return;

    const int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    int y = 30;
    const int lineHeight = 25;
    
    // Draw face detection bounding boxes
    for (const auto& bbox : result.faceDetections) {
        cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);
    }

    // Gaze zone
    {
        std::string text = std::string("Gaze: ") + result.gazeZoneStr(result.gazeZone);
        cv::Scalar color = (result.gazeZone == GazeZone::Forward) ?
            cv::Scalar(0, 255, 0) : cv::Scalar(255, 128, 0);
        cv::putText(frame, text, cv::Point(10, y), fontFace, 0.6, color, 2);
        y += lineHeight;
    }

    // Drowsiness
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "Drowsiness: %s (%.2f)",
            result.drowsinessStr(result.drowsiness), result.perclos);
        cv::Scalar color = (result.drowsiness >= DrowsinessLevel::Severe) ?
            cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 0);
        cv::putText(frame, buf, cv::Point(10, y), fontFace, 0.6, color, 2);
        y += lineHeight;
    }

    // Distraction
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "Distraction: %s",
            result.distractionStr(result.distraction));
        cv::Scalar color = (result.distraction >= DistractionLevel::Severe) ?
            cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 0);
        cv::putText(frame, buf, cv::Point(10, y), fontFace, 0.6, color, 2);
        y += lineHeight;
    }

    // Head pose
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "Yaw: %.1f\xC2\xB0 Pitch: %.1f\xC2\xB0",
            result.headPose.yaw, result.headPose.pitch);
        cv::putText(frame, buf, cv::Point(10, y), fontFace, 0.6, cv::Scalar(200, 200, 200), 2);
        y += lineHeight;
    }

    // Warning indicator
    if (result.requiresWarning()) {
        cv::rectangle(frame, cv::Rect(frame.cols - 250, 10, 230, 80),
            cv::Scalar(0, 0, 255), 3);
        cv::putText(frame, "!!! WARNING !!!", cv::Point(frame.cols - 240, 55),
            fontFace, 0.9, cv::Scalar(0, 0, 255), 2);
    }

    // Face detected indicator
    if (result.faceDetected) {
        cv::circle(frame, cv::Point(frame.cols - 30, 30), 10,
            cv::Scalar(0, 255, 0), -1);
    }
}
