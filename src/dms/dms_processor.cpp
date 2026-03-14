// ================================================================
// dms_processor.cpp - DMS Euro NCAP 2026 PoC
// Step 3: Stub DMS inference engine + HMI overlay renderer implementation
// ================================================================

#include "dms_processor.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <cmath>

using namespace dms;

// ================================================================
// Constructor
// ================================================================
DmsProcessor::DmsProcessor(const std::string& cascade_path) {
    if (!cascade_path.empty()) {
        m_hasCascade = m_faceCascade.load(cascade_path);
        if (!m_hasCascade) {
            fprintf(stderr, "[DmsProcessor] Warning: failed to load Haar cascade from '%s'\n",
                    cascade_path.c_str());
        }
    }
}

// ================================================================
// process() - stub inference per-frame
// ================================================================
DmsResult DmsProcessor::process(const cv::Mat& frame) {
    DmsResult result;
    result.timestamp = std::chrono::steady_clock::now();
    m_frameCount++;

    // ---- Face detection (Haar cascade stub) ----------
    cv::Rect face;
    if (m_hasCascade && !frame.empty()) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        m_faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(60, 60));
        if (!faces.empty()) {
            face = faces[0];  // take largest or first
            result.faceDetected = true;
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

