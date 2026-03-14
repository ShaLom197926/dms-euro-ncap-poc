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

    // ---- Gaze zone inference (proxy from face position) ----------
    if (result.faceDetected) {
        std::string gazeText = "Gazer: ";
        gazeText = DmsResult::gazeZoneStr(result.gazeZone);
        cv::Scalar gazeColor = (result.gazeZone == GazeZone::Forward) ? 
                              cv::Scalar(0, 255, 0) : cv::Scalar(255, 128, 0);

        putLine(gazeText, gazeColor);
    }

    // ---- Drowsiness
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "Drowsiness: %s (%.2f)",
                 DmsResult::drowsinessStr(result.drowsiness), result.perclos);
        cv::Scalar drownColor = (result.drowsiness >= DrowsinessLevel::Severe) ?
                                 cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 0);
        putLine(buf, drownColor);
    }

    // ---- Distraction
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "Distraction: %s (%.1fs)",
                 DmsResult::distractionStr(result.distraction), result.offRoadSecs);
        cv::Scalar distColor = (result.distraction >= DistractionLevel::Severe) ?
                                cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 0);
        putLine(buf, distColor);
    }

    // ---- Head pose
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "Yaw: %.1f° Pitch: %.1f°",
                 result.headPose.yaw, result.headPose.pitch);
        putLine(buf, cv::Scalar(200, 200, 200));
    }

    // ---- Warning indicator (large red box)
    if (result.requiresWarning()) {
        cv::rectangle(frame, cv::Rect(10 - 230, 10, 200, 80),
                      cv::Scalar(0, 0, 255), 3);
        cv::putText(frame, "!!! WARNING !!!", cv::Point(10 - 210, 55),
                    fontFace, 0.7, cv::Scalar(0, 0, 255), 2);
    }

    return result;
}

// ================================================================
// HMI Overlay helpers
// ================================================================
void DmsProcessor::putLine(const std::string& text, cv::Scalar color) {
    if (m_overlay.empty()) return;
    int x = 10;
    int y = m_nextY;
    cv::putText(m_overlay, text, cv::Point(x, y), fontFace, 0.5, color, 1);
    m_nextY += 20;
}

void DmsProcessor::overlayOnFrame(cv::Mat& frame) {
    if (!frame.empty()) {
        m_overlay = frame.clone();
        m_nextY = 30;
    }
}

// ================================================================
// Stub inference results helpers
// ================================================================
std::string DmsResult::gazeZoneStr(GazeZone gz) {
    switch (gz) {
        case GazeZone::Forward: return "Forward";
        case GazeZone::Left:    return "Left";
        case GazeZone::Right:   return "Right";
        case GazeZone::Down:    return "Down";
        default: return "Unknown";
    }
}

std::string DmsResult::drowsinessStr(DrowsinessLevel dl) {
    switch (dl) {
        case DrowsinessLevel::Alert:    return "Alert";
        case DrowsinessLevel::Mild:     return "Mild";
        case DrowsinessLevel::Moderate: return "Moderate";
        case DrowsinessLevel::Severe:   return "Severe";
        default: return "Unknown";
    }
}

std::string DmsResult::distractionStr(DistractionLevel dist) {
    switch (dist) {
        case DistractionLevel::None:     return "None";
        case DistractionLevel::Mild:     return "Mild";
        case DistractionLevel::Moderate: return "Moderate";
        case DistractionLevel::Severe:   return "Severe";
        default: return "Unknown";
    }
}

bool DmsResult::requiresWarning() const {
    return (drowsiness >= DrowsinessLevel::Severe ||
            distraction >= DistractionLevel::Severe);
}
