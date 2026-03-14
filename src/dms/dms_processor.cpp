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
