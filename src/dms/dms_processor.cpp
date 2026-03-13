// =============================================================
// dms_processor.cpp  -  DMS Euro NCAP 2026 PoC
// Step 3: Stub DMS inference engine + HMI overlay renderer implementation
// =============================================================

#include "dms_processor.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <cmath>

using namespace dms;

// =============================================================
// Constructor
// =============================================================
DmsProcessor::DmsProcessor(const std::string& cascade_path) {
    if (!cascade_path.empty()) {
        m_hasCascade = m_faceCascade.load(cascade_path);
        if (!m_hasCascade) {
            fprintf(stderr, "[DmsProcessor] Warning: failed to load Haar cascade from '%s'\n",
                    cascade_path.c_str());
        }
    }
}

// =============================================================
// process() - stub inference per-frame
// =============================================================
DmsResult DmsProcessor::process(const cv::Mat& frame) {
    DmsResult result;
    result.timestamp = std::chrono::steady_clock::now();
    m_frameCount++;

    // ---- Face detection (Haar cascade stub) ---------------------
    cv::Rect face;
    if (m_hasCascade && !frame.empty()) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        m_faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(60, 60));
        if (!faces.empty()) {
            face = faces[0];  // Take largest or first
            result.faceDetected = true;
        }
    }

    // ---- Gaze zone inference (proxy from face position) ---------
    if (result.faceDetected) {
        result.gazeZone = inferGazeZone(face, frame.size());
    } else {
        result.gazeZone = GazeZone::Unknown;
    }

    // ---- Stub head pose (proxy from face bbox center) -----------
    if (result.faceDetected) {
        int cx = face.x + face.width / 2;
        int cy = face.y + face.height / 2;
        int fw = frame.cols, fh = frame.rows;
        // yaw: face center left/right -> map to ±30°
        result.headPose.yaw   = ((cx - fw/2.f) / (fw/2.f)) * 30.f;
        // pitch: face center up/down -> map to ±20°
        result.headPose.pitch = -((cy - fh/2.f) / (fh/2.f)) * 20.f;
        result.headPose.roll  = 0.f;  // TODO: real model
    }

    // ---- Simulated PERCLOS (sine wave for PoC demo) -------------
    result.perclos = simulatePerclos();

    // ---- Drowsiness classification ------------------------------
    result.drowsiness = classifyDrowsiness(result.perclos);

    // ---- Off-road gaze tracking (for distraction) ---------------
    bool isOffRoad = (result.gazeZone == GazeZone::OffRoad);
    if (isOffRoad && !m_wasOffRoad) {
        // Started looking off-road
        m_offRoadStart = result.timestamp;
        m_wasOffRoad = true;
    } else if (isOffRoad && m_wasOffRoad) {
        // Accumulate off-road duration
        auto elapsed = std::chrono::duration<float>(result.timestamp - m_offRoadStart).count();
        m_offRoadGazeSec = elapsed;
    } else if (!isOffRoad && m_wasOffRoad) {
        // Reset
        m_offRoadGazeSec = 0.f;
        m_wasOffRoad = false;
    }

    result.offRoadGazeSec = m_offRoadGazeSec;
    result.distraction = classifyDistraction(result.gazeZone, result.offRoadGazeSec);

    return result;
}

// =============================================================
// Helper: infer gaze zone from face bbox position
// =============================================================
GazeZone DmsProcessor::inferGazeZone(const cv::Rect& face, const cv::Size& frameSize) const {
    // Proxy logic: center of face bbox normalized coords
    float cx = (face.x + face.width / 2.f) / frameSize.width;
    float cy = (face.y + face.height / 2.f) / frameSize.height;

    // Simple horizontal zones:
    //  cx < 0.3 -> looking right (passenger side) = Passenger
    //  cx > 0.7 -> looking left  (driver door)  = OffRoad
    //  0.4 < cx < 0.6 -> Forward
    //  else Infotainment / Instrument

    if (cx < 0.3f) {
        return GazeZone::Passenger;
    } else if (cx > 0.7f) {
        return GazeZone::OffRoad;
    } else if (cx >= 0.4f && cx <= 0.6f) {
        // Central gaze -> consider vertical: upper half = Instrument
        if (cy < 0.4f) {
            return GazeZone::Instrument;
        } else {
            return GazeZone::Forward;
        }
    } else {
        // Intermediate -> Infotainment / Mirror
        return (cy < 0.5f) ? GazeZone::Mirror : GazeZone::Infotainment;
    }
}

// =============================================================
// Helper: simulate PERCLOS via sine wave (for PoC demo)
// =============================================================
float DmsProcessor::simulatePerclos() const {
    // Oscillate PERCLOS over ~300 frames (30 FPS -> ~10 sec cycle)
    // Range: 0.0 - 0.8
    float phase = (m_frameCount % 300) / 300.f;
    float sine = std::sin(phase * 2.f * 3.14159f);
    // Map [-1,1] -> [0, 0.8]
    return (sine + 1.f) * 0.4f;
}

// =============================================================
// Helper: classify drowsiness from PERCLOS
// =============================================================
DrowsinessLevel DmsProcessor::classifyDrowsiness(float perclos) const {
    // Euro NCAP thresholds (approximation):
    //   PERCLOS < 0.15  -> Alert
    //   0.15 <= P < 0.35 -> Mild
    //   0.35 <= P < 0.65 -> Moderate
    //   P >= 0.65        -> Severe (warning required)
    if (perclos < 0.15f)  return DrowsinessLevel::Alert;
    if (perclos < 0.35f)  return DrowsinessLevel::Mild;
    if (perclos < 0.65f)  return DrowsinessLevel::Moderate;
    return DrowsinessLevel::Severe;
}

// =============================================================
// Helper: classify distraction from gaze zone + duration
// =============================================================
DistractionLevel DmsProcessor::classifyDistraction(GazeZone zone, float offRoadSec) const {
    // Euro NCAP thresholds:
    //   off-road gaze > 6 s  -> Severe warning
    //   off-road gaze > 2 s  -> Distracted
    //   else Mild or Attentive
    if (zone == GazeZone::OffRoad) {
        if (offRoadSec > 6.f) return DistractionLevel::Severe;
        if (offRoadSec > 2.f) return DistractionLevel::Distracted;
        return DistractionLevel::Mild;
    }
    // For other non-forward zones (Infotainment, Passenger):
    if (zone == GazeZone::Infotainment || zone == GazeZone::Passenger) {
        return DistractionLevel::Mild;
    }
    return DistractionLevel::Attentive;
}

// =============================================================
// renderOverlay() - HMI text overlay on OpenCV window
// =============================================================
void DmsProcessor::renderOverlay(cv::Mat& frame, const DmsResult& result) {
    if (frame.empty()) return;

    const int W = frame.cols;
    const int H = frame.rows;

    // Choose font
    const int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    const double fontScale = 0.6;
    const int thickness = 2;
    const int lineGap = 30;

    // Background semi-transparent box for text
    cv::Rect textBox(10, 10, 350, 300);
    cv::Mat roi = frame(textBox);
    cv::Mat overlay;
    roi.copyTo(overlay);
    cv::rectangle(overlay, cv::Point(0, 0), cv::Point(overlay.cols, overlay.rows),
                  cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(overlay, 0.5, roi, 0.5, 0, roi);

    int yPos = 40;
    auto putLine = [&](const std::string& text, const cv::Scalar& color) {
        cv::putText(frame, text, cv::Point(20, yPos), fontFace, fontScale, color, thickness);
        yPos += lineGap;
    };

    // Title
    putLine("DMS - Euro NCAP 2026", cv::Scalar(255, 255, 255));

    // Face detection status
    if (result.faceDetected) {
        putLine("Face: DETECTED", cv::Scalar(0, 255, 0));
    } else {
        putLine("Face: NOT DETECTED", cv::Scalar(0, 0, 255));
    }

    // Gaze zone
    {
        std::string gazeText = "Gaze: ";
        gazeText += DmsResult::gazeZoneStr(result.gazeZone);
        cv::Scalar gazeColor = (result.gazeZone == GazeZone::Forward) ?
                               cv::Scalar(0, 255, 0) : cv::Scalar(255, 128, 0);
        putLine(gazeText, gazeColor);
    }

    // Drowsiness
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "Drowsiness: %s (%.2f)",
                 DmsResult::drowsinessStr(result.drowsiness), result.perclos);
        cv::Scalar drowsColor = (result.drowsiness >= DrowsinessLevel::Severe) ?
                                cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 0);
        putLine(buf, drowsColor);
    }

    // Distraction
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "Distraction: %s (%.1fs)",
                 DmsResult::distractionStr(result.distraction), result.offRoadGazeSec);
        cv::Scalar distColor = (result.distraction >= DistractionLevel::Severe) ?
                               cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 0);
        putLine(buf, distColor);
    }

    // Head pose
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "Yaw: %.1f°  Pitch: %.1f°",
                 result.headPose.yaw, result.headPose.pitch);
        putLine(buf, cv::Scalar(200, 200, 200));
    }

    // Warning indicator (large red box)
    if (result.requiresWarning()) {
        cv::rectangle(frame, cv::Rect(W - 220, 10, 200, 80),
                      cv::Scalar(0, 0, 255), 3);
        cv::putText(frame, "!!! WARNING !!!", cv::Point(W - 210, 55),
                    fontFace, 0.7, cv::Scalar(0, 0, 255), 2);
    }
}
