#pragma once
// =============================================================
// dms_types.hpp  -  DMS Euro NCAP 2026 PoC
// Step 3: Shared result types for DMS inference pipeline
// =============================================================

#include <string>
#include <chrono>

namespace dms {

// ---- Gaze zone (Euro NCAP 2026 relevant zones) ------------------
enum class GazeZone : uint8_t {
    Unknown       = 0,
    Forward       = 1,  // on-road, acceptable
    Instrument    = 2,  // instrument cluster glance
    Infotainment  = 3,  // center stack / touchscreen
    Mirror        = 4,  // rear/side mirror check
    Passenger     = 5,  // towards front passenger
    OffRoad       = 6,  // extended off-road gaze -> distraction
};

// ---- Drowsiness level -------------------------------------------
enum class DrowsinessLevel : uint8_t {
    Alert    = 0,
    Mild     = 1,  // PERCLOS 0.15-0.35
    Moderate = 2,  // PERCLOS 0.35-0.65
    Severe   = 3,  // PERCLOS > 0.65  -> Euro NCAP warning trigger
};

// ---- Distraction level ------------------------------------------
enum class DistractionLevel : uint8_t {
    Attentive   = 0,
    Mild        = 1,
    Distracted  = 2,  // > 2 s continuous off-road gaze
    Severe      = 3,  // > 6 s  -> Euro NCAP warning trigger
};

// ---- Head pose (Euler angles in degrees) ------------------------
struct HeadPose {
    float yaw   = 0.f;  // positive = look right
    float pitch = 0.f;  // positive = look up
    float roll  = 0.f;  // positive = tilt right
};

// ---- Per-frame DMS result ---------------------------------------
struct DmsResult {
    // Inference outputs
    GazeZone        gazeZone        = GazeZone::Unknown;
    DrowsinessLevel drowsiness      = DrowsinessLevel::Alert;
    DistractionLevel distraction    = DistractionLevel::Attentive;
    HeadPose        headPose;

    // PERCLOS (proportion of eye closure) over rolling window
    float           perclos         = 0.f;  // 0.0 - 1.0

    // Cumulative off-road gaze duration (seconds)
    float           offRoadGazeSec  = 0.f;

    // Face detected this frame?
    bool            faceDetected    = false;

    // Timestamp of this result
    std::chrono::steady_clock::time_point timestamp;

    // Convenience: is a Euro NCAP warning required?
    bool requiresWarning() const {
        return drowsiness   >= DrowsinessLevel::Severe ||
               distraction  >= DistractionLevel::Severe;
    }

    // Human-readable gaze zone string
    static const char* gazeZoneStr(GazeZone z) {
        switch (z) {
            case GazeZone::Forward:      return "Forward";
            case GazeZone::Instrument:   return "Instrument";
            case GazeZone::Infotainment: return "Infotainment";
            case GazeZone::Mirror:       return "Mirror";
            case GazeZone::Passenger:    return "Passenger";
            case GazeZone::OffRoad:      return "OffRoad";
            default:                     return "Unknown";
        }
    }

    static const char* drowsinessStr(DrowsinessLevel d) {
        switch (d) {
            case DrowsinessLevel::Alert:    return "Alert";
            case DrowsinessLevel::Mild:     return "Mild";
            case DrowsinessLevel::Moderate: return "Moderate";
            case DrowsinessLevel::Severe:   return "SEVERE";
            default:                        return "Unknown";
        }
    }

    static const char* distractionStr(DistractionLevel d) {
        switch (d) {
            case DistractionLevel::Attentive:  return "Attentive";
            case DistractionLevel::Mild:       return "Mild";
            case DistractionLevel::Distracted: return "Distracted";
            case DistractionLevel::Severe:     return "SEVERE";
            default:                           return "Unknown";
        }
    }
};

} // namespace dms
