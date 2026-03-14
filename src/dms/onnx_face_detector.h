// ================================================================
// onnx_face_detector.h - ONNX-based face detector for DMS
// Phase 2: ONNX Runtime Integration
// ================================================================
#ifndef DMS_ONNX_FACE_DETECTOR_H_
#define DMS_ONNX_FACE_DETECTOR_H_

#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

namespace dms {

// Face detection result structure
struct FaceDetection {
    cv::Rect bbox;          // Bounding box (x, y, width, height)
    float confidence;       // Detection confidence score
    cv::Point2f landmarks[5]; // 5 facial landmarks (optional)
};

/**
 * OnnxFaceDetector - Lightweight ONNX-based face detection
 * 
 * Uses a pre-trained ONNX model (e.g., YuNet, Ultra-Light-Face-Detector)
 * for fast and accurate face detection suitable for DMS applications.
 */
class OnnxFaceDetector {
public:
    /**
     * Constructor
     * @param modelPath Path to the ONNX model file
     * @param inputWidth Model input width (default: 320)
     * @param inputHeight Model input height (default: 320)
     * @param confThreshold Confidence threshold for detections (default: 0.6)
     * @param nmsThreshold NMS IOU threshold (default: 0.3)
     */
    explicit OnnxFaceDetector(
        const std::string& modelPath,
        int inputWidth = 320,
        int inputHeight = 320,
        float confThreshold = 0.6f,
        float nmsThreshold = 0.3f
    );

    ~OnnxFaceDetector();

    /**
     * Detect faces in an image
     * @param image Input BGR image
     * @return Vector of detected faces with bounding boxes and confidence scores
     */
    std::vector<FaceDetection> detect(const cv::Mat& image);

    /**
     * Check if detector is initialized
     * @return True if model loaded successfully
     */
    bool isInitialized() const { return m_initialized; }

    /**
     * Get last error message
     * @return Error message string
     */
    std::string getLastError() const { return m_lastError; }

private:
    // ONNX Runtime session and environment
    std::unique_ptr<Ort::Env> m_env;
    std::unique_ptr<Ort::Session> m_session;
    Ort::SessionOptions m_sessionOptions;
    Ort::MemoryInfo m_memoryInfo{nullptr};

    // Model configuration
    int m_inputWidth;
    int m_inputHeight;
    float m_confThreshold;
    float m_nmsThreshold;

    // Input/Output names
    std::vector<const char*> m_inputNames;
    std::vector<const char*> m_outputNames;
    std::vector<std::string> m_inputNamesStorage;
    std::vector<std::string> m_outputNamesStorage;

    // State
    bool m_initialized;
    std::string m_lastError;

    // Helper methods
    cv::Mat preprocessImage(const cv::Mat& image);
    std::vector<FaceDetection> postprocessOutput(
        const std::vector<Ort::Value>& outputTensors,
        const cv::Size& originalSize
    );
    void applyNMS(std::vector<FaceDetection>& detections);
};

} // namespace dms

#endif // DMS_ONNX_FACE_DETECTOR_H_
