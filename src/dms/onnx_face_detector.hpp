// =============================================================
// onnx_face_detector.hpp - YOLOv8 Face Detection with ONNX Runtime
// =============================================================
#pragma once
#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

namespace dms {

struct FaceDetection {
    cv::Rect bbox;
    float confidence;
    std::vector<cv::Point2f> landmarks; // 5 landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
};

class OnnxFaceDetector {
public:
    explicit OnnxFaceDetector(const std::string& modelPath, float confThreshold = 0.5f, float iouThreshold = 0.4f);
    ~OnnxFaceDetector() = default;

    std::vector<FaceDetection> detect(const cv::Mat& frame);
    
    bool isInitialized() const { return m_initialized; }

private:
    void preprocess(const cv::Mat& frame, std::vector<float>& inputTensor, float& scaleX, float& scaleY);
    std::vector<FaceDetection> postprocess(const std::vector<float>& output, const cv::Size& originalSize, float scaleX, float scaleY);
    float computeIoU(const cv::Rect& a, const cv::Rect& b);
    std::vector<int> nonMaximumSuppression(const std::vector<FaceDetection>& boxes);

    std::unique_ptr<Ort::Env> m_env;
    std::unique_ptr<Ort::Session> m_session;
    Ort::SessionOptions m_sessionOptions;
    
    std::vector<const char*> m_inputNames;
    std::vector<const char*> m_outputNames;
    std::vector<int64_t> m_inputShape;
    
    int m_inputWidth = 640;
    int m_inputHeight = 640;
    float m_confThreshold;
    float m_iouThreshold;
    bool m_initialized = false;
};

} // namespace dms
