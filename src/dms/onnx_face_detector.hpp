#pragma once
// =============================================================
// onnx_face_detector.hpp - YuNet Face Detection with OpenCV DNN
// =============================================================
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <memory>

namespace dms {

struct FaceDetection {
    cv::Rect bbox;
    float confidence;
    std::vector<cv::Point2f> landmarks;  // 5 landmarks: right_eye, left_eye, nose, right_mouth, left_mouth
};

class OnnxFaceDetector {
public:
    explicit OnnxFaceDetector(const std::string& modelPath, float confThreshold = 0.5f, float iouThreshold = 0.4f);
    ~OnnxFaceDetector() = default;

    std::vector<FaceDetection> detect(const cv::Mat& frame);
    
    bool isInitialized() const { return m_initialized; }

private:
    void preprocess(const cv::Mat& frame, cv::Mat& blob, float& scaleX, float& scaleY);
    float computeIoU(const cv::Rect& a, const cv::Rect& b);
    std::vector<int> nonMaximumSuppression(const std::vector<FaceDetection>& boxes);

    cv::dnn::Net m_net;
    
    int m_inputWidth = 640;
    int m_inputHeight = 640;
    
    float m_confThreshold;
    float m_iouThreshold;
    bool m_initialized = false;
};

} // namespace dms
