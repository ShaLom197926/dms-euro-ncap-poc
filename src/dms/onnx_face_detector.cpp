// =============================================================
// onnx_face_detector.cpp - YuNet Face Detection using OpenCV DNN
// =============================================================
#include "onnx_face_detector.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace dms {

OnnxFaceDetector::OnnxFaceDetector(const std::string& modelPath, float confThreshold, float iouThreshold)
    : m_confThreshold(confThreshold), m_iouThreshold(iouThreshold) {
    
    try {
        // Load YuNet model using OpenCV DNN (supports ONNX)
        m_net = cv::dnn::readNetFromONNX(modelPath);
        
        if (m_net.empty()) {
            fprintf(stderr, "[OnnxFaceDetector] Failed to load model from: %s\n", modelPath.c_str());
            m_initialized = false;
            return;
        }
        
        // Set backend and target
        m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        m_initialized = true;
        printf("[OnnxFaceDetector] Initialized with model: %s (input: %dx%d)\n",
               modelPath.c_str(), m_inputWidth, m_inputHeight);
        
    } catch (const cv::Exception& e) {
        fprintf(stderr, "[OnnxFaceDetector] OpenCV DNN error: %s\n", e.what());
        m_initialized = false;
    }
}

void OnnxFaceDetector::preprocess(const cv::Mat& frame, cv::Mat& blob, float& scaleX, float& scaleY) {
    // Resize to input size
    cv::Mat resized;
    scaleX = static_cast<float>(m_inputWidth) / frame.cols;
    scaleY = static_cast<float>(m_inputHeight) / frame.rows;
    
    cv::resize(frame, resized, cv::Size(m_inputWidth, m_inputHeight));
    
    // Convert to blob (NCHW format, RGB, normalized to [0,1])
    cv::dnn::blobFromImage(resized, blob, 1.0 / 255.0, cv::Size(m_inputWidth, m_inputHeight),
                          cv::Scalar(0, 0, 0), true, false, CV_32F);
}

std::vector<FaceDetection> OnnxFaceDetector::detect(const cv::Mat& frame) {
    if (!m_initialized || frame.empty()) {
        return {};
    }
    
    try {
        // Preprocess
        cv::Mat blob;
        float scaleX, scaleY;
        preprocess(frame, blob, scaleX, scaleY);
        
        // Forward pass
        m_net.setInput(blob);
        cv::Mat output = m_net.forward();
        
        // Postprocess - YuNet output format: [1, N, 15]
        // Each detection: [x, y, w, h, conf, 5 landmarks (x,y pairs)]
        std::vector<FaceDetection> detections;
        
        if (output.dims == 3 && output.size[0] == 1) {
            int numDetections = output.size[1];
            
            for (int i = 0; i < numDetections; ++i) {
                float* data = output.ptr<float>(0, i);
                
                float confidence = data[4];
                if (confidence > m_confThreshold) {
                    // Get bbox in center format [x_center, y_center, width, height]
                    // Convert to top-left to original image size
                    float x = data[0] / scaleX;
                    float y = data[1] / scaleY;
                    float width = data[2] / scaleX;
                    float height = data[3] / scaleY;
                    
                    // Clamp to image boundaries
                    x = std::max(0.0f, std::min(x, static_cast<float>(frame.cols - 1)));
                    y = std::max(0.0f, std::min(y, static_cast<float>(frame.rows - 1)));
                    float w = std::min(width, static_cast<float>(frame.cols - x));
                    float h = std::min(height, static_cast<float>(frame.rows - y));
                    
                    FaceDetection det;
                    det.bbox = cv::Rect(static_cast<int>(x), static_cast<int>(y),
                                       static_cast<int>(w), static_cast<int>(h));
                    det.confidence = confidence;
                    
                    // Parse 5 landmarks (indices 5-14)
                    for (int j = 0; j < 5; ++j) {
                        float lx = data[5 + j * 2] / scaleX;
                        float ly = data[6 + j * 2] / scaleY;
                        det.landmarks.push_back(cv::Point2f(lx, ly));
                    }
                    
                    detections.push_back(det);
                }
            }
        }
        
        // Apply NMS
        auto indices = nonMaximumSuppression(detections);
        std::vector<FaceDetection> result;
        for (int idx : indices) {
            result.push_back(detections[idx]);
        }
        return result;
        
    } catch (const cv::Exception& e) {
        fprintf(stderr, "[OnnxFaceDetector] Inference error: %s\n", e.what());
        return {};
    }
}

std::vector<int> OnnxFaceDetector::nonMaximumSuppression(const std::vector<FaceDetection>& detections) {
    if (detections.empty()) return {};
    
    std::vector<int> indices;
    std::vector<float> scores;
    std::vector<cv::Rect> boxes;
    
    for (const auto& det : detections) {
        scores.push_back(det.confidence);
        boxes.push_back(det.bbox);
    }
    
    // Sort by confidence (descending)
    std::vector<int> sortedIndices(scores.size());
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
    std::sort(sortedIndices.begin(), sortedIndices.end(),
              [&scores](int i1, int i2) { return scores[i1] > scores[i2]; });
    
    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < sortedIndices.size(); ++i) {
        int idx = sortedIndices[i];
        if (suppressed[idx]) continue;
        
        indices.push_back(idx);
        
        for (size_t j = i + 1; j < sortedIndices.size(); ++j) {
            int jdx = sortedIndices[j];
            if (suppressed[jdx]) continue;
            
            float iou = computeIoU(boxes[idx], boxes[jdx]);
            if (iou > m_iouThreshold) {
                suppressed[jdx] = true;
            }
        }
    }
    
    return indices;
}

float OnnxFaceDetector::computeIoU(const cv::Rect& a, const cv::Rect& b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);
    
    int intersectionArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = a.area() + b.area() - intersectionArea;
    
    return (unionArea > 0) ? static_cast<float>(intersectionArea) / unionArea : 0.0f;
}

} // namespace dms
