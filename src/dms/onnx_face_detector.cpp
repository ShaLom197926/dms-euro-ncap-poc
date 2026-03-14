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
                // Convert to top-left corner format and scale to original image size
                float x_center = data[0] / scaleX;
                float y_center = data[1] / scaleY;
                float width = data[2] / scaleX;
                float height = data[3] / scaleY;
                
                // Convert center to top-left corner
                float x = x_center - width / 2.0f;
                float y = y_center - height / 2.0f;
                    // Clamp to image boundaries
                    x = std::max(0.0f, std::min(x, static_cast<float>(frame.cols - 1)));
                    y = std::max(0.0f, std::min(y, static_cast<float>(frame.rows - 1)));
                        w = std::min(width, static_cast<float>(frame.cols - x));
                        h = std::min(height, static_cast<float>(frame.rows - y));
                    det.bbox = cv::Rect(static_cast<int>(x), static_cast<int>(y),
                                      static_cast<int>(w), static_cast<int>(h));
                    det.confidence = confidence;
                    
                    // Extract 5 landmarks (right eye, left eye, nose, right mouth, left mouth)
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

float OnnxFaceDetector::computeIoU(const cv::Rect& a, const cv::Rect& b) {
    int intersectionArea = (a & b).area();
    int unionArea = a.area() + b.area() - intersectionArea;
    return unionArea > 0 ? static_cast<float>(intersectionArea) / unionArea : 0.0f;
}

std::vector<int> OnnxFaceDetector::nonMaximumSuppression(const std::vector<FaceDetection>& boxes) {
    std::vector<int> indices;
    std::vector<std::pair<float, int>> scoreIndex;
    
    for (size_t i = 0; i < boxes.size(); ++i) {
        scoreIndex.push_back({boxes[i].confidence, static_cast<int>(i)});
    }
    
    // Sort by confidence
    std::sort(scoreIndex.begin(), scoreIndex.end(), 
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::vector<bool> suppressed(boxes.size(), false);
    
    for (const auto& [score, i] : scoreIndex) {
        if (suppressed[i]) continue;
        
        indices.push_back(i);
        
        for (size_t j = 0; j < boxes.size(); ++j) {
            if (suppressed[j] || i == static_cast<int>(j)) continue;
            
            float iou = computeIoU(boxes[i].bbox, boxes[j].bbox);
            if (iou > m_iouThreshold) {
                suppressed[j] = true;
            }
        }
    }
    
    return indices;
}

} // namespace dms
