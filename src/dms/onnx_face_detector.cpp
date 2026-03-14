// =============================================================
// onnx_face_detector.cpp - YuNet Face Detection using OpenCV DNN
// =============================================================
#include "onnx_face_detector.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
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

        // YuNet output format: [1, N, 15]
        // Each row: [x1, y1, w, h, lm0x, lm0y, lm1x, lm1y, lm2x, lm2y,
        //            lm3x, lm3y, lm4x, lm4y, confidence]
        // x1,y1,w,h are already PIXEL coords in 640x640 input space — NOT normalized
        // confidence is at index 14 (last)

        std::vector<FaceDetection> detections;

        if (output.dims == 3 && output.size[0] == 1) {
            int numDetections = output.size[1];

            // Log once every 30 frames
            static int frameCount = 0;
            bool shouldLog = (++frameCount % 30 == 0);

            if (shouldLog) {
                printf("[OnnxFaceDetector] Raw candidates: %d\n", numDetections);
            }

            for (int i = 0; i < numDetections; ++i) {
                float* data = output.ptr<float>(0, i);

                // Index 14 = confidence
                float confidence = data[14];
                if (confidence < m_confThreshold) continue;

                // Coords are pixel-space in 640x640 input image — scale back to frame
                float x1 = data[0] / scaleX;
                float y1 = data[1] / scaleY;
                float w  = data[2] / scaleX;
                float h  = data[3] / scaleY;

                // Clamp to frame boundaries
                x1 = std::max(0.0f, x1);
                y1 = std::max(0.0f, y1);
                w  = std::min(w, static_cast<float>(frame.cols) - x1);
                h  = std::min(h, static_cast<float>(frame.rows) - y1);

                if (w <= 0 || h <= 0) continue;  // skip degenerate boxes

                FaceDetection det;
                det.bbox       = cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                                          static_cast<int>(w),  static_cast<int>(h));
                det.confidence = confidence;

                // Parse 5 landmarks (indices 4–13, pairs of x,y)
                for (int j = 0; j < 5; ++j) {
                    float lx = data[4 + j * 2] / scaleX;
                    float ly = data[5 + j * 2] / scaleY;
                    det.landmarks.push_back(cv::Point2f(lx, ly));
                }

                detections.push_back(det);
            }

            if (shouldLog) {
                printf("[OnnxFaceDetector] After threshold: %zu detections\n", detections.size());
            }
        }

        // Apply NMS
        auto indices = nonMaximumSuppression(detections);
        std::vector<FaceDetection> result;
        result.reserve(indices.size());
        for (int idx : indices) {
            result.push_back(detections[idx]);
        }

        // Log final result once per 30 frames
        {
            static int logCount = 0;
            if (++logCount % 30 == 0 && !result.empty()) {
                printf("[OnnxFaceDetector] Final faces after NMS: %zu\n", result.size());
                for (size_t i = 0; i < result.size() && i < 3; ++i) {
                    printf("  Face %zu: x=%d y=%d w=%d h=%d conf=%.2f\n",
                           i,
                           result[i].bbox.x, result[i].bbox.y,
                           result[i].bbox.width, result[i].bbox.height,
                           result[i].confidence);
                }
            }
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
