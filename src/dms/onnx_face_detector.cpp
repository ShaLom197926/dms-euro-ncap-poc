// =============================================================
// onnx_face_detector.cpp - YOLOv8 Face Detection Implementation
// =============================================================
#include "onnx_face_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace dms {

OnnxFaceDetector::OnnxFaceDetector(const std::string& modelPath, float confThreshold, float iouThreshold)
    : m_confThreshold(confThreshold), m_iouThreshold(iouThreshold) {
    
    try {
        // Initialize ONNX Runtime environment
        m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YOLOv8Face");
        
        // Configure session options
        m_sessionOptions.SetIntraOpNumThreads(4);
        m_sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Create session
        std::wstring wideModelPath(modelPath.begin(), modelPath.end());
        m_session = std::make_unique<Ort::Session>(*m_env, wideModelPath.c_str(), m_sessionOptions);
        
        // Get input/output info
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input info
        size_t numInputNodes = m_session->GetInputCount();
        if (numInputNodes > 0) {
            auto inputName = m_session->GetInputNameAllocated(0, allocator);
            m_inputNames.push_back(inputName.release());
            
            auto inputTypeInfo = m_session->GetInputTypeInfo(0);
            auto tensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
            m_inputShape = tensorInfo.GetShape();
            
            // YOLOv8 expects [1, 3, 640, 640]
            if (m_inputShape.size() == 4) {
                m_inputHeight = static_cast<int>(m_inputShape[2]);
                m_inputWidth = static_cast<int>(m_inputShape[3]);
            }
        }
        
        // Output info
        size_t numOutputNodes = m_session->GetOutputCount();
        if (numOutputNodes > 0) {
            auto outputName = m_session->GetOutputNameAllocated(0, allocator);
            m_outputNames.push_back(outputName.release());
        }
        
        m_initialized = true;
        printf("[OnnxFaceDetector] Initialized with model: %s (input: %dx%d)\n", 
               modelPath.c_str(), m_inputWidth, m_inputHeight);
        
    } catch (const Ort::Exception& e) {
        fprintf(stderr, "[OnnxFaceDetector] ONNX Runtime error: %s\n", e.what());
        m_initialized = false;
    }
}

void OnnxFaceDetector::preprocess(const cv::Mat& frame, std::vector<float>& inputTensor, float& scaleX, float& scaleY) {
    // Resize and pad to maintain aspect ratio
    cv::Mat resized;
    scaleX = static_cast<float>(m_inputWidth) / frame.cols;
    scaleY = static_cast<float>(m_inputHeight) / frame.rows;
    float scale = std::min(scaleX, scaleY);
    
    int newWidth = static_cast<int>(frame.cols * scale);
    int newHeight = static_cast<int>(frame.rows * scale);
    
    cv::resize(frame, resized, cv::Size(newWidth, newHeight));
    
    // Create padded image
    cv::Mat padded = cv::Mat::zeros(m_inputHeight, m_inputWidth, CV_8UC3);
    resized.copyTo(padded(cv::Rect(0, 0, newWidth, newHeight)));
    
    // Convert to RGB and normalize
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    
    // Convert to float and normalize to [0, 1]
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
    
    // HWC to CHW format
    inputTensor.resize(1 * 3 * m_inputHeight * m_inputWidth);
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    
    size_t channelSize = m_inputHeight * m_inputWidth;
    for (int c = 0; c < 3; ++c) {
        std::memcpy(inputTensor.data() + c * channelSize, 
                    channels[c].data, 
                    channelSize * sizeof(float));
    }
    
    // Store actual scales used
    scaleX = scale;
    scaleY = scale;
}

std::vector<FaceDetection> OnnxFaceDetector::detect(const cv::Mat& frame) {
    if (!m_initialized || frame.empty()) {
        return {};
    }
    
    try {
        // Preprocess
        std::vector<float> inputTensor;
        float scaleX, scaleY;
        preprocess(frame, inputTensor, scaleX, scaleY);
        
        // Create input tensor
        std::vector<int64_t> inputShape = {1, 3, m_inputHeight, m_inputWidth};
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensorOrt = Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensor.data(), inputTensor.size(),
            inputShape.data(), inputShape.size());
        
        // Run inference
        auto outputTensors = m_session->Run(
            Ort::RunOptions{nullptr},
            m_inputNames.data(), &inputTensorOrt, 1,
            m_outputNames.data(), 1);
        
        // Get output tensor
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        // Copy output data
        size_t outputSize = 1;
        for (auto dim : outputShape) {
            outputSize *= dim;
        }
        std::vector<float> output(outputData, outputData + outputSize);
        
        // Postprocess
        return postprocess(output, frame.size(), scaleX, scaleY);
        
    } catch (const Ort::Exception& e) {
        fprintf(stderr, "[OnnxFaceDetector] Inference error: %s\n", e.what());
        return {};
    }
}

std::vector<FaceDetection> OnnxFaceDetector::postprocess(
    const std::vector<float>& output, const cv::Size& originalSize, float scaleX, float scaleY) {
    
    // YOLOv8 output format: [1, 20, 8400]
    // 20 channels: 4 (bbox) + 1 (confidence) + 15 (5 landmarks * 3)
    std::vector<FaceDetection> detections;
    
    const int numDetections = 8400;
    const int numChannels = 20;
    
    for (int i = 0; i < numDetections; ++i) {
        // Get confidence
        float confidence = output[4 * numDetections + i];
        
        if (confidence > m_confThreshold) {
            // Get bbox (center_x, center_y, width, height)
            float cx = output[0 * numDetections + i] / scaleX;
            float cy = output[1 * numDetections + i] / scaleY;
            float w = output[2 * numDetections + i] / scaleX;
            float h = output[3 * numDetections + i] / scaleY;
            
            // Convert to (x, y, width, height)
            int x = static_cast<int>(cx - w / 2);
            int y = static_cast<int>(cy - h / 2);
            
            // Clamp to image boundaries
            x = std::max(0, std::min(x, originalSize.width - 1));
            y = std::max(0, std::min(y, originalSize.height - 1));
            int width = std::min(static_cast<int>(w), originalSize.width - x);
            int height = std::min(static_cast<int>(h), originalSize.height - y);
            
            FaceDetection det;
            det.bbox = cv::Rect(x, y, width, height);
            det.confidence = confidence;
            
            // Extract landmarks (5 points: left_eye, right_eye, nose, left_mouth, right_mouth)
            for (int j = 0; j < 5; ++j) {
                float lx = output[(5 + j * 3) * numDetections + i] / scaleX;
                float ly = output[(6 + j * 3) * numDetections + i] / scaleY;
                det.landmarks.push_back(cv::Point2f(lx, ly));
            }
            
            detections.push_back(det);
        }
    }
    
    // Apply NMS
    auto indices = nonMaximumSuppression(detections);
    std::vector<FaceDetection> result;
    for (int idx : indices) {
        result.push_back(detections[idx]);
    }
    
    return result;
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
