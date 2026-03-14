cmake_minimum_required(VERSION 3.18)
project(dms_app LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Sources ----------------------------------------------------------------
add_executable(dms_app
    src/main.cpp
    src/camera/camera_capture.cpp
    src/pipeline/frame_pipeline.cpp  # Step 2: producer/consumer pipeline
    src/dms/dms_processor.cpp        # Step 3: DMS inference + overlay
    src/dms/onnx_face_detector.cpp   # Step 4: ONNX YuNet face detector
)

# ---- OpenCV ----------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio imgcodecs dnn)

# ---- ONNX Runtime (Step 4: YOLO-Pose inference) ----------------------------
set(ONNXRUNTIME_ROOT "D:/onnxruntime-win-x64-1.20.2/")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include/")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib/")
set(ONNXRUNTIME_BIN_DIR "${ONNXRUNTIME_ROOT}/bin/")

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIRS}/onnxruntime_cxx_api.h")
    message(WARNING
        "[dms] ONNX Runtime not found at ${ONNXRUNTIME_ROOT}\.\n"
        "      Download from: https://github.com/microsoft/onnxruntime/releases\.\n"
        "      Extract to D:/onnxruntime-win-x64-1.20.2/"
    )
endif()

include_directories(${ONNXRUNTIME_INCLUDE_DIRS})

# ---- Threads (std::thread needs this on Linux; no-op on MSVC) -------------
find_package(Threads REQUIRED)

# ---- Include directories ---------------------------------------------------
target_include_directories(dms_app PRIVATE
    ${CMAKE_SOURCE_DIR}/src/**
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

# ---- Link libraries --------------------------------------------------------
target_link_libraries(dms_app PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
)

# Copy ONNX Runtime DLL to output directory (for Windows runtime)
if(WIN32 AND EXISTS "${ONNXRUNTIME_BIN_DIR}/onnxruntime.dll")
    add_custom_command(TARGET dms_app POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_BIN_DIR}/onnxruntime.dll"
        $<TARGET_FILE_DIR:dms_app>
    )
endif()
