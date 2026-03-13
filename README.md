# DMS Euro NCAP 2026 PoC

A **Driver Monitoring System (DMS)** proof-of-concept for Euro NCAP 2026 compliance, built with C++ and OpenCV on Windows.

## Features

### Step 1: Camera Capture + Live Display ✅
- USB camera capture using OpenCV VideoCapture
- Real-time video display with OpenCV highgui
- Cross-platform support (Windows focus)

### Step 2: Ring Buffer + Multi-threaded Pipeline ✅
- Producer-consumer architecture with separate threads
- Lock-free SPSC ring buffer for frame management
- Configurable buffer size (default: 60 frames)
- Frame statistics tracking (captured/processed/dropped)

### Step 3: DMS HMI Overlay Integration ✅
- Face detection using Haar cascade classifier
- Real-time HMI overlay with:
  - Gaze zone detection (LEFT/CENTER/RIGHT)
  - Drowsiness level monitoring (ALERT/MILD/MODERATE/SEVERE)
  - Distraction level tracking (NONE/LOW/MEDIUM/HIGH)
  - Visual indicators (bounding boxes, status text, color coding)
- Configurable detection parameters

## Project Structure

```
dms-euro-ncap-poc/
├── CMakeLists.txt              # CMake build configuration
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
└── src/
    ├── main.cpp                # Application entry point
    ├── camera/
    │   ├── camera_capture.h    # Camera interface
    │   └── camera_capture.cpp  # OpenCV camera implementation
    ├── pipeline/
    │   ├── frame_pipeline.h    # Pipeline interface
    │   ├── frame_pipeline.cpp  # Producer/consumer threads
    │   └── ring_buffer.hpp     # Lock-free SPSC ring buffer
    ├── dms/
    │   ├── dms_processor.h     # DMS processor interface
    │   ├── dms_processor.cpp   # Face detection + overlay logic
    │   └── dms_types.hpp       # DMS result types
    ├── logger/
    │   ├── dms_logger.h        # Logger interface
    │   └── dms_logger.c        # Thread-safe logger (C)
    └── utils/
        ├── platform.h          # Platform detection macros
        ├── ring_buffer.h       # Ring buffer C interface
        └── ring_buffer.c       # Ring buffer implementation
```

## Prerequisites

### Windows
- **CMake** 3.18+
- **Visual Studio 2019/2022** (MSVC compiler)
- **OpenCV** 4.x
  - Download from: https://opencv.org/releases/
  - Recommended: OpenCV 4.10.0 or later
- **ONNX Runtime** 1.24.2 (already installed)
  - Location: `D:\onnxruntime-win-x64-1.24.2`
  - Includes headers and libraries for DNN inference
- **USB Camera** (any webcam compatible with OpenCV)

### Environment Setup

Set the OpenCV and ONNX Runtime paths:

```cmd
set OpenCV_DIR=C:\opencv\build
set ONNXRUNTIME_DIR=D:\onnxruntime-win-x64-1.24.2
```

## Build Instructions

### Phase 1, Part B: Building with ONNX Runtime Integration

Since you already have ONNX Runtime installed at `D:\onnxruntime-win-x64-1.24.2`, follow these steps:

#### 1. **Clone the repository** (if not already done)
```cmd
git clone https://github.com/ShaLom197926/dms-euro-ncap-poc.git
cd dms-euro-ncap-poc
```

#### 2. **Set environment variables**
```cmd
set OpenCV_DIR=C:\opencv\build
set ONNXRUNTIME_DIR=D:\onnxruntime-win-x64-1.24.2
```

#### 3. **Create build directory**
```cmd
mkdir build
cd build
```

#### 4. **Configure with CMake** (specifying ONNX Runtime path)
```cmd
cmake -G "Visual Studio 17 2022" -A x64 ^
  -DOpenCV_DIR=C:\opencv\build ^
  -DONNXRUNTIME_DIR=D:\onnxruntime-win-x64-1.24.2 ^
  ..
```

**Expected CMake Output:**
```
-- Found OpenCV: C:/opencv/build (found version "4.10.0")
-- ONNX Runtime found at: D:/onnxruntime-win-x64-1.24.2
-- ONNX Runtime include: D:/onnxruntime-win-x64-1.24.2/include
-- ONNX Runtime library: D:/onnxruntime-win-x64-1.24.2/lib/onnxruntime.lib
-- Configuring done
-- Generating done
```

#### 5. **Build the project**
```cmd
cmake --build . --config Release
```

#### 6. **Copy ONNX Runtime DLL to executable directory**

After building, copy the ONNX Runtime DLL to your build output directory:

```cmd
copy D:\onnxruntime-win-x64-1.24.2\lib\onnxruntime.dll Release\
```

#### 7. **Verify the build**

Check that the executable and DLLs are in place:

```cmd
dir Release\
```

You should see:
- `dms_app.exe`
- `onnxruntime.dll`
- OpenCV DLLs (opencv_world4*.dll)

#### 8. **Run the application**
```cmd
Release\dms_app.exe
```

### Alternative: Using Visual Studio IDE

1. Open `build\dms_app.sln` in Visual Studio
2. Set build configuration to **Release** | **x64**
3. Build → Build Solution (F7)
4. Manually copy `onnxruntime.dll` to `Release\` folder
5. Run → Start Without Debugging (Ctrl+F5)

## Runtime Requirements

### Required DLLs

Ensure these DLLs are in your executable directory or system PATH:

#### OpenCV DLLs
```
opencv_world4xx.dll (e.g., opencv_world4100.dll)
opencv_videoio_ffmpeg4xx_64.dll
```
Location: `C:\opencv\build\x64\vc16\bin\`

#### ONNX Runtime DLL
```
onnxruntime.dll
```
Location: `D:\onnxruntime-win-x64-1.24.2\lib\`

### Haar Cascade File (Optional)
For face detection, download `haarcascade_frontalface_default.xml`:
- From: https://github.com/opencv/opencv/tree/master/data/haarcascades
- Place in project root or specify path in code

## Usage

### Basic Operation
1. Launch the application
2. Camera feed will display in window "DMS Euro NCAP 2026 - Step 3"
3. Press **Q** or **ESC** to quit

### Controls
- **Q** / **ESC**: Exit application
- Camera feed runs at ~30 FPS by default

### Expected Output
```
[main] ===== DMS Euro NCAP 2026 - Step 3 starting =====
[main] Signal handlers registered
[main] Creating OpenCV window...
[main] Window created OK
[main] DMS processor ready
[main] Starting pipeline on camera index 0...
[main] Pipeline started. Press q or ESC to quit.
...
[main] Stopping pipeline...
[main] Done. Captured=1234 Processed=1234 Dropped=0
```

### HMI Overlay Information
The live display shows:
- **Green Box**: Face detected, driver alert
- **Yellow Box**: Driver showing signs of drowsiness/distraction
- **Red Box**: Critical drowsiness or high distraction
- **Status Text**: Real-time gaze, drowsiness, and distraction levels

## Configuration

### Camera Settings
Edit `src/main.cpp`:
```cpp
constexpr int kCamera = 0;  // Change camera index
```

### Ring Buffer Size
Edit `src/pipeline/frame_pipeline.cpp`:
```cpp
constexpr size_t kBufferSize = 60;  // Adjust buffer capacity
```

### DMS Detection Parameters
Edit `src/dms/dms_processor.cpp`:
```cpp
// Haar cascade path
const char* cascadePath = "haarcascade_frontalface_default.xml";

// Face detection parameters
double scaleFactor = 1.1;
int minNeighbors = 5;
```

## Troubleshooting

### CMake cannot find OpenCV
```
CMake Error: Could not find OpenCV
```
**Solution**: Set `OpenCV_DIR` environment variable or pass via `-DOpenCV_DIR=...`

### CMake cannot find ONNX Runtime
```
CMake Warning: ONNX Runtime not found
```
**Solution**: Set `ONNXRUNTIME_DIR` environment variable or pass via `-DONNXRUNTIME_DIR=D:\onnxruntime-win-x64-1.24.2`

### Camera not opening
```
[CameraCapture] Failed to open camera 0
```
**Solutions**:
- Check camera is connected and not in use by another app
- Try different camera index (1, 2, etc.)
- Verify camera works in Windows Camera app

### Missing DLL errors

#### Missing OpenCV DLL
```
The code execution cannot proceed because opencv_world4xx.dll was not found
```
**Solution**: Copy OpenCV DLLs to executable directory or add to PATH:
```cmd
set PATH=%PATH%;C:\opencv\build\x64\vc16\bin
```

#### Missing ONNX Runtime DLL
```
The code execution cannot proceed because onnxruntime.dll was not found
```
**Solution**: Copy `onnxruntime.dll` to executable directory:
```cmd
copy D:\onnxruntime-win-x64-1.24.2\lib\onnxruntime.dll Release\
```

Or add to PATH:
```cmd
set PATH=%PATH%;D:\onnxruntime-win-x64-1.24.2\lib
```

### Low FPS or dropped frames
```
[main] Done. Captured=500 Processed=450 Dropped=50
```
**Solutions**:
- Reduce camera resolution in `camera_capture.cpp`
- Increase ring buffer size
- Build in Release mode (not Debug)

## Performance Metrics

Tested on **Windows 11** with **Logitech C920 webcam**:
- **Resolution**: 640x480
- **Frame Rate**: 30 FPS
- **Latency**: <50ms (capture to display)
- **CPU Usage**: ~15% (Intel i7, Release build)
- **Memory**: ~50 MB

## Next Steps - Phase 2: ONNX Model Integration

Now that ONNX Runtime is integrated into the build system, the next phase involves:

1. **Download/Convert DMS Model**: Obtain or convert a driver monitoring ONNX model
2. **Implement ONNX Inference**: Add model loading and inference in `dms_processor.cpp`
3. **Replace Haar Cascade**: Migrate from Haar cascade to deep learning-based detection
4. **Optimize Performance**: Tune inference for real-time performance

## Future Enhancements

- [ ] Phase 2: ONNX model integration for deep learning inference
- [ ] Eye tracking with dlib or MediaPipe
- [ ] Head pose estimation (pitch, yaw, roll)
- [ ] Data logging to CSV/JSON
- [ ] Configuration file support (YAML/JSON)
- [ ] Multi-camera support
- [ ] Performance profiling tools

## License

This is a proof-of-concept implementation for educational and research purposes.

## References

- **Euro NCAP 2026 Roadmap**: https://www.euroncap.com/en/vehicle-safety/the-ratings-explained/safety-assist/
- **OpenCV Documentation**: https://docs.opencv.org/
- **ONNX Runtime Documentation**: https://onnxruntime.ai/docs/
- **CMake Documentation**: https://cmake.org/documentation/

## Author

**ShaLom197926**
- GitHub: [@ShaLom197926](https://github.com/ShaLom197926)

## Acknowledgments

- OpenCV team for the computer vision library
- Microsoft for ONNX Runtime
- Euro NCAP for DMS guidelines
- CMake for cross-platform build support
