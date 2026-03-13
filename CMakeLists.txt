cmake_minimum_required(VERSION 3.18)
project(dms_app LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Sources -------------------------------------------------------
add_executable(dms_app
    src/main.cpp
    src/camera/camera_capture.cpp
    src/pipeline/frame_pipeline.cpp  # Step 2: producer/consumer pipeline
        src/dms/dms_processor.cpp     # Step 3: DMS inference + overlay
)

# ---- OpenCV --------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio imgcodecs)

# ---- ONNX Runtime (Step 4: YOLO-Pose inference) --------------------
set(ONNXRUNTIME_ROOT "D:/onnxruntime-win-x64-1.24.2")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")
set(ONNXRUNTIME_BIN_DIR "${ONNXRUNTIME_ROOT}/bin")

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIRS}/onnxruntime_cxx_api.h")
    message(WARNING 
        "[DMS] ONNX Runtime not found at ${ONNXRUNTIME_ROOT}\n"
        "      Download from: https://github.com/microsoft/onnxruntime/releases\n"
        "      Extract to D:/onnxruntime-win-x64-1.24.2/")
endif()

include_directories(${ONNXRUNTIME_INCLUDE_DIRS})

# ---- Threads (std::thread needs this on Linux; no-op on MSVC) ------
find_package(Threads REQUIRED)

# ---- Include directories -------------------------------------------
target_include_directories(dms_app PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    ${OpenCV_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIRS}
)

# ---- Link libraries ------------------------------------------------
target_link_libraries(dms_app PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
        "${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib"
    "${ONNXRUNTIME_LIB_DIR}/onnxruntime_providers_shared.lib"
)

# ---- MSVC-specific flags -------------------------------------------
if(MSVC)
    target_compile_definitions(dms_app PRIVATE
        _CRT_SECURE_NO_WARNINGS
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        DMS_PLATFORM_WINDOWS
    )
    target_compile_options(dms_app PRIVATE /W3 /WX- /wd4996 /wd4005)
endif()

# ---- Output to bin/ ------------------------------------------------
set_target_properties(dms_app PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ---- Windows: auto-copy OpenCV DLLs next to EXE after build --------
# This ensures the EXE can find opencv_world*.dll without modifying PATH.
if(WIN32)
    # Collect all OpenCV DLL paths from imported targets
    foreach(_comp ${OpenCV_LIBS})
        if(TARGET ${_comp})
            get_target_property(_imp ${_comp} IMPORTED_LOCATION_RELEASE)
            if(_imp)
                # Replace .lib with .dll (for MSVC import libs)
                string(REPLACE ".lib" ".dll" _dll "${_imp}")
                # Also handle the case where IMPORTED_LOCATION already is a dll
                if(NOT _dll MATCHES "\\.dll$")
                    set(_dll "")
                endif()
                if(_dll AND EXISTS "${_dll}")
                    list(APPEND _opencv_dlls "${_dll}")
                endif()
            endif()
        endif()
    endforeach()

    # Fallback: glob the OpenCV bin directory for all opencv_world*.dll
    if(NOT _opencv_dlls)
        get_filename_component(_ocv_lib_dir "${OpenCV_DIR}" ABSOLUTE)
        # Common layout: OpenCVConfig.cmake is in <opencv_build>/  
        # DLLs are in <opencv_build>/bin/ or x64/vc*/bin/
        file(GLOB _opencv_dlls_glob
            "${_ocv_lib_dir}/bin/*.dll"
            "${_ocv_lib_dir}/x64/vc*/bin/*.dll"
            "${_ocv_lib_dir}/../bin/*.dll"
        )
        list(APPEND _opencv_dlls ${_opencv_dlls_glob})
    endif()

    if(_opencv_dlls)
        add_custom_command(TARGET dms_app POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "[cmake] Copying OpenCV DLLs to EXE directory..."
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_opencv_dlls}
                "$<TARGET_FILE_DIR:dms_app>"
            COMMENT "Copying OpenCV runtime DLLs next to dms_app.exe"
        )
    else()
        message(WARNING
            "[DMS] Could not locate OpenCV DLLs for auto-copy.\n"
            "      After building, manually copy opencv_world*.dll from\n"
            "      D:/opencv/build/x64/vc16/bin/  next to dms_app.exe,\n"
            "      or add that directory to your PATH before running.")
    endif()

        # ---- Copy ONNX Runtime DLL next to EXE ---------------------------
    if(EXISTS "${ONNXRUNTIME_BIN_DIR}/onnxruntime.dll")
        add_custom_command(TARGET dms_app POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "[cmake] Copying ONNX Runtime DLL to EXE directory..."
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_BIN_DIR}/onnxruntime.dll"
                "$<TARGET_FILE_DIR:dms_app>"
            COMMENT "Copying ONNX Runtime DLL next to dms_app.exe"
        )
    else()
        message(WARNING
            "[DMS] onnxruntime.dll not found at ${ONNXRUNTIME_BIN_DIR}\n"
            "      The EXE will fail to run without this DLL.")
    endif()
endif()
