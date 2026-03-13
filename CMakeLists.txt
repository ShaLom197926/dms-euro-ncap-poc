cmake_minimum_required(VERSION 3.18)
project(dms_app LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Sources ------------------------------------------------
add_executable(dms_app
    src/main.cpp
    src/camera/camera_capture.cpp
    src/pipeline/frame_pipeline.cpp  # Step 2: producer/consumer pipeline
)

# ---- OpenCV -------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio imgcodecs)

# ---- Threads (std::thread needs this on Linux; no-op on MSVC) ---
find_package(Threads REQUIRED)

# ---- Include directories ------------------------------------
target_include_directories(dms_app PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    ${OpenCV_INCLUDE_DIRS}
)

# ---- Link libraries -----------------------------------------
target_link_libraries(dms_app PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
)

# ---- MSVC-specific flags ------------------------------------
if(MSVC)
    target_compile_definitions(dms_app PRIVATE
        _CRT_SECURE_NO_WARNINGS
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        DMS_PLATFORM_WINDOWS
    )
    target_compile_options(dms_app PRIVATE /W3 /WX- /wd4996 /wd4005)
endif()

# ---- Output to bin/ -----------------------------------------
set_target_properties(dms_app PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
