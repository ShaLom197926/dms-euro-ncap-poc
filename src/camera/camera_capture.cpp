/**
 * @file camera_capture.cpp
 * @brief OpenCV-backed UVC camera capture implementation.
 *        Uses the OpenCV C++ API wrapped in a C-compatible interface.
 *        OpenCV's VideoCapture handles DirectShow (Windows) and
 *        V4L2 (Linux/QNX) automatically based on build flags.
 */

#include "camera_capture.h"
//#include "../logger/dms_logger.h"
// Inline logger stub — avoids dependency on dms_logger.c
#include <windows.h>
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <inttypes.h>

typedef enum { DMS_LOG_DEBUG=0, DMS_LOG_INFO, DMS_LOG_WARN,
               DMS_LOG_ERROR, DMS_LOG_FATAL } DmsLogLevel;

static inline void dms_log(DmsLogLevel lvl, const char* mod, const char* fmt, ...) {
    const char* tags[] = {"DBG","INF","WRN","ERR","FTL"};
    printf("[%s][%-10s] ", tags[lvl], mod);
    va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
    printf("\n"); fflush(stdout);
}
#define DMS_LOGD(m,...) dms_log(DMS_LOG_DEBUG, m, __VA_ARGS__)
#define DMS_LOGI(m,...) dms_log(DMS_LOG_INFO,  m, __VA_ARGS__)
#define DMS_LOGW(m,...) dms_log(DMS_LOG_WARN,  m, __VA_ARGS__)
#define DMS_LOGE(m,...) dms_log(DMS_LOG_ERROR, m, __VA_ARGS__)

// Timing stub
static inline double dms_get_time_ms(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart * 1000.0 / (double)freq.QuadPart;
}


/* OpenCV headers — C++ API */
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

struct DmsCameraHandle_t {
    cv::VideoCapture cap;
    DmsCameraConfig  cfg;
    uint64_t         frame_count;
};

/* NOTE:
 *  extern "C" is applied only in the header (camera_capture.h)
 *  around the function declarations. Definitions here are plain C++
 *  to avoid linkage specification conflicts.
 */

DmsCameraHandle* dms_camera_open(const DmsCameraConfig* cfg)
{
    DMS_LOGI("CAMERA", "Opening camera index=%d at %dx%d @%d fps format=%s",
             cfg->camera_id, cfg->width, cfg->height, cfg->fps, cfg->pixel_format);

    DmsCameraHandle* handle = new DmsCameraHandle_t();
    handle->cfg          = *cfg;
    handle->frame_count  = 0;

    /* Select backend: DirectShow on Windows for lower latency with UVC cams */
#if defined(DMS_PLATFORM_WINDOWS)
    int api = cv::CAP_DSHOW;   /* DirectShow — best UVC support on Win11    */
#elif defined(DMS_PLATFORM_LINUX) || defined(DMS_PLATFORM_QNX)
    int api = cv::CAP_V4L2;
#else
    int api = cv::CAP_ANY;
#endif

    handle->cap.open(cfg->camera_id, api);

    if (!handle->cap.isOpened()) {
        DMS_LOGW("CAMERA", "DirectShow open failed, retrying with CAP_ANY...");
        handle->cap.open(cfg->camera_id, cv::CAP_ANY);
        if (!handle->cap.isOpened()) {
            DMS_LOGE("CAMERA", "FATAL: Cannot open camera index=%d", cfg->camera_id);
            delete handle;
            return NULL;
        }
    }

    /* Set pixel format BEFORE resolution/fps */
    if (strcmp(cfg->pixel_format, "MJPG") == 0) {
        handle->cap.set(cv::CAP_PROP_FOURCC,
                        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        DMS_LOGI("CAMERA", "FOURCC set to MJPG");
    } else if (strcmp(cfg->pixel_format, "YUYV") == 0) {
        handle->cap.set(cv::CAP_PROP_FOURCC,
                        cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
        DMS_LOGI("CAMERA", "FOURCC set to YUYV");
    }

    handle->cap.set(cv::CAP_PROP_FRAME_WIDTH,  cfg->width);
    handle->cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg->height);
    handle->cap.set(cv::CAP_PROP_FPS,          cfg->fps);

    /* Disable auto-buffering — minimize latency */
    handle->cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    /* Verify what we actually got */
    double actual_w   = handle->cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double actual_h   = handle->cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double actual_fps = handle->cap.get(cv::CAP_PROP_FPS);

    DMS_LOGI("CAMERA", "Camera opened: actual resolution=%dx%d @%.1f fps",
             (int)actual_w, (int)actual_h, actual_fps);

    return handle;
}

void dms_camera_get_info(DmsCameraHandle* handle, DmsCameraInfo* info)
{
    if (!handle || !info) return;

    info->width      = (int)handle->cap.get(cv::CAP_PROP_FRAME_WIDTH);
    info->height     = (int)handle->cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    info->actual_fps = handle->cap.get(cv::CAP_PROP_FPS);
    snprintf(info->backend_name, sizeof(info->backend_name),
             "OpenCV-DirectShow");
    info->is_ir_camera = false;  /* Detected in Step 2 via mean luminance check */
}

bool dms_camera_read_frame(DmsCameraHandle* handle,
                           uint8_t*         bgr_out,
                           double*          ts_ms_out)
{
    if (!handle || !bgr_out) return false;

    cv::Mat frame;
    if (!handle->cap.read(frame)) {
        DMS_LOGE("CAMERA", "cap.read() failed at frame %" PRIu64,
                 (unsigned long long)handle->frame_count);
        return false;
    }

    *ts_ms_out = dms_get_time_ms();
    handle->frame_count++;

    /* Ensure BGR output (OpenCV default is BGR — just copy) */
    if (frame.channels() == 1) {
        cv::Mat bgr;
        cv::cvtColor(frame, bgr, cv::COLOR_GRAY2BGR);
        memcpy(bgr_out, bgr.data,
               (size_t)(bgr.cols * bgr.rows * 3));
    } else if (frame.channels() == 3) {
        memcpy(bgr_out, frame.data,
               (size_t)(frame.cols * frame.rows * 3));
    } else {
        DMS_LOGE("CAMERA", "Unexpected frame channels: %d", frame.channels());
        return false;
    }

    return true;
}

void dms_camera_close(DmsCameraHandle* handle)
{
    if (!handle) return;

    DMS_LOGI("CAMERA", "Closing camera. Total frames captured: %" PRIu64,
             (unsigned long long)handle->frame_count);

    handle->cap.release();
    delete handle;
}

int dms_camera_enumerate(int* out_indices, int max_count)
{
    int found = 0;
    DMS_LOGI("CAMERA", "Enumerating available cameras (max index 8)...");

    for (int i = 0; i < 8 && found < max_count; i++) {
        cv::VideoCapture test;
        test.open(i, cv::CAP_ANY);
        if (test.isOpened()) {
            out_indices[found++] = i;
            DMS_LOGI("CAMERA", "  [FOUND] Camera index %d", i);
            test.release();
        }
    }

    DMS_LOGI("CAMERA", "Enumeration complete: %d camera(s) found", found);
    return found;
}
