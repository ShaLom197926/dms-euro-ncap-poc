#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <signal.h>
#include <inttypes.h>

/* Platform: Windows-only for now */
#include <windows.h>

/* ── Timing ───────────────────────────────────────────────────────────────── */
static inline double dms_get_time_ms(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#define DMS_SLEEP_MS(ms) Sleep(ms)
#define DMS_UNUSED(x)    (void)(x)
#define DMS_PATH_SEP     "\\"

/* ── Logger stubs (no external .c needed) ─────────────────────────────────── */
typedef enum { DMS_LOG_DEBUG=0, DMS_LOG_INFO, DMS_LOG_WARN,
               DMS_LOG_ERROR, DMS_LOG_FATAL } DmsLogLevel;

static inline void dms_logger_init(const char* d, DmsLogLevel l) { (void)d; (void)l; }
static inline void dms_logger_shutdown(void) {}

static inline void dms_log_impl(DmsLogLevel lvl, const char* mod, const char* fmt, ...) {
    const char* tags[] = {"DBG","INF","WRN","ERR","FTL"};
    printf("[%s][%-10s] ", tags[lvl], mod);
    va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
    printf("\n"); fflush(stdout);
}
#define DMS_LOGD(m,...) dms_log_impl(DMS_LOG_DEBUG, m, __VA_ARGS__)
#define DMS_LOGI(m,...) dms_log_impl(DMS_LOG_INFO,  m, __VA_ARGS__)
#define DMS_LOGW(m,...) dms_log_impl(DMS_LOG_WARN,  m, __VA_ARGS__)
#define DMS_LOGE(m,...) dms_log_impl(DMS_LOG_ERROR, m, __VA_ARGS__)
#define DMS_LOGF(m,...) dms_log_impl(DMS_LOG_FATAL, m, __VA_ARGS__)

/* ── Ring-buffer stubs (no external .c needed) ────────────────────────────── */
#define RING_BUFFER_CAPACITY 1
typedef struct { int dummy; } DmsRingBuffer;

static inline int  dms_ring_buffer_init(DmsRingBuffer* r, int w, int h)
    { (void)r;(void)w;(void)h; return 0; }
static inline void dms_ring_buffer_push(DmsRingBuffer* r, const uint8_t* d,
    int w, int h, double t, uint64_t f)
    { (void)r;(void)d;(void)w;(void)h;(void)t;(void)f; }
static inline void dms_ring_buffer_destroy(DmsRingBuffer* r) { (void)r; }

/* ── Camera (real implementation) ─────────────────────────────────────────── */
#include "camera/camera_capture.h"

/* ── OpenCV display ───────────────────────────────────────────────────────── */
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

/* ── Globals ──────────────────────────────────────────────────────────────── */
static volatile bool g_running = true;

static BOOL WINAPI ctrl_handler(DWORD t) {
    if (t == CTRL_C_EVENT || t == CTRL_BREAK_EVENT) { g_running = false; return TRUE; }
    return FALSE;
}

/* ── Config ───────────────────────────────────────────────────────────────── */
typedef struct {
    int  camera_id, width, height, fps;
    char pixel_format[16];
    bool display_window;
} DmsConfig;

static void config_defaults(DmsConfig* c) {
    c->camera_id = 0; c->width = 1280; c->height = 720; c->fps = 30;
    strncpy(c->pixel_format, "MJPG", sizeof(c->pixel_format)-1);
    c->display_window = true;
}

/* ── FPS calc ─────────────────────────────────────────────────────────────── */
typedef struct { double ts[64]; int idx, count; } FpsCalc;
static void fps_push(FpsCalc* f, double t) {
    f->ts[f->idx%64]=t; f->idx++;
    if(f->count<64) f->count++;
}
static double fps_get(FpsCalc* f) {
    if(f->count<2) return 0.0;
    double dt = f->ts[(f->idx-1+64)%64] - f->ts[(f->idx-f->count+64)%64];
    return dt>0 ? (f->count-1)/(dt/1000.0) : 0.0;
}

/* ── HUD ──────────────────────────────────────────────────────────────────── */
static void draw_hud(cv::Mat& fr, double fps, uint64_t fid, int w, int h) {
    char buf[128];
    cv::Mat ov = fr.clone();
    cv::rectangle(ov,{0,0},{fr.cols,44},cv::Scalar(0,0,0),cv::FILLED);
    cv::addWeighted(ov,0.5,fr,0.5,0,fr);
    snprintf(buf,sizeof(buf),"FPS: %.1f",fps);
    cv::putText(fr,buf,{10,30},cv::FONT_HERSHEY_SIMPLEX,0.8,{0,255,0},2);
    snprintf(buf,sizeof(buf),"Frame: %" PRIu64,(unsigned long long)fid);
    cv::putText(fr,buf,{180,30},cv::FONT_HERSHEY_SIMPLEX,0.8,{0,255,255},2);
    snprintf(buf,sizeof(buf),"%dx%d",w,h);
    cv::putText(fr,buf,{420,30},cv::FONT_HERSHEY_SIMPLEX,0.8,{200,200,200},2);
    cv::putText(fr,"DMS STEP-1: CAMERA CAPTURE [OK]",
        {10,fr.rows-15},cv::FONT_HERSHEY_SIMPLEX,0.65,{0,200,255},2);
    cv::putText(fr,"Euro NCAP 2026 DMS PoC",
        {fr.cols-320,fr.rows-15},cv::FONT_HERSHEY_SIMPLEX,0.55,{100,100,100},1);
}

/* ── Main ─────────────────────────────────────────────────────────────────── */
int main(int argc, char* argv[]) {
    DMS_UNUSED(argc); DMS_UNUSED(argv);

    /* Enable ANSI colors */
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0; GetConsoleMode(hOut,&dwMode);
    SetConsoleMode(hOut, dwMode|ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    SetConsoleCtrlHandler(ctrl_handler, TRUE);

    dms_logger_init("logs", DMS_LOG_INFO);
    DMS_LOGI("MAIN","DMS Application Step-1 starting...");

    /* Enumerate */
    int cam_idx[8];
    int n = dms_camera_enumerate(cam_idx, 8);
    if (n == 0) { DMS_LOGE("MAIN","No camera found."); return EXIT_FAILURE; }

    /* Config */
    DmsConfig cfg; config_defaults(&cfg);

    /* Open camera */
    DmsCameraConfig cc;
    memset(&cc, 0, sizeof(cc));
    cc.camera_id = cfg.camera_id;
    cc.width     = cfg.width;
    cc.height    = cfg.height;
    cc.fps       = cfg.fps;
    cc.use_directshow = true;
    strncpy(cc.pixel_format, cfg.pixel_format, sizeof(cc.pixel_format)-1);

    DmsCameraHandle* cam = dms_camera_open(&cc);
    if (!cam) { DMS_LOGE("MAIN","Camera open failed."); return EXIT_FAILURE; }

    DmsCameraInfo info;
    dms_camera_get_info(cam, &info);
    DMS_LOGI("MAIN","Camera: %dx%d @%.1f fps [%s]",
             info.width, info.height, info.actual_fps, info.backend_name);

    /* Frame buffer */
    uint8_t* buf = (uint8_t*)malloc((size_t)(info.width * info.height * 3));
    if (!buf) { dms_camera_close(cam); return EXIT_FAILURE; }

    /* Stub ring buffer */
    DmsRingBuffer rb;
    dms_ring_buffer_init(&rb, info.width, info.height);

    /* Window */
    const char* WIN = "DMS Euro NCAP 2026";
    cv::namedWindow(WIN, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::resizeWindow(WIN, info.width, info.height);
    DMS_LOGI("MAIN","Press Q or ESC to exit.");

    FpsCalc fpc = {0};
    uint64_t fid = 0;
    double ts = 0.0, t0, t1, lat = 0.0, last_log = dms_get_time_ms();

    while (g_running) {
        t0 = dms_get_time_ms();

        if (!dms_camera_read_frame(cam, buf, &ts)) {
            DMS_LOGE("MAIN","Frame read failed"); DMS_SLEEP_MS(10); continue;
        }
        fid++;
        fps_push(&fpc, ts);
        double fps = fps_get(&fpc);
        dms_ring_buffer_push(&rb, buf, info.width, info.height, ts, fid);

        cv::Mat fr(info.height, info.width, CV_8UC3, buf);
        cv::Mat disp = fr.clone();
        draw_hud(disp, fps, fid, info.width, info.height);
        cv::imshow(WIN, disp);

        int k = cv::waitKey(1);
        if (k=='q'||k=='Q'||k==27) { g_running=false; }
        if (k=='s'||k=='S') {
            char p[256];
            snprintf(p,sizeof(p),"snap_%" PRIu64 ".jpg",(unsigned long long)fid);
            cv::imwrite(p, disp);
            DMS_LOGI("MAIN","Snapshot: %s", p);
        }

        t1 = dms_get_time_ms();
        lat += (t1 - t0);
        if ((t1 - last_log) >= 5000.0) {
            DMS_LOGI("MAIN","Perf: frames=%" PRIu64 " fps=%.1f lat=%.2fms",
                     (unsigned long long)fid, fps,
                     fid>0?lat/(double)fid:0.0);
            last_log = t1;
        }
    }

    DMS_LOGI("MAIN","Total frames: %" PRIu64, (unsigned long long)fid);
    cv::destroyAllWindows();
    dms_ring_buffer_destroy(&rb);
    free(buf);
    dms_camera_close(cam);
    dms_logger_shutdown();
    return EXIT_SUCCESS;
}
