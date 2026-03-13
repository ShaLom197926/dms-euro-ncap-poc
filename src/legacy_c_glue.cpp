// legacy_c_glue.cpp
// Temporary workaround: compile C modules as C++ with C linkage.

extern "C" {

#include "utils/ring_buffer.h"
#include "logger/dms_logger.h"

// === ring_buffer.c contents ===
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int dms_ring_buffer_init(DmsRingBuffer* rb, int width, int height)
{
    memset(rb, 0, sizeof(DmsRingBuffer));
    rb->head     = 0;
    rb->tail     = 0;
    rb->count    = 0;
    rb->shutdown = false;

    for (int i = 0; i < RING_BUFFER_CAPACITY; i++) {
        rb->frames[i].data = (uint8_t*)malloc((size_t)(width * height * 3));
        if (!rb->frames[i].data) {
            fprintf(stderr, "[RingBuffer] ERROR: malloc failed for frame %d\n", i);
            return -1;
        }
        rb->frames[i].width    = width;
        rb->frames[i].height   = height;
        rb->frames[i].channels = 3;
        rb->frames[i].valid    = false;
    }

    DMS_MUTEX_INIT(rb->mutex);
    DMS_SEM_INIT(rb->sem_full, 0);
    DMS_SEM_INIT(rb->sem_empty, RING_BUFFER_CAPACITY);
    return 0;
}

void dms_ring_buffer_push(DmsRingBuffer* rb, const uint8_t* bgr_data,
                          int width, int height, double ts_ms, uint64_t frame_id)
{
    DMS_MUTEX_LOCK(rb->mutex);

    if (rb->count == RING_BUFFER_CAPACITY) {
        rb->tail = (rb->tail + 1) % RING_BUFFER_CAPACITY;
        rb->count--;
        DMS_SEM_WAIT(rb->sem_full);
    }

    DmsFrame* slot = &rb->frames[rb->head];
    memcpy(slot->data, bgr_data, (size_t)(width * height * 3));
    slot->width        = width;
    slot->height       = height;
    slot->timestamp_ms = ts_ms;
    slot->frame_id     = frame_id;
    slot->valid        = true;

    rb->head = (rb->head + 1) % RING_BUFFER_CAPACITY;
    rb->count++;

    DMS_MUTEX_UNLOCK(rb->mutex);
    DMS_SEM_POST(rb->sem_full);
}

bool dms_ring_buffer_pop(DmsRingBuffer* rb, DmsFrame* out_frame)
{
    DMS_SEM_WAIT(rb->sem_full);

    DMS_MUTEX_LOCK(rb->mutex);
    if (rb->shutdown && rb->count == 0) {
        DMS_MUTEX_UNLOCK(rb->mutex);
        return false;
    }

    DmsFrame* slot = &rb->frames[rb->tail];
    memcpy(out_frame->data, slot->data,
           (size_t)(slot->width * slot->height * 3));
    out_frame->width        = slot->width;
    out_frame->height       = slot->height;
    out_frame->timestamp_ms = slot->timestamp_ms;
    out_frame->frame_id     = slot->frame_id;
    out_frame->valid        = true;
    slot->valid             = false;

    rb->tail = (rb->tail + 1) % RING_BUFFER_CAPACITY;
    rb->count--;

    DMS_MUTEX_UNLOCK(rb->mutex);
    DMS_SEM_POST(rb->sem_empty);
    return true;
}

void dms_ring_buffer_destroy(DmsRingBuffer* rb)
{
    DMS_MUTEX_LOCK(rb->mutex);
    rb->shutdown = true;
    DMS_MUTEX_UNLOCK(rb->mutex);

    for (int i = 0; i < RING_BUFFER_CAPACITY; i++) {
        DMS_SEM_POST(rb->sem_full);
    }

    for (int i = 0; i < RING_BUFFER_CAPACITY; i++) {
        if (rb->frames[i].data) {
            free(rb->frames[i].data);
            rb->frames[i].data = NULL;
        }
    }

    DMS_MUTEX_DESTROY(rb->mutex);
    DMS_SEM_DESTROY(rb->sem_full);
    DMS_SEM_DESTROY(rb->sem_empty);
}

// === dms_logger.c contents ===
#include <stdio.h>
#include <stdarg.h>
#include <time.h>

static DmsLogLevel g_log_level = DMS_LOG_INFO;
static FILE*       g_log_file  = NULL;

static const char* level_str[] = { "DEBUG", "INFO ", "WARN ", "ERROR", "FATAL" };
static const char* level_color[] = {
    "\033[37m",
    "\033[32m",
    "\033[33m",
    "\033[31m",
    "\033[35m"
};
#define COLOR_RESET "\033[0m"

void dms_logger_init(const char* log_dir, DmsLogLevel level)
{
    g_log_level = level;
    if (log_dir) {
        char    path[512];
        time_t  now = time(NULL);
        struct tm* t = localtime(&now);
#if defined(DMS_PLATFORM_WINDOWS)
        CreateDirectoryA(log_dir, NULL);
        snprintf(path, sizeof(path),
                 "%s\\dms_%04d%02d%02d_%02d%02d%02d.log",
                 log_dir,
                 t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                 t->tm_hour, t->tm_min, t->tm_sec);
#else
        snprintf(path, sizeof(path),
                 "%s/dms_%04d%02d%02d_%02d%02d%02d.log",
                 log_dir,
                 t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                 t->tm_hour, t->tm_min, t->tm_sec);
#endif
        g_log_file = fopen(path, "w");
        if (!g_log_file) {
            fprintf(stderr,
                    "[LOGGER] WARNING: Cannot open log file: %s\n",
                    path);
        }
    }
}

void dms_log(DmsLogLevel level, const char* module, const char* fmt, ...)
{
    if (level < g_log_level) return;

    double ts_ms = dms_get_time_ms();

    char    msg[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);

    fprintf(stdout, "%s[%s][%12.3f ms][%-12s] %s%s\n",
            level_color[level], level_str[level],
            ts_ms, module, msg, COLOR_RESET);
    fflush(stdout);

    if (g_log_file) {
        fprintf(g_log_file, "[%s][%12.3f ms][%-12s] %s\n",
                level_str[level], ts_ms, module, msg);
        fflush(g_log_file);
    }
}

void dms_logger_shutdown(void)
{
    if (g_log_file) {
        fclose(g_log_file);
        g_log_file = NULL;
    }
}

} // extern "C"
