/**
 * @file ring_buffer.h
 * @brief Thread-safe ring buffer for camera frame passing between
 *        capture thread → inference thread.
 *        Pre-wired for Step 2 (multi-threaded pipeline).
 */

#ifndef DMS_RING_BUFFER_H
#define DMS_RING_BUFFER_H

#include "platform.h"

#define RING_BUFFER_CAPACITY  4   /* 4 frames deep — tunable via config later */

typedef struct {
    uint8_t*  data;          /* Raw BGR frame data                    */
    int       width;
    int       height;
    int       channels;      /* Always 3 (BGR)                        */
    double    timestamp_ms;  /* Monotonic capture timestamp (ms)      */
    uint64_t  frame_id;      /* Monotonically increasing frame counter */
    bool      valid;
} DmsFrame;

typedef struct {
    DmsFrame    frames[RING_BUFFER_CAPACITY];
    int         head;        /* Producer writes here                  */
    int         tail;        /* Consumer reads from here              */
    int         count;       /* Current occupancy                     */
    dms_mutex_t mutex;
    dms_sem_t   sem_full;    /* Signaled when a new frame is pushed   */
    dms_sem_t   sem_empty;   /* Signaled when a slot is freed         */
    bool        shutdown;    /* Signals consumer threads to exit      */
} DmsRingBuffer;

/**
 * @brief Allocate and initialize ring buffer. Allocates frame memory.
 * @param rb       Pointer to ring buffer
 * @param width    Frame width in pixels
 * @param height   Frame height in pixels
 * @return 0 on success, -1 on allocation failure
 */
#ifdef __cplusplus
extern "C" {
#endif

int  dms_ring_buffer_init(DmsRingBuffer* rb, int width, int height);

void dms_ring_buffer_push(DmsRingBuffer* rb, const uint8_t* bgr_data,
                          int width, int height,
                          double ts_ms, uint64_t frame_id);

bool dms_ring_buffer_pop(DmsRingBuffer* rb, DmsFrame* out_frame);

void dms_ring_buffer_destroy(DmsRingBuffer* rb);

#ifdef __cplusplus
}
#endif

#endif /* DMS_RING_BUFFER_H */
