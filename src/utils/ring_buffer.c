#include "ring_buffer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int dms_ring_buffer_init(DmsRingBuffer* rb, int width, int height)
{
    memset(rb, 0, sizeof(DmsRingBuffer));
    rb->head = 0;
    rb->tail = 0;
    rb->count = 0;
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
    DMS_SEM_INIT(rb->sem_full,  0);
    DMS_SEM_INIT(rb->sem_empty, RING_BUFFER_CAPACITY);
    return 0;
}

void dms_ring_buffer_push(DmsRingBuffer* rb, const uint8_t* bgr_data,
                           int width, int height, double ts_ms, uint64_t frame_id)
{
    DMS_MUTEX_LOCK(rb->mutex);

    if (rb->count == RING_BUFFER_CAPACITY) {
        /* Drop oldest frame — camera must never stall */
        rb->tail = (rb->tail + 1) % RING_BUFFER_CAPACITY;
        rb->count--;
        DMS_SEM_WAIT(rb->sem_full);   /* Rebalance semaphore count */
    }

    DmsFrame* slot = &rb->frames[rb->head];
    memcpy(slot->data, bgr_data, (size_t)(width * height * 3));
    slot->width        = width;
    slot->height       = height;
    slot->timestamp_ms = ts_ms;
    slot->frame_id     = frame_id;
    slot->valid        = true;

    rb->head  = (rb->head + 1) % RING_BUFFER_CAPACITY;
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
    slot->valid = false;

    rb->tail  = (rb->tail + 1) % RING_BUFFER_CAPACITY;
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

    /* Unblock any waiting consumers */
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
