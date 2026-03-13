#ifndef DMS_CAMERA_CAPTURE_H
#define DMS_CAMERA_CAPTURE_H

#include <stdint.h>
#include <stdbool.h>

typedef struct DmsCameraHandle_t DmsCameraHandle;

typedef struct {
    int  camera_id;
    int  width;
    int  height;
    int  fps;
    char pixel_format[16];
    bool use_directshow;
} DmsCameraConfig;

typedef struct {
    int    width;
    int    height;
    double actual_fps;
    char   backend_name[64];
    bool   is_ir_camera;
} DmsCameraInfo;

#ifdef __cplusplus
extern "C" {
#endif

DmsCameraHandle* dms_camera_open(const DmsCameraConfig* cfg);
void             dms_camera_get_info(DmsCameraHandle* handle, DmsCameraInfo* info);
bool             dms_camera_read_frame(DmsCameraHandle* handle, uint8_t* bgr_out, double* ts_ms_out);
void             dms_camera_close(DmsCameraHandle* handle);
int              dms_camera_enumerate(int* out_indices, int max_count);

#ifdef __cplusplus
}
#endif

#endif /* DMS_CAMERA_CAPTURE_H */
