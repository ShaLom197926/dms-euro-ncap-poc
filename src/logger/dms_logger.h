/**
 * @file dms_logger.h
 * @brief Lightweight async-capable logger with level filtering.
 *        Outputs to stdout + optionally to rotating CSV log file.
 *        Euro NCAP SD-201 requires persistent event logging.
 */

#ifndef DMS_LOGGER_H
#define DMS_LOGGER_H

#include "../utils/platform.h"

typedef enum {
    DMS_LOG_DEBUG = 0,
    DMS_LOG_INFO,
    DMS_LOG_WARN,
    DMS_LOG_ERROR,
    DMS_LOG_FATAL
} DmsLogLevel;

/**
 * @brief Initialize logger. Creates log directory if needed.
 * @param log_dir   Directory path for log files (NULL = stdout only)
 * @param level     Minimum log level to emit
 */
#ifdef __cplusplus
extern "C" {
#endif

void dms_logger_init(const char* log_dir, DmsLogLevel level);
void dms_log(DmsLogLevel level, const char* module, const char* fmt, ...);
void dms_logger_shutdown(void);

#ifdef __cplusplus
}
#endif

#define DMS_LOGD(mod, ...) dms_log(DMS_LOG_DEBUG, mod, __VA_ARGS__)
#define DMS_LOGI(mod, ...) dms_log(DMS_LOG_INFO,  mod, __VA_ARGS__)
#define DMS_LOGW(mod, ...) dms_log(DMS_LOG_WARN,  mod, __VA_ARGS__)
#define DMS_LOGE(mod, ...) dms_log(DMS_LOG_ERROR, mod, __VA_ARGS__)
#define DMS_LOGF(mod, ...) dms_log(DMS_LOG_FATAL, mod, __VA_ARGS__)

#endif /* DMS_LOGGER_H */

