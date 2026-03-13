#include "dms_logger.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static DmsLogLevel g_log_level = DMS_LOG_INFO;
static FILE*       g_log_file  = NULL;

static const char* level_str[] = { "DEBUG", "INFO ", "WARN ", "ERROR", "FATAL" };
static const char* level_color[] = {
    "\033[37m",   /* DEBUG — white  */
    "\033[32m",   /* INFO  — green  */
    "\033[33m",   /* WARN  — yellow */
    "\033[31m",   /* ERROR — red    */
    "\033[35m"    /* FATAL — magenta*/
};
#define COLOR_RESET "\033[0m"

void dms_logger_init(const char* log_dir, DmsLogLevel level)
{
    g_log_level = level;
    if (log_dir) {
        char path[512];
        time_t now = time(NULL);
        struct tm* t = localtime(&now);
#if defined(DMS_PLATFORM_WINDOWS)
        CreateDirectoryA(log_dir, NULL);
        snprintf(path, sizeof(path), "%s\\dms_%04d%02d%02d_%02d%02d%02d.log",
                 log_dir,
                 t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                 t->tm_hour, t->tm_min, t->tm_sec);
#else
        snprintf(path, sizeof(path), "%s/dms_%04d%02d%02d_%02d%02d%02d.log",
                 log_dir,
                 t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                 t->tm_hour, t->tm_min, t->tm_sec);
#endif
        g_log_file = fopen(path, "w");
        if (!g_log_file) {
            fprintf(stderr, "[LOGGER] WARNING: Cannot open log file: %s\n", path);
        }
    }
}

void dms_log(DmsLogLevel level, const char* module, const char* fmt, ...)
{
    if (level < g_log_level) return;

    double ts_ms = dms_get_time_ms();

    char msg[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);

    /* Console output with ANSI color (Windows 10+ supports VT100) */
    fprintf(stdout, "%s[%s][%12.3f ms][%-12s] %s%s\n",
            level_color[level], level_str[level], ts_ms, module, msg, COLOR_RESET);
    fflush(stdout);

    /* File output (no color codes) */
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
