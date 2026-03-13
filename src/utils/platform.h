/**
 * @file platform.h
 * @brief Platform abstraction layer for Windows/Linux/QNX portability.
 *        DMS Application - Euro NCAP 2026 Compliant
 *        Target: Windows 11 (MSVC) -> Qualcomm SA8255 (QNX aarch64)
 */

#ifndef DMS_PLATFORM_H
#define DMS_PLATFORM_H

/* ─── Compiler / OS Detection ─────────────────────────────────────────────── */
#if defined(_WIN32) || defined(_WIN64)
    #define DMS_PLATFORM_WINDOWS
    #define WIN32_LEAN_AND_MEAN
    #define NOMINMAX
    #include <windows.h>
    #include <winsock2.h>
#elif defined(__QNX__)
    #define DMS_PLATFORM_QNX
    #include <sys/neutrino.h>
    #include <sys/syspage.h>
#elif defined(__linux__)
    #define DMS_PLATFORM_LINUX
#else
    #error "Unsupported platform"
#endif

/* ─── Standard Includes (all platforms) ───────────────────────────────────── */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <stdbool.h>

/* ─── Threading Abstraction ───────────────────────────────────────────────── */
#if defined(DMS_PLATFORM_WINDOWS)
    #include <process.h>
    typedef HANDLE          dms_thread_t;
    typedef CRITICAL_SECTION dms_mutex_t;
    typedef HANDLE          dms_sem_t;

    #define DMS_MUTEX_INIT(m)    InitializeCriticalSection(&(m))
    #define DMS_MUTEX_LOCK(m)    EnterCriticalSection(&(m))
    #define DMS_MUTEX_UNLOCK(m)  LeaveCriticalSection(&(m))
    #define DMS_MUTEX_DESTROY(m) DeleteCriticalSection(&(m))

    #define DMS_SEM_INIT(s, val)  ((s) = CreateSemaphore(NULL, (val), INT32_MAX, NULL))
    #define DMS_SEM_WAIT(s)       WaitForSingleObject((s), INFINITE)
    #define DMS_SEM_POST(s)       ReleaseSemaphore((s), 1, NULL)
    #define DMS_SEM_DESTROY(s)    CloseHandle((s))

#else
    /* POSIX: Linux + QNX */
    #include <pthread.h>
    #include <semaphore.h>
    typedef pthread_t   dms_thread_t;
    typedef pthread_mutex_t dms_mutex_t;
    typedef sem_t       dms_sem_t;

    #define DMS_MUTEX_INIT(m)    pthread_mutex_init(&(m), NULL)
    #define DMS_MUTEX_LOCK(m)    pthread_mutex_lock(&(m))
    #define DMS_MUTEX_UNLOCK(m)  pthread_mutex_unlock(&(m))
    #define DMS_MUTEX_DESTROY(m) pthread_mutex_destroy(&(m))

    #define DMS_SEM_INIT(s, val)  sem_init(&(s), 0, (val))
    #define DMS_SEM_WAIT(s)       sem_wait(&(s))
    #define DMS_SEM_POST(s)       sem_post(&(s))
    #define DMS_SEM_DESTROY(s)    sem_destroy(&(s))
#endif

/* ─── High-Resolution Monotonic Clock ─────────────────────────────────────── */
/**
 * @brief Returns monotonic timestamp in milliseconds.
 *        Uses QueryPerformanceCounter on Windows, clock_gettime on POSIX.
 */
static inline double dms_get_time_ms(void)
{
#if defined(DMS_PLATFORM_WINDOWS)
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
#endif
}

/* ─── Sleep Abstraction ────────────────────────────────────────────────────── */
#if defined(DMS_PLATFORM_WINDOWS)
    #define DMS_SLEEP_MS(ms) Sleep((DWORD)(ms))
#else
    #include <unistd.h>
    #define DMS_SLEEP_MS(ms) usleep((ms) * 1000)
#endif

/* ─── Directory Separator ─────────────────────────────────────────────────── */
#if defined(DMS_PLATFORM_WINDOWS)
    #define DMS_PATH_SEP "\\"
#else
    #define DMS_PATH_SEP "/"
#endif

/* ─── Unused Variable Suppressor ──────────────────────────────────────────── */
#define DMS_UNUSED(x) (void)(x)

/* ─── Compile-time Assert ─────────────────────────────────────────────────── */
#define DMS_STATIC_ASSERT(cond, msg) typedef char dms_static_assert_##msg[(cond) ? 1 : -1]

#endif /* DMS_PLATFORM_H */
