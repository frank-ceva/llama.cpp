#pragma once

// NPM Shared Memory Helpers
//
// Provides cross-platform shared memory functionality for the NPM emulator.
// The CPU-side driver creates a shared memory region, and the emulator
// process attaches to it to access tensor data without copies.

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Shared memory region
// =============================================================================

struct npm_shm_region {
    char     name[64];    // Shared memory name (e.g., "/npm-shm-12345")
    void *   base;        // Mapped base address
    size_t   size;        // Total size of the region
    size_t   allocated;   // Currently allocated bytes
    int      fd;          // File descriptor (Unix) or handle (Windows)
    bool     is_owner;    // True if this process created the region
};

// =============================================================================
// Region management
// =============================================================================

// Create a new shared memory region
// Returns NULL on failure
struct npm_shm_region * npm_shm_create(size_t size);

// Attach to an existing shared memory region by name
// Returns NULL on failure
struct npm_shm_region * npm_shm_attach(const char * name, size_t size);

// Detach from / destroy the shared memory region
void npm_shm_destroy(struct npm_shm_region * region);

// =============================================================================
// Simple bump allocator within the region
// =============================================================================

// Allocate memory from the shared region
// Returns offset within the region, or (size_t)-1 on failure
size_t npm_shm_alloc(struct npm_shm_region * region, size_t size, size_t alignment);

// Get pointer from offset
static inline void * npm_shm_get_ptr(struct npm_shm_region * region, size_t offset) {
    if (!region || offset >= region->size) {
        return NULL;
    }
    return (char *)region->base + offset;
}

// Reset allocator (free all allocations)
void npm_shm_reset(struct npm_shm_region * region);

#ifdef __cplusplus
}
#endif

// =============================================================================
// Implementation (header-only for simplicity)
// =============================================================================

#ifdef NPM_SHM_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

struct npm_shm_region * npm_shm_create(size_t size) {
    struct npm_shm_region * region = (struct npm_shm_region *)malloc(sizeof(struct npm_shm_region));
    if (!region) {
        return NULL;
    }

    memset(region, 0, sizeof(*region));

#ifdef _WIN32
    // Windows: use memory-mapped file
    snprintf(region->name, sizeof(region->name), "npm-shm-%lu", (unsigned long)GetCurrentProcessId());

    HANDLE hMapFile = CreateFileMappingA(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        (DWORD)(size >> 32),
        (DWORD)(size & 0xFFFFFFFF),
        region->name
    );

    if (!hMapFile) {
        free(region);
        return NULL;
    }

    region->base = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (!region->base) {
        CloseHandle(hMapFile);
        free(region);
        return NULL;
    }

    region->fd = (int)(intptr_t)hMapFile;
#else
    // Unix: use POSIX shared memory
    snprintf(region->name, sizeof(region->name), "/npm-shm-%d", getpid());

    int fd = shm_open(region->name, O_CREAT | O_RDWR, 0600);
    if (fd < 0) {
        free(region);
        return NULL;
    }

    if (ftruncate(fd, size) < 0) {
        close(fd);
        shm_unlink(region->name);
        free(region);
        return NULL;
    }

    region->base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (region->base == MAP_FAILED) {
        close(fd);
        shm_unlink(region->name);
        free(region);
        return NULL;
    }

    region->fd = fd;
#endif

    region->size = size;
    region->allocated = 0;
    region->is_owner = true;

    return region;
}

struct npm_shm_region * npm_shm_attach(const char * name, size_t size) {
    if (!name || !size) {
        return NULL;
    }

    struct npm_shm_region * region = (struct npm_shm_region *)malloc(sizeof(struct npm_shm_region));
    if (!region) {
        return NULL;
    }

    memset(region, 0, sizeof(*region));
    strncpy(region->name, name, sizeof(region->name) - 1);

#ifdef _WIN32
    HANDLE hMapFile = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, name);
    if (!hMapFile) {
        free(region);
        return NULL;
    }

    region->base = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (!region->base) {
        CloseHandle(hMapFile);
        free(region);
        return NULL;
    }

    region->fd = (int)(intptr_t)hMapFile;
#else
    int fd = shm_open(name, O_RDWR, 0);
    if (fd < 0) {
        free(region);
        return NULL;
    }

    region->base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (region->base == MAP_FAILED) {
        close(fd);
        free(region);
        return NULL;
    }

    region->fd = fd;
#endif

    region->size = size;
    region->allocated = 0;
    region->is_owner = false;

    return region;
}

void npm_shm_destroy(struct npm_shm_region * region) {
    if (!region) {
        return;
    }

#ifdef _WIN32
    if (region->base) {
        UnmapViewOfFile(region->base);
    }
    if (region->fd) {
        CloseHandle((HANDLE)(intptr_t)region->fd);
    }
#else
    if (region->base && region->base != MAP_FAILED) {
        munmap(region->base, region->size);
    }
    if (region->fd >= 0) {
        close(region->fd);
    }
    if (region->is_owner) {
        shm_unlink(region->name);
    }
#endif

    free(region);
}

size_t npm_shm_alloc(struct npm_shm_region * region, size_t size, size_t alignment) {
    if (!region || size == 0) {
        return (size_t)-1;
    }

    // Align the current allocation pointer
    if (alignment == 0) {
        alignment = 64;  // Default alignment for cache lines
    }

    size_t aligned_offset = (region->allocated + alignment - 1) & ~(alignment - 1);
    size_t new_allocated = aligned_offset + size;

    if (new_allocated > region->size) {
        return (size_t)-1;  // Out of memory
    }

    region->allocated = new_allocated;
    return aligned_offset;
}

void npm_shm_reset(struct npm_shm_region * region) {
    if (region) {
        region->allocated = 0;
    }
}

#endif // NPM_SHM_IMPLEMENTATION
