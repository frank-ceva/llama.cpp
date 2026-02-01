#pragma once

// NPM Memory Hierarchy Model
//
// Models the three-tier memory hierarchy of NPM:
//   - DDR: External memory (represented by shared memory from client)
//   - L2:  Shared cache across all engines
//   - L1:  Per-engine local scratchpad
//
// Implements LRU eviction and tracks cache hits/misses for statistics.

#include <stddef.h>
#include <stdint.h>
#include <vector>
#include <list>
#include <unordered_map>

// Memory region types
enum npm_mem_region {
    NPM_MEM_DDR = 0,    // External DDR (shared memory from client)
    NPM_MEM_L2  = 1,    // Shared L2 cache
    NPM_MEM_L1  = 2,    // Per-engine L1 scratchpad
};

// Memory block tracking
struct npm_mem_block {
    uint64_t handle;            // Original buffer handle
    size_t   offset;            // Offset within original buffer
    size_t   size;              // Block size
    npm_mem_region location;    // Current location
    size_t   local_offset;      // Offset within L1/L2 storage
    uint64_t last_access;       // Timestamp for LRU eviction
    bool     dirty;             // Modified since last writeback
};

// L1 memory model (per engine)
struct npm_l1_model {
    size_t capacity;            // Total L1 size
    size_t used;                // Currently allocated
    std::vector<uint8_t> storage;  // Actual L1 storage

    // Block tracking (ordered by access time for LRU)
    std::list<npm_mem_block> blocks;

    npm_l1_model(size_t cap) : capacity(cap), used(0), storage(cap) {}

    bool can_fit(size_t size) const { return used + size <= capacity; }
};

// L2 memory model (shared across engines)
struct npm_l2_model {
    size_t capacity;            // Total L2 size
    size_t used;                // Currently allocated
    std::vector<uint8_t> storage;  // Actual L2 storage

    // Block tracking (ordered by access time for LRU)
    std::list<npm_mem_block> blocks;

    npm_l2_model(size_t cap) : capacity(cap), used(0), storage(cap) {}

    bool can_fit(size_t size) const { return used + size <= capacity; }
};

// Overall memory hierarchy controller
class npm_memory_hierarchy {
public:
    npm_memory_hierarchy(int num_engines, size_t l1_size_per_engine, size_t l2_size);
    ~npm_memory_hierarchy();

    // Stage data from DDR to L2
    // Returns pointer to data in L2 storage
    // If data already in L2, returns existing pointer (cache hit)
    void * stage_to_l2(uint64_t handle, size_t offset, size_t size, void * ddr_ptr);

    // Stage data from L2 to L1 for a specific engine
    // Returns pointer to data in L1 storage
    void * stage_to_l1(int engine_id, uint64_t handle, size_t offset, size_t size);

    // Writeback modified data from L1 to L2
    void writeback_l1_to_l2(int engine_id, uint64_t handle, size_t offset);

    // Writeback data from L2 to DDR
    void writeback_l2_to_ddr(uint64_t handle, size_t offset, void * ddr_ptr);

    // Mark a block as dirty (modified)
    void mark_dirty(int engine_id, uint64_t handle, size_t offset);

    // Flush all dirty data back to DDR
    void flush_all(void * ddr_base);

    // Reset memory state (clear all cached blocks)
    void reset();

    // Get statistics
    uint64_t get_l1_hits() const { return l1_hits; }
    uint64_t get_l2_hits() const { return l2_hits; }
    uint64_t get_l1_misses() const { return l1_misses; }
    uint64_t get_l2_misses() const { return l2_misses; }
    uint64_t get_total_bytes_moved() const { return total_bytes_moved; }

    // Get model info
    int get_num_engines() const { return num_engines; }
    size_t get_l1_size() const { return l1_size_per_engine; }
    size_t get_l2_size() const { return l2_size; }

private:
    int num_engines;
    size_t l1_size_per_engine;
    size_t l2_size;

    // Memory models
    std::vector<npm_l1_model> l1_models;  // One per engine
    npm_l2_model l2_model;

    // Access timestamp counter
    uint64_t access_counter;

    // Statistics
    uint64_t l1_hits;
    uint64_t l2_hits;
    uint64_t l1_misses;
    uint64_t l2_misses;
    uint64_t total_bytes_moved;

    // Helper functions
    npm_mem_block * find_block_in_l1(int engine_id, uint64_t handle, size_t offset);
    npm_mem_block * find_block_in_l2(uint64_t handle, size_t offset);
    void evict_lru_l1(int engine_id, size_t needed_size);
    void evict_lru_l2(size_t needed_size);
    size_t allocate_l1(int engine_id, size_t size);
    size_t allocate_l2(size_t size);
};
