// NPM Memory Hierarchy Model Implementation

#include "npm-memory-model.h"
#include <cstring>
#include <algorithm>

npm_memory_hierarchy::npm_memory_hierarchy(int num_engines, size_t l1_size_per_engine, size_t l2_size)
    : num_engines(num_engines)
    , l1_size_per_engine(l1_size_per_engine)
    , l2_size(l2_size)
    , l2_model(l2_size)
    , access_counter(0)
    , l1_hits(0)
    , l2_hits(0)
    , l1_misses(0)
    , l2_misses(0)
    , total_bytes_moved(0)
{
    // Create L1 models for each engine
    l1_models.reserve(num_engines);
    for (int i = 0; i < num_engines; i++) {
        l1_models.emplace_back(l1_size_per_engine);
    }
}

npm_memory_hierarchy::~npm_memory_hierarchy() {
}

npm_mem_block * npm_memory_hierarchy::find_block_in_l1(int engine_id, uint64_t handle, size_t offset) {
    if (engine_id < 0 || engine_id >= num_engines) return nullptr;

    npm_l1_model & l1 = l1_models[engine_id];
    for (auto & block : l1.blocks) {
        if (block.handle == handle && block.offset == offset) {
            return &block;
        }
    }
    return nullptr;
}

npm_mem_block * npm_memory_hierarchy::find_block_in_l2(uint64_t handle, size_t offset) {
    for (auto & block : l2_model.blocks) {
        if (block.handle == handle && block.offset == offset) {
            return &block;
        }
    }
    return nullptr;
}

void npm_memory_hierarchy::evict_lru_l1(int engine_id, size_t needed_size) {
    if (engine_id < 0 || engine_id >= num_engines) return;

    npm_l1_model & l1 = l1_models[engine_id];

    // Evict oldest blocks until we have enough space
    while (!l1.blocks.empty() && !l1.can_fit(needed_size)) {
        // Find LRU block (oldest access time)
        auto lru_it = std::min_element(l1.blocks.begin(), l1.blocks.end(),
            [](const npm_mem_block & a, const npm_mem_block & b) {
                return a.last_access < b.last_access;
            });

        if (lru_it != l1.blocks.end()) {
            // If dirty, would need to writeback (simplified: just evict)
            l1.used -= lru_it->size;
            l1.blocks.erase(lru_it);
        }
    }
}

void npm_memory_hierarchy::evict_lru_l2(size_t needed_size) {
    // Evict oldest blocks until we have enough space
    while (!l2_model.blocks.empty() && !l2_model.can_fit(needed_size)) {
        auto lru_it = std::min_element(l2_model.blocks.begin(), l2_model.blocks.end(),
            [](const npm_mem_block & a, const npm_mem_block & b) {
                return a.last_access < b.last_access;
            });

        if (lru_it != l2_model.blocks.end()) {
            l2_model.used -= lru_it->size;
            l2_model.blocks.erase(lru_it);
        }
    }
}

size_t npm_memory_hierarchy::allocate_l1(int engine_id, size_t size) {
    if (engine_id < 0 || engine_id >= num_engines) return 0;

    npm_l1_model & l1 = l1_models[engine_id];

    // Simple bump allocator (for now)
    // In a real implementation, would track free regions
    size_t offset = l1.used;
    l1.used += size;
    return offset;
}

size_t npm_memory_hierarchy::allocate_l2(size_t size) {
    size_t offset = l2_model.used;
    l2_model.used += size;
    return offset;
}

void * npm_memory_hierarchy::stage_to_l2(uint64_t handle, size_t offset, size_t size, void * ddr_ptr) {
    // Check if already in L2
    npm_mem_block * existing = find_block_in_l2(handle, offset);
    if (existing) {
        l2_hits++;
        existing->last_access = ++access_counter;
        return l2_model.storage.data() + existing->local_offset;
    }

    l2_misses++;

    // Need to bring from DDR
    // Evict if necessary
    if (!l2_model.can_fit(size)) {
        evict_lru_l2(size);
    }

    // Allocate space
    size_t local_offset = allocate_l2(size);

    // Copy from DDR
    memcpy(l2_model.storage.data() + local_offset, ddr_ptr, size);
    total_bytes_moved += size;

    // Add block entry
    npm_mem_block block;
    block.handle = handle;
    block.offset = offset;
    block.size = size;
    block.location = NPM_MEM_L2;
    block.local_offset = local_offset;
    block.last_access = ++access_counter;
    block.dirty = false;
    l2_model.blocks.push_back(block);

    return l2_model.storage.data() + local_offset;
}

void * npm_memory_hierarchy::stage_to_l1(int engine_id, uint64_t handle, size_t offset, size_t size) {
    if (engine_id < 0 || engine_id >= num_engines) return nullptr;

    npm_l1_model & l1 = l1_models[engine_id];

    // Check if already in L1
    npm_mem_block * existing = find_block_in_l1(engine_id, handle, offset);
    if (existing) {
        l1_hits++;
        existing->last_access = ++access_counter;
        return l1.storage.data() + existing->local_offset;
    }

    l1_misses++;

    // Check if in L2 (should have been staged there first)
    npm_mem_block * l2_block = find_block_in_l2(handle, offset);
    if (!l2_block) {
        // Data not in L2 - this is an error in normal usage
        // For robustness, return nullptr
        return nullptr;
    }

    // Evict from L1 if necessary
    if (!l1.can_fit(size)) {
        evict_lru_l1(engine_id, size);
    }

    // Allocate space in L1
    size_t local_offset = allocate_l1(engine_id, size);

    // Copy from L2
    memcpy(l1.storage.data() + local_offset,
           l2_model.storage.data() + l2_block->local_offset,
           size);
    total_bytes_moved += size;

    // Add block entry
    npm_mem_block block;
    block.handle = handle;
    block.offset = offset;
    block.size = size;
    block.location = NPM_MEM_L1;
    block.local_offset = local_offset;
    block.last_access = ++access_counter;
    block.dirty = false;
    l1.blocks.push_back(block);

    return l1.storage.data() + local_offset;
}

void npm_memory_hierarchy::writeback_l1_to_l2(int engine_id, uint64_t handle, size_t offset) {
    if (engine_id < 0 || engine_id >= num_engines) return;

    npm_l1_model & l1 = l1_models[engine_id];

    npm_mem_block * l1_block = find_block_in_l1(engine_id, handle, offset);
    if (!l1_block || !l1_block->dirty) return;

    npm_mem_block * l2_block = find_block_in_l2(handle, offset);
    if (!l2_block) return;

    // Copy from L1 to L2
    memcpy(l2_model.storage.data() + l2_block->local_offset,
           l1.storage.data() + l1_block->local_offset,
           l1_block->size);
    total_bytes_moved += l1_block->size;

    l1_block->dirty = false;
    l2_block->dirty = true;
}

void npm_memory_hierarchy::writeback_l2_to_ddr(uint64_t handle, size_t offset, void * ddr_ptr) {
    npm_mem_block * l2_block = find_block_in_l2(handle, offset);
    if (!l2_block || !l2_block->dirty) return;

    // Copy from L2 to DDR
    memcpy(ddr_ptr, l2_model.storage.data() + l2_block->local_offset, l2_block->size);
    total_bytes_moved += l2_block->size;

    l2_block->dirty = false;
}

void npm_memory_hierarchy::mark_dirty(int engine_id, uint64_t handle, size_t offset) {
    npm_mem_block * block = find_block_in_l1(engine_id, handle, offset);
    if (block) {
        block->dirty = true;
    }
}

void npm_memory_hierarchy::flush_all(void * ddr_base) {
    // Flush all dirty L1 blocks to L2
    for (int e = 0; e < num_engines; e++) {
        npm_l1_model & l1 = l1_models[e];
        for (auto & block : l1.blocks) {
            if (block.dirty) {
                writeback_l1_to_l2(e, block.handle, block.offset);
            }
        }
    }

    // Flush all dirty L2 blocks to DDR
    // Note: This simplified version assumes ddr_base + offset matches original registration
    // A real implementation would track DDR addresses
    for (auto & block : l2_model.blocks) {
        if (block.dirty) {
            void * ddr_ptr = (uint8_t *)ddr_base + block.offset;
            memcpy(ddr_ptr, l2_model.storage.data() + block.local_offset, block.size);
            total_bytes_moved += block.size;
            block.dirty = false;
        }
    }
}

void npm_memory_hierarchy::reset() {
    // Clear all blocks and reset statistics
    for (auto & l1 : l1_models) {
        l1.blocks.clear();
        l1.used = 0;
    }
    l2_model.blocks.clear();
    l2_model.used = 0;

    access_counter = 0;
    l1_hits = 0;
    l2_hits = 0;
    l1_misses = 0;
    l2_misses = 0;
    total_bytes_moved = 0;
}
