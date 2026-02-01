#pragma once

// NPM DMA Model
//
// Models the DMA engines for data movement between memory tiers:
//   - System DMA: DDR <-> L2 transfers
//   - L1 DMA:     L2 <-> L1 transfers (per engine)
//
// Calculates transfer cycles based on bandwidth and tracks statistics.

#include <stddef.h>
#include <stdint.h>
#include "npm-trace.h"

// DMA transfer types
enum npm_dma_type {
    NPM_DMA_DDR_TO_L2 = 0,    // System DMA: DDR -> L2
    NPM_DMA_L2_TO_DDR = 1,    // System DMA: L2 -> DDR
    NPM_DMA_L2_TO_L1  = 2,    // L1 DMA: L2 -> L1
    NPM_DMA_L1_TO_L2  = 3,    // L1 DMA: L1 -> L2
};

// DMA configuration
struct npm_dma_config {
    double system_dma_bw_gbps;   // DDR <-> L2 bandwidth (GB/s)
    double l1_dma_bw_gbps;       // L2 <-> L1 bandwidth (GB/s)
    uint64_t clock_freq_mhz;     // System clock frequency (MHz)

    // Default configuration based on typical NPM specs
    npm_dma_config()
        : system_dma_bw_gbps(50.0)
        , l1_dma_bw_gbps(100.0)
        , clock_freq_mhz(1000)
    {}
};

// DMA transfer record (for history/debugging)
struct npm_dma_transfer {
    npm_dma_type type;
    size_t size;
    uint64_t start_cycle;
    uint64_t end_cycle;
    int engine_id;          // For L1 DMA transfers
};

// DMA model
class npm_dma_model {
public:
    npm_dma_model();
    explicit npm_dma_model(const npm_dma_config & config);
    ~npm_dma_model();

    // Initiate a transfer and return the number of cycles taken
    // For overlapped execution, can overlap with compute
    uint64_t transfer(npm_dma_type type, size_t bytes, int engine_id = -1);

    // Get current cycle count
    uint64_t get_current_cycle() const { return current_cycle; }

    // Advance cycle counter (for compute simulation)
    void advance_cycles(uint64_t cycles) { current_cycle += cycles; }

    // Reset cycle counter
    void reset_cycles() { current_cycle = 0; }

    // Get configuration
    const npm_dma_config & get_config() const { return config; }

    // Set configuration
    void set_config(const npm_dma_config & cfg) { config = cfg; }

    // Set trace context
    void set_trace_ctx(npm_trace_ctx* ctx) { trace_ctx_ = ctx; }

    // Statistics
    uint64_t get_total_bytes_transferred() const { return total_bytes; }
    uint64_t get_total_transfer_cycles() const { return total_transfer_cycles; }
    uint64_t get_ddr_l2_bytes() const { return ddr_l2_bytes; }
    uint64_t get_l2_l1_bytes() const { return l2_l1_bytes; }

    // Reset statistics
    void reset_stats();

private:
    npm_dma_config config;
    uint64_t current_cycle;

    // Statistics
    uint64_t total_bytes;
    uint64_t total_transfer_cycles;
    uint64_t ddr_l2_bytes;      // DDR <-> L2 bytes
    uint64_t l2_l1_bytes;       // L2 <-> L1 bytes

    // Trace context
    npm_trace_ctx* trace_ctx_ = nullptr;

    // Calculate cycles for a transfer
    uint64_t calculate_cycles(npm_dma_type type, size_t bytes) const;
};
