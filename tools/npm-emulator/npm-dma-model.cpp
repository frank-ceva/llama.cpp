// NPM DMA Model Implementation

#include "npm-dma-model.h"
#include <cmath>

npm_dma_model::npm_dma_model()
    : current_cycle(0)
    , total_bytes(0)
    , total_transfer_cycles(0)
    , ddr_l2_bytes(0)
    , l2_l1_bytes(0)
{
}

npm_dma_model::npm_dma_model(const npm_dma_config & config)
    : config(config)
    , current_cycle(0)
    , total_bytes(0)
    , total_transfer_cycles(0)
    , ddr_l2_bytes(0)
    , l2_l1_bytes(0)
{
}

npm_dma_model::~npm_dma_model() {
}

uint64_t npm_dma_model::calculate_cycles(npm_dma_type type, size_t bytes) const {
    double bandwidth_gbps;

    switch (type) {
        case NPM_DMA_DDR_TO_L2:
        case NPM_DMA_L2_TO_DDR:
            bandwidth_gbps = config.system_dma_bw_gbps;
            break;
        case NPM_DMA_L2_TO_L1:
        case NPM_DMA_L1_TO_L2:
            bandwidth_gbps = config.l1_dma_bw_gbps;
            break;
        default:
            bandwidth_gbps = config.system_dma_bw_gbps;
            break;
    }

    // Bandwidth in bytes per second = gbps * 1e9 / 8 (bits to bytes)
    // Cycles per second = clock_freq_mhz * 1e6
    // Bytes per cycle = (bandwidth_gbps * 1e9 / 8) / (clock_freq_mhz * 1e6)
    //                 = bandwidth_gbps * 1e9 / (8 * clock_freq_mhz * 1e6)
    //                 = bandwidth_gbps * 1000 / (8 * clock_freq_mhz)
    //                 = bandwidth_gbps * 125 / clock_freq_mhz

    double bytes_per_cycle = (bandwidth_gbps * 125.0) / config.clock_freq_mhz;

    // Cycles = bytes / bytes_per_cycle
    uint64_t cycles = (uint64_t)ceil((double)bytes / bytes_per_cycle);

    // Minimum 1 cycle for any transfer
    return (cycles > 0) ? cycles : 1;
}

uint64_t npm_dma_model::transfer(npm_dma_type type, size_t bytes, int engine_id) {
    (void)engine_id;  // For future use with multiple L1 DMA engines

    uint64_t cycles = calculate_cycles(type, bytes);

    // Update cycle counter
    current_cycle += cycles;

    // Update statistics
    total_bytes += bytes;
    total_transfer_cycles += cycles;

    switch (type) {
        case NPM_DMA_DDR_TO_L2:
        case NPM_DMA_L2_TO_DDR:
            ddr_l2_bytes += bytes;
            break;
        case NPM_DMA_L2_TO_L1:
        case NPM_DMA_L1_TO_L2:
            l2_l1_bytes += bytes;
            break;
    }

    // Emit trace event
    if (trace_ctx_ && npm_trace_enabled(trace_ctx_, NPM_TRACE_DMA)) {
        npm_trace_event_type evt;
        switch (type) {
            case NPM_DMA_DDR_TO_L2: evt = NPM_TRACE_DMA_DDR_TO_L2; break;
            case NPM_DMA_L2_TO_DDR: evt = NPM_TRACE_DMA_L2_TO_DDR; break;
            case NPM_DMA_L2_TO_L1:  evt = NPM_TRACE_DMA_L2_TO_L1;  break;
            case NPM_DMA_L1_TO_L2:  evt = NPM_TRACE_DMA_L1_TO_L2;  break;
        }
        npm_trace_dma(trace_ctx_, evt, bytes, cycles, engine_id);
    }

    return cycles;
}

void npm_dma_model::reset_stats() {
    current_cycle = 0;
    total_bytes = 0;
    total_transfer_cycles = 0;
    ddr_l2_bytes = 0;
    l2_l1_bytes = 0;
}
