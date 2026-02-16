#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Fused SiLU + Q4_0 down-projection matvec with accumulate.
//
// Computes: output[row] += dot(weight_row, silu(gate) * up)
// where silu(x) = x / (1 + exp(-x))
//
// Eliminates: (1) SiLU dispatch, (2) scratch_silu buffer write/read,
//             (3) residual_add_inplace dispatch.
// Saves 2 dispatches per layer (60 total for 30-layer model).
//
// 256 threads = 8 simdgroups, each handles 1 output row (8 rows/TG).
//
// Buffer layout:
//   0: weight    Q4_0 blocks [out_dim, in_dim/32] (down projection)
//   1: gate      [in_dim] float (gate projection output)
//   2: up        [in_dim] float (up projection output)
//   3: output    [out_dim] float (accumulate target, e.g., hidden_a)
//   4: out_dim   scalar uint
//   5: in_dim    scalar uint
// ---------------------------------------------------------------------------

struct BlockQ4_0 {
    half d;
    uchar qs[16];
};

kernel void silu_matvec_q4_0_accumulate(
    device const BlockQ4_0* weight  [[buffer(0)]],
    device const float*     gate    [[buffer(1)]],
    device const float*     up      [[buffer(2)]],
    device float*           output  [[buffer(3)]],  // READ-MODIFY-WRITE
    constant uint&          out_dim [[buffer(4)]],
    constant uint&          in_dim  [[buffer(5)]],
    uint tgid      [[threadgroup_position_in_grid]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    const uint ROWS_PER_TG = 8;
    const uint row = tgid * ROWS_PER_TG + simd_id;
    if (row >= out_dim) return;

    const uint n_blocks = in_dim / 32;
    const uint row_offset = row * n_blocks;

    float sum = 0.0f;

    for (uint b = simd_lane; b < n_blocks; b += 32) {
        const BlockQ4_0 block = weight[row_offset + b];
        const float scale = float(block.d);
        const uint base = b * 32;

        for (uint i = 0; i < 16; i += 4) {
            // Compute SiLU(gate) * up inline for low nibble elements
            float4 g_lo = float4(gate[base + i],
                                 gate[base + i + 1],
                                 gate[base + i + 2],
                                 gate[base + i + 3]);
            float4 u_lo = float4(up[base + i],
                                 up[base + i + 1],
                                 up[base + i + 2],
                                 up[base + i + 3]);
            // silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)
            // Use fast::exp for ~2x faster approximate exp
            float4 silu_lo = g_lo / (1.0f + float4(fast::exp(-g_lo.x), fast::exp(-g_lo.y), fast::exp(-g_lo.z), fast::exp(-g_lo.w))) * u_lo;

            // Same for high nibble elements
            float4 g_hi = float4(gate[base + i + 16],
                                 gate[base + i + 17],
                                 gate[base + i + 18],
                                 gate[base + i + 19]);
            float4 u_hi = float4(up[base + i + 16],
                                 up[base + i + 17],
                                 up[base + i + 18],
                                 up[base + i + 19]);
            float4 silu_hi = g_hi / (1.0f + float4(fast::exp(-g_hi.x), fast::exp(-g_hi.y), fast::exp(-g_hi.z), fast::exp(-g_hi.w))) * u_hi;

            // Dequantize Q4_0 weight values
            float4 lo_vals = float4(int(block.qs[i]   & 0x0F) - 8,
                                    int(block.qs[i+1] & 0x0F) - 8,
                                    int(block.qs[i+2] & 0x0F) - 8,
                                    int(block.qs[i+3] & 0x0F) - 8) * scale;
            float4 hi_vals = float4(int(block.qs[i]   >> 4) - 8,
                                    int(block.qs[i+1] >> 4) - 8,
                                    int(block.qs[i+2] >> 4) - 8,
                                    int(block.qs[i+3] >> 4) - 8) * scale;

            sum += dot(lo_vals, silu_lo) + dot(hi_vals, silu_hi);
        }
    }

    sum = simd_sum(sum);

    if (simd_lane == 0) {
        output[row] += sum;  // ACCUMULATE
    }
}
