// Metal kernels for single-token decode of the Qwen3.5 hybrid architecture with ternary weights.
//
// Activations are stored as half and every kernel computes in float, rounding to half at the
// points where the MLX reference does, so that greedy decoding picks the same tokens.
#include <metal_stdlib>
using namespace metal;

static inline float sigmoidf(float v) { return 1.0f / (1.0f + exp(-v)); }

// silu evaluated the way a half-precision graph does it: sigmoid rounded, then the product rounded.
static inline half silu_h(half v) {
    half s = half(sigmoidf(float(v)));
    return half(float(v) * float(s));
}

// Every kernel takes a batch of tokens: per-token vectors are stored back to back, token b at
// offset b * (vector length). Decode runs the same kernels with a batch of one.

// ---------------------------------------------------------------------------------------------
// Embedding: x = scale * q + bias for one row of the packed table. Grid (cols, batch).
struct EmbedP { uint step; uint cols; uint groups; };

kernel void embed(device const uint *W [[buffer(0)]],
                  device const half *S [[buffer(1)]],
                  device const half *B [[buffer(2)]],
                  device const uint *tok [[buffer(3)]],
                  device half *x [[buffer(4)]],
                  constant EmbedP &p [[buffer(5)]],
                  uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    if (i >= p.cols) return;
    uint t = tok[p.step + gid.y];
    uint word = W[t * (p.cols >> 4) + (i >> 4)];
    float q = float((word >> ((i & 15u) * 2u)) & 3u);
    uint g = t * p.groups + (i >> 7);
    x[gid.y * p.cols + i] = half(float(S[g]) * q + float(B[g]));
}

// ---------------------------------------------------------------------------------------------
// RMSNorm over one vector, optionally adding a residual delta in place first.
// One threadgroup of 256 threads per token: groups (1, batch); n must be a multiple of 256.
struct NormP { uint n; uint add; float eps; };

kernel void add_rmsnorm(device half *xb [[buffer(0)]],
                        device const half *rb [[buffer(1)]],
                        device const half *w [[buffer(2)]],
                        device half *yb [[buffer(3)]],
                        constant NormP &p [[buffer(4)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float part[256];
    device half *x = xb + tg.y * p.n;
    device const half *r = rb + tg.y * p.n;
    device half *y = yb + tg.y * p.n;
    uint per = p.n >> 8;
    uint base = tid * per;
    float s = 0.0f;
    for (uint j = 0; j < per; ++j) {
        uint i = base + j;
        half v = x[i];
        if (p.add != 0u) {
            v = half(float(v) + float(r[i]));
            x[i] = v;
        }
        s += float(v) * float(v);
    }
    part[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint j = 0; j < 256u; ++j) tot += part[j];
    float scale = rsqrt(tot / float(p.n) + p.eps);
    for (uint j = 0; j < per; ++j) {
        uint i = base + j;
        y[i] = half(float(x[i]) * scale * float(w[i]));
    }
}

// ---------------------------------------------------------------------------------------------
// Ternary matrix products, 2-bit slots, one scale per 128 weights:
//   y[r] = sum_g s[r,g] * sum_i (q[r,i] - 1) * x[i]
// Row counts are multiples of 16 and column counts multiples of 512, so there are no tails.
struct GemvP { uint rows; uint groups; };

// Reference kernel: one thread per row, slots unpacked with shifts.
kernel void gemv_t2_r1(device const uint *W [[buffer(0)]],
                       device const half *S [[buffer(1)]],
                       device const half *x [[buffer(2)]],
                       device half *y [[buffer(3)]],
                       constant GemvP &p [[buffer(4)]],
                       uint tid [[thread_position_in_grid]]) {
    if (tid >= p.rows) return;
    device const uint *w = W + tid * p.groups * 8u;
    device const half *s = S + tid * p.groups;
    const uint4 sh = uint4(0u, 2u, 4u, 6u);
    float acc = 0.0f;
    for (uint g = 0; g < p.groups; ++g) {
        device const half4 *xp = (device const half4 *)(x + g * 128u);
        float t = 0.0f;
        for (uint j = 0; j < 8u; ++j) {
            uint bits = w[g * 8u + j];
            t += dot(float4((uint4(bits) >> sh) & 3u) - 1.0f, float4(xp[j * 4u]))
               + dot(float4((uint4(bits >> 8u) >> sh) & 3u) - 1.0f, float4(xp[j * 4u + 1u]))
               + dot(float4((uint4(bits >> 16u) >> sh) & 3u) - 1.0f, float4(xp[j * 4u + 2u]))
               + dot(float4((uint4(bits >> 24u) >> sh) & 3u) - 1.0f, float4(xp[j * 4u + 3u]));
        }
        acc += float(s[g]) * t;
    }
    y[tid] = half(acc);
}

// Production GEMV. The columns of a row are split across the 32 lanes of a SIMD group: at every
// step the group reads 32 adjacent weight words and the matching 512 inputs, each lane keeps R
// rows going (one load and conversion of the input serves R rows), and simd_sum folds the lanes.
//
// Unpacking: the slot is masked in place, giving q * 4^k, the bias 4^k is subtracted as an integer
// and the input is pre-divided by 4^k. All of it is exact, so the result is bit-identical to
// unpacking with shifts while a weight costs AND, SUB, one conversion and one multiply-add.
// Grid (32, rows / R), threadgroup (32, 1).
#define UNPACK_CONSTS                                                                             \
    const uint4 c0 = uint4(0x1u, 0x4u, 0x10u, 0x40u);                                             \
    const uint4 c1 = c0 << 8u, c2 = c0 << 16u, c3 = c0 << 24u;                                    \
    const float4 d0 = float4(1.0f, 0.25f, 0.0625f, 0.015625f);                                    \
    const float4 d1 = d0 * 0x1p-8f, d2 = d0 * 0x1p-16f, d3 = d0 * 0x1p-24f;

#define GEMV_T2(NAME, R)                                                                          \
kernel void NAME(device const uint *W [[buffer(0)]],                                              \
                 device const half *S [[buffer(1)]],                                              \
                 device const half *x [[buffer(2)]],                                              \
                 device half *y [[buffer(3)]],                                                    \
                 constant GemvP &p [[buffer(4)]],                                                 \
                 uint2 gid [[thread_position_in_grid]],                                           \
                 uint lane [[thread_index_in_simdgroup]]) {                                       \
    uint r0 = gid.y * R;                                                                          \
    uint words = p.groups * 8u;                                                                   \
    device const uint *w = W + r0 * words + lane;                                                 \
    device const half *s = S + r0 * p.groups + (lane >> 3);                                       \
    device const half4 *xp = (device const half4 *)(x + lane * 16u);                              \
    UNPACK_CONSTS                                                                                 \
    float acc[R];                                                                                 \
    for (uint i = 0; i < R; ++i) acc[i] = 0.0f;                                                   \
    for (uint b = 0; b < words; b += 32u) {                                                       \
        float4 x0 = float4(xp[b * 4u]) * d0, x1 = float4(xp[b * 4u + 1u]) * d1;                   \
        float4 x2 = float4(xp[b * 4u + 2u]) * d2, x3 = float4(xp[b * 4u + 3u]) * d3;              \
        for (uint i = 0; i < R; ++i) {                                                            \
            uint4 bits = uint4(w[i * words + b]);                                                 \
            float t = dot(float4(int4((bits & (c0 * 3u)) - c0)), x0)                              \
                    + dot(float4(int4((bits & (c1 * 3u)) - c1)), x1)                              \
                    + dot(float4(int4((bits & (c2 * 3u)) - c2)), x2)                              \
                    + dot(float4(int4((bits & (c3 * 3u)) - c3)), x3);                             \
            acc[i] += float(s[i * p.groups + (b >> 3)]) * t;                                      \
        }                                                                                         \
    }                                                                                             \
    for (uint i = 0; i < R; ++i) {                                                                \
        float v = simd_sum(acc[i]);                                                               \
        if (lane == 0u) y[r0 + i] = half(v);                                                      \
    }                                                                                             \
}

GEMV_T2(gemv_t2_sgm1, 1)
GEMV_T2(gemv_t2_sgm2, 2)
GEMV_T2(gemv_t2_sgm4, 4)
GEMV_T2(gemv_t2_sgm8, 8)
GEMV_T2(gemv_t2_sgm16, 16)

// Batched product (prefill, multi-token verification): Y[t] = W * X[t]. Same lane layout; a
// thread unpacks each weight word once into exact -1/0/+1 floats and uses it for BT tokens, so
// the weights are read once per BT tokens. X is [batch][cols], Y is [batch][rows].
// Grid (32, rows / R, batch / BT), threadgroup (32, 1, 1).
#define GEMM_T2(NAME, R, BT)                                                                      \
kernel void NAME(device const uint *W [[buffer(0)]],                                              \
                 device const half *S [[buffer(1)]],                                              \
                 device const half *x [[buffer(2)]],                                              \
                 device half *y [[buffer(3)]],                                                    \
                 constant GemvP &p [[buffer(4)]],                                                 \
                 uint3 gid [[thread_position_in_grid]],                                           \
                 uint lane [[thread_index_in_simdgroup]]) {                                       \
    uint r0 = gid.y * R;                                                                          \
    uint t0 = gid.z * BT;                                                                         \
    uint words = p.groups * 8u;                                                                   \
    uint cols4 = p.groups * 32u;                                                                  \
    device const uint *w = W + r0 * words + lane;                                                 \
    device const half *s = S + r0 * p.groups + (lane >> 3);                                       \
    device const half4 *xp = (device const half4 *)(x + lane * 16u) + t0 * cols4;                 \
    UNPACK_CONSTS                                                                                 \
    float acc[R * BT];                                                                            \
    for (uint i = 0; i < R * BT; ++i) acc[i] = 0.0f;                                              \
    for (uint b = 0; b < words; b += 32u) {                                                       \
        float4 w0[R], w1[R], w2[R], w3[R];                                                        \
        float sc[R];                                                                              \
        for (uint i = 0; i < R; ++i) {                                                            \
            uint4 bits = uint4(w[i * words + b]);                                                 \
            w0[i] = float4(int4((bits & (c0 * 3u)) - c0)) * d0;                                   \
            w1[i] = float4(int4((bits & (c1 * 3u)) - c1)) * d1;                                   \
            w2[i] = float4(int4((bits & (c2 * 3u)) - c2)) * d2;                                   \
            w3[i] = float4(int4((bits & (c3 * 3u)) - c3)) * d3;                                   \
            sc[i] = float(s[i * p.groups + (b >> 3)]);                                            \
        }                                                                                         \
        for (uint t = 0; t < BT; ++t) {                                                           \
            device const half4 *xt = xp + t * cols4 + b * 4u;                                     \
            float4 x0 = float4(xt[0]), x1 = float4(xt[1]), x2 = float4(xt[2]), x3 = float4(xt[3]);\
            for (uint i = 0; i < R; ++i) {                                                        \
                float v = dot(w0[i], x0) + dot(w1[i], x1) + dot(w2[i], x2) + dot(w3[i], x3);      \
                acc[i * BT + t] += sc[i] * v;                                                     \
            }                                                                                     \
        }                                                                                         \
    }                                                                                             \
    for (uint i = 0; i < R; ++i) {                                                                \
        for (uint t = 0; t < BT; ++t) {                                                           \
            float v = simd_sum(acc[i * BT + t]);                                                  \
            if (lane == 0u) y[(t0 + t) * p.rows + r0 + i] = half(v);                              \
        }                                                                                         \
    }                                                                                             \
}

GEMM_T2(gemm_t2_r1b4, 1, 4)
GEMM_T2(gemm_t2_r2b4, 2, 4)
GEMM_T2(gemm_t2_r2b8, 2, 8)
GEMM_T2(gemm_t2_r4b4, 4, 4)

// Batched kernel for many tokens. A threadgroup of 128 threads (4 SIMD groups) computes a tile of
// 64 rows x 32 tokens. For every block of 32 columns the threads expand the weights (scale folded
// in, which is exact: 0 or +-scale) and the inputs into threadgroup memory; then every SIMD group
// multiplies its 32 x 16 part with the 8 x 8 matrix intrinsics. Tokens past `n` are padded with
// zeros and rows past `rows` are skipped, so any batch size works.
// A matrix with few rows gives too few threadgroups to fill the GPU, so the columns can be split
// over `ksplit` threadgroups; the partial sums then go to `part` as floats and gemm_reduce adds
// them up.
// Threadgroups (ceil(rows / 64), ceil(n / 32), ksplit), 128 threads each.
struct GemmP { uint rows; uint groups; uint n; uint ksplit; };

kernel void gemm_t2_tile(device const uint *W [[buffer(0)]],
                         device const half *S [[buffer(1)]],
                         device const half *x [[buffer(2)]],
                         device half *y [[buffer(3)]],
                         constant GemmP &p [[buffer(4)]],
                         device float *part [[buffer(5)]],
                         uint3 tg [[threadgroup_position_in_grid]],
                         ushort ti [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[64 * 32]; // [row][column], reused as [row][token] for the result
    threadgroup float sb[32 * 32]; // [column][token]
    const uint words = p.groups * 8u;
    const uint cols = p.groups * 128u;

    // Weights: this thread expands one word (16 columns) of one row.
    const uint a_row = ti >> 1, a_half = ti & 1u;
    const uint row = tg.x * 64u + a_row;
    const bool a_on = row < p.rows;
    device const uint *w = W + row * words + a_half;
    device const half *s = S + row * p.groups;
    threadgroup float4 *sa_w = (threadgroup float4 *)(sa + a_row * 32u + a_half * 16u);

    // Inputs: this thread copies 8 columns of one token.
    const uint b_tok = ti >> 2, b_part = ti & 3u;
    const uint tok = tg.y * 32u + b_tok;
    const bool b_on = tok < p.n;
    device const half4 *xp = (device const half4 *)(x + tok * cols + b_part * 8u);
    threadgroup float *sb_w = sb + b_part * 8u * 32u + b_tok;

    simdgroup_float8x8 ma[4], mb[2], mc[8];
    for (uint i = 0; i < 8; ++i) mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint ry = (sg & 1u) * 32u; // first row of this SIMD group's part
    const uint tx = (sg >> 1) * 16u; // first token of it

    const uint kb_n = cols / 32u / p.ksplit;
    for (uint kb = tg.z * kb_n; kb < (tg.z + 1u) * kb_n; ++kb) {
        uint bits = 0x55555555u; // every slot 1: weight 0
        float sc = 0.0f;
        if (a_on) {
            bits = w[kb * 2u];
            sc = float(s[kb >> 2]);
        }
        for (uint j = 0; j < 4; ++j) {
            uint4 q = (uint4(bits >> (8u * j)) >> uint4(0u, 2u, 4u, 6u)) & 3u;
            sa_w[j] = float4(int4(q) - 1) * sc;
        }
        float4 x0 = 0.0f, x1 = 0.0f;
        if (b_on) {
            x0 = float4(xp[kb * 8u]);
            x1 = float4(xp[kb * 8u + 1u]);
        }
        for (uint j = 0; j < 4; ++j) {
            sb_w[j * 32u] = x0[j];
            sb_w[(4u + j) * 32u] = x1[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ik = 0; ik < 4; ++ik) {
            for (uint i = 0; i < 4; ++i) simdgroup_load(ma[i], sa, 32, ulong2(ik * 8u, ry + i * 8u));
            for (uint i = 0; i < 2; ++i) simdgroup_load(mb[i], sb, 32, ulong2(tx + i * 8u, ik * 8u));
            for (uint i = 0; i < 8; ++i) simdgroup_multiply_accumulate(mc[i], ma[i >> 1], mb[i & 1u], mc[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < 8; ++i) simdgroup_store(mc[i], sa, 32, ulong2(tx + (i & 1u) * 8u, ry + (i >> 1) * 8u));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!a_on) return;
    // This thread writes 16 tokens of one row.
    const uint t0 = tg.y * 32u + a_half * 16u;
    for (uint j = 0; j < 16; ++j) {
        if (t0 + j >= p.n) break;
        float v = sa[a_row * 32u + a_half * 16u + j];
        if (p.ksplit == 1u) {
            y[(t0 + j) * p.rows + row] = half(v);
        } else {
            part[(tg.z * p.n + t0 + j) * p.rows + row] = v;
        }
    }
}

// The same tile for short batches: 64 rows x 8 tokens per threadgroup of 64 threads (2 SIMD
// groups, 32 rows each). A pass over a few draft tokens then costs a few tokens' worth of
// arithmetic instead of a full 32-token tile.
// Threadgroups (ceil(rows / 64), ceil(n / 8), ksplit), 64 threads each.
kernel void gemm_t2_tile8(device const uint *W [[buffer(0)]],
                          device const half *S [[buffer(1)]],
                          device const half *x [[buffer(2)]],
                          device half *y [[buffer(3)]],
                          constant GemmP &p [[buffer(4)]],
                          device float *part [[buffer(5)]],
                          uint3 tg [[threadgroup_position_in_grid]],
                          ushort ti [[thread_index_in_threadgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sa[64 * 32]; // [row][column], reused as [row][token] for the result
    threadgroup float sb[32 * 8];  // [column][token]
    const uint words = p.groups * 8u;
    const uint cols = p.groups * 128u;

    // Weights: this thread expands 32 columns (two words) of one row.
    const uint row = tg.x * 64u + ti;
    const bool a_on = row < p.rows;
    device const uint *w = W + row * words;
    device const half *s = S + row * p.groups;
    threadgroup float4 *sa_w = (threadgroup float4 *)(sa + ti * 32u);

    // Inputs: this thread copies 4 columns of one token.
    const uint b_tok = ti >> 3, b_part = ti & 7u;
    const uint tok = tg.y * 8u + b_tok;
    const bool b_on = tok < p.n;
    device const half4 *xp = (device const half4 *)(x + tok * cols + b_part * 4u);
    threadgroup float *sb_w = sb + b_part * 4u * 8u + b_tok;

    simdgroup_float8x8 ma[4], mb, mc[4];
    for (uint i = 0; i < 4; ++i) mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint ry = sg * 32u; // first row of this SIMD group's part

    const uint kb_n = cols / 32u / p.ksplit;
    for (uint kb = tg.z * kb_n; kb < (tg.z + 1u) * kb_n; ++kb) {
        uint2 bits = uint2(0x55555555u); // every slot 1: weight 0
        float sc = 0.0f;
        if (a_on) {
            bits = uint2(w[kb * 2u], w[kb * 2u + 1u]);
            sc = float(s[kb >> 2]);
        }
        for (uint j = 0; j < 4; ++j) {
            uint4 q0 = (uint4(bits.x >> (8u * j)) >> uint4(0u, 2u, 4u, 6u)) & 3u;
            uint4 q1 = (uint4(bits.y >> (8u * j)) >> uint4(0u, 2u, 4u, 6u)) & 3u;
            sa_w[j] = float4(int4(q0) - 1) * sc;
            sa_w[4u + j] = float4(int4(q1) - 1) * sc;
        }
        float4 xv = 0.0f;
        if (b_on) xv = float4(xp[kb * 8u]);
        for (uint j = 0; j < 4; ++j) sb_w[j * 8u] = xv[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ik = 0; ik < 4; ++ik) {
            for (uint i = 0; i < 4; ++i) simdgroup_load(ma[i], sa, 32, ulong2(ik * 8u, ry + i * 8u));
            simdgroup_load(mb, sb, 8, ulong2(0, ik * 8u));
            for (uint i = 0; i < 4; ++i) simdgroup_multiply_accumulate(mc[i], ma[i], mb, mc[i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < 4; ++i) simdgroup_store(mc[i], sa, 32, ulong2(0, ry + i * 8u));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!a_on) return;
    const uint t0 = tg.y * 8u;
    for (uint j = 0; j < 8; ++j) {
        if (t0 + j >= p.n) break;
        float v = sa[ti * 32u + j];
        if (p.ksplit == 1u) {
            y[(t0 + j) * p.rows + row] = half(v);
        } else {
            part[(tg.z * p.n + t0 + j) * p.rows + row] = v;
        }
    }
}

// Sum of the partial results of a column-split tiled kernel. Grid (rows * n).
kernel void gemm_reduce(device const float *part [[buffer(0)]],
                        device half *y [[buffer(1)]],
                        constant GemmP &p [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    uint total = p.rows * p.n;
    float v = 0.0f;
    for (uint z = 0; z < p.ksplit; ++z) v += part[z * total + i];
    y[i] = half(v);
}

// Memory floor probe for the profiler: the access pattern of gemv_t2_sgm* with next to no
// arithmetic. Not used for inference.
kernel void probe_mem_sg(device const uint *W [[buffer(0)]],
                         device const half *S [[buffer(1)]],
                         device const half *x [[buffer(2)]],
                         device half *y [[buffer(3)]],
                         constant GemvP &p [[buffer(4)]],
                         uint2 gid [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    uint words = p.groups * 8u;
    device const uint *w = W + gid.y * words + lane;
    uint acc = 0u;
    for (uint b = 0; b < words; b += 32u) acc += popcount(w[b]);
    uint v = simd_sum(acc);
    if (lane == 0u) y[gid.y] = half(float(v & 1023u));
}

// ---------------------------------------------------------------------------------------------
// SwiGLU: m = silu(gate) * up, with gate and up stored back to back. Grid (n, batch).
kernel void swiglu(device const half *gu [[buffer(0)]],
                   device half *m [[buffer(1)]],
                   constant uint &n [[buffer(2)]],
                   uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    if (i >= n) return;
    device const half *g = gu + gid.y * 2u * n;
    m[gid.y * n + i] = half(float(silu_h(g[i])) * float(g[n + i]));
}

// ---------------------------------------------------------------------------------------------
// Gated delta net (linear attention). The recurrences run over the tokens of the batch inside
// one thread; `ps` is the per-token stride of the input projection [qkv | z | b | a].

// Depthwise conv over the last 4 inputs, silu, and (with `write`) shift of the 3-step conv state.
// Grid (n).
struct ConvP { uint n; uint batch; uint ps; uint write; };

kernel void gdn_conv(device const half *proj [[buffer(0)]],
                     device const half *cw [[buffer(1)]],
                     device half *cs [[buffer(2)]],
                     device half *out [[buffer(3)]],
                     constant ConvP &p [[buffer(4)]],
                     uint c [[thread_position_in_grid]]) {
    if (c >= p.n) return;
    half s0 = cs[c], s1 = cs[p.n + c], s2 = cs[2u * p.n + c];
    device const half *w = cw + c * 4u;
    float w0 = float(w[0]), w1 = float(w[1]), w2 = float(w[2]), w3 = float(w[3]);
    for (uint b = 0; b < p.batch; ++b) {
        half q = proj[b * p.ps + c];
        float v = w0 * float(s0) + w1 * float(s1) + w2 * float(s2) + w3 * float(q);
        out[b * p.n + c] = silu_h(half(v));
        s0 = s1;
        s1 = s2;
        s2 = q;
    }
    if (p.write == 0u) return;
    cs[c] = s0;
    cs[p.n + c] = s1;
    cs[2u * p.n + c] = s2;
}

// L2-normalise q and k per key head and fold the 1/sqrt(dk) readout scale into q.
// Grid (dk, 2 * hk, batch), threadgroup (dk, 1, 1): rows [0, hk) are q heads, [hk, 2hk) k heads.
struct QkP { uint hk; uint dk; uint cs; };

kernel void gdn_qk_norm(device const half *conv_out [[buffer(0)]],
                        device half *qk [[buffer(1)]],
                        constant QkP &p [[buffer(2)]],
                        uint3 gid [[thread_position_in_grid]]) {
    threadgroup float part[128];
    uint d = gid.x;
    uint row = gid.y;
    float v = float(conv_out[gid.z * p.cs + row * p.dk + d]);
    part[d] = v * v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint j = 0; j < p.dk; ++j) tot += part[j];
    float inv = rsqrt(float(p.dk));
    float inv2 = 1.0f / float(p.dk);
    half normed = half(v * rsqrt(tot / float(p.dk) + 1e-6f * inv2));
    float f = row < p.hk ? inv2 : float(half(inv));
    qk[(gid.z * 2u * p.hk + row) * p.dk + d] = half(float(normed) * f);
}

// State update and readout. Grid (dv, hv): one thread per row of the per-head state matrix.
// `write` = 0 leaves the stored state as it was (speculative pass); only gdn_step_sg honours it.
struct GdnP { uint hv; uint hk; uint dk; uint dv; uint b_off; uint a_off; uint v_off; uint batch; uint ps; uint cs; uint write; };

kernel void gdn_step(device const half *qkb [[buffer(0)]],
                     device const half *conv_out [[buffer(1)]],
                     device const half *projb [[buffer(2)]],
                     device const float *adt [[buffer(3)]],
                     device float *state [[buffer(4)]],
                     device half *y [[buffer(5)]],
                     constant GdnP &p [[buffer(6)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint dv = gid.x;
    uint hv = gid.y;
    if (dv >= p.dv || hv >= p.hv) return;
    uint hk = hv / (p.hv / p.hk);
    device float *s = state + (hv * p.dv + dv) * p.dk;
    for (uint b = 0; b < p.batch; ++b) {
        device const half *proj = projb + b * p.ps;
        device const half *q = qkb + (b * 2u * p.hk + hk) * p.dk;
        device const half *k = qkb + (b * 2u * p.hk + p.hk + hk) * p.dk;

        half apd = half(float(proj[p.a_off + hv]) + adt[p.hv + hv]);
        float af = float(apd);
        half sp = half(max(af, 0.0f) + log(1.0f + exp(-fabs(af))));
        float g = exp(-adt[hv] * float(sp));
        float beta = float(half(sigmoidf(float(proj[p.b_off + hv]))));

        float kv = 0.0f;
        for (uint i = 0; i < p.dk; ++i) kv += s[i] * g * float(k[i]);
        float delta = (float(conv_out[b * p.cs + p.v_off + hv * p.dv + dv]) - kv) * beta;
        float out = 0.0f;
        for (uint i = 0; i < p.dk; ++i) {
            float ns = s[i] * g + float(k[i]) * delta;
            s[i] = ns;
            out += ns * float(q[i]);
        }
        y[(b * p.hv + hv) * p.dv + dv] = half(out);
    }
}

// Same step with one state row split across the 32 lanes of a SIMD group (dk / 32 elements per
// lane), which cuts the latency of the two passes over the row; the work itself is tiny.
// Grid (32, hv * dv), threadgroup (32, 1). The two sums are folded per lane, so the result
// differs from gdn_step in the last float bits.
kernel void gdn_step_sg(device const half *qkb [[buffer(0)]],
                        device const half *conv_out [[buffer(1)]],
                        device const half *projb [[buffer(2)]],
                        device const float *adt [[buffer(3)]],
                        device float *state [[buffer(4)]],
                        device half *y [[buffer(5)]],
                        constant GdnP &p [[buffer(6)]],
                        uint2 gid [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    uint hv = gid.y / p.dv;
    uint dv = gid.y - hv * p.dv;
    uint hk = hv / (p.hv / p.hk);
    uint per = p.dk >> 5;
    uint off = lane * per;
    device float *s = state + gid.y * p.dk + off;
    float ls[8]; // this lane's part of the state row (dk <= 256)
    for (uint i = 0; i < per; ++i) ls[i] = s[i];
    for (uint b = 0; b < p.batch; ++b) {
        device const half *proj = projb + b * p.ps;
        device const half *q = qkb + (b * 2u * p.hk + hk) * p.dk + off;
        device const half *k = qkb + (b * 2u * p.hk + p.hk + hk) * p.dk + off;

        half apd = half(float(proj[p.a_off + hv]) + adt[p.hv + hv]);
        float af = float(apd);
        half sp = half(max(af, 0.0f) + log(1.0f + exp(-fabs(af))));
        float g = exp(-adt[hv] * float(sp));
        float beta = float(half(sigmoidf(float(proj[p.b_off + hv]))));

        float kv = 0.0f;
        for (uint i = 0; i < per; ++i) kv += ls[i] * g * float(k[i]);
        kv = simd_sum(kv);
        float delta = (float(conv_out[b * p.cs + p.v_off + hv * p.dv + dv]) - kv) * beta;
        float out = 0.0f;
        for (uint i = 0; i < per; ++i) {
            float ns = ls[i] * g + float(k[i]) * delta;
            ls[i] = ns;
            out += ns * float(q[i]);
        }
        out = simd_sum(out);
        if (lane == 0u) y[(b * p.hv + hv) * p.dv + dv] = half(out);
    }
    if (p.write == 0u) return;
    for (uint i = 0; i < per; ++i) s[i] = ls[i];
}

// Per-head RMSNorm of the readout, gated by silu(z). Grid (dv, hv, batch), threadgroup (dv, 1, 1).
struct GnormP { uint dv; uint hv; uint z_off; uint ps; float eps; };

kernel void gdn_out_norm(device const half *y [[buffer(0)]],
                         device const half *w [[buffer(1)]],
                         device const half *proj [[buffer(2)]],
                         device half *o [[buffer(3)]],
                         constant GnormP &p [[buffer(4)]],
                         uint3 gid [[thread_position_in_grid]]) {
    threadgroup float part[128];
    uint d = gid.x;
    uint i = gid.y * p.dv + d;
    uint base = gid.z * p.hv * p.dv;
    float v = float(y[base + i]);
    part[d] = v * v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint j = 0; j < p.dv; ++j) tot += part[j];
    half normed = half(v * rsqrt(tot / float(p.dv) + p.eps) * float(w[d]));
    float z = float(proj[gid.z * p.ps + p.z_off + i]);
    o[base + i] = half(z * sigmoidf(z) * float(normed));
}

// ---------------------------------------------------------------------------------------------
// Full attention. Token b of the batch sits at position pos + b and attends to [0, pos + b].

// Per-head RMSNorm and partial RoPE of q and k; k and v are appended to the KV cache.
// Grid (hd, nh + 2 * nkv, batch), threadgroup (hd, 1, 1).
struct PrepP { uint nh; uint nkv; uint hd; uint pos; uint rot; uint ps; float eps; float log_base; };

kernel void attn_prep(device const half *projb [[buffer(0)]],
                      device const half *qn [[buffer(1)]],
                      device const half *kn [[buffer(2)]],
                      device half *qout [[buffer(3)]],
                      device half *kc [[buffer(4)]],
                      device half *vc [[buffer(5)]],
                      constant PrepP &p [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]]) {
    threadgroup float part[256];
    threadgroup half nrm[256];
    uint d = gid.x;
    uint row = gid.y;
    uint pos = p.pos + gid.z;
    device const half *proj = projb + gid.z * p.ps;
    bool is_q = row < p.nh;
    bool is_k = !is_q && row < p.nh + p.nkv;
    uint slot = pos * p.nkv * p.hd;
    uint k_base = p.nh * 2u * p.hd;
    if (!is_q && !is_k) {
        uint h = row - p.nh - p.nkv;
        vc[slot + h * p.hd + d] = proj[k_base + (p.nkv + h) * p.hd + d];
        return;
    }
    uint h = is_q ? row : row - p.nh;
    float v = is_q ? float(proj[h * 2u * p.hd + d]) : float(proj[k_base + h * p.hd + d]);
    part[d] = v * v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint j = 0; j < p.hd; ++j) tot += part[j];
    float w = is_q ? float(qn[d]) : float(kn[d]);
    half n = half(v * rsqrt(tot / float(p.hd) + p.eps) * w);
    nrm[d] = n;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    half res = n;
    if (d < p.rot) {
        uint half_rot = p.rot >> 1;
        uint f = d < half_rot ? d : d - half_rot;
        float theta = float(pos) * exp(-float(f) * (p.log_base / float(half_rot)));
        float c = cos(theta), s = sin(theta);
        float x1 = float(nrm[f]), x2 = float(nrm[f + half_rot]);
        res = half(d < half_rot ? x1 * c - x2 * s : x1 * s + x2 * c);
    }
    if (is_q) qout[(gid.z * p.nh + h) * p.hd + d] = res;
    else kc[slot + h * p.hd + d] = res;
}

// scores[b, h, t] = scale * q[b, h] . K[t, h / (nh / nkv)]. Grid (pos + batch, nh, batch).
struct ScoreP { uint nh; uint nkv; uint hd; uint pos; uint stride; uint ps; float scale; };

kernel void attn_scores(device const half *q [[buffer(0)]],
                        device const half *kc [[buffer(1)]],
                        device float *scores [[buffer(2)]],
                        constant ScoreP &p [[buffer(3)]],
                        uint3 gid [[thread_position_in_grid]]) {
    uint t = gid.x, h = gid.y;
    if (t > p.pos + gid.z || h >= p.nh) return;
    device const half4 *qp = (device const half4 *)(q + (gid.z * p.nh + h) * p.hd);
    device const half4 *kp = (device const half4 *)(kc + (t * p.nkv + h / (p.nh / p.nkv)) * p.hd);
    float acc = 0.0f;
    for (uint i = 0; i < (p.hd >> 2); ++i) acc += dot(float4(qp[i]), float4(kp[i]));
    scores[(gid.z * p.nh + h) * p.stride + t] = float(half(acc * p.scale));
}

// Softmax over t, in place. Grid (nh, batch): one thread per head and token.
kernel void attn_softmax(device float *scores [[buffer(0)]],
                         constant ScoreP &p [[buffer(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.nh) return;
    uint T = p.pos + gid.y + 1u;
    device float *s = scores + (gid.y * p.nh + gid.x) * p.stride;
    float m = s[0];
    for (uint t = 1; t < T; ++t) m = max(m, s[t]);
    float tot = 0.0f;
    for (uint t = 0; t < T; ++t) {
        float e = exp(s[t] - m);
        s[t] = e;
        tot += e;
    }
    float inv = 1.0f / tot;
    for (uint t = 0; t < T; ++t) s[t] = float(half(s[t] * inv));
}

// o[b, h, d] = (sum_t p[b, h, t] * V[t, kvh, d]) * sigmoid(gate[b, h, d]). Grid (hd, nh, batch).
kernel void attn_out(device const float *scores [[buffer(0)]],
                     device const half *vc [[buffer(1)]],
                     device const half *proj [[buffer(2)]],
                     device half *o [[buffer(3)]],
                     constant ScoreP &p [[buffer(4)]],
                     uint3 gid [[thread_position_in_grid]]) {
    uint d = gid.x, h = gid.y;
    if (d >= p.hd || h >= p.nh) return;
    uint T = p.pos + gid.z + 1u;
    device const float *s = scores + (gid.z * p.nh + h) * p.stride;
    uint kvh = h / (p.nh / p.nkv);
    float acc = 0.0f;
    for (uint t = 0; t < T; ++t) acc += s[t] * float(vc[(t * p.nkv + kvh) * p.hd + d]);
    half gate = half(sigmoidf(float(proj[gid.z * p.ps + h * 2u * p.hd + p.hd + d])));
    o[(gid.z * p.nh + h) * p.hd + d] = half(float(half(acc)) * float(gate));
}

// The same two kernels with the positions split across the lanes of a SIMD group (lane l takes
// t = l, l + 32, ...): latency drops from T to T / 32 steps. Sums are folded per lane, so results
// differ from the serial kernels in the last float bits.

// Grid (32, nh, batch), threadgroup (32, 1, 1).
kernel void attn_softmax_sg(device float *scores [[buffer(0)]],
                            constant ScoreP &p [[buffer(1)]],
                            uint3 gid [[thread_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    uint T = p.pos + gid.z + 1u;
    device float *s = scores + (gid.z * p.nh + gid.y) * p.stride;
    float m = -INFINITY;
    for (uint t = lane; t < T; t += 32u) m = max(m, s[t]);
    m = simd_max(m);
    float tot = 0.0f;
    for (uint t = lane; t < T; t += 32u) {
        float e = exp(s[t] - m);
        s[t] = e;
        tot += e;
    }
    float inv = 1.0f / simd_sum(tot);
    for (uint t = lane; t < T; t += 32u) s[t] = float(half(s[t] * inv));
}

// Grid (32, nh * hd, batch), threadgroup (32, 1, 1).
kernel void attn_out_sg(device const float *scores [[buffer(0)]],
                        device const half *vc [[buffer(1)]],
                        device const half *proj [[buffer(2)]],
                        device half *o [[buffer(3)]],
                        constant ScoreP &p [[buffer(4)]],
                        uint3 gid [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    uint h = gid.y / p.hd;
    uint d = gid.y - h * p.hd;
    uint T = p.pos + gid.z + 1u;
    device const float *s = scores + (gid.z * p.nh + h) * p.stride;
    device const half *v = vc + (h / (p.nh / p.nkv)) * p.hd + d;
    uint vs = p.nkv * p.hd;
    float acc = 0.0f;
    for (uint t = lane; t < T; t += 32u) acc += s[t] * float(v[t * vs]);
    acc = simd_sum(acc);
    if (lane != 0u) return;
    half gate = half(sigmoidf(float(proj[gid.z * p.ps + h * 2u * p.hd + p.hd + d])));
    o[(gid.z * p.nh + h) * p.hd + d] = half(float(half(acc)) * float(gate));
}

// Reads one word of every 16 KiB page of the weights, so the pages are wired at load time instead
// of during the first tokens. Grid (pages).
kernel void touch_pages(device const uint *W [[buffer(0)]],
                        device uint *out [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    if (W[i * 4096u] == 0x7e57ab1eu) out[0] += 1u;
}

// ---------------------------------------------------------------------------------------------
// Greedy sampling: out[at + t] = argmax(logits of token t), lowest index on ties.
// One threadgroup of 1024 threads per token, grid of threadgroups (1, tokens).
struct ArgmaxP { uint n; uint at; };

kernel void argmax(device const half *all_logits [[buffer(0)]],
                   device uint *out [[buffer(1)]],
                   constant ArgmaxP &p [[buffer(2)]],
                   uint3 tg [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]]) {
    device const half *logits = all_logits + tg.y * p.n;
    threadgroup float bv[1024];
    threadgroup uint bi[1024];
    uint per = (p.n + 1023u) >> 10;
    uint lo = tid * per;
    uint hi = min(lo + per, p.n);
    float best = -INFINITY;
    uint idx = lo;
    for (uint i = lo; i < hi; ++i) {
        float v = float(logits[i]);
        if (v > best) { best = v; idx = i; }
    }
    bv[tid] = best;
    bi[tid] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0u) return;
    for (uint j = 1; j < 1024u; ++j) {
        if (bv[j] > best) { best = bv[j]; idx = bi[j]; }
    }
    out[p.at + tg.y] = idx;
}
