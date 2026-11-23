//! Forward-pass encoder: turns the forward pass of one token, or of a batch of tokens, into
//! dispatches on a Metal compute encoder.
//!
//! The token id travels GPU-side (the argmax kernel writes `tok[step + 1]`, the embedding kernel
//! reads `tok[step]`), so the CPU never has to see a token before the next one can be encoded.
//! That lets the caller keep several command buffers in flight.
//!
//! Nothing here allocates or calls through a pointer after `init`.
//!
//! The encoder is generic over the model shape: `Engine` runs the 2-bit main model, `DraftEngine`
//! the 4-bit draft model on the same device, in the same command buffers.

const std = @import("std");
const metal = @import("metal.zig");
const model = @import("model.zig");

const kernel_source: [*:0]const u8 = @embedFile("kernels.metal");

/// A quantised matrix: weight words, scales and (4-bit only) biases, as byte offsets in the file.
pub const Mat = struct { w: usize, s: usize, b: usize = 0, rows: u32, groups: u32 };

const Layer = struct {
    linear: bool,
    n1: usize,
    n2: usize,
    in: Mat,
    out: Mat,
    mlp: Mat,
    down: Mat,
    // linear attention
    conv: usize = 0,
    adt: usize = 0,
    gnorm: usize = 0,
    state: usize = 0,
    /// Offset of this layer's input projection in `lin_proj`.
    proj: usize = 0,
    // full attention
    qn: usize = 0,
    kn: usize = 0,
    kv: usize = 0,
};

// Kernel parameter blocks; field order matches the structs in kernels.metal.
const EmbedP = extern struct { step: u32, cols: u32, groups: u32 };
const NormP = extern struct { n: u32, add: u32, eps: f32 };
const GemvP = extern struct { rows: u32, groups: u32 };
const GemmP = extern struct { rows: u32, groups: u32, n: u32, ksplit: u32 };
/// Column split of the tiled kernel: `gemm_split` ways up to `split_rows` rows, 2 ways up to
/// `split2_rows`, none above. `split_part_rows` bounds rows * split for the partial sums.
const split_rows = 8192;
const split2_rows = 20480;
const split_part_rows = 2 * split2_rows;
const ConvP = extern struct { n: u32, batch: u32, ps: u32, write: u32 };
const QkP = extern struct { hk: u32, dk: u32, cs: u32 };
const GdnP = extern struct { hv: u32, hk: u32, dk: u32, dv: u32, b_off: u32, a_off: u32, v_off: u32, batch: u32, ps: u32, cs: u32, write: u32 };
const GnormP = extern struct { dv: u32, hv: u32, z_off: u32, ps: u32, eps: f32 };
const PrepP = extern struct { nh: u32, nkv: u32, hd: u32, pos: u32, rot: u32, ps: u32, eps: f32, log_base: f32 };
const ScoreP = extern struct { nh: u32, nkv: u32, hd: u32, pos: u32, stride: u32, ps: u32, scale: f32 };
const ArgmaxP = extern struct { n: u32, at: u32 };
const A4P = extern struct { rows: u32, groups: u32, n: u32 };
/// Tokens per thread of the batched 4-bit kernel; the draft's `max_batch` is a multiple of it.
pub const a4_batch = 8;

/// What a pass does with the recurrent state of the linear-attention layers: `advance` stores
/// the state after its last token, `peek` leaves the stored state alone (speculative pass).
pub const State = enum { advance, peek };

/// Parameter blocks a pass reads from GPU memory instead of the command stream, so that an
/// earlier pass on the GPU can set its position and recurrent batch (chained speculation). One
/// engine's set is `dyn_blocks` blocks of `dyn_block` bytes at `off`, in this order: EmbedP,
/// PrepP, ScoreP, ConvP, GdnP. `bound` is at least the position; it sizes the attention grid.
pub const Dyn = struct { buf: metal.Buffer, off: usize, bound: u32 };
pub const dyn_block = 256;
pub const dyn_blocks = 5;

/// Where a pass sits: a position the CPU knows, or one held in GPU memory.
pub const At = union(enum) { cpu: u32, gpu: Dyn };

// kernels.metal (chain_set) rewrites these fields in place.
comptime {
    std.debug.assert(@offsetOf(EmbedP, "step") == 0 and @offsetOf(PrepP, "pos") == 12 and @offsetOf(ScoreP, "pos") == 12);
    std.debug.assert(@offsetOf(ConvP, "batch") == 4 and @offsetOf(GdnP, "batch") == 28);
}

/// Kernel classes of one token, for the per-class time breakdown (`tern prof`).
pub const Part = enum(u5) {
    embed,
    norm,
    gemv_in,
    gemv_out,
    gemv_mlp,
    gemv_down,
    gemv_head,
    swiglu,
    gdn_conv,
    gdn_qk,
    gdn_step,
    gdn_norm,
    attn_prep,
    attn_scores,
    attn_softmax,
    attn_out,
    argmax,
};

fn size1(n: usize) metal.Size {
    return .{ .w = n, .h = 1, .d = 1 };
}

fn size2(w: usize, h: usize) metal.Size {
    return .{ .w = w, .h = h, .d = 1 };
}

fn size3(w: usize, h: usize, d: usize) metal.Size {
    return .{ .w = w, .h = h, .d = d };
}

pub const Config = struct {
    max_ctx: u32 = 4096,
    /// Most tokens one batched pass may carry (prefill, verification).
    max_batch: u32 = 32,
    /// GEMV kernel name, or "auto" for the per-shape default.
    gemv: [*:0]const u8 = "auto",
    /// Rows per GEMV thread; ignored with "auto".
    gemv_rows: u32 = 4,
    /// Batched product: rows per thread and tokens served by one read of the weights. A pass of
    /// two tokens has a lane kernel of its own.
    gemm_rows: u32 = 2,
    gemm_tokens: u32 = 4,
    /// Batches of at least this many tokens use the tiled kernel (64 rows x 32 tokens per
    /// threadgroup), which takes any batch size; 0 turns it off.
    gemm_tile: u32 = 16,
    /// Batches of at least this many tokens, and fewer than `gemm_tile`, use the 8-token tile;
    /// 0 turns it off.
    gemm_tile8: u32 = 8,
    /// Column split of the tiled kernel for matrices with few rows (1 = off, 2 or 4).
    gemm_split: u32 = 4,
    /// Lane-split variants of the latency-bound kernels. They sum in a different order than the
    /// serial kernels, so every combination has to be checked against the reference tokens.
    gdn_sg: bool = true,
    /// Bit 0: softmax of full attention, bit 1: its value sum.
    attn_sg: u32 = 1,
};

pub const Engine = EngineOf(model.cfg);
pub const DraftEngine = EngineOf(model.cfg_draft);

pub fn EngineOf(comptime cfg: type) type {
    return struct {
        const Self = @This();

        const conv_state_bytes = (cfg.conv_taps - 1) * cfg.conv_dim * 2;
        const delta_state_bytes = cfg.lin_v_heads * cfg.lin_dv * cfg.lin_dk * 4;
        const lin_state_bytes = conv_state_bytes + delta_state_bytes;
        /// Length of a linear layer's input projection [qkv | z | b | a].
        const lin_ps = cfg.conv_dim + cfg.lin_v_heads * cfg.lin_dv + 2 * cfg.lin_v_heads;

        device: metal.Device,
        queue: metal.Queue,
        lib: metal.Library,
        file: model.File,
        max_ctx: u32,
        max_batch: u32,
        /// Bit per `Part`; a cleared bit skips that kernel class (profiling only).
        mask: u32,

        weights: metal.Buffer,
        x: metal.Buffer,
        xn: metal.Buffer,
        proj: metal.Buffer,
        /// Input projections of the linear layers, one slot of `max_batch` tokens per layer: they
        /// outlive the pass so that `encodeCommit` can replay the accepted tokens of a `peek` pass.
        lin_proj: metal.Buffer,
        tmp: metal.Buffer,
        r: metal.Buffer,
        conv_out: metal.Buffer,
        qk: metal.Buffer,
        y: metal.Buffer,
        qrot: metal.Buffer,
        scores: metal.Buffer,
        logits: metal.Buffer,
        /// Greedy successor of every token of the last `encodeVerify` pass.
        pred: metal.Buffer,
        tok: metal.Buffer,
        state: metal.Buffer,
        kv: metal.Buffer,

        k_embed: metal.Pipeline,
        k_norm: metal.Pipeline,
        k_gemv: metal.Pipeline,
        /// Kernel for matrices with a long input (`down`) or very many rows (`head`): more rows per
        /// thread, so the input is re-read less often.
        k_gemv_wide: metal.Pipeline,
        k_gemm: metal.Pipeline,
        /// The lane kernel for a pass of two tokens.
        k_gemm2: metal.Pipeline,
        k_swiglu: metal.Pipeline,
        k_conv: metal.Pipeline,
        k_qk: metal.Pipeline,
        k_gdn: metal.Pipeline,
        k_gnorm: metal.Pipeline,
        k_prep: metal.Pipeline,
        k_scores: metal.Pipeline,
        k_softmax: metal.Pipeline,
        k_attn_out: metal.Pipeline,
        k_argmax: metal.Pipeline,
        /// 4-bit products: one token, two tokens, `a4_batch` tokens per thread.
        k_a4: [3]metal.Pipeline,

        /// Whether the GEMV kernel splits a row across the lanes of a SIMD group.
        gemv_simd: bool,
        gemv_rows: u32,
        gemv_wide_rows: u32,
        gemm_rows: u32,
        gemm_tokens: u32,
        gemm_tile: u32,
        gemm_tile8: u32,
        gemm_split: u32,
        k_gemm_tile: metal.Pipeline,
        k_gemm_tile8: metal.Pipeline,
        k_gemm_reduce: metal.Pipeline,
        part: metal.Buffer,
        gdn_sg: bool,
        attn_sg: u32,
        /// Per-token length of a full-attention layer's projection: [q, gate per head | k | v].
        attn_ps: u32,

        layers: [cfg.layers]Layer,
        embed_w: usize,
        embed_s: usize,
        embed_b: usize,
        final_norm: usize,
        head: Mat,

        pub fn init(path: [*:0]const u8, config: Config) !Self {
            metal.init();
            const device = try metal.Device.default();
            return initOn(device, try device.library(kernel_source), path, config);
        }

        /// Opens a model on a device and kernel library that are already set up (the draft model
        /// next to the main one).
        pub fn initOn(device: metal.Device, lib: metal.Library, path: [*:0]const u8, config: Config) !Self {
            const file = try model.File.open_(path);
            const page = 16384;
            const mapped = (file.len + page - 1) / page * page;
            if (mapped > device.maxBufferLength()) return error.ModelTooLargeForOneBuffer;
            const embed = try file.find("embed.w");
            if (embed.rows != cfg.vocab or embed.cols * (32 / cfg.bits) != cfg.hidden) return error.ModelShapeMismatch;
            const max_ctx = config.max_ctx;
            const mb: usize = config.max_batch;
            if (mb == 0 or mb % config.gemm_tokens != 0) return error.BatchNotMultipleOfGemmTokens;
            if (cfg.bits == 4 and mb % a4_batch != 0) return error.BatchNotMultipleOfA4Batch;

            var self: Self = undefined;
            self.device = device;
            self.queue = device.queue();
            self.lib = lib;
            self.file = file;
            self.max_ctx = max_ctx;
            self.max_batch = config.max_batch;
            self.mask = std.math.maxInt(u32);
            self.weights = try device.bufferNoCopy(file.base, mapped);

            self.x = try device.buffer(mb * cfg.hidden * 2);
            self.xn = try device.buffer(mb * cfg.hidden * 2);
            self.proj = try device.buffer(mb * 2 * cfg.inter * 2);
            self.lin_proj = try device.buffer(cfg.n_linear * mb * lin_ps * 2);
            self.tmp = try device.buffer(mb * cfg.inter * 2);
            self.r = try device.buffer(mb * cfg.hidden * 2);
            self.conv_out = try device.buffer(mb * cfg.conv_dim * 2);
            self.qk = try device.buffer(mb * 2 * cfg.lin_k_heads * cfg.lin_dk * 2);
            self.y = try device.buffer(mb * cfg.lin_v_heads * cfg.lin_dv * 2);
            self.qrot = try device.buffer(mb * cfg.heads * cfg.head_dim * 2);
            self.scores = try device.buffer(mb * cfg.heads * max_ctx * 4);
            self.logits = try device.buffer(mb * cfg.vocab * 2);
            self.pred = try device.buffer(mb * 4);
            self.tok = try device.buffer((@as(usize, max_ctx) + 1) * 4);
            self.state = try device.buffer(cfg.n_linear * lin_state_bytes);
            self.kv = try device.buffer(@as(usize, cfg.n_full) * 2 * kvBytes(max_ctx));

            if (cfg.bits == 4) {
                self.k_a4 = .{ try lib.pipeline("gemm_a4_r4b1"), try lib.pipeline("gemm_a4_r4b2"), try lib.pipeline("gemm_a4_r2b8") };
            }
            const gemv_name = std.mem.span(config.gemv);
            const auto = std.mem.eql(u8, gemv_name, "auto");
            self.gemv_simd = auto or std.mem.indexOf(u8, gemv_name, "_sg") != null;
            self.gemv_rows = if (auto) 4 else config.gemv_rows;
            self.gemv_wide_rows = if (auto) 16 else config.gemv_rows;
            self.k_gemv = try lib.pipeline(if (auto) "gemv_t2_sgm4" else config.gemv);
            self.k_gemv_wide = if (auto) try lib.pipeline("gemv_t2_sgm16") else self.k_gemv;
            var name_buf: [32]u8 = undefined;
            const gemm_name = try std.fmt.bufPrint(&name_buf, "gemm_t2_r{d}b{d}\x00", .{ config.gemm_rows, config.gemm_tokens });
            self.k_gemm = try lib.pipeline(@ptrCast(gemm_name.ptr));
            const gemm2_name = try std.fmt.bufPrint(&name_buf, "gemm_t2_r{d}b2\x00", .{config.gemm_rows});
            self.k_gemm2 = try lib.pipeline(@ptrCast(gemm2_name.ptr));
            self.gemm_rows = config.gemm_rows;
            self.gemm_tokens = config.gemm_tokens;
            self.gemm_tile = config.gemm_tile;
            self.k_gemm_tile = try lib.pipeline("gemm_t2_tile");
            self.k_gemm_tile8 = try lib.pipeline("gemm_t2_tile8");
            self.gemm_tile8 = config.gemm_tile8;
            self.k_gemm_reduce = try lib.pipeline("gemm_reduce");
            if (config.gemm_split != 1 and config.gemm_split != 2 and config.gemm_split != 4) return error.BadGemmSplit;
            self.gemm_split = config.gemm_split;
            self.part = try device.buffer(mb * split_part_rows * 4);

            self.k_embed = try lib.pipeline(if (cfg.bits == 4) "embed_a4" else "embed");
            self.k_norm = try lib.pipeline("add_rmsnorm");
            self.k_swiglu = try lib.pipeline("swiglu");
            self.k_conv = try lib.pipeline("gdn_conv");
            self.k_qk = try lib.pipeline("gdn_qk_norm");
            self.gdn_sg = config.gdn_sg;
            self.attn_sg = config.attn_sg;
            self.k_gdn = try lib.pipeline(if (config.gdn_sg) "gdn_step_sg" else "gdn_step");
            self.k_gnorm = try lib.pipeline("gdn_out_norm");
            self.k_prep = try lib.pipeline("attn_prep");
            self.k_scores = try lib.pipeline("attn_scores");
            self.k_softmax = try lib.pipeline(if (config.attn_sg & 1 != 0) "attn_softmax_sg" else "attn_softmax");
            self.k_attn_out = try lib.pipeline(if (config.attn_sg & 2 != 0) "attn_out_sg" else "attn_out");
            self.k_argmax = try lib.pipeline("argmax");

            // Wire the weight pages now, so the cost lands in load time and not in the first tokens.
            {
                const pool = metal.Pool.push();
                defer pool.pop();
                const cb = self.queue.commandBuffer();
                const enc = cb.encoder();
                enc.setPipeline(try lib.pipeline("touch_pages"));
                enc.setBuffer(self.weights, 0, 0);
                enc.setBuffer(self.scores, 0, 1);
                enc.dispatchThreads(size1(mapped / page), size1(256));
                enc.end();
                cb.commit();
                try cb.wait();
            }

            self.embed_w = (try file.find("embed.w")).offset;
            self.embed_s = (try file.find("embed.s")).offset;
            self.embed_b = (try file.find("embed.b")).offset;
            self.final_norm = (try file.find("norm")).offset;
            self.head = try mat(file, "head", .{});
            if (self.head.rows != cfg.vocab) return error.ModelShapeMismatch;

            var n_lin: usize = 0;
            var n_full: usize = 0;
            for (&self.layers, 0..) |*l, i| {
                l.* = .{
                    .linear = cfg.isLinear(i),
                    .n1 = (try file.findFmt("L{d}.n1", .{i})).offset,
                    .n2 = (try file.findFmt("L{d}.n2", .{i})).offset,
                    .in = try mat(file, "L{d}.in", .{i}),
                    .out = try mat(file, "L{d}.out", .{i}),
                    .mlp = try mat(file, "L{d}.mlp", .{i}),
                    .down = try mat(file, "L{d}.down", .{i}),
                };
                if (l.linear) {
                    l.conv = (try file.findFmt("L{d}.conv", .{i})).offset;
                    l.adt = (try file.findFmt("L{d}.adt", .{i})).offset;
                    l.gnorm = (try file.findFmt("L{d}.gnorm", .{i})).offset;
                    l.state = n_lin * lin_state_bytes;
                    l.proj = n_lin * mb * lin_ps * 2;
                    if (l.in.rows != lin_ps) return error.UnexpectedProjectionSize;
                    n_lin += 1;
                } else {
                    l.qn = (try file.findFmt("L{d}.qn", .{i})).offset;
                    l.kn = (try file.findFmt("L{d}.kn", .{i})).offset;
                    l.kv = n_full * 2 * kvBytes(max_ctx);
                    if (n_full == 0) self.attn_ps = l.in.rows;
                    if (l.in.rows != self.attn_ps) return error.UnexpectedProjectionSize;
                    n_full += 1;
                }
            }
            return self;
        }

        fn kvBytes(max_ctx: u32) usize {
            return @as(usize, max_ctx) * cfg.kv_heads * cfg.head_dim * 2;
        }

        pub fn mat(file: model.File, comptime fmt: []const u8, args: anytype) !Mat {
            const w = try file.findFmt(fmt ++ ".w", args);
            const s = try file.findFmt(fmt ++ ".s", args);
            // The kernels step 32 words at a time and take 4 or 16 rows per thread, without tails.
            if (s.rows != w.rows or s.cols * cfg.group != w.cols * (32 / cfg.bits) or s.cols % 4 != 0 or w.rows % 16 != 0)
                return error.UnsupportedMatrixShape;
            const b = if (cfg.bits == 4) (try file.findFmt(fmt ++ ".b", args)).offset else 0;
            return .{ .w = w.offset, .s = s.offset, .b = b, .rows = w.rows, .groups = s.cols };
        }

        pub fn tokens(self: *const Self) [*]u32 {
            return self.tok.as(u32);
        }

        // -- encoding ---------------------------------------------------------------------------
        // Every piece takes `n`, the number of tokens in the pass. Per-token vectors sit back to back
        // in the buffers, token b at b * (vector length).

        inline fn on(self: *const Self, part: Part) bool {
            return self.mask >> @backingInt(part) & 1 != 0;
        }

        /// out[t] = m * in[t] for n tokens: the GEMV kernel for one token, the tiled kernel from
        /// `gemm_tile` tokens up, the lane kernel in between (n must then be 2 or a multiple of
        /// `gemm_tokens`). The result goes to `out` at byte offset `out_off`.
        pub inline fn matmul(self: *const Self, enc: metal.Encoder, part: Part, m: Mat, in: metal.Buffer, out: metal.Buffer, out_off: usize, n: u32) void {
            if (!self.on(part)) return;
            enc.setBuffer(self.weights, m.w, 0);
            enc.setBuffer(self.weights, m.s, 1);
            enc.setBuffer(in, 0, 2);
            enc.setBuffer(out, out_off, 3);
            if (cfg.bits == 4) {
                // Any n up to `max_batch`: 4 rows per thread for one or two tokens, 2 rows and
                // `a4_batch` tokens per thread above.
                const k: usize = if (n == 1) 0 else if (n == 2) 1 else 2;
                const rows: u32 = if (k == 2) 2 else 4;
                const per: u32 = if (k == 2) a4_batch else n;
                enc.setParams(&A4P{ .rows = m.rows, .groups = m.groups, .n = n }, 4);
                enc.setBuffer(self.weights, m.b, 5);
                enc.setPipeline(self.k_a4[k]);
                enc.dispatchThreads(size3(32, m.rows / rows, (n + per - 1) / per), size3(32, 1, 1));
                return;
            }
            const wide_tile = self.gemm_tile != 0 and n >= self.gemm_tile;
            if (wide_tile or (self.gemm_tile8 != 0 and n >= self.gemm_tile8)) {
                const split: u32 = if (self.gemm_split == 1 or m.rows > split2_rows) 1 else if (m.rows <= split_rows) self.gemm_split else 2;
                const params = GemmP{ .rows = m.rows, .groups = m.groups, .n = n, .ksplit = split };
                enc.setParams(&params, 4);
                enc.setBuffer(self.part, 0, 5);
                if (wide_tile) {
                    enc.setPipeline(self.k_gemm_tile);
                    enc.dispatchGroups(size3((m.rows + 63) / 64, (n + 31) / 32, split), size3(128, 1, 1));
                } else {
                    enc.setPipeline(self.k_gemm_tile8);
                    enc.dispatchGroups(size3((m.rows + 63) / 64, (n + 7) / 8, split), size3(64, 1, 1));
                }
                if (split != 1) {
                    enc.setPipeline(self.k_gemm_reduce);
                    enc.setBuffer(self.part, 0, 0);
                    enc.setBuffer(out, out_off, 1);
                    enc.setParams(&params, 2);
                    enc.dispatchThreads(size1(m.rows * n), size1(256));
                }
                return;
            }
            enc.setParams(&GemvP{ .rows = m.rows, .groups = m.groups }, 4);
            if (n == 2) {
                enc.setPipeline(self.k_gemm2);
                enc.dispatchThreads(size3(32, m.rows / self.gemm_rows, 1), size3(32, 1, 1));
                return;
            }
            if (n > 1) {
                std.debug.assert(n % self.gemm_tokens == 0);
                enc.setPipeline(self.k_gemm);
                enc.dispatchThreads(size3(32, m.rows / self.gemm_rows, n / self.gemm_tokens), size3(32, 1, 1));
                return;
            }
            const wide = m.groups > 64 or m.rows > 100_000;
            const rows_per_thread = if (wide) self.gemv_wide_rows else self.gemv_rows;
            enc.setPipeline(if (wide) self.k_gemv_wide else self.k_gemv);
            if (self.gemv_simd) {
                enc.dispatchThreads(size2(32, m.rows / rows_per_thread), size2(32, 1));
            } else {
                enc.dispatchThreads(size1(m.rows / rows_per_thread), size1(32));
            }
        }

        /// x += r (when `add`), xn = rmsnorm(x) * weight.
        inline fn norm(self: *const Self, enc: metal.Encoder, weight: usize, add: bool, n: u32) void {
            if (!self.on(.norm)) return;
            enc.setPipeline(self.k_norm);
            enc.setBuffer(self.x, 0, 0);
            enc.setBuffer(self.r, 0, 1);
            enc.setBuffer(self.weights, weight, 2);
            enc.setBuffer(self.xn, 0, 3);
            enc.setParams(&NormP{ .n = cfg.hidden, .add = @intFromBool(add), .eps = cfg.eps }, 4);
            enc.dispatchGroups(size2(1, n), size2(256, 1));
        }

        /// Embeddings of `tok[step .. step + n]` and the first layer's input norm.
        pub fn encodeEmbed(self: *const Self, enc: metal.Encoder, step: u32, n: u32) void {
            self.embedAt(enc, .{ .cpu = step }, n);
        }

        fn embedAt(self: *const Self, enc: metal.Encoder, at: At, n: u32) void {
            if (self.on(.embed)) {
                enc.setPipeline(self.k_embed);
                enc.setBuffer(self.weights, self.embed_w, 0);
                enc.setBuffer(self.weights, self.embed_s, 1);
                enc.setBuffer(self.weights, self.embed_b, 2);
                enc.setBuffer(self.tok, 0, 3);
                enc.setBuffer(self.x, 0, 4);
                bind(enc, at, &self.embedP(cpuPos(at)), 0, 5);
                enc.dispatchThreads(size2(cfg.hidden, n), size2(256, 1));
            }
            self.norm(enc, self.layers[0].n1, false, n);
        }

        // Parameter blocks that depend on the position or on the recurrent batch.
        fn embedP(_: *const Self, step: u32) EmbedP {
            return .{ .step = step, .cols = cfg.hidden, .groups = cfg.hidden / cfg.group };
        }

        fn prepP(self: *const Self, pos: u32) PrepP {
            return .{
                .nh = cfg.heads,
                .nkv = cfg.kv_heads,
                .hd = cfg.head_dim,
                .pos = pos,
                .rot = cfg.rot_dims,
                .ps = self.attn_ps,
                .eps = cfg.eps,
                .log_base = @log(cfg.rope_base),
            };
        }

        fn scoreP(self: *const Self, pos: u32) ScoreP {
            return .{
                .nh = cfg.heads,
                .nkv = cfg.kv_heads,
                .hd = cfg.head_dim,
                .pos = pos,
                .stride = self.max_ctx,
                .ps = self.attn_ps,
                .scale = 1.0 / @sqrt(@as(f32, cfg.head_dim)),
            };
        }

        fn convP(_: *const Self, n: u32, write: bool) ConvP {
            return .{ .n = cfg.conv_dim, .batch = n, .ps = lin_ps, .write = @intFromBool(write) };
        }

        fn gdnP(_: *const Self, n: u32, write: bool) GdnP {
            const key_dim = cfg.lin_k_heads * cfg.lin_dk;
            const value_dim = cfg.lin_v_heads * cfg.lin_dv;
            return .{
                .hv = cfg.lin_v_heads,
                .hk = cfg.lin_k_heads,
                .dk = cfg.lin_dk,
                .dv = cfg.lin_dv,
                .b_off = cfg.conv_dim + value_dim,
                .a_off = cfg.conv_dim + value_dim + cfg.lin_v_heads,
                .v_off = 2 * key_dim,
                .batch = n,
                .ps = lin_ps,
                .cs = cfg.conv_dim,
                .write = @intFromBool(write),
            };
        }

        /// Fills this engine's set of GPU-held parameter blocks at byte offset `off` of `buf` for
        /// a pass at `pos` whose recurrent layers advance over `batch` tokens. Nothing that reads
        /// them may be in flight.
        pub fn dynInit(self: *const Self, buf: metal.Buffer, off: usize, pos: u32, batch: u32) void {
            const base = buf.contents() + off;
            @as(*EmbedP, @ptrCast(@alignCast(base))).* = self.embedP(pos);
            @as(*PrepP, @ptrCast(@alignCast(base + dyn_block))).* = self.prepP(pos);
            @as(*ScoreP, @ptrCast(@alignCast(base + 2 * dyn_block))).* = self.scoreP(pos);
            @as(*ConvP, @ptrCast(@alignCast(base + 3 * dyn_block))).* = self.convP(batch, true);
            @as(*GdnP, @ptrCast(@alignCast(base + 4 * dyn_block))).* = self.gdnP(batch, true);
        }

        fn cpuPos(at: At) u32 {
            return switch (at) {
                .cpu => |pos| pos,
                .gpu => 0,
            };
        }

        /// Binds parameter block `block`: from the command stream for a CPU-known position, from
        /// GPU memory otherwise.
        inline fn bind(enc: metal.Encoder, at: At, params: anytype, block: usize, index: usize) void {
            switch (at) {
                .cpu => enc.setParams(params, index),
                .gpu => |d| enc.setBuffer(d.buf, d.off + block * dyn_block, index),
            }
        }

        /// The largest batch of at most n tokens that one pass can carry (0 when n is too small).
        pub fn batchFit(self: *const Self, n: u32) u32 {
            const m = @min(n, self.max_batch);
            if (self.gemm_tile != 0 and m >= self.gemm_tile) return m;
            if (self.gemm_tile8 != 0 and m >= self.gemm_tile8) return m;
            if (m >= self.gemm_tokens) return m / self.gemm_tokens * self.gemm_tokens;
            return if (m >= 2) 2 else 0;
        }

        /// One decoder layer for n tokens at positions pos .. pos + n. Leaves the layer output in `x`
        /// and the next layer's normalised input in `xn`.
        pub fn encodeLayer(self: *const Self, enc: metal.Encoder, i: usize, pos: u32, n: u32, state: State) void {
            self.layerAt(enc, i, .{ .cpu = pos }, n, state);
        }

        /// `encodeLayer` at `at`; with a GPU-held position an `advance` pass also takes the
        /// recurrent batch from GPU memory.
        fn layerAt(self: *const Self, enc: metal.Encoder, i: usize, at: At, n: u32, state: State) void {
            const l = &self.layers[i];
            if (l.linear) {
                self.matmul(enc, .gemv_in, l.in, self.xn, self.lin_proj, l.proj, n);
                const dyn: ?Dyn = if (state == .advance and at == .gpu) at.gpu else null;
                self.encodeGdn(enc, l, n, state == .advance, dyn);
                if (self.on(.gdn_norm)) {
                    enc.setPipeline(self.k_gnorm);
                    enc.setBuffer(self.y, 0, 0);
                    enc.setBuffer(self.weights, l.gnorm, 1);
                    enc.setBuffer(self.lin_proj, l.proj, 2);
                    enc.setBuffer(self.tmp, 0, 3);
                    enc.setParams(&GnormP{ .dv = cfg.lin_dv, .hv = cfg.lin_v_heads, .z_off = cfg.conv_dim, .ps = lin_ps, .eps = cfg.eps }, 4);
                    enc.dispatchGroups(size3(1, cfg.lin_v_heads, n), size3(cfg.lin_dv, 1, 1));
                }
            } else {
                self.matmul(enc, .gemv_in, l.in, self.xn, self.proj, 0, n);
                self.encodeFullAttn(enc, l, at, n);
            }
            self.matmul(enc, .gemv_out, l.out, self.tmp, self.r, 0, n);
            self.norm(enc, l.n2, true, n);

            self.matmul(enc, .gemv_mlp, l.mlp, self.xn, self.proj, 0, n);
            if (self.on(.swiglu)) {
                enc.setPipeline(self.k_swiglu);
                enc.setBuffer(self.proj, 0, 0);
                enc.setBuffer(self.tmp, 0, 1);
                enc.setParams(&@as(u32, cfg.inter), 2);
                enc.dispatchThreads(size2(cfg.inter, n), size2(256, 1));
            }
            self.matmul(enc, .gemv_down, l.down, self.tmp, self.r, 0, n);
            self.norm(enc, if (i + 1 < cfg.layers) self.layers[i + 1].n1 else self.final_norm, true, n);
        }

        /// Gated delta net of one linear layer over the first n tokens of its slot in `lin_proj`:
        /// conv, q/k norm, and the state recurrence with its readout in `y`. With `write` the conv
        /// and delta-rule states are stored as they are after token n - 1. With `dyn` the states
        /// advance over the GPU-held batch (at most n) instead.
        fn encodeGdn(self: *const Self, enc: metal.Encoder, l: *const Layer, n: u32, write: bool, dyn: ?Dyn) void {
            const at: At = if (dyn) |d| .{ .gpu = d } else .{ .cpu = 0 };
            if (self.on(.gdn_conv)) {
                enc.setPipeline(self.k_conv);
                enc.setBuffer(self.lin_proj, l.proj, 0);
                enc.setBuffer(self.weights, l.conv, 1);
                enc.setBuffer(self.state, l.state, 2);
                enc.setBuffer(self.conv_out, 0, 3);
                bind(enc, at, &self.convP(n, write), 3, 4);
                enc.dispatchThreads(size1(cfg.conv_dim), size1(256));
            }
            if (self.on(.gdn_qk)) {
                enc.setPipeline(self.k_qk);
                enc.setBuffer(self.conv_out, 0, 0);
                enc.setBuffer(self.qk, 0, 1);
                enc.setParams(&QkP{ .hk = cfg.lin_k_heads, .dk = cfg.lin_dk, .cs = cfg.conv_dim }, 2);
                enc.dispatchGroups(size3(1, 2 * cfg.lin_k_heads, n), size3(cfg.lin_dk, 1, 1));
            }
            if (self.on(.gdn_step)) {
                enc.setPipeline(self.k_gdn);
                enc.setBuffer(self.qk, 0, 0);
                enc.setBuffer(self.conv_out, 0, 1);
                enc.setBuffer(self.lin_proj, l.proj, 2);
                enc.setBuffer(self.weights, l.adt, 3);
                enc.setBuffer(self.state, l.state + conv_state_bytes, 4);
                enc.setBuffer(self.y, 0, 5);
                bind(enc, at, &self.gdnP(n, write), 4, 6);
                if (self.gdn_sg) {
                    enc.dispatchThreads(size2(32, cfg.lin_v_heads * cfg.lin_dv), size2(32, 1));
                } else {
                    enc.dispatchThreads(size2(cfg.lin_dv, cfg.lin_v_heads), size2(32, 1));
                }
            }
        }

        fn encodeFullAttn(self: *const Self, enc: metal.Encoder, l: *const Layer, at: At, n: u32) void {
            const k_off = l.kv;
            const v_off = l.kv + kvBytes(self.max_ctx);
            const pos = cpuPos(at);
            // Threads past the position return at once, so a bound is enough for the grid.
            const reach = switch (at) {
                .cpu => |p| p,
                .gpu => |d| d.bound,
            } + n;

            if (self.on(.attn_prep)) {
                enc.setPipeline(self.k_prep);
                enc.setBuffer(self.proj, 0, 0);
                enc.setBuffer(self.weights, l.qn, 1);
                enc.setBuffer(self.weights, l.kn, 2);
                enc.setBuffer(self.qrot, 0, 3);
                enc.setBuffer(self.kv, k_off, 4);
                enc.setBuffer(self.kv, v_off, 5);
                bind(enc, at, &self.prepP(pos), 1, 6);
                enc.dispatchGroups(size3(1, cfg.heads + 2 * cfg.kv_heads, n), size3(cfg.head_dim, 1, 1));
            }

            const sp = self.scoreP(pos);
            if (self.on(.attn_scores)) {
                enc.setPipeline(self.k_scores);
                enc.setBuffer(self.qrot, 0, 0);
                enc.setBuffer(self.kv, k_off, 1);
                enc.setBuffer(self.scores, 0, 2);
                bind(enc, at, &sp, 2, 3);
                enc.dispatchThreads(size3(reach, cfg.heads, n), size3(32, 1, 1));
            }
            if (self.on(.attn_softmax)) {
                enc.setPipeline(self.k_softmax);
                enc.setBuffer(self.scores, 0, 0);
                bind(enc, at, &sp, 2, 1);
                if (self.attn_sg & 1 != 0) {
                    enc.dispatchThreads(size3(32, cfg.heads, n), size3(32, 1, 1));
                } else {
                    enc.dispatchThreads(size2(cfg.heads, n), size2(cfg.heads, 1));
                }
            }
            if (self.on(.attn_out)) {
                enc.setPipeline(self.k_attn_out);
                enc.setBuffer(self.scores, 0, 0);
                enc.setBuffer(self.kv, v_off, 1);
                enc.setBuffer(self.proj, 0, 2);
                enc.setBuffer(self.tmp, 0, 3);
                bind(enc, at, &sp, 2, 4);
                if (self.attn_sg & 2 != 0) {
                    enc.dispatchThreads(size3(32, cfg.heads * cfg.head_dim, n), size3(32, 1, 1));
                } else {
                    enc.dispatchThreads(size3(cfg.head_dim, cfg.heads, n), size3(32, 1, 1));
                }
            }
        }

        /// LM head and greedy pick for the n tokens of a pass: `out[at + t]` is the successor of token t.
        fn pick(self: *const Self, enc: metal.Encoder, n: u32, out: metal.Buffer, at: u32) void {
            self.matmul(enc, .gemv_head, self.head, self.xn, self.logits, 0, n);
            if (self.on(.argmax)) {
                enc.setPipeline(self.k_argmax);
                enc.setBuffer(self.logits, 0, 0);
                enc.setBuffer(out, 0, 1);
                enc.setParams(&ArgmaxP{ .n = cfg.vocab, .at = at }, 2);
                enc.dispatchGroups(size2(1, n), size1(1024));
            }
        }

        /// LM head and greedy pick for a single-token pass: writes `tok[step + 1]`.
        pub fn encodeHead(self: *const Self, enc: metal.Encoder, step: u32) void {
            self.pick(enc, 1, self.tok, step + 1);
        }

        /// Full forward pass for the token at `step`; with `head` it also picks the next token.
        pub fn encodeToken(self: *const Self, enc: metal.Encoder, step: u32, head: bool) void {
            self.encodeBlock(enc, step, 1);
            if (head) self.encodeHead(enc, step);
        }

        /// Forward pass without the head for the n tokens `tok[step .. step + n]` (prefill).
        pub fn encodeBlock(self: *const Self, enc: metal.Encoder, step: u32, n: u32) void {
            std.debug.assert(n <= self.max_batch);
            self.encodeEmbed(enc, step, n);
            for (0..cfg.layers) |i| self.encodeLayer(enc, i, step, n, .advance);
        }

        /// Speculative pass over `tok[step .. step + n]`, a confirmed token followed by drafts:
        /// `pred[t]` receives the greedy successor of token t and the recurrent states stay as they
        /// were. The KV cache needs no rollback, later passes overwrite the rejected positions. The
        /// caller accepts a prefix of the pass and encodes `encodeCommit` for it before any other
        /// pass. Needs the lane-split state kernel (`gdn_sg`).
        pub fn encodeVerify(self: *const Self, enc: metal.Encoder, at: At, n: u32) void {
            std.debug.assert(n <= self.max_batch and self.gdn_sg);
            self.embedAt(enc, at, n);
            for (0..cfg.layers) |i| self.layerAt(enc, i, at, n, .peek);
            self.pick(enc, n, self.pred, 0);
        }

        /// Advances the recurrent states over the first n tokens of the last `encodeVerify` pass, with
        /// the same kernels on the same inputs, so the states are those of a pass of exactly n tokens.
        pub fn encodeCommit(self: *const Self, enc: metal.Encoder, n: u32) void {
            for (&self.layers) |*l| {
                if (l.linear) self.encodeGdn(enc, l, n, true, null);
            }
        }

        /// `encodeCommit` for the GPU-held batch (at most `n`).
        pub fn encodeCommitAt(self: *const Self, enc: metal.Encoder, d: Dyn, n: u32) void {
            for (&self.layers) |*l| {
                if (l.linear) self.encodeGdn(enc, l, n, true, d);
            }
        }

        /// Forward pass over the n tokens at the GPU-held position, with the head over all of
        /// them: `pred[t]` is the pick after token t. The recurrent states advance over the
        /// GPU-held batch only, so the tokens past it may be anything (draft model of a chain).
        pub fn encodeFeedAt(self: *const Self, enc: metal.Encoder, d: Dyn, n: u32) void {
            std.debug.assert(n <= self.max_batch and self.gdn_sg);
            self.embedAt(enc, .{ .gpu = d }, n);
            for (0..cfg.layers) |i| self.layerAt(enc, i, .{ .gpu = d }, n, .advance);
            self.pick(enc, n, self.pred, 0);
        }
    };
}
