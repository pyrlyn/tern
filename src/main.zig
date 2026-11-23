//! tern command line.
//!
//!   tern gen   <model.tern> <ids-file> <n-gen> [options]   greedy generation, one JSON line
//!   tern prof  <model.tern> <position> <batch> [options]   GPU time of one token per kernel class
//!   tern prof  <model.tern> <ids-file> 2 --draft <draft.tern>   pieces of a chained pass of two
//!   tern bench <model.tern> <matrix> [iters] [batch] [options]   one matrix product in isolation
//!
//! Tokenisation stays outside for now: the benchmark harness renders the prompt once and feeds
//! the same ids to every engine.

const std = @import("std");
const metal = @import("metal.zig");
const model = @import("model.zig");
const engine_mod = @import("engine.zig");
const draft = @import("draft.zig");
const Engine = engine_mod.Engine;
const DraftEngine = engine_mod.DraftEngine;
const Chain = @import("chain.zig").Chain;
const cfg = model.cfg;

extern "c" fn clock_gettime_nsec_np(clock: c_int) u64;
extern "c" fn printf(format: [*:0]const u8, ...) c_int;
extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fread(ptr: [*]u8, size: usize, n: usize, file: *anyopaque) usize;
extern "c" fn fwrite(ptr: [*]const u8, size: usize, n: usize, file: *anyopaque) usize;
extern "c" fn fclose(file: *anyopaque) c_int;

fn now() f64 {
    return @as(f64, @floatFromInt(clock_gettime_nsec_np(8))) * 1e-9; // CLOCK_UPTIME_RAW
}

const max_lookahead = 8;
/// Tokens per command buffer when the prompt is fed one token at a time.
const serial_chunk = 8;

const Options = struct {
    engine: engine_mod.Config = .{},
    lookahead: u32 = 2,
    /// Tokens per batched prefill pass; below `gemm_tokens` the prompt is fed token by token.
    prefill_batch: u32 = 32,
    /// Speculative decoding with prompt-lookup drafts: most tokens per verification pass
    /// (0 = off) and the shortest suffix match that starts one. It needs the lane-split state
    /// kernel and is off without it.
    spec: u32 = 32,
    spec_min: u32 = 4,
    /// Draft model (`tools/convert.py` output of Qwen3.5-0.8B 4-bit): where prompt lookup has no
    /// match, it proposes one token for a pass of two.
    draft_path: ?[*:0]const u8 = null,
    /// With a draft model: time its passes against plain tokens and run the faster of the two
    /// (`draft.Gate`). Off, every position without a prompt match is a draft-model pass.
    gate: bool = true,
    dump_dir: ?[]const u8 = null,
    eos: [8]u32 = @splat(std.math.maxInt(u32)),

    /// Parses `--flag value` pairs starting at args[first].
    fn parse(args: []const [:0]const u8, first: usize) !Options {
        var opt = Options{};
        var spec_asked = false;
        var a = first;
        while (a + 1 < args.len) : (a += 2) {
            const flag = args[a];
            const value = args[a + 1];
            if (std.mem.eql(u8, flag, "--eos")) {
                _ = parseIds(value, &opt.eos);
            } else if (std.mem.eql(u8, flag, "--lookahead")) {
                opt.lookahead = @min(max_lookahead, @max(1, try std.fmt.parseInt(u32, value, 10)));
            } else if (std.mem.eql(u8, flag, "--prefill-batch")) {
                opt.prefill_batch = try std.fmt.parseInt(u32, value, 10);
            } else if (std.mem.eql(u8, flag, "--spec")) {
                opt.spec = try std.fmt.parseInt(u32, value, 10);
                spec_asked = opt.spec != 0;
            } else if (std.mem.eql(u8, flag, "--spec-min")) {
                opt.spec_min = @max(1, try std.fmt.parseInt(u32, value, 10));
            } else if (std.mem.eql(u8, flag, "--gemv")) {
                opt.engine.gemv = value.ptr; // kernel name; pair it with --gemv-rows
            } else if (std.mem.eql(u8, flag, "--gemv-rows")) {
                opt.engine.gemv_rows = try std.fmt.parseInt(u32, value, 10);
            } else if (std.mem.eql(u8, flag, "--gemm-rows")) {
                opt.engine.gemm_rows = try std.fmt.parseInt(u32, value, 10);
            } else if (std.mem.eql(u8, flag, "--gemm-tokens")) {
                opt.engine.gemm_tokens = try std.fmt.parseInt(u32, value, 10);
            } else if (std.mem.eql(u8, flag, "--gemm-tile")) {
                opt.engine.gemm_tile = try std.fmt.parseInt(u32, value, 10);
            } else if (std.mem.eql(u8, flag, "--gemm-tile8")) {
                opt.engine.gemm_tile8 = try std.fmt.parseInt(u32, value, 10);
            } else if (std.mem.eql(u8, flag, "--gemm-split")) {
                opt.engine.gemm_split = try std.fmt.parseInt(u32, value, 10);
            } else if (std.mem.eql(u8, flag, "--gdn-sg")) {
                opt.engine.gdn_sg = value[0] != '0';
            } else if (std.mem.eql(u8, flag, "--attn-sg")) {
                opt.engine.attn_sg = try std.fmt.parseInt(u32, value, 10);
            } else if (std.mem.eql(u8, flag, "--draft")) {
                opt.draft_path = value.ptr;
            } else if (std.mem.eql(u8, flag, "--gate")) {
                opt.gate = value[0] != '0';
            } else if (std.mem.eql(u8, flag, "--dump")) {
                opt.dump_dir = value;
            } else usage();
        }
        if (a != args.len) usage();
        const bt = opt.engine.gemm_tokens;
        if (!opt.engine.gdn_sg) {
            if (spec_asked) return error.SpeculationNeedsGdnSg;
            opt.spec = 0;
        }
        opt.engine.max_batch = @max(bt, (@max(opt.prefill_batch, opt.spec) + bt - 1) / bt * bt);
        if (opt.draft_path != null and opt.spec < 2) return error.DraftNeedsSpeculation;
        return opt;
    }

    /// Settings of the draft model's engine: same context, batched 4-bit kernel only.
    fn draftConfig(self: Options) engine_mod.Config {
        const ab = engine_mod.a4_batch;
        return .{
            .max_ctx = self.engine.max_ctx,
            .max_batch = (@max(self.engine.max_batch, self.spec + 1) + ab - 1) / ab * ab,
            .gemm_tokens = ab,
            .gemm_tile = 0,
            .gemm_tile8 = 0,
        };
    }
};

fn parseIds(text: []const u8, out: []u32) usize {
    var n: usize = 0;
    var i: usize = 0;
    while (i < text.len and n < out.len) {
        if (text[i] < '0' or text[i] > '9') {
            i += 1;
            continue;
        }
        var v: u32 = 0;
        while (i < text.len and text[i] >= '0' and text[i] <= '9') : (i += 1) v = v * 10 + (text[i] - '0');
        out[n] = v;
        n += 1;
    }
    return n;
}

fn usage() noreturn {
    std.debug.print(
        \\usage: tern gen   <model.tern> <ids-file> <n-gen> [options]
        \\       tern prof  <model.tern> <position> <batch> [options]
        \\       tern prof  <model.tern> <ids-file> 2 --draft <draft.tern>   (pieces of a pass of two)
        \\       tern bench <model.tern> <matrix> <iters> <batch> [options]
        \\options: --eos a,b  --lookahead k  --prefill-batch n  --dump dir
        \\         --spec n (speculative decoding: most tokens per pass, 0 = off)  --spec-min m
        \\         --draft <draft.tern> (draft model for passes of two where lookup finds nothing)
        \\         --gate 0|1 (with --draft: run its passes only while they beat plain tokens, default 1)
        \\         --gemv <kernel> --gemv-rows n  --gemm-rows r --gemm-tokens t --gemm-tile n --gemm-tile8 n --gemm-split 1|2|4
        \\         --gdn-sg 0|1  --attn-sg 0..3 (bit 0 softmax, bit 1 value sum)
        \\
    , .{});
    std.process.exit(2);
}

fn dump(dir: []const u8, name: []const u8, bytes: []const u8) !void {
    var path_buf: [512]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}\x00", .{ dir, name });
    const f = fopen(@ptrCast(path.ptr), "wb") orelse return error.DumpOpenFailed;
    defer _ = fclose(f);
    if (fwrite(bytes.ptr, 1, bytes.len, f) != bytes.len) return error.DumpWriteFailed;
}

/// Feeds the draft model `tok[fed.* .. upto]` in passes of at most its `max_batch` tokens and
/// advances `fed`.
fn feedDraft(d: *const DraftEngine, enc: metal.Encoder, fed: *u32, upto: u32) void {
    while (fed.* < upto) {
        const n = @min(d.max_batch, upto - fed.*);
        d.encodeBlock(enc, fed.*, n);
        fed.* += n;
    }
}

/// Prefill: every prompt token but the last, no head. Batches first (any size with the tiled
/// kernel, multiples of `gemm_tokens` with the lane kernel), then the tail token by token. The
/// draft model, if any, is fed in the same command buffers. Returns the GPU seconds.
fn prefill(engine: *const Engine, drafter: ?*const DraftEngine, batch: u32, n_prompt: u32, step: *u32, fed: *u32) !f64 {
    const bt = engine.gemm_tokens;
    var gpu_s: f64 = 0;
    while (step.* + 1 < n_prompt) {
        const pool = metal.Pool.push();
        defer pool.pop();
        const cb = engine.queue.commandBuffer();
        const enc = cb.encoder();
        const left = n_prompt - 1 - step.*;
        const tile = engine.gemm_tile;
        if (tile != 0 and @min(batch, left) >= tile) {
            const n = @min(batch, left);
            engine.encodeBlock(enc, step.*, n);
            step.* += n;
        } else if (batch >= bt and left >= bt) {
            const n = @min(engine.max_batch, left) / bt * bt;
            engine.encodeBlock(enc, step.*, n);
            step.* += n;
        } else {
            const end = step.* + @min(serial_chunk, left);
            while (step.* < end) : (step.* += 1) engine.encodeToken(enc, step.*, false);
        }
        if (drafter) |d| feedDraft(d, enc, fed, step.*);
        enc.end();
        cb.commit();
        try cb.wait();
        gpu_s += cb.gpuSeconds();
    }
    return gpu_s;
}

/// Counts the `got` tokens after `tok[done.*]` as output and advances `done` over them, up to
/// and including an end-of-sequence token; returns whether one came.
fn take(tok: [*]const u32, eos: []const u32, done: *u32, n_out: *u32, got: u32) bool {
    for (0..got) |_| {
        done.* += 1;
        for (eos) |e| {
            if (tok[done.*] == e) return true;
        }
        n_out.* += 1;
    }
    return false;
}

/// Chained passes to keep in flight after `tok[done]`: one while the context could reach a
/// lookup match with the up to two tokens of the pass in flight, so that the match shows before
/// another draft pass is queued (a long lookup pass is worth far more than a pass of two);
/// `lookahead` otherwise.
fn chainDepth(policy: *const draft.Policy, tok: [*]const u32, done: u32, last_step: u32, lookahead: u32) u32 {
    const near = policy.size(last_step + 1 - done) != 0 and
        draft.find(tok[0 .. done + 1], @max(1, policy.min_match -| 2)) != null;
    return if (near) 1 else lookahead;
}

/// Runs one piece of the graph in its own command buffer and waits for it (debug dumps only).
fn runPiece(engine: *const Engine, comptime piece: anytype, args: anytype) !void {
    const cb = engine.queue.commandBuffer();
    const enc = cb.encoder();
    @call(.auto, piece, .{ engine, enc } ++ args);
    enc.end();
    cb.commit();
    try cb.wait();
}

/// tern bench <model.tern> <matrix> <iters> <batch> [options]
/// Times one matrix product on one matrix of the model, e.g. `tern bench m.tern L0.mlp 30 1`.
/// Best of 5 command buffers, GPU time per token.
fn benchMatmul(args: []const [:0]const u8) !void {
    const iters = try std.fmt.parseInt(u32, args[4], 10);
    const n = try std.fmt.parseInt(u32, args[5], 10);
    var opt = try Options.parse(args, 6);
    opt.engine.max_ctx = 64;
    opt.engine.max_batch = @max(n, opt.engine.gemm_tokens);
    const engine = try Engine.init(args[2].ptr, opt.engine);
    const m = try Engine.mat(engine.file, "{s}", .{args[3]});

    // Deterministic pseudo-random input in [-1, 1).
    const cols = m.groups * cfg.group;
    const input = try engine.device.buffer(@as(usize, n) * cols * 2);
    const output = try engine.device.buffer(@as(usize, n) * m.rows * 2);
    const in = input.as(f16);
    var seed: u32 = 12345;
    for (0..@as(usize, n) * cols) |i| {
        seed = seed *% 1664525 +% 1013904223;
        in[i] = @floatCast(@as(f32, @floatFromInt(@as(i32, @bitCast(seed)) >> 8)) / 8388608.0);
    }

    var best: f64 = std.math.inf(f64);
    for (0..5) |_| {
        const pool = metal.Pool.push();
        defer pool.pop();
        const cb = engine.queue.commandBuffer();
        const enc = cb.encoder();
        for (0..iters) |_| engine.matmul(enc, .gemv_in, m, input, output, 0, n);
        enc.end();
        cb.commit();
        try cb.wait();
        best = @min(best, cb.gpuSeconds() / @as(f64, @floatFromInt(iters * n)));
    }
    const out = output.as(f16);
    var checksum: f64 = 0;
    for (0..@as(usize, n) * m.rows) |i| checksum += @as(f64, out[i]) * @as(f64, @floatFromInt(i % 7 + 1));
    const weights = @as(f64, @floatFromInt(m.rows)) * @as(f64, @floatFromInt(cols));
    _ = printf("%s batch=%u rows=%u cols=%u  %.4f ms/token  %.1f Gweights/s  checksum %.4f\n", args[3].ptr, n, m.rows, cols, best * 1e3, weights / best / 1e9, checksum);
}

/// tern prof <model.tern> <position> <batch> [options]
/// GPU time per token of each kernel class at the given context position: each class runs alone
/// (best of 5 command buffers), then the matrix products together, the rest together, and all
/// together. Kernel times do not depend on the data. Takes the main model or the draft model.
/// With `--draft` it profiles a chained pass of two instead (`profilePass`).
fn profile(args: []const [:0]const u8) !void {
    const n = try std.fmt.parseInt(u32, args[4], 10);
    var opt = try Options.parse(args, 5);
    if (opt.draft_path == null) opt.engine.max_batch = @max(n, opt.engine.gemm_tokens);
    const engine = Engine.init(args[2].ptr, opt.engine) catch |err| {
        if (err != error.ModelShapeMismatch) return err;
        var dc = opt.draftConfig();
        dc.max_batch = (@max(n, 1) + engine_mod.a4_batch - 1) / engine_mod.a4_batch * engine_mod.a4_batch;
        return profileWith(DraftEngine, try DraftEngine.init(args[2].ptr, dc), args, n);
    };
    if (opt.draft_path) |path| {
        if (n != 2) usage();
        return profilePass(engine, path, opt, args[3]);
    }
    return profileWith(Engine, engine, args, n);
}

/// `tern prof <model.tern> <ids-file> 2 --draft <draft.tern>`: GPU time of the pieces of a
/// chained pass of two after a real prompt, against a plain token. Both models prefill the
/// prompt and every piece starts from the same recurrent states. The GPU runs a piece faster
/// after a rest or after light pieces, so each one follows the same warm-up without a gap and
/// the order changes every round. Command buffers of 3; best and median of 15 rounds.
///
/// Pieces: `plain` one token with the head; `block` two tokens, states advanced, no head;
/// `verify` two tokens with the head, states left alone; `commit` the state replay for two
/// tokens; `draft` the draft model's part of a pass; `draft1` one draft-model token with the
/// head; `pass` the whole chained pass; `pass_nc` the same without the commit.
fn profilePass(engine_in: Engine, path: [*:0]const u8, opt: Options, ids_path: [:0]const u8) !void {
    var engine = engine_in;
    var d = try DraftEngine.initOn(engine.device, engine.lib, path, opt.draftConfig());
    d.tok = engine.tok;
    const ch = try Chain.init(&engine);

    var text: [1 << 16]u8 = undefined;
    const ids_file = fopen(ids_path.ptr, "rb") orelse return error.IdsOpenFailed;
    const text_len = fread(&text, 1, text.len, ids_file);
    _ = fclose(ids_file);
    const tok = engine.tokens();
    const reps = 3;
    const n_prompt: u32 = @intCast(parseIds(text[0..text_len], tok[0 .. opt.engine.max_ctx - 2 * reps - 2]));
    if (n_prompt == 0) return error.EmptyPrompt;
    var pos: u32 = 0;
    var fed: u32 = 0;
    _ = try prefill(&engine, &d, opt.prefill_batch, n_prompt, &pos, &fed);

    // Saved recurrent states after the prompt; the KV cache past it is overwritten anyway.
    const main_saved = try engine.device.buffer(Engine.state_len);
    const draft_saved = try engine.device.buffer(DraftEngine.state_len);
    @memcpy(main_saved.contents()[0..Engine.state_len], engine.state.contents()[0..Engine.state_len]);
    @memcpy(draft_saved.contents()[0..DraftEngine.state_len], d.state.contents()[0..DraftEngine.state_len]);

    const Piece = enum { plain, block, verify, commit, draft, draft1, pass, pass_nc };
    const pieces = std.enums.values(Piece);
    const rounds = 15;
    var times: [pieces.len][rounds]f64 = undefined;
    var order: [pieces.len]usize = undefined;
    for (&order, 0..) |*o, k| o.* = k;
    var seed: u32 = 12345;
    for (0..rounds) |round| {
        // A new order every round: a piece runs faster after light ones than after heavy ones.
        for (0..pieces.len - 1) |k| {
            seed = seed *% 1664525 +% 1013904223;
            std.mem.swap(usize, &order[k], &order[k + (seed >> 8) % (pieces.len - k)]);
        }
        for (order) |i| {
            const piece = pieces[i];
            @memcpy(engine.state.contents()[0..Engine.state_len], main_saved.contents()[0..Engine.state_len]);
            @memcpy(d.state.contents()[0..DraftEngine.state_len], draft_saved.contents()[0..DraftEngine.state_len]);
            ch.start(&engine, &d, pos);
            const pool = metal.Pool.push();
            defer pool.pop();
            // The same load ahead of every piece, with no gap: two passes that leave the
            // recurrent states alone, in a command buffer of their own.
            engine.mask = std.math.maxInt(u32);
            const warm = engine.queue.commandBuffer();
            const warm_enc = warm.encoder();
            for (0..2) |_| engine.encodeVerify(warm_enc, .{ .cpu = pos }, 2);
            warm_enc.end();
            warm.commit();
            const cb = engine.queue.commandBuffer();
            const enc = cb.encoder();
            // The draft-only pass runs the chain with the main model's kernels masked off.
            engine.mask = if (piece == .draft) 0 else std.math.maxInt(u32);
            for (0..reps) |r| {
                const at: u32 = @intCast(r);
                switch (piece) {
                    .plain => engine.encodeToken(enc, pos + at, true),
                    .block => engine.encodeBlock(enc, pos + 2 * at, 2),
                    .verify => engine.encodeVerify(enc, .{ .cpu = pos + 2 * at }, 2),
                    .commit => engine.encodeCommit(enc, 2),
                    .draft1 => d.encodeToken(enc, pos + at, true),
                    .draft, .pass => ch.encodePass(enc, &engine, &d, at + 1, pos + 2 * at),
                    // Sequence number 0 leaves the commit out.
                    .pass_nc => ch.encodePass(enc, &engine, &d, 0, pos + 2 * at),
                }
            }
            enc.end();
            cb.commit();
            try cb.wait();
            times[i][round] = cb.gpuSeconds() / reps;
        }
    }
    _ = printf("prompt %u tokens, pass at %u; GPU ms, best and median of %u rounds, in plain tokens\n", n_prompt, pos, @as(u32, rounds));
    for (&times) |*t| std.mem.sort(f64, t, {}, std.sort.asc(f64));
    for (pieces, times) |piece, t| {
        const best = t[0];
        const med = t[rounds / 2];
        _ = printf("%-7s %8.3f %8.3f  %5.2f %5.2f\n", @tagName(piece).ptr, best * 1e3, med * 1e3, best / times[0][0], med / times[0][rounds / 2]);
    }
}

fn profileWith(comptime E: type, engine_in: E, args: []const [:0]const u8, n: u32) !void {
    var engine = engine_in;
    const pos = try std.fmt.parseInt(u32, args[3], 10);
    const reps: u32 = if (n == 1) 4 else 1;
    const per_token = 1.0 / @as(f64, @floatFromInt(reps * n));
    const parts = std.enums.values(engine_mod.Part);
    var sum: f64 = 0;
    var gemms: u32 = 0;
    for ([_]engine_mod.Part{ .gemv_in, .gemv_out, .gemv_mlp, .gemv_down, .gemv_head }) |p| gemms |= @as(u32, 1) << @backingInt(p);
    for (0..parts.len + 3) |i| {
        engine.mask = if (i < parts.len) @as(u32, 1) << @backingInt(parts[i]) else switch (i - parts.len) {
            0 => gemms,
            1 => ~gemms,
            else => std.math.maxInt(u32),
        };
        var best: f64 = std.math.inf(f64);
        var encode_s: f64 = 0;
        for (0..5) |_| {
            const pool = metal.Pool.push();
            defer pool.pop();
            const cb = engine.queue.commandBuffer();
            const enc = cb.encoder();
            const t = now();
            for (0..reps) |_| {
                engine.encodeBlock(enc, pos, n);
                if (n == 1) engine.encodeHead(enc, pos);
            }
            encode_s = now() - t;
            enc.end();
            cb.commit();
            try cb.wait();
            best = @min(best, cb.gpuSeconds() * per_token);
        }
        if (i < parts.len) sum += best;
        const name: [*:0]const u8 = if (i < parts.len) @tagName(parts[i]).ptr else switch (i - parts.len) {
            0 => "products",
            1 => "the rest",
            else => "token",
        };
        _ = printf("%-13s %8.3f ms gpu  %6.3f ms encode\n", name, best * 1e3, encode_s * per_token * 1e3);
    }
    _ = printf("%-13s %8.3f ms gpu\n", "sum of parts", sum * 1e3);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len >= 5 and std.mem.eql(u8, args[1], "prof")) return profile(args);
    if (args.len >= 6 and std.mem.eql(u8, args[1], "bench")) return benchMatmul(args);
    if (args.len < 5 or !std.mem.eql(u8, args[1], "gen")) usage();
    const opt = try Options.parse(args, 5);
    const max_ctx = opt.engine.max_ctx;

    const t_start = now();
    var engine = try Engine.init(args[2].ptr, opt.engine);
    var drafter: ?DraftEngine = null;
    if (opt.draft_path) |path| {
        drafter = try DraftEngine.initOn(engine.device, engine.lib, path, opt.draftConfig());
        drafter.?.tok = engine.tok; // the draft reads the same token ids, at the same positions
    }
    const chain: ?Chain = if (drafter != null) try Chain.init(&engine) else null;
    const t_loaded = now();

    const tok = engine.tokens();
    var text: [1 << 16]u8 = undefined;
    const ids_file = fopen(args[3].ptr, "rb") orelse return error.IdsOpenFailed;
    const text_len = fread(&text, 1, text.len, ids_file);
    _ = fclose(ids_file);
    const n_prompt: u32 = @intCast(parseIds(text[0..text_len], tok[0 .. max_ctx - 1]));
    if (n_prompt == 0) return error.EmptyPrompt;
    const n_gen = @min(try std.fmt.parseInt(u32, args[4], 10), max_ctx - n_prompt);

    const t0 = now();
    var gpu_s: f64 = 0;

    var step: u32 = 0;
    // Prompt tokens the draft model has seen; it runs in the main model's command buffers.
    var fed: u32 = 0;
    gpu_s += try prefill(&engine, if (drafter) |*d| d else null, opt.prefill_batch, n_prompt, &step, &fed);
    const t_prefilled = now();

    if (opt.dump_dir) |dir| {
        // Last prompt token piece by piece, dumping the residual stream after every layer.
        const pool = metal.Pool.push();
        defer pool.pop();
        try runPiece(&engine, Engine.encodeEmbed, .{ step, 1 });
        try dump(dir, "embed.bin", engine.x.contents()[0 .. cfg.hidden * 2]);
        for (0..cfg.layers) |i| {
            try runPiece(&engine, Engine.encodeLayer, .{ i, step, 1, .advance });
            var name_buf: [32]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "layer_{d:0>2}.bin", .{i});
            try dump(dir, name, engine.x.contents()[0 .. cfg.hidden * 2]);
        }
        try runPiece(&engine, Engine.encodeHead, .{step});
        try dump(dir, "final_norm.bin", engine.xn.contents()[0 .. cfg.hidden * 2]);
        try dump(dir, "logits.bin", engine.logits.contents()[0 .. cfg.vocab * 2]);
        step += 1;
    }

    // Decode. Step s produces tok[s + 1]; up to `lookahead` command buffers are in flight because
    // the token id is passed on GPU-side.
    //
    // With speculation, a verification pass starts whenever the context ends in something seen
    // before and no step is in flight (`hold` stops new steps once a match shows up). The pass
    // leaves the recurrent states alone; `pending` counts its accepted tokens until the next
    // command buffer commits them. With a draft model, the steps without a match are chained
    // passes of two (chain.zig), also up to `lookahead` in flight: the GPU accepts each one.
    // The gate times those passes against plain tokens and picks the way for the next stretch.
    const last_step = n_prompt + n_gen - 2;
    var ring: [max_lookahead]metal.CommandBuffer = undefined;
    var submitted: u32 = step;
    var done: u32 = step;
    var t_first: f64 = 0;
    var t_last: f64 = 0;
    var n_out: u32 = if (opt.dump_dir != null) 1 else 0;
    // Tokens that arrived together with the first one; they are not part of the decode speed.
    var n_first: u32 = 1;
    if (n_out == 1) {
        t_first = now();
        t_last = t_first;
    }
    var policy = draft.Policy{ .min_match = opt.spec_min, .count = 0 };
    for ([_]u32{ 2, 4, 8, opt.spec }) |want| {
        const n = engine.batchFit(@min(want, opt.spec));
        if (n >= 2 and (policy.count == 0 or n > policy.sizes[policy.count - 1])) {
            policy.sizes[policy.count] = n;
            policy.count += 1;
        }
    }
    const spec = policy.count != 0;
    var by_draft = draft.Policy.Stat{};
    var gate = draft.Gate{};
    var pending: u32 = 0;
    var hold = false;
    var stopped = false;
    while (done <= last_step and !stopped) {
        if (spec and submitted == done) {
            hold = false;
            const room = last_step + 1 - done;
            const n = policy.size(room);
            if (if (n != 0) draft.find(tok[0 .. done + 1], policy.min_match) else null) |m| {
                draft.fill(tok, done, m, n - 1);
                const pool = metal.Pool.push();
                defer pool.pop();
                const cb = engine.queue.commandBuffer();
                const enc = cb.encoder();
                if (pending != 0) engine.encodeCommit(enc, pending);
                engine.encodeVerify(enc, .{ .cpu = done }, n);
                enc.end();
                const t_pass = now();
                cb.commit();
                try cb.wait();
                gpu_s += cb.gpuSeconds();
                // pred[t] follows tok[done + t]: a draft stands while it equals the prediction
                // before it, and the first prediction that is not confirmed is a new token too.
                const pred = engine.pred.as(u32);
                var got: u32 = 1;
                while (got < n and tok[done + got] == pred[got - 1]) got += 1;
                tok[done + got] = pred[got - 1];
                t_last = now();
                policy.passed(n, got, t_last - t_pass);
                pending = got;
                const first = n_out == 0;
                if (first) t_first = t_last;
                stopped = take(tok, &opt.eos, &done, &n_out, got);
                if (first) n_first = @max(1, n_out);
                submitted = done;
                continue;
            }
            if (drafter != null and room >= 2 and gate.way == .model) {
                // Chained passes until a lookup match, the end of the budget or of the sequence.
                // A pass is submitted only if it ends within the budget even when every pass
                // before it accepts two tokens.
                const d = &drafter.?;
                const ch = &chain.?;
                ch.start(&engine, d, done);
                var seq: u32 = 0;
                var seen: u32 = 0;
                var more = true;
                var depth = chainDepth(&policy, tok, done, last_step, opt.lookahead);
                var t_pass = now();
                while (true) {
                    while (more and seq - seen < depth and done + 2 * (seq - seen) + 2 <= last_step + 1) : (seq += 1) {
                        const pool = metal.Pool.push();
                        defer pool.pop();
                        const cb = engine.queue.commandBuffer();
                        metal.retain(cb.id);
                        const enc = cb.encoder();
                        if (seq == 0) {
                            if (pending != 0) engine.encodeCommit(enc, pending);
                            feedDraft(d, enc, &fed, done);
                        }
                        ch.encodePass(enc, &engine, d, seq, done + 2 * (seq - seen));
                        enc.end();
                        cb.commit();
                        ring[seq % max_lookahead] = cb;
                    }
                    if (seen == seq) break;
                    const cb = ring[seen % max_lookahead];
                    try cb.wait();
                    gpu_s += cb.gpuSeconds();
                    metal.release(cb.id);
                    const after = ch.after(seen);
                    seen += 1;
                    if (stopped) continue; // drain
                    const got = after - done;
                    t_last = now();
                    policy.modelPass(got);
                    by_draft.passes += 1;
                    by_draft.tokens += got;
                    by_draft.seconds += t_last - t_pass;
                    if (opt.gate) gate.sample(.model, got, t_last - t_pass);
                    t_pass = t_last;
                    pending = got;
                    fed = after + 1 - got;
                    const first = n_out == 0;
                    if (first) t_first = t_last;
                    stopped = take(tok, &opt.eos, &done, &n_out, got);
                    if (first) n_first = @max(1, n_out);
                    more = !stopped and gate.way == .model and (policy.size(last_step + 1 - done) == 0 or
                        draft.find(tok[0 .. done + 1], policy.min_match) == null);
                    depth = chainDepth(&policy, tok, done, last_step, opt.lookahead);
                }
                submitted = done;
                continue;
            }
        }
        while (!hold and submitted <= last_step and submitted - done < opt.lookahead) : (submitted += 1) {
            const pool = metal.Pool.push();
            defer pool.pop();
            const cb = engine.queue.commandBuffer();
            metal.retain(cb.id);
            const enc = cb.encoder();
            if (pending != 0) engine.encodeCommit(enc, pending);
            pending = 0;
            engine.encodeToken(enc, submitted, true);
            enc.end();
            cb.commit();
            ring[submitted % max_lookahead] = cb;
        }
        const cb = ring[done % max_lookahead];
        try cb.wait();
        gpu_s += cb.gpuSeconds();
        metal.release(cb.id);
        const t_before = t_last;
        t_last = now();
        if (n_out == 0) t_first = t_last;
        if (drafter != null and opt.gate and t_before != 0) gate.sample(.plain, 1, t_last - t_before);
        const produced = tok[done + 1];
        done += 1;
        for (opt.eos) |e| {
            if (produced == e) stopped = true;
        }
        if (!stopped) n_out += 1;
        if (spec) {
            policy.plain();
            // No new steps once a pass is due: a prompt match, or the gate back at the draft model.
            hold = done <= last_step and ((policy.size(last_step + 1 - done) != 0 and
                draft.find(tok[0 .. done + 1], policy.min_match) != null) or
                (drafter != null and gate.way == .model and last_step + 1 - done >= 2));
        }
    }
    // Drain command buffers submitted past an end-of-sequence token.
    while (done < submitted) : (done += 1) {
        const cb = ring[done % max_lookahead];
        try cb.wait();
        metal.release(cb.id);
    }

    const ttft = t_first - t0;
    const decode_tps = if (n_out > n_first) @as(f64, @floatFromInt(n_out - n_first)) / (t_last - t_first) else 0;
    _ = printf(
        "{\"engine\":\"tern\",\"device\":\"%s\",\"load_s\":%.3f,\"n_prompt\":%u,\"n_gen\":%u,\"ttft_s\":%.4f,\"prefill_s\":%.4f,\"prefill_tps\":%.2f,\"decode_tps\":%.2f,\"gpu_s\":%.3f,\"wall_s\":%.3f,\"lookahead\":%u,\"prefill_batch\":%u,\"spec\":[",
        engine.device.name(),
        t_loaded - t_start,
        n_prompt,
        n_out,
        ttft,
        t_prefilled - t0,
        @as(f64, @floatFromInt(n_prompt)) / ttft,
        decode_tps,
        gpu_s,
        t_last - t0,
        opt.lookahead,
        opt.prefill_batch,
    );
    for (0..policy.count) |i| {
        const st = policy.stats[i];
        _ = printf(if (i == 0) "{\"size\":%u,\"passes\":%u,\"tokens\":%u,\"s\":%.4f}" else ",{\"size\":%u,\"passes\":%u,\"tokens\":%u,\"s\":%.4f}", policy.sizes[i], st.passes, st.tokens, st.seconds);
    }
    _ = printf(
        "],\"draft\":{\"passes\":%u,\"tokens\":%u,\"s\":%.4f},\"gate\":{\"switches\":%u,\"model_ms\":%.2f,\"plain_ms\":%.2f},\"tokens\":[",
        by_draft.passes,
        by_draft.tokens,
        by_draft.seconds,
        gate.switches,
        gate.cost[0] * 1e3,
        gate.cost[1] * 1e3,
    );
    for (0..n_out) |i| _ = printf(if (i == 0) "%u" else ",%u", tok[n_prompt + i]);
    _ = printf("]}\n");
}
