//! Model shape and the flat weight file produced by tools/convert.py.

const std = @import("std");

/// Shape of Ternary Bonsai 27B (Qwen3.5 27B hybrid attention): 2-bit ternary weights, one scale
/// per 128 weights.
pub const cfg = struct {
    pub const hidden = 5120;
    pub const inter = 17408;
    pub const layers = 64;
    pub const vocab = 248320;
    pub const bits = 2;
    pub const group = 128;
    pub const eps: f32 = 1e-6;

    /// Every `full_interval`-th layer uses softmax attention, the rest are gated delta nets.
    pub const full_interval = 4;
    pub const n_full = layers / full_interval;
    pub const n_linear = layers - n_full;

    pub const heads = 24;
    pub const kv_heads = 4;
    pub const head_dim = 256;
    pub const rot_dims = 64;
    pub const rope_base: f32 = 1e7;

    pub const lin_k_heads = 16;
    pub const lin_v_heads = 48;
    pub const lin_dk = 128;
    pub const lin_dv = 128;
    pub const conv_dim = 2 * lin_k_heads * lin_dk + lin_v_heads * lin_dv;
    pub const conv_taps = 4;

    pub fn isLinear(layer: usize) bool {
        return (layer + 1) % full_interval != 0;
    }
};

/// Shape of Qwen3.5-0.8B in MLX 4-bit, the draft model: 4-bit affine weights with a scale and a
/// bias per 64 weights, the same architecture and tokenizer as the main model.
pub const cfg_draft = struct {
    pub const hidden = 1024;
    pub const inter = 3584;
    pub const layers = 24;
    pub const vocab = 248320;
    pub const bits = 4;
    pub const group = 64;
    pub const eps: f32 = 1e-6;

    pub const full_interval = 4;
    pub const n_full = layers / full_interval;
    pub const n_linear = layers - n_full;

    pub const heads = 8;
    pub const kv_heads = 2;
    pub const head_dim = 256;
    pub const rot_dims = 64;
    pub const rope_base: f32 = 1e7;

    pub const lin_k_heads = 16;
    pub const lin_v_heads = 16;
    pub const lin_dk = 128;
    pub const lin_dv = 128;
    pub const conv_dim = 2 * lin_k_heads * lin_dk + lin_v_heads * lin_dv;
    pub const conv_taps = 4;

    pub fn isLinear(layer: usize) bool {
        return (layer + 1) % full_interval != 0;
    }
};

const Entry = extern struct {
    name: [48]u8,
    offset: u64,
    nbytes: u64,
    rows: u32,
    cols: u32,
    dtype: u32,
    _pad: u32,
};

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn lseek(fd: c_int, offset: i64, whence: c_int) i64;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64) ?*anyopaque;

pub const Tensor = struct { offset: usize, rows: u32, cols: u32 };

pub const File = struct {
    base: [*]align(16384) u8,
    len: usize,
    entries: []const Entry,

    pub fn open_(path: [*:0]const u8) !File {
        const fd = open(path, 0); // O_RDONLY
        if (fd < 0) return error.OpenFailed;
        defer _ = close(fd);
        const size = lseek(fd, 0, 2); // SEEK_END
        if (size < 16) return error.BadFile;
        const len: usize = @intCast(size);
        // PROT_READ, MAP_PRIVATE
        const ptr = mmap(null, len, 1, 2, fd, 0) orelse return error.MapFailed;
        if (@intFromPtr(ptr) == std.math.maxInt(usize)) return error.MapFailed;
        const base: [*]align(16384) u8 = @ptrCast(@alignCast(ptr));
        if (!std.mem.eql(u8, base[0..8], "TERN0001")) return error.BadMagic;
        const count = std.mem.readInt(u32, base[8..12], .little);
        const table: [*]const Entry = @ptrCast(@alignCast(base + 16));
        return .{ .base = base, .len = len, .entries = table[0..count] };
    }

    pub fn find(self: File, name: []const u8) !Tensor {
        for (self.entries) |*e| {
            const stored = std.mem.sliceTo(&e.name, 0);
            if (std.mem.eql(u8, stored, name)) {
                return .{ .offset = @intCast(e.offset), .rows = e.rows, .cols = e.cols };
            }
        }
        std.debug.print("model: tensor {s} not found\n", .{name});
        return error.MissingTensor;
    }

    pub fn findFmt(self: File, comptime fmt: []const u8, args: anytype) !Tensor {
        var buf: [48]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, fmt, args);
        return self.find(name);
    }
};
