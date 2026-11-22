//! Minimal Metal bindings through the Objective-C runtime.
//!
//! Every selector is registered once in `init`; the encode path only performs direct
//! `objc_msgSend` calls with pre-resolved selectors and allocates nothing on the Zig side.

const std = @import("std");

pub const Id = ?*anyopaque;
const Sel = ?*anyopaque;

extern "c" fn objc_msgSend() void;
extern "c" fn sel_registerName(name: [*:0]const u8) Sel;
extern "c" fn objc_getClass(name: [*:0]const u8) Id;
extern "c" fn objc_autoreleasePoolPush() ?*anyopaque;
extern "c" fn objc_autoreleasePoolPop(pool: ?*anyopaque) void;
extern "c" fn MTLCreateSystemDefaultDevice() Id;

pub const Size = extern struct { w: usize, h: usize, d: usize };

var sel: struct {
    stringWithUTF8String: Sel,
    UTF8String: Sel,
    localizedDescription: Sel,
    newCommandQueue: Sel,
    newLibraryWithSource: Sel,
    newFunctionWithName: Sel,
    newComputePipelineStateWithFunction: Sel,
    newBufferWithLength: Sel,
    newBufferWithBytesNoCopy: Sel,
    contents: Sel,
    commandBuffer: Sel,
    computeCommandEncoder: Sel,
    setComputePipelineState: Sel,
    setBuffer: Sel,
    setBytes: Sel,
    dispatchThreads: Sel,
    dispatchThreadgroups: Sel,
    endEncoding: Sel,
    commit: Sel,
    waitUntilCompleted: Sel,
    status: Sel,
    @"error": Sel,
    GPUStartTime: Sel,
    GPUEndTime: Sel,
    retain: Sel,
    release: Sel,
    name: Sel,
    maxBufferLength: Sel,
} = undefined;

fn Fn(comptime T: type) type {
    return *const T;
}

inline fn cast(comptime T: type) Fn(T) {
    return @ptrCast(&objc_msgSend);
}

pub fn init() void {
    inline for (.{
        .{ "stringWithUTF8String", "stringWithUTF8String:" },
        .{ "UTF8String", "UTF8String" },
        .{ "localizedDescription", "localizedDescription" },
        .{ "newCommandQueue", "newCommandQueue" },
        .{ "newLibraryWithSource", "newLibraryWithSource:options:error:" },
        .{ "newFunctionWithName", "newFunctionWithName:" },
        .{ "newComputePipelineStateWithFunction", "newComputePipelineStateWithFunction:error:" },
        .{ "newBufferWithLength", "newBufferWithLength:options:" },
        .{ "newBufferWithBytesNoCopy", "newBufferWithBytesNoCopy:length:options:deallocator:" },
        .{ "contents", "contents" },
        .{ "commandBuffer", "commandBuffer" },
        .{ "computeCommandEncoder", "computeCommandEncoder" },
        .{ "setComputePipelineState", "setComputePipelineState:" },
        .{ "setBuffer", "setBuffer:offset:atIndex:" },
        .{ "setBytes", "setBytes:length:atIndex:" },
        .{ "dispatchThreads", "dispatchThreads:threadsPerThreadgroup:" },
        .{ "dispatchThreadgroups", "dispatchThreadgroups:threadsPerThreadgroup:" },
        .{ "endEncoding", "endEncoding" },
        .{ "commit", "commit" },
        .{ "waitUntilCompleted", "waitUntilCompleted" },
        .{ "status", "status" },
        .{ "error", "error" },
        .{ "GPUStartTime", "GPUStartTime" },
        .{ "GPUEndTime", "GPUEndTime" },
        .{ "retain", "retain" },
        .{ "release", "release" },
        .{ "name", "name" },
        .{ "maxBufferLength", "maxBufferLength" },
    }) |pair| {
        @field(sel, pair[0]) = sel_registerName(pair[1]);
    }
}

fn nsString(s: [*:0]const u8) Id {
    return cast(fn (Id, Sel, [*:0]const u8) callconv(.c) Id)(objc_getClass("NSString"), sel.stringWithUTF8String, s);
}

fn cString(ns: Id) [*:0]const u8 {
    return cast(fn (Id, Sel) callconv(.c) [*:0]const u8)(ns, sel.UTF8String);
}

fn describe(err: Id) [*:0]const u8 {
    if (err == null) return "unknown error";
    return cString(cast(fn (Id, Sel) callconv(.c) Id)(err, sel.localizedDescription));
}

pub const Pool = struct {
    handle: ?*anyopaque,
    pub fn push() Pool {
        return .{ .handle = objc_autoreleasePoolPush() };
    }
    pub fn pop(self: Pool) void {
        objc_autoreleasePoolPop(self.handle);
    }
};

pub fn retain(obj: Id) void {
    _ = cast(fn (Id, Sel) callconv(.c) Id)(obj, sel.retain);
}

pub fn release(obj: Id) void {
    cast(fn (Id, Sel) callconv(.c) void)(obj, sel.release);
}

pub const Buffer = struct {
    id: Id,

    pub fn contents(self: Buffer) [*]u8 {
        return @ptrCast(cast(fn (Id, Sel) callconv(.c) ?*anyopaque)(self.id, sel.contents).?);
    }

    pub fn as(self: Buffer, comptime T: type) [*]T {
        return @ptrCast(@alignCast(self.contents()));
    }
};

pub const Pipeline = struct { id: Id };

pub const Device = struct {
    id: Id,

    pub fn default() !Device {
        const dev = MTLCreateSystemDefaultDevice();
        if (dev == null) return error.NoMetalDevice;
        return .{ .id = dev };
    }

    pub fn name(self: Device) [*:0]const u8 {
        return cString(cast(fn (Id, Sel) callconv(.c) Id)(self.id, sel.name));
    }

    pub fn maxBufferLength(self: Device) usize {
        return cast(fn (Id, Sel) callconv(.c) usize)(self.id, sel.maxBufferLength);
    }

    pub fn queue(self: Device) Queue {
        return .{ .id = cast(fn (Id, Sel) callconv(.c) Id)(self.id, sel.newCommandQueue) };
    }

    pub fn library(self: Device, source: [*:0]const u8) !Library {
        var err: Id = null;
        const lib = cast(fn (Id, Sel, Id, Id, *Id) callconv(.c) Id)(self.id, sel.newLibraryWithSource, nsString(source), null, &err);
        if (lib == null) {
            std.debug.print("metal: shader compilation failed: {s}\n", .{describe(err)});
            return error.ShaderCompilation;
        }
        return .{ .id = lib, .device = self };
    }

    /// Zero-filled buffer in shared storage.
    pub fn buffer(self: Device, len: usize) !Buffer {
        const buf = cast(fn (Id, Sel, usize, usize) callconv(.c) Id)(self.id, sel.newBufferWithLength, len, 0);
        if (buf == null) return error.BufferAllocation;
        return .{ .id = buf };
    }

    /// Wraps page-aligned memory (for example a file mapping) without copying it.
    pub fn bufferNoCopy(self: Device, ptr: [*]u8, len: usize) !Buffer {
        const buf = cast(fn (Id, Sel, [*]u8, usize, usize, Id) callconv(.c) Id)(self.id, sel.newBufferWithBytesNoCopy, ptr, len, 0, null);
        if (buf == null) return error.BufferAllocation;
        return .{ .id = buf };
    }
};

pub const Library = struct {
    id: Id,
    device: Device,

    pub fn pipeline(self: Library, func: [*:0]const u8) !Pipeline {
        const f = cast(fn (Id, Sel, Id) callconv(.c) Id)(self.id, sel.newFunctionWithName, nsString(func));
        if (f == null) {
            std.debug.print("metal: no kernel named {s}\n", .{func});
            return error.MissingKernel;
        }
        var err: Id = null;
        const pso = cast(fn (Id, Sel, Id, *Id) callconv(.c) Id)(self.device.id, sel.newComputePipelineStateWithFunction, f, &err);
        if (pso == null) {
            std.debug.print("metal: pipeline {s} failed: {s}\n", .{ func, describe(err) });
            return error.PipelineCreation;
        }
        return .{ .id = pso };
    }
};

pub const Queue = struct {
    id: Id,

    /// The returned command buffer is autoreleased; retain it to keep it past the current pool.
    pub fn commandBuffer(self: Queue) CommandBuffer {
        return .{ .id = cast(fn (Id, Sel) callconv(.c) Id)(self.id, sel.commandBuffer) };
    }
};

pub const CommandBuffer = struct {
    id: Id,

    pub fn encoder(self: CommandBuffer) Encoder {
        return .{ .id = cast(fn (Id, Sel) callconv(.c) Id)(self.id, sel.computeCommandEncoder) };
    }

    pub fn commit(self: CommandBuffer) void {
        cast(fn (Id, Sel) callconv(.c) void)(self.id, sel.commit);
    }

    pub fn wait(self: CommandBuffer) !void {
        cast(fn (Id, Sel) callconv(.c) void)(self.id, sel.waitUntilCompleted);
        const status = cast(fn (Id, Sel) callconv(.c) usize)(self.id, sel.status);
        if (status != 4) { // MTLCommandBufferStatusCompleted
            const err = cast(fn (Id, Sel) callconv(.c) Id)(self.id, sel.@"error");
            std.debug.print("metal: command buffer failed (status {d}): {s}\n", .{ status, describe(err) });
            return error.CommandBufferFailed;
        }
    }

    /// Seconds the GPU spent executing this command buffer; valid after `wait`.
    pub fn gpuSeconds(self: CommandBuffer) f64 {
        const start = cast(fn (Id, Sel) callconv(.c) f64)(self.id, sel.GPUStartTime);
        const end = cast(fn (Id, Sel) callconv(.c) f64)(self.id, sel.GPUEndTime);
        return end - start;
    }
};

pub const Encoder = struct {
    id: Id,

    pub inline fn setPipeline(self: Encoder, p: Pipeline) void {
        cast(fn (Id, Sel, Id) callconv(.c) void)(self.id, sel.setComputePipelineState, p.id);
    }

    pub inline fn setBuffer(self: Encoder, b: Buffer, offset: usize, index: usize) void {
        cast(fn (Id, Sel, Id, usize, usize) callconv(.c) void)(self.id, sel.setBuffer, b.id, offset, index);
    }

    /// Copies a small parameter struct into the command stream.
    pub inline fn setParams(self: Encoder, params: anytype, index: usize) void {
        const T = @typeInfo(@TypeOf(params)).pointer.child;
        cast(fn (Id, Sel, *const anyopaque, usize, usize) callconv(.c) void)(self.id, sel.setBytes, params, @sizeOf(T), index);
    }

    pub inline fn dispatchThreads(self: Encoder, grid: Size, group: Size) void {
        cast(fn (Id, Sel, Size, Size) callconv(.c) void)(self.id, sel.dispatchThreads, grid, group);
    }

    pub inline fn dispatchGroups(self: Encoder, groups: Size, group: Size) void {
        cast(fn (Id, Sel, Size, Size) callconv(.c) void)(self.id, sel.dispatchThreadgroups, groups, group);
    }

    pub inline fn end(self: Encoder) void {
        cast(fn (Id, Sel) callconv(.c) void)(self.id, sel.endEncoding);
    }
};
