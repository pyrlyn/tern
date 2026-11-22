const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });

    const mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mod.linkSystemLibrary("objc", .{});
    mod.linkFramework("Foundation", .{});
    mod.linkFramework("Metal", .{});

    const exe = b.addExecutable(.{ .name = "tern", .root_module = mod });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    run.addPassthruArgs();
    b.step("run", "Run tern").dependOn(&run.step);

    // Unit tests of the code that needs neither a model nor a GPU.
    const tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("src/draft.zig"),
        .target = target,
        .optimize = optimize,
    }) });
    b.step("test", "Run unit tests").dependOn(&b.addRunArtifact(tests).step);
}
