# Backend abstraction

Design only: the compile-time backend interface, how a backend is selected, the memory model, a Zig signature sketch, and a reuse / port / rewrite analysis of the current Metal/Zig tree. Platform goals, performance baselines and the `>= 2x` bar live in a separate plan PR (`docs/cross-platform-plan` branch) and are out of scope here.

This expands [plan/T6](../plan/T6/task.md). Nothing here is implemented.

## Goal of the split

Today `src/engine.zig` talks to Metal directly through `src/metal.zig` (`objc_msgSend`, Foundation, Metal frameworks). [T6](../plan/T6/task.md) already requires a backend interface resolved at compile time, with no indirect calls on the hot path, Metal as the first implementation, and a second backend where hardware exists. After the split, `EngineOf` encodes the same graph (`encodeBlock`, `encodeVerify`, `encodeCommit`, `encodeHead`) against that interface. A backend is a compile-time package, not a vtable.

## Common interface

The interface is the surface already implied by `src/metal.zig` and the call sites in `src/engine.zig` / `src/main.zig`. Names can change; the operations cannot, or the engine has to be rewritten again.

Host types a backend must provide:

- `Device` — pick a default device, report a name and a maximum single-allocation size (Metal's `maxBufferLength` is checked at load in `Engine.initOn`).
- `Library` — compiled kernel module (Metal source today; SPIR-V, PTX/CUBIN, or a CPU function table on other backends).
- `Pipeline` — one named entry point.
- `Buffer` — device-visible storage; `contents` / `as(T)` only where the memory is host-visible.
- `Queue`, `CommandBuffer`, `Encoder` — record a pass, commit, wait, read GPU time.
- `Size` — dispatch grid (`w`, `h`, `d`), already used by `size1` / `size2` / `size3` in `src/engine.zig`.
- `Pool` — scoped lifetime for ephemeral command objects (Metal autorelease pool today; can be a no-op or an arena elsewhere).

Encoder operations the engine actually uses: `setPipeline`, `setBuffer` (buffer + offset + slot), `setParams` (small POD, Metal uses `setBytes`), `dispatchThreads` / `dispatchGroups`, `end`. The engine also needs a way to compile or load pipelines by name (`embed`, `add_rmsnorm`, `gemv_*`, `gemm_*`, `gdn_*`, `attn_*`, `argmax`, `touch_pages`, `probe_mem_sg`, `swiglu`, `gemm_reduce`, and the 4-bit `gemm_a4_*` set).

Two memory facts must be explicit on the interface, because Apple Silicon hides them:

1. **Weight placement.** Metal today maps the `.tern` file and calls `device.bufferNoCopy` (`Engine.initOn`). That is unified memory. A discrete GPU backend must copy (or stream) weights into device memory and must not assume a host pointer is a device pointer.
2. **Token buffer visibility.** The token id stays on the GPU (`tok`, `pred`): argmax writes it, the next embed reads it, and the CPU encodes ahead (`lookahead` in `src/main.zig`). A backend must support that device-side producer/consumer between command buffers. Peek/commit of linear-attention state (`State.advance` vs `State.peek`, `lin_proj`) is engine policy, not a backend opcode, but the backend must not make a `peek` pass write the state buffers.

Optional capability flags, resolved at comptime or once at init: `unified_memory`, `host_visible_weights`, `gpu_timestamp`, `simdgroup` / subgroup size, cooperative-matrix / tensor-core availability. Fall back to scalar or lane kernels when a capability is missing; do not require tensor cores to load a model.

## Backend set and selection

First implementations, one package each under `src/backend/`:

| Backend | Role |
| --- | --- |
| `metal` | Existing macOS path, first implementation of the interface (T6 check: same tokens and speed as before the split). |
| `cuda` | NVIDIA on Linux and Windows. Host API through the CUDA driver library (`libcuda` / `nvcuda.dll`) via the C ABI from Zig. Kernels as CUDA C++ or PTX, loaded at startup. |
| `vulkan` | Cross-vendor GPU on Linux and Windows (AMD, Intel, NVIDIA when CUDA is absent). Precompiled SPIR-V. Cooperative matrix (`VK_KHR_cooperative_matrix`, `VK_NV_cooperative_matrix2`) is optional and probed at init. |
| `cpu` | Fallback and the only backend that can be tested without a GPU. Zig (or a small C kernel file) with AVX2 / AVX-512 on x86_64 and NEON on arm64, plus a thread pool. Bandwidth of this path is not measured yet ([ideas.md](../ideas.md)). |

Out of scope for the first interface: HIP/ROCm as a separate package (Vulkan is the AMD coverage; a HIP backend can implement the same interface later), SYCL, DirectML / D3D12 / Windows ML (operator graphs, not custom ternary kernels; DirectML is in sustained engineering). Those are not required to satisfy T6.

Selection is compile-time, matching T6's `-Dbackend=metal` (and later `cuda`, `vulkan`, `cpu`). The hot path stays monomorphized: `const B = backend.Active;` and then `B.Encoder`, with `inline` as today. Building several backends into one binary can come later as a comptime choice of a small union of packages, still without function pointers on the encode path. Runtime work is only "which device inside the chosen backend."

### Kernel toolchain (not Zig-in-the-shader first)

Zig 0.17 has a self-hosted SPIR-V backend (`spirv64-vulkan`) and an NVPTX path that can emit PTX. As of mid-2026 the SPIR-V backend was improving but still far from complete (on the order of half of the behaviour tests passing; SPIR-V behaviour-test CI had been disabled). That is not a safe first vehicle for this kernel set. Zig-as-CUDA-host via the C driver API is the conservative host path; writing the kernels themselves in Zig/PTX can be a later experiment.

A single-source shader language (Slang) can emit SPIR-V, CUDA, and experimental MSL / HLSL. Metal support in Slang is still experimental (known buffer-index bugs in large modules). Recommendation: keep the current Metal Shading Language for the Metal backend; add CUDA C++ for `cuda` and GLSL or Slang → SPIR-V (`glslang` / `glslc`, the same family llama.cpp's Vulkan backend uses) for `vulkan`; keep `cpu` in Zig. Do not block the split on a unified shader language. A later Slang port is optional once Metal codegen is trusted.

## Memory model

The `.tern` format stays: magic `TERN0001`, 16 KiB page alignment, tensors in execution order, produced by `tools/convert.py`. That layout is for streaming and for mapping, not for Metal.

Load path:

1. Map the file read-only on the host (POSIX `mmap` today in `src/model.zig`; Windows `CreateFileMapping` / `MapViewOfFile` or the equivalent in Zig's standard library). Keep the 16 KiB alignment of the mapping.
2. On a unified device (Apple Silicon, some laptop APUs), hand the mapping to the device without a copy, then run a `touch_pages` equivalent so first-token page wiring is paid at load time.
3. On a discrete device, allocate a device buffer and upload (or stage through a host-visible window). A 7 GB weight file will not fit in a tiny BAR; plan an explicit upload with progress, and do not pin the whole host map into the GPU's limited visible window.
4. Activations, KV cache, delta-rule state, `tok` / `pred` stay in device memory. The CPU needs to read `pred` / token counts only at speculation boundaries (today a verification pass is not pipelined: the CPU waits for the outcome; that is an engine issue, not a reason to make every token host-round-trip).

`Device.maxBufferLength` (or the CUDA / Vulkan equivalent) must be checked before assuming one buffer can hold the whole weight file. If a backend cannot, the interface should allow a split mapping (several buffers + a base-offset table) without changing the `Mat` byte-offset scheme in `src/engine.zig` more than necessary.

## Zig signature sketch

Illustrative, not a commitment to these exact names. The point is a comptime package with the operations above, no trait objects.

```zig
pub const Kind = enum { metal, cuda, vulkan, cpu };

/// Chosen with `-Dbackend=...` in `build.zig`.
pub const active: Kind = @import("config").backend;
pub const B = switch (active) {
    .metal => @import("metal.zig"),
    .cuda => @import("cuda.zig"),
    .vulkan => @import("vulkan.zig"),
    .cpu => @import("cpu.zig"),
};

pub const Memory = enum { unified, discrete };

pub const Device = struct {
    pub fn default() !Device { ... }
    pub fn name(self: Device) [*:0]const u8 { ... }
    pub fn maxBufferLength(self: Device) usize { ... }
    pub fn memory(self: Device) Memory { ... }
    pub fn queue(self: Device) Queue { ... }
    pub fn library(self: Device, ir: []const u8) !Library { ... }
    pub fn buffer(self: Device, len: usize) !Buffer { ... }
    /// Unified only; discrete backends return an error and the engine uploads instead.
    pub fn bufferNoCopy(self: Device, ptr: [*]u8, len: usize) !Buffer { ... }
};

pub const Library = struct {
    pub fn pipeline(self: Library, func: [*:0]const u8) !Pipeline { ... }
};

pub const Encoder = struct {
    pub inline fn setPipeline(self: Encoder, p: Pipeline) void { ... }
    pub inline fn setBuffer(self: Encoder, b: Buffer, offset: usize, index: usize) void { ... }
    pub inline fn setParams(self: Encoder, params: anytype, index: usize) void { ... }
    pub inline fn dispatchThreads(self: Encoder, grid: Size, group: Size) void { ... }
    pub inline fn dispatchGroups(self: Encoder, groups: Size, group: Size) void { ... }
    pub inline fn end(self: Encoder) void { ... }
};

pub const CommandBuffer = struct {
    pub fn encoder(self: CommandBuffer) Encoder { ... }
    pub fn commit(self: CommandBuffer) void { ... }
    pub fn wait(self: CommandBuffer) !void { ... }
    pub fn gpuSeconds(self: CommandBuffer) f64 { ... }
};
```

`Engine.initOn` already accepts an external `Device` and `Library` so a main model and a draft model can share them. Keep that shape: the draft engine must run in the same command buffers as the main model (`src/main.zig`).

`build.zig` grows a `-Dbackend` option, links only the frameworks or libraries of that backend (today: `objc`, `Foundation`, `Metal`, `link_libc`), and for `vulkan` / `cuda` either embeds precompiled IR or invokes the shader toolchain as a build step. `zig build test` keeps testing `src/draft.zig` without a GPU; backend unit tests that need no device (SPIR-V parse, dispatch math, CPU kernels on tiny tensors) should live next to each package.

## Reuse, port, rewrite

### Reuse as-is (or with a thin port)

- `src/draft.zig` — prompt lookup, pass-size policy, unit tests already in CI. No GPU API. Keep unchanged.
- `src/model.zig` — `cfg` and `cfg_draft` shapes, `TERN0001` reader, `Entry` / `Tensor`. The POSIX `open` / `lseek` / `mmap` in `File.open_` must gain a Windows mapping path; the format and the rest of the file stay.
- `tools/convert.py` — MLX 2-bit and 4-bit to `.tern`. No Metal. Keep.
- `bench/` — prompts, rendered ids, `compare.py` / `*_bench.py`, result JSON. Harness is OS-agnostic if the `tern` binary exists. Keep; extend device names in the JSON later.
- Graph and policy in `src/engine.zig` — `EngineOf(cfg)`, `Mat` / `Layer`, `Config`, `State`, `Part`, `encodeBlock` / `encodeVerify` / `encodeCommit` / `encodeHead` / `encodeHeadLast` / `matmul` / `batchFit`. This is the reusable architecture, but every type that is `metal.Device` / `metal.Buffer` / `metal.Encoder` today has to be `B.Device` and so on. That is a mechanical edit, not a new encoder.
- CLI shape in `src/main.zig` — `gen`, `prof`, `bench`, flags, JSON line, speculation ring, draft feed. The command-buffer ring and timing calls are backend-typed; `clock_gettime_nsec_np` is macOS-only and must be replaced with a portable clock (Linux `clock_gettime`, Windows `QueryPerformanceCounter` or Zig `std.time`).

### Port (algorithm stays, source does not)

- `src/kernels.metal` — every compute kernel: `embed` / `embed_a4`, `add_rmsnorm`, ternary GEMV / lane GEMM / 8-token and 32-token tiles / `gemm_reduce`, 4-bit `gemm_a4_*`, `swiglu`, GDN chain (`gdn_conv`, `gdn_qk_norm`, `gdn_step` / `gdn_step_sg`, `gdn_out_norm`), attention (`attn_prep`, `attn_scores`, `attn_softmax` / `_sg`, `attn_out` / `_sg`), `argmax`, `touch_pages`, `probe_mem_sg`. Numeric contract stays: integer unpack of ternary (or 4-bit affine) weights, exact -1/0/+1, float sums, half storage, same summation order until a measured token check allows a change ([plan/README.md](../plan/README.md)).
- Lane-split ("sg") kernels and the tiled GEMM that uses 8×8 matrix intrinsics (see [README.md](../README.md) and [REPORT.md](../REPORT.md)) need backend-native equivalents: Metal SIMD-groups, CUDA warps / tensor cores, Vulkan subgroups / cooperative matrix. These are ports of the algorithm, not copy-paste of MSL.
- `touch_pages` and the memory-floor probe become backend-specific, same purpose.

### Rewrite from scratch

- `src/metal.zig` — stays as the Metal package under the new interface; CUDA, Vulkan and CPU are new files. Do not try to implement CUDA "through" this Objective-C wrapper.
- Dispatch, pipeline creation, and `bufferNoCopy` load in `src/engine.zig` — new per backend, behind the interface.
- `build.zig` and `.github/workflows/ci.yml` — today they assume macOS, `zig fmt`, `zig build --release=fast`, `zig build test`, and `xcrun metal -c src/kernels.metal`. Linux and Windows jobs, shader/PTX compile, and `-Dbackend=` variants are new.
- Host file mapping on Windows, portable clocks, and discrete-GPU upload.
- Any HIP, SYCL, or Direct3D backend, if ever added.

## T6 acceptance (this design)

Done means: `zig build -Dbackend=metal` produces the same greedy tokens and, on a quiet interleaved run, the same speed as the pre-split Metal binary; `EngineOf` no longer imports Metal symbols except through `src/backend/metal.zig`; at least one other backend compiles and passes the three-prompt token check on its hardware (CPU is enough for the first "second backend" if no Linux GPU is available). Shader IR for a non-Metal backend is produced in CI even when a GPU is not.

This PR does not change production code. Implementation of T6 is a later PR.
