//! Chained speculative passes with a draft model. Every pass verifies one draft token, and the
//! GPU accepts it and sets the position and counts of the next pass itself, so the CPU keeps
//! several passes in flight without reading their outcome, as it does for plain tokens.
//!
//! One pass, in one command buffer:
//!   1. the main model commits the tokens the previous pass accepted (recurrent states);
//!   2. the draft model is fed those tokens in two slots and picks after the last real one;
//!   3. `chain_draft` writes that pick as the draft for P + 1;
//!   4. the main model verifies [P, P + 1] without touching its recurrent states;
//!   5. `chain_accept` accepts one or two tokens and sets up the next pass.

const engine_mod = @import("engine.zig");
const metal = @import("metal.zig");
const Engine = engine_mod.Engine;
const DraftEngine = engine_mod.DraftEngine;
const Dyn = engine_mod.Dyn;

/// Passes whose end position the log keeps; more than can be in flight.
pub const log_len = 64;

const ChainP = extern struct { main_dyn: u32, draft_dyn: u32, seq: u32 };
const main_off = 0;
const draft_off = engine_mod.dyn_blocks * engine_mod.dyn_block;
const one: metal.Size = .{ .w = 1, .h = 1, .d = 1 };

pub const Chain = struct {
    /// {P, got, log[log_len]}, see `chain_accept` in kernels.metal.
    ctrl: metal.Buffer,
    /// GPU-held parameter blocks of both engines.
    dyn: metal.Buffer,
    k_draft: metal.Pipeline,
    k_accept: metal.Pipeline,

    pub fn init(main: *const Engine) !Chain {
        return .{
            .ctrl = try main.device.buffer((2 + log_len) * 4),
            .dyn = try main.device.buffer(2 * draft_off),
            .k_draft = try main.lib.pipeline("chain_draft"),
            .k_accept = try main.lib.pipeline("chain_accept"),
        };
    }

    /// Sets up a chain whose first pass is at `pos`: both models have seen every token before
    /// it, the draft model gets the one at `pos`. Nothing may be in flight.
    pub fn start(self: *const Chain, main: *const Engine, draft: *const DraftEngine, pos: u32) void {
        const ctrl = self.ctrl.as(u32);
        ctrl[0] = pos;
        ctrl[1] = 1;
        main.dynInit(self.dyn, main_off, pos, 1);
        draft.dynInit(self.dyn, draft_off, pos, 1);
    }

    /// Position after pass `seq` of the chain (the last confirmed token); valid once its
    /// command buffer has completed and until pass `seq + log_len` runs.
    pub fn after(self: *const Chain, seq: u32) u32 {
        return self.ctrl.as(u32)[2 + seq % log_len];
    }

    /// Encodes pass `seq` (0 for the first pass of a chain, which has nothing to commit).
    /// `bound` is at least the position of the pass.
    pub fn encodePass(self: *const Chain, enc: metal.Encoder, main: *const Engine, draft: *const DraftEngine, seq: u32, bound: u32) void {
        const m = Dyn{ .buf = self.dyn, .off = main_off, .bound = bound };
        const d = Dyn{ .buf = self.dyn, .off = draft_off, .bound = bound };
        if (seq != 0) main.encodeCommitAt(enc, m, 2);
        draft.encodeFeedAt(enc, d, 2);

        enc.setPipeline(self.k_draft);
        enc.setBuffer(draft.pred, 0, 0);
        enc.setBuffer(main.tok, 0, 1);
        enc.setBuffer(self.ctrl, 0, 2);
        enc.dispatchThreads(one, one);

        main.encodeVerify(enc, .{ .gpu = m }, 2);

        enc.setPipeline(self.k_accept);
        enc.setBuffer(main.pred, 0, 0);
        enc.setBuffer(main.tok, 0, 1);
        enc.setBuffer(self.ctrl, 0, 2);
        enc.setBuffer(self.dyn, 0, 3);
        enc.setParams(&ChainP{ .main_dyn = main_off / 4, .draft_dyn = draft_off / 4, .seq = seq % log_len }, 4);
        enc.dispatchThreads(one, one);
    }
};
