//! Draft source and depth policy of speculative decoding.
//!
//! The draft is a prompt lookup: when the last tokens of the context already occurred earlier,
//! the tokens that followed them there are proposed as the continuation. The model verifies the
//! whole draft in one batched pass and keeps the prefix it agrees with, so the output is the
//! one greedy decoding would give; the policy only decides when a pass is worth its price.

const std = @import("std");

/// Longest suffix compared by `find`.
pub const max_match = 16;

pub const Match = struct {
    /// Index of the token that followed the earlier occurrence.
    at: u32,
    /// Number of tokens the two occurrences share.
    len: u32,
};

/// The earlier occurrence of the context's suffix with the longest match (up to `max_match`
/// tokens), the most recent one among equals; null when fewer than `min_len` tokens match.
pub fn find(ctx: []const u32, min_len: u32) ?Match {
    if (ctx.len < 2) return null;
    const last = ctx.len - 1;
    var best = Match{ .at = 0, .len = 0 };
    var e = last;
    while (e > 0) {
        e -= 1;
        if (ctx[e] != ctx[last]) continue;
        var len: u32 = 1;
        while (len < max_match and len <= e and ctx[e - len] == ctx[last - len]) len += 1;
        if (len > best.len) {
            best = .{ .at = @intCast(e + 1), .len = len };
            if (len == max_match) break;
        }
    }
    return if (best.len >= min_len and best.len > 0) best else null;
}

/// Writes the continuation of `m` after `tok[last]`: `tok[last + 1 ..][0..n]`. The source may
/// run into the tokens being written, which continues a periodic text.
pub fn fill(tok: [*]u32, last: u32, m: Match, n: u32) void {
    for (0..n) |i| tok[last + 1 + i] = tok[m.at + i];
}

/// How many tokens the next verification pass carries.
///
/// A pass of 2 tokens costs about as much as 1.1 sequential tokens, one of 4 about 2.1, one of 8
/// about 3.5, one of 17 to 32 about 8 (REPORT.md). The first size is nearly free, the larger
/// ones pay off only when most of the pass is accepted. The depth starts at the smallest size,
/// moves one size up after a fully accepted pass and falls back to what was accepted otherwise.
/// A pass that yields at most two tokens without being full is a loss: it is followed by a
/// pause of `pause` plain tokens, longer after every loss in a row.
pub const Policy = struct {
    /// Pass sizes, rising; `count` of them are in use.
    sizes: [max_sizes]u32 = .{ 2, 4, 8, 32 },
    count: u32 = max_sizes,
    /// Shortest suffix match that starts a pass.
    min_match: u32 = 4,
    pause: u32 = 4,

    level: u32 = 0,
    losses: u32 = 0,
    /// Plain tokens left before the next pass may start.
    cool: u32 = 0,

    /// Per pass size: passes run, tokens they produced, wall time they took.
    stats: [max_sizes]Stat = @splat(.{}),

    pub const max_sizes = 4;

    pub const Stat = struct { passes: u32 = 0, tokens: u32 = 0, seconds: f64 = 0 };

    /// Size of the next pass given that `room` tokens are still to be generated; 0 for none.
    pub fn size(self: *const Policy, room: u32) u32 {
        if (self.cool != 0) return 0;
        var l = self.level;
        while (l > 0 and self.sizes[l] > room) l -= 1;
        return if (self.sizes[l] <= room) self.sizes[l] else 0;
    }

    /// A token was produced without a pass.
    pub fn plain(self: *Policy) void {
        if (self.cool != 0) self.cool -= 1;
    }

    /// A pass of two with a draft model's token, outside the ladder, produced `got` tokens: they
    /// count against the pause like plain ones.
    pub fn modelPass(self: *Policy, got: u32) void {
        self.cool -|= got;
    }

    /// A pass of `n` tokens (a size returned by `size`) produced `got` new tokens (1 to n).
    pub fn passed(self: *Policy, n: u32, got: u32, seconds: f64) void {
        var used: u32 = 0;
        while (self.sizes[used] != n) used += 1;
        self.stats[used].passes += 1;
        self.stats[used].tokens += got;
        self.stats[used].seconds += seconds;
        if (got == n) {
            self.losses = 0;
            if (self.level + 1 < self.count and self.sizes[self.level] == n) self.level += 1;
            return;
        }
        while (self.level > 0 and self.sizes[self.level] > got) self.level -= 1;
        if (got <= 2) {
            self.losses += 1;
            self.cool = self.pause * self.losses;
        } else {
            self.losses = 0;
        }
    }
};

/// Whether the positions without a prompt match run as passes of two with a draft model's
/// token or as plain tokens.
///
/// A pass of two costs 1.45 to 2.1 plain tokens depending on the state of the machine and
/// yields one token plus the draft model's hit rate (REPORT.md), so which way is faster is not
/// known ahead: the gate times both. It runs each way for `trial` tokens, stays with the faster
/// one for `stay_first` tokens and then tries the other one again; the stay doubles every time
/// the retrial confirms the choice. The first `settle` tokens of a stretch are not timed, the
/// GPU's speed still follows what ran before them. The passes have to be faster by `margin`:
/// two stretches differ by a few percent from noise alone, and every switch costs a little.
pub const Gate = struct {
    pub const Way = enum(u1) { model, plain };

    trial: u32 = 16,
    settle: u32 = 4,
    stay_first: u32 = 96,
    stay_max: u32 = 384,
    margin: f64 = 0.05,

    way: Way = .model,
    /// Seconds per token of each way, the mean of its last stretch and the value before it;
    /// 0 until the way has been timed.
    cost: [2]f64 = .{ 0, 0 },
    switches: u32 = 0,

    /// Length of the stay that follows the next confirmed choice; 0 before the first choice.
    stay: u32 = 0,
    /// The stretch being run is a trial of the way that was not chosen.
    trying: bool = false,
    /// Tokens of the stretch: its length, those produced so far, those timed and their time.
    len: u32 = 0,
    run: u32 = 0,
    timed: u32 = 0,
    seconds: f64 = 0,

    /// `tokens` came out one way in `seconds` of wall time. Tokens of the way the gate has
    /// already left (still in flight when it switched) are ignored.
    pub fn sample(self: *Gate, way: Way, tokens: u32, seconds: f64) void {
        if (way != self.way) return;
        if (self.len == 0) self.len = self.trial;
        if (self.run >= self.settle) {
            self.timed += tokens;
            self.seconds += seconds;
        }
        self.run += tokens;
        if (self.run < self.len) return;

        const i = @backingInt(way);
        if (self.timed != 0) {
            const c = self.seconds / @as(f64, @floatFromInt(self.timed));
            self.cost[i] = if (self.cost[i] == 0) c else (self.cost[i] + c) / 2;
        }
        const other: Way = @fromBackingInt(@intCast(1 - i));
        var next = other;
        var len = self.trial;
        if (self.trying) {
            // Both ways are timed: stay with the faster one.
            const wins = if (self.cost[0] * (1 + self.margin) <= self.cost[1]) Way.model else Way.plain;
            if (wins == way) {
                next = way;
                self.stay = self.stay_first;
            } else {
                self.stay = if (self.stay == 0) self.stay_first else @min(2 * self.stay, self.stay_max);
            }
            len = self.stay;
        }
        self.trying = !self.trying;
        if (next != way) self.switches += 1;
        self.way = next;
        self.len = len;
        self.run = 0;
        self.timed = 0;
        self.seconds = 0;
    }
};

test "gate stays with the faster way and retries the other at growing intervals" {
    var g = Gate{};
    // A pass of two takes 0.07 s and yields 1.5 tokens, a plain token takes 0.04 s.
    var plain_tokens: u32 = 0;
    var model_tokens: u32 = 0;
    var stays: [3]u32 = undefined;
    var n_stays: u32 = 0;
    var before = g.way;
    var run: u32 = 0;
    for (0..800) |k| {
        if (g.way == .model) {
            const got: u32 = if (k % 2 == 0) 2 else 1;
            g.sample(.model, got, 0.07);
            model_tokens += got;
        } else {
            g.sample(.plain, 1, 0.04);
            plain_tokens += 1;
        }
        run += 1;
        if (g.way != before) {
            if (before == .plain and n_stays < stays.len) {
                stays[n_stays] = run;
                n_stays += 1;
            }
            before = g.way;
            run = 0;
        }
    }
    // Plain: the trial of 16 and a stay of 96, then stays of 192 and 384; the model gets its
    // trials only.
    try std.testing.expectEqualSlices(u32, &[_]u32{ 16 + 96, 192, 384 }, stays[0..n_stays]);
    try std.testing.expect(model_tokens < 4 * 18);
    try std.testing.expect(g.cost[0] > g.cost[1]);
    try std.testing.expectApproxEqAbs(@as(f64, 0.04), g.cost[1], 1e-9);

    // The other way round: the model is faster, plain tokens get the trials.
    var h = Gate{};
    var plain_run: u32 = 0;
    for (0..110) |_| {
        if (h.way == .model) h.sample(.model, 2, 0.06) else {
            h.sample(.plain, 1, 0.04);
            plain_run += 1;
        }
    }
    try std.testing.expectEqual(@as(u32, 2 * 16), plain_run);
    try std.testing.expectEqual(Gate.Way.model, h.way);
    // A token of the way already left changes nothing.
    const cost = h.cost;
    h.sample(.plain, 1, 9.0);
    try std.testing.expectEqual(cost, h.cost);

    // Passes that are 2% faster do not clear the margin.
    var m = Gate{};
    for (0..8) |_| m.sample(.model, 2, 0.0784);
    for (0..16) |_| m.sample(.plain, 1, 0.04);
    try std.testing.expectEqual(Gate.Way.plain, m.way);
    try std.testing.expectEqual(@as(u32, 96), m.len);
}

test "find prefers the longest, then the most recent match" {
    const ctx = [_]u32{ 1, 2, 3, 9, 2, 3, 7, 1, 2, 3 };
    const m = find(&ctx, 2).?;
    try std.testing.expectEqual(@as(u32, 3), m.len);
    try std.testing.expectEqual(@as(u32, 3), m.at);
    try std.testing.expect(find(&ctx, 4) == null);
    try std.testing.expect(find(&[_]u32{ 1, 2, 3 }, 1) == null);
    // Equal lengths: the later occurrence wins.
    const m2 = find(&[_]u32{ 5, 6, 1, 5, 6, 2, 5, 6 }, 2).?;
    try std.testing.expectEqual(@as(u32, 5), m2.at);
}

test "fill continues a periodic text through its own output" {
    var tok = [_]u32{ 4, 5, 4, 5, 0, 0, 0, 0, 0 };
    const m = find(tok[0..4], 2).?;
    fill(&tok, 3, m, 5);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 4, 5, 4, 5, 4, 5, 4, 5, 4 }, &tok);
}

test "policy grows on full acceptance and pauses after a loss" {
    var p = Policy{};
    try std.testing.expectEqual(@as(u32, 2), p.size(100));
    p.passed(2, 2, 0.1);
    try std.testing.expectEqual(@as(u32, 4), p.size(100));
    try std.testing.expectEqual(@as(u32, 2), p.size(3));
    try std.testing.expectEqual(@as(u32, 0), p.size(1));
    p.passed(4, 4, 0.1);
    p.passed(8, 8, 0.1);
    try std.testing.expectEqual(@as(u32, 32), p.size(100));
    p.passed(32, 9, 0.1);
    try std.testing.expectEqual(@as(u32, 8), p.size(100));
    p.passed(8, 1, 0.1);
    try std.testing.expectEqual(@as(u32, 0), p.size(100));
    for (0..4) |_| p.plain();
    try std.testing.expectEqual(@as(u32, 2), p.size(100));
    try std.testing.expectEqual(@as(u32, 2), p.stats[2].passes);
    try std.testing.expectEqual(@as(u32, 8 + 1), p.stats[2].tokens);
    try std.testing.expectEqual(@as(u32, 9), p.stats[3].tokens);
    p.passed(2, 1, 0.1); // the second loss in a row: a pause of 8
    try std.testing.expectEqual(@as(u32, 8), p.cool);
    p.modelPass(2);
    p.modelPass(5);
    try std.testing.expectEqual(@as(u32, 1), p.cool);
    p.modelPass(2);
    try std.testing.expectEqual(@as(u32, 0), p.cool);
}
