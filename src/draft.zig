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
/// A pass of 4 tokens costs about as much as 2.8 sequential tokens, one of 8 about 3.6, one of
/// 17 to 32 about 8 (REPORT.md), so a pass pays off only when most of it is accepted. The depth
/// starts at the smallest size, moves one size up after a fully accepted pass and falls back to
/// what was accepted otherwise. A pass that yields at most two tokens is a loss at any size:
/// it is followed by a pause of `pause` plain tokens, longer after every loss in a row.
pub const Policy = struct {
    /// Pass sizes, rising; `count` of them are in use.
    sizes: [3]u32 = .{ 4, 8, 32 },
    count: u32 = 3,
    /// Shortest suffix match that starts a pass.
    min_match: u32 = 4,
    pause: u32 = 4,

    level: u32 = 0,
    losses: u32 = 0,
    /// Plain tokens left before the next pass may start.
    cool: u32 = 0,

    /// Per pass size: passes run, tokens they produced, wall time they took.
    stats: [3]Stat = @splat(.{}),

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
    try std.testing.expectEqual(@as(u32, 4), p.size(100));
    p.passed(4, 4, 0.1);
    try std.testing.expectEqual(@as(u32, 8), p.size(100));
    try std.testing.expectEqual(@as(u32, 4), p.size(5));
    try std.testing.expectEqual(@as(u32, 0), p.size(3));
    p.passed(8, 8, 0.1);
    try std.testing.expectEqual(@as(u32, 32), p.size(100));
    p.passed(32, 9, 0.1);
    try std.testing.expectEqual(@as(u32, 8), p.size(100));
    p.passed(8, 1, 0.1);
    try std.testing.expectEqual(@as(u32, 0), p.size(100));
    for (0..4) |_| p.plain();
    try std.testing.expectEqual(@as(u32, 4), p.size(100));
    try std.testing.expectEqual(@as(u32, 2), p.stats[1].passes);
    try std.testing.expectEqual(@as(u32, 8 + 1), p.stats[1].tokens);
    try std.testing.expectEqual(@as(u32, 9), p.stats[2].tokens);
}
