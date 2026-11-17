import os
import numpy as np
import tracking_io as tio
from games import GAME, GAMES
from starter.load_smart_tagging import load

FPS = GAMES[GAME]["fps"]
L, W = GAMES[GAME]["pitch"]
GRADED = ((11185, 90651), (110808, 190197))
LAG = 5


def build():
    T = tio.load()
    bx, by = T["bx"], T["by"]
    n = len(bx)
    seen = ~np.isnan(bx)
    with np.errstate(invalid="ignore"):
        spd = np.full(n, np.nan)
        spd[:-LAG] = np.hypot(bx[LAG:] - bx[:-LAG], by[LAG:] - by[:-LAG]) * FPS / LAG
        edge = np.minimum.reduce([bx, L - bx, by, W - by]) <= 1.5
        live = seen & (spd > 4.0) & ~edge

    def find_restart(a):
        for f in range(a + FPS, a + 12 * FPS):
            w = slice(f, f + FPS)
            if seen[w].mean() < 0.5:
                continue
            s = spd[w][seen[w]]
            s = s[~np.isnan(s)]
            if s.size == 0 or s.mean() < 3.0 or edge[w][seen[w]].mean() > 0.5:
                continue
            return f
        return None

    def find_start(a):
        lo = a - 4 * FPS
        idx = np.flatnonzero(live[lo:a + FPS])
        if idx.size == 0:
            return a, False
        return min(lo + idx[-1] + LAG, a + FPS), True

    ev = [e for e in load() if (e["code"] or "").strip().lower() == "set pieces"
          and "kick off" not in (e["tags"].get("Type") or "").lower()]
    rows = []
    for e in ev:
        a = e["start_frame"]
        s, s_ok = find_start(a)
        r = find_restart(a)
        rows.append({"type": e["tags"].get("Type"), "a": a, "s": s, "s_ok": s_ok, "r": r})
    med = {}
    for t in {x["type"] for x in rows}:
        d = [x["r"] - x["a"] for x in rows if x["type"] == t and x["r"] is not None]
        med[t] = int(np.median(d)) if d else 6 * FPS
    for x in rows:
        x["r_ok"] = x["r"] is not None
        if x["r"] is None:
            x["r"] = x["a"] + med[x["type"]]
        x["r"] = max(x["r"], x["s"] + FPS)
    dead = np.zeros(n, bool)
    for x in rows:
        dead[x["s"]:x["r"]] = True
    return rows, dead


if __name__ == "__main__":
    rows, dead = build()
    n = len(dead)
    graded = np.zeros(n, bool)
    for a, b in GRADED:
        graded[a:b + 1] = True
    print("events:", len(rows))
    for t in sorted({x["type"] for x in rows}):
        g = [x for x in rows if x["type"] == t]
        so = np.array([(x["s"] - x["a"]) / FPS for x in g])
        du = np.array([(x["r"] - x["s"]) / FPS for x in g])
        print("{:<12} n={:<3} start found {}  restart found {}".format(
            t, len(g), sum(x["s_ok"] for x in g), sum(x["r_ok"] for x in g)))
        print("   start vs window start (s) p10/50/90: {}".format(np.percentile(so, [10, 50, 90]).round(1).tolist()))
        print("   dead length (s)           p10/50/90: {}".format(np.percentile(du, [10, 50, 90]).round(1).tolist()))
    h1 = graded & (np.arange(n) < 100000)
    h2 = graded & (np.arange(n) >= 100000)
    print("set-piece dead frames: 1st half {} ({:.1f}% of half), 2nd half {} ({:.1f}% of half)".format(
        int(dead[h1].sum()), 100 * dead[h1].mean(), int(dead[h2].sum()), 100 * dead[h2].mean()))
    xs = sorted(rows, key=lambda x: x["s"])
    print("overlapping neighbours:", sum(1 for p, q in zip(xs[:-1], xs[1:]) if q["s"] < p["r"]))
    os.makedirs("cache", exist_ok=True)
    np.save("cache/dead_sp.npy", dead)
    print("saved cache/dead_sp.npy")