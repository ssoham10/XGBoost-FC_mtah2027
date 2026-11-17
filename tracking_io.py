"""Raw ATD -> numpy arrays in metres, cached. Everything we build later imports this."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from games import GAME, GAMES, require, source_frames  # noqa: E402
from starter.baseline import team_of_each_track  # noqa: E402

CACHE_DIR = os.path.join(HERE, "cache")  # gitignored: frame-level tracking must never be pushed


def _parse(raw_path, n_frames, length, width):
    px = py = ps = None
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1  # frame_id counts from 1, video frames from 0
            vals = np.array(parts[1].strip().strip(";").replace(";", ",").split(","),
                            dtype=np.float32)
            if vals.size % 3:
                raise ValueError("frame {}: {} player values, not a multiple of 3".format(i, vals.size))
            vals = vals.reshape(-1, 3)
            if px is None:
                n_slots = len(vals)
                px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
                py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
                ps = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
            if len(vals) != px.shape[1]:
                raise ValueError("frame {}: {} slots, expected {}".format(i, len(vals), px.shape[1]))
            px[i], py[i], ps[i] = vals[:, 0] * length, vals[:, 1] * width, vals[:, 2]

            b = parts[2].strip().split(",")
            if not b[0].startswith("N") and len(b) >= 3:
                ball[i] = np.array(b[:3], dtype=np.float32)
    ball[:, 0] *= length
    ball[:, 1] *= width
    return px, py, ps, ball


def load(game=GAME, data_dir=None):
    """dict: px, py, pspeed (n_frames, n_slots); bx, by, bspeed (n_frames,); team (n_slots,)."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    cache = os.path.join(CACHE_DIR, "tracking_{}.npz".format(game))
    team = team_of_each_track(meta_path)
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files} | {"team": team}
    length, width = GAMES[game]["pitch"]
    n_frames = source_frames("tactical", game)
    print("parsing raw tracking once (a minute or two) ...")
    px, py, ps, ball = _parse(raw_path, n_frames, length, width)
    out = {"px": px, "py": py, "pspeed": ps,
           "bx": ball[:, 0], "by": ball[:, 1], "bspeed": ball[:, 2]}
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out | {"team": team}