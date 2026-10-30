"""
Synthetic tracker output for tests.

Simulates holes moving along the image y axis past a counting line, with
optional detection noise, low-confidence frames, dropouts and track id
switches. Returns one list of Detection objects per frame.
"""
import random

from counting import Detection


def simulate(n_anodes=10, holes_per_anode=4, pitch=60.0, anode_gap=60.0, speed=4.0,
             start_y=700.0, direction="up", miss_prob=0.0, low_conf_prob=0.0,
             id_switch_prob=0.0, jitter=0.0, missing_holes=(), extra_tracks=0,
             frames=None, seed=0, box_h=40.0, weak_after_line=0.0, line_y=520.0, return_tracks=False):
    """
    Holes are laid out along the belt: each anode has holes_per_anode holes
    spaced by pitch, anodes are separated by an extra anode_gap. Every frame the
    belt moves by speed pixels in the counting direction.

    missing_holes: set of (anode_index, hole_index) that are never detected.
    weak_after_line: probability that a hole is only weakly detected (conf
        0.40-0.75) once it is past line_y, e.g. fumes over the far side of the
        line or holes shrinking with distance from the camera.
    return_tracks: also return {track_id: (anode_index, hole_index)}.
    """
    rng = random.Random(seed)
    holes = []  # (offset along belt, anode index, hole index)
    pos = 0.0
    for a in range(n_anodes):
        for h in range(holes_per_anode):
            if (a, h) not in missing_holes:
                holes.append((pos, a, h))
            pos += pitch
        pos += anode_gap
    span = pos
    if frames is None:
        frames = int((span + 2 * start_y) / speed)

    weak = {(a, h) for (_, a, h) in holes if rng.random() < weak_after_line}
    next_id = 1
    track_of = {}
    truth = {}
    out = []
    for f in range(frames):
        dets = []
        for (offset, a, h) in holes:
            # first hole of the train is nearest the line; later holes trail behind.
            # "down" is the mirror image of "up" (use line 1080 - line_y).
            y = start_y - (f * speed - offset)
            if direction == "down":
                y = 1080.0 - y
            if y < -box_h or y > 1080 + box_h:
                continue
            if rng.random() < miss_prob:
                continue
            key = (a, h)
            if key not in track_of or rng.random() < id_switch_prob:
                track_of[key] = next_id
                truth[next_id] = key
                next_id += 1
            cy = y + (rng.uniform(-jitter, jitter) if jitter else 0.0)
            conf = rng.uniform(0.35, 0.79) if rng.random() < low_conf_prob else rng.uniform(0.8, 0.99)
            past_line = y < line_y if direction == "up" else y > 1080.0 - line_y
            if key in weak and past_line:
                conf = rng.uniform(0.40, 0.75)
            dets.append(Detection(track_of[key], 900.0, cy - box_h / 2, 1000.0, cy + box_h / 2, conf))
        for _ in range(extra_tracks):
            if rng.random() < 0.05:
                cy = rng.uniform(0, 1080)
                dets.append(Detection(next_id, rng.uniform(0, 1800), cy - 10, rng.uniform(0, 1800) + 20,
                                      cy + 10, rng.uniform(0.3, 0.95)))
                next_id += 1
        rng.shuffle(dets)
        out.append(dets)
    return (out, truth) if return_tracks else out
