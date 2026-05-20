#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from xml.etree import ElementTree as ET


W = 1200
H = 760
VIEWBOX_X = 300.0
VIEWBOX_Y = 190.0
VIEWBOX_W = 620.0
VIEWBOX_H = 420.0
ROUTE_SCALE = 0.84
FEATURE_SPREAD = 0.72
TITLE_W = 304.0
TITLE_H = 76.0
TITLE_SHIFT_X = 18.0
TITLE_LETTER_RECTS = [
    (-148.0, -91.0, -35.0, 34.0),  # S
    (-80.0, -34.0, -35.0, 34.0),   # L
    (-23.0, 39.0, -35.0, 34.0),    # A
    (54.0, 145.0, -35.0, 34.0),    # M
]
TITLE_LINK_OVERLAP_PX = 25.0
TITLE_LINK_FULL_DIST = 130.0
TITLE_LINK_MAX_DIST = 320.0
TITLE_BASE_OPACITY = 0.0
TITLE_REVEAL_OPACITY = 1.00
TITLE_REVEAL_RADIUS = 16.0
TITLE_REVEAL_POINT_THRESHOLD = 0.08
TITLE_SPOT_FADE_WINDOW = 0.045
TITLE_FINAL_REVEAL_START = 0.74
TITLE_FINAL_REVEAL_DURATION = 0.26
ROUTE_WAVE_AMP_PX = 6.0
ROUTE_WAVE_FREQ = 8.0
ROUTE_WAVE_PHASE = 0.4
CAMERA_FOV_DEG = 150.0
FRAMES = 200
SAMPLES = FRAMES + 1
DURATION = 30.0
CYCLE_REVEAL_RAMP = 0.035

FEATURES = [
    {"t": 0.05, "d": 76, "s": 6.0},
    {"t": 0.14, "d": 54, "s": 3.1},
    {"t": 0.23, "d": 67, "s": 5.2},
    {"t": 0.34, "d": 59, "s": 3.2},
    {"t": 0.44, "d": 73, "s": 6.2},
    {"t": 0.55, "d": 61, "s": 3.4},
    {"t": 0.66, "d": 79, "s": 6.4},
    {"t": 0.77, "d": 52, "s": 2.9},
    {"t": 0.87, "d": 68, "s": 5.4},
    {"t": 0.96, "d": 75, "s": 6.1},
    {"t": 0.17, "d": -45, "s": 2.4},
    {"t": 0.28, "d": -55, "s": 3.8},
    {"t": 0.39, "d": -41, "s": 2.2},
    {"t": 0.50, "d": -52, "s": 4.0},
    {"t": 0.61, "d": -44, "s": 2.5},
    {"t": 0.72, "d": -61, "s": 4.3},
    {"t": 0.83, "d": -38, "s": 2.1},
    {"t": 0.93, "d": -54, "s": 3.5},
    {"t": 0.07, "d": -58, "s": 3.7},
]

CAMERA_OFFSETS = [-0.042, 0.0, 0.065]
CAMERA_SCALES = [0.62, 1.0, 0.82]
CAMERA_ALPHA = [0.70, 1.0, 0.70]
CAMERA_LAPS = [5.0, 3.9, 3.1]


def clamp(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ellipse_center() -> tuple[float, float]:
    return W * 0.505, H * 0.53


def _route_point_base(t: float) -> tuple[float, float]:
    cx, cy = ellipse_center()
    a = W * 0.225 * ROUTE_SCALE
    b = H * 0.235 * ROUTE_SCALE
    th = t * math.pi * 2.0
    x = cx + a * (
        0.98 * math.cos(th)
        + 0.06 * math.cos(2 * th + 0.7)
        + 0.03 * math.cos(3 * th - 1.4)
    )
    y = cy + b * (
        0.94 * math.sin(th)
        - 0.05 * math.sin(2 * th - 0.5)
        + 0.03 * math.sin(3 * th + 1.0)
    )
    return x, y


def _route_tangent_base(t: float) -> tuple[float, float]:
    p1 = _route_point_base((t + 0.002) % 1.0)
    p0 = _route_point_base((t - 0.002 + 1.0) % 1.0)
    return p1[0] - p0[0], p1[1] - p0[1]


def route_point(t: float) -> tuple[float, float]:
    # Add a subtle "wavy" displacement around the base ellipse, along the local normal.
    x, y = _route_point_base(t)
    tx, ty = _route_tangent_base(t)
    tlen = max(1e-6, math.hypot(tx, ty))
    nx = -ty / tlen
    ny = tx / tlen
    th = t * math.pi * 2.0
    wave = ROUTE_WAVE_AMP_PX * math.sin(ROUTE_WAVE_FREQ * th + ROUTE_WAVE_PHASE)
    return x + nx * wave, y + ny * wave


def route_tangent(t: float) -> tuple[float, float]:
    p1 = route_point((t + 0.002) % 1.0)
    p0 = route_point((t - 0.002 + 1.0) % 1.0)
    return p1[0] - p0[0], p1[1] - p0[1]

def fmt(v: float) -> str:
    return f"{v:.2f}"


def join(seq: list[str]) -> str:
    return ";".join(seq)


def key_times(n: int) -> str:
    return ";".join(f"{i / (n - 1):.6f}" for i in range(n))


def camera_phase(t: float, offset: float, laps: float) -> float:
    return (t * laps + offset + 1.0) % 1.0


def camera_state(t: float, offset: float, laps: float) -> tuple[float, float]:
    return route_point(camera_phase(t, offset, laps))


def cycle_reveal_gate(t: float) -> float:
    return smoothstep(0.0, CYCLE_REVEAL_RAMP, t)


def title_center() -> tuple[float, float]:
    cx, cy = ellipse_center()
    return cx + TITLE_SHIFT_X, cy


def title_letter_center(letter_index: int) -> tuple[float, float]:
    cx, cy = title_center()
    left, right, top, bottom = TITLE_LETTER_RECTS[letter_index]
    return cx + (left + right) / 2.0, cy + (top + bottom) / 2.0


def title_letter_visibility_gate(cam: tuple[float, float], letter_index: int) -> float:
    cx, cy = ellipse_center()
    view_x = cx - cam[0]
    view_y = cy - cam[1]
    view_len = max(1e-6, math.hypot(view_x, view_y))
    min_view_cos = math.cos(math.radians(CAMERA_FOV_DEG / 2.0))

    lx, ly = title_letter_center(letter_index)
    feat_x = lx - cam[0]
    feat_y = ly - cam[1]
    feat_len = max(1e-6, math.hypot(feat_x, feat_y))
    view_cos = (view_x * feat_x + view_y * feat_y) / (view_len * feat_len)
    if view_cos < min_view_cos:
        return 0.0
    return smoothstep(TITLE_LINK_MAX_DIST, TITLE_LINK_FULL_DIST, feat_len)


def ray_intersect_title_letter(
    cam: tuple[float, float],
    via: tuple[float, float],
    letter_index: int,
) -> tuple[float, float]:
    """
    Intersection of the ray (cam -> via -> ...) with one approximate letter outline.
    """
    dx = via[0] - cam[0]
    dy = via[1] - cam[1]
    dlen = math.hypot(dx, dy)
    if dlen < 1e-6:
        return title_center()
    dx /= dlen
    dy /= dlen

    cx, cy = title_center()
    left, right, top, bottom = TITLE_LETTER_RECTS[letter_index]
    left += cx
    right += cx
    top += cy
    bottom += cy
    candidates: list[tuple[float, float, float]] = []

    if abs(dx) > 1e-9:
        s = (left - cam[0]) / dx
        if s > 0:
            y = cam[1] + dy * s
            if top - 1e-6 <= y <= bottom + 1e-6:
                candidates.append((s, left, y))
        s = (right - cam[0]) / dx
        if s > 0:
            y = cam[1] + dy * s
            if top - 1e-6 <= y <= bottom + 1e-6:
                candidates.append((s, right, y))

    if abs(dy) > 1e-9:
        s = (top - cam[1]) / dy
        if s > 0:
            x = cam[0] + dx * s
            if left - 1e-6 <= x <= right + 1e-6:
                candidates.append((s, x, top))
        s = (bottom - cam[1]) / dy
        if s > 0:
            x = cam[0] + dx * s
            if left - 1e-6 <= x <= right + 1e-6:
                candidates.append((s, x, bottom))

    if not candidates:
        return cx, cy
    s, x, y = min(candidates, key=lambda it: it[0])
    return x + dx * TITLE_LINK_OVERLAP_PX, y + dy * TITLE_LINK_OVERLAP_PX


def sample_camera(offset: float, laps: float) -> tuple[list[str], list[str]]:
    cam_xs: list[str] = []
    cam_ys: list[str] = []
    for i in range(SAMPLES):
        t = i / FRAMES
        cam = camera_state(t, offset, laps)
        cam_xs.append(fmt(cam[0]))
        cam_ys.append(fmt(cam[1]))
    return cam_xs, cam_ys


def sample_camera_path(offset: float, laps: float) -> tuple[list[tuple[float, float]], list[str], float]:
    points: list[tuple[float, float]] = []
    lengths = [0.0]
    total = 0.0
    prev = None
    for i in range(SAMPLES):
        t = i / FRAMES
        point = camera_state(t, offset, laps)
        points.append(point)
        if prev is not None:
            total += math.hypot(point[0] - prev[0], point[1] - prev[1])
            lengths.append(total)
        prev = point
    return points, [fmt(v) for v in lengths], total


def feature_position(feature: dict[str, float], idx: int, t: float) -> tuple[float, float]:
    landmark = route_point(feature["t"])
    dx = landmark[0] - W * 0.51
    dy = landmark[1] - H * 0.56
    length = max(1.0, math.hypot(dx, dy))
    ux = dx / length
    uy = dy / length
    px = landmark[0] + ux * feature["d"] * FEATURE_SPREAD + math.sin(t * 2.1 + idx) * 2.0
    py = landmark[1] + uy * feature["d"] * FEATURE_SPREAD + math.cos(t * 1.7 + idx * 0.8) * 1.7
    return px, py


def sample_feature(feature: dict[str, float], idx: int) -> tuple[list[str], list[str]]:
    feat_xs: list[str] = []
    feat_ys: list[str] = []
    for i in range(SAMPLES):
        t = i / FRAMES
        px, py = feature_position(feature, idx, t)
        feat_xs.append(fmt(px))
        feat_ys.append(fmt(py))
    return feat_xs, feat_ys


def feature_reveal_strength(
    feature: dict[str, float],
    idx: int,
    t: float,
    cam_offset: float,
    cam_weight: float,
    cam_laps: float,
) -> float:
    cam = camera_state(t, cam_offset, cam_laps)
    px, py = feature_position(feature, idx, t)

    center_x, center_y = ellipse_center()
    view_x = center_x - cam[0]
    view_y = center_y - cam[1]
    feat_x = px - cam[0]
    feat_y = py - cam[1]
    view_len = max(1.0, math.hypot(view_x, view_y))
    feat_len = max(1.0, math.hypot(feat_x, feat_y))
    view_cos = (view_x * feat_x + view_y * feat_y) / (view_len * feat_len)
    min_view_cos = math.cos(math.radians(CAMERA_FOV_DEG / 2.0))
    inward_gate = 1.0 if view_cos >= min_view_cos else 0.0

    dist_gate = smoothstep(300, 82, math.hypot(px - cam[0], py - cam[1]))
    forward_delta = (feature["t"] - camera_phase(t, cam_offset, cam_laps) + 1.0) % 1.0
    if forward_delta > 0.5:
        ahead_gate = 0.0
    else:
        ahead_gate = smoothstep(0.00, 0.08, forward_delta) * (1.0 - smoothstep(0.20, 0.34, forward_delta))

    return dist_gate * ahead_gate * inward_gate * cam_weight * cycle_reveal_gate(t)


def sample_feature_reveal_opacities(feature: dict[str, float], idx: int) -> list[str]:
    values: list[str] = []
    state = 0.0
    for i in range(SAMPLES):
        t = i / FRAMES
        target = 0.0
        for cam_offset, cam_weight, cam_laps in zip(CAMERA_OFFSETS, CAMERA_ALPHA, CAMERA_LAPS):
            target = max(target, feature_reveal_strength(feature, idx, t, cam_offset, cam_weight, cam_laps))
        state = max(state, state + (target - state) * 0.18)
        values.append(f"{clamp(state * 1.35, 0.0, 0.94):.3f}")
    return values


def sample_line_opacities(
    feature: dict[str, float],
    idx: int,
    cam_offset: float,
    cam_weight: float,
    cam_laps: float,
) -> list[str]:
    line_opacities: list[str] = []

    for i in range(SAMPLES):
        t = i / FRAMES
        cam = camera_state(t, cam_offset, cam_laps)

        px, py = feature_position(feature, idx, t)

        center_x, center_y = ellipse_center()
        view_x = center_x - cam[0]
        view_y = center_y - cam[1]
        feat_x = px - cam[0]
        feat_y = py - cam[1]
        view_len = max(1.0, math.hypot(view_x, view_y))
        feat_len = max(1.0, math.hypot(feat_x, feat_y))
        view_cos = (view_x * feat_x + view_y * feat_y) / (view_len * feat_len)
        min_view_cos = math.cos(math.radians(CAMERA_FOV_DEG / 2.0))
        inward_gate = 1.0 if view_cos >= min_view_cos else 0.0

        dist_gate = smoothstep(285, 75, math.hypot(px - cam[0], py - cam[1])) * 0.82 * 0.8
        forward_delta = (feature["t"] - camera_phase(t, cam_offset, cam_laps) + 1.0) % 1.0
        if forward_delta > 0.5:
            ahead_gate = 0.0
        else:
            ahead_gate = smoothstep(0.00, 0.06, forward_delta) * (1.0 - smoothstep(0.18, 0.28, forward_delta))

        line_alpha = dist_gate * ahead_gate * inward_gate * cam_weight * cycle_reveal_gate(t)
        line_opacities.append(f"{line_alpha:.3f}")

    return line_opacities


def sample_title_edge_points(
    cam_offset: float,
    letter_index: int,
    cam_laps: float,
) -> tuple[list[str], list[str], list[float]]:
    xs: list[str] = []
    ys: list[str] = []
    gates: list[float] = []
    for i in range(SAMPLES):
        t = i / FRAMES
        cam = camera_state(t, cam_offset, cam_laps)

        letter_center = title_letter_center(letter_index)
        ex, ey = ray_intersect_title_letter(cam, letter_center, letter_index)
        xs.append(fmt(ex))
        ys.append(fmt(ey))
        gates.append(title_letter_visibility_gate(cam, letter_index))
    return xs, ys, gates


def sample_title_line_opacities(
    cam_weight: float,
    cam_idx: int,
    letter_idx: int,
    gates: list[float],
) -> list[str]:
    values: list[str] = []
    for i, gate in enumerate(gates):
        t = i / FRAMES
        pulse = 0.72 + 0.28 * max(0.0, math.sin(t * 3.0 + cam_idx * 1.2 + letter_idx * 0.55))
        values.append(f"{0.34 * cam_weight * pulse * gate * cycle_reveal_gate(t):.3f}")
    return values


def sample_title_reveal_strengths(
    cam_weight: float,
    cam_idx: int,
    letter_idx: int,
    gates: list[float],
) -> list[float]:
    values: list[float] = []
    for i, gate in enumerate(gates[:-1]):
        t = i / FRAMES
        pulse = 0.68 + 0.32 * max(0.0, math.sin(t * 3.2 + cam_idx * 1.1 + letter_idx * 0.7))
        values.append(clamp(gate * pulse * (0.72 + 0.28 * cam_weight) * 0.36 * cycle_reveal_gate(t), 0.0, 0.34))
    return values


def build_route_samples(
    start_t: float,
    span_laps: float,
    steps: int,
) -> tuple[list[tuple[float, float]], list[str], float]:
    points = [route_point((start_t + (i / steps) * span_laps) % 1.0) for i in range(steps + 1)]
    lengths = [0.0]
    total = 0.0
    for i in range(1, len(points)):
        total += math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
        lengths.append(total)
    return points, [fmt(v) for v in lengths], total


def build_svg() -> ET.Element:
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "xmlns:xlink": "http://www.w3.org/1999/xlink",
            "width": fmt(VIEWBOX_W),
            "height": fmt(VIEWBOX_H),
            "viewBox": f"{fmt(VIEWBOX_X)} {fmt(VIEWBOX_Y)} {fmt(VIEWBOX_W)} {fmt(VIEWBOX_H)}",
            "role": "img",
            "aria-label": "Animated SLAM card",
        },
    )

    defs = ET.SubElement(svg, "defs")
    style = ET.SubElement(defs, "style")
    style.text = """
      .route-main { fill: none; stroke: #f0a53a; stroke-width: 4.5; stroke-linecap: round; stroke-linejoin: round; }
      .route-dash { fill: none; stroke: #ffe59d; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; stroke-dasharray: 10 14; }
      .landmark-fill { fill: #ffad32; }
      .landmark-ring { fill: none; stroke: #ffc24f; stroke-width: 1.4; }
      .link { fill: none; stroke: #ffbd45; stroke-width: 1.6; stroke-linecap: round; opacity: 0; }
      .title-link { fill: none; stroke: #ffbd45; stroke-width: 1.35; stroke-linecap: round; opacity: 0; }
      .camera-ring { fill: none; stroke: #ffc24f; stroke-width: 1.8; }
      .camera-core { fill: #ffe0a3; }
    """.strip()
    reveal_filter = ET.SubElement(defs, "filter", {
        "id": "title-reveal-soften",
        "x": "-40%",
        "y": "-40%",
        "width": "180%",
        "height": "180%",
    })
    ET.SubElement(reveal_filter, "feGaussianBlur", {"stdDeviation": "6"})

    camera_samples = [sample_camera(offset, laps) for offset, laps in zip(CAMERA_OFFSETS, CAMERA_LAPS)]

    route_alphas = [1.00, 0.86, 0.76]
    for route_idx, (cam_offset, cam_laps, route_alpha) in enumerate(zip(CAMERA_OFFSETS, CAMERA_LAPS, route_alphas)):
        route_pts, route_reveal_lengths, route_total_len = sample_camera_path(cam_offset, cam_laps)
        route_d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in route_pts)
        route_hidden = f"0 {fmt(route_total_len)}"
        route_dash_values = [f"{length} {fmt(route_total_len)}" for length in route_reveal_lengths]

        route_group = ET.SubElement(svg, "g", {
            "id": f"route-{route_idx}",
            "opacity": f"{route_alpha:.2f}",
        })

        main = ET.SubElement(route_group, "path", {
            "class": "route-main",
            "d": route_d,
            "stroke-dasharray": route_hidden,
        })
        ET.SubElement(main, "animate", {
            "attributeName": "stroke-dasharray",
            "dur": f"{DURATION}s",
            "repeatCount": "indefinite",
            "values": join(route_dash_values),
            "keyTimes": key_times(SAMPLES),
            "calcMode": "linear",
        })

        dash = ET.SubElement(route_group, "path", {
            "class": "route-dash",
            "d": route_d,
            "stroke-dasharray": route_hidden,
        })
        ET.SubElement(dash, "animate", {
            "attributeName": "stroke-dasharray",
            "dur": f"{DURATION}s",
            "repeatCount": "indefinite",
            "values": join(route_dash_values),
            "keyTimes": key_times(SAMPLES),
            "calcMode": "linear",
        })
    feature_samples = []
    for idx, feature in enumerate(FEATURES):
        feature_samples.append((feature, *sample_feature(feature, idx), sample_feature_reveal_opacities(feature, idx)))

    title_mask = ET.SubElement(defs, "mask", {
        "id": "title-reveal-mask",
        "maskUnits": "userSpaceOnUse",
        "maskContentUnits": "userSpaceOnUse",
        "x": fmt(VIEWBOX_X),
        "y": fmt(VIEWBOX_Y),
        "width": fmt(VIEWBOX_W),
        "height": fmt(VIEWBOX_H),
    })
    ET.SubElement(title_mask, "rect", {
        "x": fmt(VIEWBOX_X),
        "y": fmt(VIEWBOX_Y),
        "width": fmt(VIEWBOX_W),
        "height": fmt(VIEWBOX_H),
        "fill": "black",
    })
    mask_group = ET.SubElement(title_mask, "g", {"filter": "url(#title-reveal-soften)"})
    for cam_idx, ((cam_xs, cam_ys), cam_offset, cam_weight, cam_laps) in enumerate(zip(camera_samples, CAMERA_OFFSETS, CAMERA_ALPHA, CAMERA_LAPS)):
        for letter_idx in range(len(TITLE_LETTER_RECTS)):
            edge_xs, edge_ys, title_gates = sample_title_edge_points(cam_offset, letter_idx, cam_laps)
            title_reveal = sample_title_reveal_strengths(cam_weight, cam_idx, letter_idx, title_gates)
            for i, strength in enumerate(title_reveal):
                if strength < TITLE_REVEAL_POINT_THRESHOLD:
                    continue
                start = i / FRAMES
                fade_end = min(0.999, start + TITLE_SPOT_FADE_WINDOW)
                reveal_spot = ET.SubElement(mask_group, "circle", {
                    "cx": edge_xs[i],
                    "cy": edge_ys[i],
                    "r": f"{TITLE_REVEAL_RADIUS:.2f}",
                    "fill": "white",
                    "opacity": "0",
                })
                ET.SubElement(reveal_spot, "animate", {
                    "attributeName": "opacity",
                    "dur": f"{DURATION}s",
                    "repeatCount": "indefinite",
                    "values": f"0;0;{strength:.3f};{strength:.3f}",
                    "keyTimes": f"0;{start:.6f};{fade_end:.6f};1",
                    "calcMode": "linear",
                })
    final_title_reveal = ET.SubElement(mask_group, "rect", {
        "x": fmt(VIEWBOX_X),
        "y": fmt(VIEWBOX_Y),
        "width": fmt(VIEWBOX_W),
        "height": fmt(VIEWBOX_H),
        "fill": "white",
        "opacity": "0",
    })
    ET.SubElement(final_title_reveal, "animate", {
        "attributeName": "opacity",
        "dur": f"{DURATION}s",
        "repeatCount": "indefinite",
        "values": "0;0;0.38;0.72;1",
        "keyTimes": (
            f"0;{TITLE_FINAL_REVEAL_START:.6f};"
            f"{TITLE_FINAL_REVEAL_START + TITLE_FINAL_REVEAL_DURATION * 0.38:.6f};"
            f"{TITLE_FINAL_REVEAL_START + TITLE_FINAL_REVEAL_DURATION * 0.78:.6f};1"
        ),
        "calcMode": "linear",
    })

    for idx, (feature, feat_xs, feat_ys, feat_opacities) in enumerate(feature_samples):
        g = ET.SubElement(svg, "g", {"id": f"landmark-{idx}"})
        r_ring = feature["s"] + 1.5
        r_fill = feature["s"]

        ring = ET.SubElement(g, "circle", {
            "class": "landmark-ring",
            "cx": feat_xs[0],
            "cy": feat_ys[0],
            "r": f"{r_ring:.2f}",
        })
        ET.SubElement(ring, "animate", {
            "attributeName": "cx",
            "dur": f"{DURATION}s",
            "repeatCount": "indefinite",
            "values": join(feat_xs),
            "keyTimes": key_times(SAMPLES),
            "calcMode": "linear",
        })
        ET.SubElement(ring, "animate", {
            "attributeName": "cy",
            "dur": f"{DURATION}s",
            "repeatCount": "indefinite",
            "values": join(feat_ys),
            "keyTimes": key_times(SAMPLES),
            "calcMode": "linear",
        })
        ET.SubElement(ring, "animate", {
            "attributeName": "opacity",
            "dur": f"{DURATION}s",
            "repeatCount": "indefinite",
            "values": join(feat_opacities),
            "keyTimes": key_times(SAMPLES),
            "calcMode": "linear",
        })

        fill = ET.SubElement(g, "circle", {
            "class": "landmark-fill",
            "cx": feat_xs[0],
            "cy": feat_ys[0],
            "r": f"{r_fill:.2f}",
        })
        ET.SubElement(fill, "animate", {
            "attributeName": "cx",
            "dur": f"{DURATION}s",
            "repeatCount": "indefinite",
            "values": join(feat_xs),
            "keyTimes": key_times(SAMPLES),
            "calcMode": "linear",
        })
        ET.SubElement(fill, "animate", {
            "attributeName": "cy",
            "dur": f"{DURATION}s",
            "repeatCount": "indefinite",
            "values": join(feat_ys),
            "keyTimes": key_times(SAMPLES),
            "calcMode": "linear",
        })
        ET.SubElement(fill, "animate", {
            "attributeName": "opacity",
            "dur": f"{DURATION}s",
            "repeatCount": "indefinite",
            "values": join(feat_opacities),
            "keyTimes": key_times(SAMPLES),
            "calcMode": "linear",
        })

    for idx, (feature, feat_xs, feat_ys, feat_opacities) in enumerate(feature_samples):
        for cam_idx, ((cam_xs, cam_ys), cam_offset, cam_weight, cam_laps) in enumerate(zip(camera_samples, CAMERA_OFFSETS, CAMERA_ALPHA, CAMERA_LAPS)):
            line_opacities = sample_line_opacities(feature, idx, cam_offset, cam_weight, cam_laps)
            # Segment A: camera -> landmark
            seg_a = ET.SubElement(svg, "line", {
                "class": "link",
                "x1": cam_xs[0],
                "y1": cam_ys[0],
                "x2": feat_xs[0],
                "y2": feat_ys[0],
            })
            ET.SubElement(seg_a, "animate", {
                "attributeName": "x1",
                "dur": f"{DURATION}s",
                "repeatCount": "indefinite",
                "values": join(cam_xs),
                "keyTimes": key_times(SAMPLES),
                "calcMode": "linear",
            })
            ET.SubElement(seg_a, "animate", {
                "attributeName": "y1",
                "dur": f"{DURATION}s",
                "repeatCount": "indefinite",
                "values": join(cam_ys),
                "keyTimes": key_times(SAMPLES),
                "calcMode": "linear",
            })
            ET.SubElement(seg_a, "animate", {
                "attributeName": "x2",
                "dur": f"{DURATION}s",
                "repeatCount": "indefinite",
                "values": join(feat_xs),
                "keyTimes": key_times(SAMPLES),
                "calcMode": "linear",
            })
            ET.SubElement(seg_a, "animate", {
                "attributeName": "y2",
                "dur": f"{DURATION}s",
                "repeatCount": "indefinite",
                "values": join(feat_ys),
                "keyTimes": key_times(SAMPLES),
                "calcMode": "linear",
            })
            ET.SubElement(seg_a, "animate", {
                "attributeName": "opacity",
                "dur": f"{DURATION}s",
                "repeatCount": "indefinite",
                "values": join(line_opacities),
                "keyTimes": key_times(SAMPLES),
                "calcMode": "linear",
            })

    for cam_idx, ((cam_xs, cam_ys), cam_offset, cam_weight, cam_laps) in enumerate(zip(camera_samples, CAMERA_OFFSETS, CAMERA_ALPHA, CAMERA_LAPS)):
        for letter_idx in range(len(TITLE_LETTER_RECTS)):
            edge_xs, edge_ys, title_gates = sample_title_edge_points(cam_offset, letter_idx, cam_laps)
            title_opacities = sample_title_line_opacities(cam_weight, cam_idx, letter_idx, title_gates)
            title_line = ET.SubElement(svg, "line", {
                "class": "title-link",
                "x1": cam_xs[0],
                "y1": cam_ys[0],
                "x2": edge_xs[0],
                "y2": edge_ys[0],
            })
            ET.SubElement(title_line, "animate", {
                "attributeName": "x1",
                "dur": f"{DURATION}s",
                "repeatCount": "indefinite",
                "values": join(cam_xs),
                "keyTimes": key_times(SAMPLES),
                "calcMode": "linear",
            })
            ET.SubElement(title_line, "animate", {
                "attributeName": "y1",
                "dur": f"{DURATION}s",
                "repeatCount": "indefinite",
                "values": join(cam_ys),
                "keyTimes": key_times(SAMPLES),
                "calcMode": "linear",
            })
            ET.SubElement(title_line, "animate", {
                "attributeName": "x2",
                "dur": f"{DURATION}s",
                "repeatCount": "indefinite",
                "values": join(edge_xs),
                "keyTimes": key_times(SAMPLES),
                "calcMode": "linear",
            })
            ET.SubElement(title_line, "animate", {
                "attributeName": "y2",
                "dur": f"{DURATION}s",
                "repeatCount": "indefinite",
                "values": join(edge_ys),
                "keyTimes": key_times(SAMPLES),
                "calcMode": "linear",
            })
            ET.SubElement(title_line, "animate", {
                "attributeName": "opacity",
                "dur": f"{DURATION}s",
                "repeatCount": "indefinite",
                "values": join(title_opacities),
                "keyTimes": key_times(SAMPLES),
                "calcMode": "linear",
            })

    for cam_idx, ((cam_xs, cam_ys), cam_scale, cam_alpha) in enumerate(zip(camera_samples, CAMERA_SCALES, CAMERA_ALPHA)):
        camera = ET.SubElement(svg, "g", {"id": f"camera-{cam_idx}"})
        cam_ring = ET.SubElement(camera, "circle", {
            "class": "camera-ring",
            "cx": cam_xs[0],
            "cy": cam_ys[0],
            "r": f"{10.5 * cam_scale:.2f}",
            "opacity": f"{cam_alpha:.2f}",
        })
        ET.SubElement(cam_ring, "animate", {
            "attributeName": "cx",
            "dur": f"{DURATION}s",
            "repeatCount": "indefinite",
            "values": join(cam_xs),
            "keyTimes": key_times(SAMPLES),
            "calcMode": "linear",
        })
        ET.SubElement(cam_ring, "animate", {
            "attributeName": "cy",
            "dur": f"{DURATION}s",
            "repeatCount": "indefinite",
            "values": join(cam_ys),
            "keyTimes": key_times(SAMPLES),
            "calcMode": "linear",
        })

        cam_core = ET.SubElement(camera, "circle", {
            "class": "camera-core",
            "cx": cam_xs[0],
            "cy": cam_ys[0],
            "r": f"{4.2 * cam_scale:.2f}",
            "opacity": f"{max(0.34, cam_alpha):.2f}",
        })
        ET.SubElement(cam_core, "animate", {
            "attributeName": "cx",
            "dur": f"{DURATION}s",
            "repeatCount": "indefinite",
            "values": join(cam_xs),
            "keyTimes": key_times(SAMPLES),
            "calcMode": "linear",
        })
        ET.SubElement(cam_core, "animate", {
            "attributeName": "cy",
            "dur": f"{DURATION}s",
            "repeatCount": "indefinite",
            "values": join(cam_ys),
            "keyTimes": key_times(SAMPLES),
            "calcMode": "linear",
        })

    title_attrs = {
        "x": fmt(title_center()[0]),
        "y": fmt(title_center()[1]),
        "text-anchor": "middle",
        "dominant-baseline": "middle",
        "font-family": "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
        "font-size": "76",
        "font-weight": "800",
        "letter-spacing": "0.18em",
    }
    title_base = ET.SubElement(svg, "text", {
        **title_attrs,
        "fill": "#ffe0a6",
        "opacity": f"{TITLE_BASE_OPACITY:.2f}",
    })
    title_base.text = "SLAM"

    title_reveal = ET.SubElement(svg, "text", {
        **title_attrs,
        "fill": "#fff2cc",
        "opacity": f"{TITLE_REVEAL_OPACITY:.2f}",
        "mask": "url(#title-reveal-mask)",
    })
    title_reveal.text = "SLAM"

    return svg


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the animated SLAM SVG asset.")
    parser.add_argument(
        "-o",
        "--output",
        default="assets/slam_ba/slam_ba.svg",
        help="Output SVG path",
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    svg = build_svg()
    ET.ElementTree(svg).write(out, encoding="utf-8", xml_declaration=True)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
