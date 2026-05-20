#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from xml.etree import ElementTree as ET


W = 1200
H = 760
FRAMES = 200
SAMPLES = FRAMES + 1
DURATION = 20.0

FEATURES = [
    {"t": 0.05, "d": 74, "s": 5.4},
    {"t": 0.12, "d": 58, "s": 3.6},
    {"t": 0.20, "d": 66, "s": 4.9},
    {"t": 0.31, "d": 60, "s": 3.1},
    {"t": 0.41, "d": 72, "s": 5.6},
    {"t": 0.53, "d": 64, "s": 3.8},
    {"t": 0.63, "d": 78, "s": 5.8},
    {"t": 0.74, "d": 56, "s": 3.2},
    {"t": 0.82, "d": 69, "s": 4.7},
    {"t": 0.90, "d": 61, "s": 3.5},
    {"t": 0.97, "d": 76, "s": 5.2},
    {"t": 0.16, "d": -30, "s": 2.8},
    {"t": 0.34, "d": -38, "s": 3.5},
    {"t": 0.49, "d": -26, "s": 2.6},
    {"t": 0.67, "d": -41, "s": 3.8},
    {"t": 0.86, "d": -33, "s": 3.1},
]

CAMERA_OFFSETS = [-0.042, 0.0, 0.065]
CAMERA_SCALES = [0.62, 1.0, 0.82]
CAMERA_ALPHA = [0.70, 1.0, 0.70]


def clamp(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def route_point(t: float) -> tuple[float, float]:
    cx = W * 0.505
    cy = H * 0.53
    a = W * 0.225
    b = H * 0.235
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


def camera_state(t: float, offset: float) -> tuple[float, float]:
    return route_point((t + offset + 1.0) % 1.0)


def sample_camera(offset: float) -> tuple[list[str], list[str]]:
    cam_xs: list[str] = []
    cam_ys: list[str] = []
    for i in range(SAMPLES):
        t = i / FRAMES
        cam = camera_state(t, offset)
        cam_xs.append(fmt(cam[0]))
        cam_ys.append(fmt(cam[1]))
    return cam_xs, cam_ys


def sample_feature(feature: dict[str, float], idx: int):
    feat_xs: list[str] = []
    feat_ys: list[str] = []
    feat_opacities: list[str] = []

    for i in range(SAMPLES):
        t = i / FRAMES

        landmark = route_point(feature["t"])
        dx = landmark[0] - W * 0.51
        dy = landmark[1] - H * 0.56
        length = max(1.0, math.hypot(dx, dy))
        ux = dx / length
        uy = dy / length

        px = landmark[0] + ux * feature["d"] + math.sin(t * 2.1 + idx) * 2.4
        py = landmark[1] + uy * feature["d"] + math.cos(t * 1.7 + idx * 0.8) * 2.0

        pulse = 0.45 + 0.55 * max(0.0, math.sin(t * 3.0 + idx * 0.7))
        alpha = 0.4 + pulse * 0.55

        feat_xs.append(fmt(px))
        feat_ys.append(fmt(py))
        feat_opacities.append(f"{alpha:.3f}")

    return feat_xs, feat_ys, feat_opacities


def sample_line_opacities(
    feature: dict[str, float],
    idx: int,
    cam_offset: float,
    cam_weight: float,
) -> list[str]:
    line_opacities: list[str] = []

    for i in range(SAMPLES):
        t = i / FRAMES
        cam = camera_state(t, cam_offset)

        landmark = route_point(feature["t"])
        dx = landmark[0] - W * 0.51
        dy = landmark[1] - H * 0.56
        length = max(1.0, math.hypot(dx, dy))
        ux = dx / length
        uy = dy / length
        px = landmark[0] + ux * feature["d"] + math.sin(t * 2.1 + idx) * 2.4
        py = landmark[1] + uy * feature["d"] + math.cos(t * 1.7 + idx * 0.8) * 2.0

        dist_gate = smoothstep(340, 90, math.hypot(px - cam[0], py - cam[1])) * 0.82 * 0.8
        forward_delta = (feature["t"] - ((t + cam_offset + 1.0) % 1.0) + 1.0) % 1.0
        if forward_delta > 0.5:
            ahead_gate = 0.0
        else:
            ahead_gate = smoothstep(0.00, 0.06, forward_delta) * (1.0 - smoothstep(0.18, 0.28, forward_delta))

        line_alpha = dist_gate * ahead_gate * cam_weight
        line_opacities.append(f"{line_alpha:.3f}")

    return line_opacities


def build_svg() -> ET.Element:
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "xmlns:xlink": "http://www.w3.org/1999/xlink",
            "width": "723.47",
            "height": "484.24",
            "viewBox": "256.74 153.15 723.47 484.24",
            "role": "img",
            "aria-label": "Animated SLAM card",
        },
    )

    defs = ET.SubElement(svg, "defs")
    style = ET.SubElement(defs, "style")
    style.text = """
      .route-shadow { fill: none; stroke: #1f1b18; stroke-width: 12; stroke-linecap: round; stroke-linejoin: round; }
      .route-main { fill: none; stroke: #e0a45f; stroke-width: 4.5; stroke-linecap: round; stroke-linejoin: round; }
      .route-dash { fill: none; stroke: #fff5d8; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; stroke-dasharray: 10 14; }
      .landmark-fill { fill: #f4b35f; }
      .landmark-ring { fill: none; stroke: #ffd27e; stroke-width: 1.4; }
      .link { fill: none; stroke: #ffcf7a; stroke-width: 1.6; stroke-linecap: round; opacity: 0; }
      .camera-ring { fill: none; stroke: #ffd27e; stroke-width: 1.8; }
      .camera-core { fill: #fff0cc; }
    """.strip()

    route_pts = [route_point(i / 360.0) for i in range(361)]
    route_d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in route_pts) + " Z"

    shadow = ET.SubElement(svg, "path", {"class": "route-shadow", "d": route_d})
    main = ET.SubElement(svg, "path", {"class": "route-main", "d": route_d})
    dash = ET.SubElement(svg, "path", {"class": "route-dash", "d": route_d})
    ET.SubElement(dash, "animate", {
        "attributeName": "stroke-dashoffset",
        "dur": f"{DURATION}s",
        "repeatCount": "indefinite",
        "values": "0;-220",
    })

    feature_samples = []
    for idx, feature in enumerate(FEATURES):
        feature_samples.append((feature, *sample_feature(feature, idx)))

    camera_samples = [sample_camera(offset) for offset in CAMERA_OFFSETS]

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
        for cam_idx, ((cam_xs, cam_ys), cam_offset, cam_weight) in enumerate(zip(camera_samples, CAMERA_OFFSETS, CAMERA_ALPHA)):
            line_opacities = sample_line_opacities(feature, idx, cam_offset, cam_weight)
            line = ET.SubElement(svg, "line", {
                "class": "link",
                "x1": feat_xs[0],
                "y1": feat_ys[0],
                "x2": cam_xs[0],
                "y2": cam_ys[0],
            })
            ET.SubElement(line, "animate", {
                "attributeName": "x1",
                "dur": f"{DURATION}s",
                "repeatCount": "indefinite",
                "values": join(feat_xs),
                "keyTimes": key_times(SAMPLES),
                "calcMode": "linear",
            })
            ET.SubElement(line, "animate", {
                "attributeName": "y1",
                "dur": f"{DURATION}s",
                "repeatCount": "indefinite",
                "values": join(feat_ys),
                "keyTimes": key_times(SAMPLES),
                "calcMode": "linear",
            })
            ET.SubElement(line, "animate", {
                "attributeName": "x2",
                "dur": f"{DURATION}s",
                "repeatCount": "indefinite",
                "values": join(cam_xs),
                "keyTimes": key_times(SAMPLES),
                "calcMode": "linear",
            })
            ET.SubElement(line, "animate", {
                "attributeName": "y2",
                "dur": f"{DURATION}s",
                "repeatCount": "indefinite",
                "values": join(cam_ys),
                "keyTimes": key_times(SAMPLES),
                "calcMode": "linear",
            })
            ET.SubElement(line, "animate", {
                "attributeName": "opacity",
                "dur": f"{DURATION}s",
                "repeatCount": "indefinite",
                "values": join(line_opacities),
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

    title = ET.SubElement(svg, "text", {
        "x": "618.47",
        "y": "395.27",
        "text-anchor": "middle",
        "dominant-baseline": "middle",
        "font-family": "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
        "font-size": "76",
        "font-weight": "800",
        "letter-spacing": "0.18em",
        "fill": "#fff7e3",
        "opacity": "0.52",
    })
    title.text = "SLAM"

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
