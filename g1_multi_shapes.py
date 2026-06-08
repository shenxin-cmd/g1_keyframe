"""
G1 右手多形状轨迹 — 闭合路径参数化库
所有形状在局部 (y, z) 平面生成，再映射到世界坐标 (x, y, z)。
"""

from __future__ import annotations

import numpy as np

SHAPE_NAMES = [
    "circle",
    "ellipse",
    "square",
    "rectangle",
    "triangle",
    "triangle_down",
    "pentagon",
    "star",
    "heart",
    "wave",
    "diamond",
    "random_polygon",
    "random_spline",
]

# 同心递减轨迹常用形状（CLI 别名见 g1_concentric_traj_gen.py）
CONCENTRIC_SHAPE_NAMES = [
    "circle",
    "ellipse",
    "triangle",
    "triangle_down",
    "square",
    "rectangle",
    "star",
    "pentagon",
    "random_polygon",
    "random_spline",
]


def _to_world(local_yz: np.ndarray, center: np.ndarray, scale: float,
              theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    yl = local_yz[:, 0]
    zl = local_yz[:, 1]
    y = center[1] + scale * (yl * c - zl * s)
    z = center[2] + scale * (yl * s + zl * c)
    x = np.full(len(local_yz), center[0], dtype=np.float64)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _resample_polyline(pts: list, N: int) -> np.ndarray:
    arr = np.array(pts, dtype=np.float64)
    if len(arr) < 2:
        arr = np.vstack([arr, arr])
    if np.linalg.norm(arr[0] - arr[-1]) > 1e-9:
        arr = np.vstack([arr, arr[0:1]])
    seg_lens = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    total = seg_lens.sum()
    if total < 1e-9:
        return np.repeat(arr[:1], N, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    targets = np.linspace(0, total, N, endpoint=False)
    out = []
    j = 0
    for t in targets:
        while j < len(seg_lens) - 1 and cum[j + 1] < t:
            j += 1
        seg_t = (t - cum[j]) / max(seg_lens[j], 1e-12)
        p = arr[j] * (1 - seg_t) + arr[j + 1] * seg_t
        out.append(p)
    return np.array(out)


def _arc_corner_local(pts: list, yc: float, zc: float, r: float,
                      angle_start: float, n: int):
    for a in np.linspace(angle_start, angle_start - np.pi / 2, n, endpoint=False):
        pts.append([yc + r * np.cos(a), zc + r * np.sin(a)])


def _rounded_rect_local(half_y: float, half_z: float, N: int,
                        corner_r_frac: float = 0.15) -> np.ndarray:
    Ly, Lz = half_y, half_z
    r = min(corner_r_frac * 2 * min(Ly, Lz), min(Ly, Lz) * 0.45)
    seg_len = 2 * Ly - 2 * r + 2 * Lz - 2 * r
    arc_len = 4 * (np.pi / 2) * r
    total = 2 * (2 * Ly - 2 * r) + 2 * (2 * Lz - 2 * r) + arc_len
    N_seg = max(2, round(N * (2 * Ly - 2 * r) / total))
    N_arc = max(2, round(N * (np.pi / 2 * r) / total))
    pts = []

    def seg(y0, z0, y1, z1, n):
        for y, z in zip(np.linspace(y0, y1, n, endpoint=False),
                        np.linspace(z0, z1, n, endpoint=False)):
            pts.append([y, z])

    seg(Ly - r, Lz, -Ly + r, Lz, N_seg)
    _arc_corner_local(pts, -Ly + r, Lz - r, r, np.pi / 2, N_arc)
    seg(-Ly, Lz - r, -Ly, -Lz + r, N_seg)
    _arc_corner_local(pts, -Ly + r, -Lz + r, r, 0.0, N_arc)
    seg(-Ly + r, -Lz, Ly - r, -Lz, N_seg)
    _arc_corner_local(pts, Ly - r, -Lz + r, r, -np.pi / 2, N_arc)
    seg(Ly, -Lz + r, Ly, Lz - r, N_seg)
    _arc_corner_local(pts, Ly - r, Lz - r, r, np.pi, N_arc)
    return _resample_polyline(pts, N)


def make_circle_local(N: int) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.stack([np.sin(t), np.cos(t)], axis=1)


def make_ellipse_local(N: int, aspect: float = 1.55) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.stack([aspect * np.sin(t), np.cos(t)], axis=1)


def make_square_local(N: int) -> np.ndarray:
    return _rounded_rect_local(1.0, 1.0, N)


def make_rectangle_local(N: int, aspect: float = 1.4) -> np.ndarray:
    return _rounded_rect_local(1.0 * aspect, 1.0, N)


def make_triangle_local(N: int) -> np.ndarray:
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 4, endpoint=True)
    pts = [[np.cos(a), np.sin(a)] for a in angles]
    return _resample_polyline(pts, N)


def make_triangle_down_local(N: int) -> np.ndarray:
    angles = np.linspace(-np.pi / 2, -np.pi / 2 + 2 * np.pi, 4, endpoint=True)
    pts = [[np.cos(a), np.sin(a)] for a in angles]
    return _resample_polyline(pts, N)


def make_polygon_local(N: int, sides: int) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, sides + 1, endpoint=True)
    pts = [[np.cos(a), np.sin(a)] for a in angles]
    return _resample_polyline(pts, N)


def make_pentagon_local(N: int) -> np.ndarray:
    return make_polygon_local(N, 5)


def make_star_local(N: int, inner_ratio: float = 0.45) -> np.ndarray:
    pts = []
    for k in range(10):
        ang = np.pi / 2 + k * np.pi / 5
        rad = 1.0 if k % 2 == 0 else inner_ratio
        pts.append([rad * np.cos(ang), rad * np.sin(ang)])
    pts.append(pts[0])
    return _resample_polyline(pts, N)


def make_heart_local(N: int) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    pts = np.stack([x, y], axis=1)
    pts -= pts.mean(axis=0)
    pts /= np.max(np.abs(pts)) + 1e-9
    return pts


def make_wave_local(N: int, n_lobes: float = 3.0) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    y = np.sin(n_lobes * t)
    z = np.cos(t)
    pts = np.stack([y, z], axis=1)
    pts /= np.max(np.linalg.norm(pts, axis=1)) + 1e-9
    return pts


def make_diamond_local(N: int) -> np.ndarray:
    pts = [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 1]]
    return _resample_polyline(pts, N)


def make_random_polygon_local(N: int, rng: np.random.Generator,
                            n_sides: int | None = None) -> np.ndarray:
    k = n_sides or int(rng.integers(5, 9))
    angles = np.sort(rng.uniform(0, 2 * np.pi, k))
    radii = rng.uniform(0.6, 1.0, k)
    pts = [[r * np.cos(a), r * np.sin(a)] for a, r in zip(angles, radii)]
    pts.append(pts[0])
    return _resample_polyline(pts, N)


def make_random_spline_local(N: int, rng: np.random.Generator,
                             n_ctrl: int = 8) -> np.ndarray:
    ctrl = rng.uniform(-1, 1, (n_ctrl, 2))
    closed = np.vstack([ctrl, ctrl[0:1]])
    t = np.linspace(0, 1, len(closed), endpoint=False)
    t_new = np.linspace(0, 1, N, endpoint=False)
    y = np.interp(t_new, t, closed[:, 0])
    z = np.interp(t_new, t, closed[:, 1])
    pts = np.stack([y, z], axis=1)
    pts -= pts.mean(axis=0)
    mx = np.max(np.linalg.norm(pts, axis=1))
    if mx > 1e-9:
        pts /= mx
    return pts


def make_shape_local(shape_name: str, N: int, rng: np.random.Generator) -> np.ndarray:
    """单位局部折线，用于计算尺度/余量。"""
    name = shape_name.lower()
    if name == "circle":
        return make_circle_local(N)
    if name == "ellipse":
        return make_ellipse_local(N)
    if name == "square":
        return make_square_local(N)
    if name == "rectangle":
        return make_rectangle_local(N)
    if name == "triangle":
        return make_triangle_local(N)
    if name in ("triangle_down", "inverted_triangle", "倒三角"):
        return make_triangle_down_local(N)
    if name == "pentagon":
        return make_pentagon_local(N)
    if name == "star":
        return make_star_local(N)
    if name == "heart":
        return make_heart_local(N)
    if name == "wave":
        return make_wave_local(N)
    if name == "diamond":
        return make_diamond_local(N)
    if name == "random_polygon":
        return make_random_polygon_local(N, rng)
    if name == "random_spline":
        return make_random_spline_local(N, rng)
    raise ValueError(f"未知形状: {shape_name}")


def make_shape_waypoints(
    shape_name: str,
    center: np.ndarray,
    scale: float,
    theta: float,
    N: int = 250,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    local = make_shape_local(shape_name, N, rng)
    return _to_world(local, np.asarray(center, dtype=np.float64), scale, theta)
