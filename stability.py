"""
Static stability analysis: center of mass, support polygon (convex hull of
stance feet, ground-projected), static stability margin (distance from CoM
projection to nearest support-polygon edge).
"""
import numpy as np

GAIT_STANCE_LEGS = {
    "Stand (all 4 planted)": ["FR", "FL", "RR", "RL"],
    "Walk — phase A (RR lifted)": ["FR", "FL", "RL"],
    "Walk — phase B (FL lifted)": ["FR", "RR", "RL"],
    "Trot — diagonal pair A (FR+RL stance)": ["FR", "RL"],
    "Trot — diagonal pair B (FL+RR stance)": ["FL", "RR"],
    "One-leg disturbance (FR lifted)": ["FL", "RR", "RL"],
}


def hip_mount_points(body_length, body_width):
    hl, hw = body_length / 2, body_width / 2
    return {
        "FR": np.array([hl * 0.75, -hw]),
        "FL": np.array([hl * 0.75, hw]),
        "RR": np.array([-hl * 0.75, -hw]),
        "RL": np.array([-hl * 0.75, hw]),
    }


def foot_ground_points(body_length, body_width, hip_offset, has_abad, abad_shift=0.0):
    """Ground-projected foot (x,y) per leg, standing neutral stance.
    If has_abad is False, foot y is RIGIDLY fixed by hip_offset — the robot
    cannot actively shift it (this is the key 8-DOF lateral limitation).
    If has_abad is True, abad_shift (m) represents an achievable active
    lateral foot adjustment from the ab/ad joint.
    """
    hips = hip_mount_points(body_length, body_width)
    pts = {}
    for name, (hx, hy) in hips.items():
        sign = 1 if hy > 0 else -1
        y_extra = (hip_offset + (abad_shift if has_abad else 0.0)) * sign
        pts[name] = np.array([hx, hy + y_extra])
    return pts


def convex_hull_2d(points):
    """Simple 2D convex hull (Andrew's monotone chain). points: (N,2) array."""
    pts = sorted(map(tuple, points))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def point_in_polygon(pt, poly):
    """Ray casting."""
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def point_to_segment_dist(p, a, b):
    p, a, b = np.array(p), np.array(a), np.array(b)
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-15), 0, 1)
    proj = a + t * ab
    return np.linalg.norm(p - proj)


def stability_margin(com_xy, stance_points_dict):
    """Returns (margin_m, hull_points, is_stable).
    margin: distance from CoM to nearest polygon edge. Positive if CoM is
    inside the support polygon (stable), negative if outside (tipping).
    With fewer than 3 stance points, the robot has no support polygon at all
    (statically unsupported / must rely on dynamic balance) — margin is
    reported as None in that case.
    """
    pts = np.array(list(stance_points_dict.values()))
    if len(pts) < 3:
        return None, pts, False
    hull = convex_hull_2d(pts)
    inside = point_in_polygon(com_xy, hull)
    n = len(hull)
    dists = [point_to_segment_dist(com_xy, hull[i], hull[(i + 1) % n]) for i in range(n)]
    min_dist = min(dists)
    margin = min_dist if inside else -min_dist
    return margin, hull, inside
