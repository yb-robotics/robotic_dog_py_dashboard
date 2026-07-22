"""
Core leg kinematics engine for the quadruped dashboard.
Standard 3-DOF leg: ab/ad (hip roll) -> hip pitch -> knee pitch.
For 2-DOF (8-DOF total robot) legs, ab/ad is fixed at 0 (no roll joint).
"""
import numpy as np


def leg_fk(theta1, theta2, theta3, l1, l2, l3):
    """Forward kinematics: joint angles -> foot position in hip frame.
    theta1: ab/ad (roll about x). theta2: hip pitch. theta3: knee pitch.
    l1: ab/ad link offset. l2: femur (upper leg). l3: tibia (lower leg).
    """
    x = l2 * np.sin(theta2) + l3 * np.sin(theta2 + theta3)
    z = -(l2 * np.cos(theta2) + l3 * np.cos(theta2 + theta3))
    y0 = l1
    y = y0 * np.cos(theta1) - z * np.sin(theta1)
    z2 = y0 * np.sin(theta1) + z * np.cos(theta1)
    return np.array([x, y, z2])


def leg_ik(x, y, z, l1, l2, l3, knee_forward=True):
    """Inverse kinematics: foot position (hip frame) -> joint angles.
    Returns (theta1, theta2, theta3) or None if unreachable.
    """
    d_yz = np.sqrt(y ** 2 + z ** 2)
    if d_yz < l1:
        return None
    l = np.sqrt(max(d_yz ** 2 - l1 ** 2, 0.0))
    zp = -l  # planar (pre ab/ad-rotation) z, always negative (leg points down)
    theta1 = np.arctan2(z, y) - np.arctan2(zp, l1) if l1 > 0 else np.arctan2(z, y) - np.arctan2(zp, 0.0)

    d = np.sqrt(x ** 2 + l ** 2)
    if d > (l2 + l3) or d < abs(l2 - l3) or d == 0:
        return None

    cos_theta3 = (d ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = np.arccos(cos_theta3)
    if not knee_forward:
        theta3 = -theta3

    alpha = np.arctan2(x, l)
    beta = np.arctan2(l3 * np.sin(theta3), l2 + l3 * np.cos(theta3))
    theta2 = alpha - beta

    return theta1, theta2, theta3


def leg_jacobian(theta1, theta2, theta3, l1, l2, l3):
    """Analytic Cartesian Jacobian ``J = d[x, y, z]/d[q_ab, q_hip, q_knee]``.

    This is the same matrix used for two important checks in the dashboard:
    ``v_foot = J q_dot`` and ``tau_joint = J.T F_foot``.  An analytic form is
    used instead of finite differences so the force/torque calculation is
    repeatable and does not depend on an arbitrary numerical step size.
    """
    q1, q2, q3 = theta1, theta2, theta3
    x = l2 * np.sin(q2) + l3 * np.sin(q2 + q3)
    z_planar = -(l2 * np.cos(q2) + l3 * np.cos(q2 + q3))
    dx2 = l2 * np.cos(q2) + l3 * np.cos(q2 + q3)
    dx3 = l3 * np.cos(q2 + q3)
    dz2 = l2 * np.sin(q2) + l3 * np.sin(q2 + q3)
    dz3 = l3 * np.sin(q2 + q3)
    c, s = np.cos(q1), np.sin(q1)

    return np.array([
        [0.0, dx2, dx3],
        [-l1 * s - z_planar * c, -dz2 * s, -dz3 * s],
        [l1 * c - z_planar * s, dz2 * c, dz3 * c],
    ])


def jacobian_condition_number(theta1, theta2, theta3, l1, l2, l3):
    """Return ``cond(J)``; a large value means near-singular motion/forces."""
    return float(np.linalg.cond(leg_jacobian(theta1, theta2, theta3, l1, l2, l3)))


def compute_workspace_boundary(l1, l2, l3, n_points=72):
    """Computes the sagittal plane workspace boundary for the leg."""
    r_max = l2 + l3
    r_min = abs(l2 - l3)
    
    boundary_points = []
    # trace outer boundary
    for angle in np.linspace(0, 2 * np.pi, n_points // 2):
        boundary_points.append((r_max * np.sin(angle), -r_max * np.cos(angle)))
    # trace inner boundary
    for angle in np.linspace(2 * np.pi, 0, n_points // 2):
        boundary_points.append((r_min * np.sin(angle), -r_min * np.cos(angle)))
        
    return {
        'r_max': r_max,
        'r_min': r_min,
        'boundary_points': boundary_points
    }


def manipulability_index(theta1, theta2, theta3, l1, l2, l3):
    """Computes manipulability index w = sqrt(det(J @ J.T))."""
    J = leg_jacobian(theta1, theta2, theta3, l1, l2, l3)
    det = np.linalg.det(J @ J.T)
    if det <= 0:
        return 0.0
    return float(np.sqrt(det))


def damped_least_squares_ik(x, y, z, l1, l2, l3, q_init=None, damping=0.01, max_iter=50, tol=1e-4):
    """Iterative IK using Damped Least Squares: Δq = J^T(JJ^T + λ²I)^{-1} * e"""
    if q_init is None:
        q = np.array([0.0, 0.0, np.pi/4])
    else:
        q = np.array(q_init, dtype=float)
        
    target_pos = np.array([x, y, z])
    
    for _ in range(max_iter):
        current_pos = leg_fk(q[0], q[1], q[2], l1, l2, l3)
        error = target_pos - current_pos
        if np.linalg.norm(error) < tol:
            return float(q[0]), float(q[1]), float(q[2])
            
        J = leg_jacobian(q[0], q[1], q[2], l1, l2, l3)
        J_T = J.T
        lambda_I = (damping ** 2) * np.eye(3)
        inv_term = np.linalg.inv(J @ J_T + lambda_I)
        delta_q = J_T @ inv_term @ error
        
        q += delta_q
        
    return None


def derive_dimensions(dof_total, standing_height_m, femur_fraction=0.52,
                      neutral_knee_deg=55.0):
    """Given total DOF (8 or 12) and desired approx standing height,
    derive a self-consistent set of link lengths + body envelope.

    The target height is hip-to-ground height at a neutral crouched posture,
    not an arbitrary fraction of the link sum.  For the selected knee bend,
    the law of cosines gives the hip-to-foot distance::

        h = sqrt(l2^2 + l3^2 + 2*l2*l3*cos(q_knee))

    The neutral hip angle is then chosen so that this vector points vertically
    down.  This permits deliberately unequal femur/shank dimensions and is
    much closer to how a real student build is dimensioned.
    - l1 (ab/ad offset), only present for 12-DOF (3 DOF/leg): 0.12 * standing_height
      For 8-DOF (2 DOF/leg, no ab/ad): l1 = 0 (rigid hip mount, no roll joint)
    - Body length (nose-to-tail hip spacing) = 1.15 * standing_height
    - Body width (hip-to-hip) = 0.55 * standing_height + 2*l1
    """
    femur_fraction = float(np.clip(femur_fraction, 0.35, 0.65))
    knee = np.radians(float(np.clip(neutral_knee_deg, 15.0, 120.0)))
    shank_fraction = 1.0 - femur_fraction
    posture_factor = np.sqrt(femur_fraction**2 + shank_fraction**2
                             + 2 * femur_fraction * shank_fraction * np.cos(knee))
    max_extension = standing_height_m / posture_factor
    l2 = max_extension * femur_fraction
    l3 = max_extension * shank_fraction
    neutral_hip = -np.arctan2(l3 * np.sin(knee), l2 + l3 * np.cos(knee))
    dof_per_leg = dof_total / 4
    has_abad = dof_per_leg >= 3
    l1 = 0.12 * standing_height_m if has_abad else 0.0
    body_length = 1.15 * standing_height_m
    body_width = 0.55 * standing_height_m + 2 * l1
    return {
        "l1": l1, "l2": l2, "l3": l3,
        "body_length": body_length, "body_width": body_width,
        "max_extension": max_extension, "has_abad": has_abad,
        "dof_per_leg": dof_per_leg,
        "neutral_knee_rad": knee, "neutral_hip_rad": neutral_hip,
        "neutral_knee_deg": neutral_knee_deg,
        "femur_fraction": femur_fraction,
        "posture_factor": posture_factor,
    }


def estimate_torque(total_mass_kg, l2, l3, legs_in_stance=2, safety_factor=2.5, g=9.81):
    """Order-of-magnitude peak joint torque (knee/hip) during stance.
    safety_factor covers dynamic loading (trot impact, disturbance, slope)
    on top of static worst-case moment arm.
    Returns dict with knee and hip torque estimates (Nm).
    """
    weight_per_leg = (total_mass_kg * g) / legs_in_stance
    leg_len = l2 + l3
    # Knee: worst case moment arm ~ full lower-leg length when leg near-extended
    knee_static = weight_per_leg * l3
    # Hip: worst case moment arm ~ full leg length
    hip_static = weight_per_leg * leg_len
    # Ab/ad: worst case moment arm ~ body half-width lever + leg length component
    abad_static = weight_per_leg * l3 * 0.5
    return {
        "knee_nm": knee_static * safety_factor,
        "hip_nm": hip_static * safety_factor,
        "abad_nm": abad_static * safety_factor,
        "safety_factor": safety_factor,
        "legs_in_stance": legs_in_stance,
    }
