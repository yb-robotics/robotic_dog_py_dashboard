"""
Gait pattern generator: walk (statically stable, duty~0.75) and trot
(dynamically stable, duty=0.5). Produces per-leg foot trajectory offsets
and full joint-angle trajectories over one gait cycle.

For 8-DOF robots (no ab/ad joint) the lateral (y) foot position is fixed —
the gait generator enforces zero lateral trajectory variation and the
dashboard must say so explicitly, rather than silently producing a 12-DOF-
looking trajectory the hardware can't actually execute.
"""
import numpy as np

GAITS = {
    "Walk": {"duty_factor": 0.75,
              "phase_offsets": {"FR": 0.0, "RL": 0.25, "FL": 0.5, "RR": 0.75}},
    "Trot": {"duty_factor": 0.5,
              "phase_offsets": {"FR": 0.0, "RL": 0.0, "FL": 0.5, "RR": 0.5}},
}


def swing_trajectory(phase, step_length, step_height):
    """Cycloid swing profile. phase in [0,1). Returns (dx, dz) offset."""
    dx = step_length * (phase - np.sin(2 * np.pi * phase) / (2 * np.pi))
    dz = step_height * 0.5 * (1 - np.cos(2 * np.pi * phase))
    return dx, dz


def stance_trajectory(phase, step_length):
    """phase in [0,1). Foot moves backward under the body at ~constant rate."""
    return -step_length * (phase - 0.5)


def foot_phase(t, period, duty_factor, phase_offset):
    cycle_phase = ((t / period) + phase_offset) % 1.0
    if cycle_phase < duty_factor:
        return True, cycle_phase / duty_factor
    return False, (cycle_phase - duty_factor) / (1 - duty_factor)


def generate_gait_cycle(gait_name, period_s, step_length_m, step_height_m,
                         n_samples=100, has_abad=True):
    """Returns dict: leg_name -> {t, x_offset, z_offset, y_offset, in_stance}
    x/z offsets are relative to the leg's neutral standing foot position.
    y_offset is always 0 here (no active lateral placement modeled in the
    basic gait — even for 12-DOF, ab/ad is reserved for balance correction,
    not nominal gait, unless the user is doing a crab-walk/turn).
    """
    gait = GAITS[gait_name]
    duty = gait["duty_factor"]
    offsets = gait["phase_offsets"]
    t_arr = np.linspace(0, period_s, n_samples, endpoint=False)

    result = {}
    for leg, phase_off in offsets.items():
        xs, zs, stances = [], [], []
        for t in t_arr:
            in_stance, local_phase = foot_phase(t, period_s, duty, phase_off)
            if in_stance:
                dx = stance_trajectory(local_phase, step_length_m)
                dz = 0.0
            else:
                dx, dz = swing_trajectory(local_phase, step_length_m, step_height_m)
                dx -= step_length_m / 2  # center swing return around stance line
            xs.append(dx)
            zs.append(dz)
            stances.append(in_stance)
        result[leg] = {"t": t_arr, "x_offset": np.array(xs), "z_offset": np.array(zs),
                        "y_offset": np.zeros_like(t_arr), "in_stance": np.array(stances)}
    return result


def stance_legs_at_time(gait_name, period_s, t):
    gait = GAITS[gait_name]
    legs = []
    for leg, phase_off in gait["phase_offsets"].items():
        in_stance, _ = foot_phase(t, period_s, gait["duty_factor"], phase_off)
        if in_stance:
            legs.append(leg)
    return legs


def estimate_max_joint_speed(step_length_m, step_height_m, period_s, duty_factor,
                               thigh_length, shank_length, standing_height_m,
                               hip_offset=0.0, n_samples=50):
    """Estimate peak hip and knee angular velocities during one gait cycle.
    
    Uses numerical differentiation of IK solutions along the swing trajectory
    to find the maximum angular velocity each joint experiences.
    Returns dict with 'hip_rad_s', 'knee_rad_s', 'hip_deg_s', 'knee_deg_s'.
    """
    from kinematics import leg_ik
    
    swing_duration = period_s * (1 - duty_factor)
    if swing_duration <= 0:
        return {'hip_rad_s': 0.0, 'knee_rad_s': 0.0, 'hip_deg_s': 0.0, 'knee_deg_s': 0.0}
    
    dt = swing_duration / n_samples
    phases = np.linspace(0, 1, n_samples, endpoint=False)
    
    hip_angles = []
    knee_angles = []
    
    for phase in phases:
        # Cycloid swing trajectory
        dx = step_length_m * (phase - np.sin(2 * np.pi * phase) / (2 * np.pi))
        dx -= step_length_m / 2  # center around stance line
        dz = step_height_m * 0.5 * (1 - np.cos(2 * np.pi * phase))
        
        foot_x = dx
        foot_y = hip_offset
        foot_z = -standing_height_m + dz
        
        sol = leg_ik(foot_x, foot_y, foot_z, hip_offset, thigh_length, shank_length)
        if sol is not None:
            _, q_hip, q_knee = sol
            hip_angles.append(q_hip)
            knee_angles.append(q_knee)
        else:
            # If unreachable, use previous value or zero
            hip_angles.append(hip_angles[-1] if hip_angles else 0.0)
            knee_angles.append(knee_angles[-1] if knee_angles else 0.0)
    
    hip_angles = np.array(hip_angles)
    knee_angles = np.array(knee_angles)
    
    # Numerical differentiation
    hip_vel = np.abs(np.diff(hip_angles) / dt)
    knee_vel = np.abs(np.diff(knee_angles) / dt)
    
    max_hip = float(np.max(hip_vel)) if len(hip_vel) > 0 else 0.0
    max_knee = float(np.max(knee_vel)) if len(knee_vel) > 0 else 0.0
    
    return {
        'hip_rad_s': max_hip,
        'knee_rad_s': max_knee,
        'hip_deg_s': np.degrees(max_hip),
        'knee_deg_s': np.degrees(max_knee),
    }

