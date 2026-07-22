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
from kinematics import leg_ik
from dynamics import joint_torque_budget

GAITS = {
    "Walk": {"duty_factor": 0.75,
              "phase_offsets": {"FR": 0.0, "RL": 0.25, "FL": 0.5, "RR": 0.75}},
    "Amble": {"duty_factor": 0.65,
              "phase_offsets": {"FR": 0.0, "RL": 0.15, "FL": 0.5, "RR": 0.65}},
    "Trot": {"duty_factor": 0.5,
              "phase_offsets": {"FR": 0.0, "RL": 0.0, "FL": 0.5, "RR": 0.5}},
    "Pace": {"duty_factor": 0.5,
             "phase_offsets": {"FR": 0.0, "RL": 0.5, "FL": 0.5, "RR": 0.0}},
    "Bound": {"duty_factor": 0.4,
              "phase_offsets": {"FR": 0.0, "RL": 0.5, "FL": 0.0, "RR": 0.5}},
    "Pronk": {"duty_factor": 0.35,
              "phase_offsets": {"FR": 0.0, "RL": 0.0, "FL": 0.0, "RR": 0.0}},
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

def recommend_gait(*, hip_peak_nm, knee_peak_nm, hip_cont_nm, knee_cont_nm,
                    robot_mass_kg, payload_kg, standing_height_m,
                    thigh_length, shank_length, hip_offset,
                    target_speed_mps, terrain, efficiency, safety_factor,
                    thigh_mass_kg=0.02, shank_mass_kg=0.02):
    recommendations = []
    
    for gait_name, gait_params in GAITS.items():
        duty_factor = gait_params["duty_factor"]
        
        if duty_factor >= 0.75:
            min_stance = 3
        elif duty_factor >= 0.5:
            min_stance = 2
        elif duty_factor >= 0.35:
            min_stance = 1
        else:
            min_stance = 0
            
        sol = leg_ik(0.0, hip_offset, -standing_height_m, hip_offset, thigh_length, shank_length)
        if sol is None:
            qa, qf, qk = 0.0, 0.0, 0.0
        else:
            qa, qf, qk = sol
            
        budget = joint_torque_budget(
            hip_flexion=qf, knee_flexion=qk, hip_abduction=qa,
            hip_offset=hip_offset, thigh_length=thigh_length, shank_length=shank_length,
            total_mass_kg=robot_mass_kg, payload_kg=payload_kg,
            thigh_mass_kg=thigh_mass_kg, shank_mass_kg=shank_mass_kg,
            thigh_com_frac=0.5, shank_com_frac=0.5,
            legs_in_stance=max(1, min_stance), dynamic_accel_mps2=9.81 if min_stance==0 else 2.0,
            impact_factor=1.5 if min_stance<2 else 1.1, 
            transmission_efficiency=efficiency,
            safety_factor=safety_factor
        )
        
        req_hip = budget["peak_required"]["hip_nm"]
        req_knee = budget["peak_required"]["knee_nm"]
        
        hip_margin = (hip_peak_nm - req_hip) / hip_peak_nm * 100 if hip_peak_nm > 0 else 0
        knee_margin = (knee_peak_nm - req_knee) / knee_peak_nm * 100 if knee_peak_nm > 0 else 0
        
        feasible = (hip_margin > 0) and (knee_margin > 0)
        
        if gait_name == "Walk": speed_suitability = "Walk for <0.3 m/s"
        elif gait_name == "Amble": speed_suitability = "Amble for 0.3-0.5"
        elif gait_name == "Trot": speed_suitability = "Trot for 0.3-1.0"
        elif gait_name == "Pace": speed_suitability = "Pace for 0.5-1.0"
        elif gait_name == "Bound": speed_suitability = "Bound for 0.8-2.0"
        elif gait_name == "Pronk": speed_suitability = "Pronk for jumping"
        else: speed_suitability = "Unknown"
        
        if gait_name in ["Walk", "Amble"]: terrain_suitability = "Walk/Amble best for uneven/stairs"
        elif gait_name == "Trot": terrain_suitability = "Trot best for flat"
        elif gait_name == "Pace": terrain_suitability = "Pace for flat only"
        elif gait_name in ["Bound", "Pronk"]: terrain_suitability = "Bound/Pronk for flat only"
        else: terrain_suitability = "Unknown"

        recommendations.append({
            "gait": gait_name,
            "feasible": feasible,
            "hip_margin_pct": hip_margin,
            "knee_margin_pct": knee_margin,
            "min_stance_legs": min_stance,
            "duty_factor": duty_factor,
            "speed_suitability": speed_suitability,
            "terrain_suitability": terrain_suitability,
            "recommendation": "Recommended" if feasible else "Not feasible"
        })
        
    return sorted(recommendations, key=lambda x: (-int(x["feasible"]), -x["min_stance_legs"]))


def worst_case_gait_margin(*, hip_peak_nm, knee_peak_nm, hip_cont_nm, knee_cont_nm,
                            robot_mass_kg, payload_kg, standing_height_m,
                            thigh_length, shank_length, hip_offset,
                            target_speed_mps, efficiency, safety_factor,
                            thigh_mass_kg=0.02, shank_mass_kg=0.02):
    worst_hip_peak = 0.0
    worst_knee_peak = 0.0
    worst_hip_cont = 0.0
    worst_knee_cont = 0.0
    worst_gait = ""
    
    for gait_name, gait_params in GAITS.items():
        duty_factor = gait_params["duty_factor"]
        
        if duty_factor >= 0.75:
            min_stance = 3
        elif duty_factor >= 0.5:
            min_stance = 2
        elif duty_factor >= 0.35:
            min_stance = 1
        else:
            min_stance = 0
            
        sol = leg_ik(0.0, hip_offset, -standing_height_m, hip_offset, thigh_length, shank_length)
        if sol is None:
            qa, qf, qk = 0.0, 0.0, 0.0
        else:
            qa, qf, qk = sol
            
        budget = joint_torque_budget(
            hip_flexion=qf, knee_flexion=qk, hip_abduction=qa,
            hip_offset=hip_offset, thigh_length=thigh_length, shank_length=shank_length,
            total_mass_kg=robot_mass_kg, payload_kg=payload_kg,
            thigh_mass_kg=thigh_mass_kg, shank_mass_kg=shank_mass_kg,
            thigh_com_frac=0.5, shank_com_frac=0.5,
            legs_in_stance=max(1, min_stance), dynamic_accel_mps2=9.81 if min_stance==0 else 2.0,
            impact_factor=1.5 if min_stance<2 else 1.1, 
            transmission_efficiency=efficiency,
            safety_factor=safety_factor
        )
        
        req_hip = budget["peak_required"]["hip_nm"]
        req_knee = budget["peak_required"]["knee_nm"]
        req_hip_cont = budget["continuous_required"]["hip_nm"]
        req_knee_cont = budget["continuous_required"]["knee_nm"]
        
        if req_hip > worst_hip_peak or req_knee > worst_knee_peak:
            worst_hip_peak = max(worst_hip_peak, req_hip)
            worst_knee_peak = max(worst_knee_peak, req_knee)
            worst_hip_cont = max(worst_hip_cont, req_hip_cont)
            worst_knee_cont = max(worst_knee_cont, req_knee_cont)
            worst_gait = gait_name
            
    hip_margin = (hip_peak_nm - worst_hip_peak) / hip_peak_nm * 100 if hip_peak_nm > 0 else 0
    knee_margin = (knee_peak_nm - worst_knee_peak) / knee_peak_nm * 100 if knee_peak_nm > 0 else 0
    
    all_feasible = (hip_margin > 0) and (knee_margin > 0)
    
    return {
        'worst_hip_peak': worst_hip_peak,
        'worst_knee_peak': worst_knee_peak,
        'worst_hip_cont': worst_hip_cont,
        'worst_knee_cont': worst_knee_cont,
        'worst_gait_name': worst_gait,
        'hip_margin_pct': hip_margin,
        'knee_margin_pct': knee_margin,
        'all_gaits_feasible': all_feasible,
        'undecided_safe': all_feasible
    }
