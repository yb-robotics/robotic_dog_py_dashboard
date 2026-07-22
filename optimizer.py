"""
"AI-assisted" design optimizer.

Honest scope note: this does NOT call a cloud LLM. It runs a local
differential-evolution search (scipy.optimize) over the leg-geometry design
space to suggest link proportions that reduce peak required torque for your
target standing height and mass — i.e. a real optimization algorithm doing
the kind of search a human would do by trial and error, much faster. This is
the practical, runs-offline interpretation of "improve the model with AI"
for a local engineering tool; a live LLM call would need your own API key
and network access and wouldn't be able to run this numeric search any
better than scipy already does.
"""
import numpy as np
from scipy.optimize import differential_evolution

from kinematics import leg_ik
from dynamics import joint_torque_budget
from motor import SERVO_PRESETS


def _candidate_peak_knee(params, standing_height_m, total_mass_kg, payload_kg,
                          thigh_mass_kg, shank_mass_kg, legs_in_stance,
                          dynamic_accel, impact_factor, efficiency, safety_factor,
                          has_abad, hip_offset):
    height_fraction, thigh_frac = params
    max_ext = standing_height_m / height_fraction
    thigh_length = max_ext * thigh_frac
    shank_length = max_ext * (1 - thigh_frac)
    if thigh_length <= 0.01 or shank_length <= 0.01:
        return 1e6

    target = (0.0, hip_offset if has_abad else 0.0, -standing_height_m)
    sol = leg_ik(*target, hip_offset, thigh_length, shank_length)
    if sol is None:
        return 1e6
    qa, qf, qk = sol

    budget = joint_torque_budget(
        hip_flexion=qf, knee_flexion=qk, hip_abduction=qa,
        hip_offset=hip_offset, thigh_length=thigh_length, shank_length=shank_length,
        total_mass_kg=total_mass_kg, payload_kg=payload_kg,
        thigh_mass_kg=thigh_mass_kg, shank_mass_kg=shank_mass_kg,
        thigh_com_frac=0.5, shank_com_frac=0.5,
        legs_in_stance=legs_in_stance, dynamic_accel_mps2=dynamic_accel,
        impact_factor=impact_factor, transmission_efficiency=efficiency,
        safety_factor=safety_factor,
    )
    return budget["peak_required"]["knee_nm"]


def _candidate_dual_objective(params, standing_height_m, total_mass_kg, payload_kg,
                          thigh_mass_kg, shank_mass_kg, legs_in_stance,
                          dynamic_accel, impact_factor, efficiency, safety_factor,
                          has_abad, hip_offset, hip_peak_nm, knee_peak_nm):
    height_fraction, thigh_frac = params
    max_ext = standing_height_m / height_fraction
    thigh_length = max_ext * thigh_frac
    shank_length = max_ext * (1 - thigh_frac)
    if thigh_length <= 0.01 or shank_length <= 0.01:
        return 1e6

    target = (0.0, hip_offset if has_abad else 0.0, -standing_height_m)
    sol = leg_ik(*target, hip_offset, thigh_length, shank_length)
    if sol is None:
        return 1e6
    qa, qf, qk = sol

    budget = joint_torque_budget(
        hip_flexion=qf, knee_flexion=qk, hip_abduction=qa,
        hip_offset=hip_offset, thigh_length=thigh_length, shank_length=shank_length,
        total_mass_kg=total_mass_kg, payload_kg=payload_kg,
        thigh_mass_kg=thigh_mass_kg, shank_mass_kg=shank_mass_kg,
        thigh_com_frac=0.5, shank_com_frac=0.5,
        legs_in_stance=legs_in_stance, dynamic_accel_mps2=dynamic_accel,
        impact_factor=impact_factor, transmission_efficiency=efficiency,
        safety_factor=safety_factor,
    )
    hip_util = budget["peak_required"]["hip_nm"] / hip_peak_nm if hip_peak_nm > 0 else 1e6
    knee_util = budget["peak_required"]["knee_nm"] / knee_peak_nm if knee_peak_nm > 0 else 1e6
    return max(hip_util, knee_util)


def optimize_leg_geometry(standing_height_m, total_mass_kg, payload_kg,
                           thigh_mass_kg, shank_mass_kg, legs_in_stance,
                           dynamic_accel, impact_factor, efficiency, safety_factor,
                           has_abad, hip_offset, seed=0):
    """Search (height_fraction in [0.60,0.90], thigh_fraction in [0.35,0.65])
    to minimize peak knee torque for the given target standing height & mass.
    Returns dict with suggested geometry + before/after torque comparison.
    """
    bounds = [(0.60, 0.90), (0.35, 0.65)]
    args = (standing_height_m, total_mass_kg, payload_kg, thigh_mass_kg, shank_mass_kg,
            legs_in_stance, dynamic_accel, impact_factor, efficiency, safety_factor,
            has_abad, hip_offset)

    baseline_peak = _candidate_peak_knee((0.78, 0.5), *args)

    result = differential_evolution(_candidate_peak_knee, bounds, args=args,
                                     seed=seed, maxiter=60, popsize=15, tol=1e-6,
                                     polish=True)

    hf, tf = result.x
    max_ext = standing_height_m / hf
    suggestion = {
        "height_fraction": hf,
        "thigh_fraction": tf,
        "thigh_length_m": max_ext * tf,
        "shank_length_m": max_ext * (1 - tf),
        "baseline_peak_knee_nm": baseline_peak,
        "optimized_peak_knee_nm": result.fun,
        "improvement_pct": (1 - result.fun / baseline_peak) * 100 if baseline_peak > 0 else 0.0,
    }
    return suggestion


def optimize_leg_proportions_for_motors(standing_height_m, total_mass_kg, payload_kg,
                                         thigh_mass_kg, shank_mass_kg, legs_in_stance,
                                         dynamic_accel, impact_factor, efficiency, safety_factor,
                                         has_abad, hip_offset,
                                         hip_peak_nm, hip_cont_nm, knee_peak_nm, knee_cont_nm,
                                         seed=0):
    bounds = [(0.60, 0.90), (0.35, 0.65)]
    args = (standing_height_m, total_mass_kg, payload_kg, thigh_mass_kg, shank_mass_kg,
            legs_in_stance, dynamic_accel, impact_factor, efficiency, safety_factor,
            has_abad, hip_offset, hip_peak_nm, knee_peak_nm)
            
    baseline_util = _candidate_dual_objective((0.78, 0.5), *args)
    result = differential_evolution(_candidate_dual_objective, bounds, args=args,
                                     seed=seed, maxiter=60, popsize=15, tol=1e-6,
                                     polish=True)
                                     
    hf, tf = result.x
    max_ext = standing_height_m / hf
    
    target = (0.0, hip_offset if has_abad else 0.0, -standing_height_m)
    thigh_length = max_ext * tf
    shank_length = max_ext * (1 - tf)
    
    sol = leg_ik(*target, hip_offset, thigh_length, shank_length)
    if sol is not None:
        qa, qf, qk = sol
        budget = joint_torque_budget(
            hip_flexion=qf, knee_flexion=qk, hip_abduction=qa,
            hip_offset=hip_offset, thigh_length=thigh_length, shank_length=shank_length,
            total_mass_kg=total_mass_kg, payload_kg=payload_kg,
            thigh_mass_kg=thigh_mass_kg, shank_mass_kg=shank_mass_kg,
            thigh_com_frac=0.5, shank_com_frac=0.5,
            legs_in_stance=legs_in_stance, dynamic_accel_mps2=dynamic_accel,
            impact_factor=impact_factor, transmission_efficiency=efficiency,
            safety_factor=safety_factor,
        )
        hip_util = budget["peak_required"]["hip_nm"] / hip_peak_nm * 100 if hip_peak_nm > 0 else 0
        knee_util = budget["peak_required"]["knee_nm"] / knee_peak_nm * 100 if knee_peak_nm > 0 else 0
    else:
        hip_util, knee_util = 100, 100

    return {
        "thigh_length_m": thigh_length,
        "shank_length_m": shank_length,
        "femur_fraction": tf,
        "hip_utilization_pct": hip_util,
        "knee_utilization_pct": knee_util,
        "improvement": (baseline_util - result.fun) / baseline_util * 100 if baseline_util > 0 else 0.0
    }


def suggest_lighter_motor_combination(standing_height_m, total_mass_kg, payload_kg,
                                      thigh_mass_kg, shank_mass_kg,
                                      legs_in_stance, dynamic_accel, impact_factor,
                                      efficiency, safety_factor, has_abad, hip_offset,
                                      current_hip_preset_name, current_knee_preset_name,
                                      dof_total=8):
    current_hip = SERVO_PRESETS.get(current_hip_preset_name)
    current_knee = SERVO_PRESETS.get(current_knee_preset_name)
    if not current_hip or not current_knee:
        return []
        
    num_motors_hip = 8 if dof_total == 12 else 4
    num_motors_knee = 4
    
    current_weight = current_hip["mass_kg"] * num_motors_hip + current_knee["mass_kg"] * num_motors_knee
    body_mass_no_motors = total_mass_kg - current_weight
    
    suggestions = []
    
    for hip_name, hip_motor in SERVO_PRESETS.items():
        for knee_name, knee_motor in SERVO_PRESETS.items():
            new_weight = hip_motor["mass_kg"] * num_motors_hip + knee_motor["mass_kg"] * num_motors_knee
            if new_weight < current_weight:
                # Sweep femur_fraction
                best_margin = -1e6
                best_tf = 0.5
                best_h_margin = 0
                best_k_margin = 0
                
                # Dynamic mass for this candidate combo
                cand_total_mass = body_mass_no_motors + new_weight
                
                for tf in np.linspace(0.35, 0.65, 10):
                    max_ext = standing_height_m / 0.78
                    thigh_length = max_ext * tf
                    shank_length = max_ext * (1 - tf)
                    
                    sol = leg_ik(0.0, hip_offset if has_abad else 0.0, -standing_height_m, hip_offset, thigh_length, shank_length)
                    if sol is None:
                        continue
                    qa, qf, qk = sol
                    
                    budget = joint_torque_budget(
                        hip_flexion=qf, knee_flexion=qk, hip_abduction=qa,
                        hip_offset=hip_offset, thigh_length=thigh_length, shank_length=shank_length,
                        total_mass_kg=cand_total_mass, payload_kg=payload_kg,
                        thigh_mass_kg=thigh_mass_kg, shank_mass_kg=shank_mass_kg,
                        thigh_com_frac=0.5, shank_com_frac=0.5,
                        legs_in_stance=legs_in_stance, dynamic_accel_mps2=dynamic_accel,
                        impact_factor=impact_factor, transmission_efficiency=efficiency,
                        safety_factor=safety_factor,
                    )
                    
                    req_hip = budget["peak_required"]["hip_nm"]
                    req_knee = budget["peak_required"]["knee_nm"]
                    
                    h_margin = (hip_motor["peak_nm"] - req_hip) / hip_motor["peak_nm"] * 100 if hip_motor["peak_nm"] > 0 else -100
                    k_margin = (knee_motor["peak_nm"] - req_knee) / knee_motor["peak_nm"] * 100 if knee_motor["peak_nm"] > 0 else -100
                    
                    if h_margin > 0 and k_margin > 0:
                        min_margin = min(h_margin, k_margin)
                        if min_margin > best_margin:
                            best_margin = min_margin
                            best_tf = tf
                            best_h_margin = h_margin
                            best_k_margin = k_margin
                            
                if best_margin > 0:
                    suggestions.append({
                        "hip_motor": hip_name,
                        "knee_motor": knee_name,
                        "femur_fraction": best_tf,
                        "total_motor_weight_kg": new_weight,
                        "weight_savings_g": (current_weight - new_weight) * 1000,
                        "hip_margin_pct": best_h_margin,
                        "knee_margin_pct": best_k_margin
                    })
                    
    return sorted(suggestions, key=lambda x: x["weight_savings_g"], reverse=True)
