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
