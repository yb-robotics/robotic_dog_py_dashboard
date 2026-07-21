"""
Leg inverse dynamics: gravity-compensation torque (2-link planar Newton-Euler
result, standard textbook derivation for a hip-knee pendulum) PLUS ground
reaction force torque (Jacobian-transpose method). Produces static,
continuous-required, and peak-required torque separately at hip and knee.

Ab/ad (hip abduction) torque uses a simplified single-plane lever estimate —
full 3D coupled ab/ad dynamics are not modeled here (documented limitation).
"""
import numpy as np
from kinematics import leg_jacobian

G = 9.81


def two_link_gravity_torque(hip_flexion, knee_flexion, thigh_mass, shank_mass,
                             thigh_com_frac, shank_com_frac, thigh_length, shank_length):
    """Classic 2-link planar pendulum gravity torque (hip measured from
    vertical/down = 0, knee relative to thigh). Returns (tau_hip, tau_knee), Nm.
    thigh_com_frac / shank_com_frac: fraction of link length from the PROXIMAL
    joint to that link's center of mass (0.5 = geometric center).
    """
    l1c = thigh_com_frac * thigh_length
    l2c = shank_com_frac * shank_length
    tau_knee = shank_mass * G * l2c * np.sin(hip_flexion + knee_flexion)
    tau_hip = ((thigh_mass * l1c + shank_mass * thigh_length) * G * np.sin(hip_flexion)
               + shank_mass * G * l2c * np.sin(hip_flexion + knee_flexion))
    return tau_hip, tau_knee


def coriolis_torque(hip_flexion, knee_flexion, hip_velocity, knee_velocity,
                    shank_mass, thigh_length, shank_com_frac, shank_length):
    """Simplified 2-link Coriolis/centrifugal torque contribution.
    Returns (tau_hip_coriolis, tau_knee_coriolis) in Nm.
    """
    l2c = shank_com_frac * shank_length
    # Coriolis: cross-coupling term from knee velocity affecting hip torque
    h = shank_mass * thigh_length * l2c * np.sin(knee_flexion)
    tau_hip = -h * knee_velocity * (2 * hip_velocity + knee_velocity)
    tau_knee = h * hip_velocity ** 2
    return tau_hip, tau_knee


def grf_joint_torque(hip_abduction, hip_flexion, knee_flexion, hip_offset,
                      thigh_length, shank_length, foot_force_xyz):
    """Ground reaction force -> joint torques via Jacobian transpose:
    tau = J^T @ F_foot. foot_force_xyz is the force the GROUND exerts ON
    THE FOOT (i.e. what the leg must react against), (Fx, Fy, Fz) in hip frame.
    Returns (tau_abduction, tau_hip_flexion, tau_knee), Nm.
    """
    J = leg_jacobian(hip_abduction, hip_flexion, knee_flexion, hip_offset, thigh_length, shank_length)
    tau = J.T @ np.array(foot_force_xyz)
    return tau[0], tau[1], tau[2]


def joint_torque_budget(*, hip_flexion, knee_flexion, hip_abduction,
                         hip_offset, thigh_length, shank_length,
                         total_mass_kg, payload_kg,
                         thigh_mass_kg, shank_mass_kg,
                         thigh_com_frac, shank_com_frac,
                         legs_in_stance, dynamic_accel_mps2, impact_factor,
                         transmission_efficiency, safety_factor,
                         hip_velocity_rad_s=0.0, knee_velocity_rad_s=0.0):
    """Full torque budget: static / continuous-required / peak-required,
    separately for hip and knee (and an ab/ad estimate).

    - static: holding position, gravity only, no dynamic acceleration, no impact.
    - continuous_required: adds the nominal dynamic acceleration term (from
      normal walking/trotting, i.e. what the motor must sustain continuously).
    - peak_required: continuous scaled by impact_factor (landing/stumble
      spikes), then divided by transmission efficiency and multiplied by the
      engineering safety factor -> this is the number to compare against a
      candidate motor's PEAK/stall rating.
    """
    carried_mass = total_mass_kg + payload_kg
    supported_mass_per_leg = carried_mass / max(legs_in_stance, 1)
    weight_per_leg = supported_mass_per_leg * G

    # --- static: pure vertical support, no acceleration, no impact ---
    tau_hip_g, tau_knee_g = two_link_gravity_torque(
        hip_flexion, knee_flexion, thigh_mass_kg, shank_mass_kg,
        thigh_com_frac, shank_com_frac, thigh_length, shank_length)
    F_static = np.array([0.0, 0.0, weight_per_leg])
    tau_ab_s, tau_hip_s, tau_knee_s = grf_joint_torque(
        hip_abduction, hip_flexion, knee_flexion, hip_offset, thigh_length, shank_length, F_static)
    static = {
        # At static equilibrium the actuator balances link gravity *minus*
        # the generalized force supplied by the ground: tau = g(q) - J^T F.
        # Adding these terms makes the same upward ground force incorrectly
        # increase both loads and was the source of inconsistent estimates.
        "hip_nm": abs(tau_hip_g - tau_hip_s),
        "knee_nm": abs(tau_knee_g - tau_knee_s),
        "abad_nm": abs(tau_ab_s),
    }

    # --- continuous-required: + nominal dynamic acceleration (normal gait) ---
    # Horizontal acceleration is represented as the fore/aft contact force.
    # Vertical acceleration is intentionally not assumed here; use impact
    # factor for touchdown/transient loading rather than double counting it.
    F_dyn = np.array([supported_mass_per_leg * dynamic_accel_mps2, 0.0, weight_per_leg])
    tau_ab_c, tau_hip_c, tau_knee_c = grf_joint_torque(
        hip_abduction, hip_flexion, knee_flexion, hip_offset, thigh_length, shank_length, F_dyn)
    # Rotational inertia torque: I = (1/3) * m * L^2
    alpha = dynamic_accel_mps2 / (thigh_length + shank_length)
    tau_inertia_hip = (1.0 / 3.0) * thigh_mass_kg * (thigh_length ** 2) * alpha
    tau_inertia_knee = (1.0 / 3.0) * shank_mass_kg * (shank_length ** 2) * alpha

    # Coriolis/centrifugal torque
    if hip_velocity_rad_s != 0.0 or knee_velocity_rad_s != 0.0:
        tau_hip_coriolis, tau_knee_coriolis = coriolis_torque(
            hip_flexion, knee_flexion, hip_velocity_rad_s, knee_velocity_rad_s,
            shank_mass_kg, thigh_length, shank_com_frac, shank_length)
    else:
        tau_hip_coriolis, tau_knee_coriolis = 0.0, 0.0

    continuous = {
        "hip_nm": abs(tau_hip_g - tau_hip_c) + tau_inertia_hip + abs(tau_hip_coriolis),
        "knee_nm": abs(tau_knee_g - tau_knee_c) + tau_inertia_knee + abs(tau_knee_coriolis),
        "abad_nm": abs(tau_ab_c),
    }

    # --- peak-required: continuous * impact factor, / efficiency, * safety ---
    peak = {
        k: (continuous[k] * impact_factor / transmission_efficiency) * safety_factor
        for k in continuous
    }
    # peak should never be reported below continuous or static — enforce monotonicity
    for k in peak:
        peak[k] = max(peak[k], continuous[k], static[k])
        continuous[k] = max(continuous[k], static[k])

    return {"static": static, "continuous_required": continuous, "peak_required": peak,
            "weight_per_leg_n": weight_per_leg, "contact_force_n": F_dyn,
            "supported_mass_per_leg_kg": supported_mass_per_leg}


def stair_climb_torque_budget(*, hip_flexion_flat, knee_flexion_flat,
                                hip_flexion_stair, knee_flexion_stair,
                                hip_abduction, hip_offset, thigh_length, shank_length,
                                total_mass_kg, payload_kg,
                                thigh_mass_kg, shank_mass_kg,
                                thigh_com_frac, shank_com_frac,
                                riser_height_m, transmission_efficiency, safety_factor):
    total_mass = total_mass_kg + payload_kg
    delta_E = total_mass * G * riser_height_m
    a_climb = G * riser_height_m / (thigh_length + shank_length)

    stair_budget = joint_torque_budget(
        hip_flexion=hip_flexion_stair, knee_flexion=knee_flexion_stair,
        hip_abduction=hip_abduction, hip_offset=hip_offset,
        thigh_length=thigh_length, shank_length=shank_length,
        total_mass_kg=total_mass_kg, payload_kg=payload_kg,
        thigh_mass_kg=thigh_mass_kg, shank_mass_kg=shank_mass_kg,
        thigh_com_frac=thigh_com_frac, shank_com_frac=shank_com_frac,
        legs_in_stance=3, dynamic_accel_mps2=a_climb, impact_factor=1.5,
        transmission_efficiency=transmission_efficiency, safety_factor=safety_factor)

    support_budget = joint_torque_budget(
        hip_flexion=hip_flexion_flat, knee_flexion=knee_flexion_flat,
        hip_abduction=hip_abduction, hip_offset=hip_offset,
        thigh_length=thigh_length, shank_length=shank_length,
        total_mass_kg=total_mass_kg, payload_kg=payload_kg,
        thigh_mass_kg=thigh_mass_kg, shank_mass_kg=shank_mass_kg,
        thigh_com_frac=thigh_com_frac, shank_com_frac=shank_com_frac,
        legs_in_stance=3, dynamic_accel_mps2=0.0, impact_factor=1.5,
        transmission_efficiency=transmission_efficiency, safety_factor=safety_factor)

    return {
        "stair_leg_hip_nm": stair_budget["peak_required"]["hip_nm"],
        "stair_leg_knee_nm": stair_budget["peak_required"]["knee_nm"],
        "support_leg_hip_nm": support_budget["peak_required"]["hip_nm"],
        "support_leg_knee_nm": support_budget["peak_required"]["knee_nm"],
        "climb_energy_j": delta_E,
        "min_motor_continuous_nm": max(stair_budget["continuous_required"]["hip_nm"],
                                       stair_budget["continuous_required"]["knee_nm"],
                                       support_budget["continuous_required"]["hip_nm"],
                                       support_budget["continuous_required"]["knee_nm"]),
        "min_motor_peak_nm": max(stair_budget["peak_required"]["hip_nm"],
                                 stair_budget["peak_required"]["knee_nm"],
                                 support_budget["peak_required"]["hip_nm"],
                                 support_budget["peak_required"]["knee_nm"])
    }


SAFETY_PRESETS = {
    "Aggressive — prototype, indoor, controlled (1.5x)": 1.5,
    "Standard — typical research quadruped (2.0x)": 2.0,
    "Conservative — outdoor / uneven terrain (2.5x)": 2.5,
    "Heavy-duty — dynamic gaits, jumping, frequent impacts (3.5x)": 3.5,
}

TRANSMISSION_EFFICIENCY_PRESETS = {
    "Servo motor (integrated gears, ~0.80)": 0.80,
    "Direct-drive / low-ratio QDD (~0.90)": 0.90,
    "Planetary gearbox, single stage (~0.85)": 0.85,
    "Cycloidal drive (~0.80)": 0.80,
    "Harmonic drive (~0.75)": 0.75,
    "Timing belt + pulley (~0.92)": 0.92,
}
