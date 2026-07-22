"""
Motor/servo candidate evaluation against a joint's torque budget.
Applies gearbox ratio + transmission efficiency to get torque AT THE JOINT,
compares against continuous-required and peak-required torque, and checks
speed against the gait's required joint angular velocity.
"""
import numpy as np


# Common servo/motor presets with official manufacturer specifications.
SERVO_PRESETS = {
    "SG90 / Micro (1.8 kg·cm @ 5V) — Micro (< 0.5 kg robot)": {
        "continuous_nm": 0.09, "peak_nm": 0.18, "rpm": 70,
        "voltage": 5.0, "current_a": 0.8, "avg_current_a": 0.20, "mass_kg": 0.009,
        "notes": "Micro servo for mini/toy quadrupeds (<500g body). Plastic gears.",
    },
    "MG996R (11 kg·cm @ 6V) — Small (< 1.2 kg robot)": {
        "continuous_nm": 0.55, "peak_nm": 1.08, "rpm": 60,
        "voltage": 6.0, "current_a": 2.5, "avg_current_a": 0.50, "mass_kg": 0.055,
        "notes": "Entry hobby servo. Suitable for small quadrupeds up to ~1.2 kg.",
    },
    "DS3218 Digital (20 kg·cm @ 6.8V) — Small/Medium (~1.5 kg)": {
        "continuous_nm": 1.0, "peak_nm": 1.96, "rpm": 55,
        "voltage": 6.8, "current_a": 3.0, "avg_current_a": 0.60, "mass_kg": 0.060,
        "notes": "Popular for small quadrupeds (~1.5 kg, standing height < 200 mm).",
    },
    "LewanSoul LX-224HV Bus (24 kg·cm @ 7.4V)": {
        "continuous_nm": 1.2, "peak_nm": 2.35, "rpm": 62,
        "voltage": 7.4, "current_a": 2.8, "avg_current_a": 0.55, "mass_kg": 0.060,
        "notes": "Serial bus servo with position/temp feedback.",
    },
    "Dynamixel XL430-W250 (1.4 Nm @ 12V)": {
        "continuous_nm": 1.0, "peak_nm": 1.4, "rpm": 57,
        "voltage": 12.0, "current_a": 1.5, "avg_current_a": 0.40, "mass_kg": 0.058,
        "notes": "Smart actuator with encoder feedback for research platforms.",
    },
}


def torque_at_speed(motor_peak_nm, motor_rpm, operating_speed_rpm):
    """Returns available torque using linear torque-speed curve:
    tau(omega) = tau_stall * (1 - omega/omega_no_load)
    """
    if operating_speed_rpm > motor_rpm:
        return 0.0
    return motor_peak_nm * (1.0 - operating_speed_rpm / motor_rpm)


def motor_operating_point(*, motor_peak_nm, motor_cont_nm, motor_rpm, required_torque_nm, required_speed_rad_s):
    required_speed_rpm = required_speed_rad_s * 60.0 / (2 * np.pi)
    avail_torque = torque_at_speed(motor_peak_nm, motor_rpm, required_speed_rpm)
    
    speed_pct = (required_speed_rpm / motor_rpm * 100) if motor_rpm > 0 else 0.0
    torque_pct = (required_torque_nm / motor_peak_nm * 100) if motor_peak_nm > 0 else 0.0
    
    in_cont = required_torque_nm <= motor_cont_nm
    
    thermal_warning = None
    if not in_cont:
        thermal_warning = "Operating above continuous torque rating. Will overheat if sustained."
    if avail_torque < required_torque_nm:
        if thermal_warning:
            thermal_warning += " Additionally, required torque exceeds available torque at this speed!"
        else:
            thermal_warning = "Required torque exceeds available torque at this speed!"
            
    return {
        'torque_available_at_speed': avail_torque,
        'speed_pct': speed_pct,
        'torque_pct': torque_pct,
        'in_continuous_region': in_cont,
        'thermal_warning': thermal_warning
    }


def evaluate_motor(*, motor_continuous_nm, motor_peak_nm, motor_rated_rpm,
                    gearbox_ratio, transmission_efficiency,
                    required_continuous_nm, required_peak_nm,
                    required_max_speed_rad_s=None,
                    margin_pass=1.15, margin_marginal=0.90,
                    rating_basis="Servo output shaft (already geared)"):
    """Returns dict: verdict ('PASS'/'MARGINAL'/'FAIL'), reasons (list[str]),
    computed available torque/speed at the joint.
    """
    is_servo = str(rating_basis).startswith("Servo")

    if is_servo:
        avail_continuous = motor_continuous_nm
        avail_peak = motor_peak_nm
        avail_speed_rad_s = motor_rated_rpm * 2 * np.pi / 60
    else:
        avail_continuous = motor_continuous_nm * gearbox_ratio * transmission_efficiency
        avail_peak = motor_peak_nm * gearbox_ratio * transmission_efficiency
        avail_speed_rad_s = (motor_rated_rpm * 2 * np.pi / 60) / max(gearbox_ratio, 1e-9)

    reasons = []
    verdict = "PASS"

    cont_ratio = avail_continuous / required_continuous_nm if required_continuous_nm > 0 else np.inf
    peak_ratio = avail_peak / required_peak_nm if required_peak_nm > 0 else np.inf

    # Continuous torque evaluation
    if cont_ratio < margin_marginal:
        verdict = "FAIL"
        reasons.append(f"Available continuous torque at joint ({avail_continuous:.2f} Nm) is "
                        f"BELOW required continuous load ({required_continuous_nm:.2f} Nm) — "
                        f"actuator will overheat under sustained normal-gait load.")
    elif cont_ratio < margin_pass:
        if verdict == "PASS":
            verdict = "MARGINAL"
        reasons.append(f"Continuous torque margin is thin: {(cont_ratio-1)*100:.0f}% "
                        f"({avail_continuous:.2f} vs {required_continuous_nm:.2f} Nm required).")
    else:
        reasons.append(f"Continuous torque OK: {avail_continuous:.2f} Nm available vs "
                        f"{required_continuous_nm:.2f} Nm required ({(cont_ratio-1)*100:.0f}% margin).")

    # Peak torque evaluation
    if peak_ratio < margin_marginal:
        verdict = "FAIL"
        reasons.append(f"Available peak/stall torque at joint ({avail_peak:.2f} Nm) is BELOW "
                        f"required peak requirement ({required_peak_nm:.2f} Nm) — actuator will stall "
                        f"during acceleration, touchdown impact, or disturbance recovery.")
    elif peak_ratio < margin_pass:
        if verdict == "PASS":
            verdict = "MARGINAL"
        reasons.append(f"Peak torque margin is thin: {(peak_ratio-1)*100:.0f}% "
                        f"({avail_peak:.2f} vs {required_peak_nm:.2f} Nm required).")
    else:
        reasons.append(f"Peak torque OK: {avail_peak:.2f} Nm available vs "
                        f"{required_peak_nm:.2f} Nm required ({(peak_ratio-1)*100:.0f}% margin).")

    # Joint speed evaluation
    if required_max_speed_rad_s is not None and required_max_speed_rad_s > 0:
        speed_ratio = avail_speed_rad_s / required_max_speed_rad_s
        if speed_ratio < margin_marginal:
            verdict = "FAIL"
            reasons.append(f"Available joint speed ({avail_speed_rad_s:.2f} rad/s) is below "
                            f"required gait speed ({required_max_speed_rad_s:.2f} rad/s) — leg will lag trajectory.")
        elif speed_ratio < margin_pass:
            if verdict == "PASS":
                verdict = "MARGINAL"
            reasons.append(f"Joint speed margin is thin: {avail_speed_rad_s:.2f} vs "
                            f"{required_max_speed_rad_s:.2f} rad/s required.")
        else:
            reasons.append(f"Speed OK: {avail_speed_rad_s:.2f} rad/s available vs "
                            f"{required_max_speed_rad_s:.2f} rad/s required.")

    return {
        "verdict": verdict, "reasons": reasons,
        "avail_continuous_nm": avail_continuous, "avail_peak_nm": avail_peak,
        "avail_speed_rad_s": avail_speed_rad_s,
        "continuous_margin_pct": (cont_ratio - 1) * 100 if np.isfinite(cont_ratio) else None,
        "peak_margin_pct": (peak_ratio - 1) * 100 if np.isfinite(peak_ratio) else None,
    }
