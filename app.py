"""Kinematics-first design dashboard for an 8- or 12-DOF quadruped.

Run: streamlit run app.py
All length inputs displayed to the designer are millimetres; calculations use SI units.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import importlib
import kinematics
import dynamics
import motor
import pdf_generator

importlib.reload(kinematics)
importlib.reload(dynamics)
importlib.reload(motor)
importlib.reload(pdf_generator)

from kinematics import (derive_dimensions, leg_fk, leg_ik, leg_jacobian,
                        jacobian_condition_number)
from dynamics import (SAFETY_PRESETS, TRANSMISSION_EFFICIENCY_PRESETS,
                      joint_torque_budget, stair_climb_torque_budget)
from motor import evaluate_motor, SERVO_PRESETS
from gait import estimate_max_joint_speed
from pdf_generator import generate_quadruped_pdf_report

try:
    from optimizer import optimize_leg_geometry
except ImportError:
    optimize_leg_geometry = None


MM = 1000.0
G = 9.81
st.set_page_config(page_title="Quadruped Design Dashboard", layout="wide")


def mm_input(label, value_mm, minimum=1.0, maximum=2000.0, step=1.0, **kwargs):
    """Return an SI value while keeping every UI length input in mm."""
    return st.number_input(label, min_value=float(minimum), max_value=float(maximum),
                           value=float(value_mm), step=float(step), **kwargs) / MM


def posture_dimensions(dof_total, standing_height_m, femur_fraction, neutral_knee_deg):
    """Geometry from actual standing posture, compatible with cached modules."""
    dimensions = derive_dimensions(dof_total, standing_height_m)
    femur_fraction = float(np.clip(femur_fraction, .35, .65))
    shank_fraction = 1.0 - femur_fraction
    knee = np.radians(float(np.clip(neutral_knee_deg, 15.0, 120.0)))
    posture_factor = np.sqrt(femur_fraction**2 + shank_fraction**2
                             + 2 * femur_fraction * shank_fraction * np.cos(knee))
    total_link_length = standing_height_m / posture_factor
    l2, l3 = total_link_length * femur_fraction, total_link_length * shank_fraction
    dimensions.update({
        "l2": l2, "l3": l3, "max_extension": total_link_length,
        "neutral_knee_rad": knee,
        "neutral_hip_rad": -np.arctan2(l3 * np.sin(knee), l2 + l3 * np.cos(knee)),
        "neutral_knee_deg": neutral_knee_deg,
        "femur_fraction": femur_fraction,
        "posture_factor": posture_factor,
    })
    return dimensions


def plot_leg(thigh, shank, q_hip, q_knee, target, reachable):
    """A clear dog-leg diagram: hip joint → thigh → knee joint → shank → paw."""
    hip = np.array([0.0, 0.0])
    knee = np.array([thigh * np.sin(q_hip), -thigh * np.cos(q_hip)])
    paw = knee + np.array([shank * np.sin(q_hip + q_knee),
                            -shank * np.cos(q_hip + q_knee)])
    fig, ax = plt.subplots(figsize=(6.3, 5.1))
    ax.plot([hip[0], knee[0]], [hip[1], knee[1]], "o-", lw=10, ms=10,
            color="#aa6a3d", label="Thigh (upper leg)")
    ax.plot([knee[0], paw[0]], [knee[1], paw[1]], "o-", lw=8, ms=10,
            color="#5d4037", label="Shank / lower leg")
    ax.scatter(*hip, s=130, c="#1d3557", zorder=5)
    ax.scatter(*knee, s=120, c="#1d3557", zorder=5)
    ax.scatter(*paw, s=150, c="#2a9d8f", marker="s", zorder=5)
    ax.annotate("Hip joint", hip, xytext=(8, 10), textcoords="offset points")
    ax.annotate("Knee joint", knee, xytext=(8, 10), textcoords="offset points")
    ax.annotate("Paw / foot", paw, xytext=(8, -18), textcoords="offset points")
    ax.scatter(target[0], target[2], marker="x", s=120,
               c="#e63946" if not reachable else "#264653", label="Requested foot target")
    reach = thigh + shank
    ax.set(xlim=(-reach * 1.12, reach * 1.12), ylim=(-reach * 1.12, reach * .30),
           xlabel="Fore / aft (m)", ylabel="Vertical (m)", title="Live leg pose (side view)")
    ax.axhline(-reach, ls="--", color="gray", lw=1, label="Maximum extension")
    ax.grid(alpha=.25); ax.set_aspect("equal"); ax.legend(fontsize=8, loc="upper right")
    return fig


def stair_pose(thigh, shank, hip_angle, knee_angle, target, standing_h,
               riser_h, tread, direction, reachable):
    """Draw a single 2D leg over one normal stair riser in the sagittal plane."""
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    floor = -standing_h
    if direction == "Climb":
        xs, zs = [-tread, 0, 0, tread * 1.4], [floor, floor, floor + riser_h, floor + riser_h]
        title = "Climb: paw landing on the higher step"
    else:
        xs, zs = [-tread, 0, 0, tread * 1.4], [floor, floor, floor - riser_h, floor - riser_h]
        title = "Descend: paw landing on the lower step"
    ax.plot(xs, zs, color="#6d6875", lw=6, solid_capstyle="butt", label="Stair profile")
    ax.fill_between(xs, zs, min(zs) - riser_h, color="#d9d9d9", alpha=.65)
    hip = np.array([0.0, 0.0])
    if reachable:
        knee = np.array([thigh * np.sin(hip_angle), -thigh * np.cos(hip_angle)])
        paw = knee + np.array([shank * np.sin(hip_angle + knee_angle),
                                -shank * np.cos(hip_angle + knee_angle)])
        ax.plot([0, knee[0]], [0, knee[1]], "o-", lw=9, ms=9, color="#aa6a3d", label="Thigh")
        ax.plot([knee[0], paw[0]], [knee[1], paw[1]], "o-", lw=7, ms=9, color="#5d4037", label="Shank")
        ax.scatter(*paw, marker="s", s=110, c="#2a9d8f", zorder=6, label="Paw")
        ax.annotate("Hip joint", hip, xytext=(8, 8), textcoords="offset points")
        ax.annotate("Knee joint", knee, xytext=(8, 8), textcoords="offset points")
    ax.scatter(target[0], target[2], marker="x", s=110, c="#e63946", zorder=7, label="Requested landing")
    span = max(thigh + shank, standing_h + riser_h)
    ax.set(xlim=(-tread * 1.2, max(tread * 1.5, target[0] + tread * .4)),
           ylim=(-span * 1.25, span * .25), xlabel="Fore / aft (m)", ylabel="Vertical (m)", title=title)
    ax.set_aspect("equal"); ax.grid(alpha=.25); ax.legend(fontsize=8, loc="upper right")
    return fig


def assess_stair_target(target, thigh, shank, hip_offset, knee_forward,
                        hip_min, hip_max, knee_min, knee_max):
    """IK and joint-limit check for a planned paw landing on a stair."""
    solution = leg_ik(*target, hip_offset, thigh, shank, knee_forward)
    if solution is None:
        return None, ["Paw landing is outside the leg workspace."]
    _, q_hip, q_knee = solution
    hip_deg, knee_deg = np.degrees(q_hip), np.degrees(q_knee)
    issues = []
    if not hip_min <= hip_deg <= hip_max:
        issues.append(f"Hip joint {hip_deg:.1f}° is outside {hip_min:.0f}° to {hip_max:.0f}°.")
    if not knee_min <= knee_deg <= knee_max:
        issues.append(f"Knee joint {knee_deg:.1f}° is outside {knee_min:.0f}° to {knee_max:.0f}°.")
    return solution, issues


def recommend_max_standing_height_for_motor(*, hip_peak_nm, hip_cont_nm, knee_peak_nm, knee_cont_nm,
                                            dof_total, femur_fraction, neutral_knee_deg, total_mass_kg,
                                            payload_kg, thigh_mass_kg, shank_mass_kg, stance_legs,
                                            dynamic_accel, impact_factor, efficiency, safety_factor):
    """Calculate maximum viable standing height (mm) supported by both Hip & Knee motor torque limits using inverse dynamics."""
    max_h_mm = None
    for h_mm in range(50, 1201, 5):
        h_m = h_mm / MM
        dims = derive_dimensions(dof_total, h_m, femur_fraction, neutral_knee_deg)
        sol = leg_ik(0.0, dims['l1'], -h_m, dims['l1'], dims['l2'], dims['l3'])
        if sol is None:
            continue
        q_ab, q_hip, q_knee = sol
        budget = joint_torque_budget(
            hip_flexion=q_hip, knee_flexion=q_knee, hip_abduction=q_ab,
            hip_offset=dims['l1'], thigh_length=dims['l2'], shank_length=dims['l3'],
            total_mass_kg=total_mass_kg, payload_kg=payload_kg,
            thigh_mass_kg=thigh_mass_kg, shank_mass_kg=shank_mass_kg,
            thigh_com_frac=0.5, shank_com_frac=0.5, legs_in_stance=stance_legs,
            dynamic_accel_mps2=dynamic_accel, impact_factor=impact_factor,
            transmission_efficiency=efficiency, safety_factor=safety_factor
        )
        hip_peak_req = budget['peak_required']['hip_nm']
        knee_peak_req = budget['peak_required']['knee_nm']
        hip_cont_req = budget['continuous_required']['hip_nm']
        knee_cont_req = budget['continuous_required']['knee_nm']

        if (hip_peak_req <= hip_peak_nm and hip_cont_req <= hip_cont_nm and
            knee_peak_req <= knee_peak_nm and knee_cont_req <= knee_cont_nm):
            max_h_mm = h_mm
    return max_h_mm


# Initialize session state payload synchronization
if "mass_payload" not in st.session_state:
    st.session_state["mass_payload"] = 0.0

# ---- TOP HEADER WITH PDF DOWNLOAD BUTTON -------------------------------------------
col_title, col_pdf = st.columns([3.2, 1.2])
with col_title:
    st.title("🐕 Quadruped Robot Design Dashboard")
    st.caption("Kinematics first → mass & dual-motor selection → torque budget → PDF Report Exporter. All dimensions in mm.")

# Placeholder for PDF download button (rendered after calculations)
pdf_button_container = col_pdf.container()

# ---- Sidebar Inputs ---------------------------------------------------------------
with st.sidebar:
    st.header("1. Beginner design wizard")
    st.caption("Start here. These values drive the geometry and give you a practical first-build direction.")
    dof_total = st.radio("Total degrees of freedom", [8, 12], index=0, horizontal=True)
    standing_height = mm_input("Target standing height (mm)", 350, 0.1, 1500, 0.1,
                               help="Example: enter 9.7 for 0.97 cm, or 970 for 0.97 m.")
    femur_fraction = st.slider("Femur share of total leg length (%)", 35, 65, 52, 1) / 100
    neutral_knee_deg = st.slider("Neutral standing knee bend (degrees)", 15, 120, 55, 1,
                                 help="A bent knee avoids the full-extension singularity. 45–70° is a sensible first prototype range.")
    terrain = st.selectbox("Primary terrain", ["Indoor flat floor", "Carpet / grass", "Small stairs", "Uneven outdoor ground"])

    # Synchronized Payload Input across sidebar and mass section
    sidebar_payload = st.number_input("Expected payload (kg)", 0.0, 50.0,
                                      float(st.session_state["mass_payload"]), 0.01,
                                      key="sidebar_payload_input",
                                      help="Extra weight carried on top of the robot (e.g. camera, arm, sensors). Changing this updates the Mass Breakdown section automatically.")
    st.session_state["mass_payload"] = sidebar_payload

    target_speed = st.number_input("Desired walking speed (m/s)", 0.0, 5.0, .25, .05)
    st.caption("0.97 cm = 9.7 mm. A value of 97 means 97 mm (9.7 cm), which is a very small robot.")

    if terrain == "Small stairs":
        std_riser_m = 0.178
        min_viable_height_mm = None
        for candidate_mm in range(50, 1501, 5):
            candidate_m = candidate_mm / MM
            test_dim = posture_dimensions(dof_total, candidate_m, femur_fraction, neutral_knee_deg)
            test_z = -candidate_m + std_riser_m
            test_sol = leg_ik(std_riser_m * 0.5, test_dim["l1"] if test_dim["has_abad"] else 0.0,
                              test_z, test_dim["l1"], test_dim["l2"], test_dim["l3"])
            if test_sol is not None:
                min_viable_height_mm = candidate_mm
                break
        stair_min_height = max(min_viable_height_mm or 300, 270 if dof_total == 12 else 300)
        if standing_height * MM < stair_min_height:
            st.error(f"⛔ CANNOT CLIMB standard 178 mm stairs at this height ({standing_height*MM:.0f} mm). "
                     f"Minimum standing height: **{stair_min_height} mm**. "
                     f"Increase height to ≥ {stair_min_height} mm, or change terrain.")
        else:
            st.success(f"Height passes the stair gate (≥ {stair_min_height} mm).")

    if dof_total == 8:
        st.warning("8-DOF is cheaper and easier, but has no active lateral balance. Choose 12-DOF for uneven ground or stairs.")
    else:
        st.success("12-DOF is the better mechanical choice for uneven terrain because each hip can place the foot laterally.")

    st.divider()
    st.header("Engineering assumptions")
    safety_name = st.selectbox("Engineering safety assumption", list(SAFETY_PRESETS), index=2,
        help="Choose the operating severity; this multiplier is applied to peak required torque.")
    safety_factor = SAFETY_PRESETS[safety_name]
    transmission_name = st.selectbox("Transmission type", list(TRANSMISSION_EFFICIENCY_PRESETS), index=1)
    efficiency = TRANSMISSION_EFFICIENCY_PRESETS[transmission_name]

derived = posture_dimensions(dof_total, standing_height, femur_fraction, neutral_knee_deg)
has_abad = derived["has_abad"]

st.subheader("Live kinematics simulator")
st.info("Drag the target sliders to move the paw live. FK is recomputed from the IK solution on every change.")
sim_controls, sim_diagram, sim_readout = st.columns([1.1, 1.8, 1.1])
with sim_controls:
    st.markdown("**Foot target in the hip frame (mm)**")
    max_reach = derived["max_extension"]
    foot_x = st.slider("Drag paw fore / aft (mm)", int(-max_reach * MM), int(max_reach * MM), 0, 1) / MM
    if has_abad:
        foot_y = st.slider("Drag paw lateral (mm)", int(-max_reach * MM), int(max_reach * MM),
                           int(derived["l1"] * MM), 1) / MM
    else:
        foot_y = 0.0
        st.caption("8-DOF: the paw has no active lateral (Y) placement.")
    foot_z = st.slider("Drag paw vertical (mm; down is negative)", int(-max_reach * MM), -1,
                       int(-standing_height * MM), 1) / MM
    knee_forward = st.toggle("Knee-forward configuration", value=True)
    use_custom = st.toggle("Set thigh and shank lengths", value=False)
    if use_custom:
        thigh = mm_input("Thigh length (mm)", derived["l2"] * MM, 1, 1500)
        shank = mm_input("Shank length (mm)", derived["l3"] * MM, 1, 1500)
    else:
        thigh, shank = derived["l2"], derived["l3"]

jacobian_condition = None
sol = leg_ik(foot_x, foot_y, foot_z, derived["l1"], thigh, shank, knee_forward)
with sim_diagram:
    if sol is None:
        st.pyplot(plot_leg(thigh, shank, 0, 0, (foot_x, foot_y, foot_z), False), use_container_width=True)
    else:
        st.pyplot(plot_leg(thigh, shank, sol[1], sol[2], (foot_x, foot_y, foot_z), True), use_container_width=True)
with sim_readout:
    st.markdown("**Pose result**")
    if sol is None:
        st.error("UNREACHABLE")
        st.caption(f"Distance must be between {abs(thigh-shank)*MM:.1f} and {(thigh+shank)*MM:.1f} mm.")
        q_ab, q_hip, q_knee = 0.0, 0.0, 0.0
    else:
        q_ab, q_hip, q_knee = sol
        st.success("REACHABLE")
        st.metric("Hip joint", f"{np.degrees(q_hip):.1f}°")
        st.metric("Knee joint", f"{np.degrees(q_knee):.1f}°")
        if has_abad:
            st.metric("Hip ab/ad joint", f"{np.degrees(q_ab):.1f}°")
        check = leg_fk(q_ab, q_hip, q_knee, derived["l1"], thigh, shank)
        st.caption(f"FK check: {np.linalg.norm(check-np.array([foot_x, foot_y, foot_z]))*MM:.4f} mm error")
        J = leg_jacobian(q_ab, q_hip, q_knee, derived["l1"], thigh, shank)
        jacobian_condition = jacobian_condition_number(q_ab, q_hip, q_knee, derived["l1"], thigh, shank)
        st.metric("Jacobian condition number", f"{jacobian_condition:.1f}")
        if jacobian_condition > 50:
            st.warning("Near a kinematic singularity: small foot movements can need very large joint speeds/forces.")
        with st.expander("See the kinematics matrix"):
            st.caption("J maps joint speed to foot speed: v = J·q̇. Its transpose maps foot force to joint torque: τ = Jᵀ·F.")
            st.dataframe(pd.DataFrame(J, index=["x", "y", "z"], columns=["ab/ad", "hip", "knee"]).round(4))

st.divider()
st.subheader("Kinematic dimensions")
d1, d2, d3, d4 = st.columns(4)
d1.metric("Body length", f"{derived['body_length']*MM:.0f} mm")
d2.metric("Body width", f"{derived['body_width']*MM:.0f} mm")
d3.metric("Thigh (upper leg)", f"{thigh*MM:.1f} mm")
d4.metric("Shank (lower leg)", f"{shank*MM:.1f} mm")
st.caption(f"Geometry model: height = √(femur² + shank² + 2·femur·shank·cos(knee bend)). At your {derived['neutral_knee_deg']:.0f}° neutral knee bend, the designed links give {standing_height*MM:.1f} mm hip height.")

# ---- Motor & Servo Selection Panel (Supports Case 1 & Case 2 per leg) ----------------
st.divider()
st.subheader("Motor / servo selection (2 Motors Per Leg)")
st.caption("Select motor models per joint. You can choose identical motors for all joints (Case 1) or independent Hip vs. Knee motors (Case 2).")

motor_mode_name = st.radio(
    "Motor Selection Mode",
    [
        "Case 1: Same Motor Model for All Joints (Both Hip & Knee identical)",
        "Case 2: Independent Motors per Joint (Higher spec Hip, Lower spec Knee)"
    ],
    index=0,
    horizontal=True,
    help="• Case 1: Uses the same motor model for both Hip and Knee joints.\n• Case 2: Allows picking different motor models for Hip vs Knee joints."
)
motor_mode = "same" if motor_mode_name.startswith("Case 1") else "independent"

if motor_mode == "same":
    servo_preset_name = st.selectbox(
        "Select Servo / Motor Model (All Joints)",
        list(SERVO_PRESETS.keys()), index=2,
        help="Select motor model applied to both Hip and Knee joints."
    )
    hip_preset = SERVO_PRESETS[servo_preset_name]
    knee_preset = SERVO_PRESETS[servo_preset_name]
    hip_preset_name = servo_preset_name
    knee_preset_name = servo_preset_name
else:
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        hip_preset_name = st.selectbox(
            "Select Hip Joint Servo / Motor Model",
            list(SERVO_PRESETS.keys()), index=3,  # e.g., DS3235
            help="Select motor model for the Hip pitch joint."
        )
        hip_preset = SERVO_PRESETS[hip_preset_name]
    with col_m2:
        knee_preset_name = st.selectbox(
            "Select Knee Joint Servo / Motor Model",
            list(SERVO_PRESETS.keys()), index=2,  # e.g., DS3218
            help="Select motor model for the Knee pitch joint."
        )
        knee_preset = SERVO_PRESETS[knee_preset_name]

# Extract official datasheet parameters for Hip and Knee
hip_cont = hip_preset["continuous_nm"]
hip_peak = hip_preset["peak_nm"]
hip_rpm = float(hip_preset["rpm"])
hip_volts = hip_preset["voltage"]
hip_curr = hip_preset["current_a"]
hip_avg_curr = hip_preset.get("avg_current_a", hip_curr * 0.25)

knee_cont = knee_preset["continuous_nm"]
knee_peak = knee_preset["peak_nm"]
knee_rpm = float(knee_preset["rpm"])
knee_volts = knee_preset["voltage"]
knee_curr = knee_preset["current_a"]
knee_avg_curr = knee_preset.get("avg_current_a", knee_curr * 0.25)

# Calculate total motor mass dynamically based on joint selection
if dof_total == 8:
    # 4 legs x 1 hip motor + 4 legs x 1 knee motor
    preset_motor_mass_total = 4 * hip_preset["mass_kg"] + 4 * knee_preset["mass_kg"]
else:
    # 12-DOF: 4 legs x (1 hip ab/ad + 1 hip pitch) + 4 legs x 1 knee motor
    preset_motor_mass_total = 8 * hip_preset["mass_kg"] + 4 * knee_preset["mass_kg"]

motor_mass_total = st.session_state.get("mass_motor", preset_motor_mass_total)
battery_mass = st.session_state.get("mass_battery", 0.20)
frame_mass = st.session_state.get("mass_frame", 0.25)
links_mass = st.session_state.get("mass_links", 0.15)
payload_mass = st.session_state.get("mass_payload", 0.0)
payload_x = st.session_state.get("mass_payload_x", 0.0) / MM
esp32_mass, pcb_mass, sensor_mass, wiring_mass, fastener_mass = .02, .03, .03, .04, .03

components = {"Motors": motor_mass_total, "Battery": battery_mass, "ESP32/controller": esp32_mass,
              "PCB": pcb_mass, "Frame": frame_mass, "Links": links_mass, "Sensors": sensor_mass,
              "Wiring": wiring_mass, "Fasteners": fastener_mass}
robot_mass = sum(components.values())
total_system_mass = robot_mass + payload_mass

# Official Datasheet Spec Card Display
st.markdown("##### 📋 Selected Motor Datasheet Specifications:")
if motor_mode == "same":
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("Peak Torque", f"{hip_peak:.2f} Nm")
    sc2.metric("Continuous Torque", f"{hip_cont:.2f} Nm")
    sc3.metric("Rated Speed", f"{hip_rpm:.0f} RPM")
    sc4.metric("Operating Voltage", f"{hip_volts:.1f} V")
    sc5.metric("Stall Current", f"{hip_curr:.1f} A")
    st.caption(f"ℹ️ **Model**: {hip_preset_name.split(' — ')[0]} | Weight: {hip_preset['mass_kg']*1000:.0f} g each ({preset_motor_mass_total:.2f} kg for {dof_total} motors).")
else:
    mc_h, mc_k = st.columns(2)
    with mc_h:
        st.info(f"**Hip Motor ({hip_preset_name.split(' — ')[0]})**: Peak: {hip_peak:.2f} Nm | Cont: {hip_cont:.2f} Nm | Speed: {hip_rpm:.0f} RPM | Voltage: {hip_volts:.1f}V | Stall Curr: {hip_curr:.1f}A | Weight: {hip_preset['mass_kg']*1000:.0f}g")
    with mc_k:
        st.info(f"**Knee Motor ({knee_preset_name.split(' — ')[0]})**: Peak: {knee_peak:.2f} Nm | Cont: {knee_cont:.2f} Nm | Speed: {knee_rpm:.0f} RPM | Voltage: {knee_volts:.1f}V | Stall Curr: {knee_curr:.1f}A | Weight: {knee_preset['mass_kg']*1000:.0f}g")

# Dual-Motor Standing Height Recommendation Engine
_thigh_m = max(links_mass / 8, .001)
_shank_m = max(links_mass / 8, .001)

rec_max_h_mm = recommend_max_standing_height_for_motor(
    hip_peak_nm=hip_peak, hip_cont_nm=hip_cont,
    knee_peak_nm=knee_peak, knee_cont_nm=knee_cont,
    dof_total=dof_total, femur_fraction=femur_fraction, neutral_knee_deg=neutral_knee_deg,
    total_mass_kg=robot_mass, payload_kg=payload_mass, thigh_mass_kg=_thigh_m,
    shank_mass_kg=_shank_m, stance_legs=2, dynamic_accel=1.0,
    impact_factor=1.5, efficiency=efficiency, safety_factor=safety_factor
)

st.markdown("### 🎯 Motor Capabilities & Standing Height Recommendation")
rc1, rc2, rc3 = st.columns(3)
rc1.metric("Robot Total System Mass", f"{total_system_mass:.2f} kg ({total_system_mass*1000:.0f} g)")
with rc2:
    if rec_max_h_mm is not None:
        st.metric("Max Safe Standing Height for Motors", f"{rec_max_h_mm} mm")
    else:
        st.metric("Max Safe Standing Height", "0 mm (Motors Too Weak)")
rc3.metric("Motor Peak Torque (Hip / Knee)", f"{hip_peak:.2f} / {knee_peak:.2f} Nm")

if rec_max_h_mm is None:
    st.error(f"⛔ MOTORS TOO WEAK: The selected motor combination cannot support a 50 mm standing height for a {total_system_mass:.2f} kg robot. Upgrade Hip or Knee motor selection!")
elif standing_height * MM > rec_max_h_mm:
    st.error(f"⛔ CANNOT GO BEYOND THIS VALUE ({rec_max_h_mm} mm): Target standing height ({standing_height*MM:.0f} mm) exceeds the maximum safe height ({rec_max_h_mm} mm) for your selected motor combination! "
             f"The selected motors can safely support a standing height up to **{rec_max_h_mm} mm** for a {total_system_mass:.2f} kg robot. "
             f"Reduce standing height in the wizard to ≤ {rec_max_h_mm} mm, or select higher-torque motors.")
else:
    st.success(f"✅ TARGET HEIGHT OK: Your target standing height ({standing_height*MM:.0f} mm) is within the motors' safe max height limit of **{rec_max_h_mm} mm**.")

# ---- Beginner-Friendly Joint Torque Calculator -------------------------------------
st.divider()
st.subheader("Joint torque calculator (Student-Friendly Guide)")
st.caption("This tool calculates how much torque (twisting force) your motors need to carry the robot's weight, walk, and absorb landing impacts.")

tc1, tc2, tc3, tc4 = st.columns(4)
with tc1:
    stance_legs = st.selectbox(
        "Number of stance legs", [1, 2, 3, 4], index=1,
        help="❓ HOW MANY FEET TOUCH THE GROUND TOGETHER:\n"
             "• 4 stance legs (Walk gait) = Robot weight is shared across 4 legs (EASIEST on motors).\n"
             "• 2 stance legs (Trot gait) = Robot weight is shared across 2 legs (DOUBLE the load per motor!).\n"
             "• 1 stance leg (Hop/Jump) = Single leg carries whole robot.\n\n"
             "💡 Rule of Thumb: If your motors are failing, switch to 4 stance legs or reduce robot mass."
    )
    dynamic_accel = st.number_input(
        "Dynamic acceleration (m/s²)", 0.0, 30.0, 1.0, .1,
        help="❓ EXTRA PUSHING FORCE WHEN STARTING OR STOPPING:\n"
             "• 0 m/s² = Standing completely still.\n"
             "• 1.0–2.0 m/s² = Normal comfortable walking.\n"
             "• >4.0 m/s² = Fast sprinting / aggressive direction changes.\n\n"
             "💡 Increasing this value increases the required motor torque."
    )
with tc2:
    impact_factor = st.number_input(
        "Impact factor (ground hit multiplier)", 1.0, 10.0, 1.5, .1,
        help="❓ SHOCK FORCE WHEN FOOT STRIKES THE GROUND:\n"
             "• 1.0 = Smooth crawling with no impact.\n"
             "• 1.5 = Normal stepping on flat floor.\n"
             "• 2.5 = Landing from a hop or stepping onto stairs.\n\n"
             "💡 Higher impact factor means the motor needs more PEAK STALL TORQUE to avoid collapsing on touchdown."
    )
    thigh_mass = st.number_input(
        "One thigh mass (kg)", .001, 5.0, max(links_mass/8, .001), .001,
        help="❓ WEIGHT OF ONE UPPER LEG (THIGH):\n"
             "Lightweight 3D-printed or carbon fiber legs require much less motor torque to swing fast!"
    )
with tc3:
    shank_mass = st.number_input(
        "One shank mass (kg)", .001, 5.0, max(links_mass/8, .001), .001,
        help="❓ WEIGHT OF ONE LOWER LEG (SHANK):\n"
             "The lower leg moves fastest; keep it as light as possible!"
    )
    thigh_com = st.slider(
        "Thigh COM from hip (% length)", 0, 100, 50,
        help="❓ CENTER OF MASS LOCATION ALONG THIGH:\n"
             "• 50% = Mass is evenly spread in the middle.\n"
             "• 20% = Mass is concentrated near the top hip joint (easier to swing!)."
    ) / 100
with tc4:
    shank_com = st.slider(
        "Shank COM from knee (% length)", 0, 100, 50,
        help="❓ CENTER OF MASS LOCATION ALONG SHANK:\n"
             "• 50% = Mass in middle.\n"
             "• 20% = Mass near top knee joint."
    ) / 100
    st.write(f"Safety Multiplier: **{safety_factor:.1f}×**")

budget = joint_torque_budget(hip_flexion=q_hip, knee_flexion=q_knee, hip_abduction=q_ab,
    hip_offset=derived["l1"], thigh_length=thigh, shank_length=shank, total_mass_kg=robot_mass,
    payload_kg=payload_mass, thigh_mass_kg=thigh_mass, shank_mass_kg=shank_mass,
    thigh_com_frac=thigh_com, shank_com_frac=shank_com, legs_in_stance=stance_legs,
    dynamic_accel_mps2=dynamic_accel, impact_factor=impact_factor,
    transmission_efficiency=efficiency, safety_factor=safety_factor)

torque_table = pd.DataFrame({name: {"Hip joint": vals["hip_nm"], "Knee joint": vals["knee_nm"]}
                             for name, vals in budget.items()
                             if isinstance(vals, dict) and "hip_nm" in vals}).round(2)
st.dataframe(torque_table, use_container_width=True)

# Evaluate Motor Verdicts (Hip Motor evaluated against Hip budget, Knee Motor evaluated against Knee budget)
eval_hip = evaluate_motor(
    motor_continuous_nm=hip_cont, motor_peak_nm=hip_peak,
    motor_rated_rpm=hip_rpm, gearbox_ratio=1.0,
    transmission_efficiency=efficiency,
    required_continuous_nm=budget["continuous_required"]["hip_nm"],
    required_peak_nm=budget["peak_required"]["hip_nm"],
    required_max_speed_rad_s=4.0,
    rating_basis="Servo output shaft (already geared)"
)

eval_knee = evaluate_motor(
    motor_continuous_nm=knee_cont, motor_peak_nm=knee_peak,
    motor_rated_rpm=knee_rpm, gearbox_ratio=1.0,
    transmission_efficiency=efficiency,
    required_continuous_nm=budget["continuous_required"]["knee_nm"],
    required_peak_nm=budget["peak_required"]["knee_nm"],
    required_max_speed_rad_s=4.0,
    rating_basis="Servo output shaft (already geared)"
)

evals = {"Hip joint": eval_hip, "Knee joint": eval_knee}

for name, result in evals.items():
    if result['verdict'] == "FAIL":
        st.error(f"❌ **{name}: `FAIL`** — " + " ".join(result["reasons"]))
    elif result['verdict'] == "MARGINAL":
        st.warning(f"⚠️ **{name}: `MARGINAL`** — " + " ".join(result["reasons"]))
    else:
        st.success(f"✅ **{name}: `PASS`** — " + " ".join(result["reasons"]))

# ---- Smart Automatic Battery & Power Recommender -----------------------------------
st.divider()
st.subheader("Smart Servo Power & Battery Recommender")
st.caption("Automatically computes ideal battery capacity, voltage, and discharge C-rating based on electrical specs of selected Hip & Knee motors.")

bc1, bc2 = st.columns(2)
with bc1:
    target_runtime_min = st.slider("Desired walking runtime (minutes)", 5, 120, 20, 5,
                                   help="How long you want the robot to walk on a single battery charge.")
    battery_type = st.selectbox(
        "Battery chemistry preference",
        ["LiPo (Lithium Polymer) — High performance, light", "Li-ion 18650 — Good runtime, standard", "NiMH / External Power Supply"],
        index=0
    )
with bc2:
    target_volts = max(hip_volts, knee_volts)
    st.markdown(f"**Electrical Profile ({dof_total} Servos):**")
    st.write(f"• Hip Motor Voltage: **{hip_volts:.1f} V** | Stall Curr: **{hip_curr:.1f} A**")
    st.write(f"• Knee Motor Voltage: **{knee_volts:.1f} V** | Stall Curr: **{knee_curr:.1f} A**")

# Calculate Dual Motor Battery Requirements
if dof_total == 8:
    num_hip = 4
    num_knee = 4
else:
    num_hip = 8
    num_knee = 4

total_avg_current = (num_hip * 0.5 * hip_avg_curr) + (num_knee * 0.5 * knee_avg_curr) + 0.35
total_peak_current = (num_hip * hip_curr) + (num_knee * knee_curr) + 0.35

req_capacity_mah = int(np.ceil((total_avg_current * (target_runtime_min / 60.0) / 0.80) * 1000 / 100) * 100)
req_c_rating = int(np.ceil(total_peak_current / (req_capacity_mah / 1000.0)))

lipo_cells = "2S (7.4V)" if target_volts <= 7.4 else ("3S (11.1V)" if target_volts <= 11.1 else "4S (14.8V)")

st.markdown("#### 🔋 Recommended Battery Specification:")
bp1, bp2, bp3, bp4 = st.columns(4)
bp1.metric("Recommended Voltage", f"{target_volts:.1f} V ({lipo_cells})")
bp2.metric("Min Battery Capacity", f"{req_capacity_mah} mAh")
bp3.metric("Min Discharge C-Rating", f"{req_c_rating}C or higher")
bp4.metric("Peak Current Draw", f"{total_peak_current:.1f} A")

st.info(f"💡 **Shopping Recommendation**: Buy a **{lipo_cells} {req_capacity_mah} mAh LiPo Battery with at least {req_c_rating}C discharge rating**. "
        f"For {dof_total} servos, use a dedicated fused UBEC power distribution board rated for at least **{total_peak_current*0.75:.0f}A continuous**.")

# ---- 2D Stair-climbing simulation --------------------------------------------------
st.divider()
st.subheader("2D stair-climbing and descending simulation")
st.caption("Sagittal-plane leg check for a single normal stair. It tests the requested paw landing against reach and joint-angle limits.")
stair1, stair2, stair3, stair4 = st.columns(4)
with stair1:
    riser = mm_input("Stair riser height (mm)", 175, 1, 500, 1)
    tread = mm_input("Stair tread depth (mm)", 280, 10, 600, 1)
with stair2:
    landing_x = mm_input("Paw landing beyond riser (mm)", 180, 1, 1000, 1)
    stair_direction = st.radio("Test direction", ["Climb", "Descend"], horizontal=True)
with stair3:
    hip_min = st.number_input("Hip joint minimum (°)", -180.0, 0.0, -90.0, 1.0)
    hip_max = st.number_input("Hip joint maximum (°)", 0.0, 180.0, 90.0, 1.0)
with stair4:
    knee_min = st.number_input("Knee joint minimum (°)", -180.0, 180.0, 0.0, 1.0)
    knee_max = st.number_input("Knee joint maximum (°)", -180.0, 180.0, 150.0, 1.0)

stair_z = -standing_height + riser if stair_direction == "Climb" else -standing_height - riser
stair_target = (landing_x, 0.0, stair_z)
stair_sol, stair_issues = assess_stair_target(stair_target, thigh, shank, derived["l1"],
                                               knee_forward, hip_min, hip_max, knee_min, knee_max)
stair_status, stair_figure = st.columns([1, 2])
with stair_status:
    if stair_sol is None:
        st.error(f"{stair_direction.upper()}: NOT REACHABLE")
    elif stair_issues:
        st.warning(f"{stair_direction.upper()}: REACHABLE, BUT JOINT-LIMIT FAILURE")
    else:
        st.success(f"{stair_direction.upper()}: KINEMATICALLY FEASIBLE")
    if stair_sol is not None:
        _, stair_hip, stair_knee = stair_sol
        st.metric("Hip joint on stair", f"{np.degrees(stair_hip):.1f}°")
        st.metric("Knee joint on stair", f"{np.degrees(stair_knee):.1f}°")
    for issue in stair_issues:
        st.write(f"• {issue}")
    feasible_risers = []
    for candidate_mm in range(10, 501, 5):
        candidate_r = candidate_mm / MM
        candidate_z = -standing_height + candidate_r if stair_direction == "Climb" else -standing_height - candidate_r
        _, candidate_issues = assess_stair_target((landing_x, 0.0, candidate_z), thigh, shank,
                                                   derived["l1"], knee_forward, hip_min, hip_max,
                                                   knee_min, knee_max)
        if not candidate_issues:
            feasible_risers.append(candidate_mm)
    if feasible_risers:
        max_riser = max(feasible_risers)
        st.info(f"{stair_direction} recommendation: maximum kinematic riser here is about {max_riser} mm.")
with stair_figure:
    if stair_sol is None:
        st.pyplot(stair_pose(thigh, shank, 0, 0, stair_target, standing_height, riser, tread,
                              stair_direction, False), use_container_width=True)
    else:
        st.pyplot(stair_pose(thigh, shank, stair_sol[1], stair_sol[2], stair_target,
                              standing_height, riser, tread, stair_direction, True), use_container_width=True)

# Stair-climb torque analysis
if stair_sol is not None and stair_direction == "Climb":
    st.subheader("Stair-climb torque analysis")
    stair_torque = stair_climb_torque_budget(
        hip_flexion_flat=q_hip, knee_flexion_flat=q_knee,
        hip_flexion_stair=stair_sol[1], knee_flexion_stair=stair_sol[2],
        hip_abduction=q_ab, hip_offset=derived["l1"],
        thigh_length=thigh, shank_length=shank,
        total_mass_kg=robot_mass, payload_kg=payload_mass,
        thigh_mass_kg=thigh_mass, shank_mass_kg=shank_mass,
        thigh_com_frac=thigh_com, shank_com_frac=shank_com,
        riser_height_m=riser, transmission_efficiency=efficiency,
        safety_factor=safety_factor)
    st1, st2, st3, st4 = st.columns(4)
    st1.metric("Climb leg hip", f"{stair_torque['stair_leg_hip_nm']:.2f} Nm")
    st2.metric("Climb leg knee", f"{stair_torque['stair_leg_knee_nm']:.2f} Nm")
    st3.metric("Support leg hip", f"{stair_torque['support_leg_hip_nm']:.2f} Nm")
    st4.metric("Support leg knee", f"{stair_torque['support_leg_knee_nm']:.2f} Nm")
    sc1, sc2 = st.columns(2)
    sc1.metric("Energy per step", f"{stair_torque['climb_energy_j']:.2f} J")
    sc2.metric("Min motor peak required", f"{stair_torque['min_motor_peak_nm']:.2f} Nm")

# ---- Dynamic test cases & Optimiser ------------------------------------------------
st.divider()
st.subheader("Dynamic test cases")
TESTS = {"Standing": (4, 0.0, 1.0), "Crouching": (4, .5, 1.1), "Walking": (3, 1.0, 1.3), "Trotting": (2, 2.0, 1.7), "Acceleration": (2, 4.0, 1.5), "Stopping": (2, 4.0, 1.5), "Slope standing": (3, 1.5, 1.2), "One-leg disturbance": (3, 3.0, 1.8), "Landing / impact": (2, 3.0, 2.5)}
rows = []
for name, (legs, accel, impact) in TESTS.items():
    test = joint_torque_budget(hip_flexion=q_hip, knee_flexion=q_knee, hip_abduction=q_ab, hip_offset=derived["l1"], thigh_length=thigh, shank_length=shank, total_mass_kg=robot_mass, payload_kg=payload_mass, thigh_mass_kg=thigh_mass, shank_mass_kg=shank_mass, thigh_com_frac=thigh_com, shank_com_frac=shank_com, legs_in_stance=legs, dynamic_accel_mps2=accel, impact_factor=impact, transmission_efficiency=efficiency, safety_factor=safety_factor)
    rows.append({"Test": name, "Stance legs": legs, "Hip peak (Nm)": round(test["peak_required"]["hip_nm"],2), "Knee peak (Nm)": round(test["peak_required"]["knee_nm"],2)})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with st.expander("AI-assisted geometry recommendation (offline numerical optimiser)"):
    st.caption("This is a local differential-evolution optimiser. It searches valid thigh/shank proportions to reduce peak knee torque.")
    if optimize_leg_geometry is None:
        st.warning("Install the optional optimiser dependency with `pip install -r requirements.txt` to enable this panel.")
    elif st.button("Find lower-torque leg proportions"):
        result = optimize_leg_geometry(standing_height, robot_mass, payload_mass, thigh_mass, shank_mass, stance_legs, dynamic_accel, impact_factor, efficiency, safety_factor, has_abad, derived["l1"])
        st.success(f"Suggested thigh {result['thigh_length_m']*MM:.1f} mm, shank {result['shank_length_m']*MM:.1f} mm; estimated knee-peak improvement {result['improvement_pct']:.1f}%.")

# ---- Mass breakdown (at the end of the page) ----------------------------------------
st.divider()
st.subheader("Mass breakdown")
ms1, ms2, ms3, ms4 = st.columns(4)
with ms1:
    _motor_mass = st.number_input("Motor/servo mass (kg total all motors)", 0.005, 10.0,
                                   float(st.session_state.get("mass_motor", preset_motor_mass_total)),
                                   0.01, key="mass_motor_input")
    st.session_state["mass_motor"] = _motor_mass
    _battery_mass = st.number_input("Battery mass (kg)", 0.01, 10.0,
                                     float(st.session_state.get("mass_battery", 0.20)),
                                     0.01, key="mass_battery_input")
    st.session_state["mass_battery"] = _battery_mass
with ms2:
    _frame_mass = st.number_input("Frame/chassis mass (kg)", 0.01, 20.0,
                                   float(st.session_state.get("mass_frame", 0.25)),
                                   0.01, key="mass_frame_input")
    st.session_state["mass_frame"] = _frame_mass
    _links_mass = st.number_input("All links mass (kg, total 4 legs)", 0.01, 20.0,
                                   float(st.session_state.get("mass_links", 0.15)),
                                   0.01, key="mass_links_input")
    st.session_state["mass_links"] = _links_mass
with ms3:
    _payload_mass = st.number_input("Payload mass (kg)", 0.0, 50.0,
                                     float(st.session_state["mass_payload"]), 0.01,
                                     key="mass_payload_input",
                                     help="Extra weight carried on top of the robot. Synchronized with sidebar input.")
    st.session_state["mass_payload"] = _payload_mass
    _payload_x = st.number_input("Payload fore/aft offset (mm)", -500.0, 500.0,
                                  float(st.session_state.get("mass_payload_x", 0.0)),
                                  1.0, key="mass_payload_x_input")
    st.session_state["mass_payload_x"] = _payload_x
with ms4:
    st.markdown("**Fixed allowances**")
    st.caption(f"ESP32/controller: {esp32_mass*1000:.0f} g")
    st.caption(f"PCB: {pcb_mass*1000:.0f} g")
    st.caption(f"Sensors: {sensor_mass*1000:.0f} g")
    st.caption(f"Wiring: {wiring_mass*1000:.0f} g")
    st.caption(f"Fasteners: {fastener_mass*1000:.0f} g")

_components = {
    "Motors": _motor_mass,
    "Battery": _battery_mass,
    "Frame": _frame_mass,
    "Links (4 legs)": _links_mass,
    "Payload": _payload_mass,
    "ESP32": esp32_mass,
    "PCB": pcb_mass,
    "Sensors": sensor_mass,
    "Wiring": wiring_mass,
    "Fasteners": fastener_mass,
}
_total = sum(_components.values())

pie_col, summary_col = st.columns([1.5, 1])
with pie_col:
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [f"{k} ({v*1000:.0f}g)" for k, v in _components.items() if v > 0.001]
    sizes = [v for v in _components.values() if v > 0.001]
    colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                                       startangle=140, textprops={'fontsize': 8})
    for t in autotexts:
        t.set_fontsize(7)
    ax.set_title("Mass distribution", fontsize=10)
    st.pyplot(fig, use_container_width=True)
with summary_col:
    st.metric("Total system mass", f"{_total:.3f} kg ({_total*1000:.0f} g)")
    st.metric("Robot mass (no payload)", f"{(_total - _payload_mass):.3f} kg")
    st.metric("Mass per leg", f"{((_motor_mass / 4) + _links_mass/4)*1000:.0f} g")

# ---- BUILD DATA FOR PDF GENERATION & RENDER DOWNLOAD BUTTON -----------------------
pdf_data = {
    'dof_total': dof_total,
    'standing_height_mm': standing_height * MM,
    'total_mass_kg': _total,
    'terrain': terrain,
    'motor_mode_name': motor_mode_name,
    'motor_mode': motor_mode,
    'hip_preset_name': hip_preset_name,
    'knee_preset_name': knee_preset_name,
    'hip_spec': hip_preset,
    'knee_spec': knee_preset,
    'hip_verdict': eval_hip['verdict'],
    'knee_verdict': eval_knee['verdict'],
    'max_safe_height_mm': rec_max_h_mm if rec_max_h_mm is not None else "N/A (Motor Weak)",
    'l1_mm': derived['l1'] * MM,
    'l2_mm': thigh * MM,
    'l3_mm': shank * MM,
    'foot_x_mm': foot_x * MM,
    'foot_z_mm': foot_z * MM,
    'q_hip_deg': np.degrees(q_hip),
    'q_knee_deg': np.degrees(q_knee),
    'jacobian_matrix': leg_jacobian(q_ab, q_hip, q_knee, derived["l1"], thigh, shank) if sol is not None else None,
    'jacobian_condition': jacobian_condition if jacobian_condition is not None else 1.0,
    'torque_budget': budget,
    'battery_calc': {
        'voltage_cell': lipo_cells,
        'req_capacity_mah': req_capacity_mah,
        'req_c_rating': req_c_rating,
        'total_peak_current': total_peak_current,
        'target_runtime_min': target_runtime_min
    },
    'mass_components': _components
}

try:
    pdf_bytes = generate_quadruped_pdf_report(pdf_data)
    pdf_button_container.download_button(
        "📄 Export Engineering PDF Report",
        data=pdf_bytes,
        file_name=f"Quadruped_Robot_Engineering_Dossier_{dof_total}DOF.pdf",
        mime="application/pdf",
        help="Click to generate and download a complete multi-page PDF engineering calculation dossier to present to your professor, team, or supervisor!"
    )
except Exception as pdf_err:
    pdf_button_container.warning(f"📄 PDF Exporter: {pdf_err}")
