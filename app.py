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
import gait as gait_mod
import pdf_generator

importlib.reload(kinematics)
importlib.reload(dynamics)
importlib.reload(motor)
importlib.reload(gait_mod)
importlib.reload(pdf_generator)

from kinematics import (derive_dimensions, leg_fk, leg_ik, leg_jacobian,
                        jacobian_condition_number, compute_workspace_boundary,
                        manipulability_index)
from dynamics import (SAFETY_PRESETS, TRANSMISSION_EFFICIENCY_PRESETS,
                      joint_torque_budget, stair_climb_torque_budget,
                      compute_mass_matrix)
from motor import evaluate_motor, SERVO_PRESETS, torque_at_speed, motor_operating_point
from gait import (estimate_max_joint_speed, recommend_gait, worst_case_gait_margin,
                  GAITS)
from pdf_generator import generate_quadruped_pdf_report

try:
    from optimizer import (optimize_leg_geometry, optimize_leg_proportions_for_motors,
                           suggest_cheaper_motor_combination)
except ImportError:
    optimize_leg_geometry = None
    optimize_leg_proportions_for_motors = None
    suggest_cheaper_motor_combination = None

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
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
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

# ---- Initialize session state variables for bidirectional synchronization & values ----
session_defaults = {
    "mass_payload": 0.0,
    "mass_motor": 0.48,
    "mass_battery": 0.20,
    "mass_frame": 0.25,
    "mass_links": 0.15,
    "mass_payload_x": 0.0,
    "knee_forward": True,
    "use_custom": False,
    "thigh_length_mm_custom": 180.0,
    "shank_length_mm_custom": 180.0,
    "foot_x_mm": 0.0,
    "foot_y_mm": 0.0,
    "foot_z_mm": -350.0,
}

for k, v in session_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---- TOP TITLE HEADER & DOWNLOAD PDF CONTAINER -------------------------------------------
col_title, col_pdf = st.columns([3.2, 1.2])
with col_title:
    st.title("🐕 Quadruped Robot Design Dashboard")
    st.caption("A multi-disciplinary design wizard: Geometry & Kinematics → Motor Selection & Torque Budget → Gaits & Curves → Mass & Power.")

pdf_button_container = col_pdf.container()

# ---- GLOBAL SIDEBAR CONTROLS -------------------------------------------------------------
with st.sidebar:
    st.header("🌐 Global Parameters")
    st.caption("Configure the baseline payload, degrees of freedom, terrain, and safety factors.")
    
    dof_total = st.radio("Total degrees of freedom", [8, 12], index=0, horizontal=True)
    terrain = st.selectbox("Primary terrain type", ["Indoor flat floor", "Carpet / grass", "Small stairs", "Uneven outdoor ground"])
    
    # Bidirectionally synchronized payload input
    sidebar_payload = st.number_input(
        "Expected payload (kg)", 0.0, 50.0,
        float(st.session_state["mass_payload"]), 0.01,
        key="sidebar_payload_input",
        help="Extra weight carried on top of the robot chassis. Instantly synchronized across tabs."
    )
    st.session_state["mass_payload"] = sidebar_payload
    
    st.divider()
    st.header("🛠️ Engineering Assumptions")
    safety_name = st.selectbox("Safety factor multiplier", list(SAFETY_PRESETS), index=2,
                               help="Applied to peak joint torque for shock load and dynamic safety margin.")
    safety_factor = SAFETY_PRESETS[safety_name]
    
    transmission_name = st.selectbox("Transmission efficiency preset", list(TRANSMISSION_EFFICIENCY_PRESETS), index=1)
    efficiency = TRANSMISSION_EFFICIENCY_PRESETS[transmission_name]

# ---- DEFINE 4 WORKFLOW TABS --------------------------------------------------------------
tab_geom, tab_actuators, tab_gaits, tab_power_opt = st.tabs([
    "📐 Step 1: Geometry & Kinematics",
    "⚙️ Step 2: Actuators & Torque Budget",
    "🏃 Step 3: Gait Sizing & Torque-Speed Curves",
    "🔋 Step 4: Mass, Power & Optimizers"
])

# =========================================================================================
# TAB 1: GEOMETRY & KINEMATICS
# =========================================================================================
with tab_geom:
    st.subheader("📐 Physical Geometry Wizard & Kinematics Simulator")
    
    # Geometry sliders
    gcol1, gcol2, gcol3 = st.columns(3)
    with gcol1:
        standing_height = mm_input(
            "Target standing height (mm)", 350.0, 50.0, 1500.0, 10.0,
            help="Target height measured from the hip axis to ground at neutral standing posture."
        )
    with gcol2:
        femur_fraction = st.slider("Thigh/femur share of total leg length (%)", 35, 65, 52, 1) / 100
    with gcol3:
        neutral_knee_deg = st.slider(
            "Neutral standing knee bend (degrees)", 15, 120, 55, 1,
            help="Bent knee avoids geometric singularity. 45°-70° is standard."
        )
        
    derived = posture_dimensions(dof_total, standing_height, femur_fraction, neutral_knee_deg)
    has_abad = derived["has_abad"]
    
    # Proportions metrics bar
    st.markdown("##### 📐 Current Physical Proportions:")
    pcol_1, pcol_2, pcol_3, pcol_4 = st.columns(4)
    pcol_1.metric("Target Standing Height", f"{standing_height*MM:.1f} mm")
    pcol_2.metric("Femur / Thigh Length", f"{derived['l2']*MM:.1f} mm")
    pcol_3.metric("Shank / Lower Leg", f"{derived['l3']*MM:.1f} mm")
    pcol_4.metric("Max Extended Leg Length", f"{derived['max_extension']*MM:.1f} mm")
    
    # Check stair gate height constraints immediately if terrain is stairs
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
            st.error(f"⛔ **HEIGHT FAILURE FOR STAIRS**: Target height ({standing_height*MM:.0f} mm) is too small to clear a standard 178 mm stair riser. Minimum required height: **{stair_min_height} mm**.")
        else:
            st.success(f"✅ Height clears standard stair climbing gate (≥ {stair_min_height} mm).")

    st.divider()
    
    # Kinematics Simulator Layout
    sim1, sim2 = st.columns([1, 1.2])
    with sim1:
        st.markdown("#### 🎮 Paw Target Displacement (Hip Frame)")
        max_reach = derived["max_extension"]
        
        foot_x = st.slider("Paw fore / aft offset (mm, X)", int(-max_reach * MM), int(max_reach * MM), int(st.session_state["foot_x_mm"]), key="foot_x_slider") / MM
        st.session_state["foot_x_mm"] = foot_x * MM
        
        if has_abad:
            foot_y = st.slider("Paw lateral offset (mm, Y)", int(-max_reach * MM), int(max_reach * MM), int(st.session_state["foot_y_mm"]), key="foot_y_slider") / MM
            st.session_state["foot_y_mm"] = foot_y * MM
        else:
            foot_y = 0.0
            st.caption("ℹ️ *8-DOF configuration: lateral leg offset (Y) is fixed.*")
            
        foot_z = st.slider("Paw vertical height (mm, Z; down is negative)", int(-max_reach * MM), -1, int(st.session_state["foot_z_mm"]), key="foot_z_slider") / MM
        st.session_state["foot_z_mm"] = foot_z * MM
        
        knee_forward = st.toggle("Knee-forward configuration", st.session_state["knee_forward"], key="knee_forward_toggle")
        st.session_state["knee_forward"] = knee_forward
        
        use_custom = st.toggle("Manually override leg link lengths", st.session_state["use_custom"], key="use_custom_toggle")
        st.session_state["use_custom"] = use_custom
        
        if use_custom:
            thigh = st.number_input("Custom Thigh length (mm)", 10.0, 1000.0, float(st.session_state["thigh_length_mm_custom"]), 1.0, key="thigh_length_custom") / MM
            st.session_state["thigh_length_mm_custom"] = thigh * MM
            shank = st.number_input("Custom Shank length (mm)", 10.0, 1000.0, float(st.session_state["shank_length_mm_custom"]), 1.0, key="shank_length_custom") / MM
            st.session_state["shank_length_mm_custom"] = shank * MM
        else:
            thigh, shank = derived["l2"], derived["l3"]
            
        # Run Inverse Kinematics solver
        sol = leg_ik(foot_x, foot_y, foot_z, derived["l1"], thigh, shank, knee_forward)
        
        if sol is None:
            st.error("❌ **UNREACHABLE TARGET**: Paw placement lies outside the leg's physical workspace boundaries.")
            q_ab, q_hip, q_knee = 0.0, 0.0, 0.0
            jacobian_condition = 999.0
        else:
            q_ab, q_hip, q_knee = sol
            st.success("✅ **REACHABLE POSTURE**")
            
            # Displays joint feedback
            r1, r2, r3 = st.columns(3)
            r1.metric("Solved Hip Ab/Ad", f"{np.degrees(q_ab):.1f}°" if has_abad else "N/A")
            r2.metric("Solved Hip Pitch", f"{np.degrees(q_hip):.1f}°")
            r3.metric("Solved Knee Pitch", f"{np.degrees(q_knee):.1f}°")
            
            check_fk = leg_fk(q_ab, q_hip, q_knee, derived["l1"], thigh, shank)
            st.caption(f"FK check error: {np.linalg.norm(check_fk-np.array([foot_x, foot_y, foot_z]))*MM:.4f} mm")

    with sim2:
        if sol is not None:
            st.pyplot(plot_leg(thigh, shank, q_hip, q_knee, (foot_x, foot_y, foot_z), True), use_container_width=True)
        else:
            st.pyplot(plot_leg(thigh, shank, 0, 0, (foot_x, foot_y, foot_z), False), use_container_width=True)
            
    st.divider()
    
    # Advanced kinematics diagnostics
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.markdown("#### 🌐 Reachable Workspace Bounds")
        ws = compute_workspace_boundary(derived["l1"], thigh, shank)
        
        fig_ws, ax_ws = plt.subplots(figsize=(6, 4))
        bp = np.array(ws['boundary_points'])
        ax_ws.fill(bp[:, 0]*MM, bp[:, 1]*MM, alpha=0.15, color='dodgerblue', label='Reachable workspace')
        ax_ws.plot(bp[:, 0]*MM, bp[:, 1]*MM, 'b-', linewidth=1.0, alpha=0.5)
        ax_ws.plot(foot_x*MM, foot_z*MM, 'r*', markersize=12, label='Current target')
        ax_ws.set_xlabel('X (mm, forward)')
        ax_ws.set_ylabel('Z (mm, downward)')
        ax_ws.set_title('Sagittal Plane Workspace')
        ax_ws.legend(fontsize=8)
        ax_ws.set_aspect('equal')
        ax_ws.grid(True, alpha=0.3)
        st.pyplot(fig_ws, use_container_width=True)
        
    with dcol2:
        st.markdown("#### 🔬 Singularity & Dynamic Coupling Matrix")
        if sol is not None:
            J = leg_jacobian(q_ab, q_hip, q_knee, derived["l1"], thigh, shank)
            jacobian_condition = jacobian_condition_number(q_ab, q_hip, q_knee, derived["l1"], thigh, shank)
            w_idx = manipulability_index(q_ab, q_hip, q_knee, derived["l1"], thigh, shank)
            
            st.metric("Jacobian Condition Number", f"{jacobian_condition:.2f}",
                      help="Measures directional sensitivity. Near 1 = isotropic; >50 = near singularity.")
            st.metric("Manipulability Index (w)", f"{w_idx:.4f}",
                      help="√det(J Jᵀ). Closer to 0 means robot loses ability to push in certain directions.")
            
            if jacobian_condition > 50:
                st.warning("⚠️ Singular Pose: Small paw movements require extremely high joint speeds.")
            else:
                st.success("✅ Kinematically well-conditioned posture.")
                
            with st.expander("Show Kinematics Jacobian Matrix J(q)"):
                st.dataframe(pd.DataFrame(J, index=["x", "y", "z"], columns=["ab/ad", "hip", "knee"]).round(4))
                
            with st.expander("Show 2-Link Mass Matrix M(q)"):
                _links_mass_est = st.session_state.get("mass_links", 0.15)
                _thigh_m = _links_mass_est / 8
                _shank_m = _links_mass_est / 8
                M = compute_mass_matrix(q_hip, q_knee, _thigh_m, _shank_m, 0.5, 0.5, thigh, shank)
                st.dataframe(pd.DataFrame(M, index=["hip", "knee"], columns=["hip", "knee"]).round(6))
                st.caption("τ_inertia = M(q)·q̈. Diagonal represents self-inertia; off-diagonals show acceleration coupling.")
        else:
            st.warning("Diagnostics unavailable: foot target is outside workspace.")

# =========================================================================================
# TAB 2: ACTUATORS & TORQUE BUDGET
# =========================================================================================
with tab_actuators:
    st.subheader("⚙️ Joint Actuator Selection & Dynamic Torque Sizing")
    
    # Motor Mode Selector
    motor_mode_name = st.radio(
        "Servo Configuration Mode",
        [
            "Case 1: Same Motor Model for All Joints (Identical Hip & Knee)",
            "Case 2: Independent Motors per Joint (Higher spec Hip, Lower spec Knee)"
        ],
        index=0,
        horizontal=True,
        help="• Case 1: Simplifies bill of materials (same servo everywhere).\n• Case 2: Optimizes weight and cost (smaller knee servo)."
    )
    motor_mode = "same" if motor_mode_name.startswith("Case 1") else "independent"
    
    # Preset selectors
    if motor_mode == "same":
        servo_preset_name = st.selectbox(
            "Select Servo / Motor Preset Model (All Joints)",
            list(SERVO_PRESETS.keys()), index=2,
            help="Applies selected model to both Hip and Knee pitch joints."
        )
        hip_preset = SERVO_PRESETS[servo_preset_name]
        knee_preset = SERVO_PRESETS[servo_preset_name]
        hip_preset_name = servo_preset_name
        knee_preset_name = servo_preset_name
    else:
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            hip_preset_name = st.selectbox(
                "Select Hip Joint Servo Model",
                list(SERVO_PRESETS.keys()), index=3, # e.g. DS3235
                help="Applies model to the Hip pitch joint (carries larger lever arms)."
            )
            hip_preset = SERVO_PRESETS[hip_preset_name]
        with mcol2:
            knee_preset_name = st.selectbox(
                "Select Knee Joint Servo Model",
                list(SERVO_PRESETS.keys()), index=2, # e.g. DS3218
                help="Applies model to the Knee pitch joint (carries lower mass load)."
            )
            knee_preset = SERVO_PRESETS[knee_preset_name]
            
    # Extract specs
    hip_cont, hip_peak, hip_rpm, hip_volts, hip_curr = hip_preset["continuous_nm"], hip_preset["peak_nm"], float(hip_preset["rpm"]), hip_preset["voltage"], hip_preset["current_a"]
    hip_avg_curr = hip_preset.get("avg_current_a", hip_curr * 0.25)
    
    knee_cont, knee_peak, knee_rpm, knee_volts, knee_curr = knee_preset["continuous_nm"], knee_preset["peak_nm"], float(knee_preset["rpm"]), knee_preset["voltage"], knee_preset["current_a"]
    knee_avg_curr = knee_preset.get("avg_current_a", knee_curr * 0.25)
    
    # Calculate preset mass and total cost
    if dof_total == 8:
        preset_motor_mass_total = 4 * hip_preset["mass_kg"] + 4 * knee_preset["mass_kg"]
        total_motor_cost = 4 * hip_preset.get("approx_price_inr", 350) + 4 * knee_preset.get("approx_price_inr", 350)
    else:
        preset_motor_mass_total = 8 * hip_preset["mass_kg"] + 4 * knee_preset["mass_kg"]
        total_motor_cost = 8 * hip_preset.get("approx_price_inr", 350) + 4 * knee_preset.get("approx_price_inr", 350)
    st.session_state["mass_motor"] = preset_motor_mass_total
    
    # Display catalog specifications
    st.markdown("##### 📋 Official Datasheet Specifications:")
    if motor_mode == "same":
        sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
        sc1.metric("Peak Stall Torque", f"{hip_peak:.2f} Nm")
        sc2.metric("Continuous Torque", f"{hip_cont:.2f} Nm")
        sc3.metric("No-load Rated Speed", f"{hip_rpm:.0f} RPM")
        sc4.metric("Operating Voltage", f"{hip_volts:.1f} V")
        sc5.metric("Stall Current", f"{hip_curr:.1f} A")
        sc6.metric("Estimated Cost", f"₹{total_motor_cost:,}")
        st.caption(f"ℹ️ **Total Actuator Mass**: {preset_motor_mass_total*1000:.0f} g for {dof_total} motors.")
    else:
        scc1, scc2, scc3 = st.columns([2.5, 2.5, 1.2])
        with scc1:
            st.info(f"**Hip Motor ({hip_preset_name.split(' — ')[0]})**: Peak: {hip_peak:.2f} Nm | Cont: {hip_cont:.2f} Nm | Speed: {hip_rpm:.0f} RPM | Voltage: {hip_volts:.1f}V | Stall: {hip_curr:.1f}A | Weight: {hip_preset['mass_kg']*1000:.0f}g")
        with scc2:
            st.info(f"**Knee Motor ({knee_preset_name.split(' — ')[0]})**: Peak: {knee_peak:.2f} Nm | Cont: {knee_cont:.2f} Nm | Speed: {knee_rpm:.0f} RPM | Voltage: {knee_volts:.1f}V | Stall: {knee_curr:.1f}A | Weight: {knee_preset['mass_kg']*1000:.0f}g")
        with scc3:
            st.metric("Total Motor Cost", f"₹{total_motor_cost:,}")

    st.divider()
    
    # Joint Torque Calculator Parameters
    st.markdown("#### 🧮 Torque Calculator Inputs")
    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        stance_legs = st.selectbox(
            "Stance legs during gait", [1, 2, 3, 4], index=1,
            help="❓ HOW MANY FEET SHARE BODY WEIGHT SIMULTANEOUSLY:\n• 4 legs (Walk) = Lowest torque per joint\n• 2 legs (Trot) = Dynamic vertical ground force doubles\n• 1 leg = Stance leg carries full body weight."
        )
        dynamic_accel = st.number_input(
            "Dynamic linear acceleration (m/s²)", 0.0, 30.0, 1.0, 0.1,
            help="❓ Extra linear pushing force when starting or stopping walking. Directly scales horizontal GRF."
        )
    with tc2:
        impact_factor = st.number_input(
            "Touchdown impact factor", 1.0, 10.0, 1.5, 0.1,
            help="❓ Multiplier for shock force on foot strike. High values represent jumps/rough landings."
        )
        thigh_mass = st.number_input("Thigh link mass (kg, single leg)", 0.001, 5.0, max(st.session_state["mass_links"]/8, 0.001), 0.001)
    with tc3:
        shank_mass = st.number_input("Shank link mass (kg, single leg)", 0.001, 5.0, max(st.session_state["mass_links"]/8, 0.001), 0.001)
        thigh_com = st.slider("Thigh COM (% from hip)", 0, 100, 50) / 100
    with tc4:
        shank_com = st.slider("Shank COM (% from knee)", 0, 100, 50) / 100
        jerk_mode = st.checkbox("Jerk Mode (1.1x safety)", value=False,
                                help="Tolerates temporary overload. Reduces safety factor to 1.1x to evaluate if the motor can practically work with some jerks.")
        
    # Recompute total mass for calculations
    motor_mass_total = st.session_state.get("mass_motor", preset_motor_mass_total)
    battery_mass = st.session_state.get("mass_battery", 0.20)
    frame_mass = st.session_state.get("mass_frame", 0.25)
    links_mass = st.session_state.get("mass_links", 0.15)
    payload_mass = st.session_state.get("mass_payload", 0.0)
    
    if jerk_mode:
        safety_factor = 1.1
        st.warning("⚠️ Jerk Mode Active: Overriding safety multiplier to 1.1x. Expect structural jitter or joint sag in reality under high loads, but motors will walk.")
    
    esp32_mass, pcb_mass, sensor_mass, wiring_mass, fastener_mass = .02, .03, .03, .04, .03
    robot_mass = motor_mass_total + battery_mass + frame_mass + links_mass + esp32_mass + pcb_mass + sensor_mass + wiring_mass + fastener_mass
    total_system_mass = robot_mass + payload_mass

    # Calculate Max Safe Height recommendation for selected motor combination
    rec_max_h_mm = recommend_max_standing_height_for_motor(
        hip_peak_nm=hip_peak, hip_cont_nm=hip_cont,
        knee_peak_nm=knee_peak, knee_cont_nm=knee_cont,
        dof_total=dof_total, femur_fraction=femur_fraction, neutral_knee_deg=neutral_knee_deg,
        total_mass_kg=robot_mass, payload_kg=payload_mass, thigh_mass_kg=thigh_mass,
        shank_mass_kg=shank_mass, stance_legs=stance_legs, dynamic_accel=dynamic_accel,
        impact_factor=impact_factor, efficiency=efficiency, safety_factor=safety_factor
    )
    
    # Run budget check
    budget = joint_torque_budget(
        hip_flexion=q_hip, knee_flexion=q_knee, hip_abduction=q_ab,
        hip_offset=derived["l1"], thigh_length=thigh, shank_length=shank,
        total_mass_kg=robot_mass, payload_kg=payload_mass,
        thigh_mass_kg=thigh_mass, shank_mass_kg=shank_mass,
        thigh_com_frac=thigh_com, shank_com_frac=shank_com,
        legs_in_stance=stance_legs, dynamic_accel_mps2=dynamic_accel,
        impact_factor=impact_factor, transmission_efficiency=efficiency,
        safety_factor=safety_factor
    )
    
    st.markdown("#### 📊 Calculated Torque Budget Table")
    torque_table = pd.DataFrame({name: {"Hip joint (Nm)": vals["hip_nm"], "Knee joint (Nm)": vals["knee_nm"], "Ab/Ad joint (Nm)": vals["abad_nm"]}
                                 for name, vals in budget.items()
                                 if isinstance(vals, dict) and "hip_nm" in vals}).round(2)
    st.dataframe(torque_table, use_container_width=True)
    
    # Motor verdicts
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
    
    st.markdown("##### 🔍 Sizing Verdicts:")
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        if eval_hip['verdict'] == "FAIL":
            st.error(f"❌ **Hip Pitch Joint: FAIL** — " + " ".join(eval_hip["reasons"]))
        elif eval_hip['verdict'] == "MARGINAL":
            st.warning(f"⚠️ **Hip Pitch Joint: MARGINAL** — " + " ".join(eval_hip["reasons"]))
        else:
            st.success(f"✅ **Hip Pitch Joint: PASS** — " + " ".join(eval_hip["reasons"]))
    with vcol2:
        if eval_knee['verdict'] == "FAIL":
            st.error(f"❌ **Knee Pitch Joint: FAIL** — " + " ".join(eval_knee["reasons"]))
        elif eval_knee['verdict'] == "MARGINAL":
            st.warning(f"⚠️ **Knee Pitch Joint: MARGINAL** — " + " ".join(eval_knee["reasons"]))
        else:
            st.success(f"✅ **Knee Pitch Joint: PASS** — " + " ".join(eval_knee["reasons"]))
            
    # Clickable Auto-Fix combination finder
    if eval_hip['verdict'] in ["FAIL", "MARGINAL"] or eval_knee['verdict'] in ["FAIL", "MARGINAL"]:
        st.markdown("---")
        st.warning("⚠️ **DESIGN FLAW DETECTED**: Selected motors are insufficient or marginal. Use the Auto-Fix engine to find alternative configurations that pass.")
        if st.button("🔍 Run Auto-Fix: Suggest Passing Motor Combinations"):
            num_legs = dof_total // 2 if not has_abad else dof_total // 3
            if num_legs not in [2, 4, 6]: num_legs = 4
            
            valid_combos = []
            for hip_name, h_motor in SERVO_PRESETS.items():
                for knee_name, k_motor in SERVO_PRESETS.items():
                    req_hip = budget["peak_required"]["hip_nm"]
                    req_knee = budget["peak_required"]["knee_nm"]
                    req_hip_cont = budget["continuous_required"]["hip_nm"]
                    req_knee_cont = budget["continuous_required"]["knee_nm"]
                    
                    h_pass = (h_motor["peak_nm"] >= req_hip and h_motor["continuous_nm"] >= req_hip_cont)
                    k_pass = (k_motor["peak_nm"] >= req_knee and k_motor["continuous_nm"] >= req_knee_cont)
                    
                    if h_pass and k_pass:
                        h_margin = (h_motor["peak_nm"] - req_hip) / h_motor["peak_nm"] * 100
                        k_margin = (k_motor["peak_nm"] - req_knee) / k_motor["peak_nm"] * 100
                        avg_margin = (h_margin + k_margin) / 2
                        total_cost = (h_motor["approx_price_inr"] + k_motor["approx_price_inr"]) * num_legs
                        total_weight = (h_motor["mass_kg"] + k_motor["mass_kg"]) * num_legs
                        
                        valid_combos.append({
                            "hip": hip_name,
                            "knee": knee_name,
                            "hip_margin": h_margin,
                            "knee_margin": k_margin,
                            "avg_margin": avg_margin,
                            "cost": total_cost,
                            "weight": total_weight
                        })
            
            if valid_combos:
                best_combo = sorted(valid_combos, key=lambda x: x["avg_margin"], reverse=True)[0]
                better_pool = [c for c in valid_combos if c["hip_margin"] >= 25 and c["knee_margin"] >= 25]
                better_combo = sorted(better_pool if better_pool else valid_combos, key=lambda x: x["cost"])[0]
                budget_combo = sorted(valid_combos, key=lambda x: x["cost"])[0]
                
                st.success("🎉 **Feasible Actuator Combinations Found!**")
                
                ac1, ac2, ac3 = st.columns(3)
                with ac1:
                    st.markdown("🏆 **1. BEST PERFORMANCE**")
                    st.write(f"• **Hip**: {best_combo['hip'].split(' — ')[0]}")
                    st.write(f"• **Knee**: {best_combo['knee'].split(' — ')[0]}")
                    st.write(f"• **Margins**: Hip: {best_combo['hip_margin']:.0f}% | Knee: {best_combo['knee_margin']:.0f}%")
                    st.write(f"• **Total Cost**: ₹{best_combo['cost']:,}")
                with ac2:
                    st.markdown("⚖️ **2. BETTER / BALANCED**")
                    st.write(f"• **Hip**: {better_combo['hip'].split(' — ')[0]}")
                    st.write(f"• **Knee**: {better_combo['knee'].split(' — ')[0]}")
                    st.write(f"• **Margins**: Hip: {better_combo['hip_margin']:.0f}% | Knee: {better_combo['knee_margin']:.0f}%")
                    st.write(f"• **Total Cost**: ₹{better_combo['cost']:,}")
                with ac3:
                    st.markdown("💰 **3. BUDGET / MAINTAINABLE**")
                    st.write(f"• **Hip**: {budget_combo['hip'].split(' — ')[0]}")
                    st.write(f"• **Knee**: {budget_combo['knee'].split(' — ')[0]}")
                    st.write(f"• **Margins**: Hip: {budget_combo['hip_margin']:.0f}% | Knee: {budget_combo['knee_margin']:.0f}%")
                    st.write(f"• **Total Cost**: ₹{budget_combo['cost']:,}")
            else:
                st.error("⛔ **NO FEASIBLE CONFIGURATIONS**: Even the most powerful motors in the catalog cannot support this robot. Consider reducing payload or chassis mass.")
            
    # Motor height recommendation card
    st.markdown("##### 🎯 Standing Height Limits:")
    if rec_max_h_mm is not None:
        if standing_height * MM > rec_max_h_mm:
            st.error(f"⚠️ **Overextended height**: Your target standing height ({standing_height*MM:.0f} mm) exceeds the maximum safe height of **{rec_max_h_mm} mm** for these motors.")
        else:
            st.success(f"✅ **Standing Height Valid**: Target height ({standing_height*MM:.0f} mm) is within the safe limit of **{rec_max_h_mm} mm**.")
    else:
        st.error("⛔ Selected motors are too weak to support standing at any height!")

    with st.expander("🛠️ Show Dynamic Test Cases (Walking, Trotting, Crouch, Land)"):
        st.caption("Standard operational test configurations evaluated on the fly.")
        TESTS = {"Standing": (4, 0.0, 1.0), "Crouching": (4, .5, 1.1), "Walking": (3, 1.0, 1.3), "Trotting": (2, 2.0, 1.7), "Acceleration": (2, 4.0, 1.5), "Stopping": (2, 4.0, 1.5), "Slope standing": (3, 1.5, 1.2), "One-leg disturbance": (3, 3.0, 1.8), "Landing / impact": (2, 3.0, 2.5)}
        rows = []
        for name, (legs, accel, impact) in TESTS.items():
            test = joint_torque_budget(hip_flexion=q_hip, knee_flexion=q_knee, hip_abduction=q_ab, hip_offset=derived["l1"], thigh_length=thigh, shank_length=shank, total_mass_kg=robot_mass, payload_kg=payload_mass, thigh_mass_kg=thigh_mass, shank_mass_kg=shank_mass, thigh_com_frac=thigh_com, shank_com_frac=shank_com, legs_in_stance=legs, dynamic_accel_mps2=accel, impact_factor=impact, transmission_efficiency=efficiency, safety_factor=safety_factor)
            rows.append({"Test Case": name, "Stance Legs": legs, "Hip Peak Required (Nm)": round(test["peak_required"]["hip_nm"], 2), "Knee Peak Required (Nm)": round(test["peak_required"]["knee_nm"], 2)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("🧗 2D Stair Climbing Feasibility Checker"):
        stair_col1, stair_col2 = st.columns(2)
        with stair_col1:
            riser = mm_input("Stair riser height (mm)", 175.0, 10.0, 500.0, 5.0, key="riser_stair_val")
            tread = mm_input("Stair tread depth (mm)", 280.0, 50.0, 600.0, 5.0, key="tread_stair_val")
            landing_x = mm_input("Paw landing fore/aft offset beyond riser (mm)", 180.0, 10.0, 800.0, 5.0)
            stair_direction = st.radio("Ascent / Descent Direction", ["Climb", "Descend"], horizontal=True)
        with stair_col2:
            hip_min = st.number_input("Hip joint lower limit (°)", -180.0, 0.0, -90.0, 5.0)
            hip_max = st.number_input("Hip joint upper limit (°)", 0.0, 180.0, 90.0, 5.0)
            knee_min = st.number_input("Knee joint lower limit (°)", -180.0, 0.0, -90.0, 5.0)
            knee_max = st.number_input("Knee joint upper limit (°)", 0.0, 180.0, 90.0, 5.0)

        stair_target = (landing_x, derived["l1"], -standing_height + (riser if stair_direction == "Climb" else -riser))
        stair_sol, stair_issues = assess_stair_target(stair_target, thigh, shank, derived["l1"], knee_forward, hip_min, hip_max, knee_min, knee_max)
        
        sc1, sc2 = st.columns([1.2, 1])
        with sc1:
            fig_stair = stair_pose(thigh, shank, stair_sol[1] if stair_sol else 0.0, stair_sol[2] if stair_sol else 0.0, stair_target, standing_height, riser, tread, stair_direction, stair_sol is not None)
            st.pyplot(fig_stair, use_container_width=True)
        with sc2:
            if stair_sol is None:
                st.error("❌ **STAIR COLLISION / OUT OF REACH**: Joint limits or geometry bounds violated.")
                for iss in stair_issues:
                    st.write(f"- {iss}")
            else:
                st.success("✅ **STAIR REACHABLE**")
                stair_torque = stair_climb_torque_budget(
                    hip_flexion_flat=q_hip, knee_flexion_flat=q_knee,
                    hip_flexion_stair=stair_sol[1], knee_flexion_stair=stair_sol[2],
                    hip_abduction=q_ab, hip_offset=derived["l1"],
                    thigh_length=thigh, shank_length=shank,
                    total_mass_kg=robot_mass, payload_kg=payload_mass,
                    thigh_mass_kg=thigh_mass, shank_mass_kg=shank_mass,
                    thigh_com_frac=thigh_com, shank_com_frac=shank_com,
                    riser_height_m=riser, transmission_efficiency=efficiency,
                    safety_factor=safety_factor
                )
                st.metric("Stair Leg Hip Peak (Nm)", f"{stair_torque['stair_leg_hip_nm']:.2f} Nm")
                st.metric("Stair Leg Knee Peak (Nm)", f"{stair_torque['stair_leg_knee_nm']:.2f} Nm")

# =========================================================================================
# TAB 3: GAITS & TORQUE-SPEED OPERATING POINTS
# =========================================================================================
with tab_gaits:
    st.subheader("🏃 Dynamic Gait Sizing & Torque-Speed Operating Points")
    
    # Target Speed & Gait Settings
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        gait_decided = st.radio(
            "Have you decided on which walking/trotting gait to use?",
            ["Not yet — show me the undecided-gait safety margin",
             "Yes — I know which gait I want to use"],
            index=0,
            help="If undecided, the system guarantees safety by testing against the absolute worst-case torque gait."
        )
    with gcol2:
        target_speed = st.slider("Target walking speed (m/s)", 0.05, 2.0, 0.3, 0.05,
                                 help="Required operational speed. Faster speeds increase dynamic inertia and joint velocities.")
        
    st.divider()
    
    # Gait recommendations evaluation
    gait_recs = recommend_gait(
        hip_peak_nm=hip_peak, knee_peak_nm=knee_peak,
        hip_cont_nm=hip_cont, knee_cont_nm=knee_cont,
        robot_mass_kg=robot_mass, payload_kg=payload_mass,
        standing_height_m=standing_height,
        thigh_length=thigh, shank_length=shank,
        hip_offset=derived["l1"],
        target_speed_mps=target_speed, terrain=terrain,
        efficiency=efficiency, safety_factor=safety_factor,
        thigh_mass_kg=thigh_mass, shank_mass_kg=shank_mass
    )

    if gait_decided.startswith("Not yet"):
        wcm = worst_case_gait_margin(
            hip_peak_nm=hip_peak, knee_peak_nm=knee_peak,
            hip_cont_nm=hip_cont, knee_cont_nm=knee_cont,
            robot_mass_kg=robot_mass, payload_kg=payload_mass,
            standing_height_m=standing_height,
            thigh_length=thigh, shank_length=shank,
            hip_offset=derived["l1"],
            target_speed_mps=target_speed,
            efficiency=efficiency, safety_factor=safety_factor,
            thigh_mass_kg=thigh_mass, shank_mass_kg=shank_mass
        )
        if wcm['undecided_safe']:
            st.success(f"✅ **SAFE FOR ALL GAITS**: Your motors can handle the worst-case gait (**{wcm['worst_gait_name']}**). Hip margin: {wcm['hip_margin_pct']:.0f}% | Knee margin: {wcm['knee_margin_pct']:.0f}%.")
        else:
            st.error(f"❌ **NOT SAFE FOR ALL GAITS**: The worst-case gait (**{wcm['worst_gait_name']}**) exceeds your motor capacities. Hip margin: {wcm['hip_margin_pct']:.0f}% | Knee margin: {wcm['knee_margin_pct']:.0f}%.")

    # Display gait table
    st.markdown("##### 🏃 Gait Feasibility Comparison:")
    gait_rows = []
    for rec in gait_recs:
        max_h_val = f"{rec['max_height_mm']} mm" if rec['max_height_mm'] > 0 else "Too Weak"
        gait_rows.append({
            "Gait Mode": rec['gait'],
            "Status": "✅ PASS" if rec['feasible'] else "❌ FAIL",
            "Duty Factor": f"{rec['duty_factor']:.2f}",
            "Min Stance Legs": rec['min_stance_legs'],
            "Hip Margin": f"{rec['hip_margin_pct']:.0f}%",
            "Knee Margin": f"{rec['knee_margin_pct']:.0f}%",
            "Max Safe Height": max_h_val,
            "Speed Fit": rec.get('speed_suitability', 'N/A'),
            "Terrain Fit": rec.get('terrain_suitability', 'N/A'),
        })
    st.dataframe(pd.DataFrame(gait_rows), use_container_width=True, hide_index=True)
    
    if gait_decided.startswith("Yes"):
        selected_gait = st.selectbox("Select your active design gait", list(GAITS.keys()), index=2)
    else:
        selected_gait = "Trot"
        
    st.divider()
    
    # Torque-Speed Operating Points
    st.markdown("#### 📈 Motor Torque-Speed Operating Point Verification")
    st.caption("Real DC motors lose torque output linearly as angular velocity increases. Below we verify if your operating points fall within safe curves.")
    
    # Estimate joint speeds from gait
    gait_period = 0.6
    step_length_m = target_speed * gait_period * GAITS[selected_gait]['duty_factor']
    step_height_m = 0.03
    joint_speed_est = estimate_max_joint_speed(
        step_length_m=step_length_m, step_height_m=step_height_m,
        period_s=gait_period, duty_factor=GAITS[selected_gait]['duty_factor'],
        thigh_length=thigh, shank_length=shank,
        standing_height_m=standing_height, hip_offset=derived["l1"]
    )
    
    hip_op = motor_operating_point(
        motor_peak_nm=hip_peak, motor_cont_nm=hip_cont, motor_rpm=hip_rpm,
        required_torque_nm=budget['continuous_required']['hip_nm'],
        required_speed_rad_s=joint_speed_est.get('peak_hip_rad_s', 2.0)
    )
    knee_op = motor_operating_point(
        motor_peak_nm=knee_peak, motor_cont_nm=knee_cont, motor_rpm=knee_rpm,
        required_torque_nm=budget['continuous_required']['knee_nm'],
        required_speed_rad_s=joint_speed_est.get('peak_knee_rad_s', 2.0)
    )
    
    col_op1, col_op2 = st.columns(2)
    with col_op1:
        st.markdown(f"**Hip Joint ({hip_preset_name.split(' — ')[0]})**")
        st.write(f"• Torque Available at Speed: **{hip_op['torque_available_at_speed']:.2f} Nm**")
        st.write(f"• Speed Utilization: **{hip_op['speed_pct']:.0f}%**")
        st.write(f"• Torque Utilization: **{hip_op['torque_pct']:.0f}%**")
        if hip_op.get('thermal_warning'):
            st.warning(f"🔥 {hip_op['thermal_warning']}")
        elif hip_op['in_continuous_region']:
            st.success("✅ Operating within continuous thermal region.")
        else:
            st.warning("⚠️ Operating in peak region (Intermittent duty only).")
            
    with col_op2:
        st.markdown(f"**Knee Joint ({knee_preset_name.split(' — ')[0]})**")
        st.write(f"• Torque Available at Speed: **{knee_op['torque_available_at_speed']:.2f} Nm**")
        st.write(f"• Speed Utilization: **{knee_op['speed_pct']:.0f}%**")
        st.write(f"• Torque Utilization: **{knee_op['torque_pct']:.0f}%**")
        if knee_op.get('thermal_warning'):
            st.warning(f"🔥 {knee_op['thermal_warning']}")
        elif knee_op['in_continuous_region']:
            st.success("✅ Operating within continuous thermal region.")
        else:
            st.warning("⚠️ Operating in peak region (Intermittent duty only).")
            
    # Curves plot
    fig_ts, (ax_h, ax_k) = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, label, peak, cont, rpm, req_t, req_s in [
        (ax_h, f"Hip ({hip_preset_name.split(' — ')[0]})", hip_peak, hip_cont, hip_rpm,
         budget['continuous_required']['hip_nm'], joint_speed_est.get('peak_hip_rad_s', 2.0)),
        (ax_k, f"Knee ({knee_preset_name.split(' — ')[0]})", knee_peak, knee_cont, knee_rpm,
         budget['continuous_required']['knee_nm'], joint_speed_est.get('peak_knee_rad_s', 2.0))
    ]:
        omega_nl = rpm * 2 * np.pi / 60
        speeds = np.linspace(0, omega_nl, 100)
        torques = peak * (1 - speeds / omega_nl)
        ax.fill_between(speeds, 0, np.minimum(torques, cont), alpha=0.2, color='green', label='Continuous Region')
        ax.fill_between(speeds, cont, torques, where=torques > cont, alpha=0.15, color='orange', label='Peak Region')
        ax.plot(speeds, torques, 'b-', lw=2, label='T-ω Curve')
        ax.axhline(y=cont, color='green', ls='--', alpha=0.6, label='Cont. Limit')
        ax.plot(req_s, req_t, 'r*', ms=12, label='Operating Point')
        ax.set_xlabel('Joint speed (rad/s)')
        ax.set_ylabel('Torque (Nm)')
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig_ts.tight_layout()
    st.pyplot(fig_ts, use_container_width=True)

# =========================================================================================
# TAB 4: MASS, POWER & OPTIMIZERS
# =========================================================================================
with tab_power_opt:
    st.subheader("🔋 Mass Sizing, Battery Sizing & Leg Optimizations")
    
    # Layout splits
    mcol1, mcol2 = st.columns([1, 1.2])
    with mcol1:
        st.markdown("#### ⚖️ Mass Component Breakdown")
        
        # Interactive number inputs bound to session state keys
        _frame_mass_ui = st.number_input(
            "Frame/chassis mass (kg)", 0.01, 20.0,
            float(st.session_state["mass_frame"]), 0.01,
            key="mass_frame_input_tab"
        )
        st.session_state["mass_frame"] = _frame_mass_ui
        
        _links_mass_ui = st.number_input(
            "All links mass (kg, total 4 legs)", 0.01, 20.0,
            float(st.session_state["mass_links"]), 0.01,
            key="mass_links_input_tab"
        )
        st.session_state["mass_links"] = _links_mass_ui
        
        _battery_mass_ui = st.number_input(
            "Battery mass (kg)", 0.01, 10.0,
            float(st.session_state["mass_battery"]), 0.01,
            key="mass_battery_input_tab"
        )
        st.session_state["mass_battery"] = _battery_mass_ui
        
        # Synchronized payload input
        _payload_mass_ui = st.number_input(
            "Payload mass (kg)", 0.0, 50.0,
            float(st.session_state["mass_payload"]), 0.01,
            key="mass_payload_input_tab"
        )
        st.session_state["mass_payload"] = _payload_mass_ui
        
        _payload_x_ui = st.number_input(
            "Payload fore/aft offset (mm)", -500.0, 500.0,
            float(st.session_state["mass_payload_x"]), 1.0,
            key="mass_payload_x_input_tab"
        )
        st.session_state["mass_payload_x"] = _payload_x_ui

        st.markdown("**Fixed Allowances**")
        st.caption(f"ESP32 Controller: {esp32_mass*1000:.0f} g | PCB PDB: {pcb_mass*1000:.0f} g")
        st.caption(f"Sensors: {sensor_mass*1000:.0f} g | Wiring: {wiring_mass*1000:.0f} g | Fasteners: {fastener_mass*1000:.0f} g")

    with mcol2:
        st.markdown("#### 📊 Mass Distribution Chart")
        
        # Calculate totals
        _components_dict = {
            "Motors": motor_mass_total,
            "Battery": _battery_mass_ui,
            "Frame": _frame_mass_ui,
            "Links (4 legs)": _links_mass_ui,
            "Payload": _payload_mass_ui,
            "ESP32": esp32_mass,
            "PCB": pcb_mass,
            "Sensors": sensor_mass,
            "Wiring": wiring_mass,
            "Fasteners": fastener_mass,
        }
        _total_mass_calc = sum(_components_dict.values())
        
        fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
        labels = [f"{k} ({v*1000:.0f}g)" for k, v in _components_dict.items() if v > 0.001]
        sizes = [v for v in _components_dict.values() if v > 0.001]
        colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
        wedges, texts, autotexts = ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=140)
        ax_pie.set_title("Chassis Mass Distribution", fontsize=10)
        st.pyplot(fig_pie, use_container_width=True)
        
        st.metric("Total System Mass (with Payload)", f"{_total_mass_calc:.3f} kg ({_total_mass_calc*1000:.0f} g)")
        st.metric("Robot Dry Mass", f"{(_total_mass_calc - _payload_mass_ui):.3f} kg")

    st.divider()

    # Smart battery calculator
    st.markdown("#### 🔋 Smart Servo Power & Battery Recommender")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        target_runtime_min = st.slider("Desired walking runtime (minutes)", 5, 120, 20, 5, key="runtime_slider_tab")
        battery_type = st.selectbox(
            "Battery chemistry preference selection",
            ["LiPo (Lithium Polymer) — High performance, light", "Li-ion 18650 — Good runtime, standard", "NiMH / External Power Supply"],
            index=0, key="chem_select_tab"
        )
    with pcol2:
        target_volts = max(hip_volts, knee_volts)
        # Calculate Dual Motor Battery Requirements
        if dof_total == 8:
            num_hip, num_knee = 4, 4
        else:
            num_hip, num_knee = 8, 4
        total_avg_current = (num_hip * 0.5 * hip_avg_curr) + (num_knee * 0.5 * knee_avg_curr) + 0.35
        total_peak_current = (num_hip * hip_curr) + (num_knee * knee_curr) + 0.35
        req_capacity_mah = int(np.ceil((total_avg_current * (target_runtime_min / 60.0) / 0.80) * 1000 / 100) * 100)
        req_c_rating = int(np.ceil(total_peak_current / (req_capacity_mah / 1000.0)))
        lipo_cells = "2S (7.4V)" if target_volts <= 7.4 else ("3S (11.1V)" if target_volts <= 11.1 else "4S (14.8V)")
        
        st.write(f"• Peak Current Demand: **{total_peak_current:.1f} A**")
        st.write(f"• Avg Walking Current: **{total_avg_current:.1f} A**")

    # Battery recommendations
    bp1, bp2, bp3 = st.columns(3)
    bp1.metric("Recommended Battery Voltage", f"{target_volts:.1f} V ({lipo_cells})")
    bp2.metric("Min Capacity Required", f"{req_capacity_mah} mAh")
    bp3.metric("Min C-Rating Required", f"{req_c_rating}C or higher")
    st.info(f"💡 **Power Recommendation**: Buy a **{lipo_cells} {req_capacity_mah} mAh LiPo Battery with at least {req_c_rating}C discharge rating**.")

    st.divider()

    # Leg proportion optimizer
    st.markdown("#### ⚙️ Motor-Aware Leg Geometry Optimizer")
    st.caption("Searches for link length distributions that maximize torque margin on both joints simultaneously.")
    
    hip_util = budget['peak_required']['hip_nm'] / hip_peak * 100 if hip_peak > 0 else 999
    knee_util = budget['peak_required']['knee_nm'] / knee_peak * 100 if knee_peak > 0 else 999
    
    uc1, uc2 = st.columns(2)
    uc1.metric("Hip Servo Utilization", f"{hip_util:.0f}%", delta=f"{'OK' if hip_util < 100 else 'OVERLOADED'}")
    uc2.metric("Knee Servo Utilization", f"{knee_util:.0f}%", delta=f"{'OK' if knee_util < 100 else 'OVERLOADED'}")

    if optimize_leg_proportions_for_motors is not None:
        with st.expander("🔧 Run Dual-Objective Leg Proportion Optimizer"):
            if st.button("🚀 Optimize leg proportions for selected motors"):
                opt_result = optimize_leg_proportions_for_motors(
                    standing_height, _total_mass_calc, _payload_mass_ui,
                    thigh_mass, shank_mass, stance_legs,
                    dynamic_accel, impact_factor, efficiency, safety_factor,
                    has_abad, derived["l1"],
                    hip_peak, hip_cont, knee_peak, knee_cont
                )
                st.success(f"✅ **Optimized split**: Thigh = **{opt_result['thigh_length_m']*MM:.1f} mm** ({opt_result['femur_fraction']:.0%}), Shank = **{opt_result['shank_length_m']*MM:.1f} mm** ({1-opt_result['femur_fraction']:.0%})")
                
                opt_c1, opt_c2, opt_c3 = st.columns(3)
                opt_c1.metric("Optimized Hip Utilization", f"{opt_result['hip_utilization_pct']:.0f}%")
                opt_c2.metric("Optimized Knee Utilization", f"{opt_result['knee_utilization_pct']:.0f}%")
                opt_c3.metric("Torque-Margin Improvement", f"{opt_result.get('improvement_pct', 0):.1f}%")

    if suggest_cheaper_motor_combination is not None:
        with st.expander("💰 Cost-Saving Motor Downsizing Suggestions"):
            if st.button("🔍 Find cheaper motor combinations"):
                suggestions = suggest_cheaper_motor_combination(
                    standing_height, _total_mass_calc, _payload_mass_ui,
                    thigh_mass, shank_mass, stance_legs,
                    dynamic_accel, impact_factor, efficiency, safety_factor,
                    has_abad, derived["l1"],
                    hip_preset_name, knee_preset_name, dof_total
                )
                if suggestions:
                    st.success(f"Found **{len(suggestions)}** cheaper motor combination(s) that meet torque requirements!")
                    cost_rows = []
                    for s in suggestions[:5]:
                        cost_rows.append({
                            "Hip Motor": s['hip_motor'],
                            "Knee Motor": s['knee_motor'],
                            "Femur Fraction": f"{s['femur_fraction']:.0%}",
                            "Total Motor Cost": f"₹{s['total_motor_cost']:,}",
                            "Savings": f"₹{s['cost_savings']:,}",
                            "Hip Margin": f"{s['hip_margin_pct']:.0f}%",
                            "Knee Margin": f"{s['knee_margin_pct']:.0f}%"
                        })
                    st.dataframe(pd.DataFrame(cost_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No cheaper motor combination found that meets torque requirements. Your current selection is already cost-optimal.")

# =========================================================================================
# PDF BUILDER DATA DICT AND DOWNLOAD BUTTON
# =========================================================================================
pdf_data = {
    'dof_total': dof_total,
    'standing_height_mm': standing_height * MM,
    'total_mass_kg': _total_mass_calc,
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
    'manipulability_index': manipulability_index(q_ab, q_hip, q_knee, derived["l1"], thigh, shank) if sol is not None else 0.0,
    'mass_matrix': compute_mass_matrix(q_hip, q_knee, thigh_mass, shank_mass, thigh_com, shank_com, thigh, shank).tolist(),
    'torque_budget': budget,
    'gait_recommendations': gait_recs,
    'motor_operating_points': {'hip': hip_op, 'knee': knee_op},
    'battery_calc': {
        'voltage_cell': lipo_cells,
        'req_capacity_mah': req_capacity_mah,
        'req_c_rating': req_c_rating,
        'total_peak_current': total_peak_current,
        'target_runtime_min': target_runtime_min
    },
    'mass_components': _components_dict
}

try:
    pdf_bytes = generate_quadruped_pdf_report(pdf_data)
    pdf_button_container.download_button(
        "📄 Export Engineering PDF Report",
        data=pdf_bytes,
        file_name=f"Quadruped_Robot_Engineering_Dossier_{dof_total}DOF.pdf",
        mime="application/pdf",
        help="Click to generate and download a complete A4 PDF engineering calculation dossier to present your project!"
    )
except Exception as pdf_err:
    pdf_button_container.warning(f"📄 PDF Exporter: {pdf_err}")
