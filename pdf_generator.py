"""
PDF Engineering Report Generator for Quadruped Robot Design.
Generates a comprehensive multi-page engineering calculation PDF report
including Kinematics (FK/IK), Jacobian Matrix, Dynamics, Dual-Motor Selection,
Power/Battery sizing, Stair Feasibility, Mass Breakdown, and Control Theory.
"""
import io
import datetime
import numpy as np

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY


def generate_quadruped_pdf_report(data: dict) -> bytes:
    """Generate a complete PDF report from quadruped calculation parameters.
    Returns bytes buffer suitable for Streamlit download_button.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    PRIMARY = colors.HexColor("#1D3557")
    SECONDARY = colors.HexColor("#457B9D")
    ACCENT = colors.HexColor("#E63946")
    DARK_BG = colors.HexColor("#F8F9FA")
    TEXT_DARK = colors.HexColor("#212529")
    SUCCESS_COLOR = colors.HexColor("#2A9D8F")
    WARNING_COLOR = colors.HexColor("#E76F51")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=PRIMARY,
        alignment=TA_LEFT,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=14,
        textColor=SECONDARY,
        alignment=TA_LEFT,
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=6
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=SECONDARY,
        spaceBefore=10,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=TEXT_DARK,
        spaceAfter=6
    )

    equation_style = ParagraphStyle(
        'EquationText',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#1F2937"),
        backColor=colors.HexColor("#F3F4F6"),
        borderColor=colors.HexColor("#E5E7EB"),
        borderWidth=1,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=6
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=TA_CENTER
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=TEXT_DARK,
        alignment=TA_LEFT
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=TEXT_DARK,
        alignment=TA_LEFT
    )

    elements = []

    # -------------------------------------------------------------------------
    # HEADER / TITLE BLOCK
    # -------------------------------------------------------------------------
    elements.append(Paragraph("🐕 QUADRUPED ROBOT ENGINEERING REPORT", title_style))
    elements.append(Paragraph(
        f"Full Mathematical Calculation, Kinematics, Dynamics & Component Sizing Dossier<br/>"
        f"Generated on: {datetime.datetime.now().strftime('%B %d, %Y - %H:%M:%S')}",
        subtitle_style
    ))
    elements.append(HRFlowable(width="100%", thickness=2, color=PRIMARY, spaceAfter=12))

    # -------------------------------------------------------------------------
    # EXECUTIVE SUMMARY CARDS TABLE
    # -------------------------------------------------------------------------
    summary_data = [
        [
            Paragraph(f"<b>DOF Total:</b> {data.get('dof_total', 8)}-DOF", table_cell_style),
            Paragraph(f"<b>Target Height:</b> {data.get('standing_height_mm', 350):.0f} mm", table_cell_style),
            Paragraph(f"<b>Total System Mass:</b> {data.get('total_mass_kg', 1.5):.2f} kg", table_cell_bold),
        ],
        [
            Paragraph(f"<b>Terrain:</b> {data.get('terrain', 'Flat')}", table_cell_style),
            Paragraph(f"<b>Motor Mode:</b> {data.get('motor_mode_name', 'Single')}", table_cell_style),
            Paragraph(f"<b>Max Safe Height:</b> {data.get('max_safe_height_mm', 'N/A')} mm", table_cell_bold),
        ]
    ]
    summary_table = Table(summary_data, colWidths=[175, 175, 175])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EBF4F6")),
        ('BORDER', (0, 0), (-1, -1), 1, colors.HexColor("#B0C4DE")),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # SECTION 1: KINEMATICS MATHEMATICAL MODEL (FK & IK)
    # -------------------------------------------------------------------------
    elements.append(Paragraph("1. Kinematics Mathematical Model (FK & IK)", h1_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=SECONDARY, spaceAfter=6))

    elements.append(Paragraph(
        "Kinematics models the geometry of motion without considering forces. The quadruped leg is modeled "
        "as a planar 2-link/3-link manipulator in the sagittal (side) plane with link lengths <i>l</i><sub>1</sub> (hip offset), "
        "<i>l</i><sub>2</sub> (thigh), and <i>l</i><sub>3</sub> (shank).",
        body_style
    ))

    elements.append(Paragraph("Forward Kinematics (FK) Equations:", h2_style))
    elements.append(Paragraph(
        "Given joint angles (q_hip, q_knee), FK computes foot position (x, y, z) in the hip frame:<br/>"
        "&nbsp;&nbsp;x = l2 * sin(q_hip) + l3 * sin(q_hip + q_knee)<br/>"
        "&nbsp;&nbsp;y = l1 (for 12-DOF with hip ab/ad angle q_ab)<br/>"
        "&nbsp;&nbsp;z = -l2 * cos(q_hip) - l3 * cos(q_hip + q_knee)",
        equation_style
    ))

    elements.append(Paragraph("Inverse Kinematics (IK) Equations (Law of Cosines):", h2_style))
    elements.append(Paragraph(
        "Given target foot landing (x, y, z), IK solves exact joint angles:<br/>"
        "&nbsp;&nbsp;R = sqrt(x^2 + z^2)<br/>"
        "&nbsp;&nbsp;cos(q_knee) = (x^2 + z^2 - l2^2 - l3^2) / (2 * l2 * l3)<br/>"
        "&nbsp;&nbsp;q_knee = atan2(+/- sqrt(1 - cos(q_knee)^2), cos(q_knee))&nbsp;&nbsp;[Knee-Forward Configuration]<br/>"
        "&nbsp;&nbsp;q_hip = atan2(x, -z) - atan2(l3 * sin(q_knee), l2 + l3 * cos(q_knee))",
        equation_style
    ))

    kin_table_data = [
        [
            Paragraph("Parameter", table_header_style),
            Paragraph("Symbol", table_header_style),
            Paragraph("Designed Value", table_header_style),
            Paragraph("Description", table_header_style)
        ],
        [Paragraph("Thigh Length", table_cell_style), Paragraph("l2", table_cell_style), Paragraph(f"{data.get('l2_mm', 0):.1f} mm", table_cell_bold), Paragraph("Upper leg link length", table_cell_style)],
        [Paragraph("Shank Length", table_cell_style), Paragraph("l3", table_cell_style), Paragraph(f"{data.get('l3_mm', 0):.1f} mm", table_cell_bold), Paragraph("Lower leg link length", table_cell_style)],
        [Paragraph("Hip Offset", table_cell_style), Paragraph("l1", table_cell_style), Paragraph(f"{data.get('l1_mm', 0):.1f} mm", table_cell_bold), Paragraph("Lateral hip abduction offset", table_cell_style)],
        [Paragraph("Current Foot X", table_cell_style), Paragraph("x", table_cell_style), Paragraph(f"{data.get('foot_x_mm', 0):.1f} mm", table_cell_style), Paragraph("Foot target fore/aft position", table_cell_style)],
        [Paragraph("Current Foot Z", table_cell_style), Paragraph("z", table_cell_style), Paragraph(f"{data.get('foot_z_mm', 0):.1f} mm", table_cell_style), Paragraph("Foot target vertical position", table_cell_style)],
        [Paragraph("Solved Hip Angle", table_cell_style), Paragraph("q_hip", table_cell_style), Paragraph(f"{data.get('q_hip_deg', 0):.1f}°", table_cell_bold), Paragraph("Hip pitch joint angle", table_cell_style)],
        [Paragraph("Solved Knee Angle", table_cell_style), Paragraph("q_knee", table_cell_style), Paragraph(f"{data.get('q_knee_deg', 0):.1f}°", table_cell_bold), Paragraph("Knee pitch joint angle", table_cell_style)],
    ]
    kin_table = Table(kin_table_data, colWidths=[120, 50, 100, 255])
    kin_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
    ]))
    elements.append(kin_table)
    elements.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # SECTION 2: DIFFERENTIAL KINEMATICS & JACOBIAN MATRIX
    # -------------------------------------------------------------------------
    elements.append(Paragraph("2. Differential Kinematics & Jacobian Matrix", h1_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=SECONDARY, spaceAfter=6))

    elements.append(Paragraph(
        "The Jacobian matrix <b>J(q)</b> relates joint angular velocities <i>q̇</i> to foot linear velocity <i>v</i>:<br/>"
        "&nbsp;&nbsp;<b>v</b> = <b>J(q)</b> * <b>q̇</b><br/>"
        "By virtual work duality, its transpose maps foot ground reaction forces <b>F<sub>ground</sub></b> to joint torques <b>τ</b>:<br/>"
        "&nbsp;&nbsp;<b>τ</b> = <b>J(q)<sup>T</sup></b> * <b>F<sub>ground</sub></b>",
        equation_style
    ))

    j_matrix = data.get('jacobian_matrix', None)
    j_cond = data.get('jacobian_condition', 1.0)
    
    j_text = "Evaluated 2D Jacobian Matrix J(q):<br/>"
    if j_matrix is not None:
        j_text += f"[ [ {j_matrix[0][0]:.4f},  {j_matrix[0][1]:.4f} ],<br/>"
        j_text += f"  [ {j_matrix[1][0]:.4f},  {j_matrix[1][1]:.4f} ] ]"
    else:
        j_text += "[ [ dX/dq_hip, dX/dq_knee ], [ dZ/dq_hip, dZ/dq_knee ] ]"

    elements.append(Paragraph(j_text, equation_style))

    cond_color = SUCCESS_COLOR if j_cond < 20 else WARNING_COLOR
    elements.append(Paragraph(
        f"<b>Jacobian Condition Number c(J):</b> {j_cond:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<i>Status: {'WELL-CONDITIONED (Far from singularity)' if j_cond < 25 else 'NEAR SINGULARITY (High Joint Velocities Required)'}</i>",
        body_style
    ))
    elements.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # SECTION 3: INVERSE DYNAMICS & JOINT TORQUE BUDGET
    # -------------------------------------------------------------------------
    elements.append(Paragraph("3. Inverse Dynamics & Joint Torque Budget", h1_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=SECONDARY, spaceAfter=6))

    elements.append(Paragraph(
        "The complete equation of motion for each leg during gait stance is formulated as:<br/>"
        "&nbsp;&nbsp;<b>τ<sub>actuator</sub></b> = <b>g(q)</b> - <b>J(q)<sup>T</sup></b> * <b>F<sub>ground</sub></b> + <b>I</b> * <b>α<sub>dynamic</sub></b> + <b>τ<sub>coriolis</sub></b><br/>"
        "where link rotational inertia <i>I</i> = (1/3)*m*L^2, angular acceleration <i>α</i> = a_dyn / L, and "
        "<i>F<sub>ground</sub></i> includes dynamic acceleration multiplier and safety factor <i>k<sub>safety</sub></i>.",
        equation_style
    ))

    t_budget = data.get('torque_budget', {})
    cont_req = t_budget.get('continuous_required', {'hip_nm': 0.0, 'knee_nm': 0.0})
    peak_req = t_budget.get('peak_required', {'hip_nm': 0.0, 'knee_nm': 0.0})
    stat_req = t_budget.get('static', {'hip_nm': 0.0, 'knee_nm': 0.0})

    torque_table_data = [
        [Paragraph("Joint Name", table_header_style), Paragraph("Static Torque (Nm)", table_header_style), Paragraph("Continuous Required (Nm)", table_header_style), Paragraph("Peak Required (Nm)", table_header_style)],
        [Paragraph("Hip Joint (Pitch)", table_cell_bold), Paragraph(f"{stat_req['hip_nm']:.2f} Nm", table_cell_style), Paragraph(f"{cont_req['hip_nm']:.2f} Nm", table_cell_bold), Paragraph(f"{peak_req['hip_nm']:.2f} Nm", table_cell_bold)],
        [Paragraph("Knee Joint (Pitch)", table_cell_bold), Paragraph(f"{stat_req['knee_nm']:.2f} Nm", table_cell_style), Paragraph(f"{cont_req['knee_nm']:.2f} Nm", table_cell_bold), Paragraph(f"{peak_req['knee_nm']:.2f} Nm", table_cell_bold)],
    ]
    torque_table_pdf = Table(torque_table_data, colWidths=[150, 125, 125, 125])
    torque_table_pdf.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
    ]))
    elements.append(torque_table_pdf)
    elements.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # SECTION 4: ACTUATOR SELECTION & DUAL-MOTOR CONFIGURATION
    # -------------------------------------------------------------------------
    elements.append(Paragraph("4. Actuator Selection & Dual-Motor Configuration", h1_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=SECONDARY, spaceAfter=6))

    motor_mode = data.get('motor_mode', 'same')
    hip_preset_name = data.get('hip_preset_name', 'DS3218')
    knee_preset_name = data.get('knee_preset_name', 'DS3218')
    hip_spec = data.get('hip_spec', {})
    knee_spec = data.get('knee_spec', {})

    if motor_mode == 'same':
        elements.append(Paragraph(
            f"<b>Configuration:</b> Case 1 — Same Motor Model for All Joints (Hip & Knee identical)<br/>"
            f"<b>Selected Motor:</b> {hip_preset_name}",
            body_style
        ))
    else:
        elements.append(Paragraph(
            f"<b>Configuration:</b> Case 2 — Independent Motors per Joint (Custom Hip vs Knee)<br/>"
            f"<b>Hip Joint Motor:</b> {hip_preset_name}<br/>"
            f"<b>Knee Joint Motor:</b> {knee_preset_name}",
            body_style
        ))

    motor_table_data = [
        [
            Paragraph("Joint / Motor", table_header_style),
            Paragraph("Selected Preset Model", table_header_style),
            Paragraph("Peak Torque", table_header_style),
            Paragraph("Continuous Torque", table_header_style),
            Paragraph("Unit Weight", table_header_style),
            Paragraph("Verdict", table_header_style)
        ],
        [
            Paragraph("Hip Joint Motor", table_cell_bold),
            Paragraph(hip_preset_name.split(' — ')[0], table_cell_style),
            Paragraph(f"{hip_spec.get('peak_nm', 0):.2f} Nm", table_cell_style),
            Paragraph(f"{hip_spec.get('continuous_nm', 0):.2f} Nm", table_cell_style),
            Paragraph(f"{hip_spec.get('mass_kg', 0)*1000:.0f} g", table_cell_style),
            Paragraph(f"<b>{data.get('hip_verdict', 'PASS')}</b>", table_cell_bold)
        ],
        [
            Paragraph("Knee Joint Motor", table_cell_bold),
            Paragraph(knee_preset_name.split(' — ')[0], table_cell_style),
            Paragraph(f"{knee_spec.get('peak_nm', 0):.2f} Nm", table_cell_style),
            Paragraph(f"{knee_spec.get('continuous_nm', 0):.2f} Nm", table_cell_style),
            Paragraph(f"{knee_spec.get('mass_kg', 0)*1000:.0f} g", table_cell_style),
            Paragraph(f"<b>{data.get('knee_verdict', 'PASS')}</b>", table_cell_bold)
        ]
    ]
    motor_table = Table(motor_table_data, colWidths=[100, 160, 65, 75, 60, 65])
    motor_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
    ]))
    elements.append(motor_table)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(
        f"<b>🎯 Motor Standing Height Recommendation:</b> The selected motor combination supports a maximum safe standing height "
        f"up to <b>{data.get('max_safe_height_mm', 'N/A')} mm</b> for a total system mass of <b>{data.get('total_mass_kg', 1.5):.2f} kg</b>.",
        body_style
    ))
    elements.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # SECTION 5: SMART BATTERY & POWER CALCULATOR
    # -------------------------------------------------------------------------
    elements.append(Paragraph("5. Smart Battery & Electrical Power System", h1_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=SECONDARY, spaceAfter=6))

    batt_data = data.get('battery_calc', {})
    elements.append(Paragraph(
        f"<b>Recommended Battery Specs:</b> {batt_data.get('voltage_cell', '2S LiPo (7.4V)')} | "
        f"Capacity: <b>{batt_data.get('req_capacity_mah', 2500)} mAh</b> | "
        f"Discharge C-Rating: <b>{batt_data.get('req_c_rating', 20)}C or higher</b><br/>"
        f"<b>Peak Current Draw:</b> {batt_data.get('total_peak_current', 0):.1f} A &nbsp;|&nbsp; "
        f"<b>Target Runtime:</b> {batt_data.get('target_runtime_min', 20)} minutes",
        body_style
    ))
    elements.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # SECTION 6: MASS BREAKDOWN TABLE
    # -------------------------------------------------------------------------
    elements.append(Paragraph("6. Complete Mass Breakdown", h1_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=SECONDARY, spaceAfter=6))

    mass_comp = data.get('mass_components', {})
    mass_rows = [[Paragraph("Subsystem Component", table_header_style), Paragraph("Mass (kg)", table_header_style), Paragraph("Mass (grams)", table_header_style), Paragraph("Share of Total", table_header_style)]]
    total_m = data.get('total_mass_kg', 1.5)
    for k, v in mass_comp.items():
        pct = (v / total_m * 100) if total_m > 0 else 0
        mass_rows.append([
            Paragraph(k, table_cell_style),
            Paragraph(f"{v:.3f} kg", table_cell_style),
            Paragraph(f"{v*1000:.0f} g", table_cell_bold),
            Paragraph(f"{pct:.1f}%", table_cell_style)
        ])
    mass_rows.append([
        Paragraph("TOTAL SYSTEM MASS", table_cell_bold),
        Paragraph(f"{total_m:.3f} kg", table_cell_bold),
        Paragraph(f"{total_m*1000:.0f} g", table_cell_bold),
        Paragraph("100.0%", table_cell_bold)
    ])
    mass_table = Table(mass_rows, colWidths=[180, 115, 115, 115])
    mass_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#EBF4F6")),
    ]))
    elements.append(mass_table)
    elements.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # SECTION 7: CONTROL THEORY & IMPLEMENTATION STRATEGY
    # -------------------------------------------------------------------------
    elements.append(Paragraph("7. Control Architecture & Implementation Strategy", h1_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=SECONDARY, spaceAfter=6))

    elements.append(Paragraph(
        "<b>Control Loop Structure:</b><br/>"
        "1. <b>Gait Trajectory Planner:</b> Generates cycloid swing and stance foot trajectories in Cartesian space.<br/>"
        "2. <b>Analytical IK Module:</b> Solves joint target angles (q_hip, q_knee) at 100 Hz.<br/>"
        "3. <b>Low-Level Joint Controller:</b> Closed-loop PD control loop on digital servos:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;u(t) = Kp * (q_target - q_actual) + Kd * (q̇_target - q̇_actual)<br/>"
        "4. <b>Hardware Recommendation:</b> Microcontroller (ESP32 / Teensy 4.1) paired with PCA9685 16-Channel PWM Servo Driver "
        "and a 6-axis MPU6050 IMU for body posture stabilization.",
        body_style
    ))

    # Build document
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
