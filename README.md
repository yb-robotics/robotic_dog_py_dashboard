# Quadruped Design Dashboard

## Run
```
pip install -r requirements.txt
streamlit run app.py
```

## What it does
- Input: total DOF (8 or 12) + approx standing height.
- Derives: femur/tibia link lengths, ab/ad offset, body length/width
  (assumptions documented inline in `kinematics.py::derive_dimensions`).
- Live IK/FK on a slider-driven foot target, with round-trip validation
  shown per-solve.
- Two-panel diagram (leg side view + body top view) redraws on every input
  change.
- Component-based weight builder (actuator/battery/frame/electronics mass
  inputs) feeding directly into torque sizing with adjustable safety factor.
- One-click generation of N random IK/FK test cases with pass/fail
  validation table + error histogram, using your current link lengths.

Everything is driven by Streamlit's automatic rerun-on-interaction — change
any slider/input and every downstream number, table, and plot updates.
