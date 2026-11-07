"""Arm frame for a multirotor / VTOL, generated at the motor positions the balance planner chose (cog_balance.lift_plan).

This closes the loop between balance and geometry: the planner says where the lift motors must be so their thrust centroid sits on the
centre of gravity; this builds a hub with one arm and one motor plate per motor at exactly those positions, and records the motor
mount points in the script as MOTOR_MOUNTS. The assembly reads that list, so after the frame is placed the balance is re-checked on
the real geometry (where the frame actually ended up, with the motors, props and ESCs at the mounts).
The script is build123d code generated from numbers: deterministic, no model writes it. Units mm; part origin = centre of the motor circle.
"""
from __future__ import annotations

import math


def arm_frame(motors, centre_xy=(0.0, 0.0), arm_d=16.0, hub_r=45.0, pitch_mm=25.0, plate_t=3.0, motor_h_mm=25.0, hub_t=8.0,
              plate_mm=None, shaft_clear_mm=8.0):
    """motors: [{x, y, spin}] in assembly mm (lift_plan['motors']); centre_xy: lift_plan['centre_xy_mm']. Returns the script and the mounts."""
    if len(motors) < 3:
        raise ValueError("need at least 3 motors")
    plate = float(plate_mm or max(32.0, pitch_mm + 12.0))
    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    rel = [(round(float(m["x"]) - cx, 3), round(float(m["y"]) - cy, 3)) for m in motors]
    spins = [int(m.get("spin", 1)) for m in motors]
    short = [i + 1 for i, (x, y) in enumerate(rel) if math.hypot(x, y) < hub_r + plate / 2]
    if short:
        raise ValueError(f"motor(s) {short} are too close to the centre for a {hub_r:g} mm hub and {plate:g} mm motor plates: "
                         "use a smaller hub or a larger motor circle")
    z = round(plate_t / 2 + motor_h_mm / 2, 3)
    mounts = [(x, y, z) for x, y in rel]
    arm = max(math.hypot(x, y) for x, y in rel)
    script = f'''from build123d import *
import math
# Balanced arm frame: hub + {len(rel)} arms + motor plates, generated at the planner's motor positions.
# Part origin = centre of the motor circle (assembly x {cx:.1f}, y {cy:.1f} when the frame is placed where it was planned).
MOTOR_MOUNTS = {[tuple(m) for m in mounts]}      # motor centres in this part's frame (read by the assembly for the balance check)
MOTOR_SPIN = {spins}                              # +1 clockwise, -1 counter-clockwise
HUB_R, HUB_T, ARM_D, PLATE, PLATE_T, PITCH, SHAFT = {hub_r}, {hub_t}, {arm_d}, {plate:.2f}, {plate_t}, {pitch_mm}, {shaft_clear_mm}

frame = Cylinder(HUB_R, HUB_T)
for x, y, z in MOTOR_MOUNTS:
    length = math.hypot(x, y)
    theta = math.degrees(math.atan2(y, x))
    frame = frame + Rot(0, 0, theta) * (Pos(length / 2, 0, 0) * (Rot(0, 90, 0) * Cylinder(ARM_D / 2, length)))
    frame = frame + Pos(x, y, 0) * Box(PLATE, PLATE, PLATE_T)
for x, y, z in MOTOR_MOUNTS:
    for sx in (-1, 1):
        for sy in (-1, 1):
            frame = frame - Pos(x + sx * PITCH / 2, y + sy * PITCH / 2, 0) * Cylinder(1.6, PLATE_T * 4)
    frame = frame - Pos(x, y, 0) * Cylinder(SHAFT / 2, PLATE_T * 4)
for k in range(4):
    a = math.radians(45 + 90 * k)
    frame = frame - Pos(25 * math.cos(a), 25 * math.sin(a), 0) * Cylinder(1.7, HUB_T * 4)      # M3 holes to bolt the hub to the fuselage
result = frame
'''
    return {"script": script, "mounts": [list(m) for m in mounts], "spins": spins, "arm_length_mm": round(arm, 1),
            "name": "Motor arm frame (balanced layout)", "material": "PETG",
            "note": "Printed-arm version: arms are solid here. Swap to carbon tubes for a flight frame."}
