"""Turn gait_*.npz from record_gait.py into report figures + a gait summary.

  python plot_gait.py gait_0p15.npz

Figures (PNG, next to the npz):
  *_foot_path.png    foot path relative to the hip (side view) - the "coupler curve"
  *_joints.png       leg joint angles over time
  *_foot_vel.png     foot velocity (numerical derivative) over time
  *_torque.png       joint torque vs the 1 Nm motor limit
Prints: step frequency, step length, duty factor, foot lift, peak joint speed/torque.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "gait_0p15.npz"
d = np.load(path)
base = path[:-4]
t, dt = d["t"], float(d["dt"])
names = [str(n) for n in d["joint_names"]]
legs = [n for n in names if n.split("_", 1)[0] in ("left", "right")]

# ---- stance / swing from contact force
contact_l = d["fz_l"] > 0.5
contact_r = d["fz_r"] > 0.5
duty_l, duty_r = contact_l.mean(), contact_r.mean()
touchdowns = np.flatnonzero(np.diff(contact_l.astype(int)) == 1)
period = float(np.mean(np.diff(touchdowns)) * dt) if len(touchdowns) > 1 else float("nan")
vx = float(d["v_base"][:, 0].mean())
fl, fr = d["foot_l"], d["foot_r"]
lift = float(fl[:, 2].max() - fl[:, 2].min())
stride = vx * period

print(f"commanded vx {float(d['vx_cmd']):.3f} m/s, achieved {vx:.3f} m/s")
print(f"gait period {period:.3f} s  (step frequency {1 / period:.2f} Hz per leg)")
print(f"stride length {stride * 100:.1f} cm, foot lift {lift * 1000:.1f} mm")
print(f"duty factor L {duty_l:.2f}  R {duty_r:.2f}")

# ---- 1. foot path relative to hip (x forward, z up)
fig, ax = plt.subplots(figsize=(6, 5))
for foot, c, lab in ((fl, contact_l, "left"), (fr, contact_r, "right")):
    ax.plot(foot[:, 0] * 100, foot[:, 2] * 100, lw=0.8, alpha=0.5, label=f"{lab} foot")
    ax.scatter(foot[c, 0] * 100, foot[c, 2] * 100, s=4, label=f"{lab} stance")
ax.set_xlabel("x relative to hip (cm, forward +)")
ax.set_ylabel("z relative to hip (cm)")
ax.set_title("Foot path relative to hip (learned 'coupler curve')")
ax.axis("equal")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(base + "_foot_path.png", dpi=150)

# ---- 2. joint angles
fig, ax = plt.subplots(figsize=(9, 5))
for n in legs:
    if n.startswith("left"):
        ax.plot(t, d["q"][:, names.index(n)], label=n)
ax.set_xlabel("time (s)")
ax.set_ylabel("joint angle (rad)")
ax.set_title("Left-leg joint angles")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(base + "_joints.png", dpi=150)

# ---- 3. foot velocity by numerical differentiation (central difference)
v_l = np.gradient(fl, dt, axis=0)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t, v_l[:, 0], label="vx (forward)")
ax.plot(t, v_l[:, 2], label="vz (up)")
ax.fill_between(t, v_l.min(), v_l.max(), where=contact_l, alpha=0.1, label="stance")
ax.set_xlabel("time (s)")
ax.set_ylabel("left foot velocity rel. hip (m/s)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(base + "_foot_vel.png", dpi=150)

# ---- 4. torques vs limit
fig, ax = plt.subplots(figsize=(9, 4))
for n in legs:
    if n.startswith("left"):
        ax.plot(t, d["tau"][:, names.index(n)], label=n)
ax.axhline(1.0, ls="--", c="k", lw=0.8)
ax.axhline(-1.0, ls="--", c="k", lw=0.8, label="motor limit")
ax.set_xlabel("time (s)")
ax.set_ylabel("torque (Nm)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(base + "_torque.png", dpi=150)

print("peak |joint speed| (rad/s) / peak |torque| (Nm) / mean mech. power (W):")
for n in legs:
    j = names.index(n)
    p = np.mean(np.abs(d["tau"][:, j] * d["qd"][:, j]))
    print(f"  {n:18s} {np.abs(d['qd'][:, j]).max():6.2f}   {np.abs(d['tau'][:, j]).max():5.2f}   {p:.3f}")
print("wrote", base + "_*.png")
