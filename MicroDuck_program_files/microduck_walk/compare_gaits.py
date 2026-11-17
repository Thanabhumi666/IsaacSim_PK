"""Compare recorded gaits from different training runs (report figure).

  python compare_gaits.py "Run 2 (hop) model_400=gait_stdfix400.npz" "Run 3 (biped) model_700=gait_0p15.npz"

Each argument is  label=file.npz  (files from record_gait.py).
Writes compare_gait_diagram.png  - footfall diagram (stance bars per foot) + foot height
       compare_summary.txt       - table: speed, duty factors, period, stride, peak torque, power
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SHOW_S = 3.0  # seconds shown in the footfall diagram
pairs = [a.rsplit("=", 1) for a in sys.argv[1:]]
if not pairs:
    sys.exit(__doc__)

rows = []
fig, axes = plt.subplots(len(pairs), 1, figsize=(10, 2.6 * len(pairs)), squeeze=False)
for ax, (label, path) in zip(axes[:, 0], pairs):
    d = np.load(path)
    t, dt = d["t"], float(d["dt"])
    names = [str(n) for n in d["joint_names"]]
    cl, cr = d["fz_l"] > 0.5, d["fz_r"] > 0.5
    n = int(SHOW_S / dt)

    # footfall diagram: one bar per stance phase
    for y, c, col in ((1, cl[:n], "tab:blue"), (0, cr[:n], "tab:orange")):
        edges = np.flatnonzero(np.diff(np.r_[0, c.astype(int), 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            ax.broken_barh([(t[s], (e - s) * dt)], (y - 0.35, 0.7), color=col)
    ax.set_yticks([0, 1], ["right foot", "left foot"])
    ax.set_xlim(0, SHOW_S)
    ax.set_title(label, fontsize=10, loc="left")
    ax2 = ax.twinx()
    ax2.plot(t[:n], (d["foot_l"][:n, 2] - d["foot_l"][:, 2].min()) * 1000, c="tab:blue", lw=0.8, alpha=0.6)
    ax2.plot(t[:n], (d["foot_r"][:n, 2] - d["foot_r"][:, 2].min()) * 1000, c="tab:orange", lw=0.8, alpha=0.6)
    ax2.set_ylabel("foot lift (mm)", fontsize=8)

    td = np.flatnonzero(np.diff(cl.astype(int)) == 1)
    if len(td) < 2:
        td = np.flatnonzero(np.diff(cr.astype(int)) == 1)
    period = float(np.mean(np.diff(td)) * dt) if len(td) > 1 else float("nan")
    vx = float(d["v_base"][:, 0].mean())
    both = float((cl & cr).mean())
    none = float((~cl & ~cr).mean())
    legs = [i for i, nm in enumerate(names) if nm.split("_", 1)[0] in ("left", "right")]
    tau = np.abs(d["tau"][:, legs])
    power = float(np.mean(np.sum(np.abs(d["tau"][:, legs] * d["qd"][:, legs]), axis=1)))
    sat = float((tau > 0.98).mean())
    rows.append((label, float(d["vx_cmd"]), vx, cl.mean(), cr.mean(), both, none, period, vx * period, power, sat))

axes[-1, 0].set_xlabel("time (s)  -  bars = foot on the ground (stance)")
fig.tight_layout()
fig.savefig("compare_gait_diagram.png", dpi=150)

hdr = (f"{'run':38s} {'cmd':>5s} {'vx':>6s} {'dutyL':>6s} {'dutyR':>6s} {'double':>6s} {'flight':>6s}"
       f" {'T(s)':>6s} {'stride(cm)':>10s} {'P(W)':>6s} {'torque@limit':>12s}")
lines = [hdr]
for r in rows:
    lines.append(f"{r[0][:38]:38s} {r[1]:5.2f} {r[2]:6.3f} {r[3]:6.2f} {r[4]:6.2f} {r[5]:6.2f} {r[6]:6.2f}"
                 f" {r[7]:6.3f} {r[8] * 100:10.1f} {r[9]:6.2f} {r[10] * 100:11.1f}%")
lines += ["", "double = both feet down, flight = no foot down (fraction of time)",
          "P = mean total mechanical power of leg joints, torque@limit = share of samples with |tau| >= 0.98 Nm"]
open("compare_summary.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("wrote compare_gait_diagram.png, compare_summary.txt")
