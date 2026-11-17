"""Limit-position analysis of the Micro Duck leg (from the URDF + recorded walking data).

  python limit_position.py

No simulator needed. Writes:
  limit_position_result.txt   all numbers
  limit_joint_usage.csv       joint limits vs. range used while walking (for Excel)
  limit_workspace.png         sagittal workspace of the ankle + joint-limit boundaries + walking paths
  limit_toggle.png            leg length / Jacobian determinant / knee torque vs. knee angle
  limit_squat.png             knee torque needed to carry half the weight vs. how deep the robot crouches
  limit_joint_usage.png       bar chart: joint limits vs. range used while walking

Sections
  1. joint limits of the leg joints (URDF)
  2. toggle (dead-centre) position of the knee: thigh and shin collinear
  3. limit positions set by the joint stops: reach forward/back, lowest/highest ankle point
  4. sagittal workspace of the ankle (hip_pitch x knee within their limits)
  5. Jacobian determinant along the knee range (singularity at the toggle position)
  6. statics along a crouch: knee torque to hold half the weight, from straight leg to the joint stops
  7. how close the walking gait (run 4) comes to the limits
"""

import glob
import math
import os
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "assets", "microduck_fixed.urdf")
STANCE = {"left_hip_pitch": -0.3, "left_knee": 0.3, "left_ankle": 0.6,
          "right_hip_pitch": 0.3, "right_knee": -0.3, "right_ankle": -0.6}
MASS_G = 9.81
lines = []


def out(s=""):
    print(s)
    lines.append(s)


def head(title):
    out()
    out("=" * 100)
    out(title)
    out("=" * 100)


# ------------------------------------------------------------------ URDF + forward kinematics
def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(a, q):
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * K @ K


def tf(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


root = ET.parse(URDF).getroot()
links = {l.get("name"): l for l in root.findall("link")}
joints = {}
for j in root.findall("joint"):
    o = j.find("origin")
    xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    r = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    ax = j.find("axis")
    lim = j.find("limit")
    joints[j.get("name")] = dict(
        type=j.get("type"), parent=j.find("parent").get("link"), child=j.find("child").get("link"),
        T=tf(rpy(*r), np.array(xyz)),
        axis=np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()]),
        lo=float(lim.get("lower", 0)) if lim is not None else 0.0,
        hi=float(lim.get("upper", 0)) if lim is not None else 0.0,
        effort=float(lim.get("effort", 0)) if lim is not None else 0.0,
    )
child_of = {v["child"]: k for k, v in joints.items()}
base = [n for n in links if n not in child_of][0]


def mass_of(link):
    m = links[link].find("inertial/mass")
    return float(m.get("value")) if m is not None else 0.0


def fk(q):
    Tl, Tj = {base: np.eye(4)}, {}
    pending = list(joints)
    while pending:
        for name in list(pending):
            j = joints[name]
            if j["parent"] in Tl:
                Tjoint = Tl[j["parent"]] @ j["T"]
                Tj[name] = Tjoint
                R = axis_angle(j["axis"], q.get(name, 0.0)) if j["type"] in ("revolute", "continuous") else np.eye(3)
                Tl[j["child"]] = Tjoint @ tf(R, np.zeros(3))
                pending.remove(name)
    return Tl, Tj


def chain(link):
    js = []
    while link in child_of:
        if joints[child_of[link]]["type"] in ("revolute", "continuous"):
            js.append(child_of[link])
        link = joints[child_of[link]]["parent"]
    return js[::-1]


Tl0, Tj0 = fk({})
LEG = {"left": chain("ankle_left"), "right": chain("ankle_right")}
FOOT = {"left": "ankle_left", "right": "ankle_right"}

# sole point (lowest mesh point of the foot) in the foot link frame
try:
    import trimesh

    def _sole(link):
        pts = []
        for vis in links[link].findall("visual"):
            m = trimesh.load(vis.find("geometry/mesh").get("filename"), force="mesh")
            o = vis.find("origin")
            T = tf(rpy(*[float(v) for v in o.get("rpy", "0 0 0").split()]),
                   np.array([float(v) for v in o.get("xyz", "0 0 0").split()])) if o is not None else np.eye(4)
            m.apply_transform(Tl0[link] @ T)
            pts.append(m.vertices)
        v = np.vstack(pts)
        s = v[v[:, 2] < v[:, 2].min() + 0.003]
        return np.array([(s[:, 0].min() + s[:, 0].max()) / 2, (s[:, 1].min() + s[:, 1].max()) / 2, v[:, 2].min()])
    SOLE_LOCAL = {s: np.linalg.inv(Tl0[f]) @ np.r_[_sole(f), 1] for s, f in FOOT.items()}
    SOLE_NOTE = "sole centre from the foot meshes"
except Exception as e:  # noqa: BLE001
    SOLE_LOCAL = {s: np.r_[0, 0, 0, 1.0] for s in FOOT}
    SOLE_NOTE = f"trimesh not available ({e}); foot point = ankle joint centre"


def leg_points(q, side):
    """hip_pitch centre, knee centre, ankle centre, sole point (base frame, m)."""
    Tl, Tj = fk(q)
    L = LEG[side]
    hp = [n for n in L if n.endswith("hip_pitch")][0]
    kn = [n for n in L if n.endswith("knee")][0]
    an = [n for n in L if n.endswith("ankle")][0]
    sole = (Tl[FOOT[side]] @ SOLE_LOCAL[side])[:3]
    return Tj[hp][:3, 3], Tj[kn][:3, 3], Tj[an][:3, 3], sole, Tj


def jac(q, side, point="ankle"):
    """Geometric linear Jacobian of the ankle centre (or sole) w.r.t. the leg joints, base fixed."""
    hip, knee, ank, sole, Tj = leg_points(q, side)
    p = ank if point == "ankle" else sole
    J = np.zeros((3, len(LEG[side])))
    for k, n in enumerate(LEG[side]):
        z = Tj[n][:3, :3] @ (joints[n]["axis"] / np.linalg.norm(joints[n]["axis"]))
        J[:, k] = np.cross(z, p - Tj[n][:3, 3])
    return J


def jn(side, suffix):
    return [n for n in LEG[side] if n.endswith(suffix)][0]


d = math.degrees
total_m = sum(mass_of(l) for l in links)
W = total_m * MASS_G
out(f"Micro Duck limit-position analysis   (URDF {os.path.basename(URDF)}, mass {total_m:.3f} kg, weight {W:.2f} N)")
out(f"foot point: {SOLE_NOTE}")
out("frame: robot base frame, x = forward, y = left, z = up; sagittal plane = x-z. Leg analysed: LEFT "
    "(right leg is its mirror image, its joint axes are flipped so the angle signs are opposite).")

# ------------------------------------------------------------------ 1. joint limits
head("1. JOINT LIMITS OF THE LEG JOINTS (from the URDF)")
out(f"{'joint':18s} {'lower (deg)':>11s} {'upper (deg)':>11s} {'range (deg)':>11s} {'stance (deg)':>12s} "
    f"{'effort (Nm)':>11s}  axis in base frame")
for side in ("left", "right"):
    for n in LEG[side]:
        j = joints[n]
        a = Tj0[n][:3, :3] @ (j["axis"] / np.linalg.norm(j["axis"]))
        out(f"{n:18s} {d(j['lo']):11.1f} {d(j['hi']):11.1f} {d(j['hi'] - j['lo']):11.1f} "
            f"{d(STANCE.get(n, 0.0)):12.1f} {j['effort']:11.2f}  [{a[0]:+.2f} {a[1]:+.2f} {a[2]:+.2f}]")

# ------------------------------------------------------------------ geometry at q = 0
side = "left"
HP, KN, AN = jn(side, "hip_pitch"), jn(side, "knee"), jn(side, "ankle")
hip0, knee0, ank0, sole0, _ = leg_points({}, side)
l1 = np.linalg.norm(knee0 - hip0)
l2 = np.linalg.norm(ank0 - knee0)
l3 = np.linalg.norm(sole0 - ank0)
l1s = np.linalg.norm((knee0 - hip0)[[0, 2]])
l2s = np.linalg.norm((ank0 - knee0)[[0, 2]])
head("2. LEG LINKS AND THE TOGGLE (DEAD-CENTRE) POSITION OF THE KNEE")
out(f"thigh  hip_pitch -> knee  : {l1 * 1000:6.1f} mm (3D)   {l1s * 1000:6.1f} mm in the sagittal plane")
out(f"shin   knee -> ankle      : {l2 * 1000:6.1f} mm (3D)   {l2s * 1000:6.1f} mm in the sagittal plane")
out(f"foot   ankle -> sole      : {l3 * 1000:6.1f} mm")
out(f"max possible hip->ankle reach if the knee could reach the toggle: l1 + l2 = {(l1 + l2) * 1000:.1f} mm "
    f"(sagittal {(l1s + l2s) * 1000:.1f} mm)")

# knee sweep (hip_pitch = ankle = 0), wide range to find the geometric toggle angle
qk_all = np.radians(np.linspace(-180, 180, 3601))
reach_all = np.array([np.linalg.norm(leg_points({KN: q}, side)[2] - hip0) for q in qk_all])
k_tog = qk_all[np.argmax(reach_all)]
k_fold = qk_all[np.argmin(reach_all)]
lo_k, hi_k = joints[KN]["lo"], joints[KN]["hi"]
in_lim = (qk_all >= lo_k) & (qk_all <= hi_k)
reach_lim_max = reach_all[in_lim].max()
reach_lim_min = reach_all[in_lim].min()
k_lim_max = qk_all[in_lim][np.argmax(reach_all[in_lim])]
k_lim_min = qk_all[in_lim][np.argmin(reach_all[in_lim])]
reach_zero = np.linalg.norm(ank0 - hip0)
out()
out(f"knee angle at the toggle position (thigh and shin collinear, leg fully stretched): {d(k_tog):+.1f} deg "
    f"-> hip->ankle {reach_all.max() * 1000:.1f} mm")
out(f"knee angle at the folded dead centre (shin folded back onto the thigh)          : {d(k_fold):+.1f} deg "
    f"-> hip->ankle {reach_all.min() * 1000:.1f} mm (geometric only, parts would collide)")
out(f"knee joint limits: {d(lo_k):+.1f} .. {d(hi_k):+.1f} deg -> toggle position is "
    f"{'INSIDE' if lo_k <= k_tog <= hi_k else 'OUTSIDE'} the joint range")
out(f"at q = 0 the leg is {d(abs(k_tog)):.1f} deg away from the toggle; hip->ankle {reach_zero * 1000:.1f} mm "
    f"= {reach_zero / reach_all.max() * 100:.1f} % of the stretched length")
out(f"within the joint limits: longest leg {reach_lim_max * 1000:.1f} mm at knee {d(k_lim_max):+.1f} deg, "
    f"shortest {reach_lim_min * 1000:.1f} mm at knee {d(k_lim_min):+.1f} deg "
    f"-> leg length can change by {(reach_lim_max - reach_lim_min) * 1000:.1f} mm")
st_reach = np.linalg.norm(leg_points(STANCE, side)[2] - leg_points(STANCE, side)[0])
out(f"standing pose (knee {d(STANCE[KN]):+.1f} deg): hip->ankle {st_reach * 1000:.1f} mm "
    f"= {st_reach / reach_all.max() * 100:.1f} % of the stretched length, {d(abs(STANCE[KN] - k_tog)):.1f} deg "
    f"from the toggle")

# ------------------------------------------------------------------ 3. limit positions of the foot
head("3. LIMIT POSITIONS SET BY THE JOINT STOPS (hip_yaw = hip_roll = 0, ankle = 0; ankle centre "
     "relative to the hip_pitch centre, mm)")
lo_h, hi_h = joints[HP]["lo"], joints[HP]["hi"]
NG = 121
qh = np.linspace(lo_h, hi_h, NG)
qk = np.linspace(lo_k, hi_k, NG)
WS = np.zeros((NG, NG, 3))
for a, h in enumerate(qh):
    for b, k in enumerate(qk):
        WS[a, b] = leg_points({HP: h, KN: k}, side)[2] - hip0
X, Z = WS[..., 0] * 1000, WS[..., 2] * 1000
for label, idx in (("furthest forward", np.unravel_index(np.argmax(X), X.shape)),
                   ("furthest back", np.unravel_index(np.argmin(X), X.shape)),
                   ("lowest (leg reaches down)", np.unravel_index(np.argmin(Z), Z.shape)),
                   ("highest (foot lifted)", np.unravel_index(np.argmax(Z), Z.shape))):
    a, b = idx
    out(f"{label:28s} x {X[a, b]:+7.1f}  z {Z[a, b]:+7.1f}   at hip_pitch {d(qh[a]):+7.1f} deg, knee {d(qk[b]):+7.1f} deg"
        f"{'  (hip at stop)' if a in (0, NG - 1) else ''}{'  (knee at stop)' if b in (0, NG - 1) else ''}")
out(f"workspace size: {X.max() - X.min():.1f} mm (x) by {Z.max() - Z.min():.1f} mm (z)")
area = 0.0
for a in range(NG - 1):  # area of the mapped grid (sum of quads, shoelace)
    for b in range(NG - 1):
        P = np.array([[X[a, b], Z[a, b]], [X[a + 1, b], Z[a + 1, b]], [X[a + 1, b + 1], Z[a + 1, b + 1]], [X[a, b + 1], Z[a, b + 1]]])
        area += 0.5 * abs(np.dot(P[:, 0], np.roll(P[:, 1], 1)) - np.dot(P[:, 1], np.roll(P[:, 0], 1)))
out(f"workspace area (sagittal, ankle centre): {area:.0f} mm^2 = {area / 100:.1f} cm^2 "
    f"(may count overlap twice if the map folds; no fold when the toggle is outside the knee range)")

# ------------------------------------------------------------------ 4/5. Jacobian determinant along the knee range
head("4. SINGULARITY: DETERMINANT OF THE PLANAR JACOBIAN  d(x,z)_ankle / d(hip_pitch, knee)")
ih, ik = LEG[side].index(HP), LEG[side].index(KN)
out("theory (planar 2R arm): det J = l1 * l2 * sin(theta_knee')  where theta_knee' is the angle between thigh "
    "and shin lines -> det = 0 at the toggle position (leg straight) -> the foot cannot move along the leg.")
kk = np.radians(np.linspace(d(lo_k), d(hi_k), 13))
out(f"{'knee (deg)':>10s} {'from toggle (deg)':>17s} {'hip->ankle (mm)':>15s} {'|det J| (cm^2/rad^2)':>20s} "
    f"{'l1 l2 |sin| (cm^2)':>18s}")
for k in kk:
    J = jac({KN: k}, side)[[0, 2]][:, [ih, ik]]
    reach = np.linalg.norm(leg_points({KN: k}, side)[2] - hip0)
    out(f"{d(k):10.1f} {d(k - k_tog):17.1f} {reach * 1000:15.1f} {abs(np.linalg.det(J)) * 1e4:20.2f} "
        f"{l1s * l2s * abs(math.sin(k - k_tog)) * 1e4:18.2f}")

# ------------------------------------------------------------------ 6. crouch statics
head(f"5. STATICS ALONG A CROUCH: torque to carry half the weight ({W / 2:.2f} N) on one sole, "
     "stance ratios scaled by c (c = 1 is the standing pose)")
out("pose(c): hip_pitch = c * stance_hip, knee = c * stance_knee, ankle = c * stance_ankle. tau = J_sole^T F, "
    "F = vertical ground reaction. c is increased until a joint stop is reached.")
ratios = [(joints[n]["hi"] if STANCE[n] > 0 else joints[n]["lo"]) / STANCE[n] for n in (HP, KN, AN)]
ratios = [r for r in ratios if r > 0]
cmax = min(min(ratios) if ratios else 3.0, 6.0)
cs = np.linspace(0.0, cmax, 61)
rows = []
for c in cs:
    q = {n: c * STANCE[n] for n in (HP, KN, AN)}
    hip, knee, ank, sole, _ = leg_points(q, side)
    Js = jac(q, side, "sole")
    tau = Js.T @ np.array([0, 0, W / 2])
    rows.append((c, d(q[KN]), (hip - sole)[2] * 1000, (sole - hip)[0] * 1000,
                 tau[LEG[side].index(HP)], tau[LEG[side].index(KN)], tau[LEG[side].index(AN)]))
rows = np.array(rows)
out(f"{'c':>5s} {'knee (deg)':>10s} {'hip height (mm)':>15s} {'sole x (mm)':>11s} {'tau_hip (Nm)':>12s} "
    f"{'tau_knee (Nm)':>13s} {'tau_ankle (Nm)':>14s} {'knee % of limit':>15s}")
for r in rows[::6]:
    out(f"{r[0]:5.2f} {r[1]:10.1f} {r[2]:15.1f} {r[3]:11.1f} {r[4]:+12.3f} {r[5]:+13.3f} {r[6]:+14.3f} "
        f"{abs(r[5]) / joints[KN]['effort'] * 100:14.1f}%")
out(f"crouch stops at c = {cmax:.2f} (first joint stop). The deeper the crouch, the larger the knee lever arm "
    "-> the knee torque grows; a straight leg (near the toggle) carries the weight with almost no knee torque.")

# ------------------------------------------------------------------ 7. walking vs limits
head("6. HOW CLOSE THE WALKING GAIT (RUN 4) COMES TO THE LIMITS")
csvs = sorted(glob.glob(os.path.join(HERE, "gait_run4_*.csv")))
walk = {}
for f in csvs:
    with open(f) as fh:
        cols = fh.readline().strip().split(",")
    data = np.loadtxt(f, delimiter=",", skiprows=1)
    walk[os.path.basename(f)[10:-4]] = (cols, data)
if not walk:
    out("no gait_run4_*.csv found - run gait_run4.bat first")
usage_rows = []
paths = {}
for tag, (cols, data) in walk.items():
    col = {c: i for i, c in enumerate(cols)}
    qrec = {n: data[:, col["q_" + n]] for s in LEG for n in LEG[s] if "q_" + n in col}
    P = []
    for t in range(0, len(data), 2):
        q = {n: v[t] for n, v in qrec.items()}
        h, _, a, _, _ = leg_points(q, side)
        P.append((a - h)[[0, 2]] * 1000)
    paths[tag] = np.array(P)
    for s in LEG:
        for n in LEG[s]:
            v = qrec[n]
            j = joints[n]
            rng = j["hi"] - j["lo"]
            usage_rows.append((tag, n, d(j["lo"]), d(j["hi"]), d(v.min()), d(v.max()),
                               (v.max() - v.min()) / rng * 100,
                               d(min(v.min() - j["lo"], j["hi"] - v.max()))))
    kv = qrec[KN]
    reach = np.linalg.norm(paths[tag] - 0, axis=1)
    out(f"vx {tag}: left knee {d(kv.min()):+.1f} .. {d(kv.max()):+.1f} deg, closest to the toggle "
        f"{d(np.min(np.abs(kv - k_tog))):.1f} deg; hip->ankle (sagittal) {reach.min():.1f} .. {reach.max():.1f} mm "
        f"= {reach.max() / (reach_all.max() * 1000) * 100:.1f} % of the stretched leg at most")
if usage_rows:
    out()
    out(f"{'speed':6s} {'joint':18s} {'limit lo':>8s} {'limit hi':>8s} {'walk min':>8s} {'walk max':>8s} "
        f"{'% range used':>12s} {'margin to stop (deg)':>20s}")
    for r in usage_rows:
        out(f"{r[0]:6s} {r[1]:18s} {r[2]:8.1f} {r[3]:8.1f} {r[4]:8.1f} {r[5]:8.1f} {r[6]:11.1f}% {r[7]:20.1f}")
    with open(os.path.join(HERE, "limit_joint_usage.csv"), "w") as fh:
        fh.write("speed,joint,limit_lo_deg,limit_hi_deg,walk_min_deg,walk_max_deg,pct_range_used,margin_to_stop_deg\n")
        for r in usage_rows:
            fh.write(",".join([r[0], r[1]] + [f"{x:.2f}" for x in r[2:]]) + "\n")

# ------------------------------------------------------------------ plots
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(X.ravel(), Z.ravel(), s=1, c="#cfd8e3", label="reachable (joint limits)")
for a, lab in ((0, "hip at lower stop"), (NG - 1, "hip at upper stop")):
    ax.plot(X[a], Z[a], lw=2, label=lab)
for b, lab in ((0, "knee at lower stop"), (NG - 1, "knee at upper stop")):
    ax.plot(X[:, b], Z[:, b], lw=2, ls="--", label=lab)
th = np.linspace(0, 2 * np.pi, 400)
ax.plot((l1s + l2s) * 1000 * np.cos(th), (l1s + l2s) * 1000 * np.sin(th), "k:", lw=1,
        label=f"toggle circle r = l1 + l2 = {(l1s + l2s) * 1000:.0f} mm")
st = leg_points(STANCE, side)
ax.plot(*((st[2] - st[0])[[0, 2]] * 1000), "k*", ms=14, label="standing pose")
for tag, P in paths.items():
    ax.plot(P[:, 0], P[:, 1], lw=1.5, label=f"walking path vx {tag}")
ax.plot(0, 0, "ko")
ax.set_aspect("equal")
ax.set_xlabel("x forward (mm), relative to hip_pitch joint")
ax.set_ylabel("z up (mm)")
ax.set_title("Left ankle workspace (sagittal plane) and limit positions")
ax.legend(fontsize=7, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "limit_workspace.png"), dpi=150)

fig, axs = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
qd = np.degrees(qk_all)
detv = np.array([abs(np.linalg.det(jac({KN: q}, side)[[0, 2]][:, [ih, ik]])) * 1e4 for q in qk_all[::10]])
axs[0].plot(qd, reach_all * 1000)
axs[0].set_ylabel("hip->ankle (mm)")
axs[1].plot(qd[::10], detv)
axs[1].set_ylabel("|det J| (cm^2)")
tq = []
for q in qk_all[::10]:
    Js = jac({KN: q}, side, "sole")
    tq.append(abs((Js.T @ np.array([0, 0, W / 2]))[ik]))
axs[2].plot(qd[::10], tq)
axs[2].axhline(joints[KN]["effort"], color="r", ls=":", label="motor limit")
axs[2].set_ylabel("|tau_knee| for W/2 (Nm)")
axs[2].set_xlabel("knee angle (deg), hip_pitch = ankle = 0")
for a in axs:
    a.axvspan(d(lo_k), d(hi_k), color="g", alpha=0.08)
    a.axvline(d(k_tog), color="k", ls="--", lw=1)
    a.grid(alpha=0.3)
axs[0].set_title("Toggle position (dashed) and knee joint range (green)")
axs[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "limit_toggle.png"), dpi=150)

fig, ax = plt.subplots(figsize=(7, 4.5))
for i, lab in ((4, "hip_pitch"), (5, "knee"), (6, "ankle")):
    ax.plot(rows[:, 1], np.abs(rows[:, i]), label=lab)
ax.axvline(d(STANCE[KN]), color="k", ls="--", lw=1, label="standing pose")
ax.set_xlabel("knee angle (deg)  (right = deeper crouch; all leg joints scaled together)")
ax.set_ylabel("|torque| to carry W/2 (Nm)")
ax.set_title("Joint torque vs. crouch depth (one leg carries half the weight)")
ax2 = ax.twiny()  # hip height on the top axis
ax2.set_xlim(ax.get_xlim())
ticks = rows[::12, 1]
ax2.set_xticks(ticks)
ax2.set_xticklabels([f"{h:.0f}" for h in rows[::12, 2]])
ax2.set_xlabel("hip height above sole (mm)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "limit_squat.png"), dpi=150)

if usage_rows:
    fig, ax = plt.subplots(figsize=(8, 4))
    names = LEG["left"]
    for i, n in enumerate(names):
        j = joints[n]
        ax.barh(i, d(j["hi"] - j["lo"]), left=d(j["lo"]), color="#dde3ea", edgecolor="k")
        vs = [r for r in usage_rows if r[1] == n]
        ax.barh(i, max(r[5] for r in vs) - min(r[4] for r in vs), left=min(r[4] for r in vs), height=0.4,
                color="#3b82f6")
        ax.plot(d(STANCE.get(n, 0.0)), i, "k|", ms=18)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("angle (deg)   grey = joint limits, blue = used while walking (all speeds), | = stance")
    ax.set_title("Left leg: joint limits vs. range used while walking (run 4)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "limit_joint_usage.png"), dpi=150)

if paths:  # close-up of the walking paths (they are small in the full workspace plot)
    fig, ax = plt.subplots(figsize=(6, 5))
    for tag, P in paths.items():
        ax.plot(P[:, 0], P[:, 1], lw=1.5, label=f"vx {tag.replace('p', '.')} m/s")
    ax.plot(*((st[2] - st[0])[[0, 2]] * 1000), "k*", ms=14, label="standing pose")
    allp = np.vstack(list(paths.values()) + [((st[2] - st[0])[[0, 2]] * 1000)[None]])
    lo_xy, hi_xy = allp.min(0) - 8, allp.max(0) + 8
    th2 = np.linspace(0, 2 * np.pi, 2000)
    for r, lab in (((l1s + l2s) * 1000, "toggle circle (leg straight)"),):
        ax.plot(r * np.cos(th2), r * np.sin(th2), "k:", lw=1, label=lab)
    ax.set_xlim(lo_xy[0], hi_xy[0])
    ax.set_ylim(lo_xy[1], hi_xy[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x forward (mm), relative to hip_pitch joint")
    ax.set_ylabel("z up (mm)")
    ax.set_title("Left ankle path while walking (run 4), close-up")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "limit_walk_zoom.png"), dpi=150)

# collect the report figures in one folder
import shutil  # noqa: E402

FIG_DIR = os.path.join(HERE, "report_figures", "limit_position")
os.makedirs(FIG_DIR, exist_ok=True)
for i, n in enumerate(["limit_workspace.png", "limit_walk_zoom.png", "limit_toggle.png", "limit_squat.png",
                       "limit_joint_usage.png"], 1):
    if os.path.exists(os.path.join(HERE, n)):
        shutil.copy(os.path.join(HERE, n), os.path.join(FIG_DIR, f"fig{i}_{n}"))
for n in ("limit_position_result.txt", "limit_joint_usage.csv"):
    if os.path.exists(os.path.join(HERE, n)):
        shutil.copy(os.path.join(HERE, n), os.path.join(FIG_DIR, n))

with open(os.path.join(HERE, "limit_position_result.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
shutil.copy(os.path.join(HERE, "limit_position_result.txt"), os.path.join(FIG_DIR, "limit_position_result.txt"))
print("wrote limit_position_result.txt, limit_joint_usage.csv, limit_workspace.png, limit_toggle.png, "
      "limit_squat.png, limit_joint_usage.png")
print("LIMIT DONE")
