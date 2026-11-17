"""Mechanism analysis of the Micro Duck straight from its URDF (no simulator needed).

  python kinematics_report.py

Prints / writes kinematics_result.txt:
  1. joint table: parent -> child link, joint axis (in base frame at q=0), joint centre, limits
  2. DOF by Gruebler/Kutzbach (spatial): floating, double support, single support
  3. leg chain geometry: hip -> knee -> ankle -> sole distances
  4. foot Jacobian (geometric, linear part) at the zero pose and at the standing pose,
     compared with the finite differences measured in Isaac Sim (find_stance.py)
  5. statics: joint torques to hold half the body weight on each foot, tau = J^T F
"""

import math
import os
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "assets", "microduck_fixed.urdf")
STANCE = {"left_hip_pitch": -0.3, "left_knee": 0.3, "left_ankle": 0.6,
          "right_hip_pitch": 0.3, "right_knee": -0.3, "right_ankle": -0.6}
G = 9.81
lines = []


def out(s=""):
    print(s)
    lines.append(s)


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


def com_local(link):
    o = links[link].find("inertial/origin")
    return np.array([float(v) for v in o.get("xyz").split()]) if o is not None else np.zeros(3)


def fk(q):
    """World (= base frame) transform of every link and of every joint frame."""
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
    """Moving (revolute) joints from the base to `link`; fixed joints are rigid, not DOF."""
    js = []
    while link in child_of:
        if joints[child_of[link]]["type"] in ("revolute", "continuous"):
            js.append(child_of[link])
        link = joints[child_of[link]]["parent"]
    return js[::-1]


# ---------------------------------------------------------------- 1. joint table
Tl0, Tj0 = fk({})
out("=" * 100)
out(f"1. JOINT TABLE  (base link: {base}; axis and centre expressed in the base frame at q = 0)")
out("=" * 100)
out(f"{'joint':16s} {'type':9s} {'parent -> child':34s} {'axis (base frame)':22s} {'centre (mm)':24s} {'limits (deg)':16s}")
for name, j in joints.items():
    a = Tj0[name][:3, :3] @ (j["axis"] / np.linalg.norm(j["axis"]))
    c = Tj0[name][:3, 3] * 1000
    lim = f"{math.degrees(j['lo']):6.1f}..{math.degrees(j['hi']):5.1f}" if j["type"] == "revolute" else "-"
    out(f"{name:16s} {j['type']:9s} {j['parent'] + ' -> ' + j['child']:34s} "
        f"[{a[0]:+.2f} {a[1]:+.2f} {a[2]:+.2f}]   [{c[0]:7.1f} {c[1]:7.1f} {c[2]:7.1f}]  {lim}")

# ---------------------------------------------------------------- 2. DOF
rev = [n for n, j in joints.items() if j["type"] in ("revolute", "continuous")]
n_links = len(links)
out()
out("=" * 100)
out("2. DOF  (Kutzbach, spatial:  M = 6(n-1) - 5 j1,  revolute joint = 1 DOF / removes 5)")
out("=" * 100)
n_fixed = len(joints) - len(rev)
out(f"links {n_links}, revolute joints {len(rev)}, fixed joints {n_fixed} "
    f"(a fixed joint welds two links into one rigid body -> {n_links - n_fixed} bodies)")
leg_l, leg_r = chain("ankle_left"), chain("ankle_right")
head = [n for n in rev if n not in leg_l + leg_r]
out(f"left leg chain : {' -> '.join(leg_l)}  ({len(leg_l)} joints)")
out(f"right leg chain: {' -> '.join(leg_r)}  ({len(leg_r)} joints)")
out(f"head joints    : {', '.join(head)}  ({len(head)} joints)")
out(f"floating robot : 6 (free base) + {len(rev)} joints = {6 + len(rev)} DOF")
# double support: both soles welded to the ground -> one closed loop ground-legL-base-legR-ground
nb = 1 + 1 + (len(leg_l) - 1) + (len(leg_r) - 1)  # ground(+feet), base, moving leg links
M2 = 6 * (nb - 1) - 5 * (len(leg_l) + len(leg_r))
out(f"double support : legs + base form a closed loop: n = {nb} (ground incl. feet, base, leg links), "
    f"j = {len(leg_l) + len(leg_r)}  ->  M = 6({nb}-1) - 5({len(leg_l) + len(leg_r)}) = {M2}  (+{len(head)} head = {M2 + len(head)})")
out(f"single support : one leg = open chain of {len(leg_l)} joints -> the base can move in only "
    f"{len(leg_l)} of its 6 directions relative to the stance foot (no ankle roll) -> sideways balance "
    f"must come from hip_roll / stepping")

# ---------------------------------------------------------------- 3. leg geometry
out()
out("=" * 100)
out("3. LEFT LEG GEOMETRY AT q = 0 (joint centres in the base frame, mm)")
out("=" * 100)
prev = None
for name in leg_l:
    c = Tj0[name][:3, 3]
    d = "" if prev is None else f"   distance from previous joint {np.linalg.norm(c - prev) * 1000:6.1f} mm"
    out(f"  {name:16s} [{c[0] * 1000:7.1f} {c[1] * 1000:7.1f} {c[2] * 1000:7.1f}]{d}")
    prev = c

# sole point: lowest mesh point of the ankle link at q=0, taken from the visual meshes if trimesh is present
try:
    import trimesh

    def sole_point(link):
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
    sole_w = {f: sole_point(f) for f in ("ankle_left", "ankle_right")}
except Exception as e:  # fall back: ankle origin
    out(f"  (trimesh not available: {e}; using ankle link origin as the foot point)")
    sole_w = {f: Tl0[f][:3, 3] for f in ("ankle_left", "ankle_right")}
sole_local = {f: np.linalg.inv(Tl0[f]) @ np.r_[sole_w[f], 1] for f in sole_w}
c = sole_w["ankle_left"]
out(f"  {'sole centre':16s} [{c[0] * 1000:7.1f} {c[1] * 1000:7.1f} {c[2] * 1000:7.1f}]   "
    f"distance from ankle {np.linalg.norm(c - Tj0[leg_l[-1]][:3, 3]) * 1000:6.1f} mm")


# ---------------------------------------------------------------- 4. Jacobian
def foot_jacobian(q, foot, chain_j):
    Tl, Tj = fk(q)
    p = (Tl[foot] @ sole_local[foot])[:3]
    J = np.zeros((6, len(chain_j)))
    for k, name in enumerate(chain_j):
        z = Tj[name][:3, :3] @ (joints[name]["axis"] / np.linalg.norm(joints[name]["axis"]))
        o = Tj[name][:3, 3]
        J[:3, k] = np.cross(z, p - o)
        J[3:, k] = z
    return J, p


out()
out("=" * 100)
out("4. LEFT FOOT JACOBIAN  v_sole = J(q) q_dot   (linear part, cm per rad; base held fixed)")
out("=" * 100)
meas = {"left_hip_pitch": -8.3, "left_knee": 6.4, "left_ankle": -2.5}  # Isaac Sim finite difference, +0.3 rad
for label, q in (("zero pose", {}), ("standing pose", STANCE)):
    J, p = foot_jacobian(q, "ankle_left", leg_l)
    out(f"-- {label}: sole at [{p[0] * 1000:.1f} {p[1] * 1000:.1f} {p[2] * 1000:.1f}] mm")
    out(f"   {'joint':16s} {'dx/dq':>8s} {'dy/dq':>8s} {'dz/dq':>8s}   {'measured dx (sim, +0.3 rad)':>28s}")
    for k, name in enumerate(leg_l):
        m = f"{meas[name]:+.1f}" if (label == "zero pose" and name in meas) else ""
        out(f"   {name:16s} {J[0, k] * 100:+8.2f} {J[1, k] * 100:+8.2f} {J[2, k] * 100:+8.2f}   {m:>28s}")
    sv = np.linalg.svd(J[:3])[1]
    out(f"   singular values of J_v: {np.round(sv * 100, 2)} cm/rad   manipulability sqrt(det(J J^T)) = "
        f"{math.sqrt(max(np.linalg.det(J[:3] @ J[:3].T), 0)) * 1e6:.2f} cm^3")
out("   (measured values are finite differences over 0.3 rad, so they differ a little from the exact derivative)")

# ---------------------------------------------------------------- 5. statics
total_m = sum(mass_of(l) for l in links)
Tl, _ = fk(STANCE)
com = sum(mass_of(l) * (Tl[l] @ np.r_[com_local(l), 1])[:3] for l in links) / total_m
out()
out("=" * 100)
out(f"5. STATICS IN THE STANDING POSE  (mass {total_m:.3f} kg, weight {total_m * G:.2f} N)")
out("=" * 100)
out(f"centre of mass [{com[0] * 1000:.1f} {com[1] * 1000:.1f} {com[2] * 1000:.1f}] mm (base frame)")
for foot, ch in (("ankle_left", leg_l), ("ankle_right", leg_r)):
    J, p = foot_jacobian(STANCE, foot, ch)
    F = np.array([0, 0, total_m * G / 2, 0, 0, 0])  # ground pushes up with half the weight
    tau = -J.T @ F  # torque the motors must supply to hold the body (sign: reaction on the leg)
    out(f"-- {foot}: ground reaction {F[2]:.2f} N up")
    for k, name in enumerate(ch):
        out(f"   {name:16s} tau = {tau[k]:+.3f} Nm   ({abs(tau[k]) / joints[name]['effort'] * 100:5.1f}% of {joints[name]['effort']:.1f} Nm limit)")

with open(os.path.join(HERE, "kinematics_result.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote kinematics_result.txt")
