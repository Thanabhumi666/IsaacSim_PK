"""Search a standing pose: one fixed-base robot per (hip_pitch, knee, ankle)
combo, same angles on both legs. Scores each pose by
  * centre of mass over the middle of the two soles (x, y),
  * soles flat and at the same height.
Prints the best poses and writes stance_result.txt.

  python find_stance.py --headless
"""

import argparse
import itertools
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--step", type=float, default=0.15)
parser.add_argument("--range", type=float, default=1.05)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import xml.etree.ElementTree as ET  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

from microduck_lab.microduck_cfg import MICRODUCK_CFG, URDF_PATH  # noqa: E402

lines = []


def out(*a):
    s = " ".join(str(x) for x in a)
    print("[STANCE] " + s, flush=True)
    lines.append(s)


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def link_vertices(link_name):
    """All visual-mesh vertices of a link, in the link frame."""
    root = ET.parse(URDF_PATH).getroot()
    link = [l for l in root.findall("link") if l.get("name") == link_name][0]
    pts = []
    for vis in link.findall("visual"):
        mesh_el = vis.find("geometry/mesh")
        m = trimesh.load(mesh_el.get("filename"), force="mesh")
        T = np.eye(4)
        o = vis.find("origin")
        if o is not None:
            T[:3, :3] = rpy_matrix(*[float(v) for v in o.get("rpy", "0 0 0").split()])
            T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
        m.apply_transform(T)
        pts.append(np.asarray(m.vertices))
    return np.vstack(pts)


verts_l = link_vertices("ankle_left")
verts_r = link_vertices("ankle_right")

vals = np.round(np.arange(-args.range, args.range + 1e-9, args.step), 3)
signs = list(itertools.product((1.0, -1.0), repeat=3))  # right = sign * left, per joint (axes may be mirrored)
combos = [tuple(v) + s for s in signs for v in itertools.product(vals, vals, vals)]  # h, k, a, sh, sk, sa
N = len(combos)
out(f"testing {N} poses")


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = MICRODUCK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


scene_cfg = SceneCfg(num_envs=N, env_spacing=0.4)
scene_cfg.robot.spawn.fix_base = True
scene_cfg.robot.init_state.pos = (0.0, 0.0, 0.3)

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
scene = InteractiveScene(scene_cfg)
sim.reset()
robot = scene["robot"]
dev = robot.device

q = robot.data.default_joint_pos.clone()
c = torch.tensor(combos, device=dev, dtype=q.dtype)
for k, jn in enumerate(("hip_pitch", "knee", "ankle")):
    q[:, robot.joint_names.index(f"left_{jn}")] = c[:, k]
    q[:, robot.joint_names.index(f"right_{jn}")] = c[:, k] * c[:, 3 + k]
robot.write_joint_state_to_sim(q, torch.zeros_like(q))
for _ in range(60):
    robot.set_joint_position_target(q)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())

origins = scene.env_origins
masses = robot.root_physx_view.get_masses().to(dev)  # (N, B)
try:
    com_b = robot.data.body_com_pos_w
except AttributeError:
    com_b = robot.data.body_pos_w
com = (com_b * masses[..., None]).sum(1) / masses.sum(1, keepdim=True) - origins

fl = robot.body_names.index("ankle_left")
fr = robot.body_names.index("ankle_right")
i0 = combos.index((0.0, 0.0, 0.0, 1.0, 1.0, 1.0))  # zero pose: soles are flat on the floor there


def sole_local(bi, verts):
    """Sole centre + sole normal in the ankle link frame, measured at the zero pose
    in world coordinates (the link frame itself is not z-up)."""
    p0 = robot.data.body_pos_w[i0, bi] - origins[i0]
    q0 = robot.data.body_quat_w[i0, bi]
    v = torch.tensor(verts, device=dev, dtype=p0.dtype)
    vw = p0 + quat_apply(q0.expand(len(v), 4), v)
    zmin = vw[:, 2].min()
    s = vw[vw[:, 2] < zmin + 0.003]
    lo, hi = s[:, :2].min(0).values, s[:, :2].max(0).values
    centre_w = torch.stack([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, zmin])
    q0inv = q0 * torch.tensor([1.0, -1.0, -1.0, -1.0], device=dev)
    centre_l = quat_apply(q0inv, centre_w - p0)
    normal_l = quat_apply(q0inv, torch.tensor([0.0, 0.0, 1.0], device=dev))
    out(f"sole {robot.body_names[bi]}: size x {float(hi[0] - lo[0]) * 1000:.1f} mm y {float(hi[1] - lo[1]) * 1000:.1f} mm, "
        f"centre (zero pose, rel base) {[round(float(x), 4) for x in centre_w - torch.tensor([0, 0, 0.3], device=dev)]}")
    return centre_l, normal_l


def sole_world(bi, centre_l, normal_l):
    p = robot.data.body_pos_w[:, bi] - origins
    qq = robot.data.body_quat_w[:, bi]
    up = quat_apply(qq, normal_l.expand(N, 3))
    return p + quat_apply(qq, centre_l.expand(N, 3)), up[:, 2]


sl, nl = sole_world(fl, *sole_local(fl, verts_l))
sr, nr = sole_world(fr, *sole_local(fr, verts_r))
mid = (sl + sr) / 2
dx = com[:, 0] - mid[:, 0]
dy = com[:, 1] - mid[:, 1]
height = com[:, 2] - torch.minimum(sl[:, 2], sr[:, 2])
flat = (1 - nl) + (1 - nr)
dz = (sl[:, 2] - sr[:, 2]).abs()
joint_err = (robot.data.joint_pos - q).abs().max(1).values
# hard filter: soles flat (< ~6 deg), same height, pose actually reached
ok = (flat < 0.012) & (dz < 0.003) & (joint_err < 0.05)
out(f"flat & reachable poses: {int(ok.sum())} / {N}")
# among those: CoM over mid-sole first, then prefer standing taller
score = 1000 * (dx.abs() + dy.abs()) - 20 * height
score = torch.where(ok, score, torch.full_like(score, 1e9))

out("single-joint effect (+0.30 rad, same sign both legs): sole L / sole R xyz, up_z L/R")
for name, cmb in (("hip_pitch", (0.3, 0.0, 0.0)), ("knee", (0.0, 0.3, 0.0)), ("ankle", (0.0, 0.0, 0.3))):
    j = min(range(N), key=lambda i: sum(abs(a - b) for a, b in zip(combos[i], cmb + (1.0, 1.0, 1.0))))
    out(f"  {name:9s} L {[round(float(v), 4) for v in sl[j]]} R {[round(float(v), 4) for v in sr[j]]} up {float(nl[j]):.3f}/{float(nr[j]):.3f}")
order = torch.argsort(score)
out("rank  L: hip_pitch knee ankle | R sign h k a | dx(mm) dy(mm) flat dz(mm) comH(mm) jerr")
for r, i in enumerate(order[:12].tolist()):
    h, k, a, sh, sk, sa = combos[i]
    out(
        f"{r:3d}   {h:+.2f} {k:+.2f} {a:+.2f} | {sh:+.0f} {sk:+.0f} {sa:+.0f} | {dx[i] * 1000:6.1f} {dy[i] * 1000:6.1f}"
        f" {flat[i]:.3f} {dz[i] * 1000:5.1f} {height[i] * 1000:6.1f} {joint_err[i]:.3f}"
    )
best = order[0].item()
h, k, a, sh, sk, sa = combos[best]
base_above_sole = 0.3 - float(torch.minimum(sl[best, 2], sr[best, 2]))
out(f"BEST left_hip_pitch={h} left_knee={k} left_ankle={a} right_hip_pitch={h * sh} right_knee={k * sk} right_ankle={a * sa}")
out(f"BEST base_above_sole={base_above_sole:.4f}")
out(f"zero pose: dx {dx[i0] * 1000:.1f} mm dy {dy[i0] * 1000:.1f} mm flat {flat[i0]:.3f}")

with open(os.path.join(HERE, "stance_result.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
sys.stdout.flush()
os._exit(0)
