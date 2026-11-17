"""Spawn one Micro Duck, print its bodies/joints/masses, drop it and report
standing height. Writes the same report to inspect_result.txt.

  python inspect_robot.py --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=3.0)
parser.add_argument("--z", type=float, default=0.5, help="spawn height of the base")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from microduck_lab.microduck_cfg import BASE_BODY, MICRODUCK_CFG  # noqa: E402

lines = []


def out(*a):
    s = " ".join(str(x) for x in a)
    print("[DUCK] " + s, flush=True)
    lines.append(s)


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2000.0))

cfg = MICRODUCK_CFG.replace(prim_path="/World/Robot")
cfg.init_state.pos = (0.0, 0.0, args.z)
robot = Articulation(cfg)
sim.reset()
# start exactly in the configured default pose (the USD itself is at q = 0)
robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))
robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.default_root_state[:, 7:]))
sim.step(render=False)
robot.update(sim.get_physics_dt())

out("bodies:", robot.body_names)
out("joints:", robot.joint_names)
masses = robot.root_physx_view.get_masses()[0]
out("masses:", [round(float(m), 4) for m in masses], "total", round(float(masses.sum()), 4))
lim = robot.data.joint_pos_limits[0]
for n, (lo, hi) in zip(robot.joint_names, lim.tolist()):
    out(f"  limit {n}: {lo:.3f} .. {hi:.3f}")

try:
    com0 = robot.data.body_com_pos_w[0]
except AttributeError:
    com0 = robot.data.body_pos_w[0]
m0 = robot.root_physx_view.get_masses()[0].to(com0.device)
c0 = (com0 * m0[:, None]).sum(0) / m0.sum()
out(f"start CoM xyz: {[round(float(v), 4) for v in c0]}")
out("start feet xyz:", [[round(float(v), 4) for v in robot.data.body_pos_w[0, robot.body_names.index(n)]] for n in ("ankle_left", "ankle_right")])
z = robot.data.body_pos_w[0, :, 2]
bi = robot.body_names.index(BASE_BODY) if BASE_BODY in robot.body_names else 0
lowest = int(torch.argmin(z))
out(f"default pose: base z {float(z[bi]):.4f}, lowest body {robot.body_names[lowest]} z {float(z[lowest]):.4f}")
out(f"base above lowest body origin: {float(z[bi] - z[lowest]):.4f} m")
out("body z rel base:", {n: round(float(v - z[bi]), 4) for n, v in zip(robot.body_names, z.tolist())})
try:
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    for name in ("base_link", "ankle_left", "ankle_right"):
        prim = stage.GetPrimAtPath(f"/World/Robot/{name}")
        if prim.IsValid():
            r = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
            out(f"bbox {name}: zmin {r.GetMin()[2] - float(z[bi]):.4f} zmax {r.GetMax()[2] - float(z[bi]):.4f} (rel base)")
except Exception as e:  # bbox is only a helper
    out("bbox failed:", e)

steps = int(args.seconds / sim.get_physics_dt())
target = robot.data.default_joint_pos.clone()
for i in range(steps):
    robot.set_joint_position_target(target)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.get_physics_dt())
    if i % 100 == 0:
        q = robot.data.root_quat_w[0]
        out(f"t={i * sim.get_physics_dt():.2f}s base z={float(robot.data.root_pos_w[0, 2]):.4f} quat={[round(float(v), 3) for v in q]}")

out("final joint pos:", {n: round(float(v), 3) for n, v in zip(robot.joint_names, robot.data.joint_pos[0].tolist())})
try:
    com = robot.data.body_com_pos_w[0]
except AttributeError:
    com = robot.data.body_pos_w[0]
m = robot.root_physx_view.get_masses()[0].to(com.device)
c = (com * m[:, None]).sum(0) / m.sum()
out(f"final CoM xyz: {[round(float(v), 4) for v in c]}  base xyz: {[round(float(v), 4) for v in robot.data.root_pos_w[0]]}")
fi = [robot.body_names.index(n) for n in ("ankle_left", "ankle_right")]
out("feet xyz:", [[round(float(v), 4) for v in robot.data.body_pos_w[0, i]] for i in fi])
try:
    inert = robot.root_physx_view.get_inertias()[0]
    out("inertia diag base/ankle_l:", [round(float(v), 6) for v in inert[0][[0, 4, 8]]], [round(float(v), 6) for v in inert[fi[0]][[0, 4, 8]]])
except Exception as e:
    out("inertia read failed:", e)
zf = robot.data.body_pos_w[0, :, 2]
out(f"settled: base z {float(robot.data.root_pos_w[0, 2]):.4f}, min body z {float(zf.min()):.4f}")
out("bodies near ground (<2cm):", [n for n, v in zip(robot.body_names, zf.tolist()) if v < float(zf.min()) + 0.02])

with open(os.path.join(HERE, "inspect_result.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
out("wrote inspect_result.txt")
# app.close() can hang on Windows; results are already on disk, so hard-exit.
sys.stdout.flush()
os._exit(0)
