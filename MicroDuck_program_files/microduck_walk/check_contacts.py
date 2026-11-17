"""Why does the right foot's contact sensor read 0?

Builds the play env (1 robot), lets it stand at the default pose, then prints:
  * contact-sensor vertical force on every body (should add up to the weight),
  * which collision prims sit under each rigid body in the USD,
  * world z of each body's lowest collider point (who is really touching the floor).
Writes contacts_result.txt.

  python check_contacts.py --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import microduck_lab  # noqa: E402,F401

lines = []


def out(*a):
    s = " ".join(str(x) for x in a)
    print("[CONTACT] " + s, flush=True)
    lines.append(s)


TASK = "MicroDuck-Flat-Play-v0"
env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
env_cfg.observations.policy.enable_corruption = False
env = gym.make(TASK, cfg=env_cfg)
env.reset()
uw = env.unwrapped
robot = uw.scene["robot"]
cs = uw.scene["contact_forces"]
act = torch.zeros(uw.num_envs, uw.action_manager.total_action_dim, device=uw.device)
for _ in range(50):  # zero action = hold default pose
    env.step(act)

masses = robot.root_physx_view.get_masses()[0]
weight = float(masses.sum()) * 9.81
out(f"robot weight {weight:.2f} N ({float(masses.sum()):.3f} kg)")
out("sensor bodies:", cs.body_names)
fz = cs.data.net_forces_w[0, :, 2]
total = 0.0
for n, f in zip(cs.body_names, fz.tolist()):
    if abs(f) > 1e-3:
        out(f"  contact Fz {n:18s} {f:7.3f} N")
    total += f
out(f"sum of sensed Fz {total:.3f} N vs weight {weight:.3f} N")

stage = omni.usd.get_context().get_stage()
root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
out("collision prims per rigid body (path relative to Robot), lowest z:")
for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        continue
    cols = []
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if p != prim and p.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        if p.HasAPI(UsdPhysics.CollisionAPI):
            cols.append(p)
    zmin = None
    for p in cols:
        r = bbox.ComputeWorldBound(p).ComputeAlignedRange()
        if not r.IsEmpty():
            zmin = r.GetMin()[2] if zmin is None else min(zmin, r.GetMin()[2])
    rel = str(prim.GetPath()).replace(str(root.GetPath()), "")
    zs = "n/a" if zmin is None else f"{zmin:.4f}"
    out(f"  {rel:40s} colliders {len(cols):2d}  lowest z {zs}  contactReport {prim.HasAPI('PhysxContactReportAPI')}")

with open(os.path.join(HERE, "contacts_result.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
sys.stdout.flush()
os._exit(0)
