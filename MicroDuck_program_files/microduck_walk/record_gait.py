"""Run a trained Micro Duck policy at a fixed walking command and record the gait
for kinematic analysis (report chapter 4).

  python record_gait.py --headless --vx 0.15 --seconds 10
  python record_gait.py --headless --checkpoint logs\\rsl_rl\\microduck_flat\\<run>\\model_3000.pt

Writes gait_<vx>.npz and gait_<vx>.csv with, every control step (0.02 s):
  t, joint pos / vel / applied torque (all joints),
  foot (ankle_left/right) position relative to its hip, in the base frame,
  foot vertical contact force, base linear velocity (base frame), base height.
"""

import argparse
import glob
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None, help="model_*.pt; default: newest under logs/rsl_rl/microduck_flat")
parser.add_argument("--vx", type=float, default=0.15, help="forward speed command (m/s)")
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--warmup", type=float, default=2.0, help="seconds skipped before recording")
parser.add_argument("--tag", default="", help="prefix for the output name, e.g. run4 -> gait_run4_0p20.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import microduck_lab  # noqa: E402,F401

rot_inv = getattr(math_utils, "quat_apply_inverse", None) or math_utils.quat_rotate_inverse

TASK = "MicroDuck-Flat-Play-v0"
ckpt = args.checkpoint
if ckpt is None:
    cands = glob.glob(os.path.join(HERE, "logs", "rsl_rl", "microduck_flat", "*", "model_*.pt"))
    if not cands:
        sys.exit("no checkpoint found")
    ckpt = max(cands, key=os.path.getmtime)
print("[GAIT] checkpoint:", ckpt, flush=True)

env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
cmd = env_cfg.commands.base_velocity
cmd.ranges.lin_vel_x = (args.vx, args.vx)
cmd.ranges.lin_vel_y = (0.0, 0.0)
cmd.ranges.ang_vel_z = (0.0, 0.0)
cmd.ranges.heading = (0.0, 0.0)
cmd.heading_command = False
cmd.rel_standing_envs = 0.0
cmd.resampling_time_range = (1e6, 1e6)
env_cfg.episode_length_s = args.warmup + args.seconds + 5.0
env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}

agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg))
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(ckpt)
policy = runner.get_inference_policy(device=env.unwrapped.device)

scene = env.unwrapped.scene
robot = scene["robot"]
contact = scene["contact_forces"]
dt = env.unwrapped.step_dt
bi = {n: robot.body_names.index(n) for n in ("ankle_left", "ankle_right", "hip_left", "hip_right")}
ci = {n: contact.body_names.index(n) for n in ("ankle_left", "ankle_right")}

rec = {k: [] for k in ("t", "q", "qd", "tau", "foot_l", "foot_r", "fz_l", "fz_r", "v_base", "z_base")}
obs = env.get_observations()
if isinstance(obs, tuple):  # older rsl_rl wrappers return (obs, extras)
    obs = obs[0]
n_warm = int(args.warmup / dt)
n_rec = int(args.seconds / dt)
with torch.inference_mode():
    for i in range(n_warm + n_rec):
        obs, _, _, _ = env.step(policy(obs))
        if i < n_warm:
            continue
        qw = robot.data.root_quat_w[0:1]
        pw = robot.data.body_pos_w[0]
        for side in ("l", "r"):
            foot, hip = ("ankle_left", "hip_left") if side == "l" else ("ankle_right", "hip_right")
            rel = (pw[bi[foot]] - pw[bi[hip]]).unsqueeze(0)
            rec[f"foot_{side}"].append(rot_inv(qw, rel)[0].cpu().numpy())
        f = contact.data.net_forces_w[0]
        rec["fz_l"].append(float(f[ci["ankle_left"], 2]))
        rec["fz_r"].append(float(f[ci["ankle_right"], 2]))
        rec["t"].append((i - n_warm) * dt)
        rec["q"].append(robot.data.joint_pos[0].cpu().numpy())
        rec["qd"].append(robot.data.joint_vel[0].cpu().numpy())
        rec["tau"].append(robot.data.applied_torque[0].cpu().numpy())
        rec["v_base"].append(robot.data.root_lin_vel_b[0].cpu().numpy())
        rec["z_base"].append(float(robot.data.root_pos_w[0, 2]))

data = {k: np.asarray(v) for k, v in rec.items()}
tag = f"{args.vx:.2f}".replace(".", "p")
if args.tag:
    tag = f"{args.tag}_{tag}"
out_npz = os.path.join(HERE, f"gait_{tag}.npz")
np.savez(out_npz, joint_names=np.array(robot.joint_names), dt=dt, vx_cmd=args.vx, **data)

cols = ["t"] + [f"q_{n}" for n in robot.joint_names] + [f"qd_{n}" for n in robot.joint_names] \
    + [f"tau_{n}" for n in robot.joint_names] \
    + ["footL_x", "footL_y", "footL_z", "footR_x", "footR_y", "footR_z", "fz_L", "fz_R", "vx", "vy", "vz", "z_base"]
table = np.column_stack([data["t"], data["q"], data["qd"], data["tau"], data["foot_l"], data["foot_r"],
                         data["fz_l"], data["fz_r"], data["v_base"], data["z_base"]])
np.savetxt(out_npz.replace(".npz", ".csv"), table, delimiter=",", header=",".join(cols), comments="", fmt="%.6f")

vx_mean = float(data["v_base"][:, 0].mean())
print(f"[GAIT] recorded {len(data['t'])} steps ({args.seconds:.1f} s), commanded vx {args.vx:.3f}, mean vx {vx_mean:.3f} m/s", flush=True)
print("[GAIT] wrote", out_npz, flush=True)
sys.stdout.flush()
os._exit(0)
