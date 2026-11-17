"""Flat-ground velocity-tracking walking task for the Micro Duck.

Built on Isaac Lab's LocomotionVelocityRoughEnvCfg (same recipe as the G1/H1
humanoid flat tasks), with rewards for a small biped.
"""

import os

import torch

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    TerminationsCfg,
)

# bodies that must never touch the floor (torso + whole head)
NO_TOUCH_BODIES = ["base_link", "neck", "neck_pitch", "yaw_roll_motion", "jaw_soft", "jaw"]


@configclass
class MicroDuckTerminations(TerminationsCfg):
    # run 3 (bipedfix) learned to lie on its side with the beak on the floor and
    # paddle with its legs: only base_link contact ended an episode. Now any
    # torso/head contact or a tilt over ~35 deg ends it.
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.6})

from .microduck_cfg import BASE_BODY, FEET_BODIES, HEAD_JOINTS, LEG_JOINTS, MICRODUCK_CFG


def feet_air_time_over_max(env, sensor_cfg: SceneEntityCfg, max_air_time: float) -> torch.Tensor:
    """Penalty: seconds each foot has been in the air beyond max_air_time (summed over feet).

    feet_air_time_positive_biped pays whenever exactly one foot is down, so a
    one-legged hop (other foot held up forever) also collects it. The stdfix run
    learned exactly that (right-foot duty factor 0.00). This term makes a foot that
    never comes down increasingly expensive, forcing the legs to alternate.
    """
    cs = env.scene.sensors[sensor_cfg.name]
    t = cs.data.current_air_time[:, sensor_cfg.body_ids]
    return torch.clamp(t - max_air_time, min=0.0, max=2.0).sum(dim=1)


@configclass
class MicroDuckRewards(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # reward = exp(-err^2 / std^2). std must match the robot's speed scale: with the
    # humanoid default std=0.25 m/s and commands <= 0.2 m/s, standing still already
    # scored 0.70 and the first run learned to stand instead of walk.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.1},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODIES),
            "threshold": 0.15,  # small robot -> short steps
        },
    )
    feet_air_too_long = RewTerm(
        func=feet_air_time_over_max,
        weight=-3.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODIES), "max_air_time": 0.4},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODIES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODIES),
        },
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw", ".*_hip_roll"])},
    )
    joint_deviation_head = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HEAD_JOINTS)},
    )


@configclass
class MicroDuckFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: MicroDuckRewards = MicroDuckRewards()
    terminations: MicroDuckTerminations = MicroDuckTerminations()

    def __post_init__(self):
        super().__post_init__()

        # --- scene: robot on a flat plane, no height scan
        self.scene.robot = MICRODUCK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        self.scene.env_spacing = 1.0

        # --- actions: legs + head, jaw stays at default
        self.actions.joint_pos.joint_names = LEG_JOINTS + HEAD_JOINTS
        self.actions.joint_pos.scale = 0.25

        # --- events (domain randomization kept light for a first run)
        self.events.push_robot = None
        self.events.add_base_mass = None
        # default shifts the base CoM by up to +-5 cm (too much for a 22 cm robot) and
        # targets a body called "base" that does not exist here -> disable
        self.events.base_com = None
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [BASE_BODY]
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        }

        # --- terminations
        # torso touching the floor ends the episode (trunk_base is merged into base_link)
        self.terminations.base_contact.params["sensor_cfg"].body_names = NO_TOUCH_BODIES

        # --- rewards tuning
        self.rewards.undesired_contacts = None
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.flat_orientation_l2.weight = -5.0  # keep the torso upright
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINTS)
        self.rewards.dof_torques_l2.weight = -1.0e-3
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINTS)

        # --- commands: small robot, small speeds (m/s, rad/s)
        # feet_air_time_positive_biped only pays when |cmd| > 0.1 m/s -> keep commands above it
        self.commands.base_velocity.ranges.lin_vel_x = (0.1, 0.25)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.rel_standing_envs = 0.1


@configclass
class MicroDuckFlatEnvCfg_PLAY(MicroDuckFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        # close-up camera that follows robot 0 (used by the GUI and by --video)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (0.45, 0.45, 0.25)
        self.viewer.lookat = (0.0, 0.0, 0.08)
        self.events.base_external_force_torque = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # MICRODUCK_LEGACY=1: play the older runs (1-3) with the terminations they were trained with
        # (only a torso contact ends an episode), so their videos show what they really learned.
        if os.environ.get("MICRODUCK_LEGACY") == "1":
            self.terminations.bad_orientation = None
            self.terminations.base_contact.params["sensor_cfg"].body_names = [BASE_BODY]
