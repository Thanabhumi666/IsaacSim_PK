"""Articulation config for the Micro Duck, spawned straight from its URDF.

Values marked TUNE are first guesses; run inspect_robot.py and adjust.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# microduck_fixed.urdf is made by fix_urdf.py (real masses/inertia + collisions);
# the original export has 0.01 kg / 1e-5 placeholders on every link.
URDF_PATH = os.environ.get(
    "MICRODUCK_URDF",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "microduck_fixed.urdf"),
)

# Names taken from microduck.urdf
LEG_JOINTS = [".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", ".*_ankle"]
HEAD_JOINTS = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
BASE_BODY = "base_link"  # trunk_base gets merged into base_link by the importer
FEET_BODIES = "ankle_.*"  # ankle_left / ankle_right carry the foot + sole meshes

BASE_HEIGHT = 0.008  # base_link origin sits near sole level; small drop at reset

# Standing pose from find_stance.py: CoM 1.2 mm from mid-sole, soles flat.
# NOTE: right-leg joint axes are mirrored w.r.t. the left leg.
STANCE = {
    "left_hip_pitch": -0.3,
    "left_knee": 0.3,
    "left_ankle": 0.6,
    "right_hip_pitch": 0.3,
    "right_knee": -0.3,
    "right_ankle": -0.6,
}
ALL_JOINTS = [
    f"{s}_{j}" for s in ("left", "right") for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
] + HEAD_JOINTS + ["jaw"]

MICRODUCK_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=URDF_PATH,
        fix_base=False,
        merge_fixed_joints=True,
        # microduck_fixed.urdf already has <collision> tags (copied from the visuals)
        collision_from_visuals=False,
        collider_type="convex_hull",
        make_instanceable=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, BASE_HEIGHT),
        joint_pos={n: STANCE.get(n, 0.0) for n in ALL_JOINTS},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # URDF says effort 1.0 Nm, velocity 10 rad/s
        "legs": ImplicitActuatorCfg(
            joint_names_expr=LEG_JOINTS,
            effort_limit_sim=1.0,
            velocity_limit_sim=10.0,
            stiffness=4.0,  # TUNE
            damping=0.2,  # TUNE
            armature=0.005,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=HEAD_JOINTS + ["jaw"],
            effort_limit_sim=1.0,
            velocity_limit_sim=10.0,
            stiffness=2.0,
            damping=0.1,
            armature=0.005,
        ),
    },
)
