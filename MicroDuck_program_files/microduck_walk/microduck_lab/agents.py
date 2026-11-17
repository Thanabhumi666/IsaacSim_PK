"""PPO settings: reuse the installed Isaac Lab G1 flat config so it always
matches the installed isaaclab_rl / rsl-rl versions."""

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1FlatPPORunnerCfg,
)


@configclass
class MicroDuckFlatPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:
            parent()
        self.experiment_name = "microduck_flat"
        self.max_iterations = 3000
        self.save_interval = 100
