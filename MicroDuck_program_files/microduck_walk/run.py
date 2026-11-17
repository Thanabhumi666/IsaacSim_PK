"""Run Isaac Lab's rsl_rl train/play scripts with the Micro Duck tasks registered.

  python run.py train --task MicroDuck-Flat-v0 --headless --num_envs 4096
  python run.py play  --task MicroDuck-Flat-Play-v0 --num_envs 32
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ISAACLAB = os.environ.get("ISAACLAB_PATH", r"C:\Users\punn_\IsaacLab")

if len(sys.argv) < 2 or sys.argv[1] not in ("train", "play"):
    sys.exit("usage: python run.py train|play [isaac lab args...]")

mode = sys.argv[1]
script = os.path.join(ISAACLAB, "scripts", "reinforcement_learning", "rsl_rl", mode + ".py")

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(script))  # for cli_args.py next to the script

import microduck_lab  # noqa: E402,F401  (registers the gym ids)

sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
