#!/usr/bin/env bash
# Install the extra Python dependencies robosuite needs to import and run in
# this robust_gymnasium fork (headless, no GUI). Run inside the target env
# (e.g. the server's mujoko conda env):
#
#     bash examples/robust_attack/install_robosuite_deps.sh
#
# These were the concrete import blockers hit when building robosuite behind the
# robust_attack interface:
#   - cv2        -> opencv-python-headless   (utils/opencv_renderer.py)
#   - termcolor  -> termcolor                (utils/log_utils.py)
#   - h5py       -> h5py                      (wrappers/demo_sampler_wrapper.py)
# Pillow is pulled in for image utilities. mujoco is assumed already installed.
set -e

PY=${PY:-python}
echo "using interpreter: $($PY -c 'import sys; print(sys.executable)')"

"$PY" -m pip install \
    opencv-python-headless \
    termcolor \
    h5py \
    Pillow

echo "--- verifying robosuite import ---"
"$PY" - <<'PYEOF'
import robust_gymnasium.envs.robosuite as suite
from robust_gymnasium.envs.robosuite.environments.base import REGISTERED_ENVS
from robust_gymnasium.envs.robosuite.wrappers.gym_wrapper import GymWrapper
from robust_gymnasium.envs.robosuite.controllers import load_controller_config
print("robosuite import OK; #envs =", len(REGISTERED_ENVS))
PYEOF
echo "robosuite deps installed and import verified."
