# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Snake locomotion environment.
"""

import gymnasium as gym

from . import agents
from .snake_env import SnakeEnv, SnakeEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Snake-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.snake:SnakeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SnakeEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.SnakePPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
