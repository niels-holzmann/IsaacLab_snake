# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Snake locomotion environment
"""

import gymnasium as gym

from . import agents, snake_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Snake-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": snake_env_cfg.SnakeEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.SnakePPORunnerCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
