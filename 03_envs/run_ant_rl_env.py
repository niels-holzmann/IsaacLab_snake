# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to run the RL environment for the ant task."""

"""Launch Isaac Sim Simulator first."""

import argparse
import time

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the ant RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv

from omni.isaac.lab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg


def main():
    """Main function."""
    # create environment configuration
    env_cfg = AntEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    start_time = time.time()

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                start_time = time.time()  # reset time on environment reset
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            #joint_efforts = torch.randn_like(env.action_manager.action)
            
            # calculate elapsed time
            elapsed_time = time.time() - start_time
            # create sinusoidal joint efforts
            frequency = 1.0  # frequency of the sine wave
            amplitude = 1.0  # amplitude of the sine wave
            joint_efforts = amplitude * torch.sin(frequency * torch.tensor(elapsed_time) * torch.ones_like(env.action_manager.action))
            
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
