# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab_assets.snake import SNAKE_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.direct.locomotion.locomotion_env_snake import SnakeLocomotionEnv


@configclass
class SnakeEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 30.0
    decimation = 2
    action_scale = 10
    num_actions = 10
    num_observations = 42
    num_states = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        #terrain_type="plane",
        terrain_type="usd",
        usd_path="/home/nholzmann/Documents/rough_plane.usd",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    '''
    #sphere
    sphere = AssetBaseCfg(
        prim_path="/World/sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.5,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
        ),
    )
    '''
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True)

    # robot
    prim_path = "/World/envs/env_.*/Robot"
    print("Prim path: ", prim_path)
    robot: ArticulationCfg = SNAKE_CFG.replace(prim_path=prim_path)
    joint_gears: list = [15, 15, 15, 15, 15, 15, 15, 15, 15, 15]

    heading_weight: float = 3.0
    up_weight: float = 0.0

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.005
    alive_reward_scale: float = 1.0
    dof_vel_scale: float = 0.2

    death_cost: float = -2.0
    termination_height: float = 1.2

    angular_velocity_scale: float = 1.0
    contact_force_scale: float = 0.1


class SnakeEnv(SnakeLocomotionEnv):
    cfg: SnakeEnvCfg

    def __init__(self, cfg: SnakeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
