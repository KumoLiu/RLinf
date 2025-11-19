from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject
from isaaclab.utils.math import quat_rotate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def lift_trocars_reward(
    env: ManagerBasedRLEnv, 
    table_height: float = 0.85483,
    lift_threshold: float = 0.05,
    asset_cfg1: SceneEntityCfg = SceneEntityCfg("trocar_1"),
    asset_cfg2: SceneEntityCfg = SceneEntityCfg("trocar_2"),
) -> torch.Tensor:
    """Reward for lifting both trocars above the table.
    
    Reward is 1.0 if both trocars are lifted above (table_height + lift_threshold), else 0.0.
    """
    # Get the rigid objects from the scene
    # The names "trocar_1" and "trocar_2" must match the attributes in the SceneCfg
    obj1: RigidObject = env.scene[asset_cfg1.name]
    obj2: RigidObject = env.scene[asset_cfg2.name]
    
    # Get positions (num_envs, 3)
    pos1 = obj1.data.root_pos_w
    pos2 = obj2.data.root_pos_w
    
    target_z = table_height + lift_threshold
    
    # Check if lifted
    is_lifted_1 = pos1[:, 2] > target_z
    is_lifted_2 = pos2[:, 2] > target_z
    
    return (is_lifted_1 & is_lifted_2).float()


def trocar_insertion_reward(
    env: ManagerBasedRLEnv,
    dist_std: float = 0.1,
    angle_std: float = 0.2,
    angle_threshold: float = 0.2, # Tolerance for parallelism (radians)
    asset_cfg1: SceneEntityCfg = SceneEntityCfg("trocar_1"),
    asset_cfg2: SceneEntityCfg = SceneEntityCfg("trocar_2"),
) -> torch.Tensor:
    """Reward for inserting trocar_2 into trocar_1 with relaxed alignment requirements.
    
    Consists of:
    1. Distance reward: encourage minimizing distance between centers.
    2. Alignment reward: encourage aligning Z-axes, with a flat tolerance threshold.
    """
    obj1: RigidObject = env.scene[asset_cfg1.name]
    obj2: RigidObject = env.scene[asset_cfg2.name]

    # Positions and Rotations
    pos1 = obj1.data.root_pos_w
    quat1 = obj1.data.root_quat_w
    pos2 = obj2.data.root_pos_w
    quat2 = obj2.data.root_quat_w

    # 1. Distance Reward (Gaussian kernel)
    # Encourages centers to be close
    dist = torch.norm(pos1 - pos2, dim=-1)
    dist_reward = torch.exp(-torch.square(dist) / (2 * dist_std**2))
    
    # 2. Alignment Reward
    # Calculate X-axis vectors (based on visual inspection, the long axis is X)
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    
    axis1 = quat_rotate(quat1, x_axis)
    axis2 = quat_rotate(quat2, x_axis)
    
    # Calculate angle between axes
    # We want them to be parallel (angle ~ 0) or anti-parallel (angle ~ pi)
    dot_prod = torch.sum(axis1 * axis2, dim=-1)
    abs_dot = torch.abs(dot_prod)
    # Clamp for numerical stability before acos
    abs_dot = torch.clamp(abs_dot, max=1.0)
    
    angle = torch.acos(abs_dot) # Result is in [0, pi/2] because of abs_dot
    
    # Threshold logic:
    # If angle <= angle_threshold, full reward (1.0)
    # Else, decay based on excess angle
    excess_angle = torch.clamp(angle - angle_threshold, min=0.0)
    align_reward = torch.exp(-torch.square(excess_angle) / (2 * angle_std**2))

    # Combine rewards
    total_reward = dist_reward * align_reward
    
    return total_reward
