# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0      
"""
public base scene configuration module
provides reusable scene element configurations, such as tables, objects, ground, lights, etc.
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import  AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaacsim.core.utils.torch.rotations import euler_angles_to_quats
from ..common_config import   CameraBaseCfg  # isort: skip
import os

project_root = os.environ.get("PROJECT_ROOT")
usd_root = "/yunl/Data"
# usd_root = "/mnt/hdd/Data"
# usd_root = "/home/nvidia/workspace/yunl/assets"

@configclass
class SurgicalSceneCfg(InteractiveSceneCfg): # inherit from the interactive scene configuration class
    """object table scene configuration class
    defines a complete scene containing robot, object, table, etc.
    """
      # 1. room wall configuration - simplified configuration to avoid rigid body property conflicts
    scene = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Scene",
        spawn=UsdFileCfg(
            usd_path=f"{usd_root}/lw_v1/surgery-room-dev-internal/assets/Assets/scene.usd",  # use simple room model
        ),
    )
    
    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",   # light in the scene
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), # light color (white)
                                     intensity=1000.0),    # light intensity
    )
