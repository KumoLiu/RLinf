# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
import torch
from dataclasses import MISSING

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import mdp

from ..common_config import G1RobotPresets, CameraPresets  # isort: skip
from ..common_event.event_manager import SimpleEvent, SimpleEventManager
from ..common_scene.base_scene_pickplace_surgical import SurgicalSceneCfg

##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(SurgicalSceneCfg):
    """object table scene configuration class
    
    inherits from G1SingleObjectSceneCfg, gets the complete G1 robot scene configuration
    can add task-specific scene elements or override default configurations here
    """
    
    # Humanoid robot w/ arms higher
    # humanoid robot configuration 
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_base_fix(init_pos=(-1.91882, 1.94, 0.81168), init_rot=(1.0, 0, 0, 0.0))
    # add camera configuration 
    front_camera = CameraPresets.g1_front_camera()
    left_wrist_camera = CameraPresets.left_dex3_wrist_camera()
    right_wrist_camera = CameraPresets.right_dex3_wrist_camera()

##
# MDP settings
##
@configclass
class ActionsCfg:
    """defines the action configuration related to robot control, using direct joint angle control
    """
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)



@configclass
class ObservationsCfg:
    """defines all available observation information
    """
    @configclass
    class PolicyCfg(ObsGroup):
        """policy group observation configuration class
        defines all state observation values for policy decision
        inherit from ObsGroup base class 
        """

        # 1. robot joint state observation
        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states, params={"enable_dds": False})
        # 2. gripper joint state observation 
        robot_gipper_state = ObsTerm(func=mdp.get_robot_dex3_joint_states, params={"enable_dds": False})

        def __post_init__(self):
            """post initialization function
            set the basic attributes of the observation group
            """
            self.enable_corruption = False  # disable observation value corruption
            self.concatenate_terms = False  # disable observation item connection

    @configclass
    class CameraImagesCfg(ObsGroup):
        """Observations from the robot's cameras."""
        front_camera = ObsTerm(
            func=base_mdp.image, params={"sensor_cfg": SceneEntityCfg("front_camera"), "data_type": "rgb"}
        )
        left_wrist_camera = ObsTerm(
            func=base_mdp.image, params={"sensor_cfg": SceneEntityCfg("left_wrist_camera"), "data_type": "rgb"}
        )
        right_wrist_camera = ObsTerm(
            func=base_mdp.image, params={"sensor_cfg": SceneEntityCfg("right_wrist_camera"), "data_type": "rgb"}
        )

        def __post_init__(self):
            self.concatenate_terms = False

    # observation groups
    # create policy observation group instance
    policy: PolicyCfg = PolicyCfg()
    camera_images: CameraImagesCfg = CameraImagesCfg()

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class RewardsCfg:
    lift_trocars = RewTerm(
        func=mdp.lift_trocars_reward,
        weight=2.0,
        params={
            "table_height": 0.85483,
            "lift_threshold": 0.05,
            "asset_cfg1": SceneEntityCfg("trocar_1"),
            "asset_cfg2": SceneEntityCfg("trocar_2"),
        }
    )
    
    insert_trocars = RewTerm(
        func=mdp.trocar_insertion_reward,
        weight=5.0,
        params={
            "dist_std": 0.1,
            "angle_std": 0.2,
            "angle_threshold": 0.15, # ~8.6 degrees tolerance
            "asset_cfg1": SceneEntityCfg("trocar_1"),
            "asset_cfg2": SceneEntityCfg("trocar_2"),
        }
    )

@configclass
class EventCfg:
    pass
    # reset_scene = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")
    # reset_object = EventTermCfg(
    #     func=mdp.reset_root_state_uniform,  # use uniform distribution reset function
    #     mode="reset",   # set event mode to reset
    #     params={
    #         # position range parameter
    #         "pose_range": {
    #             "x": [-0.05, 0.05],  # x axis position range: -0.05 to 0.0 meter
    #             "y": [-0.05, 0.05],   # y axis position range: 0.0 to 0.05 meter
    #         },
    #         # speed range parameter (empty dictionary means using default value)
    #         "velocity_range": {},
    #         # specify
    #         "asset_cfg": SceneEntityCfg("object"),
    #     },
    # )


@configclass
class PickPlaceG129DEX3JointEnvCfg(ManagerBasedRLEnvCfg):
    """uNITREE G1 robot pick place environment configuration class
    inherits from ManagerBasedRLEnvCfg, defines all configuration parameters for the entire environment
    """

    # scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True
    )
    # viewer settings
    viewer: ViewerCfg = ViewerCfg(
        eye=(-1.9, 1.90, 1.20101),
        lookat=(-1.2, 1.9, 0.6),
        cam_prim_path="/OmniverseKit_Persp",
    )
    # basic settings
    observations: ObservationsCfg = ObservationsCfg()   # observation configuration
    actions: ActionsCfg = ActionsCfg()                  # action configuration
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()    # termination configuration
    events = EventCfg()                                  # event configuration
    commands = None # command manager
    rewards: RewardsCfg = RewardsCfg()  # reward manager
    curriculum = None # curriculum manager

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1/200
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01

        self.sim.render.enable_translucency = True
        self.sim.render.carb_settings = {
            "rtx.raytracing.fractionalCutoutOpacity": True,
        }

        # create event manager
        self.event_manager = SimpleEventManager()

        self.event_manager.register("reset_all_self", SimpleEvent(
            func=lambda env: base_mdp.reset_scene_to_default(
                env,
                torch.arange(env.num_envs, device=env.device))
        ))

