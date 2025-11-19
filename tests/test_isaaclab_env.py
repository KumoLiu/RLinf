import unittest
import torch
from omegaconf import OmegaConf
import sys
import os

# This import assumes that RLinf is in the PYTHONPATH.
# It also requires Isaac Lab and the local tasks to be in the PYTHONPATH.
try:
    from rlinf.envs.isaaclab.isaaclab_env import IsaacLabEnv
except ImportError as e:
    print(f"Failed to import IsaacLabEnv. Please ensure RLinf, Isaac Lab, and your custom task repositories are in your PYTHONPATH. Error: {e}")
    IsaacLabEnv = None


@unittest.skipIf(IsaacLabEnv is None, "IsaacLabEnv could not be imported. Skipping tests.")
class TestIsaacLabEnv(unittest.TestCase):
    def setUp(self):
        """Set up the test case by creating a configuration."""
        print("Setting up IsaacLabEnv test...")
        # This configuration mimics the one defined in your YAML files.
        self.cfg = OmegaConf.create({
            'auto_reset': True,
            'ignore_terminations': False,
            'record_metrics': True,
            'isaaclab': {
                'task_name': 'Isaac-PickPlace-Surgical-G129-Dex3-Joint',
                'num_envs': 1,
                'headless': False,
                'device': 'cuda',
                'task_description': 'test description for pick and place',
                'use_proprio': True,
                'video_cfg': {
                    'save_video': True,
                    'video_base_dir': 'tests/isaaclab_video',
                }
            },
        })
        self.env = None
        print("Configuration created.")

    def test_init_reset_step(self):
        """Tests environment initialization, reset, and a single step."""
        print("Initializing IsaacLabEnv...")
        # We use a try-except block to provide a more informative error message
        # if the Isaac Sim environment fails to launch.
        try:
            self.env = IsaacLabEnv(self.cfg, seed_offset=42, total_num_processes=1)
        except Exception as e:
            self.fail(f"IsaacLabEnv initialization failed with an exception: {e}")

        self.assertIsNotNone(self.env, "Environment object should not be None after initialization.")
        self.assertEqual(self.env.num_envs, self.cfg.isaaclab.num_envs, "num_envs should match the configuration.")
        print("IsaacLabEnv initialized successfully.")

        print("Resetting environment...")
        obs, info = self.env.reset()

        # --- Validate Observation Space ---
        self.assertIn('images', obs, "Observation should contain 'images' key.")
        self.assertIn('wrist_images', obs, "Observation should contain 'wrist_images' key.")
        self.assertIn('task_descriptions', obs, "Observation should contain 'task_descriptions' key.")
        self.assertIn('proprio', obs, "Observation should contain 'proprio' key.")
        # Check image shapes (N, C, H, W)
        self.assertEqual(obs['images'].shape[0], self.cfg.isaaclab.num_envs)
        self.assertEqual(obs['wrist_images'].shape[0], self.cfg.isaaclab.num_envs)
        self.assertEqual(obs['images'].shape[1], 3, "Head camera image should have 3 (RGB) channels.")
        self.assertEqual(obs['wrist_images'].shape[2], 3, "Wrist camera image should have 3 (RGB) channels.")

        # Check proprioception shape (N, D)
        self.assertEqual(obs['proprio'].shape[0], self.cfg.isaaclab.num_envs)
        self.assertGreater(obs['proprio'].shape[1], 0, "Proprioceptive state should have a non-zero dimension.")
        print("Observation space validated.")

        print("Stepping environment with a dummy action...")
        # Create a dummy action tensor with the correct shape and device.
        # The action dimension for the g1_29dof_dex3 robot is 29.
        action_dim = 29
        dummy_action = torch.zeros((self.env.num_envs, action_dim), device=self.env.device)
        next_obs, reward, terminated, truncated, info = self.env.step(dummy_action)
        print('****** info', info)
        print('****** reward', reward)
        print('****** terminated', terminated)
        print('****** truncated', truncated)
        print("Environment stepped successfully.")

        # --- Validate Step Return Values ---
        self.assertIsInstance(next_obs, dict)
        self.assertEqual(reward.shape, (self.env.num_envs,))
        self.assertEqual(terminated.shape, (self.env.num_envs,))
        self.assertEqual(truncated.shape, (self.env.num_envs,))
        self.assertIsInstance(info, dict)
        print("Step return values validated.")

        print("Unit test for IsaacLabEnv passed.")

    # def tearDown(self):
    #     """Clean up after the test by closing the simulation app."""
    #     if hasattr(self, 'env') and self.env and hasattr(self.env, 'simulation_app'):
    #         if self.env.simulation_app.is_running():
    #             print("Closing Isaac Sim application...")
    #             self.env.simulation_app.close()
    #             print("Isaac Sim application closed.")

if __name__ == '__main__':
    unittest.main()
