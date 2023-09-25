from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.mobile_franka import MobileFranka
from omniisaacgymenvs.robots.articulations.views.mobile_franka_view import MobileFrankaView

from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.rotations import get_euler_xyz
from omni.isaac.core.prims import GeometryPrimView

import numpy as np
import torch
import math

from pxr import UsdGeom


class MobileFrankaTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.start_position_noise = self._task_cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self._task_cfg["env"]["startRotationNoise"]
        self.num_props = 1 #self._task_cfg["env"]["numProps"]

        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self._task_cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self._task_cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self._task_cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.finger_close_reward_scale = self._task_cfg["env"]["fingerCloseRewardScale"]

        self.distX_offset = 0.04
        #self.dt = 1/60.
        # these values depend on the task and how we interface with the real robot
        control_frequency = 120.0 / self._task_cfg["env"]["controlFrequencyInv"] # 30
        self.dt = 1/control_frequency

        self._num_observations = 32 - 3 - 2 #23
        self._num_actions = 11
        self._num_agents = 1

        self.initial_target_pos = np.array([2.0, 0.0, 0.5])

        # set the ranges for the target randomization
        self.x_lim = [-3, 3]
        self.y_lim = [-3, 3]
        self.z_lim = [0.2, 1.2]

        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:

        self.get_franka()
        target_cube = VisualCuboid(
            prim_path=self.default_zero_env_path + "/target_cube",
            translation=self.initial_target_pos,
            scale=np.array([0.1, 0.1, 0.1]),
            color=np.array([1, 0, 0]),
        )
        
        super().set_up_scene(scene, replicate_physics=False)

        self._mobilefrankas = MobileFrankaView(prim_paths_expr="/World/envs/.*/mobile_franka", name="franka_view")

        scene.add(self._mobilefrankas)
        scene.add(self._mobilefrankas._hands)
        scene.add(self._mobilefrankas._lfingers)
        scene.add(self._mobilefrankas._rfingers)
        scene.add(self._mobilefrankas._base)
        
        self._targets = GeometryPrimView(prim_paths_expr="/World/envs/.*/target_cube", name="target_view")
        scene.add(self._targets)

        self.init_data()
        return

    def get_franka(self):
        mobile_franka = MobileFranka(prim_path=self.default_zero_env_path + "/mobile_franka", name="mobile_franka")
        self._sim_config.apply_articulation_settings("mobile_franka", get_prim_at_path(mobile_franka.prim_path), self._sim_config.parse_actor_config("mobile_franka"))     

    def init_data(self) -> None:
        def get_env_local_pose(env_pos, xformable, device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()
            
            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device, dtype=torch.float)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(self._env_pos[0], UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/mobile_franka/panda_link7")), self._device)
        lfinger_pose = get_env_local_pose(
            self._env_pos[0], UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/mobile_franka/panda_leftfinger")), self._device
        )
        rfinger_pose = get_env_local_pose(
            self._env_pos[0], UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/mobile_franka/panda_rightfinger")), self._device
        )

        finger_pose = torch.zeros(7, device=self._device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = (tf_inverse(hand_pose[3:7], hand_pose[0:3]))

        grasp_pose_axis = 1
        franka_local_grasp_pose_rot, franka_local_pose_pos = tf_combine(hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3])
        franka_local_pose_pos += torch.tensor([0, 0.04, 0], device=self._device)
        self.franka_local_grasp_pos = franka_local_pose_pos.repeat((self._num_envs, 1))
        self.franka_local_grasp_rot = franka_local_grasp_pose_rot.repeat((self._num_envs, 1))

        # self.franka_default_dof_pos = torch.tensor(
        #     [1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self._device
        # )
        #[0.00017897569768976496, -0.7856239589326517, 1.8711715534358575e-05, -2.3559680300009447, -5.8659626880341875e-06, 1.5717616294461347, 0.7853945309207777]
        self.mobile_franka_default_dof_pos = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, -0.7856, 0.0, -2.356, 0.0, 1.572, 0.7854, 0.035, 0.035], device=self._device
        )

        self.target_positions = torch.zeros((self._num_envs, 3), device=self._device)

        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

    def get_observations(self) -> dict:
        hand_pos, hand_rot = self._mobilefrankas._hands.get_world_poses(clone=False)
        hand_pos = hand_pos - self._env_pos
        franka_dof_pos = self._mobilefrankas.get_joint_positions(clone=False)
        franka_dof_vel = self._mobilefrankas.get_joint_velocities(clone=False)
        self.franka_dof_pos = franka_dof_pos

        base_pos, base_rot = self._mobilefrankas._base.get_world_poses(clone=False)
        base_pos = base_pos - self._env_pos 
        base_pos_xy = base_pos[:, :2]

        # yaw is in range 0-2pi do I want it to be -pi to pi
        roll, pitch, base_yaw = get_euler_xyz(base_rot)
        base_yaw = base_yaw.unsqueeze(1)
        
        base_vel = self._mobilefrankas._base.get_velocities(clone=False)
        base_vel_xy = base_vel[:, :2]
        base_angvel_z = base_vel[:, -1].unsqueeze(1)

        self.franka_lfinger_pos, self.franka_lfinger_rot = self._mobilefrankas._lfingers.get_world_poses(clone=False)
        self.franka_lfinger_pos = self.franka_lfinger_pos - self._env_pos
        
        # panda arm joint positions scaled
        arm_dof_pos_scaled = (
            2.0
            * (franka_dof_pos[:, 3:] - self.franka_dof_lower_limits[3:])
            / (self.franka_dof_upper_limits[3:] - self.franka_dof_lower_limits[3:])
            - 1.0
        )

        self.to_target = self.target_positions - self.franka_lfinger_pos

        obs = torch.hstack((
            base_pos_xy, 
            base_yaw, 
            arm_dof_pos_scaled,
            #base_vel_xy, 
            #base_angvel_z, 
            franka_dof_vel[:, 3:] * self.dof_vel_scale,
            self.franka_lfinger_pos,
            self.target_positions
        )).to(dtype=torch.float32)

        self.obs_buf = obs

        observations = {
            self._mobilefrankas.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        raw_actions = actions.clone().to(self._device)
        
        base_actions = raw_actions[:, :2]
        arm_actions = raw_actions[:, 2:]
        

        combined_actions = torch.hstack((
            base_actions[:,0].unsqueeze(1),
            torch.zeros((base_actions.shape[0], 1), device=self._device),
            base_actions[:,1].unsqueeze(1),
            arm_actions
        ))

        self.actions = combined_actions
        
        targets = self.franka_dof_targets + self.franka_dof_speed_scales * self.dt * combined_actions * self.action_scale # * 0.1
        self.franka_dof_targets[:] = torch.clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(self._mobilefrankas.count, dtype=torch.int32, device=self._device)

        action_x = combined_actions[:, 0] * 1.0 # * 0.5
        action_y = torch.zeros(self._mobilefrankas.count, device=self._device)
        action_yaw = combined_actions[:, 2] * 0.75 # * 0.5

        vel_targets = self._calculate_velocity_targets(action_x, action_y, action_yaw)

        # set the position targets for base joints to the current position
        self.franka_dof_targets[:, :3] = self.franka_dof_pos[:, :3]
        artic_vel_targets = torch.zeros_like(self.franka_dof_targets)
        artic_vel_targets[:, :3] = vel_targets
        self._mobilefrankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)
        self._mobilefrankas.set_joint_velocity_targets(artic_vel_targets)
    
    def _calculate_velocity_targets(self, action_x, action_y, action_yaw):
        current_yaw = self.franka_dof_pos[:, 2]
        new_yaw = current_yaw# + action_yaw
        new_x = torch.cos(new_yaw) * action_x - torch.sin(new_yaw) * action_y
        new_y = torch.sin(new_yaw) * action_x - torch.cos(new_yaw) * action_y
        
        return torch.transpose(torch.stack([new_x, new_y, action_yaw]), 0, 1)

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # reset franka
        pos = torch.clamp(
            self.mobile_franka_default_dof_pos.unsqueeze(0)
            + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self._device) - 0.5),
            self.franka_dof_lower_limits,
            self.franka_dof_upper_limits,
        )
        # randomize the yaw from 0 to 360 in degrees
        pos[:, 2] = torch.rand((len(env_ids),), device=self._device) * 359.0
        dof_pos = torch.zeros((num_indices, self._mobilefrankas.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._mobilefrankas.num_dof), device=self._device)
        dof_pos[:, :] = pos
        self.franka_dof_targets[env_ids, :] = pos
        self.franka_dof_pos[env_ids, :] = pos

        self._mobilefrankas.set_joint_position_targets(self.franka_dof_targets[env_ids], indices=indices)
        self._mobilefrankas.set_joint_positions(dof_pos, indices=indices)
        self._mobilefrankas.set_joint_velocities(dof_vel, indices=indices)

        rands = torch.rand((num_indices, 3), device=self._device)
        
        # modify rands to be in the range of the limits
        rands[:, 0] = rands[:, 0] * (self.x_lim[1] - self.x_lim[0]) + self.x_lim[0]
        rands[:, 1] = rands[:, 1] * (self.y_lim[1] - self.y_lim[0]) + self.y_lim[0]
        rands[:, 2] = rands[:, 2] * (self.z_lim[1] - self.z_lim[0]) + self.z_lim[0]

        self.target_positions[env_ids] = rands
        self._targets.set_world_poses(self._env_pos + self.target_positions)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        """setup initial values for dof related things. This is run only once when the environment is initialized."""
        self.num_franka_dofs = self._mobilefrankas.num_dof
        self.franka_dof_pos = torch.zeros((self.num_envs, self.num_franka_dofs), device=self._device)
        dof_limits = self._mobilefrankas.get_dof_limits()
        self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.franka_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)

        # control the joint speeds with these
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[self._mobilefrankas.gripper_indices] = 0.1
        self.franka_dof_speed_scales[self._mobilefrankas._base_indices] = 0.1
        
        self.franka_dof_targets = torch.zeros(
            (self._num_envs, self.num_franka_dofs), dtype=torch.float, device=self._device
        )

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(torch.square(self.actions[:, 2:]), dim=-1)

        distance_to_target = torch.norm(self.to_target, p=2, dim=-1) # / self.dt

        arm_joint_dof_pos = self.franka_dof_pos[:, 3:-2]
        penalty_joint_limit = self._joint_limit_penalty(arm_joint_dof_pos)
        
        reward = torch.zeros_like(self.rew_buf)
        reward +=  0.5 * torch.exp(-1.2 * distance_to_target) - self.action_penalty_scale * action_penalty - 0.06 * penalty_joint_limit
        self.extras["rewards/distance_to_target"] = torch.mean(distance_to_target)
        self.extras["rewards/penalty_joint_limit"] = torch.mean(penalty_joint_limit)
        self.extras["rewards/action_penalty"] = torch.mean(action_penalty)
        self.rew_buf[:] = reward
    
    def _joint_limit_penalty(self, values):
        # neutral position of joints
        neutral = torch.tensor([0,0,0,-1.5,0,2.0,0], device=self._device)
        # weights for each joint how much to penalize them incase they differ a lot from neutral
        weights = torch.tensor([1.5, 1, 1.5, 1, 1, 2.0, 1], device=self._device)
        return torch.sum(torch.abs(values-neutral) * weights, axis=1)

    def is_done(self) -> None:
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

