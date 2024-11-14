# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from functorch import vmap

import omni.isaac.core.utils.torch as torch_utils
import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion
import omni.isaac.core.utils.prims as prim_utils
import torch
import torch.distributions as D

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec
from omni.isaac.debug_draw import _debug_draw

from ..utils import lemniscate, lemniscate_v, pentagram, scale_time
import collections
import numpy as np
from omni_drones.utils.torch import quaternion_to_euler

class Track(IsaacEnv):
    r"""
    A basic control task. The goal for the agent is to track a reference 
    lemniscate trajectory in the 3D space.

    ## Observation
    
    - `rpos` (3 * `future_traj_steps`): The relative position of the drone to the 
      reference positions in the future `future_traj_steps` time steps.
    - `root_state` (16 + `num_rotors`): The basic information of the drone (except its position), 
      containing its rotation (in quaternion), velocities (linear and angular), 
      heading and up vectors, and the current throttle.
    - `time_encoding` (optional): The time encoding, which is a 4-dimensional
      vector encoding the current progress of the episode.

    ## Reward

    - `pos`: Reward for tracking the trajectory, computed from the position
      error as {math}`\exp(-a * \text{pos_error})`.
    - `up`: Reward computed from the uprightness of the drone to discourage
      large tilting.
    - `spin`: Reward computed from the spin of the drone to discourage spinning.
    - `effort`: Reward computed from the effort of the drone to optimize the
      energy consumption.
    - `action_smoothness`: Reward that encourages smoother drone actions, computed based on the throttle difference of the drone.

    The total reward is computed as follows:
    ```{math}
        r = r_\text{pos} + r_\text{pos} * (r_\text{up} + r_\text{heading}) + r_\text{effort} + r_\text{action_smoothness}
    ```

    ## Episode End

    The episode ends when the tracking error is larger than `reset_thres`, or
    when the drone is too close to the ground, or when the episode reaches 
    the maximum length.

    ## Config

    | Parameter               | Type  | Default       | Description |
    |-------------------------|-------|---------------|-------------|
    | `drone_model`           | str   | "hummingbird" | Specifies the model of the drone being used in the environment. |
    | `reset_thres`           | float | 0.5           | Threshold for the distance between the drone and its target, upon exceeding which the episode will be reset. |
    | `future_traj_steps`     | int   | 4             | Number of future trajectory steps the drone needs to predict. |
    | `reward_distance_scale` | float | 1.2           | Scales the reward based on the distance between the drone and its target. |
    | `time_encoding`         | bool  | True          | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |


    """
    def __init__(self, cfg, headless):
        self.reset_thres = cfg.task.reset_thres
        self.reward_acc_weight_init = cfg.task.reward_acc_weight_init
        self.reward_acc_weight_lr = cfg.task.reward_acc_weight_lr
        self.reward_acc_max = cfg.task.reward_acc_max
        self.reward_jerk_weight_init = cfg.task.reward_jerk_weight_init
        self.reward_jerk_weight_lr = cfg.task.reward_jerk_weight_lr
        self.reward_jerk_max = cfg.task.reward_jerk_max
        # action norm and smoothness
        self.reward_action_smoothness_weight_init = cfg.task.reward_action_smoothness_weight_init
        self.reward_action_smoothness_weight_lr = cfg.task.reward_action_smoothness_weight_lr
        self.reward_smoothness_max = cfg.task.reward_smoothness_max
        self.reward_action_norm_weight_init = cfg.task.reward_action_norm_weight_init
        self.reward_action_norm_weight_lr = cfg.task.reward_action_norm_weight_lr
        self.reward_norm_max = cfg.task.reward_norm_max
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.time_encoding = cfg.task.time_encoding
        self.future_traj_steps = int(cfg.task.future_traj_steps)
        assert self.future_traj_steps > 0
        self.intrinsics = cfg.task.intrinsics
        self.wind = cfg.task.wind
        self.use_eval = cfg.task.use_eval
        self.num_drones = 1
        self.use_throttle2critic = cfg.task.use_throttle2critic
        self.action_history_step = cfg.task.action_history_step
        self.trajectory_scale = cfg.task.trajectory_scale # 'slow', 'normal', 'fast'
        self.reward_spin_weight = cfg.task.reward_spin_weight
        self.reward_yaw_weight = cfg.task.reward_yaw_weight
        self.reward_up_weight = cfg.task.reward_up_weight
        self.use_random_init = cfg.task.use_random_init
        self.use_ab_wolrd_pos = cfg.task.use_ab_wolrd_pos
        self.use_vel_init = cfg.task.use_vel_init
        self.use_obs_noise = cfg.task.use_obs_noise
        self.obs_noise_scale = cfg.task.obs_noise_scale
        self.use_infessible_done = cfg.task.use_infessible_done
        self.sim_data = []
        self.sim_rpy = []
        self.action_data = []

        super().__init__(cfg, headless)

        self.drone.initialize()
        randomization = self.cfg.task.get("randomization", None)
        if randomization is not None:
            if "drone" in self.cfg.task.randomization:
                self.drone.setup_randomization(self.cfg.task.randomization["drone"])

        if self.wind:
            if randomization is not None:
                if "wind" in self.cfg.task.randomization:
                    cfg = self.cfg.task.randomization["wind"]
                    # for phase in ("train", "eval"):
                    wind_intensity_scale = cfg['train'].get("intensity", None)
                    self.wind_intensity_low = wind_intensity_scale[0]
                    self.wind_intensity_high = wind_intensity_scale[1]
            else:
                self.wind_intensity_low = 0
                self.wind_intensity_high = 2
            self.wind_w = torch.zeros(self.num_envs, 3, 8, device=self.device)
            self.wind_i = torch.zeros(self.num_envs, 1, device=self.device)
        
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )
        self.traj_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi
        )

        if self.trajectory_scale == 'slow':
            self.T_scale_dist = D.Uniform(
                torch.tensor(14.8, device=self.device),
                torch.tensor(15.2, device=self.device)
            )
        elif self.trajectory_scale == 'normal':
            self.T_scale_dist = D.Uniform(
                torch.tensor(5.3, device=self.device),
                torch.tensor(5.7, device=self.device)
            )
        elif self.trajectory_scale == 'fast':
            self.T_scale_dist = D.Uniform(
                torch.tensor(3.3, device=self.device),
                torch.tensor(3.7, device=self.device)
            )
        else:
            self.T_scale_dist = D.Uniform(
                torch.tensor(3.3, device=self.device),
                torch.tensor(15.2, device=self.device)
            )
        
        # eval
        if self.use_eval:
            self.init_rpy_dist = D.Uniform(
                torch.tensor([-.0, -.0, 0.], device=self.device) * torch.pi,
                torch.tensor([0., 0., 0.], device=self.device) * torch.pi
            )
            self.traj_rpy_dist = D.Uniform(
                torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
                torch.tensor([0., 0., 0.], device=self.device) * torch.pi
            )
            if self.trajectory_scale == 'slow':
                self.T_scale_dist = D.Uniform(
                    torch.tensor(15.0, device=self.device),
                    torch.tensor(15.0, device=self.device)
                )
            elif self.trajectory_scale == 'normal':
                self.T_scale_dist = D.Uniform(
                    torch.tensor(5.5, device=self.device),
                    torch.tensor(5.5, device=self.device)
                )
            elif self.trajectory_scale == 'fast':
                self.T_scale_dist = D.Uniform(
                    torch.tensor(3.5, device=self.device),
                    torch.tensor(3.5, device=self.device)
                )
        
        self.origin = torch.tensor([0., 0., 1.], device=self.device)

        self.traj_t0 = torch.ones(self.num_envs, device=self.device)
        self.T_scale = torch.zeros(self.num_envs, device=self.device)
        self.traj_rot = torch.zeros(self.num_envs, 4, device=self.device)

        self.last_linear_v = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_angular_v = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_linear_a = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_angular_a = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_linear_jerk = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_angular_jerk = torch.zeros(self.num_envs, 1, device=self.device)

        self.target_pos = torch.zeros(self.num_envs, self.future_traj_steps, 3, device=self.device)

        self.alpha = 0.8

        self.draw = _debug_draw.acquire_debug_draw_interface()
        
        self.prev_actions = torch.zeros(self.num_envs, self.num_drones, 4, device=self.device)
        self.count = 0 # episode of RL training

    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        self.drone.spawn(translations=[(0.0, 0.0, 1.5)])
        return ["/World/defaultGroundPlane"]
    
    def _set_specs(self):
        self.use_obs_norm = self.cfg.task.use_obs_norm
        if self.use_ab_wolrd_pos:
            drone_state_dim = 3 + 3 + 3 + 3 + 3 + 3 # pos, linear vel, body rate, heading, lateral, up
        else:
            # drone_state_dim = 4 + 3 + 3 + 3 + 3 # quat, linear vel, heading, lateral, up
            drone_state_dim = 3 + 3 + 3 + 3 # linear vel, heading, lateral, up
        obs_dim = drone_state_dim + 3 * self.future_traj_steps
        
        self.time_encoding_dim = 4
        if self.time_encoding:
            obs_dim += self.time_encoding_dim
        if self.intrinsics:
            obs_dim += sum(spec.shape[-1] for name, spec in self.drone.info_spec.items())
        
        if self.use_obs_norm:
            rpos_scale = [0.1, 0.1, 0.1] * self.future_traj_steps
            vel_scale = [0.1, 0.1, 0.1]
            rotation_scale = [1.0] * 9
            self.obs_norm_scale = torch.tensor(rpos_scale + vel_scale + rotation_scale).to(self.device)
        
        # action history
        self.action_history = self.cfg.task.action_history_step if self.cfg.task.use_action_history else 0
        self.action_history_buffer = collections.deque(maxlen=self.action_history)
        
        if self.action_history > 0:
            obs_dim += self.action_history * 4
        
        state_dim = obs_dim + 4
        if self.use_throttle2critic:
            state_dim += 4
        
        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": UnboundedContinuousTensorSpec((1, obs_dim)),
                "state": UnboundedContinuousTensorSpec((state_dim)), # add motor speed
            }
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": {
                "action": self.drone.action_spec.unsqueeze(0),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": {
                "reward": UnboundedContinuousTensorSpec((1, 1))
            }
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "state"),
        )
        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "tracking_error": UnboundedContinuousTensorSpec(1),
            "tracking_error_ema": UnboundedContinuousTensorSpec(1),
            "action_error_order1_mean": UnboundedContinuousTensorSpec(1),
            "action_error_order1_max": UnboundedContinuousTensorSpec(1),
            "action_error_order2_mean": UnboundedContinuousTensorSpec(1),
            "action_error_order2_max": UnboundedContinuousTensorSpec(1),
            "smoothness_mean": UnboundedContinuousTensorSpec(1),
            "smoothness_max": UnboundedContinuousTensorSpec(1),
            "drone_state": UnboundedContinuousTensorSpec(13),
            "reward_pos": UnboundedContinuousTensorSpec(1),
            "reward_up": UnboundedContinuousTensorSpec(1),
            "reward_spin": UnboundedContinuousTensorSpec(1),
            "reward_action_smoothness": UnboundedContinuousTensorSpec(1),
            "reward_action_smoothness_scale": UnboundedContinuousTensorSpec(1),
            "reward_action_norm_scale": UnboundedContinuousTensorSpec(1),
            "linear_v_max": UnboundedContinuousTensorSpec(1),
            "angular_v_max": UnboundedContinuousTensorSpec(1),
            "linear_a_max": UnboundedContinuousTensorSpec(1),
            "angular_a_max": UnboundedContinuousTensorSpec(1),
            "linear_jerk_max": UnboundedContinuousTensorSpec(1),
            "angular_jerk_max": UnboundedContinuousTensorSpec(1),
            "linear_v_mean": UnboundedContinuousTensorSpec(1),
            "angular_v_mean": UnboundedContinuousTensorSpec(1),
            "linear_a_mean": UnboundedContinuousTensorSpec(1),
            "angular_a_mean": UnboundedContinuousTensorSpec(1),
            "linear_jerk_mean": UnboundedContinuousTensorSpec(1),
            "angular_jerk_mean": UnboundedContinuousTensorSpec(1),
            "obs_range": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
            "prev_action": torch.stack([self.drone.action_spec] * self.drone.n, 0).to(self.device),
            "policy_action": torch.stack([self.drone.action_spec] * self.drone.n, 0).to(self.device),
            # "prev_prev_action": torch.stack([self.drone.action_spec] * self.drone.n, 0).to(self.device),
        }).expand(self.num_envs).to(self.device)
        # info_spec = self.drone.info_spec.to(self.device)
        self.observation_spec["info"] = info_spec
        self.observation_spec["stats"] = stats_spec
        self.info = info_spec.zero()
        self.stats = stats_spec.zero()

        self.random_latency = self.cfg.task.random_latency
        self.latency = self.cfg.task.latency_step if self.cfg.task.latency else 0
        # self.obs_buffer = collections.deque(maxlen=self.latency)
        self.root_state_buffer = collections.deque(maxlen=self.latency + 1)
        
    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        self.traj_rot[env_ids] = euler_to_quaternion(self.traj_rpy_dist.sample(env_ids.shape))
        self.T_scale[env_ids] = self.T_scale_dist.sample(env_ids.shape)
        if self.use_random_init:
            self.traj_t0[env_ids] = torch.rand(env_ids.shape).to(self.device) * self.T_scale[env_ids] # 0 ~ T
        else:
            self.traj_t0[env_ids] = 0.25 * self.T_scale[env_ids]
            # self.traj_t0[env_ids] = 0.4 * self.T_scale[env_ids]

        t0 = torch.zeros(len(env_ids), device=self.device)
        pos, linear_v = lemniscate_v(t0 + self.traj_t0[env_ids], self.T_scale[env_ids])
        pos = pos + self.origin
        rot = euler_to_quaternion(self.init_rpy_dist.sample(env_ids.shape))
        vel = torch.zeros(len(env_ids), 1, 6, device=self.device)
        if self.use_vel_init:
            vel[..., :3] = linear_v.unsqueeze(1)
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids], rot, env_ids
        )
        self.drone.set_velocities(vel, env_ids)
        
        # set last values
        self.last_linear_v[env_ids] = torch.norm(vel[..., :3], dim=-1)
        self.last_angular_v[env_ids] = torch.norm(vel[..., 3:], dim=-1)
        self.last_linear_a[env_ids] = torch.zeros_like(self.last_linear_v[env_ids])
        self.last_angular_a[env_ids] = torch.zeros_like(self.last_angular_v[env_ids])
        self.last_linear_jerk[env_ids] = torch.zeros_like(self.last_linear_a[env_ids])
        self.last_angular_jerk[env_ids] = torch.zeros_like(self.last_angular_a[env_ids])

        self.stats[env_ids] = 0.

        # init prev_actions: hover
        cmd_init = 2.0 * (self.drone.throttle[env_ids]) ** 2 - 1.0
        self.info['prev_action'][env_ids, :, 3] = cmd_init.mean(dim=-1)
        self.prev_actions[env_ids] = self.info['prev_action'][env_ids].clone()
        
        # add init_action to self.action_history_buffer
        for _ in range(self.action_history):
            self.action_history_buffer.append(self.prev_actions) # add all prev_actions, not len(env_ids)
        
        if self._should_render(0) and (env_ids == self.central_env_idx).any() :
            # visualize the trajectory
            self.draw.clear_lines()
            traj_vis = self._compute_traj(self.max_episode_length, self.central_env_idx.unsqueeze(0))[0]
            traj_vis = traj_vis + self.envs_positions[self.central_env_idx]
            point_list_0 = traj_vis[:-1].tolist()
            point_list_1 = traj_vis[1:].tolist()
            colors = [(1.0, 1.0, 1.0, 1.0) for _ in range(len(point_list_0))]
            sizes = [1 for _ in range(len(point_list_0))]
            self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)
            
        if self.wind:
            self.wind_i[env_ids] = torch.rand(*env_ids.shape, 1, device=self.device) * (self.wind_intensity_high-self.wind_intensity_low) + self.wind_intensity_low
            self.wind_w[env_ids] = torch.randn(*env_ids.shape, 3, 8, device=self.device)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.info["prev_action"] = tensordict[("info", "prev_action")]
        self.info["policy_action"] = tensordict[("info", "policy_action")]
        # self.info["prev_prev_action"] = tensordict[("info", "prev_prev_action")]
        self.policy_actions = tensordict[("info", "policy_action")].clone()
        self.prev_actions = self.info["prev_action"].clone()
        # self.prev_prev_actions = self.info["prev_prev_action"].clone()
        
        self.action_error_order1 = tensordict[("stats", "action_error_order1")].clone()
        self.stats["action_error_order1_mean"].add_(self.action_error_order1.mean(dim=-1).unsqueeze(-1))
        self.stats["action_error_order1_max"].set_(torch.max(self.stats["action_error_order1_max"], self.action_error_order1.mean(dim=-1).unsqueeze(-1)))
        # self.action_error_order2 = tensordict[("stats", "action_error_order2")].clone()
        # self.stats["action_error_order2_mean"].add_(self.action_error_order2.mean(dim=-1).unsqueeze(-1))
        # self.stats["action_error_order2_max"].set_(torch.max(self.stats["action_error_order2_max"], self.action_error_order2.mean(dim=-1).unsqueeze(-1)))

        self.effort = self.drone.apply_action(actions)

        if self.wind:
            t = (self.progress_buf * self.dt).reshape(-1, 1, 1)
            self.wind_force = self.wind_i * torch.sin(t * self.wind_w).sum(-1)
            wind_forces = self.drone.MASS_0 * self.wind_force
            wind_forces = wind_forces.unsqueeze(1).expand(*self.drone.shape, 3)
            self.drone.base_link.apply_forces(wind_forces, is_global=True)

    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()
        self.info["drone_state"][:] = self.root_state[..., :13]

        if self.cfg.task.latency:
            self.root_state_buffer.append(self.root_state)
            # set t and target pos to the real values
            if self.random_latency:
                random_indices = torch.randint(0, len(self.root_state_buffer), (self.num_envs,), device=self.device)
                root_state = torch.stack(list(self.root_state_buffer))[random_indices, torch.arange(self.num_envs)]
            else:
                root_state = self.root_state_buffer[0]
        else:
            root_state = self.root_state

        self.target_pos[:] = self._compute_traj(self.future_traj_steps, step_size=5)
        
        self.rpos = self.target_pos - root_state[..., :3]
        if self.use_ab_wolrd_pos:
            # pos, rpos, linear velocity, body rate, heading, lateral, up
            obs = [
                root_state[..., :3],
                self.rpos.flatten(1).unsqueeze(1),
                root_state[..., 7:10],
                root_state[..., 16:19], root_state[..., 19:28],
            ]
        else:
            # rpos, linear velocity, body rate, heading, lateral, up
            obs = [
                self.rpos.flatten(1).unsqueeze(1),
                # root_state[..., 3:7], # quat
                root_state[..., 7:10], # linear v
                root_state[..., 19:28], # rotation
            ]
        self.stats['drone_state'] = root_state[..., :13].squeeze(1).clone()
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        if self.intrinsics:
            obs.append(self.drone.get_info())

        self.stats["smoothness_mean"].add_(self.drone.throttle_difference)
        self.stats["smoothness_max"].set_(torch.max(self.drone.throttle_difference, self.stats["smoothness_max"]))
        # linear_v, angular_v
        self.linear_v = torch.norm(self.root_state[..., 7:10], dim=-1)
        self.angular_v = torch.norm(self.root_state[..., 10:13], dim=-1)
        self.stats["linear_v_max"].set_(torch.max(self.stats["linear_v_max"], torch.abs(self.linear_v)))
        self.stats["linear_v_mean"].add_(self.linear_v)
        self.stats["angular_v_max"].set_(torch.max(self.stats["angular_v_max"], torch.abs(self.angular_v)))
        self.stats["angular_v_mean"].add_(self.angular_v)
        # linear_a, angular_a
        self.linear_a = torch.abs(self.linear_v - self.last_linear_v) / self.dt
        self.angular_a = torch.abs(self.angular_v - self.last_angular_v) / self.dt
        self.stats["linear_a_max"].set_(torch.max(self.stats["linear_a_max"], torch.abs(self.linear_a)))
        self.stats["linear_a_mean"].add_(self.linear_a)
        self.stats["angular_a_max"].set_(torch.max(self.stats["angular_a_max"], torch.abs(self.angular_a)))
        self.stats["angular_a_mean"].add_(self.angular_a)
        # linear_jerk, angular_jerk
        self.linear_jerk = torch.abs(self.linear_a - self.last_linear_a) / self.dt
        self.angular_jerk = torch.abs(self.angular_a - self.last_angular_a) / self.dt
        self.stats["linear_jerk_max"].set_(torch.max(self.stats["linear_jerk_max"], torch.abs(self.linear_jerk)))
        self.stats["linear_jerk_mean"].add_(self.linear_jerk)
        self.stats["angular_jerk_max"].set_(torch.max(self.stats["angular_jerk_max"], torch.abs(self.angular_jerk)))
        self.stats["angular_jerk_mean"].add_(self.angular_jerk)
        
        # set last
        self.last_linear_v = self.linear_v.clone()
        self.last_angular_v = self.angular_v.clone()
        self.last_linear_a = self.linear_a.clone()
        self.last_angular_a = self.angular_a.clone()
        self.last_linear_jerk = self.linear_jerk.clone()
        self.last_angular_jerk = self.angular_jerk.clone()
        
        obs = torch.cat(obs, dim=-1)
        
        if self.use_obs_norm:
            obs = obs * self.obs_norm_scale.unsqueeze(0).unsqueeze(0).repeat(self.num_envs, 1, 1)

        self.stats["obs_range"].set_(torch.max(torch.abs(obs), dim=-1).values)

        # add action history to actor
        if self.action_history > 0:
            self.action_history_buffer.append(self.prev_actions)
            all_action_history = torch.concat(list(self.action_history_buffer), dim=-1)
            obs = torch.cat([obs, all_action_history], dim=-1)

        # state: obs + t + throttle
        t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
        state = torch.concat([obs, t.expand(-1, self.time_encoding_dim).unsqueeze(1)], dim=-1).squeeze(1)
        if self.use_throttle2critic:
            # add throttle to critic, throttle is the ground truth, w.o. action noise
            state = torch.concat([state, self.root_state[:, 0, 28:]], dim=-1)

        # state: ground truth, obs: add noise to obs (without action history)
        if self.use_obs_noise:
            if self.action_history > 0:
                obs_noise_dim = obs.shape[-1] - self.action_history * 4
                obs[..., :obs_noise_dim] *= torch.randn(self.num_envs, 1, obs_noise_dim, device=self.device) * self.obs_noise_scale + 1 # add a gaussian noise of mean 0 and variance self.obs_noise_scale**2
            else:
                obs *= torch.randn(obs.shape, device=self.device) * self.obs_noise_scale + 1 # add a gaussian noise of mean 0 and variance self.obs_noise_scale**2

        if self.use_eval:
            self.sim_data.append(obs[0].clone())
            self.sim_rpy.append(self.drone.vel_b[0, :, 3:].clone())

        return TensorDict({
            "agents": {
                "observation": obs,
                "state": state,
            },
            "stats": self.stats,  
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        # pos reward
        distance = torch.norm(self.rpos[:, [0]], dim=-1)
        self.stats["tracking_error"].add_(-distance)
        self.stats["tracking_error_ema"].lerp_(distance, (1-self.alpha))
        
        reward_pos = self.reward_distance_scale * torch.exp(-distance)
        
        # uprightness
        tiltage = torch.abs(1 - self.drone.up[..., 2])
        reward_up = self.reward_up_weight * 0.5 / (1.0 + torch.square(tiltage))

        # reward action norm
        self.reward_action_norm_weight = min(self.reward_action_norm_weight_init + self.reward_action_norm_weight_lr * self.count, self.reward_norm_max)
        reward_action_norm = self.reward_action_norm_weight * torch.exp(-torch.norm(self.policy_actions, dim=-1))

        # reward action smooth
        self.reward_action_smoothness_weight = min(self.reward_action_smoothness_weight_init + self.reward_action_smoothness_weight_lr * self.count, self.reward_smoothness_max)
        not_begin_flag = (self.progress_buf > 1).unsqueeze(1)
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.action_error_order1) * not_begin_flag.float()
        
        # reward acc
        self.reward_acc_weight = min(self.reward_acc_weight_init + self.reward_acc_weight_lr * self.count, self.reward_acc_max)
        reward_acc = self.reward_acc_weight * torch.exp(-self.linear_a)
        # reward jerk
        self.reward_jerk_weight = min(self.reward_jerk_weight_init + self.reward_jerk_weight_lr * self.count, self.reward_jerk_max)
        reward_jerk = self.reward_jerk_weight * torch.exp(-self.linear_jerk)

        # spin reward, fixed z
        spin = torch.square(self.drone.vel_b[..., -1])
        reward_spin = self.reward_spin_weight * 0.5 / (1.0 + torch.square(spin))

        # yaw reward
        rpy = quaternion_to_euler(self.drone.rot)
        reward_yaw = self.reward_yaw_weight * 0.5 / (1.0 + torch.square(rpy[..., -1]))

        reward = (
            reward_pos
            + reward_pos * (reward_up + reward_spin + reward_yaw)
            + reward_action_norm
            + reward_action_smoothness
            + reward_acc
            + reward_jerk
        )
        
        self.stats['reward_pos'].add_(reward_pos)
        self.stats['reward_action_smoothness'].add_(reward_action_smoothness)
        self.stats['reward_spin'].add_(reward_pos * reward_spin)
        self.stats['reward_up'].add_(reward_pos * reward_up)
        self.stats['reward_action_smoothness_scale'].set_(self.reward_action_smoothness_weight * torch.ones(self.num_envs, 1, device=self.device))
        self.stats['reward_action_norm_scale'].set_(self.reward_action_norm_weight * torch.ones(self.num_envs, 1, device=self.device))

        if self.use_infessible_done:
            done = (
                (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
                | (self.drone.pos[..., 2] < 0.1)
                | (distance > self.reset_thres)
            )
        else:
            done = (
                (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
                | (self.drone.pos[..., 2] < 0.1)
            )
        
        if self.use_eval:
            self.action_data.append(self.prev_actions[0].clone())
            if done[0]:
                torch.save(self.sim_data, 'sim_state.pt')
                torch.save(self.sim_rpy, 'sim_rpy.pt')
                torch.save(self.action_data, 'sim_action.pt')

        ep_len = self.progress_buf.unsqueeze(-1)
        self.stats["tracking_error"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['action_error_order1_mean'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['action_error_order2_mean'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['smoothness_mean'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['reward_pos'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['reward_spin'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['reward_up'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['reward_action_smoothness'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["linear_v_mean"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["angular_v_mean"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["linear_a_mean"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["angular_a_mean"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1)
                },
                "done": done,
            },
            self.batch_size,
        )
        
    def _compute_traj(self, steps: int, env_ids=None, step_size: float=1.):
        if env_ids is None:
            env_ids = ...
        # discrete t
        t = self.progress_buf[env_ids].unsqueeze(1) + step_size * torch.arange(steps, device=self.device)
        # t: [num_envs, steps], continuous t
        t = self.traj_t0[env_ids].unsqueeze(1) + t * self.dt
        # target_pos: [num_envs, steps, 3]
        target_pos, _ = vmap(lemniscate_v)(t, self.T_scale[env_ids].unsqueeze(-1))

        return self.origin + target_pos

