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

class Rotate(IsaacEnv):
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
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.time_encoding = cfg.task.time_encoding
        self.future_traj_steps = int(cfg.task.future_traj_steps)
        assert self.future_traj_steps > 0
        self.intrinsics = cfg.task.intrinsics
        self.wind = cfg.task.wind
        self.use_eval = cfg.task.use_eval
        self.num_drones = 1
        self.use_rotor2critic = cfg.task.use_rotor2critic
        self.action_history_step = cfg.task.action_history_step

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
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        )
        # self.traj_c_dist = D.Uniform(
        #     torch.tensor(-0.6, device=self.device),
        #     torch.tensor(0.6, device=self.device)
        # )
        # self.traj_scale_dist = D.Uniform(
        #     torch.tensor([0.5, 0.5, 0.25], device=self.device),
        #     torch.tensor([1.2, 1.2, 0.25], device=self.device)
        # )

        self.v_scale_dist = D.Uniform(
            torch.tensor(0.5, device=self.device),
            torch.tensor(1.2, device=self.device)
        )

        # self.v_scale_dist = D.Uniform(
        #     torch.tensor(2.0, device=self.device),
        #     torch.tensor(2.2, device=self.device)
        # )
        
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
            # self.traj_scale_dist = D.Uniform(
            #     torch.tensor([1.2, 1.2, 0.25], device=self.device),
            #     torch.tensor([1.2, 1.2, 0.25], device=self.device)
            # )
            self.v_scale_dist = D.Uniform(
                torch.tensor(1.0, device=self.device),
                torch.tensor(1.0, device=self.device)
                )
            # self.traj_c_dist = D.Uniform(
            #     torch.tensor(0.0, device=self.device),
            #     torch.tensor(0.0, device=self.device)
            # )
        
        self.origin = torch.tensor([0., 0., 1.], device=self.device)

        self.traj_t0 = torch.pi / 2.0
        # self.traj_c = torch.zeros(self.num_envs, device=self.device)
        # self.traj_scale = torch.zeros(self.num_envs, 3, device=self.device)
        self.v_scale = torch.zeros(self.num_envs, device=self.device)
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
        # self.prev_prev_actions = torch.zeros(self.num_envs, self.num_drones, 4, device=self.device)

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
        drone_state_dim = 3 + 3 + 4 + 3 + 3 # position, velocity, quaternion, heading, up
        obs_dim = drone_state_dim + 3 * (self.future_traj_steps-1)
        if self.time_encoding:
            self.time_encoding_dim = 4
            obs_dim += self.time_encoding_dim
        if self.intrinsics:
            obs_dim += sum(spec.shape[-1] for name, spec in self.drone.info_spec.items())
        
        # action history
        self.action_history = self.cfg.task.action_history_step if self.cfg.task.use_action_history else 0
        self.action_history_buffer = collections.deque(maxlen=self.action_history)

        state_dim = obs_dim + 4
        
        if self.action_history > 0:
            obs_dim += self.action_history * 4
        
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
        }).expand(self.num_envs).to(self.device)
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
            "prev_action": torch.stack([self.drone.action_spec] * self.drone.n, 0).to(self.device),
            "prev_prev_action": torch.stack([self.drone.action_spec] * self.drone.n, 0).to(self.device),
        }).expand(self.num_envs).to(self.device)
        # info_spec = self.drone.info_spec.to(self.device)
        self.observation_spec["info"] = info_spec
        self.observation_spec["stats"] = stats_spec
        self.info = info_spec.zero()
        self.stats = stats_spec.zero()

        self.random_latency = self.cfg.task.random_latency
        self.latency = self.cfg.task.latency_step if self.cfg.task.latency else 0
        # self.obs_buffer = collections.deque(maxlen=self.latency)
        self.root_state_buffer = collections.deque(maxlen=self.latency)
        
    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        self.traj_rot[env_ids] = euler_to_quaternion(self.traj_rpy_dist.sample(env_ids.shape))
        # self.traj_scale[env_ids] = self.traj_scale_dist.sample(env_ids.shape)
        self.v_scale[env_ids] = self.v_scale_dist.sample(env_ids.shape)

        t0 = torch.zeros(len(env_ids), device=self.device)
        # pos, _ = lemniscate(t0 + self.traj_t0, self.traj_c[env_ids])
        # pos, _ = lemniscate(t0 + self.traj_t0)
        pos = lemniscate_v(t0 + self.traj_t0, self.v_scale[env_ids])
        pos = pos + self.origin
        # if self.use_eval:
        #     pos = torch.zeros(len(env_ids), 3, device=self.device)
        #     pos = pos + self.origin
        rot = euler_to_quaternion(self.init_rpy_dist.sample(env_ids.shape))
        vel = torch.zeros(len(env_ids), 1, 6, device=self.device)
        # vel[..., :3] = linear_v.unsqueeze(1)
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

        cmd_init = 2.0 * (self.drone.throttle[env_ids]) ** 2 - 1.0
        max_thrust_ratio = self.drone.params['max_thrust_ratio']
        self.info['prev_action'][env_ids, :, 3] = (0.5 * (max_thrust_ratio + cmd_init)).mean(dim=-1)
        self.info['prev_prev_action'][env_ids, :, 3] = (0.5 * (max_thrust_ratio + cmd_init)).mean(dim=-1)
        self.prev_actions[env_ids] = self.info['prev_action'][env_ids]
        # self.prev_prev_actions[env_ids] = self.info['prev_prev_action'][env_ids]
        
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
        self.info["prev_prev_action"] = tensordict[("info", "prev_prev_action")]
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
        obs = [
            self.rpos.flatten(1).unsqueeze(1),
            root_state[..., 3:10], root_state[..., 13:19],
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
        
        # add throttle to critic
        if self.use_rotor2critic:
            state = torch.concat([obs, self.drone.throttle], dim=-1).squeeze(1)
        else:
            state = obs.squeeze(1)
        
        # add action history to actor
        if self.action_history > 0:
            self.action_history_buffer.append(self.prev_actions)
            all_action_history = torch.concat(list(self.action_history_buffer), dim=-1)
            obs = torch.cat([obs, all_action_history], dim=-1)

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
        reward_up = 0.5 / (1.0 + torch.square(tiltage))

        # effort
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.action_error_order1)
        # if self.use_action_error_order2:
        #     reward_action_smoothness += self.reward_action_smoothness_weight * torch.exp(-self.action_error_order2)
        
        # spin reward, fixed z
        spin = torch.square(self.drone.vel[..., -1])
        reward_spin = 0.5 / (1.0 + torch.square(spin))

        reward = (
            reward_pos
            + reward_pos * (reward_up + reward_spin)
            + reward_action_smoothness
        )
        
        self.stats['reward_pos'].add_(reward_pos)
        self.stats['reward_action_smoothness'].add_(reward_action_smoothness)
        self.stats['reward_spin'].add_(reward_pos * reward_spin)
        self.stats['reward_up'].add_(reward_pos * reward_up)

        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            | (self.drone.pos[..., 2] < 0.1)
            # | (distance > self.reset_thres)
        )

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
        t = self.progress_buf[env_ids].unsqueeze(1) + step_size * torch.arange(steps, device=self.device)
        t = self.traj_t0 + scale_time(t * self.dt)
        traj_rot = self.traj_rot[env_ids].unsqueeze(1).expand(-1, t.shape[1], 4)
        
        # target_pos = vmap(lemniscate)(t)
        target_pos = vmap(lemniscate_v)(t, self.v_scale[env_ids].unsqueeze(-1))
        # target_pos = vmap(torch_utils.quat_rotate)(traj_rot, target_pos) * self.traj_scale[env_ids].unsqueeze(1)
        target_pos = vmap(torch_utils.quat_rotate)(traj_rot, target_pos)

        return self.origin + target_pos
