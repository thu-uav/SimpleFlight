import functorch
import torch
import torch.distributions as D

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.objects as objects
import numpy as np

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView, RigidPrimView
from omni_drones.utils.torch import euler_to_quaternion, quat_axis, quat_rotate_inverse

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec
import collections
from torch.distributions import Normal

def attach_payload(parent_path):
    from omni.isaac.core import objects
    import omni.physx.scripts.utils as script_utils
    from pxr import UsdPhysics

    payload_prim = objects.DynamicCuboid(
        prim_path=parent_path + "/payload",
        scale=torch.tensor([0.1, 0.1, .15]),
        mass=0.0001
    ).prim

    parent_prim = prim_utils.get_prim_at_path(parent_path + "/base_link")
    stage = prim_utils.get_current_stage()
    joint = script_utils.createJoint(stage, "Prismatic", payload_prim, parent_prim)
    UsdPhysics.DriveAPI.Apply(joint, "linear")
    joint.GetAttribute("physics:lowerLimit").Set(-0.15)
    joint.GetAttribute("physics:upperLimit").Set(0.15)
    joint.GetAttribute("physics:axis").Set("Z")
    joint.GetAttribute("drive:linear:physics:damping").Set(10.)
    joint.GetAttribute("drive:linear:physics:stiffness").Set(10000.)

class Hover(IsaacEnv):
    r"""
    A basic control task. The goal for the agent is to maintain a stable
    position and heading in mid-air without drifting. 

    Observation
    -----------
    The observation space consists of the following part:

    - `rpos` (3): The position relative to the target hovering position.
    - `root_state` (16 + num_rotors): The basic information of the drone (except its position), 
      containing its rotation (in quaternion), velocities (linear and angular), 
      heading and up vectors, and the current throttle.
    - `rheading` (3): The difference between the reference heading and the current heading.
    - *time_encoding*:

    Reward 
    ------
    - pos: 
    - heading_alignment:
    - up:
    - spin:

    The total reward is 

    .. math:: 
    
        r = r_\text{pos} + r_\text{pos} * (r_\text{up} + r_\text{heading})

    Episode End
    -----------
    - Termination: 

    Config
    ------


    """
    def __init__(self, cfg, headless):
        # self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.time_encoding = cfg.task.time_encoding
        self.use_eval = cfg.task.use_eval
        self.use_disturbance = cfg.task.use_disturbance
        
        self.randomization = cfg.task.get("randomization", {})
        self.has_payload = "payload" in self.randomization.keys()

        super().__init__(cfg, headless)

        self.drone.initialize()
        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])
        if "payload" in self.randomization:
            payload_cfg = self.randomization["payload"]
            self.payload_z_dist = D.Uniform(
                torch.tensor([payload_cfg["z"][0]], device=self.device),
                torch.tensor([payload_cfg["z"][1]], device=self.device)
            )
            self.payload_mass_dist = D.Uniform(
                torch.tensor([payload_cfg["mass"][0]], device=self.device),
                torch.tensor([payload_cfg["mass"][1]], device=self.device)
            )
            self.payload = RigidPrimView(
                f"/World/envs/env_*/{self.drone.name}_*/payload",
                reset_xform_properties=False,
                shape=(-1, self.drone.n)
            )
            self.payload.initialize()
        
        self.target_vis = ArticulationView(
            "/World/envs/env_*/target",
            reset_xform_properties=False
        )
        self.target_vis.initialize()
        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        self.init_pos_dist = D.Uniform(
            torch.tensor([-1., -1., 0.05], device=self.device),
            torch.tensor([1., 1., 2.0], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-0.2, -0.2, 0.0], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 0.5], device=self.device) * torch.pi
        )
        self.target_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi
        )
        # self.target_rpy_dist = D.Uniform(
        #     torch.tensor([-0.1, -0.1, 0.0], device=self.device) * torch.pi,
        #     torch.tensor([0.1, 0.1, 0.5], device=self.device) * torch.pi
        # )
        
        self.force_disturbance_dist = Normal(0.0, 0.03 * 9.81 / 20)
        self.torques_disturbance_dist = Normal(0.0, 0.03 * 9.81 / 10000)
        
        if self.use_eval:
            self.init_pos_dist = D.Uniform(
                torch.tensor([0.0, 0.0, 0.0], device=self.device),
                torch.tensor([0.0, 0.0, 0.0], device=self.device)
            )
            self.init_rpy_dist = D.Uniform(
                torch.tensor([0.0, 0.0, 0.0], device=self.device) * torch.pi,
                torch.tensor([0.0, 0.0, 0.0], device=self.device) * torch.pi
            )
            self.target_rpy_dist = D.Uniform(
                torch.tensor([0.0, 0.0, 0.0], device=self.device) * torch.pi,
                torch.tensor([0.0, 0.0, 0.0], device=self.device) * torch.pi
            )

        self.target_pos = torch.tensor([[0.0, 0.0, 1.]], device=self.device)
        self.target_heading = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.alpha = 0.8
        
        self.force_disturbance = torch.zeros(self.num_envs, 3, device=self.device)
        self.torques_disturbance = torch.zeros(self.num_envs, 3, device=self.device)

        self.last_linear_v = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_angular_v = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_linear_a = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_angular_a = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_linear_jerk = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_angular_jerk = torch.zeros(self.num_envs, 1, device=self.device)

        self.prev_actions = torch.zeros(self.num_envs, 1, 4, device=self.device)

    def _design_scene(self):
        import omni_drones.utils.kit as kit_utils
        import omni.isaac.core.utils.prims as prim_utils

        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 0.05)])[0]
        if self.has_payload:
            attach_payload(drone_prim.GetPath().pathString)

        target_vis_prim = prim_utils.create_prim(
            prim_path="/World/envs/env_0/target",
            usd_path=self.drone.usd_path,
            translation=(0.0, 0.0, 1.),
        )

        kit_utils.set_nested_collision_properties(
            target_vis_prim.GetPath(), 
            collision_enabled=False
        )
        kit_utils.set_nested_rigid_body_properties(
            target_vis_prim.GetPath(),
            disable_gravity=True
        )

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        # drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = 15
        self.time_encoding_dim = 4
        if self.cfg.task.time_encoding:
            observation_dim += self.time_encoding_dim

        state_dim = observation_dim + 4

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                #"observation": UnboundedContinuousTensorSpec((1, observation_dim-6), device=self.device),   remove throttle
                "observation": UnboundedContinuousTensorSpec((1, observation_dim), device=self.device),
                "state": UnboundedContinuousTensorSpec((state_dim), device=self.device), # add motor speed
                # "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device)
            })
        }).expand(self.num_envs)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec.unsqueeze(0),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1))
            })
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            # state_key=("agents", "intrinsics")
            state_key=("agents", "state"),
        )

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "pos_bonus": UnboundedContinuousTensorSpec(1),
            "head_bonus": UnboundedContinuousTensorSpec(1),
            "reward_pos": UnboundedContinuousTensorSpec(1),
            "reward_head": UnboundedContinuousTensorSpec(1),
            "reward_up": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "pos_error": UnboundedContinuousTensorSpec(1),
            "heading_alignment": UnboundedContinuousTensorSpec(1),
            "action_error_order1_mean": UnboundedContinuousTensorSpec(1),
            "action_error_order1_max": UnboundedContinuousTensorSpec(1),
            "smoothness_mean": UnboundedContinuousTensorSpec(1),
            "smoothness_max": UnboundedContinuousTensorSpec(1),
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
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        
        self.force_disturbance[env_ids] = self.force_disturbance_dist.sample((*env_ids.shape, 3)).to(self.device)
        self.torques_disturbance[env_ids] = self.torques_disturbance_dist.sample((*env_ids.shape, 3)).to(self.device)
        
        pos = self.init_pos_dist.sample((*env_ids.shape, 1))
        rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids].unsqueeze(1), rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)

        if self.has_payload:
            # TODO@btx0424: workout a better way 
            payload_z = self.payload_z_dist.sample(env_ids.shape)
            joint_indices = torch.tensor([self.drone._view._dof_indices["PrismaticJoint"]], device=self.device)
            self.drone._view.set_joint_positions(
                payload_z, env_indices=env_ids, joint_indices=joint_indices)
            self.drone._view.set_joint_position_targets(
                payload_z, env_indices=env_ids, joint_indices=joint_indices)
            self.drone._view.set_joint_velocities(
                torch.zeros(len(env_ids), 1, device=self.device), 
                env_indices=env_ids, joint_indices=joint_indices)
            
            payload_mass = self.payload_mass_dist.sample(env_ids.shape+(1,)) * self.drone.masses[env_ids]
            self.payload.set_masses(payload_mass, env_indices=env_ids)

        target_rpy = self.target_rpy_dist.sample((*env_ids.shape, 1))
        target_rot = euler_to_quaternion(target_rpy)
        self.target_heading[env_ids] = quat_axis(target_rot.squeeze(1), 0).unsqueeze(1)
        self.target_vis.set_world_poses(orientations=target_rot, env_indices=env_ids)

        # set last values
        self.last_linear_v[env_ids] = torch.norm(self.init_vels[env_ids][..., :3], dim=-1)
        self.last_angular_v[env_ids] = torch.norm(self.init_vels[env_ids][..., 3:], dim=-1)
        self.last_linear_a[env_ids] = torch.zeros_like(self.last_linear_v[env_ids])
        self.last_angular_a[env_ids] = torch.zeros_like(self.last_angular_v[env_ids])
        self.last_linear_jerk[env_ids] = torch.zeros_like(self.last_linear_a[env_ids])
        self.last_angular_jerk[env_ids] = torch.zeros_like(self.last_angular_a[env_ids])

        self.stats[env_ids] = 0.

        # init prev_actions: hover
        cmd_init = 2.0 * (self.drone.throttle[env_ids]) ** 2 - 1.0
        self.info['prev_action'][env_ids, :, 3] = cmd_init.mean(dim=-1)
        self.prev_actions[env_ids] = self.info['prev_action'][env_ids].clone()

        self.linear_v_episode = torch.zeros_like(self.stats["linear_v_mean"])
        self.angular_v_episode = torch.zeros_like(self.stats["angular_v_mean"])
        self.linear_a_episode = torch.zeros_like(self.stats["linear_a_mean"])
        self.angular_a_episode = torch.zeros_like(self.stats["angular_a_mean"])
        self.linear_jerk_episode = torch.zeros_like(self.stats["linear_jerk_mean"])
        self.angular_jerk_episode = torch.zeros_like(self.stats["angular_jerk_mean"])

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        if self.cfg.task.action_noise:
            actions *= torch.randn(actions.shape, device=self.device) * 0.1 + 1

        self.info["prev_action"] = tensordict[("info", "prev_action")]
        # self.info["prev_prev_action"] = tensordict[("info", "prev_prev_action")]
        self.prev_actions = self.info["prev_action"].clone()
        # self.prev_prev_actions = self.info["prev_prev_action"].clone()
        
        self.action_error_order1 = tensordict[("stats", "action_error_order1")].clone()
        self.stats["action_error_order1_mean"].add_(self.action_error_order1.mean(dim=-1).unsqueeze(-1))
        self.stats["action_error_order1_max"].set_(torch.max(self.stats["action_error_order1_max"], self.action_error_order1.mean(dim=-1).unsqueeze(-1)))
        # self.action_error_order2 = tensordict[("stats", "action_error_order2")].clone()
        # self.stats["action_error_order2_mean"].add_(self.action_error_order2.mean(dim=-1).unsqueeze(-1))
        # self.stats["action_error_order2_max"].set_(torch.max(self.stats["action_error_order2_max"], self.action_error_order2.mean(dim=-1).unsqueeze(-1)))
      
        if self.use_disturbance:
            disturbance = torch.concat([self.force_disturbance, self.torques_disturbance_dist], dim=-1)
            self.effort = self.drone.apply_action_disturbance(actions, disturbance)
        else:
            self.effort = self.drone.apply_action(actions)
        
    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()
        self.info["drone_state"][:] = self.root_state[..., :13]

        # relative position and heading
        self.rpos = self.target_pos - self.root_state[..., :3]
        self.rheading = self.target_heading - self.root_state[..., 19:22]
        
        obs = [
            self.rpos.flatten(1).unsqueeze(1),
            # self.root_state[..., 3:7], # quat
            self.root_state[..., 7:10], # linear v
            self.root_state[..., 19:28], # rotation
        ]

        obs = torch.cat(obs, dim=-1)

        # add time encoding
        t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
        state = torch.concat([obs, t.expand(-1, self.time_encoding_dim).unsqueeze(1)], dim=-1).squeeze(1)
        
        # linear_v, angular_v
        self.linear_v = torch.norm(self.root_state[..., 7:10], dim=-1)
        self.angular_v = torch.norm(self.root_state[..., 10:13], dim=-1)
        self.stats["linear_v_max"].set_(torch.max(self.stats["linear_v_max"], torch.abs(self.linear_v)))
        self.linear_v_episode.add_(torch.abs(self.linear_v))
        self.stats["linear_v_mean"].set_(self.linear_v_episode / (self.progress_buf + 1.0).unsqueeze(1))
        self.stats["angular_v_max"].set_(torch.max(self.stats["angular_v_max"], torch.abs(self.angular_v)))
        self.angular_v_episode.add_(torch.abs(self.angular_v))
        self.stats["angular_v_mean"].set_(self.angular_v_episode / (self.progress_buf + 1.0).unsqueeze(1))
        # linear_a, angular_a
        self.linear_a = torch.abs(self.linear_v - self.last_linear_v) / self.dt
        self.angular_a = torch.abs(self.angular_v - self.last_angular_v) / self.dt
        self.stats["linear_a_max"].set_(torch.max(self.stats["linear_a_max"], torch.abs(self.linear_a)))
        self.linear_a_episode.add_(torch.abs(self.linear_a))
        self.stats["linear_a_mean"].set_(self.linear_a_episode / (self.progress_buf + 1.0).unsqueeze(1))
        self.stats["angular_a_max"].set_(torch.max(self.stats["angular_a_max"], torch.abs(self.angular_a)))
        self.angular_a_episode.add_(torch.abs(self.angular_a))
        self.stats["angular_a_mean"].set_(self.angular_a_episode / (self.progress_buf + 1.0).unsqueeze(1))
        # linear_jerk, angular_jerk
        self.linear_jerk = torch.abs(self.linear_a - self.last_linear_a) / self.dt
        self.angular_jerk = torch.abs(self.angular_a - self.last_angular_a) / self.dt
        self.stats["linear_jerk_max"].set_(torch.max(self.stats["linear_jerk_max"], torch.abs(self.linear_jerk)))
        self.linear_jerk_episode.add_(torch.abs(self.linear_jerk))
        self.stats["linear_jerk_mean"].set_(self.linear_jerk_episode / (self.progress_buf + 1.0).unsqueeze(1))
        self.stats["angular_jerk_max"].set_(torch.max(self.stats["angular_jerk_max"], torch.abs(self.angular_jerk)))
        self.angular_jerk_episode.add_(torch.abs(self.angular_jerk))
        self.stats["angular_jerk_mean"].set_(self.angular_jerk_episode / (self.progress_buf + 1.0).unsqueeze(1))
        
        # set last
        self.last_linear_v = self.linear_v.clone()
        self.last_angular_v = self.angular_v.clone()
        self.last_linear_a = self.linear_a.clone()
        self.last_angular_a = self.angular_a.clone()
        self.last_linear_jerk = self.linear_jerk.clone()
        self.last_angular_jerk = self.angular_jerk.clone()

        return TensorDict({
            "agents": {
                "observation": obs,
                "state": state,
            },
            "stats": self.stats,
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        # pose reward
        pos_error = torch.norm(self.rpos, dim=-1)
        head_error = torch.norm(self.rheading, dim=-1)
        
        heading_alignment = torch.sum(self.drone.heading * self.target_heading, dim=-1)
        
        # check done
        distance = torch.norm(torch.cat([self.rpos], dim=-1), dim=-1)

        reward_pos = - pos_error * self.reward_distance_scale
        reward_pos_bonus = ((pos_error <= 0.02) * 10).float()
        
        reward_head = - head_error * (reward_pos_bonus > 0)
        reward_head_bonus = ((head_error <= 0.02) * 10 * (reward_pos_bonus > 0)).float()

        # uprightness
        reward_up = torch.square((self.drone.up[..., 2] + 1) / 2)

        # effort
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.action_error_order1)

        self.stats["smoothness_mean"].add_(self.drone.throttle_difference)
        self.stats["smoothness_max"].set_(torch.max(self.drone.throttle_difference, self.stats["smoothness_max"]))
        
        reward = (
            reward_pos
            + reward_pos_bonus
            + reward_head 
            + reward_head_bonus
            + reward_up
            + reward_action_smoothness
        )

        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        )

        ep_len = self.progress_buf.unsqueeze(-1)
        self.stats['action_error_order1_mean'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['smoothness_mean'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )

        self.stats["pos_error"].lerp_(pos_error, (1-self.alpha))
        self.stats["heading_alignment"].lerp_(heading_alignment, (1-self.alpha))
        self.stats["return"] += reward
        # bonus
        self.stats["reward_pos"] = reward_pos
        self.stats["reward_head"] = reward_head
        self.stats["pos_bonus"] = reward_pos_bonus
        self.stats["head_bonus"] = reward_head_bonus
        # self.stats["reward_up"] = reward_up

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
