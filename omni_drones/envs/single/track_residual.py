from functorch import vmap

import omni.isaac.core.utils.torch as torch_utils
import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion, quat_rotate_inverse
import omni.isaac.core.utils.prims as prim_utils
import torch
import torch.distributions as D
import os

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec
from omni.isaac.debug_draw import _debug_draw

from ..utils import lemniscate, lemniscate_v, pentagram, scale_time
from ..utils.chained_polynomial import ChainedPolynomial
from ..utils.zigzag import RandomZigzag
from ..utils.pointed_star import NPointedStar
from ..utils.lemniscate import Lemniscate
import collections
import numpy as np

from omni_drones.controllers import PID_controller_flightmare as _PID_controller_flightmare


class TrackResidual(IsaacEnv):
    def __init__(self, cfg, headless):
        self.reset_thres = cfg.task.reset_thres
        self.eval_no_reset = cfg.task.get("eval_no_reset", False)  # eval 时禁用 distance>reset_thres 的提前 reset
        self.reward_acc_weight_init = cfg.task.reward_acc_weight_init
        self.reward_acc_weight_lr = cfg.task.reward_acc_weight_lr
        self.reward_acc_max = cfg.task.reward_acc_max
        self.reward_jerk_weight_init = cfg.task.reward_jerk_weight_init
        self.reward_jerk_weight_lr = cfg.task.reward_jerk_weight_lr
        self.reward_jerk_max = cfg.task.reward_jerk_max
        self.reward_snap_weight_init = cfg.task.reward_snap_weight_init
        self.reward_snap_weight_lr = cfg.task.reward_snap_weight_lr
        self.reward_snap_max = cfg.task.reward_snap_max

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

        self.wind = cfg.task.wind
        self.use_eval = cfg.task.use_eval
        self.num_drones = 1
        self.use_rotor2critic = cfg.task.use_rotor2critic
        self.action_history_step = cfg.task.action_history_step
        self.reward_spin_weight = cfg.task.reward_spin_weight
        self.reward_up_weight = cfg.task.reward_up_weight
        self.use_ab_wolrd_pos = cfg.task.use_ab_wolrd_pos
        self.eval_traj = cfg.task.eval_traj

        self.sim_data = []
        self.sim_rpy = []
        self.action_data = []

        # residual policy settings
        self.residual_alpha = float(cfg.task.get("residual_alpha", 0.3))
        self.residual_penalty_coef = float(cfg.task.get("residual_penalty_coef", 0.01))
        self.residual_smoothness_coef = float(cfg.task.get("residual_smoothness_coef", 0.01))

        self.delta_c_limit = float(cfg.task.get("delta_c_limit", 0.15))
        self.delta_rate_limit = float(cfg.task.get("delta_rate_limit", 0.20))
        self.comp_rpos_steps = int(cfg.task.get("comp_rpos_steps", 6))

        super().__init__(cfg, headless)

        self.drone.initialize()

        # ========================================================
        # 关键修复 1：
        # total_mass 统一为 [num_envs, 1]
        # get_body_masses() 在 Isaac Sim 中返回 [num_envs, num_drones, num_links] (3D)
        # 用 reshape(num_envs, -1) 先展平，再 sum(-1, keepdim=True) 得到 [num_envs, 1]
        # ========================================================
        self.total_mass = self.drone._view.get_body_masses().reshape(self.num_envs, -1).sum(-1, keepdim=True)  # [num_envs, 1]

        # 创建 PID 控制器，用于将 CTBR 转换为电机指令
        self.pid_controller = _PID_controller_flightmare(
            self.cfg.sim.dt, self.drone.params, self.device
        ).to(self.device)
        self.pid_reset_flag = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        print("===== TrackResidual mass check =====")
        print("air.yaml mass -> self.drone.mass:", self.drone.mass)
        print("self.drone.MASS_0:", self.drone.MASS_0)
        print("self.drone.masses[0]:", self.drone.masses[0])
        print("base_link.get_masses()[0]:", self.drone.base_link.get_masses()[0])
        print("body masses sum env0:", self.drone._view.get_body_masses()[0].sum())
        print("self.total_mass shape:", self.total_mass.shape)
        print("self.total_mass[0]:", self.total_mass[0])
        print("===================================")

        randomization = self.cfg.task.get("randomization", None)
        if randomization is not None:
            if "drone" in self.cfg.task.randomization:
                self.drone.setup_randomization(self.cfg.task.randomization["drone"])

        if self.wind:
            # [2026-05-06 重构] wind 作为总开关，disturbance.mode 决定类型（sinsum/composite），enable 参数已移除
            # 原版备份：omni_drones/envs/single/track_residual_backup_20260506.py
            # 旧代码（sinsum 独立初始化）：
            # if randomization is not None and "wind" in self.cfg.task.randomization:
            #     cfg_wind = self.cfg.task.randomization["wind"]
            #     wind_intensity_scale = cfg_wind["train"].get("intensity", None)
            #     self.wind_intensity_low = wind_intensity_scale[0]
            #     self.wind_intensity_high = wind_intensity_scale[1]
            # else:
            #     self.wind_intensity_low = 0
            #     self.wind_intensity_high = 2
            # self.wind_w = torch.zeros(self.num_envs, 3, 8, device=self.device)
            # self.wind_i = torch.zeros(self.num_envs, 1, device=self.device)
            # 旧代码（独立 disturbance_enable 块）：
            # dist_cfg = cfg.task.get("disturbance", None)
            # self.disturbance_enable = (dist_cfg is not None) and dist_cfg.get("enable", False)
            # if self.disturbance_enable:
            dist_cfg = self.cfg.task.get("disturbance", {}) or {}
            self.dist_cfg = dist_cfg
            self.dist_mode = dist_cfg.get("mode", "sinsum")
            self.dist_horizontal_only = dist_cfg.get("horizontal_only", True)
            self.dist_clip_total = dist_cfg.get("clip_total", True)
            self.dist_max_acc_train = dist_cfg.get("max_total_acc_train", 3.0)
            self.dist_max_acc_eval = dist_cfg.get("max_total_acc_eval", 3.5)
            self.dist_max_acc = self.dist_max_acc_train if not self.use_eval else self.dist_max_acc_eval

            # 初始化 sinsum 模式所需的 buffer
            if self.dist_mode == "sinsum":
                if randomization is not None and "wind" in self.cfg.task.randomization:
                    cfg_wind = self.cfg.task.randomization["wind"]
                    wind_intensity_scale = cfg_wind["train"].get("intensity", None)
                    self.dist_intensity_low = wind_intensity_scale[0]
                    self.dist_intensity_high = wind_intensity_scale[1]
                elif dist_cfg.get("sinsum", {}).get("intensity_range"):
                    intensity_range = dist_cfg["sinsum"]["intensity_range"]
                    self.dist_intensity_low = intensity_range[0]
                    self.dist_intensity_high = intensity_range[1]
                else:
                    self.dist_intensity_low = 0.0
                    self.dist_intensity_high = 2.0
                self.dist_num_freqs = dist_cfg.get("sinsum", {}).get("num_freqs", 8)
                self.wind_w = torch.zeros(self.num_envs, 3, self.dist_num_freqs, device=self.device)
                self.wind_i = torch.zeros(self.num_envs, 1, device=self.device)

            # 初始化 composite 模式所需的 buffer
            elif self.dist_mode == "composite":
                self.dist_bias = torch.zeros(self.num_envs, 3, device=self.device)
                self.dist_gm = torch.zeros(self.num_envs, 3, device=self.device)
                self.dist_gm_sigma = torch.zeros(self.num_envs, 1, device=self.device)
                self.dist_gm_tau = torch.zeros(self.num_envs, 1, device=self.device)
                self.dist_gm_alpha = torch.zeros(self.num_envs, 1, device=self.device)
                self.dist_swing_amp = torch.zeros(self.num_envs, 3, device=self.device)
                self.dist_swing_freq = torch.zeros(self.num_envs, 1, device=self.device)
                self.dist_swing_phase = torch.zeros(self.num_envs, 3, device=self.device)

            self.dist_acc = torch.zeros(self.num_envs, 3, device=self.device)
            self.dist_force = torch.zeros(self.num_envs, 3, device=self.device)

        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )

        self.origin = torch.tensor([0., 0., 1.], device=self.device)
        self.ref = [
            ChainedPolynomial(
                num_trajs=self.num_envs,
                scale=2.5,
                use_y=True,
                min_dt=1.5,
                max_dt=4.0,
                degree=5,
                origin=self.origin,
                device=self.device
            ),
            RandomZigzag(
                num_trajs=self.num_envs,
                max_D=[1.0, 1.0, 0.0],
                min_dt=1.0,
                max_dt=1.5,
                diff_axis=True,
                origin=self.origin,
                device=self.device
            )
        ]
        self.ref_style_seq = torch.randint(0, 2, (self.num_envs,), device=self.device)
        self.traj_t0 = torch.zeros(self.num_envs, 1, device=self.device)

        if self.use_eval:
            self.init_rpy_dist = D.Uniform(
                torch.tensor([-.0, -.0, 0.], device=self.device) * torch.pi,
                torch.tensor([0., 0., 0.], device=self.device) * torch.pi
            )
            if self.eval_traj == "poly":
                self.ref = ChainedPolynomial(
                    num_trajs=self.num_envs,
                    scale=2.5,
                    use_y=True,
                    min_dt=1.5,
                    max_dt=4.0,
                    degree=5,
                    origin=self.origin,
                    device=self.device
                )
            elif self.eval_traj == "zigzag":
                self.ref = RandomZigzag(
                    num_trajs=self.num_envs,
                    max_D=[1.0, 1.0, 0.0],
                    min_dt=1.0,
                    max_dt=1.5,
                    diff_axis=True,
                    origin=self.origin,
                    device=self.device
                )
            elif self.eval_traj == "pentagram":
                self.ref = NPointedStar(
                    num_trajs=self.num_envs,
                    num_points=5,
                    origin=self.origin,
                    speed=1.0,
                    radius=0.7,
                    device=self.device
                )
            elif self.eval_traj == "slow":
                self.ref = Lemniscate(T=15.0, origin=self.origin, device=self.device)
                self.traj_t0 = torch.ones(self.num_envs, 1, device=self.device) * 15.0 / 4
            elif self.eval_traj == "normal":
                self.ref = Lemniscate(T=5.5, origin=self.origin, device=self.device)
                self.traj_t0 = torch.ones(self.num_envs, 1, device=self.device) * 5.5 / 4
            elif self.eval_traj == "fast":
                self.ref = Lemniscate(T=3.5, origin=self.origin, device=self.device)
                self.traj_t0 = torch.ones(self.num_envs, 1, device=self.device) * 3.5 / 4

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
        self.count = 0

        # residual-specific buffers
        self.base_actions = torch.zeros(self.num_envs, self.num_drones, 4, device=self.device)
        self.comp_actions = torch.zeros(self.num_envs, self.num_drones, 4, device=self.device)
        self.prev_comp_actions = torch.zeros(self.num_envs, self.num_drones, 4, device=self.device)
        self.final_actions = torch.zeros(self.num_envs, self.num_drones, 4, device=self.device)

        self.pred_disturbance = torch.zeros(self.num_envs, 3, device=self.device)
        self.gt_disturbance = torch.zeros(self.num_envs, 3, device=self.device)
        self.wind_accel_buf = torch.zeros(self.num_envs, 3, device=self.device)
        self.expose_gt_disturbance = bool(cfg.task.get("expose_gt_disturbance", False))
        self.v_body = torch.zeros(self.num_envs, 3, device=self.device)
        self.w_body = torch.zeros(self.num_envs, 3, device=self.device)
        self.R_flat = torch.zeros(self.num_envs, 9, device=self.device)
        self.rpos0 = torch.zeros(self.num_envs, 3, device=self.device)
        self.vel_error_body = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_distance = torch.zeros(self.num_envs, 1, device=self.device)

    def _apply_eval_traj(self):
        """切换评估轨迹类型，重建 self.ref。在设置 eval_traj 后调用。"""
        self.use_eval = True
        # 切换轨迹时先清掉上一条轨迹的时间相位，避免 poly/zigzag 继承
        # slow/normal/fast 的 lemniscate 相位偏移。
        self.traj_t0 = torch.zeros(self.num_envs, 1, device=self.device)
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.0, -.0, 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi
        )
        if self.eval_traj == "poly":
            self.ref = ChainedPolynomial(
                num_trajs=self.num_envs,
                scale=2.5,
                use_y=True,
                min_dt=1.5,
                max_dt=4.0,
                degree=5,
                origin=self.origin,
                device=self.device
            )
        elif self.eval_traj == "zigzag":
            self.ref = RandomZigzag(
                num_trajs=self.num_envs,
                max_D=[1.0, 1.0, 0.0],
                min_dt=1.0,
                max_dt=1.5,
                diff_axis=True,
                origin=self.origin,
                device=self.device
            )
        elif self.eval_traj == "pentagram":
            self.ref = NPointedStar(
                num_trajs=self.num_envs,
                num_points=5,
                origin=self.origin,
                speed=1.0,
                radius=0.7,
                device=self.device
            )
        elif self.eval_traj == "slow":
            self.ref = Lemniscate(T=15.0, origin=self.origin, device=self.device)
            self.traj_t0 = torch.ones(self.num_envs, 1, device=self.device) * 15.0 / 4
        elif self.eval_traj == "normal":
            self.ref = Lemniscate(T=5.5, origin=self.origin, device=self.device)
            self.traj_t0 = torch.ones(self.num_envs, 1, device=self.device) * 5.5 / 4
        elif self.eval_traj == "fast":
            self.ref = Lemniscate(T=3.5, origin=self.origin, device=self.device)
            self.traj_t0 = torch.ones(self.num_envs, 1, device=self.device) * 3.5 / 4
        else:
            raise ValueError(f"Unknown eval_traj: {self.eval_traj}")

    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        if self.use_local_usd:
            usd_path = os.path.join(
                os.path.dirname(__file__), os.pardir, "assets", "default_environment.usd"
            )
            kit_utils.create_ground_plane(
                "/World/defaultGroundPlane",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                usd_path=usd_path
            )
        else:
            kit_utils.create_ground_plane(
                "/World/defaultGroundPlane",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )

        self.drone.spawn(translations=[(0.0, 0.0, 1.5)])
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        # =========================================================
        # 1. 先定义“冻结主策略”原本需要的 observation/state 维度
        #    这部分严格沿用原 Track 的定义
        # =========================================================
        if self.use_ab_wolrd_pos:
            base_drone_state_dim = 3 + 3 + 3 + 3 + 3 + 3
        else:
            base_drone_state_dim = 3 + 3 + 3 + 3

        self.time_encoding_dim = self.cfg.task.time_encoding_dim
        self.base_obs_dim = base_drone_state_dim + 3 * self.future_traj_steps
        if self.time_encoding:
            self.base_obs_dim += self.time_encoding_dim

        self.action_history = self.cfg.task.action_history_step if self.cfg.task.use_action_history else 0
        self.action_history_buffer = collections.deque(maxlen=self.action_history)

        if self.action_history > 0:
            self.base_obs_dim += self.action_history * 4

        if self.time_encoding:
            self.base_state_dim = self.base_obs_dim
        else:
            self.base_state_dim = self.base_obs_dim + self.time_encoding_dim

        # =========================================================
        # 2. residual policy observation 维度
        # =========================================================
        # P0: rpos in body frame; P1: comp_rpos_steps steps; P2: vel_error_body(3)
        # [a_hat(3)+u_base(4)+v_body(3)+w_body(3)+R_flat(9)+rpos_body(S*3)+prev_rpos0_body(3)+vel_error_body(3)]
        self.comp_obs_dim = 3 + 4 + 3 + 3 + 9 + self.comp_rpos_steps * 3 + 3 + 3

        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": UnboundedContinuousTensorSpec((1, self.comp_obs_dim)),
                "state": UnboundedContinuousTensorSpec((self.comp_obs_dim,)),
            }
        }).expand(self.num_envs).to(self.device)

        self.action_spec = CompositeSpec({
            "agents": {
                "action": UnboundedContinuousTensorSpec((1, 4)),
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
            "return_pure": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "tracking_error": UnboundedContinuousTensorSpec(1),
            "tracking_error_ema": UnboundedContinuousTensorSpec(1),
            "tracking_error_max": UnboundedContinuousTensorSpec(1),
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
            "reward_action_norm": UnboundedContinuousTensorSpec(1),
            "reward_acc": UnboundedContinuousTensorSpec(1),
            "reward_jerk": UnboundedContinuousTensorSpec(1),
            "reward_action_smoothness_scale": UnboundedContinuousTensorSpec(1),
            "reward_action_norm_scale": UnboundedContinuousTensorSpec(1),
            "linear_v_max": UnboundedContinuousTensorSpec(1),
            "angular_v_max": UnboundedContinuousTensorSpec(1),
            "linear_a_max": UnboundedContinuousTensorSpec(1),
            "angular_a_max": UnboundedContinuousTensorSpec(1),
            "linear_jerk_max": UnboundedContinuousTensorSpec(1),
            "angular_jerk_max": UnboundedContinuousTensorSpec(1),
            "linear_snap_max": UnboundedContinuousTensorSpec(1),
            "linear_v_mean": UnboundedContinuousTensorSpec(1),
            "angular_v_mean": UnboundedContinuousTensorSpec(1),
            "linear_a_mean": UnboundedContinuousTensorSpec(1),
            "angular_a_mean": UnboundedContinuousTensorSpec(1),
            "linear_jerk_mean": UnboundedContinuousTensorSpec(1),
            "angular_jerk_mean": UnboundedContinuousTensorSpec(1),
            "linear_snap_mean": UnboundedContinuousTensorSpec(1),
            "obs_range": UnboundedContinuousTensorSpec(1),
            "residual_norm_max": UnboundedContinuousTensorSpec(1),
            "residual_norm_mean": UnboundedContinuousTensorSpec(1),
            "residual_smoothness_max": UnboundedContinuousTensorSpec(1),
            "residual_smoothness_mean": UnboundedContinuousTensorSpec(1),
            "reward_tracking_delta": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)

        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
            "prev_action": UnboundedContinuousTensorSpec((self.drone.n, 4), device=self.device),
            "policy_action": UnboundedContinuousTensorSpec((self.drone.n, 4), device=self.device),

            "base_action": UnboundedContinuousTensorSpec((self.drone.n, 4), device=self.device),
            "comp_action": UnboundedContinuousTensorSpec((self.drone.n, 4), device=self.device),
            "final_action": UnboundedContinuousTensorSpec((self.drone.n, 4), device=self.device),
            "prev_comp_action": UnboundedContinuousTensorSpec((self.drone.n, 4), device=self.device),

            "pred_disturbance": UnboundedContinuousTensorSpec((3,), device=self.device),
            "gt_disturbance": UnboundedContinuousTensorSpec((3,), device=self.device),
            "v_body": UnboundedContinuousTensorSpec((3,), device=self.device),
            "w_body": UnboundedContinuousTensorSpec((3,), device=self.device),
            "R_flat": UnboundedContinuousTensorSpec((9,), device=self.device),
            "rpos0": UnboundedContinuousTensorSpec((3,), device=self.device),
            "rpos_steps": UnboundedContinuousTensorSpec((self.comp_rpos_steps, 3), device=self.device),
            "prev_rpos0": UnboundedContinuousTensorSpec((3,), device=self.device),
            "vel_error_body": UnboundedContinuousTensorSpec((3,), device=self.device),
            "target_vel": UnboundedContinuousTensorSpec((3,), device=self.device),

            "base_obs": UnboundedContinuousTensorSpec((self.drone.n, self.base_obs_dim), device=self.device),
            "base_state": UnboundedContinuousTensorSpec((self.base_state_dim,), device=self.device),

            # 扰动模型相关的信息
            "disturbance_acc": UnboundedContinuousTensorSpec((3,), device=self.device),
            "disturbance_bias": UnboundedContinuousTensorSpec((3,), device=self.device),
            "disturbance_gm": UnboundedContinuousTensorSpec((3,), device=self.device),
            "disturbance_swing": UnboundedContinuousTensorSpec((3,), device=self.device),
        }).expand(self.num_envs).to(self.device)

        self.observation_spec["info"] = info_spec
        self.observation_spec["stats"] = stats_spec

        self.info = info_spec.zero()
        self.stats = stats_spec.zero()

        self.random_latency = self.cfg.task.random_latency
        self.latency = self.cfg.task.latency_step if self.cfg.task.latency else 0
        self.root_state_buffer = collections.deque(maxlen=self.latency + 1)

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)

        if not self.use_eval:
            self.ref[0].reset(env_ids)
            self.ref[1].reset(env_ids)
            self.ref_style_seq[env_ids] = torch.randint(0, 2, (len(env_ids),), device=self.device)

        if self.use_eval:
            self.ref.reset(env_ids)

        pos = torch.zeros(len(env_ids), 3, device=self.device) + self.origin
        rot = euler_to_quaternion(self.init_rpy_dist.sample(env_ids.shape))
        vel = torch.zeros(len(env_ids), 1, 6, device=self.device)

        self.drone.set_world_poses(pos + self.envs_positions[env_ids], rot, env_ids)
        self.drone.set_velocities(vel, env_ids)

        self.last_linear_v[env_ids] = torch.norm(vel[..., :3], dim=-1)
        self.last_angular_v[env_ids] = torch.norm(vel[..., 3:], dim=-1)
        self.last_linear_a[env_ids] = torch.zeros_like(self.last_linear_v[env_ids])
        self.last_angular_a[env_ids] = torch.zeros_like(self.last_angular_v[env_ids])
        self.last_linear_jerk[env_ids] = torch.zeros_like(self.last_linear_a[env_ids])
        self.last_angular_jerk[env_ids] = torch.zeros_like(self.last_angular_a[env_ids])

        self.stats[env_ids] = 0.

        cmd_init = 2.0 * (self.drone.throttle[env_ids]) ** 2 - 1.0
        self.info["prev_action"][env_ids, :, 3] = cmd_init.mean(dim=-1)
        self.prev_actions[env_ids] = self.info["prev_action"][env_ids].clone()

        self.base_actions[env_ids] = 0.
        self.comp_actions[env_ids] = 0.
        self.prev_comp_actions[env_ids] = 0.
        self.final_actions[env_ids] = 0.
        self.pred_disturbance[env_ids] = 0.
        self.gt_disturbance[env_ids] = 0.
        self.v_body[env_ids] = 0.
        self.w_body[env_ids] = 0.
        self.R_flat[env_ids] = 0.
        self.rpos0[env_ids] = 0.
        self.vel_error_body[env_ids] = 0.
        self.last_distance[env_ids] = 0.

        if self._should_render(0) and (env_ids == self.central_env_idx).any():
            self.draw.clear_lines()
            traj_vis = self._compute_traj(self.max_episode_length, self.central_env_idx.unsqueeze(0))[0]
            traj_vis = traj_vis + self.envs_positions[self.central_env_idx]
            point_list_0 = traj_vis[:-1].tolist()
            point_list_1 = traj_vis[1:].tolist()
            colors = [(1.0, 1.0, 1.0, 1.0) for _ in range(len(point_list_0))]
            sizes = [1 for _ in range(len(point_list_0))]
            self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)

        self.pid_reset_flag[env_ids] = True

        # [2026-05-06 重构] 统一通过 _reset_disturbance 处理所有扰动 reset（sinsum/composite）
        # 旧代码（旧 sinsum 直接 reset）：
        # if self.wind:
        #     self.wind_i[env_ids] = (
        #         torch.rand(*env_ids.shape, 1, device=self.device)
        #         * (self.wind_intensity_high - self.wind_intensity_low)
        #         + self.wind_intensity_low
        #     )
        #     self.wind_w[env_ids] = torch.randn(*env_ids.shape, 3, 8, device=self.device)
        # if self.disturbance_enable:
        #     self._reset_disturbance(env_ids)
        if self.wind:
            self._reset_disturbance(env_ids)

    def _limit_comp_action(self, comp_action: torch.Tensor) -> torch.Tensor:
        # With tanh=true policy, comp_action is already in (-1, 1).
        # residual_alpha scales the effective range: max compensation = residual_alpha * 1.0
        # Hard per-dimension clamp removed to avoid PPO gradient dead-zones.
        return comp_action

    def _pre_sim_step(self, tensordict: TensorDictBase):
        comp_action = tensordict[("agents", "action")]
        base_action = tensordict[("info", "base_action")]  # CTBR [-1,1] (tanh 已在 composed_policy 中完成)
        pred_disturbance = tensordict[("info", "pred_disturbance")]

        comp_action = self._limit_comp_action(comp_action)

        # 合成最终 CTBR = 基础 CTBR + alpha * 残差补偿
        final_ctbr = base_action + self.residual_alpha * comp_action
        final_ctbr = torch.clamp(final_ctbr, -1.0, 1.0)

        self.base_actions = base_action.clone()
        self.comp_actions = comp_action.clone()
        self.final_actions = final_ctbr.clone()

        self.info["base_action"] = self.base_actions.clone()
        self.info["comp_action"] = self.comp_actions.clone()
        self.info["final_action"] = self.final_actions.clone()
        self.info["prev_comp_action"] = self.prev_comp_actions.clone()
        self.info["pred_disturbance"] = pred_disturbance.clone()

        # CTBR 级别的 policy_action (用于 reward 计算)
        self.policy_actions = final_ctbr.clone()

        # 先计算 action smoothness (当前 CTBR vs 上一步 CTBR)
        self.action_error_order1 = (final_ctbr - self.prev_actions).abs()
        self.stats["action_error_order1_mean"].add_(self.action_error_order1.mean(dim=-1))
        self.stats["action_error_order1_max"].set_(
            torch.max(self.stats["action_error_order1_max"], self.action_error_order1.mean(dim=-1))
        )

        # 更新 prev_action 为当前 CTBR（供下一步使用）
        self.info["prev_action"] = final_ctbr.clone()
        self.info["policy_action"] = final_ctbr.clone()
        self.prev_actions = final_ctbr.clone()

        residual_norm = torch.norm(self.comp_actions, dim=-1)
        residual_smooth = torch.norm(self.comp_actions - self.prev_comp_actions, dim=-1)

        self.stats["residual_norm_mean"].add_(residual_norm)
        self.stats["residual_norm_max"].set_(torch.max(self.stats["residual_norm_max"], residual_norm))
        self.stats["residual_smoothness_mean"].add_(residual_smooth)
        self.stats["residual_smoothness_max"].set_(
            torch.max(self.stats["residual_smoothness_max"], residual_smooth)
        )

        # ========================================================
        # 将 CTBR 通过 PID 控制器转换为电机指令
        # 与 PIDRateController_flightmare 的处理一致：
        #   rate  部分: [-1,1] * pi -> [-pi, pi] rad/s
        #   thrust部分: [-1,1] -> (x+1)/2 * 15.0 -> [0, 15] m/s^2
        # ========================================================
        target_rate = final_ctbr[..., :3] * torch.pi
        target_thrust = (final_ctbr[..., 3:] + 1.0) / 2.0 * 15.0

        drone_state = tensordict[("info", "drone_state")][..., :13]
        reset_pid = self.pid_reset_flag.unsqueeze(-1).expand(-1, drone_state.shape[-2])

        motor_cmds = self.pid_controller(
            drone_state,
            target_rate=target_rate,
            target_thrust=target_thrust,
            reset_pid=reset_pid,
        )
        torch.nan_to_num_(motor_cmds, 0.0)
        self.pid_reset_flag[:] = False

        self.effort = self.drone.apply_action(motor_cmds)

        # [2026-05-06 重构] 统一通过 _update_and_apply_disturbance 处理所有扰动施力（sinsum/composite）
        # 旧代码（旧 sinsum 内联施力）：
        # if self.wind:
        #     t = (self.progress_buf * self.dt).reshape(self.num_envs, 1, 1)
        #     wind_accel = self.wind_i * torch.sin(t * self.wind_w).sum(-1)
        #     mass = self.drone._view.get_body_masses().reshape(self.num_envs, -1).sum(-1, keepdim=True)
        #     wind_forces = (mass * wind_accel).view(self.num_envs, 1, 3)
        #     self.drone.base_link.apply_forces(wind_forces, is_global=True)
        #     if self.expose_gt_disturbance:
        #         self.wind_accel_buf[:] = wind_accel
        # if self.disturbance_enable:
        #     self._update_and_apply_disturbance()
        if self.wind:
            self._update_and_apply_disturbance()

        # ----------------------------------------------------------------
        # GT 扰动真值计算（机体系，与 PINN 输出坐标系一致）
        # drag: F_drag = -drag_coef * v * |v|  =>  a_drag = F_drag / mass
        # wind: a_wind 已在上方 wind_accel 中计算（世界系）
        # 转机体系: quat_rotate_inverse(rot, a_world)
        # 仅当 expose_gt_disturbance=True 时执行（对应 disturbance_input=gt 模式）
        # ----------------------------------------------------------------
        if self.expose_gt_disturbance:
            vel_w = self.drone.vel[..., :3]                            # [N, 1, 3] 世界系
            vel_n = torch.norm(vel_w, dim=-1, keepdim=True)            # [N, 1, 1]
            mass_exp = self.total_mass.unsqueeze(1)                    # [N, 1, 1]
            drag_acc_world = -(self.drone.drag_coef * vel_w * vel_n) / mass_exp  # [N, 1, 3]
            # [2026-05-06 重构] 外部加速度源统一通过 wind 开关控制，dist_acc 在 _update_and_apply_disturbance 中已更新
            # 旧代码: if self.disturbance_enable: ... elif self.wind: ... else: ...
            if self.wind:
                ext_acc_world = self.dist_acc.unsqueeze(1)             # [N, 1, 3]
            else:
                ext_acc_world = torch.zeros(self.num_envs, 1, 3, device=self.device)
            total_dist_world = ext_acc_world + drag_acc_world          # [N, 1, 3]
            gt_dist_body = quat_rotate_inverse(self.drone.rot, total_dist_world)   # [N, 1, 3]
            self.gt_disturbance[:] = gt_dist_body.squeeze(1)          # [N, 3]
            self.info["gt_disturbance"] = self.gt_disturbance.clone()


    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()
        self.info["drone_state"][:] = self.root_state[..., :13]

        if self.cfg.task.latency:
            self.root_state_buffer.append(self.root_state)
            if self.random_latency:
                random_indices = torch.randint(
                    0, len(self.root_state_buffer), (self.num_envs,), device=self.device
                )
                root_state = torch.stack(list(self.root_state_buffer))[random_indices, torch.arange(self.num_envs)]
            else:
                root_state = self.root_state_buffer[0]
        else:
            root_state = self.root_state

        self.target_pos[:] = self._compute_traj(self.future_traj_steps, step_size=5)
        self.rpos = self.target_pos - root_state[..., :3]
        prev_rpos0 = self.rpos0.clone()  # 保存上一步rpos0，提供速度误差信息
        self.rpos0 = self.rpos[:, 0, :].clone()

        if self.use_ab_wolrd_pos:
            base_obs_list = [
                root_state[..., :3],
                self.rpos.flatten(1).unsqueeze(1),
                root_state[..., 7:10],
                root_state[..., 16:19],
                root_state[..., 19:28],
            ]
        else:
            base_obs_list = [
                self.rpos.flatten(1).unsqueeze(1),
                root_state[..., 7:10],
                root_state[..., 19:28],
            ]

        if self.time_encoding:
            t_enc = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            base_obs_list.append(t_enc.expand(-1, self.time_encoding_dim).unsqueeze(1))

        base_obs = torch.cat(base_obs_list, dim=-1)

        t_enc = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
        if self.time_encoding:
            base_state = base_obs.squeeze(1)
        else:
            base_state = torch.cat(
                [base_obs, t_enc.expand(-1, self.time_encoding_dim).unsqueeze(1)],
                dim=-1
            ).squeeze(1)

        if self.action_history > 0:
            self.action_history_buffer.append(self.prev_actions)
            all_action_history = torch.cat(list(self.action_history_buffer), dim=-1)
            base_obs = torch.cat([base_obs, all_action_history], dim=-1)

        self.info["base_obs"][:] = base_obs
        self.info["base_state"][:] = base_state

        self.stats["drone_state"] = root_state[..., :13].squeeze(1).clone()

        self.stats["smoothness_mean"].add_(self.drone.throttle_difference)
        self.stats["smoothness_max"].set_(torch.max(self.drone.throttle_difference, self.stats["smoothness_max"]))

        self.linear_v = torch.norm(self.root_state[..., 7:10], dim=-1)
        self.angular_v = torch.norm(self.root_state[..., 10:13], dim=-1)
        self.stats["linear_v_max"].set_(torch.max(self.stats["linear_v_max"], torch.abs(self.linear_v)))
        self.stats["linear_v_mean"].add_(self.linear_v)
        self.stats["angular_v_max"].set_(torch.max(self.stats["angular_v_max"], torch.abs(self.angular_v)))
        self.stats["angular_v_mean"].add_(self.angular_v)

        self.linear_a = torch.abs(self.linear_v - self.last_linear_v) / self.dt
        self.angular_a = torch.abs(self.angular_v - self.last_angular_v) / self.dt
        self.stats["linear_a_max"].set_(torch.max(self.stats["linear_a_max"], torch.abs(self.linear_a)))
        self.stats["linear_a_mean"].add_(self.linear_a)
        self.stats["angular_a_max"].set_(torch.max(self.stats["angular_a_max"], torch.abs(self.angular_a)))
        self.stats["angular_a_mean"].add_(self.angular_a)

        self.linear_jerk = torch.abs(self.linear_a - self.last_linear_a) / self.dt
        self.angular_jerk = torch.abs(self.angular_a - self.last_angular_a) / self.dt
        self.stats["linear_jerk_max"].set_(torch.max(self.stats["linear_jerk_max"], torch.abs(self.linear_jerk)))
        self.stats["linear_jerk_mean"].add_(self.linear_jerk)
        self.stats["angular_jerk_max"].set_(torch.max(self.stats["angular_jerk_max"], torch.abs(self.angular_jerk)))
        self.stats["angular_jerk_mean"].add_(self.angular_jerk)

        self.linear_snap = torch.abs(self.linear_jerk - self.last_linear_jerk) / self.dt
        self.stats["linear_snap_max"].set_(torch.max(self.stats["linear_snap_max"], torch.abs(self.linear_snap)))
        self.stats["linear_snap_mean"].add_(self.linear_snap)

        self.last_linear_v = self.linear_v.clone()
        self.last_angular_v = self.angular_v.clone()
        self.last_linear_a = self.linear_a.clone()
        self.last_angular_a = self.angular_a.clone()
        self.last_linear_jerk = self.linear_jerk.clone()
        self.last_angular_jerk = self.angular_jerk.clone()

        self.v_body = self.drone.vel_b[..., 0, 0:3].clone()
        self.w_body = self.drone.vel_b[..., 0, 3:6].clone()
        self.R_flat = root_state[..., 19:28].squeeze(1).clone()

        self.info["v_body"][:] = self.v_body
        self.info["w_body"][:] = self.w_body
        self.info["R_flat"][:] = self.R_flat
        self.info["rpos0"][:] = self.rpos0

        # --- P0: world -> body frame for rpos ---
        R_mat = self.R_flat.reshape(self.num_envs, 3, 3)  # [N, 3, 3]
        # P1: use comp_rpos_steps steps
        rpos_world = self.rpos[:, 0:self.comp_rpos_steps, :]  # [N, S, 3]
        # R_mat columns are body axes in world frame  =>  body = R^T @ world
        rpos_body = torch.einsum('nij,nsj->nsi', R_mat, rpos_world)  # [N, S, 3]
        self.info["rpos_steps"][:] = rpos_body.clone()

        # prev_rpos0 also to body frame
        prev_rpos0_body = torch.einsum('nij,nj->ni', R_mat, prev_rpos0)  # [N, 3]
        self.info["prev_rpos0"][:] = prev_rpos0_body

        # --- P2: velocity error in body frame ---
        # target velocity approximation: (rpos0 - prev_rpos0) / dt is noisy
        # use finite difference on target trajectory instead
        target_vel_world = (self.target_pos[:, 1, :] - self.target_pos[:, 0, :]) / (5.0 * self.dt)  # [N, 3]
        current_vel_world = root_state[..., 7:10].squeeze(1)  # [N, 3]
        vel_error_world = target_vel_world - current_vel_world  # [N, 3]
        vel_error_body = torch.einsum('nij,nj->ni', R_mat, vel_error_world)  # [N, 3]
        self.vel_error_body = vel_error_body
        self.info["vel_error_body"][:] = vel_error_body
        self.info["target_vel"][:] = target_vel_world

        # --- Build comp_obs ---
        rpos_body_flat = rpos_body.flatten(1).unsqueeze(1)  # [N, 1, S*3]
        comp_obs = torch.cat([
            self.info["pred_disturbance"].unsqueeze(1),           # a_hat(3)
            self.info["base_action"],                              # u_base(4)
            self.v_body.unsqueeze(1),                              # v_body(3)
            self.w_body.unsqueeze(1),                              # w_body(3)
            self.R_flat.unsqueeze(1),                              # R_flat(9)
            rpos_body_flat,                                        # rpos_body(S*3)
            prev_rpos0_body.unsqueeze(1),                          # prev_rpos0_body(3)
            vel_error_body.unsqueeze(1),                           # vel_error_body(3)
        ], dim=-1)  # total = 3+4+3+3+9+S*3+3+3

        comp_state = comp_obs.squeeze(1)

        self.stats["obs_range"].set_(torch.max(torch.abs(comp_obs), dim=-1).values)

        if self.use_eval:
            self.sim_data.append(comp_obs[0].clone())
            self.sim_rpy.append(self.drone.vel_b[0, :, 3:].clone())

        return TensorDict({
            "agents": {
                "observation": comp_obs,
                "state": comp_state,
            },
            "stats": self.stats,
            "info": self.info,
        }, self.batch_size)

    def _compute_reward_and_done(self):
        distance = torch.norm(self.rpos[:, [0]], dim=-1)
        self.stats["tracking_error"].add_(-distance)
        self.stats["tracking_error_ema"].lerp_(distance, (1 - self.alpha))
        self.stats["tracking_error_max"].set_(
            torch.max(self.stats["tracking_error_max"], distance)
        )

        reward_pos = self.reward_distance_scale * torch.exp(-distance)

        tiltage = torch.abs(1 - self.drone.up[..., 2])
        reward_up = self.reward_up_weight * 0.5 / (1.0 + torch.square(tiltage))

        self.reward_action_norm_weight = min(
            self.reward_action_norm_weight_init + self.reward_action_norm_weight_lr * self.count,
            self.reward_norm_max
        )
        # 修复：使用base_actions避免comp_action被双重惩罚
        reward_action_norm = self.reward_action_norm_weight * torch.exp(-torch.norm(self.base_actions, dim=-1))

        self.reward_action_smoothness_weight = min(
            self.reward_action_smoothness_weight_init + self.reward_action_smoothness_weight_lr * self.count,
            self.reward_smoothness_max
        )

        # ========================================================
        # 关键修复 3：
        # self.action_error_order1: [num_envs, 1, 4]
        # 先在动作维上取均值 -> [num_envs, 1]
        # not_begin_flag 也保持 [num_envs, 1]
        # 最终 reward_action_smoothness 必须是 [num_envs, 1]
        # ========================================================
        action_error_order1_mean = self.action_error_order1.mean(dim=-1)
        not_begin_flag = (self.progress_buf > 1).unsqueeze(-1).float()
        reward_action_smoothness = (
            self.reward_action_smoothness_weight
            * torch.exp(-action_error_order1_mean)
            * not_begin_flag
        )

        self.reward_acc_weight = min(
            self.reward_acc_weight_init + self.reward_acc_weight_lr * self.count,
            self.reward_acc_max
        )
        reward_acc = self.reward_acc_weight * torch.exp(-self.linear_a)

        self.reward_jerk_weight = min(
            self.reward_jerk_weight_init + self.reward_jerk_weight_lr * self.count,
            self.reward_jerk_max
        )
        reward_jerk = self.reward_jerk_weight * torch.exp(-self.linear_jerk)

        self.reward_snap_weight = min(
            self.reward_snap_weight_init + self.reward_snap_weight_lr * self.count,
            self.reward_snap_max
        )
        reward_snap = self.reward_snap_weight * torch.exp(-self.linear_snap)

        spin = torch.square(self.drone.vel_b[..., -1])
        reward_spin = self.reward_spin_weight * 0.5 / (1.0 + torch.square(spin))

        # Potential-based shaping: 每步即时跟踪改善奖励，加速RL学习
        # F = coef*(dist(t-1)-dist(t)) > 0 当靠近目标时，不改变最优策略
        not_first_step = (self.progress_buf > 1).unsqueeze(-1).float()
        reward_tracking_delta = 20.0 * (self.last_distance - distance) * not_first_step
        self.last_distance = distance.clone()

        residual_norm_penalty = self.residual_penalty_coef * torch.norm(self.comp_actions, dim=-1)
        residual_smooth_penalty = self.residual_smoothness_coef * torch.norm(
            self.comp_actions - self.prev_comp_actions, dim=-1
        )

        # reward_pure: 与 track.py 的 reward 完全一致，可直接对比
        reward_pure = (
            reward_pos
            + reward_pos * (reward_up + reward_spin)
            + reward_action_norm
            + reward_action_smoothness
            + reward_acc
            + reward_jerk
            + reward_snap
        )
        # reward: 训练用奖励 = reward_pure + tracking_delta shaping - residual penalties
        reward = reward_pure + reward_tracking_delta - residual_norm_penalty - residual_smooth_penalty

        self.stats["reward_pos"].add_(reward_pos)
        self.stats["reward_action_smoothness"].add_(reward_action_smoothness)
        self.stats["reward_action_norm"].add_(reward_action_norm)
        self.stats["reward_acc"].add_(reward_acc)
        self.stats["reward_jerk"].add_(reward_jerk)
        self.stats["reward_spin"].add_(reward_pos * reward_spin)
        self.stats["reward_up"].add_(reward_pos * reward_up)
        self.stats["reward_tracking_delta"].add_(reward_tracking_delta)
        self.stats["reward_action_smoothness_scale"].set_(
            self.reward_action_smoothness_weight * torch.ones(self.num_envs, 1, device=self.device)
        )
        self.stats["reward_action_norm_scale"].set_(
            self.reward_action_norm_weight * torch.ones(self.num_envs, 1, device=self.device)
        )

        # [2026-05-06] eval_no_reset=True 时禁用所有提前 reset（包括 distance>reset_thres 和 z<0.1 坠机）
        thres_reset = distance > self.reset_thres
        z_crash = self.drone.pos[..., 2] < 0.1
        if self.use_eval and self.eval_no_reset:
            thres_reset = torch.zeros_like(thres_reset)
            z_crash = torch.zeros_like(z_crash)
        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            | z_crash
            | thres_reset
        )

        if self.use_eval:
            self.action_data.append(self.final_actions[0].clone())
            if done[0]:
                torch.save(self.sim_data, "sim_state.pt")
                torch.save(self.sim_rpy, "sim_rpy.pt")
                torch.save(self.action_data, "sim_action.pt")

        ep_len = self.progress_buf.unsqueeze(-1)

        self.stats["tracking_error"].div_(torch.where(done, ep_len, torch.ones_like(ep_len)))
        self.stats["action_error_order1_mean"].div_(torch.where(done, ep_len, torch.ones_like(ep_len)))
        self.stats["action_error_order2_mean"].div_(torch.where(done, ep_len, torch.ones_like(ep_len)))
        self.stats["smoothness_mean"].div_(torch.where(done, ep_len, torch.ones_like(ep_len)))
        self.stats["reward_pos"].div_(torch.where(done, ep_len, torch.ones_like(ep_len)))
        self.stats["reward_spin"].div_(torch.where(done, ep_len, torch.ones_like(ep_len)))
        self.stats["reward_up"].div_(torch.where(done, ep_len, torch.ones_like(ep_len)))
        self.stats["reward_action_smoothness"].div_(torch.where(done, ep_len, torch.ones_like(ep_len)))
        self.stats["reward_action_norm"].div_(torch.where(done, ep_len, torch.ones_like(ep_len)))
        self.stats["reward_acc"].div_(torch.where(done, ep_len, torch.ones_like(ep_len)))
        self.stats["reward_jerk"].div_(torch.where(done, ep_len, torch.ones_like(ep_len)))
        self.stats["linear_v_mean"].div_(torch.where(done, ep_len, torch.ones_like(ep_len)))
        self.stats["angular_v_mean"].div_(torch.where(done, ep_len, torch.ones_like(ep_len)))
        self.stats["linear_a_mean"].div_(torch.where(done, ep_len, torch.ones_like(ep_len)))
        self.stats["angular_a_mean"].div_(torch.where(done, ep_len, torch.ones_like(ep_len)))
        self.stats["linear_jerk_mean"].div_(torch.where(done, ep_len, torch.ones_like(ep_len)))
        self.stats["linear_snap_mean"].div_(torch.where(done, ep_len, torch.ones_like(ep_len)))
        self.stats["residual_norm_mean"].div_(torch.where(done, ep_len, torch.ones_like(ep_len)))
        self.stats["residual_smoothness_mean"].div_(torch.where(done, ep_len, torch.ones_like(ep_len)))
        self.stats["reward_tracking_delta"].div_(torch.where(done, ep_len, torch.ones_like(ep_len)))

        self.stats["return"] += reward
        self.stats["return_pure"] += reward_pure
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        # 更新 prev_comp_actions（必须在 reward 计算之后，供下一步 smoothness penalty 使用）
        self.prev_comp_actions = self.comp_actions.clone()

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1)
                },
                "done": done,
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_traj(self, steps: int, env_ids=None, step_size: float = 1.0):
        if env_ids is None:
            env_ids = ...

        t = self.progress_buf.unsqueeze(1) + step_size * torch.arange(steps, device=self.device)
        t = self.traj_t0 + t * self.dt

        if not self.use_eval:
            smooth = self.ref[0].batch_pos(t)
            zigzag = self.ref[1].batch_pos(t)
            target_pos = (
                smooth * (1 - self.ref_style_seq).unsqueeze(1).unsqueeze(1)
                + zigzag * self.ref_style_seq.unsqueeze(1).unsqueeze(1)
            )
        else:
            target_pos = []
            for ti in range(t.shape[1]):
                target_pos.append(self.ref.pos(t[:, ti]))
            target_pos = torch.stack(target_pos, dim=1)[env_ids]

        return target_pos
    def _reset_disturbance(self, env_ids: torch.Tensor):
        """
        重置指定环境的扰动模型参数。
        对 env_ids 中的环境采样 bias/GM/swing 参数。
        [2026-05-06 重构] 移除 disturbance_enable guard（由 wind 统一控制）。
        # 旧代码: if not self.disturbance_enable: return
        """
        n = len(env_ids)
        device = self.device

        # 选择 train 或 eval 范围
        if self.use_eval:
            bias_range = self.dist_cfg["bias"].get("range_eval", [-2.0, 2.0])
            gm_sigma_range = self.dist_cfg["gauss_markov"].get("sigma_range_eval", [0.8, 1.5])
            swing_amp_range = self.dist_cfg["swing"].get("amp_range_eval", [0.0, 1.2])
        else:
            bias_range = self.dist_cfg["bias"].get("range_train", [-2.0, 2.0])
            gm_sigma_range = self.dist_cfg["gauss_markov"].get("sigma_range_train", [0.2, 0.8])
            swing_amp_range = self.dist_cfg["swing"].get("amp_range_train", [0.0, 0.8])

        # ============================================================
        # sinsum 模式：采样强度和频率
        # ============================================================
        if self.dist_mode == "sinsum":
            self.wind_i[env_ids] = (
                torch.rand(n, 1, device=device)
                * (self.dist_intensity_high - self.dist_intensity_low)
                + self.dist_intensity_low
            )
            self.wind_w[env_ids] = torch.randn(n, 3, self.dist_num_freqs, device=device)

        # ============================================================
        # composite 模式：采样 bias、GM、swing 参数
        # ============================================================
        elif self.dist_mode == "composite":
            # 分量 1：常值偏置
            if self.dist_cfg["bias"].get("enable", True):
                # 每轴独立均匀采样 U(bias_range[0], bias_range[1])
                self.dist_bias[env_ids] = (
                    torch.rand(n, 3, device=device)
                    * (bias_range[1] - bias_range[0])
                    + bias_range[0]
                )
            else:
                self.dist_bias[env_ids] = 0.0

            # 分量 2：Gauss-Markov 随机扰动
            if self.dist_cfg["gauss_markov"].get("enable", True):
                sigma = (
                    torch.rand(n, 1, device=device)
                    * (gm_sigma_range[1] - gm_sigma_range[0])
                    + gm_sigma_range[0]
                )
                tau_range = self.dist_cfg["gauss_markov"].get("tau_range", [0.5, 2.0])
                tau = (
                    torch.rand(n, 1, device=device)
                    * (tau_range[1] - tau_range[0])
                    + tau_range[0]
                )
                alpha = torch.exp(-self.dt / tau)
                self.dist_gm_sigma[env_ids] = sigma
                self.dist_gm_tau[env_ids] = tau
                self.dist_gm_alpha[env_ids] = alpha
                if self.dist_cfg["gauss_markov"].get("reset_to_zero", True):
                    self.dist_gm[env_ids] = 0.0
            else:
                self.dist_gm[env_ids] = 0.0

            # 分量 3：摆动正弦（悬挂负载）
            if self.dist_cfg["swing"].get("enable", True):
                amp_x = (
                    torch.rand(n, device=device)
                    * (swing_amp_range[1] - swing_amp_range[0])
                    + swing_amp_range[0]
                )
                amp_y = (
                    torch.rand(n, device=device)
                    * (swing_amp_range[1] - swing_amp_range[0])
                    + swing_amp_range[0]
                )
                freq_range = self.dist_cfg["swing"].get("freq_range", [0.3, 1.2])
                freq = (
                    torch.rand(n, 1, device=device)
                    * (freq_range[1] - freq_range[0])
                    + freq_range[0]
                )
                if self.dist_cfg["swing"].get("random_phase", True):
                    phase_x = 2.0 * torch.pi * torch.rand(n, device=device)
                    phase_y = 2.0 * torch.pi * torch.rand(n, device=device)
                else:
                    phase_x = torch.zeros(n, device=device)
                    phase_y = torch.zeros(n, device=device)

                self.dist_swing_amp[env_ids, 0] = amp_x
                self.dist_swing_amp[env_ids, 1] = amp_y
                self.dist_swing_freq[env_ids] = freq
                self.dist_swing_phase[env_ids, 0] = phase_x
                self.dist_swing_phase[env_ids, 1] = phase_y
                if not self.dist_horizontal_only:
                    # Z 轴摆动分量
                    amp_z = (
                        torch.rand(n, device=device)
                        * (swing_amp_range[1] - swing_amp_range[0])
                        + swing_amp_range[0]
                    )
                    phase_z = (
                        2.0 * torch.pi * torch.rand(n, device=device)
                        if self.dist_cfg["swing"].get("random_phase", True)
                        else torch.zeros(n, device=device)
                    )
                    self.dist_swing_amp[env_ids, 2] = amp_z
                    self.dist_swing_phase[env_ids, 2] = phase_z
                else:
                    self.dist_swing_amp[env_ids, 2] = 0.0
                    self.dist_swing_phase[env_ids, 2] = 0.0
            else:
                self.dist_swing_amp[env_ids] = 0.0
                self.dist_swing_freq[env_ids] = 0.0
                self.dist_swing_phase[env_ids] = 0.0

    def _update_and_apply_disturbance(self):
        """
        每个仿真步更新并应用扰动模型。
        计算最终扰动加速度，转换为力，并应用到无人机。
        [2026-05-06 重构] 移除 disturbance_enable guard（由 wind 统一控制）。
        # 旧代码: if not self.disturbance_enable: return
        """
        device = self.device

        # ============================================================
        # sinsum 模式：多频率正弦叠加
        # ============================================================
        if self.dist_mode == "sinsum":
            t = (self.progress_buf * self.dt).reshape(self.num_envs, 1, 1)
            a_dist = self.wind_i * torch.sin(t * self.wind_w).sum(-1)  # [num_envs, 3]
            self.dist_acc[:] = a_dist

        # ============================================================
        # composite 模式：bias + GM + swing
        # ============================================================
        elif self.dist_mode == "composite":
            # 分量 1：常值偏置
            a_bias = self.dist_bias.clone()

            # 分量 2：Gauss-Markov 随机扰动
            eps = torch.randn_like(self.dist_gm)
            alpha = self.dist_gm_alpha
            sigma = self.dist_gm_sigma
            a_gm = alpha * self.dist_gm + sigma * torch.sqrt(1.0 - alpha ** 2) * eps
            self.dist_gm[:] = a_gm
            if self.dist_horizontal_only:
                a_gm[:, 2] = 0.0  # 仅水平方向

            # 分量 3：摆动正弦
            t = (self.progress_buf.float() * self.dt).reshape(-1, 1)  # [num_envs, 1]
            phase = self.dist_swing_phase  # [num_envs, 2]
            freq = self.dist_swing_freq  # [num_envs, 1]
            amp = self.dist_swing_amp  # [num_envs, 2]

            # amp: [N,3], phase: [N,3], freq: [N,1], t: [N,1] -> 广播得到 [N,3]
            a_swing = amp * torch.sin(2.0 * torch.pi * freq * t + phase)  # [num_envs, 3]

            # 求和：a_dist = bias + gm + swing
            a_dist = a_bias + a_gm + a_swing

            # 仅水平方向
            if self.dist_horizontal_only:
                a_dist[:, 2] = 0.0

            # 截断总水平范数
            if self.dist_clip_total:
                max_acc = self.dist_max_acc if hasattr(self, 'dist_max_acc') else self.dist_max_acc_train
                if self.dist_horizontal_only:
                    # 仅截断水平（XY）范数
                    xy = a_dist[:, :2]
                    norm_xy = torch.linalg.norm(xy, dim=-1, keepdim=True).clamp_min(1e-6)
                    scale = torch.clamp(max_acc / norm_xy, max=1.0)
                    a_dist[:, :2] = xy * scale
                else:
                    # 截断 3D 总范数
                    norm_3d = torch.linalg.norm(a_dist, dim=-1, keepdim=True).clamp_min(1e-6)
                    scale = torch.clamp(max_acc / norm_3d, max=1.0)
                    a_dist = a_dist * scale

            self.dist_acc[:] = a_dist

        # 记录各个分量用于调试/日志
        if self.dist_mode == "composite":
            self.info["disturbance_acc"] = self.dist_acc.clone()
            self.info["disturbance_bias"] = self.dist_bias.clone()
            self.info["disturbance_gm"] = self.dist_gm.clone()
            # disturbance_swing 需要在这里计算，现在先使用总的减去其他两个
            self.info["disturbance_swing"] = self.dist_acc - self.dist_bias - self.dist_gm

        # ============================================================
        # 将加速度扰动转换为力并应用
        # ============================================================
        # 使用无人机质量（如果有随机化则使用当前质量，否则使用 MASS_0）
        mass = self.total_mass  # [num_envs, 1]
        dist_force = mass * self.dist_acc  # [num_envs, 3]
        self.dist_force[:] = dist_force

        # 扩展为无人机形状 [num_envs, num_drones, 3]
        dist_force_expanded = dist_force.unsqueeze(1).expand(self.num_envs, self.drone.n, 3)

        # 应用全局力
        self.drone.base_link.apply_forces(dist_force_expanded, is_global=True)
