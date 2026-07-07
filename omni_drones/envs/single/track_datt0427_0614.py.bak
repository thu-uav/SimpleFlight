"""
TrackDATT0427  ---  DATT 自适应基线环境（0427 扰动模型更新版）
====================================================
在 TrackDATT 基础上新增 composite 扰动模型，支持三种工作模式：

  disturbance.mode: "sinsum"（兼容原有行为）
      与 TrackDATT wind_mode="sinusoidal" 等价。
      时变多频率正弦叠加，参数由 disturbance.sinsum 控制。

  disturbance.mode: "composite"（新型三分量模型）
      偏置 + Gauss-Markov随机 + 摆动正弦。
      更真实的扰动模型，适合扰动观测器训练。

  disturbance.enable=false（回退到原 TrackDATT 行为）
      wind_mode: "sinusoidal" 或 "constant" 均可，与 TrackDATT 完全一致。

在所有模式下，info["gt_wind"] 均暴露当前扰动加速度（世界系，m/s^2）。
当 include_gt_wind=True 时，将 gt_wind 附加到 obs（oracle 输入）。
"""

import torch

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec

from .track import Track


class TrackDATT0427(Track):
    """DATT baseline environment（0427 扰动模型更新版）。
    继承 :class:`Track`。
    新增 composite 扰动模型（偏置 + GM随机 + 摆动正弦）。
    扰动模式通过 disturbance.mode 控制（见模块文档）。
    """

    def __init__(self, cfg, headless):
        # 必须在 super().__init__() 之前设置，因为 super() 会调用 _set_specs()
        self.include_gt_wind: bool = bool(cfg.task.get("include_gt_wind", True))
        self.wind_mode: str = str(cfg.task.get("wind_mode", "sinusoidal"))
        self.wind_max: float = float(cfg.task.get("wind_max", 1.0))
        self.e_dim: int = 3

        super().__init__(cfg, headless)

        # Episode-constant wind buffer（wind_mode="constant" 时使用）
        self.episode_wind = torch.zeros(self.num_envs, self.e_dim, device=self.device)

        # ----------------------------------------------------------------
        # 初始化新扰动模型
        # ----------------------------------------------------------------
        dist_cfg = cfg.task.get("disturbance", None)
        self.disturbance_enable = (dist_cfg is not None) and dist_cfg.get("enable", False)

        if self.disturbance_enable:
            self.dist_cfg = dist_cfg
            self.dist_mode = dist_cfg.get("mode", "composite")
            self.dist_horizontal_only = dist_cfg.get("horizontal_only", True)
            self.dist_clip_total = dist_cfg.get("clip_total", True)
            self.dist_max_acc_train = dist_cfg.get("max_total_acc_train", 3.0)
            self.dist_max_acc_eval = dist_cfg.get("max_total_acc_eval", 3.5)

            # sinsum 模式：复用 Track 已初始化的 wind_w / wind_i buffer
            # Track.__init__ 在 wind=True 时已分配这些 buffer，无需重复分配
            if self.dist_mode == "sinsum":
                sinsum_cfg = dist_cfg.get("sinsum", {})
                if sinsum_cfg.get("intensity_range"):
                    intensity_range = sinsum_cfg["intensity_range"]
                    self.dist_intensity_low = intensity_range[0]
                    self.dist_intensity_high = intensity_range[1]
                else:
                    self.dist_intensity_low = 0.0
                    self.dist_intensity_high = 2.0
                self.dist_num_freqs = sinsum_cfg.get("num_freqs", 8)

            # composite 模式：初始化 bias/GM/swing buffer，并禁用 Track 的原有风应用
            elif self.dist_mode == "composite":
                # 禁用 Track._pre_sim_step 中的风力施加，改由 composite 模型接管
                # 原值：self.wind = True（来自 cfg.task.wind=true 时）
                self.wind = False

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

    # ------------------------------------------------------------------
    def _set_specs(self):
        super()._set_specs()

        if self.include_gt_wind:
            agents_spec = self.observation_spec["agents"]
            old_obs_dim = agents_spec["observation"].shape[-1]
            old_state_dim = agents_spec["state"].shape[-1]

            self.observation_spec["agents"] = CompositeSpec({
                "observation": UnboundedContinuousTensorSpec(
                    (1, old_obs_dim + self.e_dim)),
                "state": UnboundedContinuousTensorSpec(
                    (old_state_dim + self.e_dim,)),
            }).expand(self.num_envs).to(self.device)

        # 重建 info_spec，包含 gt_wind 及扰动模型相关信息（供观测器训练使用）
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec(
                (self.drone.n, 13), device=self.device),
            "prev_action": torch.stack(
                [self.drone.action_spec] * self.drone.n, 0).to(self.device),
            "policy_action": torch.stack(
                [self.drone.action_spec] * self.drone.n, 0).to(self.device),
            "gt_wind": UnboundedContinuousTensorSpec(
                (self.e_dim,), device=self.device),
            # 扰动模型各分量（disturbance.enable=true 时填充，否则保持零）
            "disturbance_acc": UnboundedContinuousTensorSpec((3,), device=self.device),
            "disturbance_bias": UnboundedContinuousTensorSpec((3,), device=self.device),
            "disturbance_gm": UnboundedContinuousTensorSpec((3,), device=self.device),
            "disturbance_swing": UnboundedContinuousTensorSpec((3,), device=self.device),
        }).expand(self.num_envs).to(self.device)

        self.observation_spec["info"] = info_spec
        self.info = info_spec.zero()

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if not hasattr(self, "episode_wind"):
            return
        # constant 模式：采样 episode 级别常值风（原 TrackDATT 逻辑，保持不变）
        if self.wind_mode == "constant":
            n = len(env_ids)
            self.episode_wind[env_ids] = (
                torch.rand(n, self.e_dim, device=self.device) * 2.0 - 1.0
            ) * self.wind_max

        # 重置新扰动模型（sinsum / composite 模式）
        if self.disturbance_enable:
            self._reset_disturbance(env_ids)

    # ------------------------------------------------------------------
    def _pre_sim_step(self, tensordict: TensorDictBase):
        # ---- disturbance 未启用：回退到原 TrackDATT 逻辑 ----
        if not self.disturbance_enable:
            if self.wind_mode == "sinusoidal":
                # 原 TrackDATT sinusoidal 分支：直接调用 Track._pre_sim_step
                # （Track 会应用 sinusoidal wind）
                super()._pre_sim_step(tensordict)
                return

            # 原 TrackDATT constant 分支（完整保留）
            actions = tensordict[("agents", "action")]
            self.info["prev_action"] = tensordict[("info", "prev_action")]
            self.info["policy_action"] = tensordict[("info", "policy_action")]
            self.policy_actions = tensordict[("info", "policy_action")].clone()
            self.prev_actions = self.info["prev_action"].clone()

            self.action_error_order1 = tensordict[("stats", "action_error_order1")].clone()
            self.stats["action_error_order1_mean"].add_(
                self.action_error_order1.mean(dim=-1).unsqueeze(-1))
            self.stats["action_error_order1_max"].set_(torch.max(
                self.stats["action_error_order1_max"],
                self.action_error_order1.mean(dim=-1).unsqueeze(-1)))

            self.effort = self.drone.apply_action(actions)

            if self.wind:
                wind_forces = (
                    self.total_mass.reshape(self.num_envs, 1, 1)
                    * self.episode_wind.unsqueeze(1)
                )
                self.drone.base_link.apply_forces(wind_forces, is_global=True)
            return

        # ---- disturbance 已启用 ----
        if self.dist_mode == "sinsum":
            # sinsum 模式：wind_i/wind_w 已由 _reset_disturbance 用新参数覆盖
            # self.wind=True，Track._pre_sim_step 会正常应用 sinusoidal wind
            # 原 TrackDATT sinusoidal 分支：super()._pre_sim_step(tensordict)
            super()._pre_sim_step(tensordict)
            return

        elif self.dist_mode == "composite":
            # composite 模式：self.wind=False，Track 不施加风力
            # 先走 Track 的 action 处理流程（无风力）
            # 原 TrackDATT sinusoidal 分支：super()._pre_sim_step(tensordict)（当时 self.wind=True）
            super()._pre_sim_step(tensordict)
            # 再施加新扰动模型
            self._update_and_apply_disturbance()

    # ------------------------------------------------------------------
    def _compute_state_and_obs(self):
        td = super()._compute_state_and_obs()

        # 更新 episode_wind（暴露给 policy 的 gt_wind 加速度，世界系）
        if self.disturbance_enable and self.dist_mode == "composite":
            # composite 模式：用新扰动模型的总加速度作为 gt_wind
            # 原值：self.wind_force（Track._pre_sim_step 计算的正弦风加速度）
            self.episode_wind[:] = self.dist_acc

        elif self.disturbance_enable and self.dist_mode == "sinsum":
            # sinsum 模式：wind_force 由 Track._pre_sim_step 设置
            if hasattr(self, "wind_force"):
                self.episode_wind[:] = self.wind_force

        elif self.wind_mode == "sinusoidal":
            # 原 TrackDATT sinusoidal 分支（disturbance.enable=false 时的回退）
            if self.wind and hasattr(self, "wind_force"):
                self.episode_wind[:] = self.wind_force
        # constant 模式：episode_wind 已在 _reset_idx 中采样，此处不更新

        self.info["gt_wind"] = self.episode_wind.clone()

        if self.include_gt_wind:
            obs = td["agents"]["observation"]        # [N, 1, obs_dim]
            state = td["agents"]["state"]            # [N, state_dim]
            gt_obs = self.episode_wind.unsqueeze(1)  # [N, 1, 3]
            gt_state = self.episode_wind             # [N, 3]
            td["agents"]["observation"] = torch.cat([obs, gt_obs], dim=-1)
            td["agents"]["state"] = torch.cat([state, gt_state], dim=-1)

        return td

    # ------------------------------------------------------------------
    # 扰动模型方法（与 TrackResidual 同名方法逻辑一致）
    # ------------------------------------------------------------------

    def _reset_disturbance(self, env_ids: torch.Tensor):
        """重置指定环境的扰动模型参数。
        sinsum 模式：用 disturbance.sinsum 参数覆盖 Track 的默认 wind_i/wind_w。
        composite 模式：采样 bias / GM / swing 参数。
        """
        if not self.disturbance_enable:
            return

        n = len(env_ids)
        device = self.device

        # 根据 train / eval 选择参数范围
        if self.use_eval:
            bias_range = self.dist_cfg["bias"].get("range_eval", [-2.0, 2.0])
            gm_sigma_range = self.dist_cfg["gauss_markov"].get("sigma_range_eval", [0.8, 1.5])
            swing_amp_range = self.dist_cfg["swing"].get("amp_range_eval", [0.0, 1.2])
        else:
            bias_range = self.dist_cfg["bias"].get("range_train", [-2.0, 2.0])
            gm_sigma_range = self.dist_cfg["gauss_markov"].get("sigma_range_train", [0.2, 0.8])
            swing_amp_range = self.dist_cfg["swing"].get("amp_range_train", [0.0, 0.8])

        # ---- sinsum 模式 ----
        if self.dist_mode == "sinsum":
            self.wind_i[env_ids] = (
                torch.rand(n, 1, device=device)
                * (self.dist_intensity_high - self.dist_intensity_low)
                + self.dist_intensity_low
            )
            self.wind_w[env_ids] = torch.randn(n, 3, self.dist_num_freqs, device=device)

        # ---- composite 模式 ----
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
        """每个仿真步更新并应用扰动模型。
        sinsum 模式：由 Track._pre_sim_step 中的 wind 代码处理，此处无需操作。
        composite 模式：GM更新 -> swing计算 -> 求和 -> clip -> apply_forces。
        """
        if not self.disturbance_enable:
            return

        # sinsum 模式由 Track 的 wind 逻辑处理，无需再次施加
        if self.dist_mode == "sinsum":
            return

        # ---- composite 模式：bias + GM + swing ----
        if self.dist_mode == "composite":
            a_bias = self.dist_bias.clone()

            # Gauss-Markov 更新：a_gm[k+1] = alpha * a_gm[k] + sigma * sqrt(1-alpha^2) * eps
            eps = torch.randn_like(self.dist_gm)
            alpha = self.dist_gm_alpha   # [N, 1]
            sigma = self.dist_gm_sigma   # [N, 1]
            a_gm = alpha * self.dist_gm + sigma * torch.sqrt(1.0 - alpha ** 2) * eps
            self.dist_gm[:] = a_gm
            if self.dist_horizontal_only:
                a_gm[:, 2] = 0.0  # 仅水平方向

            # 摆动正弦：a_swing(t) = A * sin(2*pi*f_s*t + phi)，全3轴广播
            t = (self.progress_buf.float() * self.dt).reshape(-1, 1)  # [N, 1]
            # dist_swing_amp: [N,3], dist_swing_phase: [N,3], dist_swing_freq: [N,1]
            a_swing = self.dist_swing_amp * torch.sin(
                2.0 * torch.pi * self.dist_swing_freq * t + self.dist_swing_phase
            )  # [N, 3]

            # 三分量求和
            a_dist = a_bias + a_gm + a_swing

            # 强制水平方向
            if self.dist_horizontal_only:
                a_dist[:, 2] = 0.0

            # 截断总水平范数
            if self.dist_clip_total:
                max_acc = self.dist_max_acc_eval if self.use_eval else self.dist_max_acc_train
                xy = a_dist[:, :2]
                norm_xy = torch.linalg.norm(xy, dim=-1, keepdim=True).clamp_min(1e-6)
                scale = torch.clamp(max_acc / norm_xy, max=1.0)
                a_dist[:, :2] = xy * scale

            self.dist_acc[:] = a_dist

            # 记录各分量（用于 wandb 日志 / 观测器训练标签）
            self.info["disturbance_acc"] = self.dist_acc.clone()
            self.info["disturbance_bias"] = self.dist_bias.clone()
            self.info["disturbance_gm"] = self.dist_gm.clone()
            self.info["disturbance_swing"] = a_swing.clone()

        # 将加速度扰动转换为力并应用（world frame）
        # self.total_mass: [num_envs, 1, 1]（来自 Track.__init__）
        dist_force = self.total_mass.reshape(self.num_envs, 1, 1) * self.dist_acc.unsqueeze(1)
        # 原值：wind_forces = self.total_mass.reshape(N,1,1) * self.wind_force.unsqueeze(1)
        self.drone.base_link.apply_forces(dist_force, is_global=True)
