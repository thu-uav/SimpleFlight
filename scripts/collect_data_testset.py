# =============================================================================
# [修改记录 - 2026-05-19]
# 修改文件：collect_data_test.py
# 修改内容：extract_and_save() 内额外保存 "resets" mask（done 信号）。
#   - 新增: done_mask = trajs[("next","done")].squeeze(-1).contiguous().bool()
#   - dataset 中新增 key "resets": done_mask.cpu()
# 原版备份：collect_data_test_20260519.py
# =============================================================================
import logging
import os
import time
import hydra
import torch
import numpy as np
import datetime
from omegaconf import OmegaConf

from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.utils.torchrl import SyncDataCollector, AgentSpec
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction, 
    FromDiscreteAction,
    ravel_composite,
    History
)
from omni_drones.learning import (
    MAPPOPolicy, HAPPOPolicy, QMIXPolicy, DQNPolicy, SACPolicy, TD3Policy, 
    MATD3Policy, TDMPCPolicy, Policy, PPOPolicy, PPOAdaptivePolicy, PPORNNPolicy
)

from setproctitle import setproctitle
from torchrl.envs.transforms import (
    TransformedEnv, 
    InitTracker, 
    Compose,
)
from tqdm import tqdm

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_collect_pinn0504")
def main(cfg):
    # 1. 强制覆盖配置，确保采集模式正确
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    cfg.task.env.num_envs = 4096  # [建议] 强制设置较大的并行数量以快速采集
    cfg.headless = True           # [强制] 开启无头模式，无需渲染，极大提高速度
    cfg.task.use_eval = True      # [test版本] 强制使用 eval 模式以支持指定轨迹
    cfg.task.name = "TrackPINNTest0518"  # [test版本] 使用新环境类
    
    print(f"Start Data Collection with {cfg.task.env.num_envs} environments...")
    print(OmegaConf.to_yaml(cfg))

    # 2. 初始化仿真应用
    simulation_app = init_simulation_app(cfg)
    
    # 3. 导入环境类 (这里会自动加载我们修改过的 TrackPINN)
    from omni_drones.envs.isaac_env import IsaacEnv
    
    # 获取算法类
    algos = {
        "ppo": PPOPolicy, "ppo_adaptive": PPOAdaptivePolicy, "ppo_rnn": PPORNNPolicy,
        "mappo": MAPPOPolicy, "happo": HAPPOPolicy, "qmix": QMIXPolicy, "dqn": DQNPolicy,
        "sac": SACPolicy, "td3": TD3Policy, "matd3": MATD3Policy, "tdmpc": TDMPCPolicy,
        "test": Policy
    }

    # 4. 初始化环境
    # 注意：这里会加载 yaml 中指定的 task.name，稍后在 yaml 中我们需将其改为 TrackPINN
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=True) # 强制 headless

    # 5. 设置 Transforms (为了适配 Policy 的输入格式，必须保持与训练时一致)
    transforms = [InitTracker()]
    
    # 根据配置添加 flatten 等变换
    if cfg.task.get("flatten_obs", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    if cfg.task.get("flatten_state", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "state")))
    if cfg.task.get("history", False):
        transforms.append(History([("agents", "observation")], steps=4))
    
    # Action transform
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform == "PIDrate":
            from omni_drones.controllers import PIDRateController as _PIDRateController
            from omni_drones.utils.torchrl.transforms import PIDRateController
            controller = _PIDRateController(cfg.sim.dt, 9.81, base_env.drone.params).to(base_env.device)
            transforms.append(PIDRateController(controller))
        elif action_transform == "PIDrate_FM":
            from omni_drones.controllers import PID_controller_flightmare as _PID_controller_flightmare
            from omni_drones.utils.torchrl.transforms import PIDRateController_flightmare
            controller = _PID_controller_flightmare(cfg.sim.dt, base_env.drone.params, base_env.device).to(base_env.device)
            transforms.append(PIDRateController_flightmare(controller))
        # ... (保留原有的其他 action transform 逻辑以防万一)

    env = TransformedEnv(base_env, Compose(*transforms))
    env.set_seed(cfg.seed)

    # 6. 加载策略 (Actor)
    agent_spec: AgentSpec = env.agent_spec["drone"]
    policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=agent_spec, device="cuda")

    if cfg.model_dir is not None:
        print(f"Loading nominal policy from: {cfg.model_dir}")
        policy.load_state_dict(torch.load(cfg.model_dir))
    else:
        raise ValueError("Error: 'model_dir' must be provided to load the nominal policy!")

    # 7. 开始采集数据
    @torch.no_grad()
    def collect():
        env.eval()
        # policy.eval()
        # [替换为] 对内部模块分别调用 eval
        if hasattr(policy, "actor"):
            policy.actor.eval()
        if hasattr(policy, "critic"):
            policy.critic.eval()
        
        print("Collecting data rollout...")
        # rollout 会自动运行 env.reset() 并收集 max_episode_length 步的数据
        # 返回的 trajs 是一个 TensorDict，形状通常为 [num_envs, max_steps]
        trajs = env.rollout(
            max_steps=base_env.max_episode_length,
            policy=lambda x: policy(x, deterministic=True), # 采集数据建议用确定性策略
            auto_reset=True,
            break_when_any_done=False,
            return_contiguous=False  # [2026-05-06 优化] 避免全量 contiguous 导致 OOM；只对用到的 key 单独提取
        )
        return trajs

    # =========================================================
    # [test版本] 顺序采集 3 种轨迹，各保存独立文件
    # 轨迹: normal(8字T=5.5s) / fast(8字T=3.5s) / pentagram(五角星)
    # 扰动: 每次 env.reset() 时自动重新随机化，无需额外处理
    # =========================================================
    TRAJ_LIST = ['normal', 'fast', 'pentagram']

    def extract_and_save(trajs, traj_name, save_dir, timestamp):
        print(f"[{traj_name}] Extracting dataset...")
        pinn_state = trajs[("next", "info", "pinn_features")].contiguous()
        actions = trajs[("info", "policy_action")].contiguous()
        if actions.dim() == 4:
            actions = actions.squeeze(2)
        gt_disturbance = trajs[("next", "info", "gt_disturbance")].contiguous()

        # [2026-05-19 新增] 提取 done 信号，记录时间轴断点
        # shape: [Envs, Time] bool —— True=该步 episode 末尾（之后时间轴断裂）
        done_mask = trajs[("next", "done")].squeeze(-1).contiguous().bool()  # [Envs, Time]

        if torch.isnan(pinn_state).any() or torch.isnan(gt_disturbance).any():
            print(f"[{traj_name}] Warning: NaNs detected!")

        X_inputs = torch.cat([pinn_state, actions], dim=-1)
        Y_labels = gt_disturbance

        dataset = {
            "inputs": X_inputs.cpu(),   # [Envs, Time, 19]
            "labels": Y_labels.cpu(),   # [Envs, Time, 3]
            # [2026-05-19 新增] 时间轴断点 mask，供训练侧过滤跨断点窗口
            "resets": done_mask.cpu(),  # [Envs, Time] bool — True=episode末尾（断点）
            "metadata": {
                "traj_type": traj_name,
                "features": ["v_body(3)", "w_body(3)", "R_flat(9)", "action_ctbr(4)"],
                "labels": ["acc_res_body(3)"],
                "description": f"PINN Dataset - traj:{traj_name} (3D Tensor: Batch x Time x Feat)"
            }
        }

        folder_name = f"dataset_{traj_name}_{timestamp}"
        folder_path = os.path.join(save_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        save_path = os.path.join(folder_path, f"dataset_test_{traj_name}_{timestamp}.pt")
        torch.save(dataset, save_path)

        X_flat = X_inputs.reshape(-1, X_inputs.shape[-1])
        print(f"[{traj_name}] Saved! Envs={X_inputs.shape[0]}, Time={X_inputs.shape[1]}, "
              f"Total samples={X_flat.shape[0]}, Path={os.path.abspath(save_path)}")
        return save_path

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = "collected_data"
    os.makedirs(save_dir, exist_ok=True)

    saved_paths = []
    for traj_name in TRAJ_LIST:
        print(f"\n{'='*50}")
        print(f"[Round] 切换到轨迹: {traj_name}")
        print(f"{'='*50}")

        # 切换轨迹类型（动态替换 self.ref，无需重启）
        base_env.switch_eval_traj(traj_name)

        # 重置所有 env（触发扰动重新随机化）
        env.reset()

        # 采集本轮数据
        trajs = collect()

        # 提取并保存
        path = extract_and_save(trajs, traj_name, save_dir, timestamp)
        saved_paths.append(path)

    print(f"\n{'='*50}")
    print("全部采集完成！")
    for p in saved_paths:
        print(f"  {p}")

    simulation_app.close()

if __name__ == "__main__":
    main()