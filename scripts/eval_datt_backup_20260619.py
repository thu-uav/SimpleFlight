"""
eval_datt.py  —  Evaluation script for DATT-trained policies
这个轨迹会一次性依次验证6个轨迹
=============================================================

Evaluates a saved checkpoint across multiple trajectory types and logs
per-trajectory stats to wandb.

Usage
-----
python eval_datt.py \
    model_dir=outputs/track_datt/<run>/wandb/run-xxx/files/checkpoint_final.pt \
    headless=true \
    [traj_types="[slow,normal,fast,poly,zigzag,pentagram]"]

The default trajectory set is: slow, normal, fast, poly, zigzag, pentagram.
Pass a comma-separated list via the traj_types override to restrict/reorder.
"""

import os

import hydra
import torch
import numpy as np
import wandb

from omegaconf import OmegaConf

from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.utils.torchrl import AgentSpec
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    History,
)
from omni_drones.utils.wandb import init_wandb
from omni_drones.learning import (
    MAPPOPolicy,
    HAPPOPolicy,
    QMIXPolicy,
    DQNPolicy,
    SACPolicy,
    TD3Policy,
    MATD3Policy,
    TDMPCPolicy,
    Policy,
    PPOPolicy,
    PPOAdaptivePolicy,
    PPORNNPolicy,
)

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Every:
    def __init__(self, func, steps):
        self.func = func
        self.steps = steps
        self.i = 0

    def __call__(self, *args, **kwargs):
        if self.i % self.steps == 0:
            self.func(*args, **kwargs)
        self.i += 1


# ---------------------------------------------------------------------------
# All trajectory types supported by Track / TrackDATT
# ---------------------------------------------------------------------------
ALL_TRAJ_TYPES = ["slow", "normal", "fast", "poly", "zigzag", "pentagram"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train_datt")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # Reduce num_envs for eval to avoid OOM (can override: eval_num_envs=128)
    eval_num_envs = cfg.get("eval_num_envs", 64)
    cfg.env.num_envs = eval_num_envs

    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    # Must import TrackDATT to register it in IsaacEnv.REGISTRY
    from omni_drones.envs.isaac_env import IsaacEnv
    from omni_drones.envs.single.track_datt import TrackDATT  # noqa: F401

    algos = {
        "ppo": PPOPolicy,
        "ppo_adaptive": PPOAdaptivePolicy,
        "ppo_rnn": PPORNNPolicy,
        "mappo": MAPPOPolicy,
        "happo": HAPPOPolicy,
        "qmix": QMIXPolicy,
        "dqn": DQNPolicy,
        "sac": SACPolicy,
        "td3": TD3Policy,
        "matd3": MATD3Policy,
        "tdmpc": TDMPCPolicy,
        "test": Policy,
    }

    # -----------------------------------------------------------------------
    # Build environment (identical to train_datt.py)
    # -----------------------------------------------------------------------
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]

    if cfg.task.get("flatten_obs", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    if cfg.task.get("flatten_state", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "state")))
    if (
        cfg.task.get("flatten_intrinsics", True)
        and ("agents", "intrinsics") in base_env.observation_spec.keys(True)
    ):
        transforms.append(
            ravel_composite(base_env.observation_spec, ("agents", "intrinsics"), start_dim=-1)
        )
    if cfg.task.get("history", False):
        transforms.append(History([("agents", "observation")], steps=4))

    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transforms.append(FromMultiDiscreteAction(nbins=nbins))
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transforms.append(FromDiscreteAction(nbins=nbins))
        elif action_transform == "velocity":
            from omni_drones.controllers import LeePositionController
            from omni_drones.utils.torchrl.transforms import VelController
            controller = LeePositionController(9.81, base_env.drone.params).to(base_env.device)
            transforms.append(VelController(controller))
        elif action_transform == "attitude":
            from omni_drones.controllers import AttitudeController as Controller
            from omni_drones.utils.torchrl.transforms import AttitudeController
            controller = Controller(9.81, base_env.drone.params).to(base_env.device)
            transforms.append(AttitudeController(controller))
        elif action_transform == "rate":
            from omni_drones.controllers import RateController as _RateController
            from omni_drones.utils.torchrl.transforms import RateController
            controller = _RateController(9.81, base_env.drone.params).to(base_env.device)
            transforms.append(RateController(controller))
        elif action_transform == "PIDrate":
            from omni_drones.controllers import PIDRateController as _PIDRateController
            from omni_drones.utils.torchrl.transforms import PIDRateController
            controller = _PIDRateController(cfg.sim.dt, 9.81, base_env.drone.params).to(base_env.device)
            transforms.append(PIDRateController(controller))
        elif action_transform == "PIDrate_FM":
            from omni_drones.controllers import PID_controller_flightmare as _PID_controller_flightmare
            from omni_drones.utils.torchrl.transforms import PIDRateController_flightmare
            controller = _PID_controller_flightmare(
                cfg.sim.dt, base_env.drone.params, base_env.device
            ).to(base_env.device)
            transforms.append(PIDRateController_flightmare(controller))
        elif not action_transform.lower() == "none":
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    # -----------------------------------------------------------------------
    # Build policy and load checkpoint
    # -----------------------------------------------------------------------
    agent_spec: AgentSpec = env.agent_spec["drone"]
    policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=agent_spec, device="cuda")

    if cfg.model_dir is None:
        raise ValueError(
            "model_dir must be specified, "
            "e.g. model_dir=outputs/track_datt/.../checkpoint_final.pt"
        )
    policy.load_state_dict(torch.load(cfg.model_dir))
    print(f"Loaded checkpoint from: {cfg.model_dir}")

    # -----------------------------------------------------------------------
    # Evaluate across trajectory types
    # -----------------------------------------------------------------------

    record_video = cfg.get("record_video", True)

    @torch.no_grad()
    def evaluate(traj_type: str, seed: int = 0):
        """Run one deterministic episode on the given trajectory type."""
        # Switch trajectory before the episode resets
        base_env.eval_traj = traj_type
        base_env._apply_eval_traj()  # rebuilds self.ref for the new traj type
        # [20260506] 强制覆盖，避免 hydra struct 模式导致 cfg.task.get 读不到 yaml 里的值
        base_env.eval_no_reset = False                        #####    [20260506] 注意修改，是否启用reset
        # [20260510] 控制是否取消 Z 方向风扰动（True=只保留 XY，False=全三轴）
        base_env.dist_horizontal_only = cfg.get("dist_horizontal_only", False)

        frames = []
        if record_video:
            base_env.enable_render(True)
        base_env.eval()
        env.eval()
        env.set_seed(seed)

        t = tqdm(total=base_env.max_episode_length, desc=traj_type)

        def record_frame(*args, **kwargs):
            frame = env.base_env.render(mode="rgb_array")
            frames.append(frame)
            t.update(2)

        trajs = env.rollout(
            max_steps=base_env.max_episode_length,
            policy=lambda x: policy(x, deterministic=True),
            callback=Every(record_frame, 2) if record_video else None,
            auto_reset=True,
            break_when_any_done=False,
            return_contiguous=False,
        ).clone()
        t.close()

        base_env.enable_render(not cfg.headless)
        env.reset()

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        # 统计 crash 环境数（仅 eval_no_reset=False 时有意义）
        num_crashed = 0
        if not base_env.eval_no_reset:
            num_crashed = int((first_done < base_env.max_episode_length - 1).sum().item())

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape + (1,) * (tensor.ndim - 2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            f"eval/{traj_type}/stats.{k}": torch.nanmean(v.float()).item()
            for k, v in traj_stats.items()
        }
        info[f"eval/{traj_type}/num_crashed"] = num_crashed

        if record_video and len(frames):
            video_array = np.stack(frames).transpose(0, 3, 1, 2)
            frames.clear()
            info[f"eval/{traj_type}/recording"] = wandb.Video(
                video_array, fps=0.5 / cfg.sim.dt, format="mp4"
            )

        return info

    # Choose which trajectories to evaluate
    traj_types = cfg.get("traj_types", ALL_TRAJ_TYPES)
    if isinstance(traj_types, str):
        traj_types = [t.strip() for t in traj_types.split(",")]

    all_info = {}
    summary_rows = []

    for i, traj in enumerate(traj_types):
        print(f"\n[{i+1}/{len(traj_types)}] Evaluating trajectory: {traj}")
        info = evaluate(traj_type=traj, seed=i)
        all_info.update(info)
        tracking_err  = info.get(f"eval/{traj}/stats.tracking_error", float("nan"))
        err_max       = info.get(f"eval/{traj}/stats.tracking_error_max", float("nan"))  # [20260506]
        ret           = info.get(f"eval/{traj}/stats.return",         float("nan"))
        episode_len   = info.get(f"eval/{traj}/stats.episode_len",    float("nan"))
        num_crashed    = int(info.get(f"eval/{traj}/num_crashed", 0))
        summary_rows.append((traj, tracking_err, err_max, ret, episode_len, num_crashed))

    # Log everything in one wandb step
    run.log(all_info)

    # [20260506] 打印汇总表格（含 error_max 列）
    print("\n" + "=" * 98)
    print(f"{'Trajectory':<15}  {'tracking_error':>15}  {'error_max':>12}  {'return':>12}  {'episode_len':>12}  {'crashed':>9}")
    print("-" * 98)
    for traj, err, err_max, ret, ep_len, crashed in summary_rows:
        n = base_env.num_envs
        print(f"{traj:<15}  {err:>15.4f}  {err_max:>12.4f}  {ret:>12.2f}  {ep_len:>12.1f}  {crashed:>4}/{n:<4}")
    print("=" * 98)

    wandb.finish()
    simulation_app.close()


if __name__ == "__main__":
    main()
