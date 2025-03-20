import torch
import torch.nn as nn
import torch.nn.functional as F
import torchrl.data
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, make_functional
from tensordict.nn import make_functional, TensorDictModule, TensorDictParams
from functorch import vmap
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    TensorDictReplayBuffer,
    TensorSpec,
)
from torchrl.data import UnboundedContinuousTensorSpec as UnboundedTensorSpec
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives.utils import hold_out_net

print(torchrl.data.__file__)
import copy

from tqdm import tqdm

from omni_drones.utils.torchrl import AgentSpec

from .common import soft_update

class MADDPGPolicy(object):
    def __init__(
        self,
        cfg,
        agent_spec: AgentSpec,
        device: str = "cuda",
    ) -> None:
        self.cfg = cfg
        self.agent_spec = agent_spec
        self.device = device
        print("self.agent_spec.action_spec", self.agent_spec.action_spec)
        self.gradient_steps = int(cfg.gradient_steps)
        self.batch_size = int(cfg.batch_size)
        self.buffer_size = int(cfg.buffer_size)

        self.target_noise = self.cfg.target_noise
        self.policy_noise = self.cfg.policy_noise
        self.noise_clip = self.cfg.noise_clip

        self.action_dim = self.agent_spec.action_spec.shape[-1]

        if cfg.get("agent_name"):
            self.obs_name = ("agents", cfg.agent_name + "_observation")
            self.state_name = ("agents", cfg.agent_name + "_state")
            self.act_name = ("agents", cfg.agent_name + "_action")
            self.reward_name = ("agents", cfg.agent_name + "_reward")
        else:
            self.obs_name = ("agents", "observation")
            self.state_name = ("agents", "state")
            self.act_name = ("agents", "action")
            self.reward_name = ("agents", "reward")

        self.make_model()

        self.replay_buffer = TensorDictReplayBuffer(
            batch_size=self.batch_size,
            storage=LazyTensorStorage(max_size=self.buffer_size, device="cpu"),
            sampler=RandomSampler(),
        )

    def make_model(self):
        self.policy_in_keys = [self.obs_name]
        print("self.policy_in_keys: ", self.policy_in_keys)

        self.policy_out_keys = [self.act_name, f"{self.agent_spec.name}.logp"]

        def create_actor():
            encoder = make_encoder(self.cfg.actor, self.agent_spec.observation_spec)

            return TensorDictModule(
                nn.Sequential(
                    encoder,
                    nn.ELU(),
                    nn.Linear(encoder.output_shape.numel(), self.action_dim),
                    nn.Tanh(),
                ),
                in_keys=self.policy_in_keys,
                out_keys=self.policy_out_keys,
            ).to(self.device)

        if self.cfg.share_actor:

            self.actor = create_actor()
            self.actor_opt = torch.optim.Adam(
                self.actor.parameters(), lr=self.cfg.actor.lr
            )
            self.actor_params = make_functional(self.actor).expand(self.agent_spec.n)
            self.actor_target_params = self.actor_params.clone()
        else:
            actors = nn.ModuleList([create_actor() for _ in range(self.agent_spec.n)])
            self.actor = actors[0]
            self.actor_opt = torch.optim.Adam(actors.parameters(), lr=self.cfg.actor.lr)
            # 收集所有演员的参数
            self.actor_params = torch.stack(
                [make_functional(actor) for actor in actors]
            )
            self.actor_target_params = self.actor_params.clone()

        self.value_in_keys = [self.obs_name, self.act_name]
        self.value_out_keys = [f"{self.agent_spec.name}.q"]
        self.critic = Critic(
            self.cfg.critic,
            self.agent_spec.n,
            self.agent_spec.observation_spec,
            self.agent_spec.action_spec,
        ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=self.cfg.critic.lr
        )
        self.critic_loss_fn = {"mse": F.mse_loss, "smooth_l1": F.smooth_l1_loss}[
            self.cfg.critic_loss
        ]

    def __call__(
        self, tensordict: TensorDict, deterministic: bool = False
    ) -> TensorDict:

        actor_output = self._call_actor(tensordict, self.actor_params)
        # 添加噪声到动作输出
        if deterministic == False:
            action_noise = (
                actor_output[self.act_name]
                .clone()
                .normal_(0, self.policy_noise)
                .clamp_(-self.noise_clip, self.noise_clip)
            )
            actor_output[self.act_name].add_(action_noise)
        actor_output[self.act_name].batch_size = tensordict.batch_size
        tensordict.update(actor_output)
        return tensordict

    def _call_actor(self, tensordict: TensorDict, params: TensorDict):

        actor_input = tensordict.select(*self.policy_in_keys)
        actor_input.batch_size = [*actor_input.batch_size, self.agent_spec.n]
        actor_output = vmap(self.actor, in_dims=(1, 0), out_dims=1)(actor_input, params)
        return actor_output

    def train_op(self, data: TensorDict):
        self.replay_buffer.extend(data.to("cpu").reshape(-1))

        if len(self.replay_buffer) < self.cfg.buffer_size:
            print(f"{len(self.replay_buffer)} < {self.cfg.buffer_size}")
            return {}

        infos_critic = []
        infos_actor = []

        with tqdm(range(1, self.gradient_steps + 1)) as t:
            for gradient_step in t:
                transition = self.replay_buffer.sample(self.batch_size).to(self.device)

                state = transition[self.obs_name]
                actions_taken = transition[self.act_name]
                reward = transition[("next", self.reward_name)]
                next_dones = transition[("next", "done")].float().unsqueeze(-1)
                next_state = transition[("next", self.obs_name)]

                with torch.no_grad():
                    # use target actor to get next step actions
                    next_action: torch.Tensor = self._call_actor(
                        transition["next"], self.actor_target_params
                    )[self.act_name]

                    if self.target_noise > 0:
                        action_noise = (
                            next_action.clone()
                            .normal_(0, self.target_noise)
                            .clamp_(-self.noise_clip, self.noise_clip)
                        )
                        next_action = torch.clamp(next_action + action_noise, -1, 1)

                    # 计算下一个状态的Q值，原来MATD3是计算2个target critic后取最小值
                    next_qs = self.critic_target(
                        next_state, next_action
                    )  # [batchsize,2,1]

                    target_q = (
                        reward + self.cfg.gamma * (1 - next_dones) * next_qs
                    ).detach()  # [batchsize, 2 ,1]，这里的2对于agent num

                    assert not torch.isinf(target_q).any()
                    assert not torch.isnan(target_q).any()

                qs = self.critic(state, actions_taken)  # [batchsize, 2, 1]

                critic_loss = self.critic_loss_fn(qs, target_q)
                self.critic_opt.zero_grad()
                critic_loss.backward()
                critic_grad_norm = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.cfg.max_grad_norm
                )
                self.critic_opt.step()
                infos_critic.append(
                    TensorDict(
                        {
                            "critic_loss": critic_loss,
                            "critic_grad_norm": critic_grad_norm,
                            "q_taken": qs.mean(),
                        },
                        [],
                    )
                )

                # if (gradient_step + 1) % self.cfg.actor_delay == 0:
                with hold_out_net(self.critic):
                    actor_output = self._call_actor(transition, self.actor_params)
                    actions_new = actor_output[self.act_name]

                    actor_losses = []
                    for a in range(self.agent_spec.n):
                        actions = actions_taken.clone()
                        actions[..., a, :] = actions_new[..., a, :]
                        qs = self.critic(state, actions)
                        actor_losses.append(-qs.mean())

                    actor_loss = torch.stack(actor_losses).sum()
                    self.actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_grad_norm = nn.utils.clip_grad_norm_(
                        self.actor_opt.param_groups[0]["params"], self.cfg.max_grad_norm
                    )
                    self.actor_opt.step()

                    infos_actor.append(
                        TensorDict(
                            {
                                "actor_loss": actor_loss,
                                "actor_grad_norm": actor_grad_norm,
                            },
                            [],
                        )
                    )

                    with torch.no_grad():
                        soft_update_td(
                            self.actor_target_params, self.actor_params, self.cfg.tau
                        )
                        soft_update(self.critic_target, self.critic, self.cfg.tau)

                t.set_postfix({"critic_loss": critic_loss.item()})

        infos = {**torch.stack(infos_actor), **torch.stack(infos_critic)}
        infos = {k: torch.mean(v).item() for k, v in infos.items()}
        return infos

    def state_dict(self):
        state_dict = {
            "critic": self.critic.state_dict(),
            "actor_params": self.actor_params,
        }
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.actor_params = TensorDictParams(state_dict["actor_params"].to_tensordict())
        self.critic.load_state_dict(state_dict["critic"])

def soft_update_td(target_params: TensorDict, params: TensorDict, tau: float):
    for target_param, param in zip(
        target_params.values(True, True), params.values(True, True)
    ):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


from .common import make_encoder
from .modules.networks import ENCODERS_MAP, MLP


class Critic(nn.Module):
    def __init__(
        self,
        cfg,
        num_agents: int,
        state_spec: TensorSpec,
        action_spec: BoundedTensorSpec,
        num_critics: int = 2,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_agents = num_agents
        self.act_space = action_spec
        self.state_spec = state_spec
        self.num_critics = num_critics

        # self.critics = nn.ModuleList([
        #     self._make_critic() for _ in range(self.num_critics)
        # ])
        self.critic = nn.ModuleList([self._make_critic() for _ in range(1)])

    def _make_critic(self):

        if isinstance(self.state_spec, (BoundedTensorSpec, UnboundedTensorSpec)):
            action_dim = self.act_space.shape[-1]
            state_dim = self.state_spec.shape[-1]
            num_units = [
                action_dim * self.num_agents
                + state_dim * self.num_agents,  # 输入是act和state的cat
                *self.cfg["hidden_units"],
            ]  # 各层神经元个数
            base = MLP(num_units)
        elif isinstance(self.state_spec, CompositeSpec):

            encoder_cls = ENCODERS_MAP[self.cfg.attn_encoder]
            base = encoder_cls(CompositeSpec(self.state_spec))
        else:
            raise NotImplementedError

        v_out = nn.Linear(base.output_shape.numel(), self.num_agents)
        return nn.Sequential(base, v_out)

    def forward(self, state: torch.Tensor, actions: torch.Tensor):
        """
        Args:
            state: (batch_size,num_agents, state_dim)
            actions: (batch_size, num_agents, action_dim)

        Returns:
            (batch_size, num_agents, num_critics) - 各个评论者的Q值
        """

        state = state.flatten(1)  # [batch_size, num_agents*state_dim]
        actions = actions.flatten(1)
        x = torch.cat(
            [state, actions], dim=-1
        )  # 将状态和动作拼接 [batch_size, num_agents*(state_dim+action_dim)]

        # return torch.stack([critic(x) for critic in self.critics], dim=-1)
        return self.critic[0](x).unsqueeze(-1)


def soft_update_params(target: TensorDict, source: TensorDict, tau: float): ...