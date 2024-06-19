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


from functools import partial
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from torch import Tensor
from torchrl.data import CompositeSpec, TensorSpec
from torch.nn.utils.parametrizations import spectral_norm

def register(_map: Dict, name=None):
    def decorator(func):
        _name = name or func.__name__
        assert _name not in _map
        _map[_name] = func
        return func

    return decorator


class MLP(nn.Module):
    def __init__(
        self,
        num_units: Sequence[int],
        normalization: Union[str, nn.Module] = None,
        activation_class: nn.Module = nn.ELU,
        activation_kwargs: Optional[Dict] = None,
        use_sn = False,
    ):
        super().__init__()
        layers = []
        if activation_kwargs is not None:
            activation_class = partial(activation_class, **activation_kwargs)
        if isinstance(normalization, str):
            normalization = getattr(nn, normalization, None)
        for i, (in_dim, out_dim) in enumerate(zip(num_units[:-1], num_units[1:])):
            layer = nn.Linear(in_dim, out_dim)
            if use_sn:
                layer = spectral_norm(layer)
            layers.append(layer)
            if i < len(num_units) - 1:
                layers.append(activation_class())
            if normalization is not None:
                layers.append(normalization(out_dim))
        self.layers = nn.Sequential(*layers)
        self.input_dim = num_units[0]
        self.output_shape = torch.Size((num_units[-1],))

    def forward(self, x: torch.Tensor):
        return self.layers(x)


def split(x, split_shapes, split_sizes):
    return [
        xi.unflatten(-1, shape)
        for xi, shape in zip(torch.split(x, split_sizes, dim=-1), split_shapes)
    ]


def ij(a: torch.Tensor):
    ai = a.unsqueeze(-2).expand(*a.shape[:-2], a.shape[-2], a.shape[-2], a.shape[-1])
    aj = ai.transpose(-2, -3)
    aij = torch.cat([ai, aj], dim=-1)
    return aij


class LFF(nn.Module):
    """Learnable Fourier Features.

    Ideally should help with learning the high-frequency parts for coordinate-like inputs.

    https://openreview.net/forum?id=uTqvj8i3xv

    """

    def __init__(
        self,
        input_size,
        sigma: float = 0.01,
        fourier_dim=256,
        embed_dim=72,
        cat_input=True,
    ) -> None:
        super().__init__()
        b_shape = (input_size, fourier_dim)
        self.cat_input = cat_input
        self.B = nn.Parameter(
            torch.normal(torch.zeros(b_shape), torch.full(b_shape, sigma))
        )
        if self.cat_input:
            self.linear = nn.Linear(fourier_dim * 2 + input_size, embed_dim)
        else:
            self.linear = nn.Linear(fourier_dim * 2, embed_dim)

    def forward(self, x: Tensor):
        proj = torch.matmul(x, self.B) * (2 * torch.pi)
        if self.cat_input:
            ff = torch.cat([torch.sin(proj), torch.cos(proj), x], dim=-1)
        else:
            ff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.linear(ff)


class SplitEmbedding(nn.Module):
    def __init__(
        self,
        input_spec: CompositeSpec,
        embed_dim: int = 72,
        layer_norm=True,
        embed_type="linear",
        use_sn=False,
    ) -> None:
        super().__init__()
        if any(isinstance(spec, CompositeSpec) for spec in input_spec.values()):
            raise ValueError("Nesting is not supported.")
        self.input_spec = {k: v for k, v in input_spec.items() if not k.endswith("mask")}
        self.embed_dim = embed_dim
        self.num_entities = sum(spec.shape[-2] for spec in self.input_spec.values())

        if embed_type == "linear":
            self.embed = nn.ModuleDict(
                {
                    key: 
                        spectral_norm(nn.Linear(value.shape[-1], self.embed_dim)) if use_sn 
                        else nn.Linear(value.shape[-1], self.embed_dim)
                    for key, value in self.input_spec.items()
                }
            )
        else:
            raise NotImplementedError(embed_type)

        if layer_norm:
            self.layer_norm = nn.LayerNorm(embed_dim)
            # self.layer_norm = nn.LayerNorm(
            #     (self.num_entities, embed_dim)
            # )  # somehow faster
        # print(input_spec)

    def forward(self, tensordict: TensorDict):
        embeddings = torch.cat(
            [self.embed[key](tensordict[key]) for key in self.input_spec.keys()], dim=-2
        )
        if hasattr(self, "layer_norm"):
            embeddings = self.layer_norm(embeddings)
        return embeddings


################################## Attn Encoders ##################################
ENCODERS_MAP = {}


@register(ENCODERS_MAP)
class RelationEncoder(nn.Module):
    """
    f(sum_ij g(a_i, a_j))
    """

    def __init__(
        self,
        input_spec: CompositeSpec,
        *,
        embed_dim: int = 72,
        embed_type: str = "linear",
        layer_norm=True,
        f_units=(256, 128),
    ) -> None:
        super().__init__()
        self.output_shape = torch.Size((f_units[-1],))
        self.split_embed = SplitEmbedding(input_spec, embed_dim, layer_norm, embed_type)
        if layer_norm:
            self.g = nn.Sequential(
                MLP([embed_dim * 2, f_units[0]]), nn.LayerNorm(f_units[0])
            )
        else:
            self.g = MLP([embed_dim * 2, f_units[0]])
        self.f = MLP(f_units)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        a: torch.Tensor = self.split_embed(x)
        aij = ij(a)
        g_aij = self.g(aij)
        if mask is not None:
            if mask.shape == x.shape[:-1]:
                mask = mask.unsqueeze(-1)
            elif not mask.dim() == x.dim():
                raise RuntimeError(mask.shape)
            g_aij *= ij(mask).all(-1)
        return self.f(torch.sum(g_aij, dim=(-3, -2)))


@register(ENCODERS_MAP)
class PartialRelationEncoder(nn.Module):
    """
    f(sum_j g(a_i, a_j)), i=0, j!=i
    """

    def __init__(
        self,
        input_spec: CompositeSpec,
        *,
        embed_dim: int = 72,
        embed_type: str = "linear",
        layer_norm=True,
        f_units=(256, 128),
        use_sn=False
    ) -> None:
        super().__init__()
        self.output_shape = torch.Size((f_units[-1],))
        self.split_embed = SplitEmbedding(input_spec, embed_dim, layer_norm, embed_type, use_sn=use_sn)
        if layer_norm:
            self.g = nn.Sequential(
                MLP([embed_dim * 2, f_units[0]]), nn.LayerNorm(f_units[0])
            )
        else:
            self.g = MLP([embed_dim * 2, f_units[0]])
        self.f = MLP(f_units)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        a: torch.Tensor = self.split_embed(x)
        ai, aj = a.split([1, a.shape[-2] - 1], dim=-2)
        aij = torch.cat([ai.broadcast_to(aj.shape), aj], dim=-1)
        g_aij = self.g(aij)
        if mask is not None:
            if mask.shape == x.shape[:-1]:
                mask = mask.unsqueeze(-1)
            elif not mask.dim() == x.dim():
                raise RuntimeError(mask.shape)
            g_aij *= mask[..., 1:, :]
        return self.f(torch.sum(g_aij, dim=-2))


@register(ENCODERS_MAP)
class PartialAttentionEncoder(nn.Module):
    def __init__(
        self,
        input_spec: CompositeSpec,
        *,
        query_index=0,
        embed_dim: int = 128,
        embed_type: str = "linear",
        num_heads: int = 1,
        layer_norm=False,
        norm_first=False,
        attention_type=0,
        self_attention=False,
        use_sn=False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.split_embed = SplitEmbedding(input_spec, embed_dim, layer_norm, embed_type, use_sn=use_sn)
        self.attention_type = attention_type
        self.self_attention = self_attention
        if self_attention:
            self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.self_norm1 = nn.LayerNorm(embed_dim)
            self.self_norm2 = nn.LayerNorm(embed_dim)
            self.self_linear1 = spectral_norm(nn.Linear(embed_dim, embed_dim)) if use_sn else nn.Linear(embed_dim, embed_dim)
            self.self_activation = F.gelu
            self.self_linear2 = spectral_norm(nn.Linear(embed_dim, embed_dim)) if use_sn else nn.Linear(embed_dim, embed_dim)

        if attention_type != -1 and attention_type != 6:
            # no attention, directly calculate the max or mean of all ball information
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        if attention_type == 0:
            # query = obs_self, key & value = all_obs
            self.query_index = [0]
            self.source_index = 0
            output_shape = self.embed_dim
        elif attention_type == 1 or attention_type == 5:
            # query = obs_self + obs_other, key & value = all_obs
            self.query_index = list([0,1,2,3,4,5])
            self.source_index = 0
            output_shape = self.embed_dim*6
        elif attention_type == 2:
            # query = obs_self, key & value = obs_ball
            self.query_index = [0]
            self.source_index = 6
            output_shape = self.embed_dim*7
        elif attention_type == 3:
            # query = obs_self, key & value = obs_other + obs_ball
            self.query_index = [0]
            self.source_index = 1
            output_shape = self.embed_dim*2
        elif attention_type == 4:
            # 2 attention module
            # the first one: query = obs_self, key & value = obs_other
            # the second one: query = obs_self, key & value = obs_ball
            self.query_index = [0]
            self.source_index = 1
            self.source_index2 = 6
            output_shape = self.embed_dim*3
            self.attn2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        elif attention_type == -1:
            output_shape = self.embed_dim*7
        elif attention_type == 6:
            output_shape = self.embed_dim

        if attention_type != 6:
            # dim_feedforward = embed_dim
            self.linear1 = nn.Linear(embed_dim, embed_dim)
            self.activation = F.gelu
            self.linear2 = nn.Linear(embed_dim, embed_dim)

            # self.norm_first = norm_first
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
        self.output_shape = torch.Size((output_shape,))

    def forward(self, obs: Tensor, 
                # key_padding_mask: Optional[Tensor] = None
                ):
        """
        Args:
            x: 
                a tensordict of size (batch, N)
                e.g. TensorDict(
                        fields={
                            attn_obs_ball: Tensor(shape=torch.Size([32, 6, 1, 7]), device=cuda:0, dtype=torch.float32, is_shared=True),
                            obs_others: Tensor(shape=torch.Size([32, 6, 5, 14]), device=cuda:0, dtype=torch.float32, is_shared=True),
                            obs_self: Tensor(shape=torch.Size([32, 6, 1, 27]), device=cuda:0, dtype=torch.float32, is_shared=True)},
                        batch_size=torch.Size([32, 6]),
                        device=cuda:0,
                        is_shared=True)
            padding_mask: (batch, N)
        """

        x = self.split_embed(obs) # [batch, N, num_entities, embed_dim], for actor; e.g. [32, 6, 7, 128]; [bsz, num_entities, embed_dim] for critic
        key_padding_mask = None
        
        if "attn_ball_mask" in obs.keys() and "attn_static_mask" in obs.keys():
            # [bsz, N, ball_num] or [bsz, ball_num]
            ball_mask = obs["attn_ball_mask"]
            static_mask = obs["attn_static_mask"]
            self_mask = torch.zeros(obs["obs_self"].shape[:-1], dtype=bool, device=ball_mask.device)
            others_mask = torch.zeros(obs["obs_others"].shape[:-1], dtype=bool, device=ball_mask.device)
            
            mask = torch.cat([self_mask, others_mask, ball_mask, static_mask], dim=-1)
            # print("mask.shape =", mask.shape)
            key_padding_mask = mask.reshape(-1, mask.shape[-1]) #[batch*N, num_entities]
        # print(key_padding_mask)
        original_shape = x.shape[:-2]

        x = x.reshape(-1, x.shape[-2], x.shape[-1]) # [bsz*N, num_entities, embed_dim]
        if key_padding_mask is not None:
            x -= key_padding_mask.unsqueeze(-1)*x

        if self.self_attention:
            x = self._self_attn(x, key_padding_mask)
        if self.attention_type != 6:
            x2 = self.norm1(x)

        if self.attention_type == 2 or self.attention_type == 3:
            attention_output = x[:, self.query_index] + self._pa_block(x2, key_padding_mask)
            x = torch.cat([x[:, :self.source_index], attention_output, ], dim=-2)
        elif self.attention_type == 4:
            attention_output1 = x[:, self.query_index] + self._pa_block2_1(x2, key_padding_mask)
            attention_output2 = x[:, self.query_index] + self._pa_block2_2(x2, key_padding_mask)
            x = torch.cat([x[:, :self.source_index], attention_output1, attention_output2], dim=-2)
        elif self.attention_type == 5:
            out_max = torch.zeros_like(x[:, self.query_index])
            out_sum = torch.zeros_like(x[:, self.query_index])
            for i in range(6, x2.shape[-2]):
                x3 = x2[:, [0,1,2,3,4,5,i]]
                attn_res = self._pa_block(x3, key_padding_mask)
                out_max = torch.max(out_max, attn_res)
                out_sum = out_sum + attn_res
            x = x[:, self.query_index] + out_max # or sum or mean?
        elif self.attention_type == -1:
            x_base = x[:, [0,1,2,3,4,5]]
            out_max = torch.zeros_like(x[:, [0]])
            out_sum = torch.zeros_like(x[:, [0]])
            for i in range(6, x2.shape[-2]):
                out_max = torch.max(out_max, x[:, [i]])
                out_sum = out_sum + x[:, [i]]
            x = torch.cat([x_base, out_max], dim=-2)
        elif self.attention_type == 6:
            x = self._self_attn(x, key_padding_mask)
            x = x.mean(-2)
        else:
            x = x[:, self.query_index] + self._pa_block(x2, key_padding_mask) # original implementation

        # print('cross_attn', x[0][0])
        if self.attention_type != 6:
            x = x + self._ff_block(self.norm2(x))
        # print('final',x.reshape(*original_shape, -1)[0][0])
        return x.reshape(*original_shape, -1)

    def _self_attn(self, x, key_padding_mask = None):
        x2 = self.self_norm1(x)
        x = x + self.self_attn(x2, x2, x2, key_padding_mask, need_weights=False,)[0]
        x = x + self.self_linear2(self.self_activation(self.self_linear1(self.self_norm2(x))))
        return x

    def _pa_block(self, x: Tensor, key_padding_mask: Optional[Tensor] = None):
        # self.attn(query, key, value)
        if key_padding_mask is not None:
            key_padding_mask=key_padding_mask[:, self.source_index:]
        x = self.attn(
            x[:, self.query_index],
            x[:, self.source_index:],
            x[:, self.source_index:],
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return x

    def _pa_block2_1(self, x: Tensor, key_padding_mask: Optional[Tensor] = None):
        if key_padding_mask is not None:
            key_padding_mask=key_padding_mask[:, self.source_index:self.source_index2]
        x = self.attn(
            x[:, self.query_index],
            x[:, self.source_index:self.source_index2],
            x[:, self.source_index:self.source_index2],
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return x

    def _pa_block2_2(self, x: Tensor, key_padding_mask: Optional[Tensor] = None):
        if key_padding_mask is not None:
            key_padding_mask=key_padding_mask[:, self.source_index2:]
        x = self.attn2(
            x[:, self.query_index],
            x[:, self.source_index2:],
            x[:, self.source_index2:],
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return x

    def _ff_block(self, x: Tensor):
        x = self.linear2(self.activation(self.linear1(x)))
        return x
    

################################## Vision Encoders ##################################
def get_output_shape(net, input_size):
    _x = torch.zeros(input_size).unsqueeze(0)
    _out = net(_x)
    return _out.shape

class MixedEncoder(nn.Module):
    def __init__(
        self,
        cfg,
        vision_obs_names: List[str] = [],
        vision_encoder: nn.Module = None,
        state_encoder: Optional[nn.Module] = None,
        combine_mode: str = "concat",
    ):
        super().__init__()
        self.vision_obs_names = vision_obs_names
        self.vision_encoder = vision_encoder
        self.state_encoder = state_encoder
        self.combine_mode = combine_mode

        input_dim = self.vision_encoder.output_shape.numel()
        if self.state_encoder is not None:
            input_dim += self.state_encoder.output_shape.numel()

        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            MLP(
                num_units=[input_dim] + cfg.hidden_units, 
                normalization=nn.LayerNorm if cfg.get("layer_norm", False) else None
            ),
        )

        self.output_shape = torch.Size((cfg.hidden_units[-1],))

    def forward(self, x: TensorDict):
        feats = [
            self.vision_encoder(x[obs_name]) for obs_name in x.keys() 
            if obs_name in self.vision_obs_names
        ]
        if self.state_encoder is not None:
            # TODO: attn_encoder with TensorDict input
            feats += [
                self.state_encoder(x[obs_name]) for obs_name in x.keys() 
                if obs_name not in self.vision_obs_names
            ]
        if self.combine_mode == "concat":
            feats = torch.cat(feats, dim=-1)
        else:
            raise NotImplementedError
        return self.mlp(feats)


VISION_ENCODER_MAP = {}

@register(VISION_ENCODER_MAP)
class MobileNetV3Small(nn.Module):
    def __init__(
        self,
        input_size: torch.Size,
    ) -> None:
        super().__init__()
        self.input_size = input_size[2:]
        assert self.input_size[0] in [1, 3]

        self._setup_backbone()
        self.output_shape = get_output_shape(self.forward, self.input_size)

    def _backbone_fn(self, x):
        x = self._backbone_transform(x)
        x = self._backbone(x)
        return x

    def _setup_backbone(self):
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        _transform = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms

        full_backbone = mobilenet_v3_small(weights="IMAGENET1K_V1")
        for module in full_backbone.modules():
            if isinstance(module, nn.modules.batchnorm.BatchNorm2d):
                module.track_running_stats = False

        self._backbone = nn.Sequential(
            full_backbone.features,
            full_backbone.avgpool
        )
        self._backbone_transform = _transform()

    def forward(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self._backbone_fn(x)
        x = x.flatten(1)
        return x
    

@register(VISION_ENCODER_MAP)
class MobilNetV3Large(MobileNetV3Small):
    def _setup_backbone(self):
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        _transform = MobileNet_V3_Large_Weights.IMAGENET1K_V1.transforms

        full_backbone = mobilenet_v3_large(weights="IMAGENET1K_V1")
        for module in full_backbone.modules():
            if isinstance(module, nn.modules.batchnorm.BatchNorm2d):
                module.track_running_stats = False

        self._backbone = nn.Sequential(
            full_backbone.features,
            full_backbone.avgpool
        )
        self._backbone_transform = _transform()
        