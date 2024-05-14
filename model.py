import math
from collections import OrderedDict
from typing import List, Optional

import torch
from comfy import model_management
from comfy.model_patcher import ModelPatcher
from safetensors.torch import load_file, load_model
from torch import nn

from .activations import get_activation
from .utils import remove_weights


class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, time_embedding_dim: Optional[int] = None):
        super().__init__()

        if time_embedding_dim is None:
            time_embedding_dim = embedding_dim

        self.silu = nn.SiLU()
        self.linear = nn.Linear(time_embedding_dim, 2 * embedding_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, timestep_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(timestep_embedding))
        shift, scale = emb.view(len(x), 1, -1).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.square(torch.relu(x))


class PerceiverAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, time_embedding_dim: Optional[int] = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("sq_relu", SquaredReLU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )

        self.ln_1 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_2 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_ff = AdaLayerNorm(d_model, time_embedding_dim)

    def attention(self, q: torch.Tensor, kv: torch.Tensor):
        attn_output, attn_output_weights = self.attn(q, kv, kv, need_weights=False)
        return attn_output

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        timestep_embedding: Optional[torch.Tensor] = None,
    ):
        normed_latents = self.ln_1(latents, timestep_embedding)
        latents = latents + self.attention(
            q=normed_latents,
            kv=torch.cat([normed_latents, self.ln_2(x, timestep_embedding)], dim=1),
        )
        return latents + self.mlp(self.ln_ff(latents, timestep_embedding))


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        width: int = 768,
        layers: int = 6,
        heads: int = 8,
        num_latents: int = 64,
        output_dim: Optional[int] = None,
        input_dim: Optional[int] = None,
        time_embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.latents = nn.Parameter(width**-0.5 * torch.randn(num_latents, width))
        self.time_aware_linear = nn.Linear(time_embedding_dim or width, width, bias=True)

        if input_dim is not None:
            self.proj_in = nn.Linear(input_dim, width)

        self.perceiver_blocks = nn.Sequential(
            *[PerceiverAttentionBlock(width, heads, time_embedding_dim=time_embedding_dim) for _ in range(layers)]
        )

        if output_dim is not None:
            self.proj_out = nn.Sequential(nn.Linear(width, output_dim), nn.LayerNorm(output_dim))

    def forward(self, x: torch.Tensor, timestep_embedding: torch.Tensor, **kwargs):
        learnable_latents = self.latents.unsqueeze(dim=0).repeat(len(x), 1, 1)
        latents = learnable_latents + self.time_aware_linear(torch.nn.functional.silu(timestep_embedding))
        if self.input_dim is not None:
            x = self.proj_in(x)
        for p_block in self.perceiver_blocks:
            latents = p_block(x, latents, timestep_embedding=timestep_embedding)

        if self.output_dim is not None:
            latents = self.proj_out(latents)

        return latents


class GatedPerceiverAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        time_embedding_dim: Optional[int] = None,
        separated_gate=False,
        **kwargs,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("sq_relu", SquaredReLU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )

        self.ln_1 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_2 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_ff = AdaLayerNorm(d_model, time_embedding_dim)

        self.gate_attn = nn.Parameter(torch.tensor([0.0]))
        self.gate_ff = nn.Parameter(torch.tensor([0.0]))

        self.separated_gate = separated_gate
        if self.separated_gate:
            self.gate_separated = nn.Linear(d_model, 1)

    def attention(self, q: torch.Tensor, kv: torch.Tensor):
        attn_output, _ = self.attn(q, kv, kv, need_weights=False)
        return attn_output

    def get_gate_scale(self, latents, scale: float):
        if self.separated_gate:
            separated_gate = torch.nn.functional.softmax(self.gate_separated(latents), dim=1)  # B x num_tokens x 1
            return scale * separated_gate
        return scale

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        timestep_embedding: Optional[torch.Tensor] = None,
        gate_scale=1.0,
    ):
        normed_latents = self.ln_1(latents, timestep_embedding)

        scale = self.get_gate_scale(latents, gate_scale)
        attn_gate = self.gate_attn.tanh() * scale
        ffn_gate = self.gate_ff.tanh() * scale

        latents = latents + attn_gate * self.attention(
            q=normed_latents, kv=torch.cat([normed_latents, self.ln_2(x, timestep_embedding)], dim=1)
        )
        return latents + ffn_gate * self.mlp(self.ln_ff(latents, timestep_embedding))


class GatedPerceiver(nn.Module):
    def __init__(
        self,
        input_dim: Optional[int] = None,
        width: int = 768,
        layers: int = 6,
        heads: int = 8,
        time_embedding_dim: Optional[int] = 768,
        separated_gate=True,
    ):
        super().__init__()
        # input_dim from additional_input_dim
        self.input_dim = input_dim or width
        self.additional_proj_in = nn.Linear(self.input_dim, width)
        self.gated_perceiver_blocks = nn.Sequential(
            *[
                GatedPerceiverAttentionBlock(
                    width,
                    heads,
                    time_embedding_dim=time_embedding_dim,
                    separated_gate=separated_gate,
                )
                for _ in range(layers)
            ]
        )
        self.load_device = model_management.text_encoder_device()
        self.offload_device = model_management.text_encoder_offload_device()
        self.dtype = model_management.text_encoder_dtype(self.load_device)
        self.to(self.dtype) # type: ignore
        self.patcher = ModelPatcher(self, load_device=self.load_device, offload_device=self.offload_device)

    @classmethod
    def from_pretrained(cls, path):
        # TODO: Better model structure analysis
        sd = load_file(path)
        sd = {
            k.replace("connector.", "").replace("mapper.", ""): v
            for k, v in sd.items()
            if ('additional' in k or 'gated' in k)
        }
        input_dim = sd["additional_proj_in.weight"].shape[1]
        separated_gate = "gated_perceiver_blocks.0.gate_separated.weight" in sd
        gp = GatedPerceiver(input_dim=input_dim, separated_gate=separated_gate)
        gp.load_state_dict(sd)
        return gp

    def load_model(self):
        model_management.load_model_gpu(self.patcher)
        return self.patcher


def finegrained_attention_merge(
    layers: List[GatedPerceiverAttentionBlock],
    gate_scales: List[float],
    conditions: List[torch.Tensor],
    latents: torch.Tensor,
    timestep_embedding: torch.Tensor,
):
    normed_latents = [layer.ln_1(latents, timestep_embedding) for layer in layers]

    # gates
    scales = [layer.get_gate_scale(latents, s) for layer, s in zip(layers, gate_scales)]
    attn_gates = [s * layer.gate_attn.tanh() for layer, s in zip(layers, scales)]
    ffn_gates = [s * layer.gate_ff.tanh() for layer, s in zip(layers, scales)]

    # attention
    attn_results = [
        layer.attention(q=nl, kv=torch.cat([nl, layer.ln_2(c, timestep_embedding)], dim=1))
        for layer, nl, c in zip(layers, normed_latents, conditions)
    ]

    latents = latents + sum([attn_gate * ar for attn_gate, ar in zip(attn_gates, attn_results)])
    ffn_results = [layer.mlp(layer.ln_ff(latents, timestep_embedding)) for layer in layers]
    return latents + sum([ffn_gate * ffn_r for ffn_gate, ffn_r in zip(ffn_gates, ffn_results)])


class GatedMultiPerceiverResampler(nn.Module):
    def __init__(
        self,
        width: int = 768,
        layers: int = 6,
        heads: int = 8,
        num_latents: int = 64,
        output_dim: Optional[int] = None,
        input_dim: Optional[int] = None,
        time_embedding_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.latents = nn.Parameter(width**-0.5 * torch.randn(num_latents, width))
        self.time_aware_linear = nn.Linear(time_embedding_dim or width, width, bias=True)

        if input_dim is not None:
            self.proj_in = nn.Linear(input_dim, width)

        self.perceiver_blocks = nn.Sequential(
            *[PerceiverAttentionBlock(width, heads, time_embedding_dim=time_embedding_dim) for _ in range(layers)]
        )

        if output_dim is not None:
            self.proj_out = nn.Sequential(nn.Linear(width, output_dim), nn.LayerNorm(output_dim))

    def forward(
        self,
        x: torch.Tensor,
        timestep_embedding: torch.Tensor,
        emmas: Optional[list] = None,
        **kwargs,
    ):
        learnable_latents = self.latents.unsqueeze(dim=0).repeat(len(x), 1, 1)
        latents = learnable_latents + self.time_aware_linear(torch.nn.functional.silu(timestep_embedding))
        if self.input_dim is not None:
            x = self.proj_in(x)

        emmas = emmas or []
        [e[0].load_model() for e in emmas]
        additional_x = [
            ap.additional_proj_in(c.to(device=ap.load_device, dtype=ap.dtype))
            for ap, c, _ in emmas
        ]

        for layer_id, p_block in enumerate(self.perceiver_blocks):
            latents = p_block(x, latents, timestep_embedding=timestep_embedding)
            latents = finegrained_attention_merge(
                [ap.gated_perceiver_blocks[layer_id] for ap, _, _ in emmas],  # type: ignore
                [s for _, _, s in emmas],
                additional_x,
                latents,
                timestep_embedding,
            )

        if self.output_dim is not None:
            latents = self.proj_out(latents)

        return latents


class T5TextEmbedder:
    def __init__(self, pretrained_path="google/flan-t5-xl", max_length=None, dtype=None, legacy=True):
        self.load_device = model_management.text_encoder_device()
        self.offload_device = model_management.text_encoder_offload_device()
        self.dtype = dtype if dtype is not None else model_management.text_encoder_dtype(self.load_device)
        self.output_device = model_management.intermediate_device()
        self.max_length = max_length
        from transformers import T5EncoderModel, T5Tokenizer

        self.model = T5EncoderModel.from_pretrained(pretrained_path).to(self.dtype)  # type: ignore
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_path, legacy=legacy)
        self.patcher = ModelPatcher(self.model, load_device=self.load_device, offload_device=self.offload_device)

    def load_model(self):
        model_management.load_model_gpu(self.patcher)
        return self.patcher

    def __call__(self, caption, text_input_ids=None, attention_mask=None, max_length=None, **kwargs):
        self.load_model()
        # remove a1111/comfyui prompt weight, t5 embedder currently does not accept weight
        caption = remove_weights(caption)
        if max_length is None:
            max_length = self.max_length

        if text_input_ids is None or attention_mask is None:
            if max_length is not None:
                text_inputs = self.tokenizer(
                    caption,
                    return_tensors="pt",
                    add_special_tokens=True,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                )
            else:
                text_inputs = self.tokenizer(caption, return_tensors="pt", add_special_tokens=True)
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
        text_input_ids = text_input_ids.to(self.model.device)  # type: ignore
        attention_mask = attention_mask.to(self.model.device)  # type: ignore
        outputs = self.model(text_input_ids, attention_mask=attention_mask)  # type: ignore

        return outputs.last_hidden_state.to(self.output_device)


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: Optional[int] = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()
        linear_cls = nn.Linear

        self.linear_1 = linear_cls(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = linear_cls(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)  # type: ignore
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )


class ELLAModel(nn.Module):
    def __init__(
        self,
        time_channel=320,
        time_embed_dim=768,
        act_fn: str = "silu",
        out_dim: Optional[int] = None,
        width=768,
        layers=6,
        heads=8,
        num_latents=64,
        input_dim=2048,
    ):
        super().__init__()
        self.position = Timesteps(time_channel, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(
            in_channels=time_channel,
            time_embed_dim=time_embed_dim,
            act_fn=act_fn,
            out_dim=out_dim,  # type: ignore
        )

        self.connector = GatedMultiPerceiverResampler(
            width=width,
            layers=layers,
            heads=heads,
            num_latents=num_latents,
            input_dim=input_dim,
            time_embedding_dim=time_embed_dim,
        )

    def forward(self, timesteps: torch.Tensor, t5_embeds: torch.Tensor, emmas: Optional[list] = None, **kwargs):
        device = t5_embeds.device
        dtype = t5_embeds.dtype

        ori_time_feature = self.position(timesteps.view(-1)).to(device, dtype=dtype)
        ori_time_feature = ori_time_feature.unsqueeze(dim=1) if ori_time_feature.ndim == 2 else ori_time_feature
        ori_time_feature = ori_time_feature.expand(len(t5_embeds), -1, -1)
        time_embedding = self.time_embedding(ori_time_feature)

        return self.connector(t5_embeds, timestep_embedding=time_embedding, emmas=emmas)


class ELLA:
    def __init__(self, path: str, no_init=False, **kwargs) -> None:
        if no_init:
            return
        load_device = model_management.text_encoder_device()
        offload_device = model_management.text_encoder_offload_device()
        self.model = ELLAModel()
        unexpected_keys = load_model(self.model, path, strict=False)
        print(f"{unexpected_keys}")
        self.dtype = model_management.text_encoder_dtype(load_device)
        self.model.to(self.dtype)  # type: ignore
        self.patcher = ModelPatcher(self.model, load_device=load_device, offload_device=offload_device)

    def clone(self):
        n = ELLA("", no_init=True)
        n.dtype = self.dtype
        n.model = self.model
        n.patcher = self.patcher.clone()
        return n

    def load_model(self):
        model_management.load_model_gpu(self.patcher)
        return self.patcher

    def __call__(self, timesteps: torch.Tensor, t5_embeds: torch.Tensor, emmas: Optional[list] = None, **kwargs):
        self.load_model()
        timesteps = timesteps.to(device=self.patcher.load_device, dtype=torch.int64)
        t5_embeds = t5_embeds.to(device=self.patcher.load_device, dtype=self.dtype)  # type: ignore
        cond = self.model(timesteps, t5_embeds, emmas=emmas, **kwargs)
        return cond.to(model_management.intermediate_device())
