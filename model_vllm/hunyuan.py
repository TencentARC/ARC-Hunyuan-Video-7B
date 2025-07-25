from functools import partial
from typing import Any, Dict, Iterable, Optional, Set, Tuple, Type, Union, List

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import (
    get_rope,
    RotaryEmbedding,
    MRotaryEmbedding,
    _apply_rotary_emb,
)
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput

from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    AutoWeightsLoader,
)


class DynamicNTKAlphaRotaryEmbedding(RotaryEmbedding):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        scaling_alpha: float,
        dtype: torch.dtype,
        is_neox_style: bool = True,
    ) -> None:
        self.scaling_alpha = scaling_alpha
        super().__init__(
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
        )

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        # NOTE(woosuk): self.max_position_embeddings is the original
        # maximum length before applying the rope scaling.
        # Thus, the maximum length after applying the rope scaling is
        # self.max_position_embeddings * self.scaling_alpha.
        max_len = self.max_position_embeddings * self.scaling_alpha
        base = self.base * self.scaling_alpha ** (
            self.rotary_dim / (self.rotary_dim - 2)
        )
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim)
        )
        t = torch.arange(max_len, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class DynamicNTKAlphaMRotaryEmbedding(MRotaryEmbedding):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        scaling_alpha: float,
        dtype: torch.dtype,
        mrope_section: Optional[List[int]] = None,
        is_neox_style: bool = True,
    ) -> None:
        self.scaling_alpha = scaling_alpha
        assert len(mrope_section) == 4, "Currently only 4D is supported"
        mrope_section = [int(x * rotary_dim // 2) for x in mrope_section]

        # MRotaryEmbedding will enlarge the max_position_embeddings by 4
        # To keep consistent with the original max_position_embeddings,
        # we need to divide the max_position_embeddings by 4
        max_position_embeddings = max_position_embeddings // 4

        super().__init__(
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
            mrope_section,
        )
    
    def _compute_cos_sin_cache(self) -> torch.Tensor:
        max_len = self.max_position_embeddings * self.scaling_alpha
        # max_len = self.max_position_embeddings  # Check this
        base = self.base * self.scaling_alpha ** (
            self.rotary_dim / (self.rotary_dim - 2)
        )
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim)
        )
        t = torch.arange(max_len, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache
    

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """XDRope implementation following apply_rotary_pos_emb_xdrope pattern.

        Args:
            positions:
                [num_tokens,] (text only) or
                [4, num_tokens] (4D positions with multimodal inputs)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        assert positions.ndim == 2, f"positions must be 2D, but got {positions.shape}"

        num_tokens = positions.shape[-1]
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)

        x_dim = len(self.mrope_section)

        cos = cos.permute(1, 0, 2).reshape(-1, x_dim, self.rotary_dim)
        sin = sin.permute(1, 0, 2).reshape(-1, x_dim, self.rotary_dim)

        xdrope_section = self.mrope_section * 2
        assert sum(xdrope_section) == self.rotary_dim

        cos = torch.cat([
            m[:, i % x_dim] for i, m in enumerate(cos.split(xdrope_section, dim=-1))
        ], dim=-1)
        sin = torch.cat([
            m[:, i % x_dim] for i, m in enumerate(sin.split(xdrope_section, dim=-1))
        ], dim=-1)

        cos = cos.view(1, -1, self.rotary_dim)
        sin = sin.view(1, -1, self.rotary_dim)

        cos = cos.permute(1, 0, 2)
        sin = sin.permute(1, 0, 2)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query = (query * cos) + rotate_half(query) * sin
        query = query.reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key = (key * cos) + rotate_half(key) * sin
        key = key.reshape(key_shape)

        return query, key


    @classmethod
    def get_input_positions(
        cls,
        input_tokens: List[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Optional[Union[List[List[int]], torch.Tensor]],
        video_grid_thw: Optional[Union[List[List[int]], torch.Tensor]],
        second_per_grid_ts: Optional[List[float]],
        context_len: int = 0,
        seq_len: Optional[int] = None,
        audio_feature_lengths: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
    ) -> Tuple[List[List[int]], int]:
        """Get xdrope input positions and delta value."""

        image_grid_thw = [] if image_grid_thw is None else image_grid_thw
        video_grid_thw = [] if video_grid_thw is None else video_grid_thw
        second_per_grid_ts = [] if second_per_grid_ts is None else \
            second_per_grid_ts

        llm_positions, mrope_position_delta = \
            cls.get_input_positions_tensor(
                input_tokens=input_tokens,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                context_len=context_len,
                seq_len=seq_len,
                audio_feature_lengths=audio_feature_lengths,
                use_audio_in_video=use_audio_in_video,
            )

        return llm_positions.tolist(), mrope_position_delta


    @classmethod
    def get_input_positions_tensor(
        cls,
        input_tokens: List[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[List[List[int]], torch.Tensor],
        video_grid_thw: Union[List[List[int]], torch.Tensor],
        second_per_grid_ts: List[float],
        context_len: int = 0,
        seq_len: Optional[int] = None,
        audio_feature_lengths: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
    ) -> Tuple[torch.Tensor, int]:
        return cls._vl_get_input_positions_tensor(
            input_tokens=input_tokens,
            hf_config=hf_config,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            context_len=context_len,
            seq_len=seq_len,
        )


    @classmethod
    def _vl_get_input_positions_tensor(
        cls,
        input_tokens: List[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[List[List[int]], torch.Tensor],
        video_grid_thw: Union[List[List[int]], torch.Tensor],
        second_per_grid_ts: List[float],
        context_len: int = 0,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Get xdrope input positions following get_xdrope_position_ids pattern."""


        image_token_id = hf_config.image_token_id
        vision_start_token_id = hf_config.vision_start_token_id

        input_tokens_tensor = torch.tensor(input_tokens)

        # Initialize 4D position embeddings (following xdrope pattern)
        seq_length = len(input_tokens)
        position_ids_seq = torch.arange(seq_length)  # Sequential positions
        position_ids_t = position_ids_seq.clone()
        position_ids_x = position_ids_seq.clone()
        position_ids_y = position_ids_seq.clone()

        vision_start_indices = torch.argwhere(
            input_tokens_tensor == vision_start_token_id).squeeze(1)
        
        if len(vision_start_indices) == 0:
            # No vision tokens, return 4D sequential positions
            llm_positions = torch.stack([position_ids_seq, position_ids_x, position_ids_y, position_ids_t])
            mrope_position_delta = 0
            llm_positions = llm_positions[:, context_len:seq_len]
            return llm_positions, mrope_position_delta

        # Process vision tokens using image_grid_thw information
        image_index, video_index = 0, 0
        current_pos = 0
        
        for start_idx in vision_start_indices:
            start_idx = start_idx.item()
            
            # Determine if this is image or video token
            if start_idx + 1 < len(input_tokens):
                next_token = input_tokens[start_idx + 1]
                is_image = (next_token == image_token_id)
                
                if is_image and image_index < len(image_grid_thw):
                    t, h, w = image_grid_thw[image_index]
                    image_index += 1
                else:
                    continue
                
                # Calculate grid dimensions
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t, h, w
                )
                
                # Find end of vision tokens (approximate)
                vision_token_count = llm_grid_t * llm_grid_h * llm_grid_w
                end_idx = min(start_idx + vision_token_count + 2, seq_length)  # +2 for start/end tokens
                
                # Apply xdrope position assignment pattern
                if end_idx > start_idx + 2:  # Ensure we have vision tokens
                    # Reset time dimension for vision tokens (following get_xdrope_position_ids)
                    position_ids_t[start_idx + 2:end_idx] = current_pos
                    current_pos += 1
                    
                    # Calculate row and column for 2D layout
                    vision_tokens_between = end_idx - start_idx - 2  # excluding start/end
                    if llm_grid_h > 0:
                        tokens_per_row = llm_grid_w
                        num_rows = llm_grid_h
                        
                        # Assign x,y coordinates following the pattern
                        idx_xy = 0
                        for rr in range(num_rows):
                            for cc in range(tokens_per_row):
                                if start_idx + 2 + idx_xy < end_idx:
                                    position_ids_x[start_idx + 2 + idx_xy] = cc
                                    position_ids_y[start_idx + 2 + idx_xy] = rr
                                    idx_xy += 1

        # Stack into 4D positions
        llm_positions = torch.stack([position_ids_seq, position_ids_x, position_ids_y, position_ids_t])
        mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
        llm_positions = llm_positions[:, context_len:seq_len]

        return llm_positions, mrope_position_delta


    @staticmethod
    def get_next_input_positions(
        mrope_position_delta: int,
        context_len: int,
        seq_len: int,
    ) -> List[List[int]]:
        return [
            list(
                range(context_len + mrope_position_delta,
                      seq_len + mrope_position_delta)) for _ in range(4)  # Changed from 3 to 4
        ]


    @staticmethod
    def get_next_input_positions_tensor(
        mrope_position_delta: int,
        context_len: int,
        seq_len: int,
    ) -> torch.Tensor:
        return torch.arange(
            mrope_position_delta + context_len,
            mrope_position_delta + seq_len,
        ).expand(4, -1)  # Changed from 3 to 4


class HunyuanMLP(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        intermediat_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_and_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediat_size] * 2,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_and_up_proj",
        )
        # self.down_proj = ReplicatedLinear(
            # intermediat_size,
            # hidden_size,
            # bias=config.mlp_bias,
            # quant_config=quant_config,
            # prefix=f"{prefix}.down_proj",
        # )
        self.down_proj = nn.Linear(
            intermediat_size,
            hidden_size,
            bias=config.mlp_bias,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_and_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class HunYuanAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        attention_bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.total_num_heads = num_heads
        assert self.total_num_heads % self.tp_size == 0
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= self.tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert self.tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.key_value_groups = int(self.num_heads / self.num_kv_heads)
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.wqkv",
        )

        # self.o_proj = RowParallelLinear(
        #     self.total_num_heads * self.head_dim,
        #     hidden_size,
        #     bias=attention_bias,
        #     quant_config=quant_config,
        #     prefix=f"{prefix}.wo",
        # )
        self.o_proj = nn.Linear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
        )

        self.query_layernorm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if config.use_qk_norm
            else None
        )
        self.key_layernorm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if config.use_qk_norm
            else None
        )

        self.rotary_emb = (
            DynamicNTKAlphaMRotaryEmbedding(
                self.head_dim,
                self.head_dim,
                max_position_embeddings,
                int(rope_theta),
                scaling_alpha=rope_scaling["alpha"],
                dtype=torch.get_default_dtype(),
                mrope_section=rope_scaling["mrope_section"],
            )
            if config.use_rotary_pos_emb
            else None
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def split_qkv(self, qkv: torch.Tensor):
        seq_len = qkv.shape[0]
        if self.tp_size > 1:
            qkv_map = [self.q_size, self.kv_size, self.kv_size] * self.tp_size
            qkv = tensor_model_parallel_all_gather(qkv)
            qkv = torch.split(qkv, qkv_map, dim=-1)
            qkv = qkv[::3] + qkv[1::3] + qkv[2::3]
            qkv = torch.cat(qkv, dim=-1)

        qkv = qkv.view(
            seq_len,
            self.total_num_kv_heads,
            self.key_value_groups + 2,
            self.head_dim,
        )
        q, k, v = torch.split(qkv, [self.key_value_groups, 1, 1], dim=-2)
        q = q.reshape(seq_len, self.q_size * self.tp_size)
        k = k.reshape(seq_len, self.kv_size * self.tp_size)
        v = v.reshape(seq_len, self.kv_size * self.tp_size)

        if self.tp_size > 1:
            splitter = partial(
                split_tensor_along_last_dim, num_partitions=self.tp_size
            )
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
            v = splitter(v)[self.tp_rank]
        return q, k, v

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = self.split_qkv(qkv)

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)


        if self.query_layernorm is not None:
            q = q.reshape(-1, self.num_heads, self.head_dim)
            q = self.query_layernorm(q).reshape(-1, self.q_size)

        if self.key_layernorm is not None:
            k = k.reshape(-1, self.num_kv_heads, self.head_dim)
            k = self.key_layernorm(k).reshape(-1, self.kv_size)

        attn_output = self.attn(q, k, v)
        output = self.o_proj(attn_output)
        return output


class HunYuanDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig,
        quant_config: QuantizationConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = HunYuanAttention(
            config,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
            config.attention_bias,
            cache_config,
            quant_config,
            prefix=f"{prefix}.attention",
        )
        self.mlp = HunyuanMLP(
            config,
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual
            )

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class HunYuanModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: Type[HunYuanDecoderLayer] = HunYuanDecoderLayer,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )  # TODO: This does not support padding_idx, check if this is an issue

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: layer_type(
                config, cache_config, quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)
        if not get_pp_group().is_first_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class HunYuanForCausalLM(nn.Module, SupportsPP):

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        model_type: Type[HunYuanModel] = HunYuanModel,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.model = model_type(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        assert positions.ndim == 2, f"positions must be 2D, but got {positions.shape}"
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(
            self.lm_head, hidden_states, sampling_metadata
        )
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
