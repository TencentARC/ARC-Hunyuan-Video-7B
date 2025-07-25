from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import Literal, Optional, Set, Tuple, TypedDict, TypeVar, Union, Any
from copy import deepcopy

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from transformers import (
    BatchEncoding,
    PretrainedConfig,
    TensorType,
    WhisperFeatureExtractor,
)
import math
import logging

from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargs,
    NestedTensors,
    MultiModalDataDict,
    MultiModalInputs,
)
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import AnyTokenizer

from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    flatten_bn,
    init_vllm_registered_model,
    maybe_prefix,
    merge_multimodal_embeddings,
    WeightsMapper,
)
from vllm.model_executor.models.whisper import WhisperEncoder
from vllm.multimodal.parse import (
    MultiModalDataParser,
    ModalityData,
    ModalityDataItems,
    DictEmbeddingItems,
    ProcessorBatchItems,
)
from vllm.multimodal.inputs import ImageItem
from vllm.transformers_utils.tokenizer import decode_tokens
from vllm.multimodal.hasher import MultiModalHasher


logger = logging.getLogger(__name__)


IMG_START = "<img>"
IMG_END = "</img>"
IMG_CONTEXT = "<IMG_CONTEXT>"


def _hunyuan_field_config(hf_inputs: Mapping[str, torch.Tensor]):

    image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
    image_grid_sizes = image_grid_thw.prod(-1)

    return dict(
        pixel_values_flat=MultiModalFieldConfig.batched("image"),
        image_embeds=MultiModalFieldConfig.batched("image"),
        image_grid_thw=MultiModalFieldConfig.batched("image"),
    )


class HunyuanMultiModalDataParser(MultiModalDataParser):

    def _parse_image_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[ImageItem]],
    ) -> ModalityDataItems[Any, Any]:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="image",
                required_fields={"image_embeds", "image_grid_thw"},
                fields_factory=_hunyuan_field_config,
            )

        return super()._parse_image_data(data)


class HunyuanImageEmbedInputs(TypedDict):
    type: Literal["image_embeds"]
    data: Union[torch.Tensor, list[torch.Tensor]]
    """ 
    A tensor of shape `(num_images, total_image_feature_size, hidden_size)`
    or a list of tensors of shape `(total_image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """

    image_grid_thw: torch.Tensor


class BaseHunyuanProcessor(ABC):

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: AnyTokenizer,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.num_image_token = config.num_image_token

    @property
    @abstractmethod
    def image_token_id(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_image_replace(
        self,
    ) -> PromptUpdateDetails[str]:
        raise NotImplementedError

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> Mapping[str, NestedTensors]:
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]

        if images is not None:
            raise NotImplementedError("Image processing not implemented")

        text_inputs = self.tokenizer(text)

        output = {
            **BatchEncoding(text_inputs, tensor_type=return_tensors),
        }
        return output


class HunyuanProcessor(BaseHunyuanProcessor):

    @property
    def image_token_id(self) -> int:
        image_token_id = self.tokenizer.get_vocab()[IMG_CONTEXT]
        return image_token_id

    def get_image_replace(
        self,
    ) -> PromptUpdateDetails[str]:
        replace_features = IMG_CONTEXT * self.num_image_token
        replace_full = IMG_START + replace_features + IMG_END

        return PromptUpdateDetails.select_text(replace_full, IMG_CONTEXT)
        # return PromptUpdateDetails(full=replace_full, features=replace_features)


class BaseHunyuanProcessingInfo(BaseProcessingInfo):

    @abstractmethod
    def get_hf_processor(
        self,
        **kwargs: object,
    ) -> BaseHunyuanProcessor:
        raise NotImplementedError

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"image": self.get_max_image_tokens()}

    def get_max_image_tokens(self) -> int:
        processor = self.get_hf_processor()
        num_image_token = processor.num_image_token
        return num_image_token


_I = TypeVar("_I", bound=BaseHunyuanProcessingInfo)


class HunyuanDummyInputsBuilder(BaseDummyInputsBuilder[_I]):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        num_images = mm_counts.get("image", 0)

        num_image_token = self.info.get_hf_processor().num_image_token
        hidden_size = self.info.get_hf_processor().config.hidden_size

        grid_hw = int((math.sqrt(4 * num_image_token - 7) - 1) / 2)

        grid_thw = torch.tensor([[1, grid_hw, grid_hw]])
        grid_thw = grid_thw.repeat(num_images, 1)

        mm_data = {
            "image": {
                "image_embeds": torch.randn(
                    num_images,
                    num_image_token,
                    hidden_size,
                    dtype=torch.bfloat16,
                ),
                "image_grid_thw": grid_thw,
            }
        }

        return ProcessorInputs(
            prompt_text="<image>" * num_images,
            mm_data=mm_data,
        )


class HunyuanMultiModalProcessor(BaseMultiModalProcessor[_I]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]:
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )
        return processed_outputs

    def _get_data_parser(self) -> HunyuanMultiModalDataParser:
        return HunyuanMultiModalDataParser()

    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, NestedTensors],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _hunyuan_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        image_replace = hf_processor.get_image_replace()

        return [
            PromptReplacement(
                modality="image",
                target="<image>",
                replacement=image_replace,
            ),
        ]


class HunyuanProcessingInfo(BaseHunyuanProcessingInfo):

    def get_hf_processor(
        self,
        **kwargs: object,
    ) -> HunyuanProcessor:
        return self.ctx.init_processor(
            HunyuanProcessor,
            config=self.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            **kwargs,
        )


@MULTIMODAL_REGISTRY.register_processor(
    HunyuanMultiModalProcessor,
    info=HunyuanProcessingInfo,
    dummy_inputs=HunyuanDummyInputsBuilder,
)
class HunyuanVideoModel(nn.Module, SupportsMultiModal, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.config = config

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["HunYuanForCausalLM"],
        )

        self.system_message = None
        self.num_samples = 0

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler
        else:
            raise NotImplementedError

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:

        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                self.config.image_token_id,
            )
        return inputs_embeds

    def get_multimodal_embeddings(
        self, **kwargs: object
    ) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)

        if image_input is None:
            return None

        return image_input["data"]

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Optional[HunyuanImageEmbedInputs]:
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if image_embeds is None:
            return None

        if not isinstance(image_embeds, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of image embeddings. "
                f"Got type: {type(image_embeds)}"
            )
        
        image_embeds = image_embeds.to(self.config.torch_dtype)

        return HunyuanImageEmbedInputs(
            type="image_embeds",
            data=flatten_bn(image_embeds),
            image_grid_thw=flatten_bn(image_grid_thw),
        )

    def _process_image_input(
        self, image_input: HunyuanImageEmbedInputs
    ) -> MultiModalEmbeddings:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["data"]
        
        merge_size = 1  # TODO: Check this
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return image_embeds.split(sizes.tolist())

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[SamplerOutput, IntermediateTensors]:

        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None
        elif inputs_embeds is None:
            # raise ValueError(f"v0 not supported, {kwargs}")
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(
            hidden_states, sampling_metadata
        )

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Set[str]:
        loader = AutoWeightsLoader(self)
        weights = list(weights)
        for i, (k, v) in enumerate(weights):
            # The order of SiLU and Mul is different in VLLM
            if ".mlp.gate_and_up_proj.weight" in k:
                v1, v2 = v.chunk(2, dim=0)
                weights[i] = (k, torch.cat([v2, v1], dim=0))

        # Filter out weights that are not in the language model (vit, whisper, mlp2)
        weights = [(k, v) for k, v in weights if k.startswith("language_model")]

        if "language_model.lm_head.weight" not in weights:
            logger.warning(
                "langauge.lm_head.weight not found in weights, "
                "will try to load it from language_model.embed_tokens.weight"
            )
            weights.append(("language_model.lm_head.weight", self.language_model.model.embed_tokens.weight))

        return loader.load_weights(weights)
