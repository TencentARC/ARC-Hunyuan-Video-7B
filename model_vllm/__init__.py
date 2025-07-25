from vllm import ModelRegistry
from .hunyuan import HunYuanForCausalLM
from .hunyuan_video import HunyuanVideoModel
from .video_audio_encoder import VideoAudioEncoder
from .video_audio_llm import VideoAudioLLM

ModelRegistry.register_model("HunYuanForCausalLM", HunYuanForCausalLM)
ModelRegistry.register_model("HunyuanVideoModel", HunyuanVideoModel)
