import torch
import torch.nn as nn
import math
from transformers.modeling_utils import no_init_weights
from transformers import WhisperModel, AutoConfig

from transformers import Qwen2_5_VisionTransformerPretrainedModel

class VideoAudioEncoder(nn.Module):
    def __init__(self, config, max_num_frames=300):
        super().__init__()
        self.max_num_frames = max_num_frames
        self.image_size = config.image_size
        self.num_frame_per_token = int(self.image_size * self.image_size / (14 * 14 * 4 * 2))
        self.num_audio_per_second = 50
        config.vision_config._attn_implementation = "flash_attention_2"

        with no_init_weights():
            # Initialize vision model
            self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
            whisper_path = 'openai/whisper-large-v3'
            self.speech_encoder = WhisperModel.from_pretrained(whisper_path).encoder.to(torch.bfloat16)

        speech_dim = self.speech_encoder.config.d_model
        llm_hidden_size = config.vision_config.out_hidden_size
        self.mlp_speech = nn.Sequential(
            nn.LayerNorm(speech_dim),
            nn.Linear(speech_dim, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        ).to(torch.bfloat16)

    @torch.no_grad()
    def extract_image_feature(self, pixel_values, video_grid_thw):
        """Extract features from image tensors using vision model"""
        vit_embeds = self.visual(pixel_values, grid_thw=video_grid_thw)
        return vit_embeds

    @torch.no_grad()
    def extract_audio_feature(self, values_audios):
        """Extract features from audio tensors using speech encoder"""
        values_audios = values_audios.squeeze(0).reshape(-1, 128, values_audios.shape[-1])
        speech_embeds = self.speech_encoder(values_audios, return_dict=True).last_hidden_state
        speech_embeds = speech_embeds.reshape(-1, speech_embeds.shape[-1])
        speech_embeds = self.mlp_speech(speech_embeds)
        return speech_embeds

    def create_mixed_embeddings(self, video_embeds, speech_embeds):
        """Create mixed embeddings from visual and audio features"""
        # Reshape audio embeddings to match video frames
        duration = int(video_embeds.shape[0] / self.num_frame_per_token)
        speech_embeds_trunc = speech_embeds.reshape(-1, self.num_audio_per_second, speech_embeds.shape[-1])[:duration]
        
        video_embeds_reshape = video_embeds.reshape(-1, self.num_frame_per_token*2, video_embeds.shape[-1])
        speech_embeds_trunc_reshape = speech_embeds_trunc.reshape(-1, self.num_audio_per_second*2, speech_embeds_trunc.shape[-1])
        num_pad_token = video_embeds_reshape.shape[1] - speech_embeds_trunc_reshape.shape[1]

        zero_padding = torch.zeros(speech_embeds_trunc_reshape.shape[0], num_pad_token, speech_embeds_trunc_reshape.shape[-1]).to(speech_embeds_trunc_reshape.dtype).to(speech_embeds_trunc_reshape.device)
        speech_embeds_trunc_reshape_pad = torch.cat((speech_embeds_trunc_reshape, zero_padding), dim=1)
        mixed_embeds = video_embeds_reshape + speech_embeds_trunc_reshape_pad
        mixed_embeds = mixed_embeds.reshape(-1, mixed_embeds.shape[-1])

        return mixed_embeds

    def forward(self, device, pixel_values, video_grid_thw, audio_values):
        """
        Encode images and audio to create mixed embeddings

        Args:
            pixel_values (torch.Tensor): Batch of images from video (processed frames)
            audio_values (torch.Tensor): Processed audio features
        Returns:
            mixed_embeds (torch.Tensor): Mixed embeddings combining vision and audio
        """

        # Extract features
        with torch.no_grad(), torch.autocast(device.type, torch.bfloat16):
            vit_embeds = self.extract_image_feature(pixel_values, video_grid_thw)

        with torch.no_grad():
            audio_embeds = self.extract_audio_feature(audio_values)

            # Create mixed embeddings
            mixed_embeds = self.create_mixed_embeddings(
                vit_embeds, audio_embeds
            )

        return mixed_embeds
