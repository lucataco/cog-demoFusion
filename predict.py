# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import sys
import torch
from PIL import Image
from typing import List
from diffusers import DiffusionPipeline
from weights_downloader import WeightsDownloader
sys.path.extend(['/DemoFusion'])
from pipeline_demofusion_sdxl import DemoFusionSDXLPipeline


MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
MODEL_CACHE="model-cache"
SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-upcast-fix.tar"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        WeightsDownloader.download_if_not_exists(SDXL_URL, MODEL_CACHE)
        pipe = DemoFusionSDXLPipeline.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        self.pipe = pipe.to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="blurry, ugly, duplicate, poorly drawn, deformed, mosaic",
        ),
        width: int = Input(
            description="Width of output image",
            default=3072,
        ),
        height: int = Input(
            description="Height of output image",
            default=3072,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        view_batch_size: int = Input(
            description="The batch size for multiple denoising paths",
            default=16,
        ),
        stride: int = Input(
            description="The stride of moving local patches",
            default=16,
        ),
        cosine_scale_1: float = Input(
            description="Control the strength of skip-residual",
            default=3.0,
        ),
        cosine_scale_2: float = Input(
            description="Control the strength of dilated sampling",
            default=1.0,
        ),
        cosine_scale_3: float = Input(
            description="Control the strength of the Gaussian filter",
            default=1.0,
        ),
        sigma: float = Input(
            description="The standard value of the Gaussian filter",
            default=1.0,
        ),
        multi_decoder: bool = Input(
            description="Use multiple decoders",
            default=True,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        images = self.pipe(
            prompt=prompt, 
            negative_prompt=negative_prompt,
            generator=generator,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale = guidance_scale,
            view_batch_size=view_batch_size,
            stride=stride,
            cosine_scale_1=cosine_scale_1,
            cosine_scale_2=cosine_scale_2,
            cosine_scale_3=cosine_scale_3,
            sigma=sigma, 
            multi_decoder=multi_decoder
        )

        output_paths = []
        for i, image in enumerate(images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths