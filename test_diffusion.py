from diffusers.pipelines.diffusion.pipeline_diffusion import DiffusionPipeline
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

import torch

model_id = "sd-legacy/stable-diffusion-v1-5"
dummy_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
unet_config = dummy_pipe.unet.config
unet_config["in_channels"] = 3
unet_config["out_channels"] = 3

pipe = DiffusionPipeline(
    text_encoder=dummy_pipe.text_encoder,
    tokenizer=dummy_pipe.tokenizer,
    unet=UNet2DConditionModel.from_config(unet_config).to(torch.float16),
    scheduler=dummy_pipe.scheduler,
    safety_checker=dummy_pipe.safety_checker,
    feature_extractor=dummy_pipe.feature_extractor,
).to("cuda")

prompt = "A photo of a cat"
image = pipe(
    prompt,
    height=64,
    width=64,
).images[0]

print(image.size)