
import os
import torch
import argparse
from PIL import Image
from rembg import remove
from diffusers import StableDiffusionPipeline

from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

parser = argparse.ArgumentParser(description="Generate 3D model from prompt using Stable Diffusion and Hunyuan3D")
parser.add_argument("prompt", type=str, help="Text prompt for image generation")
args = parser.parse_args()
prompt = f"{args.prompt}, centered in the frame"

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
image = pipe(prompt).images[0]

image_no_bg = remove(image)

output_path = "assets/demo.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
image_no_bg.save(output_path)

print("image is generated!")

# let's generate a mesh first
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(image='assets/demo.png')[0]

print("mesh is generated!")

pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(mesh, image='assets/demo.png')

print("texture is generated!")

output_path = "logs/result.glb"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
mesh.export(output_path)