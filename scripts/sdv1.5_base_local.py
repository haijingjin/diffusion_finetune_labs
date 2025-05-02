from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path
import datetime

# Define paths
script_dir = Path(__file__).parent  # Gets directory where script is located
main_dir = script_dir.parent  # Goes up one level to main directory
output_dir = main_dir / "image_output"  
model_cache_dir = main_dir / "model_cache"

# Create output directory if it doesn't exist
output_dir.mkdir(exist_ok=True)
model_cache_dir.mkdir(exist_ok=True)
# Use stable diffusion v1.5
model_id = "runwayml/stable-diffusion-v1-5"  
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load pipeline (no "fp16" on CPU)
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    cache_dir=model_cache_dir
)
pipe = pipe.to(device)

# Enable attention slicing if on CPU/low memory
if device == "cpu":
    pipe.enable_attention_slicing()

# Generate image (fast, lower quality but great for testing)
prompt = "a cute cartoon fox under a sakura tree, pastel colors"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]  # Fewer steps = fast

# Save to output folder
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = output_dir / f"fox_{timestamp}.png"
image.save(output_path)

print(f"Done! Image saved to: {output_path}")