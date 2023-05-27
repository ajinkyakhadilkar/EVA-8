import argparse, os
import cv2
import gradio as gr
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from imwatermark import WatermarkEncoder

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

torch.set_grad_enabled(False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )
    opt = parser.parse_args()
    return opt


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img



def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    return sampler

def inpainting(masked_image, mask, prompt, seed, ddim_steps, scale, w=512, h=512, num_samples=1):
    
    config = OmegaConf.load(f"{opt.config}")
    device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(f"{opt.ckpt}")["state_dict"], strict=False)
    model = model.to(device)
    sampler = DDIMSampler(model)
  
 
    prng = np.random.RandomState(seed)
    rndm_latent = prng.randn(num_samples, 4, h // 8, w // 8)
    rndm_latent = torch.from_numpy(rndm_latent).to(
        device=device, dtype=torch.float32)
        
    masked_image = masked_image.to(device)
    mask = mask.to(device)
    
    with torch.no_grad():
        #encode prompt
        prompt_enc =  model.cond_stage_model.encode(prompt)
        #interpolate mask to latent vector size
        mask_latent =  torch.nn.functional.interpolate(mask.float(), size= (h // 8, w // 8))
        #encode masked_image
        masked_image_enc = model.get_first_stage_encoding(model.encode_first_stage(masked_image.float()))
        #interpolate masked image  to latent vector size
        #masked_image_enc = torch.nn.functional.interpolate(masked_image_enc, size=(64,64))
        
        print(masked_image.size())
        print(mask_latent.size())
        print(masked_image_enc.size())
        
        #concatenate masked image and mask vectors
        c_cat = torch.cat([mask_latent, masked_image_enc], dim=1)
        
        #cond
        cond = {"c_concat": [c_cat], "c_crossattn": [prompt_enc]}
        
        
        #uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples, "")
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
        
        shape = [model.channels, h // 8, w // 8]
        samples_cfg, intermediates = sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=1.0,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=rndm_latent,
        )
        x_samples_ddim = model.decode_first_stage(samples_cfg)

        result = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                             min=0.0, max=1.0)
        print('RESULT CALCULATED')
        #result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
        img = result.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255
    return [Image.fromarray(img.astype(np.uint8))]
        


def predict(input_image, prompt, ddim_steps, scale, seed):
    ''''
    print(input_image)
    print(prompt)
    mask = input_image['mask']
    mask = np.array(mask.convert("L"))
    mask = mask[None].transpose(1,2,0)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    ''' 
    image = input_image["image"].convert("RGB")
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    
    mask = input_image['mask'].convert("RGB")
    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)
    masked_image = image * (mask < 0.5)
    return inpainting(masked_image, mask, prompt, seed, ddim_steps, scale)


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Stable Diffusion Inpainting")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', tool='sketch', type="pil")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                ddim_steps = gr.Slider(label="Steps", minimum=1,
                                       maximum=50, value=45, step=1)
                scale = gr.Slider(
                    label="Guidance Scale", minimum=0.1, maximum=30.0, value=10, step=0.1
                )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=2147483647,
                    step=1,
                    randomize=True,
                )
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(
                grid=[2], height="auto")

    run_button.click(fn=predict, inputs=[
                     input_image, prompt, ddim_steps, scale, seed], outputs=[gallery])





if __name__ == "__main__":
    opt = parse_args()
    block.launch(share=True)
