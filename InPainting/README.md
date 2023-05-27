# InPainting implementation for Stable Diffusion

This repository showcases an advanced implementation of an in-painting mechanism designed specifically for the stable diffusion model. The implementation takes three inputs: an image, a mask, and a prompt. The mask in the image is then effectively replaced with a generation that is intelligently generated based on the provided prompt. Official repository: [Stability-AI/stablediffusion](https://github.com/Stability-AI/stablediffusion)

## Usage

### Requirements

Install (and uninstall) the dependencies using pip.

```
#Dependencies from latent diffusion
pip install cudatoolkit==11.0 numpy==1.19.2
pip install pudb==2019.2 torch-fidelity==0.3.0 -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers -e git+https://github.com/openai/CLIP.git@main#egg=clip -e .

#Dependencies for stable diffusion (addded on to previous dependenies)
pip install --upgrade albumentations==1.3.0 opencv-python==4.6.0.66 imageio==2.9.0 imageio-ffmpeg==0.4.2 pytorch-lightning==1.4.2 omegaconf==2.1.1 test-tube>=0.7.5 streamlit==1.12.1 einops==0.3.0 transformers==4.19.2 webdataset==0.2.5 kornia==0.6 open_clip_torch==2.0.2 invisible-watermark>=0.1.5 streamlit-drawable-canvas==0.8.0 torchmetrics==0.6.0 diffusers

#Add or remove dependencies due to version conflicts
pip uninstall -y torchtext
pip install torch==1.13.1 torchaudio==0.13.1 torchvision==0.14.1+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install pytorch-lightning==1.1.8 pytorch-lightning-bolts==0.3.2.post1
pip install gradio
```

Download Stable Diffusion 2 InPainting checkpoint [512-inpainting-ema.ckpt](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/tree/main)


### Run the script

```
python scripts/inpainting.py --ckpt 512-inpainting-ema.ckpt --config configs/stable-diffusion/v2-inpainting-inference.yaml --device cuda
```

