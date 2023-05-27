# ControlNet with Canny Edges

ControlNet is a neural network structure to control diffusion models by adding extra conditions. Official repository: [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)

This model of ControlNet is trained for Canny Edges. It takes Canny Edges and a prompt as an input and generates images from them.

## Usage

### Requirements
Install (and uninstall) the dependencies using pip.

```
!pip install gradio==3.16.2 albumentations==1.3.0 opencv-contrib-python imageio==2.9.0 imageio-ffmpeg==0.4.2 pytorch-lightning==1.5.0 omegaconf==2.1.1 test-tube>=0.7.5 streamlit==1.12.1 einops==0.3.0 transformers==4.19.2 webdataset==0.2.5 kornia==0.6 open_clip_torch==2.0.2 invisible-watermark>=0.1.5 streamlit-drawable-canvas==0.8.0 torchmetrics==0.6.0 timm==0.6.12 addict==2.4.0 yapf==0.32.0 prettytable==3.6.0 safetensors==0.2.7 basicsr==1.4.2
!pip install torchtext==0.6.0
!pip uninstall -y keras
```

