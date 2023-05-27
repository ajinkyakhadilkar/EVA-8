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

Download the trained checkpoint [here](https://drive.google.com/file/d/10lH_Yl0OLqEu1-98LzX1_2QyjuZvSk7o/view?usp=sharing).

### Run the script
```
python gradio_canny2image.py
```

## Training

### Dataset
This model was trained on randomly picked 200 classes from [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k) validation dataset. 

This model is trained on these [200 classes](https://github.com/ajinkyakhadilkar/EVA-8/files/11581466/classes.txt).

The images are resized to 512x512 for the training script.
```
python resize_images.py
```

Generate Canny edge maps for the images.
```
python edge_detection.py
```
Here is how the images are transformed after running the script.


![ILSVRC2012_val_00000075_n01795545](https://github.com/ajinkyakhadilkar/EVA-8/assets/27129645/73077755-1c80-4087-92bf-a35501b5f595)
![ILSVRC2012_val_00000075_n01795545](https://github.com/ajinkyakhadilkar/EVA-8/assets/27129645/6ce5db42-0007-4846-839e-2cfa367fe476)


The source and target images are ready. Next step is to generate prompts for these images. I used [BLIP](https://github.com/salesforce/BLIP) generated captions as prompts.
```
python blip_captions.py
```
This will generate BLIP captions in JSON format. We will need to re-formate the JSON to get the prompts in a desired format for the trainer.
```
python transform_prompts.py
```

Here is a sample from the file generated from above scripts.
```json
{"source": "ILSVRC2012_val_00000075_n01795545.JPEG", "target": "ILSVRC2012_val_00000075_n01795545.JPEG", "prompt": "there is a black bird with a red head and white wings"}
{"source": "ILSVRC2012_val_00000077_n02087394.JPEG", "target": "ILSVRC2012_val_00000077_n02087394.JPEG", "prompt": "there is a dog that is standing on a rock in the woods"}
{"source": "ILSVRC2012_val_00000079_n02091635.JPEG", "target": "ILSVRC2012_val_00000079_n02091635.JPEG", "prompt": "there is a dog that is sitting on a chair with its tongue out"}

```

The dataset is now ready for training. We will pass the source images, target images and image prompts to the trainer.
