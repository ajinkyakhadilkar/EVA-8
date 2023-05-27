# ControlNet with Canny Edges

ControlNet is a neural network structure to control diffusion models by adding extra conditions. Official repository: [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)

This model of ControlNet is trained for Canny Edges. It takes Canny Edges and a prompt as an input and generates images from them. Underlying pre-trained model for control is StableDiffusion 2.1.

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

## Training the ControlNet

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

The data is now ready for training. We will pass the source images, target images and image prompts to the trainer.


### Training 

The training steps, including data creation mentioned above is explained in the official repository [Train a ControlNet to Control SD
](https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md)

Once the checkpoint is created and the script is ready, run the script.
```
python tutorial_train_sd21.py
```

#### Training Logs
Sample training logs for 1 epoch.

<details>
  <summary>Training Logs</summary>
  
    logging improved.
    2023-05-22 10:42:41.413255: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-05-22 10:42:41.468687: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-05-22 10:42:42.553027: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    No module 'xformers'. Proceeding without it.
    ControlLDM: Running in eps-prediction mode
    DiffusionWrapper has 865.91 M params.
    making attention of type 'vanilla' with 512 in_channels
    Working with z of shape (1, 4, 32, 32) = 4096 dimensions.
    making attention of type 'vanilla' with 512 in_channels
    Loaded model config from [./models/cldm_v21.yaml]
    Loaded state_dict from [./models/checkpoint-epoch=01.ckpt]
    GPU available: True, used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    /usr/local/lib/python3.10/dist-packages/pytorch_lightning/trainer/configuration_validator.py:118: UserWarning: You defined a `validation_step` but have no `val_dataloader`. Skipping val loop.
      rank_zero_warn("You defined a `validation_step` but have no `val_dataloader`. Skipping val loop.")
    /usr/local/lib/python3.10/dist-packages/pytorch_lightning/trainer/configuration_validator.py:280: LightningDeprecationWarning: Base `LightningModule.on_train_batch_start` hook signature has changed in v1.5. The `dataloader_idx` argument will be removed in v1.7.
      rank_zero_deprecation(
    /usr/local/lib/python3.10/dist-packages/pytorch_lightning/trainer/configuration_validator.py:287: LightningDeprecationWarning: Base `Callback.on_train_batch_end` hook signature has changed in v1.5. The `dataloader_idx` argument will be removed in v1.7.
      rank_zero_deprecation(
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

      | Name              | Type                   | Params
    -------------------------------------------------------------
    0 | model             | DiffusionWrapper       | 865 M 
    1 | first_stage_model | AutoencoderKL          | 83.7 M
    2 | cond_stage_model  | FrozenOpenCLIPEmbedder | 354 M 
    3 | control_model     | ControlNet             | 364 M 
    -------------------------------------------------------------
    1.2 B     Trainable params
    437 M     Non-trainable params
    1.7 B     Total params
    6,671.302 Total estimated model params size (MB)
    /usr/local/lib/python3.10/dist-packages/pytorch_lightning/callbacks/model_checkpoint.py:617: UserWarning: Checkpoint directory /content/drive/MyDrive/EVA_Capstone/ControlNet/models exists and is not empty.
      rank_zero_warn(f"Checkpoint directory {dirpath} exists and is not empty.")
    /usr/local/lib/python3.10/dist-packages/pytorch_lightning/trainer/data_loading.py:110: UserWarning: The dataloader, train_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 12 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      rank_zero_warn(
    Epoch 0:   0% 0/1000 [00:00<?, ?it/s] /usr/local/lib/python3.10/dist-packages/pytorch_lightning/utilities/data.py:56: UserWarning: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 10. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
      warning_cache.warn(
    Data shape for DDIM sampling is (4, 4, 64, 64), eta 0.0
    Running DDIM Sampling with 50 timesteps

    DDIM Sampler:   0% 0/50 [00:00<?, ?it/s]
    DDIM Sampler:   2% 1/50 [00:00<00:23,  2.12it/s]
    DDIM Sampler:   4% 2/50 [00:00<00:21,  2.19it/s]
    DDIM Sampler:   6% 3/50 [00:01<00:21,  2.21it/s]
    DDIM Sampler:   8% 4/50 [00:01<00:20,  2.22it/s]
    DDIM Sampler:  10% 5/50 [00:02<00:20,  2.23it/s]
    DDIM Sampler:  12% 6/50 [00:02<00:19,  2.23it/s]
    DDIM Sampler:  14% 7/50 [00:03<00:19,  2.24it/s]
    DDIM Sampler:  16% 8/50 [00:03<00:18,  2.24it/s]
    DDIM Sampler:  18% 9/50 [00:04<00:18,  2.24it/s]
    DDIM Sampler:  20% 10/50 [00:04<00:17,  2.24it/s]
    DDIM Sampler:  22% 11/50 [00:04<00:17,  2.24it/s]
    DDIM Sampler:  24% 12/50 [00:05<00:16,  2.24it/s]
    DDIM Sampler:  26% 13/50 [00:05<00:16,  2.24it/s]
    DDIM Sampler:  28% 14/50 [00:06<00:16,  2.24it/s]
    DDIM Sampler:  30% 15/50 [00:06<00:15,  2.24it/s]
    DDIM Sampler:  32% 16/50 [00:07<00:15,  2.24it/s]
    DDIM Sampler:  34% 17/50 [00:07<00:14,  2.24it/s]
    DDIM Sampler:  36% 18/50 [00:08<00:14,  2.24it/s]
    DDIM Sampler:  38% 19/50 [00:08<00:13,  2.24it/s]
    DDIM Sampler:  40% 20/50 [00:08<00:13,  2.24it/s]
    DDIM Sampler:  42% 21/50 [00:09<00:12,  2.24it/s]
    DDIM Sampler:  44% 22/50 [00:09<00:12,  2.24it/s]
    DDIM Sampler:  46% 23/50 [00:10<00:12,  2.24it/s]
    DDIM Sampler:  48% 24/50 [00:10<00:11,  2.24it/s]
    DDIM Sampler:  50% 25/50 [00:11<00:11,  2.24it/s]
    DDIM Sampler:  52% 26/50 [00:11<00:10,  2.24it/s]
    DDIM Sampler:  54% 27/50 [00:12<00:10,  2.24it/s]
    DDIM Sampler:  56% 28/50 [00:12<00:09,  2.24it/s]
    DDIM Sampler:  58% 29/50 [00:12<00:09,  2.24it/s]
    DDIM Sampler:  60% 30/50 [00:13<00:08,  2.24it/s]
    DDIM Sampler:  62% 31/50 [00:13<00:08,  2.24it/s]
    DDIM Sampler:  64% 32/50 [00:14<00:08,  2.24it/s]
    DDIM Sampler:  66% 33/50 [00:14<00:07,  2.24it/s]
    DDIM Sampler:  68% 34/50 [00:15<00:07,  2.24it/s]
    DDIM Sampler:  70% 35/50 [00:15<00:06,  2.24it/s]
    DDIM Sampler:  72% 36/50 [00:16<00:06,  2.24it/s]
    DDIM Sampler:  74% 37/50 [00:16<00:05,  2.24it/s]
    DDIM Sampler:  76% 38/50 [00:16<00:05,  2.24it/s]
    DDIM Sampler:  78% 39/50 [00:17<00:04,  2.24it/s]
    DDIM Sampler:  80% 40/50 [00:17<00:04,  2.24it/s]
    DDIM Sampler:  82% 41/50 [00:18<00:04,  2.24it/s]
    DDIM Sampler:  84% 42/50 [00:18<00:03,  2.24it/s]
    DDIM Sampler:  86% 43/50 [00:19<00:03,  2.24it/s]
    DDIM Sampler:  88% 44/50 [00:19<00:02,  2.24it/s]
    DDIM Sampler:  90% 45/50 [00:20<00:02,  2.24it/s]
    DDIM Sampler:  92% 46/50 [00:20<00:01,  2.24it/s]
    DDIM Sampler:  94% 47/50 [00:21<00:01,  2.24it/s]
    DDIM Sampler:  96% 48/50 [00:21<00:00,  2.24it/s]
    DDIM Sampler:  98% 49/50 [00:21<00:00,  2.24it/s]
    DDIM Sampler: 100% 50/50 [00:22<00:00,  2.24it/s]
    Epoch 0:  30% 300/1000 [1:25:47<3:20:11, 17.16s/it, loss=0.148, v_num=16, train/loss_simple_step=0.103, train/loss_vlb_step=0.000478, train/loss_step=0.103, global_step=299.0] Data shape for DDIM sampling is (4, 4, 64, 64), eta 0.0
    Running DDIM Sampling with 50 timesteps

    DDIM Sampler:   0% 0/50 [00:00<?, ?it/s]
    DDIM Sampler:   2% 1/50 [00:00<00:21,  2.24it/s]
    DDIM Sampler:   4% 2/50 [00:00<00:21,  2.24it/s]
    DDIM Sampler:   6% 3/50 [00:01<00:20,  2.24it/s]
    DDIM Sampler:   8% 4/50 [00:01<00:20,  2.24it/s]
    DDIM Sampler:  10% 5/50 [00:02<00:20,  2.24it/s]
    DDIM Sampler:  12% 6/50 [00:02<00:19,  2.24it/s]
    DDIM Sampler:  14% 7/50 [00:03<00:19,  2.24it/s]
    DDIM Sampler:  16% 8/50 [00:03<00:18,  2.24it/s]
    DDIM Sampler:  18% 9/50 [00:04<00:18,  2.24it/s]
    DDIM Sampler:  20% 10/50 [00:04<00:17,  2.24it/s]
    DDIM Sampler:  22% 11/50 [00:04<00:17,  2.24it/s]
    DDIM Sampler:  24% 12/50 [00:05<00:16,  2.24it/s]
    DDIM Sampler:  26% 13/50 [00:05<00:16,  2.24it/s]
    DDIM Sampler:  28% 14/50 [00:06<00:16,  2.24it/s]
    DDIM Sampler:  30% 15/50 [00:06<00:15,  2.24it/s]
    DDIM Sampler:  32% 16/50 [00:07<00:15,  2.24it/s]
    DDIM Sampler:  34% 17/50 [00:07<00:14,  2.24it/s]
    DDIM Sampler:  36% 18/50 [00:08<00:14,  2.24it/s]
    DDIM Sampler:  38% 19/50 [00:08<00:13,  2.24it/s]
    DDIM Sampler:  40% 20/50 [00:08<00:13,  2.24it/s]
    DDIM Sampler:  42% 21/50 [00:09<00:12,  2.24it/s]
    DDIM Sampler:  44% 22/50 [00:09<00:12,  2.24it/s]
    DDIM Sampler:  46% 23/50 [00:10<00:12,  2.24it/s]
    DDIM Sampler:  48% 24/50 [00:10<00:11,  2.24it/s]
    DDIM Sampler:  50% 25/50 [00:11<00:11,  2.24it/s]
    DDIM Sampler:  52% 26/50 [00:11<00:10,  2.24it/s]
    DDIM Sampler:  54% 27/50 [00:12<00:10,  2.24it/s]
    DDIM Sampler:  56% 28/50 [00:12<00:09,  2.24it/s]
    DDIM Sampler:  58% 29/50 [00:12<00:09,  2.24it/s]
    DDIM Sampler:  60% 30/50 [00:13<00:08,  2.24it/s]
    DDIM Sampler:  62% 31/50 [00:13<00:08,  2.24it/s]
    DDIM Sampler:  64% 32/50 [00:14<00:08,  2.24it/s]
    DDIM Sampler:  66% 33/50 [00:14<00:07,  2.24it/s]
    DDIM Sampler:  68% 34/50 [00:15<00:07,  2.24it/s]
    DDIM Sampler:  70% 35/50 [00:15<00:06,  2.24it/s]
    DDIM Sampler:  72% 36/50 [00:16<00:06,  2.24it/s]
    DDIM Sampler:  74% 37/50 [00:16<00:05,  2.24it/s]
    DDIM Sampler:  76% 38/50 [00:16<00:05,  2.24it/s]
    DDIM Sampler:  78% 39/50 [00:17<00:04,  2.24it/s]
    DDIM Sampler:  80% 40/50 [00:17<00:04,  2.24it/s]
    DDIM Sampler:  82% 41/50 [00:18<00:04,  2.24it/s]
    DDIM Sampler:  84% 42/50 [00:18<00:03,  2.24it/s]
    DDIM Sampler:  86% 43/50 [00:19<00:03,  2.24it/s]
    DDIM Sampler:  88% 44/50 [00:19<00:02,  2.24it/s]
    DDIM Sampler:  90% 45/50 [00:20<00:02,  2.24it/s]
    DDIM Sampler:  92% 46/50 [00:20<00:01,  2.24it/s]
    DDIM Sampler:  94% 47/50 [00:20<00:01,  2.24it/s]
    DDIM Sampler:  96% 48/50 [00:21<00:00,  2.24it/s]
    DDIM Sampler:  98% 49/50 [00:21<00:00,  2.24it/s]
    DDIM Sampler: 100% 50/50 [00:22<00:00,  2.24it/s]
    Epoch 0:  60% 600/1000 [2:49:20<1:52:53, 16.93s/it, loss=0.16, v_num=16, train/loss_simple_step=0.312, train/loss_vlb_step=0.00454, train/loss_step=0.312, global_step=599.0]  Data shape for DDIM sampling is (4, 4, 64, 64), eta 0.0
    Running DDIM Sampling with 50 timesteps

    DDIM Sampler:   0% 0/50 [00:00<?, ?it/s]
    DDIM Sampler:   2% 1/50 [00:00<00:21,  2.24it/s]
    DDIM Sampler:   4% 2/50 [00:00<00:21,  2.24it/s]
    DDIM Sampler:   6% 3/50 [00:01<00:20,  2.24it/s]
    DDIM Sampler:   8% 4/50 [00:01<00:20,  2.24it/s]
    DDIM Sampler:  10% 5/50 [00:02<00:20,  2.24it/s]
    DDIM Sampler:  12% 6/50 [00:02<00:19,  2.24it/s]
    DDIM Sampler:  14% 7/50 [00:03<00:19,  2.24it/s]
    DDIM Sampler:  16% 8/50 [00:03<00:18,  2.24it/s]
    DDIM Sampler:  18% 9/50 [00:04<00:18,  2.24it/s]
    DDIM Sampler:  20% 10/50 [00:04<00:17,  2.24it/s]
    DDIM Sampler:  22% 11/50 [00:04<00:17,  2.24it/s]
    DDIM Sampler:  24% 12/50 [00:05<00:16,  2.24it/s]
    DDIM Sampler:  26% 13/50 [00:05<00:16,  2.24it/s]
    DDIM Sampler:  28% 14/50 [00:06<00:16,  2.24it/s]
    DDIM Sampler:  30% 15/50 [00:06<00:15,  2.24it/s]
    DDIM Sampler:  32% 16/50 [00:07<00:15,  2.24it/s]
    DDIM Sampler:  34% 17/50 [00:07<00:14,  2.24it/s]
    DDIM Sampler:  36% 18/50 [00:08<00:14,  2.24it/s]
    DDIM Sampler:  38% 19/50 [00:08<00:13,  2.24it/s]
    DDIM Sampler:  40% 20/50 [00:08<00:13,  2.24it/s]
    DDIM Sampler:  42% 21/50 [00:09<00:12,  2.24it/s]
    DDIM Sampler:  44% 22/50 [00:09<00:12,  2.24it/s]
    DDIM Sampler:  46% 23/50 [00:10<00:12,  2.24it/s]
    DDIM Sampler:  48% 24/50 [00:10<00:11,  2.24it/s]
    DDIM Sampler:  50% 25/50 [00:11<00:11,  2.24it/s]
    DDIM Sampler:  52% 26/50 [00:11<00:10,  2.24it/s]
    DDIM Sampler:  54% 27/50 [00:12<00:10,  2.24it/s]
    DDIM Sampler:  56% 28/50 [00:12<00:09,  2.24it/s]
    DDIM Sampler:  58% 29/50 [00:12<00:09,  2.24it/s]
    DDIM Sampler:  60% 30/50 [00:13<00:08,  2.24it/s]
    DDIM Sampler:  62% 31/50 [00:13<00:08,  2.24it/s]
    DDIM Sampler:  64% 32/50 [00:14<00:08,  2.24it/s]
    DDIM Sampler:  66% 33/50 [00:14<00:07,  2.24it/s]
    DDIM Sampler:  68% 34/50 [00:15<00:07,  2.24it/s]
    DDIM Sampler:  70% 35/50 [00:15<00:06,  2.24it/s]
    DDIM Sampler:  72% 36/50 [00:16<00:06,  2.24it/s]
    DDIM Sampler:  74% 37/50 [00:16<00:05,  2.24it/s]
    DDIM Sampler:  76% 38/50 [00:16<00:05,  2.24it/s]
    DDIM Sampler:  78% 39/50 [00:17<00:04,  2.24it/s]
    DDIM Sampler:  80% 40/50 [00:17<00:04,  2.24it/s]
    DDIM Sampler:  82% 41/50 [00:18<00:04,  2.24it/s]
    DDIM Sampler:  84% 42/50 [00:18<00:03,  2.24it/s]
    DDIM Sampler:  86% 43/50 [00:19<00:03,  2.24it/s]
    DDIM Sampler:  88% 44/50 [00:19<00:02,  2.24it/s]
    DDIM Sampler:  90% 45/50 [00:20<00:02,  2.24it/s]
    DDIM Sampler:  92% 46/50 [00:20<00:01,  2.24it/s]
    DDIM Sampler:  94% 47/50 [00:20<00:01,  2.24it/s]
    DDIM Sampler:  96% 48/50 [00:21<00:00,  2.24it/s]
    DDIM Sampler:  98% 49/50 [00:21<00:00,  2.24it/s]
    DDIM Sampler: 100% 50/50 [00:22<00:00,  2.24it/s]
    Epoch 0:  90% 900/1000 [4:13:24<28:09, 16.89s/it, loss=0.16, v_num=16, train/loss_simple_step=0.176, train/loss_vlb_step=0.00286, train/loss_step=0.176, global_step=899.0] Data shape for DDIM sampling is (4, 4, 64, 64), eta 0.0
    Running DDIM Sampling with 50 timesteps

    DDIM Sampler:   0% 0/50 [00:00<?, ?it/s]
    DDIM Sampler:   2% 1/50 [00:00<00:21,  2.24it/s]
    DDIM Sampler:   4% 2/50 [00:00<00:21,  2.24it/s]
    DDIM Sampler:   6% 3/50 [00:01<00:20,  2.24it/s]
    DDIM Sampler:   8% 4/50 [00:01<00:20,  2.24it/s]
    DDIM Sampler:  10% 5/50 [00:02<00:20,  2.24it/s]
    DDIM Sampler:  12% 6/50 [00:02<00:19,  2.24it/s]
    DDIM Sampler:  14% 7/50 [00:03<00:19,  2.24it/s]
    DDIM Sampler:  16% 8/50 [00:03<00:18,  2.24it/s]
    DDIM Sampler:  18% 9/50 [00:04<00:18,  2.24it/s]
    DDIM Sampler:  20% 10/50 [00:04<00:17,  2.24it/s]
    DDIM Sampler:  22% 11/50 [00:04<00:17,  2.24it/s]
    DDIM Sampler:  24% 12/50 [00:05<00:16,  2.24it/s]
    DDIM Sampler:  26% 13/50 [00:05<00:16,  2.24it/s]
    DDIM Sampler:  28% 14/50 [00:06<00:16,  2.24it/s]
    DDIM Sampler:  30% 15/50 [00:06<00:15,  2.24it/s]
    DDIM Sampler:  32% 16/50 [00:07<00:15,  2.24it/s]
    DDIM Sampler:  34% 17/50 [00:07<00:14,  2.24it/s]
    DDIM Sampler:  36% 18/50 [00:08<00:14,  2.24it/s]
    DDIM Sampler:  38% 19/50 [00:08<00:13,  2.24it/s]
    DDIM Sampler:  40% 20/50 [00:08<00:13,  2.24it/s]
    DDIM Sampler:  42% 21/50 [00:09<00:12,  2.24it/s]
    DDIM Sampler:  44% 22/50 [00:09<00:12,  2.24it/s]
    DDIM Sampler:  46% 23/50 [00:10<00:12,  2.24it/s]
    DDIM Sampler:  48% 24/50 [00:10<00:11,  2.24it/s]
    DDIM Sampler:  50% 25/50 [00:11<00:11,  2.24it/s]
    DDIM Sampler:  52% 26/50 [00:11<00:10,  2.24it/s]
    DDIM Sampler:  54% 27/50 [00:12<00:10,  2.24it/s]
    DDIM Sampler:  56% 28/50 [00:12<00:09,  2.24it/s]
    DDIM Sampler:  58% 29/50 [00:12<00:09,  2.24it/s]
    DDIM Sampler:  60% 30/50 [00:13<00:08,  2.24it/s]
    DDIM Sampler:  62% 31/50 [00:13<00:08,  2.24it/s]
    DDIM Sampler:  64% 32/50 [00:14<00:08,  2.24it/s]
    DDIM Sampler:  66% 33/50 [00:14<00:07,  2.24it/s]
    DDIM Sampler:  68% 34/50 [00:15<00:07,  2.24it/s]
    DDIM Sampler:  70% 35/50 [00:15<00:06,  2.24it/s]
    DDIM Sampler:  72% 36/50 [00:16<00:06,  2.24it/s]
    DDIM Sampler:  74% 37/50 [00:16<00:05,  2.24it/s]
    DDIM Sampler:  76% 38/50 [00:16<00:05,  2.24it/s]
    DDIM Sampler:  78% 39/50 [00:17<00:04,  2.24it/s]
    DDIM Sampler:  80% 40/50 [00:17<00:04,  2.24it/s]
    DDIM Sampler:  82% 41/50 [00:18<00:04,  2.24it/s]
    DDIM Sampler:  84% 42/50 [00:18<00:03,  2.24it/s]
    DDIM Sampler:  86% 43/50 [00:19<00:03,  2.24it/s]
    DDIM Sampler:  88% 44/50 [00:19<00:02,  2.24it/s]
    DDIM Sampler:  90% 45/50 [00:20<00:02,  2.24it/s]
    DDIM Sampler:  92% 46/50 [00:20<00:01,  2.24it/s]
    DDIM Sampler:  94% 47/50 [00:20<00:01,  2.24it/s]
    DDIM Sampler:  96% 48/50 [00:21<00:00,  2.24it/s]
    DDIM Sampler:  98% 49/50 [00:21<00:00,  2.24it/s]
    DDIM Sampler: 100% 50/50 [00:22<00:00,  2.24it/s]

</details>

The checkpoint is after 5 epochs of training on NVidia A100 GPU.
