
# Insight on CIFAR-10 images on Resnet-18 network

This assignment with train the Resnet-18 model on CIFAR-10 dataset. We will see a sample of misclassified images and plot their GradCams to give a general idea about what the network was looking at when classifying those images.

### Summary of Resnet 18

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 64, 32, 32]           1,728
           BatchNorm2d-2           [-1, 64, 32, 32]             128
                Conv2d-3           [-1, 64, 32, 32]          36,864
           BatchNorm2d-4           [-1, 64, 32, 32]             128
                Conv2d-5           [-1, 64, 32, 32]          36,864
           BatchNorm2d-6           [-1, 64, 32, 32]             128
            BasicBlock-7           [-1, 64, 32, 32]               0
                Conv2d-8           [-1, 64, 32, 32]          36,864
           BatchNorm2d-9           [-1, 64, 32, 32]             128
               Conv2d-10           [-1, 64, 32, 32]          36,864
          BatchNorm2d-11           [-1, 64, 32, 32]             128
           BasicBlock-12           [-1, 64, 32, 32]               0
               Conv2d-13          [-1, 128, 16, 16]          73,728
          BatchNorm2d-14          [-1, 128, 16, 16]             256
               Conv2d-15          [-1, 128, 16, 16]         147,456
          BatchNorm2d-16          [-1, 128, 16, 16]             256
               Conv2d-17          [-1, 128, 16, 16]           8,192
          BatchNorm2d-18          [-1, 128, 16, 16]             256
           BasicBlock-19          [-1, 128, 16, 16]               0
               Conv2d-20          [-1, 128, 16, 16]         147,456
          BatchNorm2d-21          [-1, 128, 16, 16]             256
               Conv2d-22          [-1, 128, 16, 16]         147,456
          BatchNorm2d-23          [-1, 128, 16, 16]             256
           BasicBlock-24          [-1, 128, 16, 16]               0
               Conv2d-25            [-1, 256, 8, 8]         294,912
          BatchNorm2d-26            [-1, 256, 8, 8]             512
               Conv2d-27            [-1, 256, 8, 8]         589,824
          BatchNorm2d-28            [-1, 256, 8, 8]             512
               Conv2d-29            [-1, 256, 8, 8]          32,768
          BatchNorm2d-30            [-1, 256, 8, 8]             512
           BasicBlock-31            [-1, 256, 8, 8]               0
               Conv2d-32            [-1, 256, 8, 8]         589,824
          BatchNorm2d-33            [-1, 256, 8, 8]             512
               Conv2d-34            [-1, 256, 8, 8]         589,824
          BatchNorm2d-35            [-1, 256, 8, 8]             512
           BasicBlock-36            [-1, 256, 8, 8]               0
               Conv2d-37            [-1, 512, 4, 4]       1,179,648
          BatchNorm2d-38            [-1, 512, 4, 4]           1,024
               Conv2d-39            [-1, 512, 4, 4]       2,359,296
          BatchNorm2d-40            [-1, 512, 4, 4]           1,024
               Conv2d-41            [-1, 512, 4, 4]         131,072
          BatchNorm2d-42            [-1, 512, 4, 4]           1,024
           BasicBlock-43            [-1, 512, 4, 4]               0
               Conv2d-44            [-1, 512, 4, 4]       2,359,296
          BatchNorm2d-45            [-1, 512, 4, 4]           1,024
               Conv2d-46            [-1, 512, 4, 4]       2,359,296
          BatchNorm2d-47            [-1, 512, 4, 4]           1,024
           BasicBlock-48            [-1, 512, 4, 4]               0
               Linear-49                   [-1, 10]           5,130
               ResNet-50                   [-1, 10]               0
    ================================================================
    Total params: 11,173,962
    Trainable params: 11,173,962
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 11.25
    Params size (MB): 42.63
    Estimated Total Size (MB): 53.89
    ----------------------------------------------------------------
    
    
### Training the network

I trained the network for 20 epochs.

    Epoch=0 Loss=1.6135490242477573 batch_id=390 Accuracy=40.21%: 100%|██████████| 391/391 [00:43<00:00,  8.98it/s]

    Test set: Average loss: 1.3122, Accuracy: 5260/10000 (52.60%)

    Epoch=1 Loss=1.201545045503875 batch_id=390 Accuracy=56.54%: 100%|██████████| 391/391 [00:41<00:00,  9.48it/s]

    Test set: Average loss: 1.0917, Accuracy: 6078/10000 (60.78%)

    Epoch=2 Loss=0.9955618401317645 batch_id=390 Accuracy=64.25%: 100%|██████████| 391/391 [00:40<00:00,  9.65it/s]

    Test set: Average loss: 1.0139, Accuracy: 6429/10000 (64.29%)

    Epoch=3 Loss=0.8568977029122355 batch_id=390 Accuracy=69.49%: 100%|██████████| 391/391 [00:40<00:00,  9.60it/s]

    Test set: Average loss: 0.8394, Accuracy: 7044/10000 (70.44%)

    Epoch=4 Loss=0.7470100218682643 batch_id=390 Accuracy=73.76%: 100%|██████████| 391/391 [00:40<00:00,  9.62it/s]

    Test set: Average loss: 0.7422, Accuracy: 7488/10000 (74.88%)

    Epoch=5 Loss=0.6646932834554511 batch_id=390 Accuracy=76.76%: 100%|██████████| 391/391 [00:40<00:00,  9.62it/s]

    Test set: Average loss: 0.6796, Accuracy: 7637/10000 (76.37%)

    Epoch=6 Loss=0.601199871210186 batch_id=390 Accuracy=79.03%: 100%|██████████| 391/391 [00:40<00:00,  9.59it/s]

    Test set: Average loss: 0.6757, Accuracy: 7694/10000 (76.94%)

    Epoch=7 Loss=0.5545394865753096 batch_id=390 Accuracy=80.69%: 100%|██████████| 391/391 [00:40<00:00,  9.60it/s]

    Test set: Average loss: 0.6295, Accuracy: 7854/10000 (78.54%)

    Epoch=8 Loss=0.5084875420383785 batch_id=390 Accuracy=82.22%: 100%|██████████| 391/391 [00:41<00:00,  9.48it/s]

    Test set: Average loss: 0.5935, Accuracy: 7962/10000 (79.62%)

    Epoch=9 Loss=0.4770503038030756 batch_id=390 Accuracy=83.66%: 100%|██████████| 391/391 [00:40<00:00,  9.58it/s]

    Test set: Average loss: 0.5512, Accuracy: 8149/10000 (81.49%)

    Epoch=10 Loss=0.4463933108712706 batch_id=390 Accuracy=84.60%: 100%|██████████| 391/391 [00:40<00:00,  9.56it/s]

    Test set: Average loss: 0.5281, Accuracy: 8234/10000 (82.34%)

    Epoch=11 Loss=0.41926693923942876 batch_id=390 Accuracy=85.60%: 100%|██████████| 391/391 [00:40<00:00,  9.62it/s]

    Test set: Average loss: 0.5218, Accuracy: 8293/10000 (82.93%)

    Epoch=12 Loss=0.391926050757813 batch_id=390 Accuracy=86.50%: 100%|██████████| 391/391 [00:40<00:00,  9.68it/s]

    Test set: Average loss: 0.5019, Accuracy: 8325/10000 (83.25%)

    Epoch=13 Loss=0.37174972785098476 batch_id=390 Accuracy=87.03%: 100%|██████████| 391/391 [00:40<00:00,  9.71it/s]

    Test set: Average loss: 0.5060, Accuracy: 8345/10000 (83.45%)

    Epoch=14 Loss=0.3528844207296591 batch_id=390 Accuracy=87.78%: 100%|██████████| 391/391 [00:40<00:00,  9.57it/s]

    Test set: Average loss: 0.4563, Accuracy: 8514/10000 (85.14%)

    Epoch=15 Loss=0.3319216242912785 batch_id=390 Accuracy=88.41%: 100%|██████████| 391/391 [00:40<00:00,  9.68it/s]

    Test set: Average loss: 0.4588, Accuracy: 8496/10000 (84.96%)

    Epoch=16 Loss=0.31570308504964384 batch_id=390 Accuracy=88.98%: 100%|██████████| 391/391 [00:40<00:00,  9.60it/s]

    Test set: Average loss: 0.4405, Accuracy: 8554/10000 (85.54%)

    Epoch=17 Loss=0.30366044520112256 batch_id=390 Accuracy=89.48%: 100%|██████████| 391/391 [00:40<00:00,  9.67it/s]

    Test set: Average loss: 0.4314, Accuracy: 8575/10000 (85.75%)

    Epoch=18 Loss=0.286467830352771 batch_id=390 Accuracy=90.04%: 100%|██████████| 391/391 [00:40<00:00,  9.66it/s]

    Test set: Average loss: 0.4595, Accuracy: 8530/10000 (85.30%)

    Epoch=19 Loss=0.27008855529605885 batch_id=390 Accuracy=90.60%: 100%|██████████| 391/391 [00:40<00:00,  9.58it/s]

    Test set: Average loss: 0.4355, Accuracy: 8624/10000 (86.24%)
    
    
### Misclassified Images

The misclassified images can be found in [Misclassified.md](https://github.com/ajinkyakhadilkar/EVA-8/blob/main/Assignment7/Misclassified.md).

*Note: Normalized images are shown in the plots*

### GradCam of Misclassified Images

The GradCam plots of the misclassified images can be found in [GradCam.md](https://github.com/ajinkyakhadilkar/EVA-8/blob/main/Assignment7/GradCam.md).

*ToDo: The GradCam plots are messed up, again possibily due to poor handling of normalization*
