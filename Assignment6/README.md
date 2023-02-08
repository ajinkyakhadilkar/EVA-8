## CIFAR10 with Albumebtations

![download](https://user-images.githubusercontent.com/27129645/217647205-ed8ce160-be9f-4002-9499-678d219f1c2e.png)


#### Model

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1            [-1, 6, 32, 32]             456
                  ReLU-2            [-1, 6, 32, 32]               0
                Conv2d-3           [-1, 16, 34, 34]             880
                  ReLU-4           [-1, 16, 34, 34]               0
                Conv2d-5           [-1, 16, 16, 16]           2,320
                  ReLU-6           [-1, 16, 16, 16]               0
                Conv2d-7           [-1, 16, 16, 16]           2,320
                Conv2d-8            [-1, 8, 16, 16]             136
                  ReLU-9            [-1, 8, 16, 16]               0
               Conv2d-10              [-1, 8, 7, 7]             584
                 ReLU-11              [-1, 8, 7, 7]               0
               Conv2d-12             [-1, 16, 7, 7]           3,216
                 ReLU-13             [-1, 16, 7, 7]               0
               Conv2d-14             [-1, 16, 3, 3]           2,320
                 ReLU-15             [-1, 16, 3, 3]               0
    AdaptiveAvgPool2d-16             [-1, 16, 1, 1]               0
               Linear-17                   [-1, 10]             170
    ================================================================
    Total params: 12,402
    Trainable params: 12,402
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 0.52
    Params size (MB): 0.05
    Estimated Total Size (MB): 0.58
    ----------------------------------------------------------------

#### Transformations using Albumentation
[x] Horizontal Flip - 50%
[x] Shift Scale Rotate - 50%
[x] Coarse Dropout
[x] Normalize

    A.Compose([A.HorizontalFlip(p=0.5),
                              A.ShiftScaleRotate(p=0.5),
                              A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_height=16, min_width=16, min_holes = 1, fill_value=-1.69, mask_fill_value = None),
                              A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                              A.pytorch.transforms.ToTensorV2(),]))
