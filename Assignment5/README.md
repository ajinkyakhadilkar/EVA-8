# Group Normalization, Layer Normalization, and Batch Normalization with L1 Reguralization

![graphs](https://user-images.githubusercontent.com/27129645/216755331-4b72f073-c96a-49f2-a474-b505e57595cb.jpeg)


## 1. Group Normalization


#### Summary

    ----------------------------------------------------------------
          Layer (type)               Output Shape         Param #
    ================================================================
              Conv2d-1           [-1, 10, 28, 28]             100
                ReLU-2           [-1, 10, 28, 28]               0
           GroupNorm-3           [-1, 10, 28, 28]              20
           Dropout2d-4           [-1, 10, 28, 28]               0
              Conv2d-5           [-1, 16, 28, 28]           1,456
                ReLU-6           [-1, 16, 28, 28]               0
           GroupNorm-7           [-1, 16, 28, 28]              32
           Dropout2d-8           [-1, 16, 28, 28]               0
           MaxPool2d-9           [-1, 16, 14, 14]               0
             Conv2d-10            [-1, 8, 14, 14]             136
               ReLU-11            [-1, 8, 14, 14]               0
          GroupNorm-12            [-1, 8, 14, 14]              16
          Dropout2d-13            [-1, 8, 14, 14]               0
             Conv2d-14           [-1, 16, 14, 14]           1,168
               ReLU-15           [-1, 16, 14, 14]               0
          GroupNorm-16           [-1, 16, 14, 14]              32
          Dropout2d-17           [-1, 16, 14, 14]               0
             Conv2d-18           [-1, 16, 14, 14]           2,320
               ReLU-19           [-1, 16, 14, 14]               0
          GroupNorm-20           [-1, 16, 14, 14]              32
          Dropout2d-21           [-1, 16, 14, 14]               0
             Conv2d-22           [-1, 16, 14, 14]           2,320
               ReLU-23           [-1, 16, 14, 14]               0
          GroupNorm-24           [-1, 16, 14, 14]              32
          Dropout2d-25           [-1, 16, 14, 14]               0
          MaxPool2d-26             [-1, 16, 7, 7]               0
             Conv2d-27              [-1, 8, 7, 7]             136
               ReLU-28              [-1, 8, 7, 7]               0
          GroupNorm-29              [-1, 8, 7, 7]              16
          Dropout2d-30              [-1, 8, 7, 7]               0
             Conv2d-31             [-1, 10, 5, 5]             730
               ReLU-32             [-1, 10, 5, 5]               0
          GroupNorm-33             [-1, 10, 5, 5]              20
          Dropout2d-34             [-1, 10, 5, 5]               0
             Conv2d-35             [-1, 16, 3, 3]           1,456
               ReLU-36             [-1, 16, 3, 3]               0
          GroupNorm-37             [-1, 16, 3, 3]              32
          Dropout2d-38             [-1, 16, 3, 3]               0
             Conv2d-39             [-1, 10, 3, 3]             170
          AvgPool2d-40             [-1, 10, 1, 1]               0
    ================================================================
    Total params: 10,224
    Trainable params: 10,224
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 1.01
    Params size (MB): 0.04
    Estimated Total Size (MB): 1.05
    ----------------------------------------------------------------


#### Training Logs

    Epoch=1 Loss=0.12392862141132355 batch_id=937 Accuracy=88.02%: 100%|██████████| 938/938 [00:25<00:00, 36.14it/s]

    Test set: Average loss: 0.0681, Accuracy: 9808/10000 (98.08%)

    Epoch=2 Loss=0.04138115793466568 batch_id=937 Accuracy=97.20%: 100%|██████████| 938/938 [00:22<00:00, 41.00it/s]

    Test set: Average loss: 0.0516, Accuracy: 9847/10000 (98.47%)

    Epoch=3 Loss=0.08628048002719879 batch_id=937 Accuracy=97.67%: 100%|██████████| 938/938 [00:23<00:00, 40.74it/s]

    Test set: Average loss: 0.0389, Accuracy: 9878/10000 (98.78%)

    Epoch=4 Loss=0.01633160375058651 batch_id=937 Accuracy=98.06%: 100%|██████████| 938/938 [00:24<00:00, 38.94it/s]

    Test set: Average loss: 0.0359, Accuracy: 9892/10000 (98.92%)

    Epoch=5 Loss=0.009081628173589706 batch_id=937 Accuracy=98.27%: 100%|██████████| 938/938 [00:23<00:00, 39.67it/s]

    Test set: Average loss: 0.0368, Accuracy: 9890/10000 (98.90%)

    Epoch=6 Loss=0.02881023846566677 batch_id=937 Accuracy=98.36%: 100%|██████████| 938/938 [00:23<00:00, 40.08it/s]

    Test set: Average loss: 0.0335, Accuracy: 9905/10000 (99.05%)

    Epoch=7 Loss=0.24833381175994873 batch_id=937 Accuracy=98.40%: 100%|██████████| 938/938 [00:23<00:00, 40.01it/s]

    Test set: Average loss: 0.0314, Accuracy: 9905/10000 (99.05%)

    Epoch=8 Loss=0.15637683868408203 batch_id=937 Accuracy=98.52%: 100%|██████████| 938/938 [00:23<00:00, 39.85it/s]

    Test set: Average loss: 0.0306, Accuracy: 9898/10000 (98.98%)

    Epoch=9 Loss=0.2758753299713135 batch_id=937 Accuracy=98.61%: 100%|██████████| 938/938 [00:23<00:00, 39.97it/s]

    Test set: Average loss: 0.0276, Accuracy: 9910/10000 (99.10%)

    Epoch=10 Loss=0.16807948052883148 batch_id=937 Accuracy=98.61%: 100%|██████████| 938/938 [00:23<00:00, 40.04it/s]

    Test set: Average loss: 0.0301, Accuracy: 9914/10000 (99.14%)

    Epoch=11 Loss=0.009748771786689758 batch_id=937 Accuracy=98.67%: 100%|██████████| 938/938 [00:23<00:00, 39.53it/s]

    Test set: Average loss: 0.0294, Accuracy: 9906/10000 (99.06%)

    Epoch=12 Loss=0.029194261878728867 batch_id=937 Accuracy=98.70%: 100%|██████████| 938/938 [00:23<00:00, 39.99it/s]

    Test set: Average loss: 0.0310, Accuracy: 9906/10000 (99.06%)

    Epoch=13 Loss=0.052009087055921555 batch_id=937 Accuracy=98.82%: 100%|██████████| 938/938 [00:23<00:00, 40.27it/s]

    Test set: Average loss: 0.0302, Accuracy: 9907/10000 (99.07%)

    Epoch=14 Loss=0.09740747511386871 batch_id=937 Accuracy=98.87%: 100%|██████████| 938/938 [00:23<00:00, 39.86it/s]

    Test set: Average loss: 0.0289, Accuracy: 9912/10000 (99.12%)

    Epoch=15 Loss=0.014592999592423439 batch_id=937 Accuracy=98.89%: 100%|██████████| 938/938 [00:23<00:00, 39.70it/s]

    Test set: Average loss: 0.0269, Accuracy: 9920/10000 (99.20%)

#### Misclassified Images

Images

![download](https://user-images.githubusercontent.com/27129645/216755382-90c6ab67-0abf-4d85-bab9-032185823090.png)

Correct Labels:

    6 6 9 8 1 
    9 6 5 8 8 

Prediction : 

    0 0 1 9 3 
    7 5 3 0 7 


## 2. Layer Normalization

#### Summary

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 10, 28, 28]             100
                  ReLU-2           [-1, 10, 28, 28]               0
             LayerNorm-3           [-1, 10, 28, 28]          15,680
             Dropout2d-4           [-1, 10, 28, 28]               0
                Conv2d-5           [-1, 16, 28, 28]           1,456
                  ReLU-6           [-1, 16, 28, 28]               0
             LayerNorm-7           [-1, 16, 28, 28]          25,088
             Dropout2d-8           [-1, 16, 28, 28]               0
             MaxPool2d-9           [-1, 16, 14, 14]               0
               Conv2d-10            [-1, 8, 14, 14]             136
                 ReLU-11            [-1, 8, 14, 14]               0
            LayerNorm-12            [-1, 8, 14, 14]           3,136
            Dropout2d-13            [-1, 8, 14, 14]               0
               Conv2d-14           [-1, 16, 14, 14]           1,168
                 ReLU-15           [-1, 16, 14, 14]               0
            LayerNorm-16           [-1, 16, 14, 14]           6,272
            Dropout2d-17           [-1, 16, 14, 14]               0
               Conv2d-18           [-1, 16, 14, 14]           2,320
                 ReLU-19           [-1, 16, 14, 14]               0
            LayerNorm-20           [-1, 16, 14, 14]           6,272
            Dropout2d-21           [-1, 16, 14, 14]               0
               Conv2d-22           [-1, 16, 14, 14]           2,320
                 ReLU-23           [-1, 16, 14, 14]               0
            LayerNorm-24           [-1, 16, 14, 14]           6,272
            Dropout2d-25           [-1, 16, 14, 14]               0
            MaxPool2d-26             [-1, 16, 7, 7]               0
               Conv2d-27              [-1, 8, 7, 7]             136
                 ReLU-28              [-1, 8, 7, 7]               0
            LayerNorm-29              [-1, 8, 7, 7]             784
            Dropout2d-30              [-1, 8, 7, 7]               0
               Conv2d-31             [-1, 10, 5, 5]             730
                 ReLU-32             [-1, 10, 5, 5]               0
            LayerNorm-33             [-1, 10, 5, 5]             500
            Dropout2d-34             [-1, 10, 5, 5]               0
               Conv2d-35             [-1, 16, 3, 3]           1,456
                 ReLU-36             [-1, 16, 3, 3]               0
            LayerNorm-37             [-1, 16, 3, 3]             288
            Dropout2d-38             [-1, 16, 3, 3]               0
               Conv2d-39             [-1, 10, 3, 3]             170
            AvgPool2d-40             [-1, 10, 1, 1]               0
    ================================================================
    Total params: 74,284
    Trainable params: 74,284
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 1.01
    Params size (MB): 0.28
    Estimated Total Size (MB): 1.30
    ----------------------------------------------------------------

#### Training Logs

    Epoch=1 Loss=0.17002348601818085 batch_id=937 Accuracy=90.24%: 100%|██████████| 938/938 [00:22<00:00, 41.14it/s]

    Test set: Average loss: 0.0764, Accuracy: 9766/10000 (97.66%)

    Epoch=2 Loss=0.05211994796991348 batch_id=937 Accuracy=97.21%: 100%|██████████| 938/938 [00:23<00:00, 40.28it/s]

    Test set: Average loss: 0.0530, Accuracy: 9831/10000 (98.31%)

    Epoch=3 Loss=0.10989350825548172 batch_id=937 Accuracy=97.81%: 100%|██████████| 938/938 [00:23<00:00, 40.64it/s]

    Test set: Average loss: 0.0441, Accuracy: 9876/10000 (98.76%)

    Epoch=4 Loss=0.08162811398506165 batch_id=937 Accuracy=98.12%: 100%|██████████| 938/938 [00:22<00:00, 40.91it/s]

    Test set: Average loss: 0.0378, Accuracy: 9873/10000 (98.73%)

    Epoch=5 Loss=0.01683547906577587 batch_id=937 Accuracy=98.30%: 100%|██████████| 938/938 [00:23<00:00, 40.57it/s]

    Test set: Average loss: 0.0340, Accuracy: 9890/10000 (98.90%)

    Epoch=6 Loss=0.013773527927696705 batch_id=937 Accuracy=98.44%: 100%|██████████| 938/938 [00:23<00:00, 40.75it/s]

    Test set: Average loss: 0.0323, Accuracy: 9901/10000 (99.01%)

    Epoch=7 Loss=0.10910176485776901 batch_id=937 Accuracy=98.56%: 100%|██████████| 938/938 [00:22<00:00, 40.81it/s]

    Test set: Average loss: 0.0356, Accuracy: 9890/10000 (98.90%)

    Epoch=8 Loss=0.1834798902273178 batch_id=937 Accuracy=98.72%: 100%|██████████| 938/938 [00:23<00:00, 40.63it/s]

    Test set: Average loss: 0.0341, Accuracy: 9881/10000 (98.81%)

    Epoch=9 Loss=0.005808831658214331 batch_id=937 Accuracy=98.75%: 100%|██████████| 938/938 [00:22<00:00, 40.94it/s]

    Test set: Average loss: 0.0328, Accuracy: 9897/10000 (98.97%)

    Epoch=10 Loss=0.10751212388277054 batch_id=937 Accuracy=98.78%: 100%|██████████| 938/938 [00:22<00:00, 41.13it/s]

    Test set: Average loss: 0.0280, Accuracy: 9909/10000 (99.09%)

    Epoch=11 Loss=0.14841964840888977 batch_id=937 Accuracy=98.84%: 100%|██████████| 938/938 [00:23<00:00, 40.50it/s]

    Test set: Average loss: 0.0311, Accuracy: 9906/10000 (99.06%)

    Epoch=12 Loss=0.018209289759397507 batch_id=937 Accuracy=98.85%: 100%|██████████| 938/938 [00:22<00:00, 40.91it/s]

    Test set: Average loss: 0.0252, Accuracy: 9915/10000 (99.15%)

    Epoch=13 Loss=0.02086230181157589 batch_id=937 Accuracy=98.89%: 100%|██████████| 938/938 [00:23<00:00, 40.29it/s]

    Test set: Average loss: 0.0280, Accuracy: 9914/10000 (99.14%)

    Epoch=14 Loss=0.01370526384562254 batch_id=937 Accuracy=98.98%: 100%|██████████| 938/938 [00:23<00:00, 40.18it/s]

    Test set: Average loss: 0.0287, Accuracy: 9904/10000 (99.04%)

    Epoch=15 Loss=0.015676043927669525 batch_id=937 Accuracy=98.98%: 100%|██████████| 938/938 [00:22<00:00, 40.91it/s]

    Test set: Average loss: 0.0269, Accuracy: 9911/10000 (99.11%)

#### Misclassified Images

Images

![download](https://user-images.githubusercontent.com/27129645/216755408-e489c0c3-6eaf-4b4b-b25d-c106832ec69e.png)

Correct Labels: 

    3 4 9 4 8 
    3 8 6 5 3 

Prediction : 

    5 9 4 9 6 
    2 3 1 0 2 


## 3. L1 Reguralization with Batch Normalization


#### Summary

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 10, 28, 28]             100
                  ReLU-2           [-1, 10, 28, 28]               0
           BatchNorm2d-3           [-1, 10, 28, 28]              20
             Dropout2d-4           [-1, 10, 28, 28]               0
                Conv2d-5           [-1, 16, 28, 28]           1,456
                  ReLU-6           [-1, 16, 28, 28]               0
           BatchNorm2d-7           [-1, 16, 28, 28]              32
             Dropout2d-8           [-1, 16, 28, 28]               0
             MaxPool2d-9           [-1, 16, 14, 14]               0
               Conv2d-10            [-1, 8, 14, 14]             136
                 ReLU-11            [-1, 8, 14, 14]               0
          BatchNorm2d-12            [-1, 8, 14, 14]              16
            Dropout2d-13            [-1, 8, 14, 14]               0
               Conv2d-14           [-1, 16, 14, 14]           1,168
                 ReLU-15           [-1, 16, 14, 14]               0
          BatchNorm2d-16           [-1, 16, 14, 14]              32
            Dropout2d-17           [-1, 16, 14, 14]               0
               Conv2d-18           [-1, 16, 14, 14]           2,320
                 ReLU-19           [-1, 16, 14, 14]               0
          BatchNorm2d-20           [-1, 16, 14, 14]              32
            Dropout2d-21           [-1, 16, 14, 14]               0
               Conv2d-22           [-1, 16, 14, 14]           2,320
                 ReLU-23           [-1, 16, 14, 14]               0
          BatchNorm2d-24           [-1, 16, 14, 14]              32
            Dropout2d-25           [-1, 16, 14, 14]               0
            MaxPool2d-26             [-1, 16, 7, 7]               0
               Conv2d-27              [-1, 8, 7, 7]             136
                 ReLU-28              [-1, 8, 7, 7]               0
          BatchNorm2d-29              [-1, 8, 7, 7]              16
            Dropout2d-30              [-1, 8, 7, 7]               0
               Conv2d-31             [-1, 10, 5, 5]             730
                 ReLU-32             [-1, 10, 5, 5]               0
          BatchNorm2d-33             [-1, 10, 5, 5]              20
            Dropout2d-34             [-1, 10, 5, 5]               0
               Conv2d-35             [-1, 16, 3, 3]           1,456
                 ReLU-36             [-1, 16, 3, 3]               0
          BatchNorm2d-37             [-1, 16, 3, 3]              32
            Dropout2d-38             [-1, 16, 3, 3]               0
               Conv2d-39             [-1, 10, 3, 3]             170
            AvgPool2d-40             [-1, 10, 1, 1]               0
    ================================================================
    Total params: 10,224
    Trainable params: 10,224
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 1.01
    Params size (MB): 0.04
    Estimated Total Size (MB): 1.05
    ----------------------------------------------------------------
    

#### Training Logs

    Epoch=1 Loss=0.36217600107192993 batch_id=937 Accuracy=90.41%: 100%|██████████| 938/938 [00:26<00:00, 36.00it/s]
    Test set: Average loss: 0.0643, Accuracy: 9793/10000 (97.93%)

    Epoch=2 Loss=0.3840905427932739 batch_id=937 Accuracy=96.59%: 100%|██████████| 938/938 [00:25<00:00, 36.33it/s]
    Test set: Average loss: 0.0604, Accuracy: 9814/10000 (98.14%)

    Epoch=3 Loss=0.2833991050720215 batch_id=937 Accuracy=97.06%: 100%|██████████| 938/938 [00:25<00:00, 36.15it/s]
    Test set: Average loss: 0.0503, Accuracy: 9849/10000 (98.49%)

    Epoch=4 Loss=0.6140828132629395 batch_id=937 Accuracy=97.22%: 100%|██████████| 938/938 [00:25<00:00, 36.09it/s]
    Test set: Average loss: 0.0439, Accuracy: 9867/10000 (98.67%)

    Epoch=5 Loss=0.24126075208187103 batch_id=937 Accuracy=97.32%: 100%|██████████| 938/938 [00:25<00:00, 36.27it/s]
    Test set: Average loss: 0.0435, Accuracy: 9867/10000 (98.67%)

    Epoch=6 Loss=0.3590990900993347 batch_id=937 Accuracy=97.42%: 100%|██████████| 938/938 [00:26<00:00, 35.74it/s]
    Test set: Average loss: 0.0452, Accuracy: 9873/10000 (98.73%)

    Epoch=7 Loss=0.2772316634654999 batch_id=937 Accuracy=97.39%: 100%|██████████| 938/938 [00:26<00:00, 35.05it/s]
    Test set: Average loss: 0.0421, Accuracy: 9863/10000 (98.63%)

    Epoch=8 Loss=0.3154352605342865 batch_id=937 Accuracy=97.45%: 100%|██████████| 938/938 [00:25<00:00, 36.29it/s]
    Test set: Average loss: 0.0386, Accuracy: 9889/10000 (98.89%)

    Epoch=9 Loss=0.2768058180809021 batch_id=937 Accuracy=97.54%: 100%|██████████| 938/938 [00:25<00:00, 36.36it/s]
    Test set: Average loss: 0.0378, Accuracy: 9885/10000 (98.85%)

    Epoch=10 Loss=0.2064865678548813 batch_id=937 Accuracy=97.58%: 100%|██████████| 938/938 [00:25<00:00, 36.30it/s]
    Test set: Average loss: 0.0371, Accuracy: 9898/10000 (98.98%)

    Epoch=11 Loss=0.335493803024292 batch_id=937 Accuracy=97.61%: 100%|██████████| 938/938 [00:25<00:00, 36.62it/s]
    Test set: Average loss: 0.0488, Accuracy: 9830/10000 (98.30%)

    Epoch=12 Loss=0.19641457498073578 batch_id=937 Accuracy=97.50%: 100%|██████████| 938/938 [00:25<00:00, 36.94it/s]
    Test set: Average loss: 0.0369, Accuracy: 9884/10000 (98.84%)

    Epoch=13 Loss=0.20581068098545074 batch_id=937 Accuracy=97.62%: 100%|██████████| 938/938 [00:25<00:00, 36.45it/s]
    Test set: Average loss: 0.0468, Accuracy: 9845/10000 (98.45%)

    Epoch=14 Loss=0.22532521188259125 batch_id=937 Accuracy=97.59%: 100%|██████████| 938/938 [00:26<00:00, 36.03it/s]
    Test set: Average loss: 0.0337, Accuracy: 9892/10000 (98.92%)

    Epoch=15 Loss=0.2543012201786041 batch_id=937 Accuracy=97.74%: 100%|██████████| 938/938 [00:25<00:00, 36.43it/s]
    Test set: Average loss: 0.0390, Accuracy: 9873/10000 (98.73%)

#### Missclassified Images

Images
![download](https://user-images.githubusercontent.com/27129645/216755436-fd69f956-43ba-4826-8696-b741ed9619e6.png)

Correct Labels:

    4 4 7 9 9 
    7 9 3 3 1 

Prediction : 

    9 7 2 7 4 
    2 7 5 7 7 
