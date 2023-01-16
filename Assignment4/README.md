### Assignment 4

Given the network increase its validation accuracy to 99.4% in 15 epochs and under 10,000 parameters.

---------

#### Part-1 Setup and Skeleton

**Target**: 98%

**Explanation for the Target**: This is the first iteration of this model. The objective of this model is to follow expand and squeeze approach along with increase in receptive field to be atleast equal to the image size. This model is not optimized in any way, it uses 10 convolutions and 2 poolings. Hence we should not expect a 99+% accuracy. All these convolutions will also result in a very high number of parameters, but should give satisfactory results.

**Result**:

Max Training Accuracy: 99.92%

Max Validation Accuracy: 99.18%

**Analysis**:

We see that the training accuracy greatly exceeds the test accuracy from epoch 5. This model is over-fitting. We also have 250k+ parameters. We will need to reduce the parameters and optimize the model.

**Colab Link**: https://colab.research.google.com/drive/1sHwp2d7y-yEL7kDInXn1s0QH7Le6Qe4X?usp=sharing

**Model**

    Requirement already satisfied: torchsummary in /usr/local/lib/python3.8/dist-packages (1.5.1)
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 32, 28, 28]             320
                  ReLU-2           [-1, 32, 28, 28]               0
                Conv2d-3           [-1, 64, 28, 28]          18,496
                  ReLU-4           [-1, 64, 28, 28]               0
                Conv2d-5          [-1, 128, 28, 28]          73,856
                  ReLU-6          [-1, 128, 28, 28]               0
             MaxPool2d-7          [-1, 128, 14, 14]               0
                Conv2d-8           [-1, 32, 14, 14]           4,128
                  ReLU-9           [-1, 32, 14, 14]               0
               Conv2d-10           [-1, 64, 14, 14]          18,496
                 ReLU-11           [-1, 64, 14, 14]               0
               Conv2d-12          [-1, 128, 14, 14]          73,856
                 ReLU-13          [-1, 128, 14, 14]               0
            MaxPool2d-14            [-1, 128, 7, 7]               0
               Conv2d-15             [-1, 32, 7, 7]           4,128
                 ReLU-16             [-1, 32, 7, 7]               0
               Conv2d-17             [-1, 32, 5, 5]           9,248
                 ReLU-18             [-1, 32, 5, 5]               0
               Conv2d-19             [-1, 64, 3, 3]          18,496
                 ReLU-20             [-1, 64, 3, 3]               0
               Conv2d-21             [-1, 64, 1, 1]          36,928
                 ReLU-22             [-1, 64, 1, 1]               0
               Conv2d-23             [-1, 10, 1, 1]             650
    ================================================================
    Total params: 258,602
    Trainable params: 258,602
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 3.63
    Params size (MB): 0.99
    Estimated Total Size (MB): 4.62
    ----------------------------------------------------------------

**Training & Validation Logs**

    Epoch=1 Loss=1.3924113512039185 batch_id=937 Accuracy=11.78%: 100%|██████████| 938/938 [00:26<00:00, 35.60it/s]

    Test set: Average loss: 1.2818, Accuracy: 5603/10000 (56.03%)

    Epoch=2 Loss=0.06642214208841324 batch_id=937 Accuracy=95.17%: 100%|██████████| 938/938 [00:22<00:00, 41.13it/s]

    Test set: Average loss: 0.0563, Accuracy: 9806/10000 (98.06%)

    Epoch=3 Loss=0.005162568762898445 batch_id=937 Accuracy=98.67%: 100%|██████████| 938/938 [00:23<00:00, 40.46it/s]

    Test set: Average loss: 0.0477, Accuracy: 9852/10000 (98.52%)

    Epoch=4 Loss=0.07617976516485214 batch_id=937 Accuracy=99.16%: 100%|██████████| 938/938 [00:23<00:00, 40.41it/s]

    Test set: Average loss: 0.0389, Accuracy: 9876/10000 (98.76%)

    Epoch=5 Loss=0.017995836213231087 batch_id=937 Accuracy=99.34%: 100%|██████████| 938/938 [00:23<00:00, 39.71it/s]

    Test set: Average loss: 0.0413, Accuracy: 9874/10000 (98.74%)

    Epoch=6 Loss=0.0050855474546551704 batch_id=937 Accuracy=99.48%: 100%|██████████| 938/938 [00:23<00:00, 40.30it/s]

    Test set: Average loss: 0.0310, Accuracy: 9906/10000 (99.06%)

    Epoch=7 Loss=0.00440952880308032 batch_id=937 Accuracy=99.59%: 100%|██████████| 938/938 [00:22<00:00, 40.80it/s]

    Test set: Average loss: 0.0369, Accuracy: 9889/10000 (98.89%)

    Epoch=8 Loss=0.014854561537504196 batch_id=937 Accuracy=99.71%: 100%|██████████| 938/938 [00:22<00:00, 41.10it/s]

    Test set: Average loss: 0.0351, Accuracy: 9905/10000 (99.05%)

    Epoch=9 Loss=0.00013227369345258921 batch_id=937 Accuracy=99.76%: 100%|██████████| 938/938 [00:23<00:00, 39.41it/s]

    Test set: Average loss: 0.0341, Accuracy: 9900/10000 (99.00%)

    Epoch=10 Loss=0.020665206015110016 batch_id=937 Accuracy=99.81%: 100%|██████████| 938/938 [00:22<00:00, 41.00it/s]

    Test set: Average loss: 0.0353, Accuracy: 9900/10000 (99.00%)

    Epoch=11 Loss=0.00038087277789600194 batch_id=937 Accuracy=99.85%: 100%|██████████| 938/938 [00:22<00:00, 41.30it/s]

    Test set: Average loss: 0.0285, Accuracy: 9920/10000 (99.20%)

    Epoch=12 Loss=0.0005236738361418247 batch_id=937 Accuracy=99.87%: 100%|██████████| 938/938 [00:22<00:00, 40.80it/s]

    Test set: Average loss: 0.0375, Accuracy: 9914/10000 (99.14%)

    Epoch=13 Loss=0.00023113461793400347 batch_id=937 Accuracy=99.88%: 100%|██████████| 938/938 [00:23<00:00, 40.69it/s]

    Test set: Average loss: 0.0284, Accuracy: 9918/10000 (99.18%)

    Epoch=14 Loss=2.557629341026768e-05 batch_id=937 Accuracy=99.92%: 100%|██████████| 938/938 [00:23<00:00, 40.43it/s]

    Test set: Average loss: 0.0302, Accuracy: 9918/10000 (99.18%)

    Epoch=15 Loss=0.005096742417663336 batch_id=937 Accuracy=99.89%: 100%|██████████| 938/938 [00:23<00:00, 40.36it/s]

    Test set: Average loss: 0.0303, Accuracy: 9914/10000 (99.14%)


------

#### Part-2  BatchNorm and DropOut

**Target**: 99%

**Explanation for the Target**: This iteration will reduce the total parameters in the model. It will also introduce image augmentation and reguralization. We will add BatchNormalization and DropOut to each layer in the model. We can expect the model to be efficient than before, so a 99% accuracy is feasible.

**Result**:

Max Training Accuracy: 81.16%

Max Validation Accuracy: 98.51%


**Analysis**: It is clear that is model is under-fitting and there is scope for improvement. Towards the end of the training, we can observe that the training accuracy started oscillating in 80.40s and the test accuracy too was revolving around 98.3%. This could mean the learning rate is higher than expected, but since this is a severely under-fitting model it is early to conclude on the learning rate. We are still using a convolution layer as the final layer to reduce the channels.

This model did achieve to keep the parameters under 10,000. But, there is a huge need for improvement. We will try to introduce Global Average Pooling and StepLR in the next iteration.

**Colab Link: ** https://colab.research.google.com/drive/1L0-7bx4d9OsQkDo-R_wMB5l_k1jJyMzY?usp=sharing


**Model**

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
                Conv2d-9           [-1, 16, 28, 28]           2,320
                 ReLU-10           [-1, 16, 28, 28]               0
          BatchNorm2d-11           [-1, 16, 28, 28]              32
            Dropout2d-12           [-1, 16, 28, 28]               0
            MaxPool2d-13           [-1, 16, 14, 14]               0
               Conv2d-14            [-1, 1, 14, 14]              17
                 ReLU-15            [-1, 1, 14, 14]               0
          BatchNorm2d-16            [-1, 1, 14, 14]               2
            Dropout2d-17            [-1, 1, 14, 14]               0
               Conv2d-18           [-1, 10, 14, 14]             100
                 ReLU-19           [-1, 10, 14, 14]               0
          BatchNorm2d-20           [-1, 10, 14, 14]              20
            Dropout2d-21           [-1, 10, 14, 14]               0
               Conv2d-22           [-1, 16, 14, 14]           1,456
                 ReLU-23           [-1, 16, 14, 14]               0
          BatchNorm2d-24           [-1, 16, 14, 14]              32
            Dropout2d-25           [-1, 16, 14, 14]               0
            MaxPool2d-26             [-1, 16, 7, 7]               0
               Conv2d-27              [-1, 1, 7, 7]              17
                 ReLU-28              [-1, 1, 7, 7]               0
          BatchNorm2d-29              [-1, 1, 7, 7]               2
            Dropout2d-30              [-1, 1, 7, 7]               0
               Conv2d-31             [-1, 10, 5, 5]             100
                 ReLU-32             [-1, 10, 5, 5]               0
          BatchNorm2d-33             [-1, 10, 5, 5]              20
            Dropout2d-34             [-1, 10, 5, 5]               0
               Conv2d-35             [-1, 16, 3, 3]           1,456
                 ReLU-36             [-1, 16, 3, 3]               0
          BatchNorm2d-37             [-1, 16, 3, 3]              32
            Dropout2d-38             [-1, 16, 3, 3]               0
               Conv2d-39             [-1, 16, 1, 1]           2,320
                 ReLU-40             [-1, 16, 1, 1]               0
          BatchNorm2d-41             [-1, 16, 1, 1]              32
            Dropout2d-42             [-1, 16, 1, 1]               0
               Conv2d-43             [-1, 10, 1, 1]             170
    ================================================================
    Total params: 9,736
    Trainable params: 9,736
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 1.21
    Params size (MB): 0.04
    Estimated Total Size (MB): 1.25
    ----------------------------------------------------------------
    
** Training & Validation logs **

    Epoch=1 Loss=0.3884112536907196 batch_id=937 Accuracy=71.13%: 100%|██████████| 938/938 [00:28<00:00, 32.68it/s]

    Test set: Average loss: 0.1288, Accuracy: 9612/10000 (96.12%)

    Epoch=2 Loss=0.673706591129303 batch_id=937 Accuracy=78.37%: 100%|██████████| 938/938 [00:25<00:00, 37.43it/s]

    Test set: Average loss: 0.0986, Accuracy: 9701/10000 (97.01%)

    Epoch=3 Loss=0.9472307562828064 batch_id=937 Accuracy=79.10%: 100%|██████████| 938/938 [00:24<00:00, 37.93it/s]

    Test set: Average loss: 0.0816, Accuracy: 9765/10000 (97.65%)

    Epoch=4 Loss=0.6899013519287109 batch_id=937 Accuracy=79.97%: 100%|██████████| 938/938 [00:24<00:00, 37.86it/s]

    Test set: Average loss: 0.0729, Accuracy: 9781/10000 (97.81%)

    Epoch=5 Loss=0.43427279591560364 batch_id=937 Accuracy=79.82%: 100%|██████████| 938/938 [00:24<00:00, 37.80it/s]

    Test set: Average loss: 0.0667, Accuracy: 9790/10000 (97.90%)

    Epoch=6 Loss=0.6314380168914795 batch_id=937 Accuracy=79.87%: 100%|██████████| 938/938 [00:24<00:00, 38.23it/s]

    Test set: Average loss: 0.0670, Accuracy: 9783/10000 (97.83%)

    Epoch=7 Loss=0.8027915358543396 batch_id=937 Accuracy=80.18%: 100%|██████████| 938/938 [00:24<00:00, 38.16it/s]

    Test set: Average loss: 0.0579, Accuracy: 9828/10000 (98.28%)

    Epoch=8 Loss=0.7604907155036926 batch_id=937 Accuracy=80.33%: 100%|██████████| 938/938 [00:25<00:00, 37.42it/s]

    Test set: Average loss: 0.0619, Accuracy: 9815/10000 (98.15%)

    Epoch=9 Loss=0.386752188205719 batch_id=937 Accuracy=80.53%: 100%|██████████| 938/938 [00:24<00:00, 37.83it/s]

    Test set: Average loss: 0.0610, Accuracy: 9809/10000 (98.09%)

    Epoch=10 Loss=0.5457329750061035 batch_id=937 Accuracy=80.64%: 100%|██████████| 938/938 [00:24<00:00, 38.43it/s]

    Test set: Average loss: 0.0574, Accuracy: 9823/10000 (98.23%)

    Epoch=11 Loss=0.5186195969581604 batch_id=937 Accuracy=80.61%: 100%|██████████| 938/938 [00:24<00:00, 37.91it/s]

    Test set: Average loss: 0.0537, Accuracy: 9833/10000 (98.33%)

    Epoch=12 Loss=1.0028375387191772 batch_id=937 Accuracy=80.61%: 100%|██████████| 938/938 [00:24<00:00, 37.79it/s]

    Test set: Average loss: 0.0591, Accuracy: 9812/10000 (98.12%)

    Epoch=13 Loss=0.8140256404876709 batch_id=937 Accuracy=80.68%: 100%|██████████| 938/938 [00:24<00:00, 38.01it/s]

    Test set: Average loss: 0.0561, Accuracy: 9832/10000 (98.32%)

    Epoch=14 Loss=0.535012423992157 batch_id=937 Accuracy=80.64%: 100%|██████████| 938/938 [00:24<00:00, 37.93it/s]

    Test set: Average loss: 0.0519, Accuracy: 9834/10000 (98.34%)

    Epoch=15 Loss=0.7439004778862 batch_id=937 Accuracy=81.16%: 100%|██████████| 938/938 [00:24<00:00, 38.08it/s]

    Test set: Average loss: 0.0496, Accuracy: 9851/10000 (98.51%)
    
-----

#### Part-3 Fine Tuning

**Target**: 99.4% validation accuracy

**Explanation for the Target**: This iteration will introduce Global Average Pooling to the model. The last iteration showed a severly underfitted model, liekly due to introduction of btach normalization and dropout. We can reduce it by reducing the DropOut rate to 0.05. We will also move the pooling layer to receptive field 5, as patterns are recognized at that layer. The last model also had some extra room for parameters. We will increase the parameter count slightly to make use of the target parameters. We are hoping to achieve the final targeted accuracy of this assignment.

**Result**:

Max Training Accuracy: 98.85%

Max Validation Accuracy: 99.49%


**Analysis**: We see the accuracy revolves around 99.4% in the final epochs. It is safe to say that we have achieved the target.

**Colab Link: ** https://colab.research.google.com/drive/1YRu3rrZFiSqHx3AnFGAkyj6_yBulfMpN?usp=sharing

**Model**

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
    
** Training & Validation Logs **

    Epoch=1 Loss=0.14346933364868164 batch_id=937 Accuracy=90.11%: 100%|██████████| 938/938 [00:25<00:00, 37.10it/s]

    Test set: Average loss: 0.0447, Accuracy: 9853/10000 (98.53%)

    Epoch=2 Loss=0.028531590476632118 batch_id=937 Accuracy=97.17%: 100%|██████████| 938/938 [00:24<00:00, 38.02it/s]

    Test set: Average loss: 0.0376, Accuracy: 9888/10000 (98.88%)

    Epoch=3 Loss=0.16348429024219513 batch_id=937 Accuracy=97.68%: 100%|██████████| 938/938 [00:24<00:00, 38.67it/s]

    Test set: Average loss: 0.0303, Accuracy: 9910/10000 (99.10%)

    Epoch=4 Loss=0.03897513076663017 batch_id=937 Accuracy=97.93%: 100%|██████████| 938/938 [00:24<00:00, 38.40it/s]

    Test set: Average loss: 0.0274, Accuracy: 9911/10000 (99.11%)

    Epoch=5 Loss=0.009880609810352325 batch_id=937 Accuracy=98.10%: 100%|██████████| 938/938 [00:24<00:00, 38.80it/s]

    Test set: Average loss: 0.0260, Accuracy: 9917/10000 (99.17%)

    Epoch=6 Loss=0.033547207713127136 batch_id=937 Accuracy=98.29%: 100%|██████████| 938/938 [00:24<00:00, 38.51it/s]

    Test set: Average loss: 0.0256, Accuracy: 9923/10000 (99.23%)

    Epoch=7 Loss=0.05691586434841156 batch_id=937 Accuracy=98.35%: 100%|██████████| 938/938 [00:24<00:00, 38.55it/s]

    Test set: Average loss: 0.0248, Accuracy: 9928/10000 (99.28%)

    Epoch=8 Loss=0.15364857017993927 batch_id=937 Accuracy=98.47%: 100%|██████████| 938/938 [00:24<00:00, 38.46it/s]

    Test set: Average loss: 0.0238, Accuracy: 9923/10000 (99.23%)

    Epoch=9 Loss=0.06393135339021683 batch_id=937 Accuracy=98.46%: 100%|██████████| 938/938 [00:24<00:00, 38.37it/s]

    Test set: Average loss: 0.0216, Accuracy: 9939/10000 (99.39%)

    Epoch=10 Loss=0.1645202338695526 batch_id=937 Accuracy=98.63%: 100%|██████████| 938/938 [00:25<00:00, 37.44it/s]

    Test set: Average loss: 0.0201, Accuracy: 9925/10000 (99.25%)

    Epoch=11 Loss=0.020322799682617188 batch_id=937 Accuracy=98.64%: 100%|██████████| 938/938 [00:24<00:00, 38.17it/s]

    Test set: Average loss: 0.0195, Accuracy: 9940/10000 (99.40%)

    Epoch=12 Loss=0.022528648376464844 batch_id=937 Accuracy=98.68%: 100%|██████████| 938/938 [00:24<00:00, 38.32it/s]

    Test set: Average loss: 0.0193, Accuracy: 9949/10000 (99.49%)

    Epoch=13 Loss=0.031043240800499916 batch_id=937 Accuracy=98.83%: 100%|██████████| 938/938 [00:24<00:00, 38.22it/s]

    Test set: Average loss: 0.0209, Accuracy: 9933/10000 (99.33%)

    Epoch=14 Loss=0.19452056288719177 batch_id=937 Accuracy=98.85%: 100%|██████████| 938/938 [00:24<00:00, 38.70it/s]

    Test set: Average loss: 0.0197, Accuracy: 9937/10000 (99.37%)

    Epoch=15 Loss=0.08037396520376205 batch_id=937 Accuracy=98.82%: 100%|██████████| 938/938 [00:24<00:00, 38.37it/s]

    Test set: Average loss: 0.0179, Accuracy: 9946/10000 (99.46%)
