# Assignment 9: Transformer Basics

Link to pytorch_cifar library: [pytorch-cifar-for-EVA](https://github.com/ajinkyakhadilkar/pytorch-cifar-for-EVA)

## Summary
In this assignment, I implement a basic representation of Vision Transformer. The image is increased till 48 channels, converted to 1x1x48 using Global Average Pooling. The Ultimus block transforms it to size 8 query, key and value vectors using fully connected layers. After calculating the attention scores, we transform them to 48 output channels using a fully connected layer. These 48 values are added to the original 48 channels passed to the Ultimus block. The Ultimus block is repeated 4 times before reducing the channels to 10 for the output layer.

### Model

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 16, 32, 32]             448
                  ReLU-2           [-1, 16, 32, 32]               0
                Conv2d-3           [-1, 32, 32, 32]           4,640
                  ReLU-4           [-1, 32, 32, 32]               0
                Conv2d-5           [-1, 48, 32, 32]          13,872
                  ReLU-6           [-1, 48, 32, 32]               0
     AdaptiveAvgPool2d-7             [-1, 48, 1, 1]               0
                Linear-8                    [-1, 8]             392
                Linear-9                    [-1, 8]             392
               Linear-10                    [-1, 8]             392
               Linear-11                   [-1, 48]             432
              Ultimus-12                   [-1, 48]               0
               Linear-13                    [-1, 8]             392
               Linear-14                    [-1, 8]             392
               Linear-15                    [-1, 8]             392
               Linear-16                   [-1, 48]             432
              Ultimus-17                   [-1, 48]               0
               Linear-18                    [-1, 8]             392
               Linear-19                    [-1, 8]             392
               Linear-20                    [-1, 8]             392
               Linear-21                   [-1, 48]             432
              Ultimus-22                   [-1, 48]               0
               Linear-23                    [-1, 8]             392
               Linear-24                    [-1, 8]             392
               Linear-25                    [-1, 8]             392
               Linear-26                   [-1, 48]             432
              Ultimus-27                   [-1, 48]               0
               Linear-28                   [-1, 10]             490
     TransformerModel-29                   [-1, 10]               0
    ================================================================
    Total params: 25,882
    Trainable params: 25,882
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 1.50
    Params size (MB): 0.10
    Estimated Total Size (MB): 1.61
    ----------------------------------------------------------------
    
    
### The Ultimus Block 
This block implements the transformer using query, key and value tensors.

``` python
class Ultimus(nn.Module):
    def __init__(self):
        super(Ultimus, self).__init__()
        self.fc1_k = nn.Linear(48, 8)
        self.fc2_q = nn.Linear(48, 8)
        self.fc3_v = nn.Linear(48, 8)
        self.out = nn.Linear(8, 48)
       
    def forward(self, x):
        orig = x
        k = self.fc1_k(x)
        q = self.fc2_q(x)
        v = self.fc3_v(x)
        am = F.softmax(torch.div(torch.multiply(q, k),(8**0.5)))
        z = torch.multiply(v, am)
        out = self.out(z)
        out = torch.add(out,orig)
        return out
```

### Range Test for One Cycle Policy

``` python
import matplotlib.pyplot as plt

main.start_training(1, is_range_test=True)
print(main.lr)
print(main.train_losses)
def plot(lr, losses, start=10, end=-5):
    plt.xlabel("Learning Rate")
    plt.ylabel("Losses")
    plt.plot(lr, losses)
    plt.xscale('log')
print(main.lr)
print(main.train_losses)
plot(main.lr, main.train_losses)
```
![range_test](https://user-images.githubusercontent.com/27129645/221429940-a299238d-65b8-44a6-b5a2-5ac4ce891fe8.jpg)

A Max LR of 0.02 looks like the ideal value for the One Cycle Policy


### Training Logs
    Epoch=0 LR=0.0026267510015656713 Loss=2.031066616477869 batch_id=390 Accuracy=23.32%: 100%|█| 391/391 [00:13<00:00, 29.

    Test set: Average loss: 1.8646, Accuracy: 2970/10000 (29.70%)

    Epoch=1 LR=0.0074264657372516835 Loss=1.8016941346170958 batch_id=390 Accuracy=31.37%: 100%|█| 391/391 [00:12<00:00, 31

    Test set: Average loss: 1.6896, Accuracy: 3612/10000 (36.12%)

    Epoch=2 LR=0.013363445782935248 Loss=1.68522828466752 batch_id=390 Accuracy=36.19%: 100%|█| 391/391 [00:12<00:00, 30.64

    Test set: Average loss: 1.6661, Accuracy: 3777/10000 (37.77%)

    Epoch=3 LR=0.018167018983702946 Loss=1.5987703019700696 batch_id=390 Accuracy=40.62%: 100%|█| 391/391 [00:13<00:00, 30.

    Test set: Average loss: 1.4947, Accuracy: 4543/10000 (45.43%)

    Epoch=4 LR=0.019999999912520917 Loss=1.5008331214070625 batch_id=390 Accuracy=44.75%: 100%|█| 391/391 [00:12<00:00, 30.

    Test set: Average loss: 1.4661, Accuracy: 4705/10000 (47.05%)

    Epoch=5 LR=0.019863407248932475 Loss=1.4356180325798367 batch_id=390 Accuracy=47.26%: 100%|█| 391/391 [00:12<00:00, 30.

    Test set: Average loss: 1.4203, Accuracy: 4790/10000 (47.90%)

    Epoch=6 LR=0.019457790242192245 Loss=1.3830962979885013 batch_id=390 Accuracy=49.43%: 100%|█| 391/391 [00:12<00:00, 30.

    Test set: Average loss: 1.3164, Accuracy: 5141/10000 (51.41%)

    Epoch=7 LR=0.018794212137341757 Loss=1.3587120844580023 batch_id=390 Accuracy=50.47%: 100%|█| 391/391 [00:12<00:00, 31.

    Test set: Average loss: 1.3686, Accuracy: 5031/10000 (50.31%)

    Epoch=8 LR=0.01789077209456453 Loss=1.3248384254972647 batch_id=390 Accuracy=52.00%: 100%|█| 391/391 [00:12<00:00, 30.1

    Test set: Average loss: 1.4475, Accuracy: 4880/10000 (48.80%)

    Epoch=9 LR=0.016772111532754447 Loss=1.290140253198726 batch_id=390 Accuracy=53.42%: 100%|█| 391/391 [00:12<00:00, 30.5

    Test set: Average loss: 1.2994, Accuracy: 5300/10000 (53.00%)

    Epoch=10 LR=0.015468742032313564 Loss=1.2614626491161258 batch_id=390 Accuracy=54.52%: 100%|█| 391/391 [00:12<00:00, 31

    Test set: Average loss: 1.2559, Accuracy: 5545/10000 (55.45%)

    Epoch=11 LR=0.014016213128698756 Loss=1.2431139997814014 batch_id=390 Accuracy=55.12%: 100%|█| 391/391 [00:12<00:00, 31

    Test set: Average loss: 1.2407, Accuracy: 5453/10000 (54.53%)

    Epoch=12 LR=0.012454142695232777 Loss=1.2109728914392575 batch_id=390 Accuracy=56.44%: 100%|█| 391/391 [00:12<00:00, 30

    Test set: Average loss: 1.2178, Accuracy: 5612/10000 (56.12%)

    Epoch=13 LR=0.01082513636158692 Loss=1.1843270961280978 batch_id=390 Accuracy=57.37%: 100%|█| 391/391 [00:12<00:00, 30.

    Test set: Average loss: 1.1335, Accuracy: 5930/10000 (59.30%)

    Epoch=14 LR=0.009173625440905952 Loss=1.1673536294561517 batch_id=390 Accuracy=58.15%: 100%|█| 391/391 [00:12<00:00, 30

    Test set: Average loss: 1.1616, Accuracy: 5799/10000 (57.99%)

    Epoch=15 LR=0.007544655061230937 Loss=1.1296061695079365 batch_id=390 Accuracy=59.60%: 100%|█| 391/391 [00:13<00:00, 30

    Test set: Average loss: 1.1518, Accuracy: 5897/10000 (58.97%)

    Epoch=16 LR=0.005982655555058443 Loss=1.1106056180756416 batch_id=390 Accuracy=60.13%: 100%|█| 391/391 [00:12<00:00, 30

    Test set: Average loss: 1.1642, Accuracy: 5768/10000 (57.68%)

    Epoch=17 LR=0.004530230617510627 Loss=1.0832886915377644 batch_id=390 Accuracy=61.30%: 100%|█| 391/391 [00:12<00:00, 30

    Test set: Average loss: 1.0884, Accuracy: 6052/10000 (60.52%)

    Epoch=18 LR=0.0032269952862252088 Loss=1.0576721373421456 batch_id=390 Accuracy=62.16%: 100%|█| 391/391 [00:12<00:00, 3

    Test set: Average loss: 1.0870, Accuracy: 6116/10000 (61.16%)

    Epoch=19 LR=0.002108495437181735 Loss=1.0270280434042596 batch_id=390 Accuracy=63.32%: 100%|█| 391/391 [00:13<00:00, 29

    Test set: Average loss: 1.0624, Accuracy: 6273/10000 (62.73%)

    Epoch=20 LR=0.0012052382673252257 Loss=1.0135519548755167 batch_id=390 Accuracy=63.90%: 100%|█| 391/391 [00:12<00:00, 3

    Test set: Average loss: 1.0185, Accuracy: 6345/10000 (63.45%)

    Epoch=21 LR=0.0005418602076720879 Loss=0.9867091438044673 batch_id=390 Accuracy=64.96%: 100%|█| 391/391 [00:13<00:00, 2

    Test set: Average loss: 0.9971, Accuracy: 6433/10000 (64.33%)

    Epoch=22 LR=0.00013645496215277061 Loss=0.970521935569051 batch_id=390 Accuracy=65.63%: 100%|█| 391/391 [00:13<00:00, 2

    Test set: Average loss: 0.9942, Accuracy: 6432/10000 (64.32%)

    Epoch=23 LR=8e-08 Loss=0.9640273300887984 batch_id=390 Accuracy=65.86%: 100%|████████| 391/391 [00:12<00:00, 30.37it/s]

    Test set: Average loss: 0.9913, Accuracy: 6443/10000 (64.43%)

The training accuracy for 24 epochs is 65.86% and validation accuract is 64.43%. The model is slightly overfitting and the training can be improved by introducing more reguralization.

![train_test_loss](https://user-images.githubusercontent.com/27129645/221430028-e2d4297b-4aca-491b-bc6b-78b3a995fa1e.jpg)
_A graphical representation of training and validation losses_
