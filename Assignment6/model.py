import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional Block #1
        self.convblock1 = nn.Sequential( # 32x32 > 32x32 | jin=1 | RF=5
          nn.Conv2d(3, 16, 5, padding=2),
          nn.ReLU()
        )
        # Convolutional Block #2
        self.convblock2 = nn.Sequential(
          nn.Conv2d(16, 32, 3, dilation=1, padding=2), # 32x32 > 32x32 | jin=1 | RF=9
          nn.ReLU()
        )
        # Transitional Block #1
        self.sconv1 = nn.Sequential( # 32x32 > 15x15 | jin=1 | RF=11
          nn.Conv2d(16, 16, 3, stride=2), 
          nn.ReLU()
        )

        # Convolutional Block #3
        # Depthwise Separable Convolution
        self.convblock3 = nn.Sequential( # 15x15 > 15x15 | jin=2 | RF = 15
          nn.Conv2d(16, 16, 3, groups=1, padding=1),
          nn.Conv2d(16, 8, 1),
          nn.ReLU()
        )
        # Transitional Block #2
        self.sconv2 = nn.Sequential( # 15x15 > 7x7 | jin=2 | RF=19
          nn.Conv2d(8, 8, 3, stride=2),
          nn.ReLU()
        )

        #Convolutional Block #4
        self.convblock4 = nn.Sequential( # 7x7 > 7x7 | jin=4 | RF=35
          nn.Conv2d(8, 16, 5, padding=2),
          nn.ReLU()
        )
        # Transitional Block #3
        self.sconv3 = nn.Sequential( # 7x7 > 3x3 | jin=4 | RF=43
          nn.Conv2d(16, 16, 3, stride=2), 
          nn.ReLU()
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1,1)) # 3x3 > 1x1 | jin=8 | RF=59

        # Fully Connected Layer / Output Layer
        self.out = nn.Linear(16 * 1 * 1, 10)

        '''
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        '''

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.sconv1(x)
        x = self.convblock3(x)
        x = self.sconv2(x)
        x = self.convblock4(x)
        x = self.sconv3(x)
        x = self.gap(x)
        x = x.view(-1, 16*1*1)
        x = self.out(x)
       
        '''
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        '''
        return x


#net = Net()