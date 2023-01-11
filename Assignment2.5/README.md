# Assignment 2.5
------------
This network combines the predicted classes of MNIST dataset and predicts its sum with a random number.

------------
### Model
![NN_background](https://user-images.githubusercontent.com/27129645/211094396-55de01a2-5b60-49b5-8b2b-7f78dd78d41c.png)

#### Data Generation

MNIST data is taken from torchvision MNIST dataset, and the random number is gererated in the code via the *get_random_num()* function


#### Combining the inputs
Majority of the layers in the network work on the input from MNIST dataset. These layers output 10 classes predicting the handwritten numbers. Theses classes are contactenated with a one-hot encoded random number and fed into a fully connected layer, which outputs the 18 classes of possible sum.

------------


------------
### Results
#### Sample Manual Test

![image](https://user-images.githubusercontent.com/27129645/211918405-4963c2f6-b633-4d19-9898-60b18289202c.png)

#### Training logs
    epoch: 0 total_correct_mnist: 53706 total_correct_sum: 17776 loss: 2064.8953717714176
    epoch: 1 total_correct_mnist: 57398 total_correct_sum: 24674 loss: 1517.9885627613403
    epoch: 2 total_correct_mnist: 57648 total_correct_sum: 28366 loss: 1374.942780266516
    epoch: 3 total_correct_mnist: 57848 total_correct_sum: 30649 loss: 1290.4936307533644
    epoch: 4 total_correct_mnist: 57951 total_correct_sum: 32807 loss: 1225.9846690070117
    epoch: 5 total_correct_mnist: 58001 total_correct_sum: 34011 loss: 1195.7824630569667
    epoch: 6 total_correct_mnist: 58130 total_correct_sum: 34954 loss: 1153.8741208070423
    epoch: 7 total_correct_mnist: 58105 total_correct_sum: 35625 loss: 1134.3785340045579
    epoch: 8 total_correct_mnist: 58171 total_correct_sum: 36373 loss: 1111.2108169090934
    epoch: 9 total_correct_mnist: 58173 total_correct_sum: 36729 loss: 1096.41110892687
    epoch: 10 total_correct_mnist: 58035 total_correct_sum: 37198 loss: 1095.3350114913192
    epoch: 11 total_correct_mnist: 58124 total_correct_sum: 37951 loss: 1069.6860689698951
    epoch: 12 total_correct_mnist: 58086 total_correct_sum: 38311 loss: 1066.410830948269
    epoch: 13 total_correct_mnist: 58181 total_correct_sum: 38739 loss: 1050.1118965654168
    epoch: 14 total_correct_mnist: 58224 total_correct_sum: 39099 loss: 1024.786804831354
    epoch: 15 total_correct_mnist: 58161 total_correct_sum: 39039 loss: 1044.100810224656
    epoch: 16 total_correct_mnist: 58282 total_correct_sum: 39669 loss: 1012.4794547611382
    epoch: 17 total_correct_mnist: 58268 total_correct_sum: 39627 loss: 1013.3098483732902
    epoch: 18 total_correct_mnist: 58159 total_correct_sum: 39835 loss: 1014.990610840614
    epoch: 19 total_correct_mnist: 58300 total_correct_sum: 39873 loss: 1008.1037108855089
    epoch: 20 total_correct_mnist: 58289 total_correct_sum: 40495 loss: 981.3746898045065
    epoch: 21 total_correct_mnist: 58304 total_correct_sum: 40599 loss: 983.3554388250341
    epoch: 22 total_correct_mnist: 58284 total_correct_sum: 40726 loss: 988.5886807050556
    epoch: 23 total_correct_mnist: 58247 total_correct_sum: 40499 loss: 994.7420769802993
    epoch: 24 total_correct_mnist: 58197 total_correct_sum: 40773 loss: 991.5066843263339
    epoch: 25 total_correct_mnist: 58209 total_correct_sum: 40978 loss: 987.2922348523862
    epoch: 26 total_correct_mnist: 58215 total_correct_sum: 41399 loss: 981.0216184346937
    epoch: 27 total_correct_mnist: 58246 total_correct_sum: 41858 loss: 968.2437847593101
    epoch: 28 total_correct_mnist: 58182 total_correct_sum: 41359 loss: 981.2451879631262
    epoch: 29 total_correct_mnist: 58168 total_correct_sum: 41654 loss: 978.3359849010594

**Total correct MNIST images after 30 epochs:** 58168, **Accuracy**: 96.94%

**Total correct overall (MNIST+sum) after 30 epochs**: 41654, **Accuracy**: 69.42%
