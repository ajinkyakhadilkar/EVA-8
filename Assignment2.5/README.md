# Assignment 2.5
------------
This network combines the predicted classes of MNIST dataset and predicts its sum with a random number.

------------
### Model
![NN_background](https://user-images.githubusercontent.com/27129645/211094396-55de01a2-5b60-49b5-8b2b-7f78dd78d41c.png)

#### Data Generation

MNIST data is taken from torchvision MNIST dataset, and the random number is gererated in the code via the *get_random_num()* function


#### Combining the inputs
Majority of the layers in the network work on the input from MNIST dataset. These layers output 10 classes predicting the handwritten numbers. Theses classes are contactenated with one random number and fed into a fully connected layer, which outputs the 18 classes of possible sum.

------------


------------
### Results
#### Sample Manual Test

![image](https://user-images.githubusercontent.com/27129645/211095543-16694d8e-6cca-4676-adf8-ac819283dad8.png)


#### Total correct MNIST
#### Total correct overall (MNIST+sum)

###Accuracy with test data
