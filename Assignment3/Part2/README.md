## Assignment-3 Part-2
Given the NN model, increase its test accuracy to 99.4%

**Constraints**
- [x] Parameters under 20k
- [x] Less than 20 epochs

**Use**

- [x] Batch Normalization
- [x] DropOut
- [x] A fully connected layer
- [x] Global Average Pooling


------------

### Model

This solution applies squeeze-and-expand approach with BatchNorm and DropOut applied to each layer. At the end Global Average Pooling is applied and a final fully connected layer calcualtes the output classes. log_softmax along with NLL loss function is used in this model. The total number of parameters is 18.6k and the test accuracy was 99.26%.

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1            [-1, 8, 28, 28]              80
           BatchNorm2d-2            [-1, 8, 28, 28]              16
             Dropout2d-3            [-1, 8, 28, 28]               0
                Conv2d-4           [-1, 16, 28, 28]           1,168
           BatchNorm2d-5           [-1, 16, 28, 28]              32
             Dropout2d-6           [-1, 16, 28, 28]               0
             MaxPool2d-7           [-1, 16, 14, 14]               0
                Conv2d-8           [-1, 32, 14, 14]           4,640
           BatchNorm2d-9           [-1, 32, 14, 14]              64
            Dropout2d-10           [-1, 32, 14, 14]               0
               Conv2d-11            [-1, 8, 14, 14]             264
            MaxPool2d-12              [-1, 8, 7, 7]               0
               Conv2d-13             [-1, 16, 7, 7]           1,168
          BatchNorm2d-14             [-1, 16, 7, 7]              32
            Dropout2d-15             [-1, 16, 7, 7]               0
               Conv2d-16             [-1, 32, 7, 7]           4,640
          BatchNorm2d-17             [-1, 32, 7, 7]              64
            Dropout2d-18             [-1, 32, 7, 7]               0
               Conv2d-19              [-1, 8, 7, 7]             264
               Conv2d-20             [-1, 16, 7, 7]           1,168
          BatchNorm2d-21             [-1, 16, 7, 7]              32
            Dropout2d-22             [-1, 16, 7, 7]               0
               Conv2d-23             [-1, 32, 7, 7]           4,640
          BatchNorm2d-24             [-1, 32, 7, 7]              64
            Dropout2d-25             [-1, 32, 7, 7]               0
    AdaptiveAvgPool2d-26             [-1, 32, 1, 1]               0
               Linear-27                   [-1, 10]             330
    ================================================================
	Total params: 18,666
	Trainable params: 18,666
	Non-trainable params: 0
	----------------------------------------------------------------
	Input size (MB): 0.00
	Forward/backward pass size (MB): 0.72
	Params size (MB): 0.07
	Estimated Total Size (MB): 0.80


------------

### Training/Testing Logs
	  0%|          | 0/469 [00:00<?, ?it/s]<ipython-input-10-24526b341a74>:76: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
	  return F.log_softmax(x)
	loss=0.25160324573516846 batch_id=468: 100%|██████████| 469/469 [01:25<00:00,  5.47it/s]
	Test set: Average loss: 0.1197, Accuracy: 9660/10000 (96.60%)

	loss=0.09150660783052444 batch_id=468: 100%|██████████| 469/469 [01:22<00:00,  5.70it/s]
	Test set: Average loss: 0.0687, Accuracy: 9811/10000 (98.11%)

	loss=0.08957917243242264 batch_id=468: 100%|██████████| 469/469 [01:23<00:00,  5.65it/s]
	Test set: Average loss: 0.0447, Accuracy: 9860/10000 (98.60%)

	loss=0.07676941156387329 batch_id=468: 100%|██████████| 469/469 [01:22<00:00,  5.69it/s]
	Test set: Average loss: 0.0434, Accuracy: 9865/10000 (98.65%)

	loss=0.09344455599784851 batch_id=468: 100%|██████████| 469/469 [01:22<00:00,  5.67it/s]
	Test set: Average loss: 0.0382, Accuracy: 9868/10000 (98.68%)

	loss=0.03506695106625557 batch_id=468: 100%|██████████| 469/469 [01:22<00:00,  5.70it/s]
	Test set: Average loss: 0.0338, Accuracy: 9889/10000 (98.89%)

	loss=0.14049971103668213 batch_id=468: 100%|██████████| 469/469 [01:22<00:00,  5.69it/s]
	Test set: Average loss: 0.0359, Accuracy: 9883/10000 (98.83%)

	loss=0.06398900598287582 batch_id=468: 100%|██████████| 469/469 [01:22<00:00,  5.66it/s]
	Test set: Average loss: 0.0377, Accuracy: 9880/10000 (98.80%)

	loss=0.04205959662795067 batch_id=468: 100%|██████████| 469/469 [01:22<00:00,  5.71it/s]
	Test set: Average loss: 0.0291, Accuracy: 9901/10000 (99.01%)

	loss=0.046533580869436264 batch_id=468: 100%|██████████| 469/469 [01:22<00:00,  5.66it/s]
	Test set: Average loss: 0.0309, Accuracy: 9904/10000 (99.04%)

	loss=0.045987021178007126 batch_id=468: 100%|██████████| 469/469 [01:22<00:00,  5.69it/s]
	Test set: Average loss: 0.0279, Accuracy: 9901/10000 (99.01%)

	loss=0.08294835686683655 batch_id=468: 100%|██████████| 469/469 [01:22<00:00,  5.66it/s]
	Test set: Average loss: 0.0268, Accuracy: 9897/10000 (98.97%)

	loss=0.029629571363329887 batch_id=468: 100%|██████████| 469/469 [01:22<00:00,  5.69it/s]
	Test set: Average loss: 0.0254, Accuracy: 9910/10000 (99.10%)

	loss=0.054499149322509766 batch_id=468: 100%|██████████| 469/469 [01:21<00:00,  5.74it/s]
	Test set: Average loss: 0.0284, Accuracy: 9898/10000 (98.98%)

	loss=0.06466329842805862 batch_id=468: 100%|██████████| 469/469 [01:22<00:00,  5.66it/s]
	Test set: Average loss: 0.0241, Accuracy: 9922/10000 (99.22%)

	loss=0.053656529635190964 batch_id=468: 100%|██████████| 469/469 [01:22<00:00,  5.71it/s]
	Test set: Average loss: 0.0253, Accuracy: 9923/10000 (99.23%)

	loss=0.058036137372255325 batch_id=468: 100%|██████████| 469/469 [01:22<00:00,  5.66it/s]
	Test set: Average loss: 0.0250, Accuracy: 9915/10000 (99.15%)

	loss=0.03384203836321831 batch_id=468: 100%|██████████| 469/469 [01:23<00:00,  5.64it/s]
	Test set: Average loss: 0.0272, Accuracy: 9907/10000 (99.07%)

	loss=0.09015195816755295 batch_id=468: 100%|██████████| 469/469 [01:23<00:00,  5.59it/s]
	Test set: Average loss: 0.0234, Accuracy: 9926/10000 (99.26%)


**Test Accuracy:** 99.26%
