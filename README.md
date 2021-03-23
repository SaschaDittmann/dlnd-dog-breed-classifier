# Dog Breed Classifier in PyTorch
This repo contains my solution of project 2 of the [Udacity Deep Learning Nanodegree™️](https://www.udacity.com/course/deep-learning-nanodegree--nd101) program.

It is implemented by using the OpenCV and PyTorch libraries.

**Please check out Udacity's original repo [here](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-dog-classification)**

## Project Overview

Welcome to the Convolutional Neural Networks (CNN) project in the AI Nanodegree! In this project, you will learn how to build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images. Given an image of a dog, your algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

[![Sample Output](https://github.com/udacity/deep-learning-v2-pytorch/raw/master/project-dog-classification/images/sample_dog_output.png)](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/project-dog-classification/images/sample_dog_output.png)

Along with exploring state-of-the-art CNN models for classification and localization, you will make important design decisions about the user experience for your app. Our goal is that by completing this lab, you understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline. Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer. Your imperfect solution will nonetheless create a fun user experience!

## Used Datasets

I used the following two datasets for training the models:

* Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
* Download the [human_dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)

I used images from [Unsplash](https://unsplash.com/) to test the final algorithm

## Creating a CNN to Classify Dog Breeds from Scratch

My CNN structure for this task looks like this:

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 16, 224, 224]             448
         MaxPool2d-2         [-1, 16, 112, 112]               0
            Conv2d-3         [-1, 32, 112, 112]           4,640
         MaxPool2d-4           [-1, 32, 56, 56]               0
            Conv2d-5           [-1, 64, 56, 56]          18,496
         MaxPool2d-6           [-1, 64, 28, 28]               0
           Dropout-7           [-1, 64, 28, 28]               0
            Linear-8                  [-1, 512]      25,690,624
           Dropout-9                  [-1, 512]               0
           Linear-10                  [-1, 133]          68,229
================================================================
Total params: 25,782,437
Trainable params: 25,782,437
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 13.79
Params size (MB): 98.35
Estimated Total Size (MB): 112.72
----------------------------------------------------------------
```

-----

With this model I achieved an accuracy of up to **26%** with **100 epochs**

## Creating a CNN to Classify Dog Breeds using Transfer Learning

For the **Transfer Learning** task, I used pre-trained model VGG 16-layer model (configuration “D”) with batch normalization.

More information about "Very Deep Convolutional Networks For Large-Scale Image Recognition" [here](https://arxiv.org/pdf/1409.1556.pdf)

To classify the dog breeds I added the following layers:
```
model_transfer.classifier = nn.Sequential(
                                         nn.Linear(25088, 512),
                                         nn.ReLU(),
                                         nn.Linear(512, 133),
                                         nn.LogSoftmax(dim=1)
                                         )
```

With this model I achieved an accuracy of up to **86%** with **30 epochs**
