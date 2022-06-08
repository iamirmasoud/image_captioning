# Automatic Image Captioning

## Project Overview
In this project, you will create a neural network architecture to automatically generate captions from images.

In this project, I'll create a neural network architecture consisting of both CNNs and LSTMs to automatically generate captions from images.

In this project, I design and train a CNN-RNN (Convolutional Neural Network - Recurrent Neural Network) model for  automatically generating image captions. The network is trained on the Microsoft Common Objects in COntext [(MS COCO)](http://cocodataset.org/#home) dataset. The image captioning model is displayed below.

![Image Captioning Model](images/cnn_rnn_model.png?raw=true) [Image source](https://arxiv.org/pdf/1411.4555.pdf)


After using the Microsoft Common Objects in COntext [MS COCO](http://cocodataset.org/#home "MS COCO") dataset to train your network, you will test your network on novel images!

![Example](https://video.udacity-data.com/topher/2018/March/5ab588e3_image-captioning/image-captioning.png)



# Instructions  
1. Clone this repo: https://github.com/cocodataset/cocoapi  
```
git clone https://github.com/cocodataset/cocoapi.git  
```

2. Setup the coco API (also described in the readme [here](https://github.com/cocodataset/cocoapi)) 
```
cd cocoapi/PythonAPI  
make  
cd ..
```

3. Download some specific data from here: http://cocodataset.org/#download (described below)

* Under **Annotations**, download:
  * **2014 Train/Val annotations [241MB]** (extract captions_train2014.json and captions_val2014.json, and place at locations cocoapi/annotations/captions_train2014.json and cocoapi/annotations/captions_val2014.json, respectively)  
  * **2014 Testing Image info [1MB]** (extract image_info_test2014.json and place at location cocoapi/annotations/image_info_test2014.json)

* Under **Images**, download:
  * **2014 Train images [83K/13GB]** (extract the train2014 folder and place at location cocoapi/images/train2014/)
  * **2014 Val images [41K/6GB]** (extract the val2014 folder and place at location cocoapi/images/val2014/)
  * **2014 Test images [41K/6GB]** (extract the test2014 folder and place at location cocoapi/images/test2014/)

4. The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order (`0_Dataset.ipynb, 1_Preliminaries.ipynb, 2_Training.ipynb, 3_Inference.ipynb`).


### Dependencies

Before you can experiment with the code, you'll have to make sure that you have all the libraries and dependencies required to support this project. You will mainly need Python3.7+, PyTorch and its torchvision, OpenCV, Matplotlib. You can install  dependencies using `pip3 install -r requirements.txt`.


## Getting the Files

### Data

The Microsoft **C**ommon **O**bjects in **CO**ntext (MS COCO) dataset is a large-scale dataset for scene understanding.  The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms.  

![Sample Coco Example](images/coco-examples.jpg)

You can read more about the dataset on the [website](http://cocodataset.org/#home) or in the [research paper](https://arxiv.org/pdf/1405.0312.pdf).

To obtain and explore the dataset, you can use either the [COCO API](https://github.com/cocodataset/cocoapi), or run the __0.Dataset Exploration.ipynb__.

### Model Download

The core architecture used to achieve this task follows an encoder-decoder architecture, where the encoder is a pretrained ResNet CNN on ImageNet, and the decoder is a basic one-layer LSTM.

![encoder-decoder-architecture](images/encoder-decoder.png)

You can use my pre-trained model for your own experimentation. To use it, [download](https://www.dropbox.com/sh/z95hjvylmqqzqr0/AABRRLCVHzRSnUico5xV_-Y0a?raw=1). After downloading, unzip the file and place the contained pickle models under the subdirectory `models`.

Please feel free to experiment with alternative architectures, such as bidirectional LSTM with attention mechanisms.




## Project Description
The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order:

0_Dataset.ipynb
1_Preliminaries.ipynb
2_Training.ipynb
3_Inference.ipynb

## LSTM Decoder
In the project, we pass all our inputs as a sequence to an LSTM. A sequence looks like this: first a feature vector that is extracted from an input image, then a start word, then the next word, the next word, and so on!

## Embedding Dimension
The LSTM is defined such that, as it sequentially looks at inputs, it expects that each individual input in a sequence is of a consistent size and so we embed the feature vector and each word so that they are embed_size.

Sequential Inputs
So, an LSTM looks at inputs sequentially. In PyTorch, there are two ways to do this.

The first is pretty intuitive: for all the inputs in a sequence, which in this case would be a feature from an image, a start word, the next word, the next word, and so on (until the end of a sequence/batch), you loop through each input like so:



for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)
The second approach, which this project uses, is to give the LSTM our entire sequence and have it produce a set of outputs and the last hidden state:

 ***the first value returned by LSTM is all of the hidden states throughout***
 
 
 ***the sequence. the second is just the most recent hidden state***

## Add the extra 2nd dimension

inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)




Project instructions are found in the [Udacity README](README_Udacity.md).

## Method:

[The Dataset notebook](0_Dataset.ipynb) initializes the [COCO API](https://github.com/cocodataset/cocoapi) (the "pycocotools" library) used to access data from the MS COCO (Common Objects in Context) dataset, which is "commonly used to train and benchmark object detection, segmentation, and captioning algorithms." The notebook also depicts the processing pipeline using the following diagram:

![foo](images/encoder-decoder.png)

The left half of the diagram depicts the "EncoderCNN", which encodes the critical information contained in a regular picture file into a "feature vector" of a specific size. That feature vector is fed into the "DecoderRNN" on the right half of the diagram (which is "unfolded" in time - each box labeled "LSTM" represents the same cell at a different time step). Each word appearing as output at the top is fed back to the network as input (at the bottom) in a subsequent time step, until the entire caption is generated. The arrow pointing right that connects the LSTM boxes together represents hidden state information, which represents the network's "memory", also fed back to the LSTM at each time step.

[The Architecture notebook](1_Preliminaries.ipynb) uses the pycocotools, torchvision transforms, and NLTK to preprocess the images and the captions for network training. It also explores the EncoderCNN, which is taken pretrained from [torchvision.models, the ResNet50 architecture](https://pytorch.org/docs/master/torchvision/models.html#id3).

The implementations of the EncoderCNN, which is supplied, and the DecoderRNN, which is left to the student, are found in the [model.py](model.py) file.

In [the Training notebook](2_Training.ipynb) one finds selection of hyperparameter values and EncoderRNN training. The hyperparameter selections are explained.

[The Inference notebook](3_Inference.ipynb) contains the testing of the trained networks to generate captions for additional images. No rigorous validation or accuracy measurement was performed, only sample images were generated. See below.

## Results:
Here are some predictions from my model.

### Good results
![sample_171](samples/sample_171.png?raw=true)<br/>
![sample_440](samples/sample_440.png?raw=true)<br/>
![sample_457](samples/sample_457.png?raw=true)<br/>
![sample_002](samples/sample_002.png?raw=true)<br/>
![sample_029](samples/sample_029.png?raw=true)<br/>
![sample_107](samples/sample_107.png?raw=true)<br/>
![sample_202](samples/sample_202.png?raw=true)


### Not so good results

![sample_296](samples/sample_296.png?raw=true)<br/>
![sample_008](samples/sample_008.png?raw=true)<br/>
![sample_193](samples/sample_193.png?raw=true)<br/>
![sample_034](samples/sample_034.png?raw=true)<br/>
![sample_326](samples/sample_326.png?raw=true)<br/>
![sample_366](samples/sample_366.png?raw=true)<br/>
![sample_498](samples/sample_498.png?raw=true)

### More samples
There are 500 predictions in the samples folder.


Steps for additional improvement would be exploring the hyperparameter, other architectures, and also training with more epochs.


COCO API - http://cocodataset.org/

COCO is a large image dataset designed for object detection, segmentation, person keypoints detection, stuff segmentation, and caption generation. This package provides Matlab, Python, and Lua APIs that assists in loading, parsing, and visualizing the annotations in COCO. Please visit http://cocodataset.org/ for more information on COCO, including for the data, paper, and tutorials. The exact format of the annotations is also described on the COCO website. The Matlab and Python APIs are complete, the Lua API provides only basic functionality.

In addition to this API, please download both the COCO images and annotations in order to run the demos and use the API. Both are available on the project website.
-Please download, unzip, and place the images in: coco/images/
-Please download and place the annotations in: coco/annotations/
For substantially more details on the API please see http://cocodataset.org/#download.

After downloading the images and annotations, run the Matlab, Python, or Lua demos for example usage.

To install:
-For Matlab, add coco/MatlabApi to the Matlab path (OSX/Linux binaries provided)
-For Python, run "make" under coco/PythonAPI
-For Lua, run “luarocks make LuaAPI/rocks/coco-scm-1.rockspec” under coco/


Note: This project is a part of [Udacity Computer Vision Nanodegree Program](https://www.udacity.com/course/computer-vision-nanodegree--nd891)