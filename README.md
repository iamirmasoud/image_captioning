# Automatic Image Captioning

## Project Overview

In this project, I'll create a neural network architecture consisting of both CNNs (Encoder) and LSTMs (Decoder) to automatically generate captions from images.
The network will be trained on the Microsoft Common Objects in COntext [(MS COCO)](http://cocodataset.org/#home) dataset. The image captioning model is displayed below.

![Image Captioning Model](images/cnn_rnn_model.png?raw=true) [Image source](https://arxiv.org/pdf/1411.4555.pdf)

After training the network, I will test the network on novel images!

## Preparing the environment
**Note**: I have developed this project on __Linux__. It can surely be run on Windows and Mac with some little changes.

1. Clone the repository, and navigate to the downloaded folder.
```
git clone https://github.com/iamirmasoud/image_captioning.git
cd image_captioning
```

2. Create (and activate) a new environment, named `captioning_env` with Python 3.7. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	```shell
	conda create -n captioning_env python=3.7
	source activate captioning_env
	```
	
	At this point your command line should look something like: `(captioning_env) <User>:image_captioning <user>$`. The `(captioning_env)` indicates that your environment has been activated, and you can proceed with further package installations.

6. Before you can experiment with the code, you'll have to make sure that you have all the libraries and dependencies required to support this project. You will mainly need Python3.7+, PyTorch and its torchvision, OpenCV, Matplotlib. You can install  dependencies using:
```
pip install -r requirements.txt
```

7. Navigate back to the repo. (Also, your source environment should still be activated at this point.)
```shell
cd image_captioning
```

8. Open the directory of notebooks, using the below command. You'll see all of the project files appear in your local environment; open the first notebook and follow the instructions.
```shell
jupyter notebook
```

9. Once you open any of the project notebooks, make sure you are in the correct `captioning_env` environment by clicking `Kernel > Change Kernel > captioning_env`.


## Dataset
### About MS COCO dataset
The Microsoft **C**ommon **O**bjects in **CO**ntext (MS COCO) dataset is a large-scale dataset for scene understanding.  The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms.  

![Sample Coco Example](images/coco-examples.jpg)

You can read more about the dataset on the [website](http://cocodataset.org/#home), [research paper](https://arxiv.org/pdf/1405.0312.pdf), or Appendix section at the end of this page.

### Install COCO API

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

## Jupyter Notebooks
The project is structured as a series of Jupyter notebooks that should be run in sequential order:

[The Dataset Exploration notebook](0_Dataset_Exploration.ipynb) initializes the [COCO API](https://github.com/cocodataset/cocoapi) (the "pycocotools" library) used to access data from the MS COCO (Common Objects in Context) dataset, which is "commonly used to train and benchmark object detection, segmentation, and captioning algorithms." The notebook also depicts the processing pipeline using the following diagram:

![Architecture](images/encoder-decoder.png)

The left half of the diagram depicts the "EncoderCNN", which encodes the critical information contained in a regular picture file into a "feature vector" of a specific size. That feature vector is fed into the "DecoderRNN" on the right half of the diagram (which is "unfolded" in time - each box labeled "LSTM" represents the same cell at a different time step). Each word appearing as output at the top is fed back to the network as input (at the bottom) in a subsequent time step until the entire caption is generated. The arrow pointing right that connects the LSTM boxes together represents hidden state information, which represents the network's "memory", also fed back to the LSTM at each time step.

### [1. The Architecture notebook](1_Architecture.ipynb) 

This notebook uses the pycocotools, torchvision transforms, and NLTK to preprocess the images and the captions for network training. It also explores details of EncoderCNN, which is taken pretrained from [torchvision.models, the ResNet50 architecture](https://pytorch.org/docs/master/torchvision/models.html#id3). The implementations of the EncoderRNN is found in the [model.py](model.py) file.

The core architecture used to achieve this task follows an encoder-decoder architecture, where the encoder is a pretrained ResNet CNN on ImageNet, and the decoder is a basic one-layer LSTM.

![encoder-decoder-architecture](images/encoder-decoder.png)

#### LSTM Decoder
In the project, we pass all our inputs as a sequence to an LSTM. A sequence looks like this: first a feature vector that is extracted from an input image, then a start word, then the next word, the next word, and so on.

#### Embedding Dimension
The LSTM is defined such that, as it sequentially looks at inputs, it expects that each individual input in a sequence is of a consistent size and so we embed the feature vector and each word so that they are `embed_size`.

#### Use the trained model
You can use my trained model in the `models` directory for your own experimentation.

Feel free to experiment with alternative architectures, such as bidirectional LSTM with attention mechanisms!

### [2. The Training notebook](2_Training.ipynb) 

This notebook provides the selection of hyperparameter values and EncoderRNN training. The hyperparameter selection is also explained.

### [3. The Inference notebook](3_Inference.ipynb) 

This notebook contains the testing of the trained networks to generate captions for additional images. No rigorous validation or accuracy measurement was performed, only sample images were generated. 


## Results
Here are some predictions from the model.

### Some good results
![sample_171](samples/sample_171.png?raw=true)<br/>
![sample_440](samples/sample_440.png?raw=true)<br/>
![sample_457](samples/sample_457.png?raw=true)<br/>
![sample_002](samples/sample_002.png?raw=true)<br/>
![sample_029](samples/sample_029.png?raw=true)<br/>
![sample_107](samples/sample_107.png?raw=true)<br/>
![sample_202](samples/sample_202.png?raw=true)


### Some not so good results

![sample_296](samples/sample_296.png?raw=true)<br/>
![sample_008](samples/sample_008.png?raw=true)<br/>
![sample_193](samples/sample_193.png?raw=true)<br/>
![sample_034](samples/sample_034.png?raw=true)<br/>
![sample_326](samples/sample_326.png?raw=true)<br/>
![sample_366](samples/sample_366.png?raw=true)<br/>
![sample_498](samples/sample_498.png?raw=true)

### More samples
There are more prediction samples in the samples folder.

## Future work
Steps for additional improvement would be exploring the hyperparameter and other architectures, and also training with more epochs.

## Appendix: More about COCO dataset API
COCO is a large image dataset designed for object detection, segmentation, person keypoints detection, stuff segmentation, and caption generation. This package provides Matlab, Python, and Lua APIs that assists in loading, parsing, and visualizing the annotations in COCO. Please visit http://cocodataset.org/ for more information on COCO, including the data, paper, and tutorials. The exact format of the annotations is also described on the COCO website. The Matlab and Python APIs are complete, the Lua API provides only basic functionality.

In addition to this API, please download both the COCO images and annotations in order to run the demos and use the API. Both are available on the project website.
- Please download, unzip, and place the images in: coco/images/
- Please download and place the annotations in: coco/annotations/
For substantially more details on the API please see [COCO Home Page](http://cocodataset.org/#home).

After downloading the images and annotations, run the Matlab, Python, or Lua demos for example usage.

To install:
-For Matlab, add coco/MatlabApi to the Matlab path (OSX/Linux binaries provided)
-For Python, run "make" under coco/PythonAPI
-For Lua, run “luarocks make LuaAPI/rocks/coco-scm-1.rockspec” under coco/


Note: This project is a part of [Udacity Computer Vision Nanodegree Program](https://www.udacity.com/course/computer-vision-nanodegree--nd891)