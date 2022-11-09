
# Title: End-to-End Image Classification

## Introduction
In this project, A template has been made for any general purpose Image Classifications problems. For demonstration, I choose  five fruit classes. However, you can choose any class of images by using Hugging face image collection API and train and deploy model immediately. I use Pytorch, and Streamlit for training and deployment. Pre-trained Models: Resnet18 and Densnet109 models. However, you can find tons of pre-trained image classification models from 
[Pytorch Hub](https://pytorch.org/hub/),[TensorFlow Hub ](https://www.tensorflow.org/hub), [Model Zoo](https://modelzoo.co/)
and [Hugging Face](https://huggingface.co/docs/hub/models-the-hub).

## How to create project environment and install packages:
This project is divided into three parts:
 - Image Collections (IMAGE-COLLECTION_AND_SPLITFOLDER)
 - Training (BACKEND)
 - Deployment (FRONEND)


Create Environment and Installation Packages

```bash
conda create --name <environment name> python=3.8
conda activate <environment name>
pip install -r requirements.txt
```
In case you have difficulties with installation of specific version of torch and torchvision use the following commands to install:
```bash
pip install torch==1.7.1 --no-cache-dir
pip install torchvision==0.8.2  --no-cache-dir
```
## RUN the App
To run the app, Go to __FRONEND__ folder and shoot this command:              
```bash
streamlit run Resnet.py
```


<img align="center" width="1000" height="500" src="https://github.com/Helal-Chowdhury/IMAGE-CLASSIFICATION/blob/main/image.jpg">




## From web UI 
 - select model
 - upload image
 - click the **Predict** button to predict

Note: in free subscription of Github does not support to push a single filesize of more than 100 Mbyte. However, train the model and save it and test with Densene109 in your machine. Implementation of single model and multi models both are provided in __FRONEND__ folder

