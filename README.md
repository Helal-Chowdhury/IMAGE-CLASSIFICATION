
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

1. Create environment
```bash
conda create -n <yourenvname> python=x.x anaconda
```
2. Activate environment
```bash
conda activate <yourenvname>
```
3. install packages
```bash
conda env update -n <yourenvname> --file environment.yaml
```
Some packages have pip dependency. In this case, use  pip to install the packages.
```bash
pip install <package name>
```
## RUN the App
To run the app, Go to __FRONEND__ and shoot this command:              

```bash
streamlit run Resnet.py
```


<img align="left" width="1000" height="500" src="https://github.com/Helal-Chowdhury/IMAGE-CLASSIFICATION/blob/main/image.jpg">







## From web UI 
 - select model
 - upload image
 - click the **Predict** button to predict

Note: in free subscription of Github does not support to push a single filesize of more than 100 Mbyte. However, train the model and save it and test with Densene109 in your machine. Implementation of single model and multi models both are provided in __FRONEND__ folder

